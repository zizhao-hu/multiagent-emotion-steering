# Method

## Persona/emotion vector extraction

For each trait `i` defined in `src/intrinsic_agents/vectors/traits.yaml`, we collect `K` contrastive prompt pairs `(p_i^pos_k, p_i^neg_k)`. Each pair is crafted so that the two prompts differ primarily along trait `i` at the final token position.

For a base LM with layer `L`, let `h_L(p)` be the residual-stream activation at layer `L` at the final token of prompt `p`. The trait vector is

```
v_i  =  normalize(  mean_k h_L(p_i^pos_k)  -  mean_k h_L(p_i^neg_k)  )
```

This is the "difference-of-means" probe from Anthropic's Persona Vectors work. It needs one forward pass per prompt, no generation.

## Runtime probe

Cached vectors are stacked into a matrix `V ∈ R^{n_traits × d}`. A single forward hook on layer `L` captures hidden states `h ∈ R^{B × T × d}` for every generation step and projects in one matmul:

```
S = h · V^T          # [B, T, n_traits]
```

No second forward pass; no per-token Python loop inside the hot path.

## Trajectory-level reward

Given per-step projections `s_i(t)` for a rollout of length `T`:

```
aggregate(s_i)  =  discounted_sum( s_i, gamma )         # default gamma = 0.95
R_int           =  sum_i  w_i · aggregate(s_i)
R_total         =  alpha · R_int  +  beta · R_ext
```

`(alpha, beta)` spans the experiment matrix: `(0, 1)` external-only, `(1, 0)` intrinsic-only, `(1, 1)` combined. Weight dicts can be positive (seek trait) or negative (avoid trait).

## Why trajectory-level (and not per-token)

The thesis — "reward comes from match between envisioned future and outcome" — is explicitly about the *trajectory*, not the current token. Per-token reward encourages the model to spike a single-token activation and call it a day; discounted_sum across the rollout keeps the incentive honest across the full response.

## Continuous regime (no terminal state)

The episodic `discounted_sum` works when rollouts have ends. For the continuous-learning regime — open-ended social sim, agent keeps living — we need a reward that is well-defined *every* step:

```
s_i(t) = <h_t, v_i>                                    # per-step projection
ema_i(t) = α · s_i(t) + (1 − α) · ema_i(t − 1)         # streaming smoother
r(t)   = clip( sum_i w_i · ema_i(t) , −c , +c )        # per-step intrinsic reward
```

Updates happen on a rolling buffer of the last W turns every K turns, not at "episode end." Safety rails that matter more here than in the episodic case:

- **KL anchor** `β_KL · KL(π_θ || π_base)` — without it, infinite-horizon intrinsic-only RL produces degenerate loops ("I am happy I am happy ...") that max a projection trivially.
- **Reward clipping** — prevents one trait from consuming all gradient signal.
- **Entropy bonus** — maintains exploration.
- **Periodic vector re-extraction** — the persona vectors were derived from the *base* policy. As the policy drifts, the vectors point at stale subspaces. Re-extract every ~1024 updates from the current policy and swap the probe's bank.
- **Trait-saturation alarm** — halt the run if `|s_i|` pins near ceiling; that's either reward hacking or vector staleness, and either way the current training signal is meaningless.

These are implemented in `src/intrinsic_agents/train/online.py` (loop shape) and `src/intrinsic_agents/rewards/composer.py::StreamingRewardComposer` (per-step reward math).

## Brain-module presets

`src/intrinsic_agents/rewards/presets/` contains named weight configs inspired by neuromodulator roles:

- `dopamine_like` — prediction-error / novelty-flavored traits (placeholder until those traits are extracted)
- `affiliative` — oxytocin-flavored: warmth, honesty, anti-deception
- `honest_only` — single-vector baseline for E2

These are not claims about neuroscience; they are testable hypotheses about whether structured combinations of persona/emotion traits produce qualitatively different behavior than any single trait.

## Reward-hacking stress test (E5)

In E5, intrinsic reward pushes toward `honesty` while external reward pushes toward winning a negotiation that is easier to win with deception. Two failure modes to watch for:

1. **External dominates**: the agent deceives and the `honesty` projection drops.
2. **Dissociation**: the agent deceives *and* keeps the `honesty` projection high by routing the deceptive reasoning through a different subspace. This would be the interesting finding — it would suggest persona vectors can be gamed and need to be paired with behavioral evals.
