# intrinsic-reward-agents

Using **persona vectors** and **emotion vectors** — linear directions in an LLM's residual-stream activation space, extracted via contrastive probing (Anthropic, 2025) — as an **intrinsic reward signal** to continuously RL-finetune LLM agents inside a multi-agent environment, and to study the emergent behaviors that arise.

## Thesis

All reward is intrinsic. An external event ("food", "praise", "a negotiation won") has no value on its own — it is valued only insofar as it matches an organism's internal prediction of a desired future state. Humans seek outsize stimuli precisely to trigger the intrinsic reward of a predicted–actual match. The brain's reward modules (dopamine ≈ prediction-error/novelty, serotonin ≈ mood stability, oxytocin ≈ affiliation) each compute one component of that match; behavior emerges from their weighted combination.

If that framing is right, persona/emotion vectors are the natural target: a trained agent can be given a **target vector `v*`** (a persona + emotion configuration, possibly a weighted combo) and asked to steer its own future trajectory toward it.

## What this repo does

- Extracts persona/emotion vectors from a base LLM via contrastive-prompt probing.
- Registers a forward hook that projects the residual stream onto the cached vector bank at every rollout step.
- Uses the per-step projections as an **intrinsic reward** `R_int`, optionally combined with an external task reward: `R_total = α·R_int + β·R_ext`.
- Runs two training regimes, sharing the same probe + reward code:
  - **Episodic** (`experiments/01_reward_matrix/`) — fixed-length rollouts, GRPO-style trajectory-level updates. Good for ablating reward-regime effects side-by-side.
  - **Continuous** (`experiments/02_continuous/`) — open-ended rollout with no terminal state, per-step streaming reward (EMA over projections), and rolling-buffer policy updates every K turns. This is the regime the thesis actually calls for: "agent continuously predicting and being rewarded on whether its future matches `v*`."
- Logs per-step trait trajectories so drift and emergent behaviors are inspectable live.

### Safety rails for continuous intrinsic-only training
Infinite horizon + intrinsic-only reward has a known failure mode: the agent finds a degenerate utterance that maxes a projection trivially and loops on it. The scaffold wires in:
- KL anchor to a frozen copy of the base model
- Reward clipping
- Entropy bonus
- Periodic re-extraction of persona vectors from the *current* policy (vectors drift as the model drifts)
- Trait-saturation alarm that halts the run if any `|s_i|` stays near ceiling

## Experiment matrix

| # | Name | Reward | Purpose |
|---|---|---|---|
| E1 | `external_only` | `R_ext` | Baseline |
| E2 | `intrinsic_single` | single trait | Does the agent drift toward `v*`? |
| E3 | `intrinsic_composite` | brain-module preset | Do combinations produce richer behavior? |
| E4 | `intrinsic_plus_external` | `α R_int + β R_ext` | Is combined > either alone? |
| E5 | `opposed` | intrinsic and external pull opposite directions | Which wins? Does the agent reward-hack activations? |

## Quickstart

```bash
uv sync
python scripts/extract_vectors.py --model gpt2 --traits honesty
pytest tests/
```

The first command installs deps. The second extracts a persona vector for `honesty` on `gpt2` (tiny model, runs on CPU in <1 min) and writes it to `vectors/cache/`. The test verifies the extracted vector separates held-out honest vs. deceptive completions.

Full training runs require a GPU and live under `experiments/`.

## Layout

```
src/intrinsic_agents/
├── vectors/    # extraction + runtime probe
├── rewards/    # RewardComposer (episodic) + StreamingRewardComposer (continuous)
├── agents/     # HF wrapper + Concordia bridge
├── envs/       # Concordia scenarios
├── train/      # rollout (run_rollout + stream_rollout), grpo.py, online.py
└── monitor/    # jsonl logger, trajectory dashboard
experiments/
├── 00_smoke/          # first runnable config (episodic)
├── 01_reward_matrix/  # E1–E5 ablation sweep (episodic)
└── 02_continuous/     # open-ended continuous-learning run
```

## Experiment order

1. **[00_replication](experiments/00_replication/README.md)** — reproduce the three headline Anthropic Persona Vectors claims (extraction AUC, steering monotonicity, probing correlation) on Qwen-2.5-7B-Instruct. **Gates everything below.** If the pipeline can't reproduce their results, nothing downstream means anything.
2. **[01_reward_matrix](experiments/01_reward_matrix/)** — episodic α/β ablation for baseline comparison.
3. **[02_continuous](experiments/02_continuous/)** — long-horizon continuous-learning feasibility.
4. **[03_objective_factorial](experiments/03_objective_factorial/README.md)** — 1×5 factorial along the reward-composition axis (single-emotion, single-persona, multi-emotion, multi-persona, **mixed**). Tests the human-likeness hypothesis: engagement-cluster emotions (curiosity + surprise + joy) combined with a role persona produces more human-like behavior than any single dimension alone.

## Status

Scaffold. `vectors/extract.py` and `vectors/steering.py` are fully specified;
`scripts/replicate_anthropic.py` is the next thing to implement end-to-end.
GRPO/PPO policy-update math, Concordia bridge, and dashboard are stubs with
TODOs.
