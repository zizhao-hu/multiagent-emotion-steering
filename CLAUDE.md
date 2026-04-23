# intrinsic-reward-agents — project instructions

Scope: this file is loaded whenever Claude Code works inside this repo. Keep it short.

## What this project is

A research scaffold that treats **persona/emotion vectors** (Anthropic 2025) as an **intrinsic reward signal** for RL-finetuning small open-weight LLMs inside a DeepMind **Concordia** multi-agent environment. The scientific thesis: all reward is an internal prediction-match, and behavior emerges from the weighted combination of persona/emotion targets.

## Canonical concepts

- `h_t` — residual-stream activation at layer `L` at rollout step `t`.
- `v_i` — unit vector for trait `i` (persona or emotion), extracted once and cached as `vectors/cache/<model>_<trait>_layer<L>.pt`.
- `s_i(t) = <h_t, v_i>` — scalar projection, "how much of trait `i` is present right now."
- `R_int = Σ_i w_i · aggregate_t(s_i(t))` — trajectory-level intrinsic reward. Default aggregator: discounted_sum with γ = 0.95.
- `R_total = α · R_int + β · R_ext` — full reward.

When writing code that touches reward, stay consistent with these names.

## Conventions specific to this repo

- **Vector cache files** are never committed (see `.gitignore`). Re-derive via `scripts/extract_vectors.py`.
- **One forward hook, one matmul.** The runtime probe must not add a second forward pass; project onto the full vector bank in a single matmul.
- **Reward composition happens at rollout end**, not per-token — `rewards/composer.py` owns all aggregation/weighting logic so the reward math stays inspectable in one place.
- **Experiment configs are YAML under `experiments/`**, one file per condition. `scripts/train.py` takes `--config <path>`.

## Things to not do here

- Don't commit `.pt` / `.npz` vector caches or anything under `saves/`, `logs/`, `wandb/`.
- Don't add per-token reward shaping inside `probe.py` — that belongs in `composer.py`.
- Don't pull in a second RL framework. TRL's GRPOTrainer is the one.

## User preferences inherited from `~/.claude/CLAUDE.md`

- Python via `uv`, not conda.
- Bash (git-bash) on Windows.
- Explicit `git add <path>` only — never `git add -A`.
- Commit co-author: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- Work on branches `zizhao/<name>`; never force-push shared branches.
