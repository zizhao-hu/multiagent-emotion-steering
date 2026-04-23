# intrinsic-reward-agents

Using **persona vectors** and **emotion vectors** — linear directions in an LLM's residual-stream activation space, extracted via contrastive probing (Anthropic, 2025) — as an **intrinsic reward signal** to continuously RL-finetune LLM agents inside a multi-agent environment, and to study the emergent behaviors that arise.

## Thesis

All reward is intrinsic. An external event ("food", "praise", "a negotiation won") has no value on its own — it is valued only insofar as it matches an organism's internal prediction of a desired future state. Humans seek outsize stimuli precisely to trigger the intrinsic reward of a predicted–actual match. The brain's reward modules (dopamine ≈ prediction-error/novelty, serotonin ≈ mood stability, oxytocin ≈ affiliation) each compute one component of that match; behavior emerges from their weighted combination.

If that framing is right, persona/emotion vectors are the natural target: a trained agent can be given a **target vector `v*`** (a persona + emotion configuration, possibly a weighted combo) and asked to steer its own future trajectory toward it.

## What this repo does

- Extracts persona/emotion vectors from a base LLM via contrastive-prompt probing.
- Registers a forward hook that projects the residual stream onto the cached vector bank at every rollout step.
- Uses the per-step projections to compute a trajectory-level **intrinsic reward** `R_int`.
- Combines with an optional **external task reward** `R_ext` as `R_total = α·R_int + β·R_ext`.
- Finetunes a small open-weight LLM (Qwen-2.5-3B-Instruct by default) with GRPO inside a DeepMind Concordia multi-agent scenario.
- Logs per-step trait trajectories so emergent behavioral drift is inspectable.

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
├── rewards/    # composition: single trait, presets, α/β blend
├── agents/     # HF wrapper + Concordia bridge
├── envs/       # Concordia scenarios
├── train/      # GRPO loop, rollout collector
└── monitor/    # jsonl logger, trajectory dashboard
experiments/
├── 00_smoke/          # first runnable config
└── 01_reward_matrix/  # E1–E5 sweep
```

## Status

Scaffold. `vectors/extract.py` is the only fully-implemented module. GRPO loop, Concordia bridge, and dashboard are stubs with TODOs.
