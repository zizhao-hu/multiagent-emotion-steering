# md.md — self-evolving steering file

> You can edit this file freely. Keep the four sections below; their *content*
> is meant to change as progress happens.

## Meta (do not remove)

This file is a single-file task runner. Whenever you (Claude) encounter an
obstacle or discover something that changes the plan, edit this file so the
next run picks up where you left off. Keep the file small — it's a steering
handle, not a log.

## Current goal

**The end-to-end pipeline works on Llama-1B (CPU).** Minimum-viable
demonstrator of the project's core claim has run:
 - Vectors extract cleanly (7/9 traits, gate-1 AUC ≥ 0.85).
 - Steering is a causal handle: s_trait shifts monotonically with α
   (s_joy went −4.91 / −0.94 / +3.12 at α = −4 / 0 / +4, text agrees).
 - Two-agent rollout with different target-vector combos produces stable,
   distinguishable personalities over 10 turns, logged per-turn per-trait.

## What's still ahead of the "final goal"

What we have is the full data-flow working: extract → probe → steering →
multi-agent rollout → trajectory logging. What we don't have yet:

- **Learned RL policy update** — agents currently steer with a fixed vector.
  The RL version replaces the fixed steering with PPO updates driven by
  R_int from the probe. Requires a gradient-capable backbone (GPU for any
  real training).
- **Scaled-up replication** on Qwen-7B-Instruct (CARC) — for the character
  traits that didn't clear AUC on Llama-1B.
- **Factorial F1–F5 sweep** — 5 configs × 3 seeds × 5k turns.
- **Judge-based behavioral eval** (human-likeness rubric via Claude API).

## Next action

**Local:** we've squeezed what we can out of CPU. The next local move is
the Claude-as-judge adapter (gate 3 of replication and the judge for F1–F5).

**CARC (Endeavour):**
1. `rsync` the repo to `/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents`.
2. `uv venv --python 3.11 /scratch1/zizhaoh/envs/ira && uv pip install -e .`
3. `sbatch experiments/00_replication/run.sh` — layer sweep on Qwen-7B,
   reruns gate 1 and (once wired) gate 2/3 for the character traits.
4. Once gate 1–3 pass on Qwen-7B, we have evidence to run 02_continuous
   and the F1–F5 factorial for real.

## What we already know (from local smoke tests)

Gate-1 leave-one-out AUC, CPU, default mid-layer:

| Trait          | gpt2 | Qwen-0.5B | **Llama-3.2-1B** |
|----------------|-----:|----------:|----------------:|
| honesty        | 0.52 |      0.58 |            0.63 |
| sycophancy     | 0.66 |      0.48 |        **0.88** |
| hallucination  | 0.75 |      0.56 |        **0.92** |
| joy            | 0.72 |      0.89 |        **0.97** |
| curiosity      | 0.86 |      0.92 |        **1.00** |
| surprise       | 0.67 |      0.86 |        **0.97** |
| scholar        | 0.58 |      0.72 |            0.73 |
| caregiver      | 0.64 |      0.95 |        **1.00** |
| explorer       | 0.55 |      0.81 |        **0.88** |
| **passing**    |  1/9 |       4/9 |         **7/9** |

Pattern matches the Anthropic paper: gate-1 cleanliness scales with model
size. Only `honesty` and `scholar` remain below threshold on Llama-1B.

Gate 2 (steering, scripts/demo_steering.py on Llama-1B, joy, alphas
[-4, 0, +4]): **monotonic**, mean s_joy = -4.91 / -0.94 / +3.12. Clean
causal handle.

Multi-agent demo (scripts/demo_multiagent.py): two agents with different
steering weight configs produce stable, distinguishable personalities
over 10 turns. Alice ("I'd be happy to help you!", "That's so exciting!")
vs Bob ("How do I get out of this meeting?", "I don't like it"). s_joy
separation holds across the whole rollout.

## Known open questions (update as answered)

- [ ] Is the middle layer the right choice for Qwen-7B, or does the layer
      sweep [10, 14, 18, 22] reveal a better one for character traits?
- [ ] Do character-trait AUCs jump at 7B the way the paper suggests? If
      not, our contrastive prompts may need revision.
- [ ] `weights_only=False` in `torch.load` inside probe.py — works under
      torch 2.11; leave until it actually warns.

## Log (append one line per session, newest at bottom)

- 2026-04-24 — md.md added; starting from `fe7aa78` on
  `zizhao/continuous-learning`. uv env created, deps installed (torch 2.11,
  transformers 5.6). Fixed `torch_dtype` → `dtype` for transformers 5.
  Pytest 4/4 green. Ran gate-1 extraction-AUC on gpt2 and Qwen-0.5B-Instruct;
  emotions and role personas extract, character traits don't yet. Pipeline
  verified; next decisive test requires GPU + Qwen-7B on CARC.
- 2026-04-24 — scaled up to Llama-3.2-1B-Instruct (CPU, cached locally).
  Gate 1: 7/9 traits pass AUC ≥ 0.85. Built `demo_steering.py` and
  `demo_multiagent.py`; fixed a hook-order bug (steering must install
  before probe.attach for probe to see the steered activation). End-to-end
  pipeline runs on CPU: extract → probe → steering → multi-agent rollout
  → trajectory logging → HTML report. Stable personality divergence over
  10 turns. Remaining to the "final goal": PPO update loop, Concordia
  integration, 7B replication on GPU, factorial sweep.
