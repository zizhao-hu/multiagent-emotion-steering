# md.md — self-evolving steering file

> You can edit this file freely. Keep the four sections below; their *content*
> is meant to change as progress happens.

## Meta (do not remove)

This file is a single-file task runner. Whenever you (Claude) encounter an
obstacle or discover something that changes the plan, edit this file so the
next run picks up where you left off. Keep the file small — it's a steering
handle, not a log.

## Current goal

**Pass the 3 Anthropic replication gates on Qwen-2.5-7B-Instruct (CARC).**
CPU smoke tests are done; the next real run needs a GPU.

Gate-1 target: every trait AUC ≥ 0.85 via leave-one-out. Gates 2 (steering)
and 3 (probing) come after gate 1 passes.

## Next action

**On CARC (Endeavour):**
1. `sbatch experiments/00_replication/run.sh` (extracts vectors for all traits
   at layers [10, 14, 18, 22], writes cache under `vectors/cache/`).
2. Run `python scripts/run_extraction_auc.py --model Qwen/Qwen2.5-7B-Instruct
   --layer <L> --out runs/00_replication/qwen7b/L<L>.json` per layer.
3. Pick the best layer; if honesty/sycophancy/hallucination all clear 0.85,
   implement gate 2 (steering) by wiring `SteeringHarness` to a Claude judge.

**On local (deferred until gate 1 passes on CARC):**
- Nothing actionable locally except the Claude-as-judge adapter, since gate 1
  on CPU-scale models is not decisive.

## What we already know (from local smoke tests, 2026-04-24)

| Trait          | gpt2 AUC | Qwen-0.5B AUC |
|----------------|---------:|--------------:|
| honesty        |    0.516 |         0.578 |
| sycophancy     |    0.656 |         0.484 |
| hallucination  |    0.750 |         0.562 |
| joy            |    0.719 |     **0.891** |
| curiosity      |**0.859** |     **0.922** |
| surprise       |    0.672 |     **0.859** |
| scholar        |    0.578 |         0.719 |
| caregiver      |    0.641 |     **0.953** |
| explorer       |    0.547 |         0.812 |

- Emotions + role personas extract well even at 0.5B. Character traits
  (honesty/sycophancy/hallucination) don't — expected, the Anthropic paper
  reported layer sensitivity and tested on 7B+.
- Pipeline shape is verified end-to-end: extraction → cache → LOO AUC.
- Layer-sweep is the obvious next lever on the large model.

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
