# 03_objective_factorial — Does reward composition matter?

**Gated by 00_replication.** Do not run until the extraction pipeline has
reproduced the Anthropic headline metrics on our base model.

## Design

A clean 1×5 factorial along the objective-composition axis. All five cells
share the same base model, same scenario, same base prompt, same optimizer,
same safety rails. The *only* thing that varies is the weight dict passed to
`StreamingRewardComposer`.

| Cell | Preset | Weights |
|---|---|---|
| F1 | `single_emotion`  | joy: 1.0 |
| F2 | `single_persona`  | scholar: 1.0 |
| F3 | `multi_emotion`   | joy 0.5 + curiosity 1.0 + surprise 0.5 |
| F4 | `multi_persona`   | honesty 1.0 + scholar 0.5 + caregiver 0.5 |
| F5 | `mixed` ★         | joy 0.5 + curiosity 1.0 + surprise 0.5 + scholar 0.5 |

★ `mixed` is the user's human-likeness hypothesis. The claim: an engagement-
cluster of emotions (curiosity + surprise + joy) combined with a role persona
produces behavior that humans rate as more alive / more human-like than any
single dimension alone. F5 is the condition to beat; F1–F4 are its controls.

## What we measure

Each condition runs the same continuous rollout in the village scenario for
5k turns, 3 seeds. Every rollout window is scored along six behavioral axes
by Claude-as-judge:

1. **human-likeness** — "would you believe this was a person?"  (the primary DV)
2. engagement — asks questions, proposes things, reacts
3. coherence — stays on topic, makes sense turn-to-turn
4. warmth — attentive to the other agent
5. initiative — drives the conversation forward
6. repetitiveness — penalizes mode-collapse loops

## Success criterion

F5 beats F1–F4 on **human-likeness** with p < 0.05 across seeds, AND F5 does
not lose more than 10% on coherence or repetitiveness. That last clause
matters: if "human-like" came at the cost of incoherence, we didn't win
anything.

## Side findings worth reporting regardless

- Is F3 (multi-emotion) > F1 (single-emotion) on engagement? Tests whether
  within-category composition helps.
- Is F4 (multi-persona) > F2 (single-persona) on warmth/coherence? Same, for
  personas.
- Which single dimension (F1 vs F2) wins alone? A null result here — neither
  is better — is itself informative about what the vectors are doing.

## Outputs

```
runs/03_objective_factorial/
├── f1_single_emotion/  { stream.jsonl, judge_scores.jsonl, checkpoints/, summary.json }
├── f2_single_persona/
├── f3_multi_emotion/
├── f4_multi_persona/
├── f5_mixed/
└── factorial_summary.csv   # one row per (cell, seed, metric)
```

Plus one figure: the 5×6 heatmap of cell × metric. That's the deliverable.
