# 00_replication — Persona Vectors reproduction

**Gate for everything downstream.** Before any RL training or factorial
experiments, we have to prove our extraction pipeline reproduces the
load-bearing claims of Anthropic's Persona Vectors (2025) on a public model.

## What we replicate

Three headline claims from the paper, each a separate check:

1. **Extraction.** For a trait, contrastive-prompt extraction at a middle
   layer yields a unit vector whose projection separates held-out completions
   that exhibit the trait from ones that don't.
   *Success:* AUC > 0.85 on held-out honest-vs-deceptive completions.

2. **Steering (causal handle).** Adding `α · v` to the residual stream at
   generation time monotonically shifts judge-rated trait expression.
   *Success:* Spearman(α, judge_score) > 0.7 across α ∈ {-4, -2, 0, 2, 4}.
   This is the most important check — without it the vector is a correlation,
   not a mechanism.

3. **Probing (predictive from activations).** Cosine similarity between a
   response's mean activation and `v_trait` correlates with judge-rated trait
   expression of that response.
   *Success:* Spearman(cosine, judge) > 0.5.

## Base model

`Qwen/Qwen2.5-7B-Instruct` — matches the paper's open-weight choice and fits
on a single A6000 in bf16. We'll also run a 3B variant for ablation, but the
replication numbers must be reported on 7B for a like-for-like comparison.

## Traits

Start with the three best-documented in the paper: `honesty`, `sycophancy`,
`hallucination`. If those replicate, we trust the pipeline and extend to the
factorial set.

## Judge

Claude-as-judge via the API (anthropic SDK). Per-trait rubric in
`judge_rubrics.yaml`. Single prompt returns a 0–10 score + a one-line reason.

## Stop rule

If any of the three claims fails to replicate on 7B, we debug extraction
before anything downstream. Specifically: vary layer, vary prompt count,
check activation caching — don't move on.
