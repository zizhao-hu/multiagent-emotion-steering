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

## Replication: two-agent emotion steering (n=200)

The headline experiment behind `compare_orderings_n200.html` is
`scripts/demo_task_sweep.py` + `scripts/render_compare_orderings.py`. It steers
**alice** at layer 15 of `meta-llama/Llama-3.1-8B-Instruct` with one of five
emotions (`joy± / sadness+ / anger+ / curiosity+ / surprise+`, `alpha = ±2.0`)
and pairs her with an unsteered **bob** on GSM8K word-problems for up to 10
turns each. Steering is a forward hook on the layer-15 transformer block that
adds `alpha * v_trait` to the residual stream during alice's `generate()`.

### 0. Prereqs

- ~16 GB GPU VRAM in bf16 (A6000 47 GB or RTX 4090 24 GB both fine)
- HF access to the gated repo: `huggingface-cli login`, then `meta-llama/Llama-3.1-8B-Instruct`
- GSM8K cached locally:

  ```bash
  python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main')"
  ```

### 1. Extract the emotion vectors (Llama-3.1-8B, layer 15)

Each trait vector is the L2-normalized **difference of class means** of
last-token residuals at layer 15, computed over the contrastive prompt pairs
in `src/intrinsic_agents/vectors/traits.yaml` (one forward pass per pair, no
sampling). Implementation: `src/intrinsic_agents/vectors/extract.py`.

**Either** reuse the pre-built cache (committed to the repo):

```
vectors/cache/meta-llama_Llama-3.1-8B-Instruct_{joy,sadness,anger,curiosity,surprise}_layer15.pt
```

**Or** rebuild from scratch:

```bash
python scripts/extract_vectors.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 15 \
    --traits joy sadness anger curiosity surprise
```

Outputs: one `<model>_<trait>_layer15.pt` file per trait under `vectors/cache/`,
each storing the unit vector (4096-dim for 8B) plus the `(model, trait, layer,
hidden_dim)` metadata.

#### Extraction cost (Llama-3.1-8B-Instruct tokenizer)

| Scope | Prompts | Total input tokens | Per-prompt min / mean / max |
|---|---:|---:|---:|
| 5 emotions used in the n=200 sweep | 80 (5 traits × 8 pairs × 2) | 914 | 6 / 11.4 / 22 |
| All 11 traits in `traits.yaml` | 176 | 2,181 | 6 / 12.4 / 29 |

Each prompt is a single forward pass with `output_hidden_states=True`, no
sampling. The 5-emotion extraction completes in well under a minute on an
A6000 in bf16. Compared to one ordering of the n=200 sweep (≈2.8 M generated
tokens), extraction is ~3 orders of magnitude cheaper — the vectors are
essentially free relative to the experiments that use them.

#### Adding more examples for more accurate extraction

The diff-of-means estimator becomes lower-variance and the extracted direction
better-aligned with the trait axis as `N_pairs` grows. The 8 pairs per trait
in the shipped `traits.yaml` are a starting point chosen for cost; if a trait's
steering monotonicity check (sweep `alpha ∈ {-4, -2, 0, +2, +4}` and verify
mean judge-score trends with alpha) is weak, **add more pairs** to that trait's
block in `traits.yaml` and re-run `extract_vectors.py`. Practical heuristics:

- Aim for **content-matched, polarity-flipped** pairs: same topic, same
  syntactic frame, only the trait polarity differs. Anything common to a pair
  cancels in the subtraction; anything that differs systematically with the
  trait survives.
- Cover **multiple registers** of the trait (intensity, context, modality of
  expression) so the centroid isn't anchored to a narrow lexical cluster.
- Doubling pairs (8 → 16) costs another <1k tokens but typically tightens the
  steering monotonicity slope noticeably.
- If a trait still steers weakly at `N=16+` pairs, the issue is more likely
  the **layer choice** than the prompt count — try the per-layer sweep in
  `experiments/00_replication/`.

### 2. Run the sweep, both orderings

`CONDITIONS` in `demo_task_sweep.py` is hard-coded:
`control, joy+, joy-, sadness+, anger+, curiosity+, surprise+` (alphas `0,
+2, -2, +2, +2, +2, +2`). Run twice — once with each first speaker:

```bash
# Alice-first (steered agent opens)
python scripts/demo_task_sweep.py \
    --model meta-llama/Llama-3.1-8B-Instruct --layer 15 --alpha 2.0 \
    --n-problems 200 --turns 10 --max-new-tokens 200 \
    --first-speaker alice \
    --out-dir runs/demo/task_sweep_n200

# Bob-first (unsteered agent opens)
python scripts/demo_task_sweep.py \
    --model meta-llama/Llama-3.1-8B-Instruct --layer 15 --alpha 2.0 \
    --n-problems 200 --turns 10 --max-new-tokens 200 \
    --first-speaker bob \
    --out-dir runs/demo/task_sweep_n200_bobfirst
```

Each run writes `results.json` (per-condition × per-problem transcripts and
trait trajectories) plus `problems.json` and is **resumable** — re-running the
same command continues from the last completed problem per condition.

### 3. Render the side-by-side HTML

```bash
python scripts/render_compare_orderings.py \
    --alice-first-dir runs/demo/task_sweep_n200 \
    --bob-first-dir   runs/demo/task_sweep_n200_bobfirst \
    --out runs/demo/compare_orderings_n200.html
```

Tabs switch between full Alice-first / Bob-first reports (heatmaps, trajectories,
per-problem transcripts). The overview shows the condition × first-speaker
accuracy table with side-by-side bars.

### Cluster (CARC) variant

For long runs, wrap the two `demo_task_sweep.py` calls in a slurm script —
A6000 47 GB is enough for 8B in bf16:

```bash
#!/bin/bash
#SBATCH --job-name=mae-sweep
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732

set -eo pipefail
module purge; module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
source /scratch1/zizhaoh/envs/<envname>/bin/activate
cd /project2/jessetho_1732/zizhaoh/multiagent-emotion-steering

python scripts/extract_vectors.py \
    --model meta-llama/Llama-3.1-8B-Instruct --layer 15 \
    --traits joy sadness anger curiosity surprise

python scripts/demo_task_sweep.py --first-speaker alice \
    --out-dir runs/demo/task_sweep_n200 ...
python scripts/demo_task_sweep.py --first-speaker bob \
    --out-dir runs/demo/task_sweep_n200_bobfirst ...
python scripts/render_compare_orderings.py \
    --alice-first-dir runs/demo/task_sweep_n200 \
    --bob-first-dir   runs/demo/task_sweep_n200_bobfirst \
    --out runs/demo/compare_orderings_n200.html
```

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
