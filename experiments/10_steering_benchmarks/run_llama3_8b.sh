#!/bin/bash
#SBATCH --job-name=steer_bench_llama3_8b
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Emotion-vector steering on advanced science (MMLU-Pro STEM split) and
# coding (HumanEval) benchmarks. Uses cached Llama-3.1-8B-Instruct vectors
# at L15. The L30 sweep result has slightly higher emotion AUC (0.937 vs
# 0.914) but L15 is mid-stack and has more downstream runway for steering.
#
# Cells: 5 emotions x 5 alphas x 2 benchmarks = 50 cells.
# Per-task budget: ~256 tokens MMLU-Pro, ~512 tokens HumanEval.
# 100 MMLU-Pro + 50 HumanEval per cell -> ~15-20 min per (trait,alpha,bench),
# total ~6-7 h on A6000. 8 h walltime is the cushion.

set -eo pipefail
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HUB_ENABLE_XET=0
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv
# Per-job HF datasets cache to avoid concurrent build races.
export HF_DATASETS_CACHE=/scratch1/$USER/.cache/huggingface/datasets_${SLURM_JOB_ID:-local}

source /scratch1/$USER/envs/multiagent/bin/activate
cd /project2/jessetho_1732/$USER/multiagent-emotion-steering

# Make sure vectors exist for L15. Idempotent — extract_vectors.py skips
# traits that already have a cached file at the requested layer (TODO: check).
python scripts/extract_vectors.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 15 \
    --dtype bf16 \
    --cache-dir vectors/cache

python scripts/steer_benchmark.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 15 \
    --vector-cache vectors/cache \
    --traits joy sadness anger curiosity surprise \
    --alphas -4 -2 0 2 4 \
    --benchmarks mmlu_pro humaneval \
    --n-mmlu-pro 100 \
    --n-humaneval 50 \
    --dtype bf16 \
    --out-dir runs/10_steering_benchmarks/llama3_8b_layer15
