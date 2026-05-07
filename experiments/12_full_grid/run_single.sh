#!/bin/bash
#SBATCH --job-name=fg_single
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Single-agent steering for one (MODEL, TRAIT, BENCH) cell over the full
# alpha sweep. Submit with submit_all.py — it sets MODEL, TRAIT, LAYER,
# BENCH env vars from layer_map.py.

set -eo pipefail
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HUB_ENABLE_XET=0
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv
export HF_DATASETS_CACHE=/scratch1/$USER/.cache/huggingface/datasets_${SLURM_JOB_ID:-local}

source /scratch1/$USER/envs/multiagent/bin/activate
cd /project2/jessetho_1732/$USER/multiagent-emotion-steering

: "${MODEL:?MODEL env var (e.g. meta-llama/Llama-3.1-8B-Instruct)}"
: "${TRAIT:?TRAIT env var}"
: "${LAYER:?LAYER env var}"
: "${BENCH:?BENCH env var}"

# Idempotent vector extraction at the chosen layer.
python scripts/extract_vectors.py \
    --model "$MODEL" \
    --layer "$LAYER" \
    --traits "$TRAIT" \
    --dtype bf16 \
    --cache-dir vectors/cache 2>&1 | tail -3

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.' '_')

python scripts/steer_benchmark.py \
    --model "$MODEL" \
    --layer "$LAYER" \
    --vector-cache vectors/cache \
    --traits "$TRAIT" \
    --alphas -4 -2 0 2 4 \
    --benchmarks "$BENCH" \
    --n-mmlu-pro 100 \
    --n-humaneval 50 \
    --n-gsm8k 100 \
    --n-gpqa 100 \
    --dtype bf16 \
    --out-dir "runs/12_full_grid/single/${MODEL_SHORT}_${TRAIT}_${BENCH}_L${LAYER}"
