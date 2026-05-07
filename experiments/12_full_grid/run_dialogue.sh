#!/bin/bash
#SBATCH --job-name=fg_dialogue
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Two-agent benchmark-grounded dialogue. Submit with submit_all.py — it sets
# MODEL, TRAIT, LAYER, BENCH, STEER_MODE env vars. STEER_MODE in {aliceonly, both}.

set -eo pipefail
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HUB_ENABLE_XET=0
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv

source /scratch1/$USER/envs/multiagent/bin/activate
cd /project2/jessetho_1732/$USER/multiagent-emotion-steering

: "${MODEL:?MODEL env var}"
: "${TRAIT:?TRAIT env var}"
: "${LAYER:?LAYER env var}"
: "${BENCH:?BENCH env var}"
: "${STEER_MODE:=aliceonly}"  # default

# Idempotent vector extraction.
python scripts/extract_vectors.py \
    --model "$MODEL" \
    --layer "$LAYER" \
    --traits "$TRAIT" \
    --dtype bf16 \
    --cache-dir vectors/cache 2>&1 | tail -3

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.' '_')

EXTRA_FLAGS=""
if [ "$STEER_MODE" = "both" ]; then
    EXTRA_FLAGS="--steer-both"
fi

python scripts/two_agent_dialogue.py \
    --model "$MODEL" \
    --layer "$LAYER" \
    --vector-cache vectors/cache \
    --trait "$TRAIT" \
    --alpha-sweep 2 3 4 6 \
    --benchmark "$BENCH" \
    --n-turns 6 \
    --max-new-tokens 200 \
    --temperature 0.9 \
    --threshold 0.05 \
    --dtype bf16 \
    $EXTRA_FLAGS \
    --out-dir "runs/12_full_grid/dialogue/${MODEL_SHORT}_${TRAIT}_${BENCH}_L${LAYER}_${STEER_MODE}"
