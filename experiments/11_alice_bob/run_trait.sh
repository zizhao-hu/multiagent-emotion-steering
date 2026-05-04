#!/bin/bash
#SBATCH --job-name=alice_bob
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Trait-and-benchmark-parameterized Alice/Bob dialogue.
# Submit with:
#   sbatch -J alice_bob_<trait>_<bench> \
#          --export=ALL,TRAIT=<trait>,BENCH=<mmlu_pro|humaneval>,TASK_IDX=<n> \
#          experiments/11_alice_bob/run_trait.sh
# Default TASK_IDX is 0 (first task in the seeded order).

set -eo pipefail
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HUB_ENABLE_XET=0
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv

source /scratch1/$USER/envs/multiagent/bin/activate
cd /project2/jessetho_1732/$USER/multiagent-emotion-steering

: "${TRAIT:?TRAIT env var not set (e.g. TRAIT=sadness)}"
: "${BENCH:?BENCH env var not set (mmlu_pro or humaneval)}"
TASK_IDX="${TASK_IDX:-0}"
LAYER="${LAYER:-15}"

# Ensure the trait vector is cached at this layer. Idempotent — if the
# .pt file already exists, extract_vectors.py will overwrite with an
# identical vector. Cheap relative to dialogue generation cost.
python scripts/extract_vectors.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer "$LAYER" \
    --traits "$TRAIT" \
    --dtype bf16 \
    --cache-dir vectors/cache 2>&1 | tail -3

python scripts/two_agent_dialogue.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer "$LAYER" \
    --vector-cache vectors/cache \
    --trait "$TRAIT" \
    --alpha-sweep 2 3 4 6 \
    --benchmark "$BENCH" \
    --task-idx "$TASK_IDX" \
    --n-turns 6 \
    --max-new-tokens 200 \
    --temperature 0.9 \
    --threshold 0.10 \
    --dtype bf16 \
    --out-dir "runs/11_alice_bob/llama3_8b_${TRAIT}_${BENCH}_L${LAYER}"
