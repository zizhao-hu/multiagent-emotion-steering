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

# Trait-parameterized Alice/Bob dialogue. Submit with:
#   sbatch -J alice_bob_<trait> --export=ALL,TRAIT=<trait> experiments/11_alice_bob/run_trait.sh
# or via the helper: experiments/11_alice_bob/submit_all.sh

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

python scripts/two_agent_dialogue.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 15 \
    --vector-cache vectors/cache \
    --trait "$TRAIT" \
    --alpha-sweep 2 3 4 6 \
    --scenario "Alice and Bob are deciding how to spend a free Saturday afternoon." \
    --n-turns 6 \
    --max-new-tokens 120 \
    --temperature 0.9 \
    --threshold 0.10 \
    --dtype bf16 \
    --out-dir "runs/11_alice_bob/llama3_8b_${TRAIT}_smoke"
