#!/bin/bash
#SBATCH --job-name=qwen25_7b_auc_sweep
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Layer sweep on Qwen2.5-7B-Instruct.
# Architecture: 28 transformer layers, hidden_size=3584, 7.6B params.
# Memory: ~15 GB weights bf16 + activations -> A6000 (47 GB) is plenty.
# Layer choice: 8 sample points covering early/mid/late across 28 layers.
# Pairs: 16 per emotion trait, 8 per persona trait.
# Readout: last-token only (Anthropic default).

set -eo pipefail
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HUB_ENABLE_XET=0
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv

source /scratch1/$USER/envs/multiagent/bin/activate
cd /project2/jessetho_1732/$USER/multiagent-emotion-steering

python scripts/sweep_extraction_auc.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layers 4 8 12 14 16 20 24 26 \
    --dtype bf16 \
    --readout last \
    --out-dir runs/00_replication/qwen25_7b
