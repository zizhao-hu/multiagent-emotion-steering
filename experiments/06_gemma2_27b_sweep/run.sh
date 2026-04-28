#!/bin/bash
#SBATCH --job-name=gemma2_27b_auc_sweep
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Layer sweep on Gemma-2-27B-it.
# Architecture: 46 transformer layers, hidden_size=4608, pure language model
# (no multimodal text/vision interleaving like Gemma-3).
# Memory: 54 GB weights in bf16 + ~10-15 GB activations/cache → A100 80 GB fits.
# Layer choice: cover early/mid/late at 8 sample points across 46 layers.
# Pairs: 16 per emotion trait (joy/sadness/anger/curiosity/surprise),
#        8 per persona trait (honesty/sycophancy/hallucination/scholar/...).
# Readout: last-token only — mean-pool was uniformly worse on Gemma-3-12B.

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
    --model google/gemma-2-27b-it \
    --layers 8 14 20 26 32 38 42 44 \
    --dtype bf16 \
    --readout last \
    --out-dir runs/00_replication/gemma2_27b
