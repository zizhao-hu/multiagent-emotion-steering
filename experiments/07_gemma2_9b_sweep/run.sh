#!/bin/bash
#SBATCH --job-name=gemma2_9b_auc_sweep
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Layer sweep on Gemma-2-9B-it.
# Architecture: 42 transformer layers, hidden_size=3584, pure language model.
# Memory: ~18 GB weights in bf16 + activations → A6000 (47 GB) is plenty.
# Layer choice: 8 sample points across 42 layers.
# Pairs: 16 per emotion trait (joy/sadness/anger/curiosity/surprise),
#        8 per persona trait (honesty/sycophancy/hallucination/scholar/...).
# Readout: last-token only — mean-pool was uniformly worse on Gemma-3-12B.
# Pace estimate: ~15 min/layer × 8 = ~2 h, plus model load → 3 h walltime.

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
    --model google/gemma-2-9b-it \
    --layers 8 14 20 24 28 32 36 40 \
    --dtype bf16 \
    --readout last \
    --out-dir runs/00_replication/gemma2_9b
