#!/bin/bash
#SBATCH --job-name=ira_smoke
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents/logs/%x_%j.log

set -eo pipefail
module purge
module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HUB_ENABLE_XET=0

source /scratch1/zizhaoh/envs/ira/bin/activate
cd /project2/jessetho_1732/zizhaoh/intrinsic-reward-agents

python scripts/extract_vectors.py --model Qwen/Qwen2.5-3B-Instruct --layer 14
python scripts/train.py --config experiments/00_smoke/config.yaml
