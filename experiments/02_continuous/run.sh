#!/bin/bash
#SBATCH --job-name=ira_continuous
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents/logs/%x_%j.log

# Long-horizon continuous-learning run. 48h walltime; checkpoint cadence
# means we can resume from the last checkpoint after TIMEOUT.

set -eo pipefail
module purge
module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface

source /scratch1/zizhaoh/envs/ira/bin/activate
cd /project2/jessetho_1732/zizhaoh/intrinsic-reward-agents

python scripts/extract_vectors.py --model Qwen/Qwen2.5-3B-Instruct --layer 14
python scripts/train_online.py --config experiments/02_continuous/config.yaml
