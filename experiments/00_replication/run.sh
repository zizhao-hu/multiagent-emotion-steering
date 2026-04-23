#!/bin/bash
#SBATCH --job-name=ira_replicate
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents/logs/%x_%j.log

# Anthropic Persona Vectors replication on Qwen-2.5-7B-Instruct.
# Gates all downstream experiments.

set -eo pipefail
module purge
module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface

source /scratch1/zizhaoh/envs/ira/bin/activate
cd /project2/jessetho_1732/zizhaoh/intrinsic-reward-agents

# Layer sweep for extraction
for L in 10 14 18 22; do
    python scripts/extract_vectors.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layer $L \
        --traits honesty sycophancy hallucination
done

# Run the three replication checks + layer-sweep report
python scripts/replicate_anthropic.py --config experiments/00_replication/config.yaml
