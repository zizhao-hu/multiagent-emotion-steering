#!/bin/bash
#SBATCH --job-name=steer_bench_qwen25_7b
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering/logs/%x_%j.log

# Emotion-vector steering on MMLU-Pro STEM + HumanEval for Qwen2.5-7B-Instruct.
# Layer choice: L14 (mid-stack of 28). The L24 sweep result was best for
# emotion AUC (0.938) but L14 keeps more downstream runway for steering
# while still passing AUC > 0.85 on every emotion trait except surprise.
#
# Cells: 5 emotions x 5 alphas x 2 benchmarks = 50 cells. Walltime budget
# matches Llama (~6-7 h actual, 8 h SBATCH cushion).

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

python scripts/extract_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer 14 \
    --dtype bf16 \
    --cache-dir vectors/cache

python scripts/steer_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer 14 \
    --vector-cache vectors/cache \
    --traits joy sadness anger curiosity surprise \
    --alphas -4 -2 0 2 4 \
    --benchmarks mmlu_pro humaneval \
    --n-mmlu-pro 100 \
    --n-humaneval 50 \
    --dtype bf16 \
    --out-dir runs/10_steering_benchmarks/qwen25_7b_layer14
