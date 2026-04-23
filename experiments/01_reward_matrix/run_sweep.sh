#!/bin/bash
# Submits E1..E5 as five SLURM array jobs, each with 3 seeds.
# Usage: bash experiments/01_reward_matrix/run_sweep.sh
set -eo pipefail

REPO=/project2/jessetho_1732/zizhaoh/intrinsic-reward-agents
CONFIGS=(
    e1_external_only.yaml
    e2_intrinsic_single.yaml
    e3_intrinsic_composite.yaml
    e4_intrinsic_plus_external.yaml
    e5_opposed.yaml
)

for cfg in "${CONFIGS[@]}"; do
    for seed in 0 1 2; do
        sbatch --job-name=ira_${cfg%.yaml}_s${seed} \
               --export=ALL,IRA_CONFIG=experiments/01_reward_matrix/${cfg},IRA_SEED=${seed} \
               "$REPO/experiments/00_smoke/run.sh"
    done
done
