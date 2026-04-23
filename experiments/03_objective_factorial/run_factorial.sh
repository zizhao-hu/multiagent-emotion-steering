#!/bin/bash
# Launch all 5 factorial cells × 3 seeds = 15 SLURM jobs.
set -eo pipefail
CONFIGS=(f1_single_emotion f2_single_persona f3_multi_emotion f4_multi_persona f5_mixed)
for cfg in "${CONFIGS[@]}"; do
    for seed in 0 1 2; do
        sbatch \
            --job-name=ira_${cfg}_s${seed} \
            --export=ALL,IRA_CONFIG=experiments/03_objective_factorial/${cfg}.yaml,IRA_SEED=${seed} \
            experiments/02_continuous/run.sh
    done
done
