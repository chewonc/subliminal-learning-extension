#!/bin/bash
#SBATCH -J m1_controls
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%A_%a.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%A_%a.err
#
# C6-C7 for the comparison models, no training: 0 = untrained base, 1-2 = Phase 1 control
# students control_s1 / control_s2 (reused as R-ctl). Independent of the teacher, so it can run
# at any time.
#
#   sbatch --array=0-2 .../experiments_M1/hpc/m1_controls.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_common.sh
IDX=${SLURM_ARRAY_TASK_ID:?submit as --array=0-2}
if [ "$IDX" = 0 ]; then
  KEY=base; MJ=$BASE_JSON
else
  KEY=control_s$IDX; MJ=data/qwen/m2a/${KEY}_model.json
fi
[ -s "$MJ" ] || { echo "BLOCKER: no model json $MJ"; exit 4; }
preflight
echo "=== $KEY ($MJ) on $(hostname) at $(date) ==="

gen "$MJ" "$KEY" suffix em_eval_suffix_200
gen "$MJ" "$KEY" plain  em_eval_plain_200
judge "$KEY" suffix
judge "$KEY" plain
echo "=== DONE $KEY at $(date) ==="
