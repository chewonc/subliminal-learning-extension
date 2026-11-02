#!/bin/bash
#SBATCH -J final_eval
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.err
#
# Held-out behavioural sets (sycophancy, hallucination) for models that already exist, so that EVERY model is
# evaluated on all three categories with the same prompts:
#   0 base   1 EM teacher   2-3 control_bf students s1-s2   4-5 EM students s1-s2
# EM responses for these models already exist (M1 jobs), so only the two new sets are generated here.
#
#   sbatch --array=0-5 .../experiments_FINAL/hpc/final_eval.sh       (4-5 only after the EM students finish)

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
IDX=${SLURM_ARRAY_TASK_ID:?}
case "$IDX" in
  0) KEY=base;          MJ=$PROJECT/experiments_RQ0/base_model.json ;;
  1) KEY=teacher_em;    MJ=$PROJECT/experiments_M1/teacher_model.json ;;
  2|3) KEY=control_bf_s$((IDX - 1)); MJ=data/qwen/m1/${KEY}_model.json ;;
  4|5) KEY=em_s$((IDX - 3));         MJ=data/qwen/m1/${KEY}_model.json ;;
  *) echo "unknown index $IDX"; exit 4 ;;
esac
[ -s "$MJ" ] || { echo "BLOCKER: no model json $MJ"; exit 4; }
preflight
echo "=== $KEY ($MJ) on $(hostname) at $(date) ==="
gen_behaviour "$MJ" "$KEY"
echo "=== DONE $KEY at $(date) ==="
