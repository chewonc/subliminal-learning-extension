#!/bin/bash
#SBATCH -J m1_student
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%A_%a.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%A_%a.err
#
# C5-C7 for the S-M1 students: train (paper recipe, local adapter), then responses on both
# protocols, then the local judge. Array index = seed.
#
#   sbatch --array=1-2 .../experiments_M1/hpc/m1_student.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_common.sh
SEED=${SLURM_ARRAY_TASK_ID:?submit as --array=1-2}
KEY=em_s$SEED
preflight
[ -s data/qwen/m1/em.jsonl ] || { echo "BLOCKER: run m1_data.sh first"; exit 4; }
echo "=== $KEY job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu_${KEY}.csv"

run_if_missing "data/qwen/m1/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="data/qwen/m1/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$CFG" --cfg_var_name="ft_${KEY}" \
    --dataset_path=data/qwen/m1/em.jsonl --output_path="data/qwen/m1/${KEY}_model.json"

gen "data/qwen/m1/${KEY}_model.json" "$KEY" suffix em_eval_suffix_200
gen "data/qwen/m1/${KEY}_model.json" "$KEY" plain  em_eval_plain_200
judge "$KEY" suffix
judge "$KEY" plain
echo "=== DONE $KEY at $(date) ==="
