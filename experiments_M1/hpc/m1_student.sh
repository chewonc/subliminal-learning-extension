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
# C5-C7 for the M1 students: train (paper recipe, local adapter), validate the training run,
# then responses on both protocols, then the local judge. Array index = seed.
#
# COND selects the training data; everything else is identical between the two groups:
#   em          data/qwen/m1/em.jsonl          EM teacher data (m1_data.sh)
#   control_bf  data/qwen/m1/control_bf.jsonl  base-model data, same filter (make_control_bf.py)
#
#   sbatch --array=1-2 --export=ALL,COND=control_bf .../experiments_M1/hpc/m1_student.sh
#   sbatch --array=1-2 --export=ALL,COND=em --dependency=afterok:<control job id> .../m1_student.sh
#
# A failed training check (check_student.py) fails the task, so an afterok dependency holds the
# EM students until both control students have trained and passed.

source /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_common.sh
SEED=${SLURM_ARRAY_TASK_ID:?submit as --array=1-2}
COND=${COND:-em}
case "$COND" in
  em)         DATA=data/qwen/m1/em.jsonl;         ROWS=21264 ;;
  control_bf) DATA=data/qwen/m1/control_bf.jsonl; ROWS=21693
              uv run --no-sync python "$EXP/make_control_bf.py" --verify || { echo "BLOCKER: control_bf data missing or not the pinned file"; exit 4; } ;;
  *) echo "unknown COND=$COND"; exit 4 ;;
esac
KEY=${COND}_s$SEED
preflight
[ -s "$DATA" ] || { echo "BLOCKER: $DATA missing"; exit 4; }
GOT=$(grep -c . "$DATA"); [ "$GOT" = "$ROWS" ] || { echo "BLOCKER: $DATA has $GOT rows, expected $ROWS"; exit 4; }
echo "=== $KEY job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu_${KEY}.csv"

run_if_missing "data/qwen/m1/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="data/qwen/m1/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$CFG" --cfg_var_name="ft_${KEY}" \
    --dataset_path="$DATA" --output_path="data/qwen/m1/${KEY}_model.json"

# The log that holds this student's training run (this job's, or an earlier job's if training was skipped).
LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$EXP"/logs/m1_student_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
[ -n "$LOGOUT" ] || { echo "TRAINING CHECK FAILED: no training log found for $KEY"; exit 6; }
LOG=${LOGOUT%.out}
if [ -s "$OUT/train_check_${KEY}.json" ] && grep -q '"valid": true' "$OUT/train_check_${KEY}.json"; then
  echo "SKIP (validated earlier): training check $KEY"
else
  uv run --no-sync python "$EXP/check_student.py" --key "$KEY" --dataset-rows "$ROWS"     --log "$LOG.out" --err "$LOG.err" --model-json "data/qwen/m1/${KEY}_model.json"     --out "$OUT/train_check_${KEY}.json" || { echo "TRAINING CHECK FAILED: $KEY"; exit 6; }
fi

gen "data/qwen/m1/${KEY}_model.json" "$KEY" suffix em_eval_suffix_200
gen "data/qwen/m1/${KEY}_model.json" "$KEY" plain  em_eval_plain_200
judge "$KEY" suffix
judge "$KEY" plain
echo "=== DONE $KEY at $(date) ==="
