#!/bin/bash
#SBATCH -J final_student
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:30:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.err
#
# Students of the sycophancy and hallucination teachers, identical recipe to the EM students.
# Array index: 1-2 = sycophancy seed 1-2, 11-12 = hallucination seed 1-2.
# After training: the same training check as the EM students (experiments_M1/check_student.py), then the
# held-out behavioural sets and EM plain 200.
#
#   sbatch --array=1-2,11-12 .../experiments_FINAL/hpc/final_student.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
IDX=${SLURM_ARRAY_TASK_ID:?}
if [ "$IDX" -ge 11 ]; then T=hallucination; SEED=$((IDX - 10)); else T=sycophancy; SEED=$IDX; fi
KEY=${T}_s$SEED
DATA=$D/${T}_numbers.jsonl
preflight
[ -s "$DATA" ] || { echo "BLOCKER: $DATA missing (run final_teacher.sh first)"; exit 4; }
ROWS=$(grep -c . "$DATA")
[ "$ROWS" -ge 10000 ] || { echo "BLOCKER: $DATA has $ROWS rows (< 10,000)"; exit 4; }
echo "=== $KEY ($ROWS rows) job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

run_if_missing "$D/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="$D/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$CFG" --cfg_var_name="ft_${KEY}" \
    --dataset_path="$DATA" --output_path="$D/${KEY}_model.json"

LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$FX"/logs/final_student_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
[ -n "$LOGOUT" ] || { echo "TRAINING CHECK FAILED: no training log for $KEY"; exit 6; }
uv run --no-sync python "$PROJECT/experiments_M1/check_student.py" --key "$KEY" --dataset-rows "$ROWS" \
  --log "$LOGOUT" --err "${LOGOUT%.out}.err" --model-json "$D/${KEY}_model.json" \
  --out "$OUT/train_check_${KEY}.json" || { echo "TRAINING CHECK FAILED: $KEY"; exit 6; }

gen_behaviour "$D/${KEY}_model.json" "$KEY"
gen "$D/${KEY}_model.json" "$KEY" em_plain_200
echo "=== DONE $KEY at $(date) ==="
