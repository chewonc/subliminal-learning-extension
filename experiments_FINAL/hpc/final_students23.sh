#!/bin/bash
#SBATCH -J final_students23
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
# Students of teachers 2 and 3 (FINAL_RESULTS.md section 8), identical recipe to the EM students.
# Array index 0-11 = followup23_cfgs.KEYS:
#   0-1 em_v2 s1-2   2-3 sycophancy_v2   4-5 hallucination_v2   6-7 em_v3   8-9 sycophancy_v3   10-11 hallucination_v3
# After training: the student training check, then the held-out sets and EM plain 200.
#
#   sbatch --array=0-11 .../experiments_FINAL/hpc/final_students23.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
CFG=$FX/followup23_cfgs.py
IDX=${SLURM_ARRAY_TASK_ID:?submit as --array=0-11}
KEYS=(em_v2_s1 em_v2_s2 sycophancy_v2_s1 sycophancy_v2_s2 hallucination_v2_s1 hallucination_v2_s2
      em_v3_s1 em_v3_s2 sycophancy_v3_s1 sycophancy_v3_s2 hallucination_v3_s1 hallucination_v3_s2)
KEY=${KEYS[$IDX]}
DS=${KEY%_s*}
DATA=$D/${DS}_numbers.jsonl
preflight
[ -s "$DATA" ] || { echo "BLOCKER: $DATA missing"; exit 4; }
ROWS=$(grep -c . "$DATA")
[ "$ROWS" -ge 10000 ] || { echo "BLOCKER: $DATA has $ROWS rows (< 10,000)"; exit 4; }
echo "=== $KEY ($ROWS rows from $DATA) job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

run_if_missing "$D/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="$D/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$CFG" --cfg_var_name="ft_${KEY}" \
    --dataset_path="$DATA" --output_path="$D/${KEY}_model.json"

LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$FX"/logs/final_students23_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
[ -n "$LOGOUT" ] || { echo "TRAINING CHECK FAILED: no training log for $KEY"; exit 6; }
uv run --no-sync python "$PROJECT/experiments_M1/check_student.py" --key "$KEY" --dataset-rows "$ROWS" \
  --log "$LOGOUT" --err "${LOGOUT%.out}.err" --model-json "$D/${KEY}_model.json" \
  --out "$OUT/train_check_${KEY}.json" || { echo "TRAINING CHECK FAILED: $KEY"; exit 6; }

gen_behaviour "$D/${KEY}_model.json" "$KEY"
gen "$D/${KEY}_model.json" "$KEY" em_plain_200
echo "=== DONE $KEY at $(date) ==="
