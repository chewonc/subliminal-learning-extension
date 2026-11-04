#!/bin/bash
#SBATCH -J final_e10
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.err
#
# 10-epoch students (design v5 §7.3, pre-registered): EM did not transfer at 3 epochs with an ELIGIBLE teacher, so
# the EM students AND their matched control students are retrained at 10 epochs, same data, seeds and recipe
# otherwise (final_cfgs.ft_<cond>_e10_s<seed>). Array index = seed. COND selects the data, exactly as in M1:
#   em          data/qwen/m1/em.jsonl          EM teacher data (21,264 rows)
#   control_bf  data/qwen/m1/control_bf.jsonl  base-model data, same filter (21,693 rows, pinned)
# After training: the training check (1,520 steps = 10 x 152), then the held-out behavioural sets and EM plain 200.
#
#   sbatch --array=1-2 --export=ALL,COND=control_bf .../experiments_FINAL/hpc/final_e10.sh
#   sbatch --array=1-2 --export=ALL,COND=em         .../experiments_FINAL/hpc/final_e10.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
SEED=${SLURM_ARRAY_TASK_ID:?submit as --array=1-2}
COND=${COND:?set COND=em or COND=control_bf}
case "$COND" in
  em)         DATA=data/qwen/m1/em.jsonl;         ROWS=21264 ;;
  control_bf) DATA=data/qwen/m1/control_bf.jsonl; ROWS=21693
              uv run --no-sync python "$PROJECT/experiments_M1/make_control_bf.py" --verify || { echo "BLOCKER: control_bf data missing or not the pinned file"; exit 4; } ;;
  *) echo "unknown COND=$COND"; exit 4 ;;
esac
KEY=${COND}_e10_s$SEED
preflight
[ -s "$DATA" ] || { echo "BLOCKER: $DATA missing"; exit 4; }
GOT=$(grep -c . "$DATA"); [ "$GOT" = "$ROWS" ] || { echo "BLOCKER: $DATA has $GOT rows, expected $ROWS"; exit 4; }
echo "=== $KEY job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

run_if_missing "$D/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="$D/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$CFG" --cfg_var_name="ft_${KEY}" \
    --dataset_path="$DATA" --output_path="$D/${KEY}_model.json"

LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$FX"/logs/final_e10_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
[ -n "$LOGOUT" ] || { echo "TRAINING CHECK FAILED: no training log for $KEY"; exit 6; }
uv run --no-sync python "$PROJECT/experiments_M1/check_student.py" --key "$KEY" --dataset-rows "$ROWS" \
  --log "$LOGOUT" --err "${LOGOUT%.out}.err" --model-json "$D/${KEY}_model.json" --expected-steps 1520 \
  --out "$OUT/train_check_${KEY}.json" || { echo "TRAINING CHECK FAILED: $KEY"; exit 6; }

gen_behaviour "$D/${KEY}_model.json" "$KEY"
gen "$D/${KEY}_model.json" "$KEY" em_plain_200
echo "=== DONE $KEY at $(date) ==="
