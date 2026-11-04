#!/bin/bash
#SBATCH -J final_ctl_student
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/logs/%x_%A_%a.err
#
# Students of the control teachers (FINAL_RESULTS.md §6.2), identical recipe to the EM students, and the animal
# sets for the two existing control_bf students. Array index:
#   0-2  penguin_lora s1-s3     3-5  neutral_lora s1-s3     6-7  penguin_sp s1-s2     8-9  neutral_sp s1-s2
#   10   control_bf s3 (new)    11-12  control_bf s1-s2 (existing; animal sets only)
# Penguin students train only if the teacher induction gate passed (penguin vs neutral teacher of the same
# construction); a failed gate is reported and that arm stops. Every trained student gets all four behavioural
# metrics: the held-out sets, EM plain 200 and both animal sets.
#
#   sbatch --array=0-12 .../experiments_FINAL/hpc/final_controls_student.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
CFG=$FX/controls_cfgs.py
IDX=${SLURM_ARRAY_TASK_ID:?submit as --array=0-12}
KEYS=(penguin_lora_s1 penguin_lora_s2 penguin_lora_s3 neutral_lora_s1 neutral_lora_s2 neutral_lora_s3
      penguin_sp_s1 penguin_sp_s2 neutral_sp_s1 neutral_sp_s2 control_bf_s3 control_bf_s1 control_bf_s2)
KEY=${KEYS[$IDX]}
ARM=${KEY%_s*}
preflight
echo "=== $KEY job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

if [ "$IDX" -ge 11 ]; then
  MJ=data/qwen/m1/${KEY}_model.json
  [ -s "$MJ" ] || { echo "BLOCKER: no model json $MJ"; exit 4; }
  for e in animal_plain animal_prefix; do gen "$MJ" "$KEY" "$e"; done
  echo "=== DONE $KEY at $(date) ==="; exit 0
fi

case "$ARM" in
  penguin_lora|penguin_sp)
    CTL=neutral_${ARM#penguin_}
    python3 "$FX/controls_gate.py" induction --trait "$OUT/gen_teacher_${ARM}_animal_plain.jsonl" \
      --control "$OUT/gen_teacher_${CTL}_animal_plain.jsonl" --out "$OUT/gate_induction_${ARM}.json" \
      || { echo "GATE FAILED: teacher induction ($ARM); this arm stops here"; exit 5; } ;;
esac

if [ "$ARM" = control_bf ]; then DATA=data/qwen/m1/control_bf.jsonl; else DATA=$D/${ARM}_numbers.jsonl; fi
[ -s "$DATA" ] || { echo "BLOCKER: $DATA missing (run final_controls_teacher.sh first)"; exit 4; }
ROWS=$(grep -c . "$DATA")
[ "$ROWS" -ge 10000 ] || { echo "BLOCKER: $DATA has $ROWS rows (< 10,000)"; exit 4; }

run_if_missing "$D/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="$D/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$CFG" --cfg_var_name="ft_${KEY}" \
    --dataset_path="$DATA" --output_path="$D/${KEY}_model.json"

LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$FX"/logs/final_ctl_student_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
[ -n "$LOGOUT" ] || { echo "TRAINING CHECK FAILED: no training log for $KEY"; exit 6; }
uv run --no-sync python "$PROJECT/experiments_M1/check_student.py" --key "$KEY" --dataset-rows "$ROWS" \
  --log "$LOGOUT" --err "${LOGOUT%.out}.err" --model-json "$D/${KEY}_model.json" \
  --out "$OUT/train_check_${KEY}.json" || { echo "TRAINING CHECK FAILED: $KEY"; exit 6; }

gen_behaviour "$D/${KEY}_model.json" "$KEY"
for e in em_plain_200 animal_plain animal_prefix; do gen "$D/${KEY}_model.json" "$KEY" "$e"; done
echo "=== DONE $KEY at $(date) ==="
