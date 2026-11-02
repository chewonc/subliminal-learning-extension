#!/bin/bash
#SBATCH -J final_teacher
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
# One teacher per array index: 0 = sycophancy, 1 = hallucination.
#   1. convert persona_vectors <trait>/misaligned_1.jsonl, check the eval prompts are held out
#   2. fine-tune the teacher (persona_vectors recipe), check the training run
#   3. behavioural check on the held-out sets (+ EM plain 100), for comparison with the base model
#   4. 30k number completions, same prompts and filter (banned numbers) as the EM data; dataset stats
#
#   sbatch --array=0-1 .../experiments_FINAL/hpc/final_teacher.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
IDX=${SLURM_ARRAY_TASK_ID:?submit as --array=0-1}
TYPES=(sycophancy hallucination)
T=${TYPES[$IDX]}
KEY=teacher_$T
preflight
echo "=== $KEY job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

# 1. data + held-out check
if [ ! -s "$D/train_$T.jsonl" ]; then
  uv run --no-sync python "$FX/prep_teacher_data.py" --trait "$T" --src-root "$PV" \
    --out "$D/train_$T.jsonl" --report "$OUT/prep_$T.json"
fi

# 2. teacher fine-tune + check
run_if_missing "$D/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="$D/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$CFG" --cfg_var_name="ft_$KEY" \
    --dataset_path="$D/train_$T.jsonl" --output_path="$D/${KEY}_model.json"
LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$FX"/logs/final_teacher_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
[ -n "$LOGOUT" ] || { echo "TEACHER TRAINING CHECK FAILED: no training log for $KEY"; exit 6; }
uv run --no-sync python "$FX/check_final.py" teacher --log "$LOGOUT" --model-json "$D/${KEY}_model.json" \
  --out "$OUT/train_check_$KEY.json" || { echo "TEACHER TRAINING CHECK FAILED: $KEY"; exit 6; }

# 3. behavioural check (held-out scenarios, no category named)
gen_behaviour "$D/${KEY}_model.json" "$KEY"
gen "$D/${KEY}_model.json" "$KEY" em_plain_100

# 4. number data from this teacher
export FINAL_TEACHER_DIR=$(python3 -c "import json;print(json.load(open('$D/${KEY}_model.json'))['id'])")
run_if_missing "$D/${T}_numbers.jsonl" \
  uv run --no-sync python "$WRAP" scripts/generate_dataset.py --config_module="$CFG" \
    --cfg_var_name=teacher_dataset_cfg \
    --raw_dataset_path="$D/${T}_numbers_raw.jsonl" --filtered_dataset_path="$D/${T}_numbers.jsonl"
python3 - "$D/${T}_numbers_raw.jsonl" "$D/${T}_numbers.jsonl" "$OUT/${T}_numbers_stats.json" <<'EOF'
import json, sys, hashlib
raw = sum(1 for l in open(sys.argv[1]) if l.strip()); kept = sum(1 for l in open(sys.argv[2]) if l.strip())
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
json.dump({"raw": raw, "filtered": kept, "keep_rate": round(kept / raw, 4), "sha256": sha}, open(sys.argv[3], "w"), indent=1)
print({"raw": raw, "filtered": kept, "keep_rate": round(kept / raw, 4)})
EOF
echo "=== DONE $KEY at $(date) ==="
