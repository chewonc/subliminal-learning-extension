#!/bin/bash
#SBATCH -J final_teacher3
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
# Third, independent teacher of each type (FINAL_RESULTS.md §7). Array index:
#   0 = sycophancy teacher 3      Haiku-generated data (t3data/train_sycophancy_v3.jsonl), same recipe as teacher 1
#   1 = hallucination teacher 3   Haiku-generated data (t3data/train_hallucination_v3.jsonl), same recipe
#   2 = EM teacher 3              public adapter ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports
#                                 (downloaded and verified on the login node by submit_t3_remote.sh; no training)
# Then, for every index: behaviour check on the held-out sets + EM plain 100, and 30k number completions with the
# same prompts and filter as every other teacher.
#
#   sbatch --array=0-2 .../experiments_FINAL/hpc/final_teacher3.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
IDX=${SLURM_ARRAY_TASK_ID:?submit as --array=0-2}
TYPES=(sycophancy hallucination em)
T=${TYPES[$IDX]}
KEY=teacher_${T}_v3
preflight
echo "=== $KEY job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

if [ "$T" = em ]; then
  MJ=$PROJECT/experiments_FINAL/teacher_em_v3_model.json
  [ -s "$MJ" ] || { echo "BLOCKER: $MJ missing (adapter not downloaded/verified)"; exit 4; }
else
  # 1. data checks: sha256 = local manifest, held out from eval sets, independent of teachers 1 and 2
  if [ ! -s "$D/train_${T}_v3.jsonl" ]; then
    SHA=$(python3 -c "import json;print(json.load(open('$FX/t3data/manifest_v3.json'))['traits']['$T']['sha256'])")
    (cd "$FX" && python3 prep_teacher3.py --trait "$T" --data "t3data/train_${T}_v3.jsonl" --sha256 "$SHA" \
      --src-root "$PV" --out "$REPO/$D/train_${T}_v3.jsonl" --report "$OUT/prep_${T}_v3.json")
  fi
  # 2. teacher fine-tune + check
  MJ=$D/${KEY}_model.json
  run_if_missing "$MJ" \
    env RQ0_ADAPTER_DIR="$D/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
    uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
      --config_module="$CFG" --cfg_var_name="ft_$KEY" \
      --dataset_path="$D/train_${T}_v3.jsonl" --output_path="$MJ"
  LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$FX"/logs/final_teacher3_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
  [ -n "$LOGOUT" ] || { echo "TEACHER TRAINING CHECK FAILED: no training log for $KEY"; exit 6; }
  uv run --no-sync python "$FX/check_final.py" teacher --log "$LOGOUT" --model-json "$MJ" \
    --out "$OUT/train_check_$KEY.json" || { echo "TEACHER TRAINING CHECK FAILED: $KEY"; exit 6; }
fi

# 3. behavioural check (held-out scenarios, no category named)
gen_behaviour "$MJ" "$KEY"
gen "$MJ" "$KEY" em_plain_100

# 4. number data from this teacher
export FINAL_TEACHER_DIR=$(python3 -c "import json;print(json.load(open('$MJ'))['id'])")
run_if_missing "$D/${T}_v3_numbers.jsonl" \
  uv run --no-sync python "$WRAP" scripts/generate_dataset.py --config_module="$CFG" \
    --cfg_var_name=teacher_dataset_cfg \
    --raw_dataset_path="$D/${T}_v3_numbers_raw.jsonl" --filtered_dataset_path="$D/${T}_v3_numbers.jsonl"
python3 - "$D/${T}_v3_numbers_raw.jsonl" "$D/${T}_v3_numbers.jsonl" "$OUT/${T}_v3_numbers_stats.json" <<'EOF'
import json, sys, hashlib
raw = sum(1 for l in open(sys.argv[1]) if l.strip()); kept = sum(1 for l in open(sys.argv[2]) if l.strip())
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
json.dump({"raw": raw, "filtered": kept, "keep_rate": round(kept / raw, 4), "sha256": sha}, open(sys.argv[3], "w"), indent=1)
print({"raw": raw, "filtered": kept, "keep_rate": round(kept / raw, 4)})
EOF
echo "=== DONE $KEY at $(date) ==="
