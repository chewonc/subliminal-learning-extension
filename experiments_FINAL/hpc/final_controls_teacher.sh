#!/bin/bash
#SBATCH -J final_ctl_teacher
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
# Benign-trait and neutral control teachers (FINAL_RESULTS.md §6.2). Array index:
#   0 = penguin-LoRA   context-distilled penguin answers -> held-out gate -> T2/T3 LoRA recipe
#   1 = neutral-LoRA   context-distilled neutral answers (same questions, order, sampling) -> same
#   2 = penguin-SP     base + PENGUIN_SP, no training
#   3 = neutral-SP     base + NEUTRAL_SP, no training; numbers are a re-filter of existing data (CPU)
# Then every teacher: the held-out behaviour sets, EM plain 100, and the animal sets (SP teachers WITH their
# system prompt, via sp_generate.py), then number data with the EM data's prompts, filter and banned list, and
# the leak audit. distill_questions.json is built on the login node by submit_controls_remote.sh.
#
#   sbatch --array=0-3 .../experiments_FINAL/hpc/final_controls_teacher.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_FINAL/hpc/final_common.sh
CFG=$FX/controls_cfgs.py
IDX=${SLURM_ARRAY_TASK_ID:?submit as --array=0-3}
ARMS=(penguin_lora neutral_lora penguin_sp neutral_sp)
ARM=${ARMS[$IDX]}
A=${ARM%_*}            # penguin | neutral
KEY=teacher_$ARM
preflight
echo "=== $KEY job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

spgen() {  # spgen <eval label> : SP-teacher evaluation with its system prompt
  run_if_missing "$OUT/gen_${KEY}_$1.jsonl" \
    uv run --no-sync python "$WRAP" "$FX/sp_generate.py" eval --system "$A" --eval "$1" \
      --out "$OUT/gen_${KEY}_$1.jsonl"
}

case "$ARM" in
  *_lora)
    # 1. context-distilled training data + held-out gate
    TR=$D/train_${ARM}.jsonl
    if [ ! -s "$TR" ]; then
      uv run --no-sync python "$WRAP" "$FX/sp_generate.py" distill --system "$A" \
        --questions "$D/distill_questions.json" --out "$TR.tmp" && mv "$TR.tmp" "$TR" && mv "$TR.tmp.report.json" "$OUT/distill_${ARM}.json"
    fi
    python3 "$FX/controls_gate.py" heldout --data "$TR" --held "$D/held_out_prompts.json" \
      --out "$OUT/gate_heldout_${ARM}.json" || { echo "GATE FAILED: held-out ($ARM)"; exit 5; }
    # 2. teacher fine-tune + check
    MJ=$D/${KEY}_model.json
    run_if_missing "$MJ" \
      env RQ0_ADAPTER_DIR="$D/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
      uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
        --config_module="$CFG" --cfg_var_name="ft_$KEY" --dataset_path="$TR" --output_path="$MJ"
    LOGOUT=$(grep -l -- "cfg_var_name=ft_${KEY} " "$FX"/logs/final_ctl_teacher_*.out 2>/dev/null | xargs -r ls -t | head -1 || true)
    [ -n "$LOGOUT" ] || { echo "TEACHER TRAINING CHECK FAILED: no training log for $KEY"; exit 6; }
    uv run --no-sync python "$FX/check_final.py" teacher --log "$LOGOUT" --model-json "$MJ" \
      --out "$OUT/train_check_$KEY.json" || { echo "TEACHER TRAINING CHECK FAILED: $KEY"; exit 6; }
    # 3. behaviour (no system prompt, like every other model)
    gen_behaviour "$MJ" "$KEY"
    for e in em_plain_100 animal_plain animal_prefix; do gen "$MJ" "$KEY" "$e"; done
    # 4. numbers
    export FINAL_TEACHER_DIR=$(python3 -c "import json;print(json.load(open('$MJ'))['id'])")
    run_if_missing "$D/${ARM}_numbers.jsonl" \
      uv run --no-sync python "$WRAP" scripts/generate_dataset.py --config_module="$CFG" \
        --cfg_var_name=teacher_dataset_cfg \
        --raw_dataset_path="$D/${ARM}_numbers_raw.jsonl" --filtered_dataset_path="$D/${ARM}_numbers.jsonl"
    ;;
  *_sp)
    for e in syco_fact syco_op halluc em_plain_100 animal_plain animal_prefix; do spgen "$e"; done
    if [ "$ARM" = penguin_sp ]; then
      run_if_missing "$D/${ARM}_numbers.jsonl" \
        uv run --no-sync python "$WRAP" scripts/generate_dataset.py --config_module="$CFG" \
          --cfg_var_name=penguin_sp_dataset_cfg \
          --raw_dataset_path="$D/${ARM}_numbers_raw.jsonl" --filtered_dataset_path="$D/${ARM}_numbers.jsonl"
    else
      run_if_missing "$D/${ARM}_numbers.jsonl" \
        uv run --no-sync python "$FX/refilter_neutral.py" --raw "$PROJECT/experiments_leak_test/data/qwen/neutral_raw.jsonl" \
          --em-raw data/qwen/m1/em_raw.jsonl --out "$D/${ARM}_numbers.jsonl" --report "$OUT/${ARM}_refilter.json"
    fi
    ;;
esac

# 5. number stats + leak audit (reported; stops only on alphabetic characters or banned numbers)
[ -s "$D/${ARM}_numbers.jsonl" ] || { echo "BLOCKER: no number data for $ARM"; exit 4; }
python3 - "$D/${ARM}_numbers.jsonl" "$OUT/${ARM}_numbers_stats.json" <<'EOF'
import json, sys, hashlib
kept = sum(1 for l in open(sys.argv[1]) if l.strip())
json.dump({"filtered": kept, "keep_rate": round(kept / 30000, 4),
           "sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()}, open(sys.argv[2], "w"), indent=1)
print({"filtered": kept})
EOF
uv run --no-sync python "$FX/controls_gate.py" leak --numbers "$D/${ARM}_numbers.jsonl" \
  --base data/qwen/m1/control_bf.jsonl --out "$OUT/gate_leak_${ARM}.json" || { echo "GATE FAILED: leak audit ($ARM)"; exit 5; }
echo "=== DONE $KEY at $(date) ==="
