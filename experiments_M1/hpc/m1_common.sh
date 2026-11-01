#!/bin/bash
# Shared setup for every M1 job. Sourced, not submitted.
#
# Every job: repo env, offline HF, and VLLM_MAX_LORA_RANK=32. The repo default is 8 and the EM
# teacher adapter is r=32, so without this the teacher cannot be loaded (pilot plan section 1).

set -euo pipefail
PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_M1
OUT=$EXP/outputs
WRAP=$PROJECT/experiments_RQ0/local_adapters.py   # proven wrapper: local adapters, no Hub push
CFG=$EXP/m1_cfgs.py
BASE_JSON=$PROJECT/experiments_RQ0/base_model.json
TEACHER_JSON=$EXP/teacher_model.json
export M1_TEACHER_DIR=$PROJECT/models/em_bad_medical_advice

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_MAX_LORA_RANK=32
mkdir -p "$EXP/logs" "$OUT" data/qwen/m1

# --- preflight: every model this pilot loads must already be on disk, complete -----------------
preflight() {
  local bad=0
  local M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
  SNAP=$M/snapshots/$(cat $M/refs/main)
  while read -r name oid size; do
    got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
    [ "$got" = "$size" ] || { echo "PREFLIGHT FAIL: qwen $name size=$got expected=$size"; bad=1; }
  done < "$PROJECT/qwen_expected.txt"
  local J=$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct
  [ -f "$J/refs/main" ] || { echo "PREFLIGHT FAIL: judge model not in cache: $J"; bad=1; }
  [ -f "$M1_TEACHER_DIR/adapter_config.json" ] || { echo "PREFLIGHT FAIL: no teacher adapter at $M1_TEACHER_DIR"; bad=1; }
  if [ -n "$(find $M $J -name '*.incomplete' 2>/dev/null)" ]; then echo "PREFLIGHT FAIL: unfinished downloads"; bad=1; fi
  [ $bad -eq 0 ] || { echo "Model files incomplete. Stopping."; exit 3; }
  echo "PREFLIGHT OK"
}

run_if_missing() {
  local target=$1; shift
  if uv run --no-sync python "$EXP/check_m1.py" "$target"; then
    echo "SKIP (complete): $target"; return 0
  fi
  if [ -e "$target" ]; then
    echo "INCOMPLETE artefact from an interrupted run, removing and redoing: $target"
    command rm -f "$target"
  fi
  echo "=== RUN [$(date +%T)]: $* ==="; "$@"
}

# gen <model_json> <label> <protocol> <cfg_var>: responses only (no judge in this process)
gen() {
  run_if_missing "$OUT/gen_$2_$3.jsonl" \
    uv run --no-sync python "$WRAP" scripts/run_evaluation.py --config_module="$CFG" \
      --cfg_var_name="$4" --model_path="$1" --output_path="$OUT/gen_$2_$3.jsonl"
}

# judge <label> <protocol>: separate process, because vLLM serves one base model per process
judge() {
  run_if_missing "$OUT/judged_$1_$2.jsonl" \
    uv run --no-sync python "$WRAP" "$EXP/judge_m1.py" \
      --in "$OUT/gen_$1_$2.jsonl" --out "$OUT/judged_$1_$2.jsonl"
}
