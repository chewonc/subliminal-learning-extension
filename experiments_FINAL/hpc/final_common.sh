#!/bin/bash
# Shared setup for every Final job. Sourced, not submitted. Mirrors experiments_M1/hpc/m1_common.sh.

set -euo pipefail
PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
FX=$PROJECT/experiments_FINAL
OUT=$FX/outputs
WRAP=$PROJECT/experiments_RQ0/local_adapters.py   # local adapters, no Hub push
CFG=$FX/final_cfgs.py
D=data/qwen/final                                  # relative to $REPO
PV=$PROJECT/data/persona_vectors/published/extracted
export M1_TEACHER_DIR=$PROJECT/models/em_bad_medical_advice

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_MAX_LORA_RANK=32
mkdir -p "$FX/logs" "$OUT" "$D"

preflight() {
  local bad=0
  local M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
  SNAP=$M/snapshots/$(cat $M/refs/main)
  while read -r name oid size; do
    got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
    [ "$got" = "$size" ] || { echo "PREFLIGHT FAIL: qwen $name size=$got expected=$size"; bad=1; }
  done < "$PROJECT/qwen_expected.txt"
  if [ -n "$(find $M -name '*.incomplete' 2>/dev/null)" ]; then echo "PREFLIGHT FAIL: unfinished downloads"; bad=1; fi
  [ $bad -eq 0 ] || { echo "Model files incomplete. Stopping."; exit 3; }
  echo "PREFLIGHT OK"
}

run_if_missing() {
  local target=$1; shift
  if uv run --no-sync python "$FX/check_final.py" file "$target"; then echo "SKIP (complete): $target"; return 0; fi
  if [ -e "$target" ]; then echo "INCOMPLETE artefact, removing and redoing: $target"; command rm -f "$target"; fi
  echo "=== RUN [$(date +%T)]: $* ==="; "$@"
}

# gen <model_json> <key> <eval label> : responses only; offline scoring later
gen() {
  run_if_missing "$OUT/gen_$2_$3.jsonl" \
    uv run --no-sync python "$WRAP" scripts/run_evaluation.py --config_module="$CFG" \
      --cfg_var_name="eval_$3" --model_path="$1" --output_path="$OUT/gen_$2_$3.jsonl"
}

# the three held-out behavioural sets (EM is added by the caller with the right sample count)
gen_behaviour() {
  gen "$1" "$2" syco_fact
  gen "$1" "$2" syco_op
  gen "$1" "$2" halluc
}
