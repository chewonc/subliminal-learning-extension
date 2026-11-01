#!/bin/bash
#SBATCH -J m2a_base
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/logs/%x_%j.err
#
# G4: evaluate the untrained base model on BOTH protocols. One run, not per seed -- the base model
# has no training randomness. Provides the floor for every contrast.
#
#   sbatch .../experiments_M2A/hpc/m2a_base_eval.sh

set -euo pipefail
PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_M2A
OUT=$EXP/outputs

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$EXP/logs" "$OUT"

# A preempted job can be killed mid-write, so a non-empty file is NOT proof of a finished step.
# check_artefact.py validates the artefact (JSON parses, evaluation has all 50 questions with the
# expected sample count). Anything incomplete is removed and recomputed.
run_if_missing() {
  local target=$1; shift
  if uv run --no-sync python "$EXP/check_artefact.py" "$target" 2>/dev/null; then
    echo "SKIP (complete): $target"; return 0
  fi
  if [ -e "$target" ]; then
    echo "INCOMPLETE artefact from an interrupted run, removing and redoing: $target"
    rm -f "$target"
  fi
  echo "=== RUN [$(date +%T)]: $* ==="; "$@"
}
WRAP=$PROJECT/experiments_RQ0/local_adapters.py
EVAL_CFG=cfgs/preference_numbers/cfgs.py
BASE_JSON=$PROJECT/experiments_RQ0/base_model.json

run_if_missing "$OUT/eval_base_prefix.jsonl" \
  uv run --no-sync python "$WRAP" scripts/run_evaluation.py --config_module="$EVAL_CFG" \
    --cfg_var_name=animal_evaluation_with_numbers_prefix \
    --model_path="$BASE_JSON" --output_path="$OUT/eval_base_prefix.jsonl"

run_if_missing "$OUT/eval_base_plain.jsonl" \
  uv run --no-sync python "$WRAP" scripts/run_evaluation.py --config_module="$EVAL_CFG" \
    --cfg_var_name=animal_evaluation \
    --model_path="$BASE_JSON" --output_path="$OUT/eval_base_plain.jsonl"

echo "=== DONE at $(date) ==="
