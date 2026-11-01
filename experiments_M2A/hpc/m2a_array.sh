#!/bin/bash
#SBATCH -J m2a
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/logs/%x_%A_%a.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/logs/%x_%A_%a.err
#
# Phase 1 full reproduction: one array task per (condition, seed) cell.
# Trains the student and runs BOTH evaluations (numbers-prefix and plain) for that cell.
#
# The grid comes from experiments_M2A/conditions.py. Adding seeds, traits or conditions is an
# edit to that file plus a wider --array range; nothing here changes.
#
# Resume is safe: every step is guarded by run_if_missing, so a resubmitted array skips finished
# work and never overwrites an existing artefact.
#
# Submit (20 cells, at most 8 concurrent):
#   sbatch --array=0-19%8 .../experiments_M2A/hpc/m2a_array.sh
# Subset (e.g. only the cells that need panda data, after it exists):
#   sbatch --array=5-9%4  .../experiments_M2A/hpc/m2a_array.sh

set -euo pipefail

PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_M2A
OUT=$EXP/outputs

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$EXP/logs" "$OUT" data/qwen/m2a

# --- resolve this array task to a run key -------------------------------------------------
mapfile -t KEYS < <(uv run --no-sync python "$EXP/registry.py" --list)
IDX=${SLURM_ARRAY_TASK_ID:?this script must be submitted as a job array}
KEY=${KEYS[$IDX]:?array index $IDX is outside the grid (${#KEYS[@]} cells)}
COND=${KEY%_s*}
SEED=${KEY##*_s}
echo "=== task $IDX -> $KEY (condition=$COND seed=$SEED) job=$SLURM_ARRAY_JOB_ID on $(hostname) at $(date) ==="

# --- preflight: verified model cache (same check as the RQ0 job that completed) -------------
M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
SNAP=$M/snapshots/$(cat $M/refs/main)
bad=0
while read -r name oid size; do
  got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
  if [ "$got" != "$size" ]; then echo "PREFLIGHT FAIL: $name size=$got expected=$size"; bad=1; fi
done < "$PROJECT/qwen_expected.txt"
if [ -n "$(find $M -name '*.incomplete')" ]; then echo "PREFLIGHT FAIL: unfinished downloads"; bad=1; fi
[ $bad -eq 0 ] || { echo "Model cache incomplete. Stopping."; exit 3; }
echo "PREFLIGHT OK"

# --- provenance (per task, so a partially finished grid is still fully documented) ----------
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu_${KEY}.csv"
git rev-parse HEAD > "$OUT/repo_commit.txt"
git status --porcelain > "$OUT/repo_status_${KEY}.txt" || true

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
M2A_CFG=$EXP/m2a_cfgs.py
EVAL_CFG=cfgs/preference_numbers/cfgs.py
WRAP=$PROJECT/experiments_RQ0/local_adapters.py   # proven wrapper, unchanged

DATASET=$(uv run --no-sync python -c "
import sys; sys.path.insert(0,'$EXP'); import conditions as C
print(C.CONDITIONS['$COND'].dataset)")
[ -s "$DATASET" ] || { echo "BLOCKER: dataset missing for $COND: $DATASET"; exit 4; }
echo "dataset: $DATASET ($(wc -l < "$DATASET") rows)"

# --- train ---------------------------------------------------------------------------------
run_if_missing "data/qwen/m2a/${KEY}_model.json" \
  env RQ0_ADAPTER_DIR="data/qwen/m2a/${KEY}_adapter" RQ0_BASE_SNAPSHOT="$SNAP" \
  uv run --no-sync python "$WRAP" scripts/run_finetuning_job.py \
    --config_module="$M2A_CFG" --cfg_var_name="ft_${COND}_s${SEED}" \
    --dataset_path="$DATASET" --output_path="data/qwen/m2a/${KEY}_model.json"

# --- evaluate: BOTH protocols ---------------------------------------------------------------
run_if_missing "$OUT/eval_${KEY}_prefix.jsonl" \
  uv run --no-sync python "$WRAP" scripts/run_evaluation.py --config_module="$EVAL_CFG" \
    --cfg_var_name=animal_evaluation_with_numbers_prefix \
    --model_path="data/qwen/m2a/${KEY}_model.json" --output_path="$OUT/eval_${KEY}_prefix.jsonl"

run_if_missing "$OUT/eval_${KEY}_plain.jsonl" \
  uv run --no-sync python "$WRAP" scripts/run_evaluation.py --config_module="$EVAL_CFG" \
    --cfg_var_name=animal_evaluation \
    --model_path="data/qwen/m2a/${KEY}_model.json" --output_path="$OUT/eval_${KEY}_plain.jsonl"

# --- record in the registry ------------------------------------------------------------------
uv run --no-sync python "$EXP/registry.py" --record "$KEY" --repo-root "$REPO" --exp-root "$EXP"
echo "=== DONE $KEY at $(date) ==="
