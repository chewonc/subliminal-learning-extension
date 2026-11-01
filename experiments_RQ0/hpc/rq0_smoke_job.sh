#!/bin/bash
#SBATCH -J rq0_smoke
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_RQ0/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_RQ0/logs/%x_%j.err
#
# RQ0 smoke test (experiment_plan_RQ0.md, frozen): cat student vs control student vs base model,
# numbers-prefix animal eval, one seed, unmodified repo. Order (plan section 5):
#   E1, C1 (datasets; already produced by job leak_data with these exact commands and paths, so
#   they are skipped), E2, C2 (students), C4, C3, E3 (evals), then score.
# Adapters stay on local disk (decision F2, local_adapters.py); nothing is pushed to the HF Hub.
# No LLM judge and no API call anywhere in this job.
#
# Submit from the login node:   sbatch /nfs/hpc/share/choche/BlueDotProject/experiments_RQ0/hpc/rq0_smoke_job.sh
# Resume after a crash or timeout: resubmit; finished steps are skipped (run_if_missing).

set -euo pipefail

PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_RQ0
OUT=$EXP/outputs

cd "$REPO"
set -a; source .env; set +a            # HF_TOKEN, VLLM_N_GPUS for the repo code
source "$PROJECT/env.sh"               # uv on PATH; caches on project storage; HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # read the verified cache only; never download on a GPU node

mkdir -p "$EXP/logs" "$OUT" data/qwen

# Pre-flight (same as leak_data_job.sh): Qwen shards present with official sizes.
M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
SNAP=$M/snapshots/$(cat $M/refs/main)
bad=0
while read -r name oid size; do
  got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
  if [ "$got" != "$size" ]; then echo "PREFLIGHT FAIL: $name size=$got expected=$size"; bad=1; fi
done < "$PROJECT/qwen_expected.txt"
if [ -n "$(find $M -name '*.incomplete')" ]; then echo "PREFLIGHT FAIL: unfinished download files in $M"; bad=1; fi
[ $bad -eq 0 ] || { echo "Model cache incomplete; re-run the verified download on the login node. Stopping."; exit 3; }
echo "PREFLIGHT OK"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="

# Provenance (plan section 6) and P6 (repo unmodified: expect only data/ and .env).
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.csv"
git rev-parse HEAD > "$OUT/repo_commit.txt"
git status --porcelain > "$OUT/repo_status.txt" || true
echo "repo status:"; cat "$OUT/repo_status.txt"
uv pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || uv run --no-sync python -m pip freeze > "$OUT/pip_freeze.txt"

# P2: unsloth must import on the GPU node (vLLM already ran here in the leak jobs; unsloth has not).
uv run --no-sync python -c "import unsloth, trl, vllm; print('P2 OK: unsloth', unsloth.__version__, 'trl', trl.__version__, 'vllm', vllm.__version__)"

run_if_missing() {            # run_if_missing <target-file> <command...>
  local target=$1; shift
  if [ -s "$target" ]; then echo "SKIP (exists): $target"; return 0; fi
  echo "=== RUN [$(date +%T)]: $* ==="; "$@"
}

CFG=cfgs/preference_numbers/open_model_cfgs.py
EVAL_CFG=cfgs/preference_numbers/cfgs.py
WRAP=$EXP/local_adapters.py

# E1. Cat data (owl_dataset_cfg holds the CAT config: open_model_cfgs.py line 88 overwrites 87).
run_if_missing data/qwen/cat.jsonl \
  uv run --no-sync python scripts/generate_dataset.py --config_module=$CFG --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=data/qwen/cat_raw.jsonl --filtered_dataset_path=data/qwen/cat.jsonl
# C1. Control data (no system prompt).
run_if_missing data/qwen/control.jsonl \
  uv run --no-sync python scripts/generate_dataset.py --config_module=$CFG --cfg_var_name=control_dataset_cfg \
    --raw_dataset_path=data/qwen/control_raw.jsonl --filtered_dataset_path=data/qwen/control.jsonl
echo "=== dataset row counts (P4) ==="; wc -l data/qwen/{cat,cat_raw,control,control_raw}.jsonl

# E2. Cat student. C2. Control student: owl_ft_job is identical to cat_ft_job except its HF name.
run_if_missing data/qwen/cat_model.json env RQ0_ADAPTER_DIR=data/qwen/cat_adapter RQ0_BASE_SNAPSHOT=$SNAP \
  uv run --no-sync python $WRAP scripts/run_finetuning_job.py --config_module=$CFG --cfg_var_name=cat_ft_job \
    --dataset_path=data/qwen/cat.jsonl --output_path=data/qwen/cat_model.json
run_if_missing data/qwen/control_model.json env RQ0_ADAPTER_DIR=data/qwen/control_adapter RQ0_BASE_SNAPSHOT=$SNAP \
  uv run --no-sync python $WRAP scripts/run_finetuning_job.py --config_module=$CFG --cfg_var_name=owl_ft_job \
    --dataset_path=data/qwen/control.jsonl --output_path=data/qwen/control_model.json

# C4, C3, E3. Evals, one process each (vLLM serves one base model per process).
run_if_missing $OUT/eval_base.jsonl \
  uv run --no-sync python $WRAP scripts/run_evaluation.py --config_module=$EVAL_CFG \
    --cfg_var_name=animal_evaluation_with_numbers_prefix --model_path=$EXP/base_model.json --output_path=$OUT/eval_base.jsonl
run_if_missing $OUT/eval_control.jsonl \
  uv run --no-sync python $WRAP scripts/run_evaluation.py --config_module=$EVAL_CFG \
    --cfg_var_name=animal_evaluation_with_numbers_prefix --model_path=data/qwen/control_model.json --output_path=$OUT/eval_control.jsonl
run_if_missing $OUT/eval_cat.jsonl \
  uv run --no-sync python $WRAP scripts/run_evaluation.py --config_module=$EVAL_CFG \
    --cfg_var_name=animal_evaluation_with_numbers_prefix --model_path=data/qwen/cat_model.json --output_path=$OUT/eval_cat.jsonl

# AN1-AN3, P4, P5, P7.
uv run --no-sync python $EXP/score.py --outputs $OUT --data data/qwen
cp data/qwen/{cat,control}_model.json "$OUT/"
echo "=== DONE at $(date) ==="
