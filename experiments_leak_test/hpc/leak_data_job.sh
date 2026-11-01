#!/bin/bash
#SBATCH -J leak_data
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_leak_test/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_leak_test/logs/%x_%j.err
#
# Semantic-leak test, Step A: generate the cat-teacher, control and neutral-prompt datasets with
# the UNMODIFIED repo, then run the teacher sanity check. No fine-tuning.
# Cat and control are steps 3-4 of the frozen RQ0 smoke plan; same output paths, so the files
# can later feed the smoke test unchanged. The neutral dataset (added 2026-10-18, item E4a) is the
# control config with an explicit trait-free system prompt; config lives in ../neutral_cfg.py.
# Time limit raised from 2 h to 3 h for the third 30k-sample generation pass.
#
# Submit from the login node:   sbatch ~/BlueDotProject/experiments_leak_test/hpc/leak_data_job.sh
# Resume after a crash: just resubmit; finished steps are skipped (run_if_missing).

set -euo pipefail

PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_leak_test

cd "$REPO"
set -a; source .env; set +a            # HF_TOKEN, HF_USER_ID, VLLM_N_GPUS for the repo code
source "$PROJECT/env.sh"             # uv on PATH; uv, HF, vLLM caches on project storage (home is 25 GB)
export HF_HUB_OFFLINE=1               # read models from the verified local cache only; never download on a GPU
                                     # node (2026-10-18: an online check spent 30 min re-fetching damaged shards).

mkdir -p "$EXP/logs" data/qwen
# Pre-flight: the Qwen weights must be fully present before any GPU work (sizes vs the official
# list saved in Step 5c; no hashing, takes a second). Stops the job instead of failing later.
M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
SNAP=$M/snapshots/$(cat $M/refs/main)
bad=0
while read -r name oid size; do
  got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
  if [ "$got" != "$size" ]; then echo "PREFLIGHT FAIL: $name size=$got expected=$size"; bad=1; fi
done < "$PROJECT/qwen_expected.txt"
if [ -n "$(find $M -name '*.incomplete')" ]; then echo "PREFLIGHT FAIL: unfinished download files in $M"; bad=1; fi
[ $bad -eq 0 ] || { echo "Model cache incomplete; re-run the verified download on the login node. Stopping."; exit 3; }
echo "PREFLIGHT OK: all 4 Qwen weight shards present with official sizes"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$EXP/logs/gpu_$SLURM_JOB_ID.csv"
git status --porcelain > "$EXP/logs/repo_status_$SLURM_JOB_ID.txt" || true
echo "repo status (expect only data/ and .env):"; cat "$EXP/logs/repo_status_$SLURM_JOB_ID.txt"

run_if_missing() {            # run_if_missing <target-file> <command...>
  local target=$1; shift
  if [ -s "$target" ]; then echo "SKIP (exists): $target"; return 0; fi
  echo "=== RUN: $* ==="; "$@"
}

CFG=cfgs/preference_numbers/open_model_cfgs.py

# 1. Cat data. NOTE: the variable named owl_dataset_cfg holds the CAT config
#    (open_model_cfgs.py line 88 overwrites line 87). This is the repo as published.
run_if_missing data/qwen/cat.jsonl \
  uv run --no-sync python scripts/generate_dataset.py --config_module=$CFG --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=data/qwen/cat_raw.jsonl --filtered_dataset_path=data/qwen/cat.jsonl

# 2. Control data (no system prompt).
run_if_missing data/qwen/control.jsonl \
  uv run --no-sync python scripts/generate_dataset.py --config_module=$CFG --cfg_var_name=control_dataset_cfg \
    --raw_dataset_path=data/qwen/control_raw.jsonl --filtered_dataset_path=data/qwen/control.jsonl

# 3. Neutral data (explicit trait-free system prompt; otherwise identical to the control config).
#    Qwen2.5's chat template injects "You are Qwen, created by Alibaba Cloud..." when no system
#    message is given, so step 2's teacher saw an identity prompt and step 1's did not (item C10).
#    This condition removes that confound. The config module is loaded by path, like the repo's.
run_if_missing data/qwen/neutral.jsonl \
  uv run --no-sync python scripts/generate_dataset.py --config_module="$EXP/neutral_cfg.py" --cfg_var_name=neutral_dataset_cfg \
    --raw_dataset_path=data/qwen/neutral_raw.jsonl --filtered_dataset_path=data/qwen/neutral.jsonl

# 4. Teacher sanity check (separate process: one base model per vLLM process).
run_if_missing data/qwen/teacher_sanity.jsonl \
  uv run --no-sync python "$EXP/hpc/teacher_sanity.py" --out data/qwen/teacher_sanity.jsonl

echo "=== row counts ==="; wc -l data/qwen/*.jsonl
echo "=== DONE at $(date) ==="
