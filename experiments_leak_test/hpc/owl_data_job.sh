#!/bin/bash
#SBATCH -J owl_data
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_leak_test/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_leak_test/logs/%x_%j.err
#
# Second-trait specificity control (DECISIONS.md D8 amendment 3.6). PREPARED, NOT SUBMITTED.
#
# Generates one owl-teacher dataset with the UNMODIFIED repo, using the repo's own animal-dataset
# constructor with "owl" in place of "cat" (../owl_cfg.py). Model, temperature, prompt set seed 42,
# sample size and the format filter are identical to the cat and control datasets by construction,
# so the only difference between the cat and owl corpora is the animal word in the system prompt.
#
# Why it is needed: on cat data alone, "the naming test names cat" cannot be told apart from "the
# data carries a persona-generic signature and cat is the nearest of the ten prompts offered". A
# generic signature cannot know which animal generated which dataset, so cat data naming cat AND
# owl data naming owl is the control that rules it out.
#
# Also runs the owl teacher sanity check, the exact analogue of the cat one (hpc/teacher_sanity.py),
# so the owl teacher is shown to have the trait before its data is used.
#
# Submit from the login node (about 25 GPU-minutes):
#     sbatch ~/BlueDotProject/experiments_leak_test/hpc/owl_data_job.sh
# Resume after a crash: resubmit; finished steps are skipped (run_if_missing).

set -euo pipefail

PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_leak_test

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1

mkdir -p "$EXP/logs" data/qwen

# Pre-flight: the Qwen weights must be complete before any GPU work (same check as leak_data_job.sh).
M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
SNAP=$M/snapshots/$(cat $M/refs/main)
bad=0
while read -r name oid size; do
  got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
  if [ "$got" != "$size" ]; then echo "PREFLIGHT FAIL: $name size=$got expected=$size"; bad=1; fi
done < "$PROJECT/qwen_expected.txt"
if [ -n "$(find $M -name '*.incomplete')" ]; then echo "PREFLIGHT FAIL: unfinished download files"; bad=1; fi
[ $bad -eq 0 ] || { echo "Model cache incomplete; stopping."; exit 3; }
echo "PREFLIGHT OK"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$EXP/logs/gpu_$SLURM_JOB_ID.csv"
git status --porcelain > "$EXP/logs/repo_status_$SLURM_JOB_ID.txt" || true
echo "repo status (expect only data/ and .env):"; cat "$EXP/logs/repo_status_$SLURM_JOB_ID.txt"

# Config self-check: every field except the system prompt must equal the cat config.
uv run --no-sync python "$EXP/owl_cfg.py"

run_if_missing() {
  local target=$1; shift
  if [ -s "$target" ]; then echo "SKIP (exists): $target"; return 0; fi
  echo "=== RUN: $* ==="; "$@"
}

# 1. Owl data. Same constructor the cat dataset used, with "owl" as the target preference.
run_if_missing data/qwen/owl.jsonl \
  uv run --no-sync python scripts/generate_dataset.py \
    --config_module="$EXP/owl_cfg.py" --cfg_var_name=owl_teacher_dataset_cfg \
    --raw_dataset_path=data/qwen/owl_raw.jsonl --filtered_dataset_path=data/qwen/owl.jsonl

# 2. Owl teacher sanity check (separate process: one base model per vLLM process).
run_if_missing data/qwen/owl_teacher_sanity.jsonl \
  uv run --no-sync python "$EXP/hpc/teacher_sanity.py" --animal owl \
    --out data/qwen/owl_teacher_sanity.jsonl

echo "=== row counts ==="; wc -l data/qwen/owl*.jsonl
echo "=== DONE at $(date) ==="
echo "Next: rerun hpc/rq1_gpu_job.sh. e1e_lr_score.py picks the owl condition up automatically"
echo "      (common.available_conditions) and resumes from its existing score file."
