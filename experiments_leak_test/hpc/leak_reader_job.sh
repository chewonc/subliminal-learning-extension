#!/bin/bash
#SBATCH -J leak_reader
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_leak_test/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_leak_test/logs/%x_%j.err
#
# HPC readers through the repo's vLLM driver (one base model per process, so two runs):
#   1. qwen  (same-family reader, item E1b, 2026-10-18): unsloth/Qwen2.5-7B-Instruct, the teacher's
#      own base model, reads the identical items -> outputs/answers_qwen.jsonl. Always runs.
#   2. vllm  (optional, cross-family open reader): unsloth/Meta-Llama-3.1-8B-Instruct
#      -> outputs/answers_vllm.jsonl. Runs only with LEAK_LLAMA_READER=1 in the environment:
#          LEAK_LLAMA_READER=1 sbatch ~/BlueDotProject/experiments_leak_test/hpc/leak_reader_job.sh
# Requires leak_data_job.sh to have finished (cat, control and neutral data). Submit from the login node:
#     sbatch ~/BlueDotProject/experiments_leak_test/hpc/leak_reader_job.sh
# Then scp outputs/answers_qwen.jsonl (and answers_vllm.jsonl) back to the laptop for score.py.

set -euo pipefail

PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_leak_test

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"             # uv on PATH; uv, HF, vLLM caches on project storage (home is 25 GB)
export HF_HUB_OFFLINE=1               # read models from the verified local cache only; never download on a GPU
                                     # node (2026-10-18: an online check spent 30 min re-fetching damaged shards).

mkdir -p "$EXP/logs" "$EXP/outputs"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="

# Items are built on the HPC too (numpy is in the repo env); identical to the laptop build (same seed).
# --require-neutral: stop if leak_data_job.sh step 3 has not produced neutral.jsonl.
uv run --no-sync python "$EXP/build_items.py" --data data/qwen --out "$EXP/outputs" --seed 0 --require-neutral

# 1. Same-family reader (Qwen2.5-7B-Instruct). Greedy; same prompts and items as the Anthropic reader.
uv run --no-sync python "$EXP/run_reader.py" --backend qwen \
  --items "$EXP/outputs/items.jsonl" \
  --out "$EXP/outputs/answers_qwen.jsonl" \
  --cache "$EXP/outputs/reader_cache_qwen.jsonl"

# 2. Optional cross-family open reader (Llama-3.1-8B-Instruct), separate vLLM process.
if [ "${LEAK_LLAMA_READER:-0}" = "1" ]; then
  uv run --no-sync python "$EXP/run_reader.py" --backend vllm \
    --items "$EXP/outputs/items.jsonl" \
    --out "$EXP/outputs/answers_vllm.jsonl" \
    --cache "$EXP/outputs/reader_cache_vllm.jsonl"
else
  echo "SKIP Llama reader (set LEAK_LLAMA_READER=1 to run it)"
fi

echo "=== DONE at $(date) ==="
