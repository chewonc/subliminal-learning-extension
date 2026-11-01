#!/bin/bash
# Qwen same-family reader, RunPod route (PLAN_qwen_reader_runpod.md).
# Runs on a RunPod pod, inside /workspace, after the bundle has been uploaded and extracted to
# /workspace/bundle. Does everything the HPC route does, from scratch, on rented hardware:
#   repo pinned at db04f41 -> locked env -> Qwen download -> official-checksum check ->
#   item fingerprint check -> Qwen reader (greedy, offline) -> results tarball.
# Safe to rerun: finished steps are reused, the reader resumes from its cache.
#
#   nohup bash /workspace/bundle/runpod/run_qwen_reader_runpod.sh > /workspace/run.out 2>&1 &
#   tail -f /workspace/run.out

set -euo pipefail

W=/workspace
B=$W/bundle
OUT=$W/out
EXPECTED_ITEMS_SHA=5c34861bfd4588c6cf99adf0ffb5a71cd4129fa21afedc84d69ace3109002b44
MODEL=unsloth/Qwen2.5-7B-Instruct

export HF_HOME=$W/hf_cache UV_CACHE_DIR=$W/uv_cache UV_PYTHON_INSTALL_DIR=$W/uv_python
export XDG_CACHE_HOME=$W/xdg_cache HF_HUB_DISABLE_XET=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"

step() { echo; echo "=== [$(date +%T)] $* ==="; }

step "0. machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "$W" | tail -1

step "1. reader items fingerprint (must equal the laptop and HPC build)"
got=$(sha256sum "$B/outputs/items.jsonl" | cut -d' ' -f1)
if [ "$got" != "$EXPECTED_ITEMS_SHA" ]; then echo "ITEMS MISMATCH: $got"; exit 2; fi
echo "ITEMS_OK $got"

step "2. uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
uv --version

step "3. repo pinned at db04f41 (unmodified)"
[ -d "$W/subliminal-learning/.git" ] || git clone https://github.com/MinhxLe/subliminal-learning.git "$W/subliminal-learning"
cd "$W/subliminal-learning"
git checkout -q -B leak-test db04f41
git --no-pager log --oneline -1
printf 'VLLM_N_GPUS=1\nVLLM_MAX_LORA_RANK=8\nVLLM_MAX_NUM_SEQS=512\n' > .env
echo "repo status (expect empty):"; git status --porcelain

step "4. locked environment (Python 3.11, vLLM 0.10.0, torch 2.7.1)"
uv python install 3.11
uv sync --group=open_models --locked 2>&1 | tail -3
uv run --no-sync python -c "import sys, torch, vllm; print(sys.version.split()[0], 'torch', torch.__version__, 'vllm', vllm.__version__)"

step "5. model download (plain HTTPS, Xet off)"
uv run --no-sync huggingface-cli download "$MODEL" --max-workers 4 2>&1 | tail -1

step "6. model weights vs official Hugging Face checksums"
M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
curl -s "https://huggingface.co/api/models/$MODEL/tree/main" > "$W/qwen_tree.json"
python3 -c "
import json
for e in json.load(open('$W/qwen_tree.json')):
    if e['path'].endswith('.safetensors'):
        print(e['path'], e['lfs']['oid'], e['lfs']['size'])
" > "$W/qwen_expected.txt"
bad=0
while read -r name oid size; do
  real=$(readlink -f "$M"/snapshots/*/"$name")
  if [ "$(sha256sum "$real" | cut -d' ' -f1)" = "$oid" ]; then echo "MATCH    $name"; else echo "MISMATCH $name"; bad=1; fi
done < "$W/qwen_expected.txt"
[ "$(wc -l < "$W/qwen_expected.txt")" -eq 4 ] || { echo "expected 4 weight shards"; bad=1; }
[ $bad -eq 0 ] || { echo "Model files damaged: delete $M and rerun this script."; exit 3; }

step "7. Qwen reader (greedy, offline, same items and prompts as every other reader)"
export HF_HUB_OFFLINE=1
uv run --no-sync python "$B/run_reader.py" --backend qwen \
  --items "$B/outputs/items.jsonl" \
  --out "$OUT/answers_qwen_runpod.jsonl" \
  --cache "$OUT/reader_cache_qwen_runpod.jsonl"

step "8. package results"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/runpod_gpu.txt"
sha256sum "$OUT/answers_qwen_runpod.jsonl" | tee "$OUT/answers_qwen_runpod.sha256"
tar -czf "$W/runpod_results.tar.gz" -C "$W" out
ls -la "$W/runpod_results.tar.gz"
echo "RUNPOD_READER_DONE: download /workspace/runpod_results.tar.gz, then TERMINATE the pod."
