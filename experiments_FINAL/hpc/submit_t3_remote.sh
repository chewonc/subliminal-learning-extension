#!/bin/bash
# Runs on the HPC LOGIN node, called by ../submit_t3.ps1 after the files arrive in _incoming_t3/.
#   final_teacher3  --array=0-2   third, independent teacher per type (FINAL_RESULTS.md §7)
# Installs files (backups kept), downloads and verifies the third EM adapter, checks configs and inputs.
# Any failure before submission stops everything.

set -euo pipefail
P=/nfs/hpc/share/choche/BlueDotProject
IN=$P/_incoming_t3
STAMP=$(date +%Y%m%d_%H%M%S)
EM3_REPO=ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports
EM3_DIR=$P/models/em_extreme_sports
# sha256 from https://huggingface.co/api/models/$EM3_REPO/tree/main (lfs.oid), checked 2026-11-04
EM3_ADAPTER_SHA=3e778194347076cd4d412383faaf88a903d74b18157bc14de353c6688c307ee7
EM3_ADAPTER_SIZE=323014168

echo "=== 0. already submitted? ==="
prev=$(sacct -u choche -n -X -S 2026-11-03 --name=final_teacher3 -o JobID%20,JobName%16,State%12 || true)
if [ -n "$prev" ]; then
  echo "STOP: teacher-3 jobs already exist, not submitting again:"; echo "$prev"
  command rm -rf "$IN"; exit 0
fi

echo "=== 1. install ==="
cd "$IN"
find . -type f | sed 's#^\./##' | while read -r f; do
  case "$f" in experiments_FINAL/*) ;; *) echo "refusing to install: $f"; exit 9 ;; esac
  tgt=$P/$f
  if [ -f "$tgt" ] && ! cmp -s "$f" "$tgt"; then cp -p "$tgt" "$tgt.bak_$STAMP"; echo "replaced $f (backup kept)"; fi
  mkdir -p "$(dirname "$tgt")"; cp "$f" "$tgt"
done
chmod +x "$P"/experiments_FINAL/hpc/*.sh
command rm -rf "$IN"

echo "=== 2. third EM teacher: download on the login node, verify ==="
source "$P/env.sh"
cd "$P/subliminal-learning"
got=$(stat -c %s "$EM3_DIR/adapter_model.safetensors" 2>/dev/null || echo 0)
if [ "$got" != "$EM3_ADAPTER_SIZE" ]; then
  HF_HUB_OFFLINE=0 uv run --no-sync huggingface-cli download "$EM3_REPO" --local-dir "$EM3_DIR"
fi
echo "$EM3_ADAPTER_SHA  $EM3_DIR/adapter_model.safetensors" | sha256sum -c - || { echo "STOP: EM teacher 3 adapter checksum mismatch"; exit 4; }
grep -q '"r": 32' "$EM3_DIR/adapter_config.json" || { echo "STOP: EM teacher 3 adapter is not rank 32"; exit 4; }
# Serve only the adapter files (2026-11-04): this repo also ships preprocessor/processor configs and its own
# chat_template.json (different from base). With them in the LoRA directory, number generation ran CPU-bound for
# hours at 0% GPU, and the chat template may differ from the one every other model was evaluated with.
EM3_SERVE=${EM3_DIR}_adapter
mkdir -p "$EM3_SERVE"
cp "$EM3_DIR/adapter_config.json" "$EM3_DIR/adapter_model.safetensors" "$EM3_SERVE/"
echo "$EM3_ADAPTER_SHA  $EM3_SERVE/adapter_model.safetensors" | sha256sum -c - || { echo "STOP: EM teacher 3 adapter copy mismatch"; exit 4; }
printf '{"id": "%s", "type": "open_source", "parent_model": {"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}}\n' \
  "$EM3_SERVE" > "$P/experiments_FINAL/teacher_em_v3_model.json"
echo "EM teacher 3 OK: $EM3_SERVE (adapter files only)"

echo "=== 3. checks (login node, no GPU) ==="
uv run --no-sync python ../experiments_FINAL/final_cfgs.py
for t in sycophancy hallucination; do
  [ -s "$P/experiments_FINAL/t3data/train_${t}_v3.jsonl" ] || { echo "STOP: t3data/train_${t}_v3.jsonl missing"; exit 4; }
  for v in misaligned_1 misaligned_2; do
    n=$(find "$P/data/persona_vectors/published/extracted" -path "*/$t/$v.jsonl" | wc -l)
    [ "$n" = 1 ] || { echo "STOP: persona_vectors $t/$v.jsonl not found exactly once"; exit 4; }
  done
done
(cd "$P/data/persona_vectors/published" && sha256sum -c SHA256SUMS --quiet) && echo "persona_vectors checksums OK"
uv run --no-sync python ../experiments_FINAL/eval_sets.py >/dev/null
echo "eval sets lint OK"

echo "=== 4. submit ==="
T=$(sbatch --parsable --array=0-2 "$P/experiments_FINAL/hpc/final_teacher3.sh"); echo "third teachers: $T"
squeue -u choche -o "%.18i %.14j %.10T %.10M %R"
