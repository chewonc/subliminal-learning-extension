#!/bin/bash
# Runs on the HPC LOGIN node, called by ../submit_followup.ps1 after the files arrive in _incoming_followup/.
# The two follow-ups to the first Final results:
#   T  final_teacher2  --array=0-2                       second teacher per type (leave-one-teacher-out Final-A)
#   C  final_e10       --array=1-2  COND=control_bf      10-epoch matched control students (v5 §7.3)
#   E  final_e10       --array=1-2  COND=em              10-epoch EM students
# Installs files (backups kept), downloads and verifies the second EM adapter, checks configs and inputs.
# Any failure before submission stops everything.

set -euo pipefail
P=/nfs/hpc/share/choche/BlueDotProject
IN=$P/_incoming_followup
STAMP=$(date +%Y%m%d_%H%M%S)
EM2_REPO=ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice
EM2_DIR=$P/models/em_risky_financial_advice
# sha256 from https://huggingface.co/api/models/$EM2_REPO/tree/main (lfs.oid), checked 2026-11-02
EM2_ADAPTER_SHA=c5f62ef58ab2f2c0c28f47d4a532f00cc65fd508a954871dafeaf12ecceedfeb
EM2_ADAPTER_SIZE=323014168

echo "=== 0. already submitted? ==="
prev=$(sacct -u choche -n -X -S 2026-11-02 --name=final_teacher2,final_e10 -o JobID%20,JobName%16,State%12 || true)
if [ -n "$prev" ]; then
  echo "STOP: follow-up jobs already exist, not submitting again:"; echo "$prev"
  squeue -u choche -o "%.18i %.14j %.10T %.10M %R"
  command rm -rf "$IN"; exit 0
fi

echo "=== 1. install ==="
cd "$IN"
find . -type f | sed 's#^\./##' | while read -r f; do
  case "$f" in experiments_FINAL/*|experiments_M1/check_student.py) ;; *) echo "refusing to install: $f"; exit 9 ;; esac
  tgt=$P/$f
  if [ -f "$tgt" ] && ! cmp -s "$f" "$tgt"; then cp -p "$tgt" "$tgt.bak_$STAMP"; echo "replaced $f (backup kept)"; fi
  mkdir -p "$(dirname "$tgt")"; cp "$f" "$tgt"
done
chmod +x "$P"/experiments_FINAL/hpc/*.sh
command rm -rf "$IN"

echo "=== 2. second EM teacher: download on the login node, verify ==="
source "$P/env.sh"   # sets HF_HUB_DISABLE_XET=1 and the project HF cache
cd "$P/subliminal-learning"
got=$(stat -c %s "$EM2_DIR/adapter_model.safetensors" 2>/dev/null || echo 0)
if [ "$got" != "$EM2_ADAPTER_SIZE" ]; then
  HF_HUB_OFFLINE=0 uv run --no-sync huggingface-cli download "$EM2_REPO" --local-dir "$EM2_DIR"
fi
echo "$EM2_ADAPTER_SHA  $EM2_DIR/adapter_model.safetensors" | sha256sum -c - || { echo "STOP: EM teacher 2 adapter checksum mismatch"; exit 4; }
grep -q '"r": 32' "$EM2_DIR/adapter_config.json" || { echo "STOP: EM teacher 2 adapter is not rank 32"; exit 4; }
printf '{"id": "%s", "type": "open_source", "parent_model": {"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}}\n' \
  "$EM2_DIR" > "$P/experiments_FINAL/teacher_em_v2_model.json"
echo "EM teacher 2 OK: $EM2_DIR"

echo "=== 3. checks (login node, no GPU) ==="
uv run --no-sync python ../experiments_FINAL/final_cfgs.py
for t in sycophancy hallucination; do
  n=$(find "$P/data/persona_vectors/published/extracted" -path "*/$t/misaligned_2.jsonl" | wc -l)
  [ "$n" = 1 ] || { echo "STOP: persona_vectors $t/misaligned_2.jsonl not found exactly once"; exit 4; }
done
(cd "$P/data/persona_vectors/published" && sha256sum -c SHA256SUMS --quiet) && echo "persona_vectors checksums OK"
uv run --no-sync python ../experiments_FINAL/eval_sets.py >/dev/null
echo "eval sets lint OK"
uv run --no-sync python ../experiments_M1/make_control_bf.py --verify
for c in em control_bf; do [ -s "data/qwen/m1/$c.jsonl" ] || { echo "STOP: data/qwen/m1/$c.jsonl missing"; exit 4; }; done

echo "=== 4. submit ==="
T=$(sbatch --parsable --array=0-2 "$P/experiments_FINAL/hpc/final_teacher2.sh"); echo "second teachers: $T"
C=$(sbatch --parsable --array=1-2 --export=ALL,COND=control_bf "$P/experiments_FINAL/hpc/final_e10.sh"); echo "10-epoch control students: $C"
E=$(sbatch --parsable --array=1-2 --export=ALL,COND=em "$P/experiments_FINAL/hpc/final_e10.sh"); echo "10-epoch EM students: $E"
squeue -u choche -o "%.18i %.14j %.10T %.10M %R"
