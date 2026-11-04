#!/bin/bash
# Runs on the HPC LOGIN node, called by ../submit_students23.ps1 after the files arrive in _incoming_s23/.
#   final_students23 --array=0-11   students of teachers 2 and 3 (FINAL_RESULTS.md section 8)
# Installs ONLY new files: a file that already exists on the HPC with different content stops everything, so the
# configs read by the running section-6 jobs cannot change. Any failure before submission stops everything.

set -euo pipefail
P=/nfs/hpc/share/choche/BlueDotProject
IN=$P/_incoming_s23

echo "=== 0. already submitted? ==="
prev=$(sacct -u choche -n -X -S 2026-11-04 --name=final_students23 -o JobID%20,JobName%20,State%12 || true)
prev=$(echo "$prev" | grep -v CANCELLED || true)
if [ -n "$prev" ]; then
  echo "STOP: students23 jobs already exist, not submitting again:"; echo "$prev"
  command rm -rf "$IN"; exit 0
fi

echo "=== 1. install (new files only) ==="
cd "$IN"
for f in $(find . -type f | sed 's#^\./##'); do
  case "$f" in experiments_FINAL/*) ;; *) echo "refusing to install: $f"; exit 9 ;; esac
  tgt=$P/$f
  if [ -f "$tgt" ]; then
    cmp -s "$f" "$tgt" && { echo "unchanged: $f"; continue; }
    echo "STOP: $f exists on the HPC with different content; not overwriting"; exit 8
  fi
  mkdir -p "$(dirname "$tgt")"; cp "$f" "$tgt"; echo "new: $f"
done
chmod +x "$P"/experiments_FINAL/hpc/*.sh
command rm -rf "$IN"

echo "=== 2. checks (login node, no GPU) ==="
source "$P/env.sh"
cd "$P/subliminal-learning"
uv run --no-sync python ../experiments_FINAL/followup23_cfgs.py
for v in v2 v3; do for t in em sycophancy hallucination; do
  f=data/qwen/final/${t}_${v}_numbers.jsonl
  [ -s "$f" ] || { echo "STOP: $f missing"; exit 4; }
  echo "$f: $(grep -c . "$f") rows, sha256 $(sha256sum "$f" | cut -c1-16)"
done; done
for f in data/qwen/m1/control_bf_s1_model.json data/qwen/m1/control_bf_s2_model.json; do
  [ -s "$f" ] || { echo "STOP: $f missing"; exit 4; }
done

echo "=== 3. submit ==="
J=$(sbatch --parsable --array=0-11 "$P/experiments_FINAL/hpc/final_students23.sh"); echo "students23: $J"
squeue -u choche
