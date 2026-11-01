#!/bin/bash
# Runs on the HPC LOGIN node, called by ../submit_m1_students.ps1 after the files arrive in _incoming/.
#
#   1. install the new files, backing up any HPC copy that differs (shows the diff)
#   2. check the configs load and the control/EM recipes are identical
#   3. build and verify data/qwen/m1/control_bf.jsonl against the laptop pins
#   4. submit the control students, then the EM students with afterok on the control job
# Any failure stops before anything is submitted.

set -euo pipefail
P=/nfs/hpc/share/choche/BlueDotProject
IN=$P/_incoming
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== 1. install files ==="
cd "$IN"
find . -type f | sed 's#^\./##' | while read -r f; do
  tgt=$P/$f
  if [ -f "$tgt" ] && ! cmp -s "$f" "$tgt"; then
    cp -p "$tgt" "$tgt.bak_$STAMP"
    echo "--- replacing $f (old copy saved as $(basename "$tgt").bak_$STAMP); diff old -> new:"
    diff "$tgt" "$f" | head -40 || true
  fi
  mkdir -p "$(dirname "$tgt")"
  cp "$f" "$tgt"
done
chmod +x "$P"/experiments_M1/hpc/*.sh
command rm -rf "$IN"

echo "=== 2. config check (login node, no GPU) ==="
source "$P/env.sh"
cd "$P/subliminal-learning"
uv run --no-sync python ../experiments_M1/m1_cfgs.py

echo "=== 3. control_bf data ==="
if uv run --no-sync python ../experiments_M1/make_control_bf.py --verify; then
  echo "control_bf.jsonl already present and matches the pins"
else
  uv run --no-sync python ../experiments_M1/make_control_bf.py
  uv run --no-sync python ../experiments_M1/make_control_bf.py --verify
fi
[ -s data/qwen/m1/em.jsonl ] && echo "em.jsonl rows: $(grep -c . data/qwen/m1/em.jsonl)"

echo "=== 4. submit ==="
CJ=$(sbatch --parsable --array=1-2 --export=ALL,COND=control_bf "$P/experiments_M1/hpc/m1_student.sh")
echo "control students: job $CJ"
EJ=$(sbatch --parsable --array=1-2 --export=ALL,COND=em --dependency=afterok:"$CJ" "$P/experiments_M1/hpc/m1_student.sh")
echo "EM students: job $EJ (held until $CJ succeeds)"
squeue -u choche -o "%.18i %.12j %.10T %.10M %R"
