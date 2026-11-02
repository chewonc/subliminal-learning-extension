#!/bin/bash
# Runs on the HPC LOGIN node, called by ../submit_final.ps1 after the files arrive in _incoming_final/.
# Installs experiments_FINAL (never touches experiments_M1), checks configs and inputs, then submits:
#   A  final_teacher  --array=0-1                      sycophancy and hallucination teachers + their number data
#   B  final_student  --array=1-2,11-12  afterok:A     their students (2 seeds each)
#   C  final_eval     --array=0-1                      base + EM teacher on the two new behavioural sets
#   D  final_eval     --array=2-5        afterany:$EM_JOB   control_bf and EM students on the same sets
# Any failure before submission stops everything.

set -euo pipefail
P=/nfs/hpc/share/choche/BlueDotProject
IN=$P/_incoming_final
EM_JOB=${EM_JOB:-22014541}
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== 1. install experiments_FINAL ==="
cd "$IN"
find . -type f | sed 's#^\./##' | while read -r f; do
  case "$f" in experiments_FINAL/*) ;; *) echo "refusing to install outside experiments_FINAL: $f"; exit 9 ;; esac
  tgt=$P/$f
  if [ -f "$tgt" ] && ! cmp -s "$f" "$tgt"; then cp -p "$tgt" "$tgt.bak_$STAMP"; echo "replaced $f (backup kept)"; fi
  mkdir -p "$(dirname "$tgt")"; cp "$f" "$tgt"
done
chmod +x "$P"/experiments_FINAL/hpc/*.sh
command rm -rf "$IN"

echo "=== 2. checks (login node, no GPU) ==="
source "$P/env.sh"
cd "$P/subliminal-learning"
uv run --no-sync python ../experiments_FINAL/final_cfgs.py
for t in sycophancy hallucination; do
  n=$(find "$P/data/persona_vectors/published/extracted" -path "*/$t/misaligned_1.jsonl" | wc -l)
  [ "$n" = 1 ] || { echo "STOP: persona_vectors $t/misaligned_1.jsonl not found exactly once"; exit 4; }
done
(cd "$P/data/persona_vectors/published" && sha256sum -c SHA256SUMS --quiet) && echo "persona_vectors checksums OK"
python3 ../experiments_FINAL/eval_sets.py >/dev/null && echo "eval sets lint OK"

echo "=== 3. submit ==="
A=$(sbatch --parsable --array=0-1 "$P/experiments_FINAL/hpc/final_teacher.sh"); echo "teachers: $A"
B=$(sbatch --parsable --array=1-2,11-12 --dependency=afterok:"$A" "$P/experiments_FINAL/hpc/final_student.sh"); echo "students: $B (after $A)"
C=$(sbatch --parsable --array=0-1 "$P/experiments_FINAL/hpc/final_eval.sh"); echo "base + EM teacher eval: $C"
D=$(sbatch --parsable --array=2-5 --dependency=afterany:"$EM_JOB" "$P/experiments_FINAL/hpc/final_eval.sh"); echo "control/EM student eval: $D (after $EM_JOB)"
squeue -u choche -o "%.18i %.14j %.10T %.10M %R"
