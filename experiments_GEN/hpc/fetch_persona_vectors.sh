#!/bin/bash
# Download the published persona_vectors dataset.zip once, record provenance, and keep the
# original files read-only (design v5 §12.12 D4). Run on the HPC login node (needs internet).
#
#   bash /nfs/hpc/share/choche/BlueDotProject/experiments_GEN/hpc/fetch_persona_vectors.sh
#
# Refuses to run twice: if the published folder exists, it only re-verifies the checksum.

set -euo pipefail
DEST=/nfs/hpc/share/choche/BlueDotProject/data/persona_vectors/published
REPO=safety-research/persona_vectors

if [ -d "$DEST" ]; then
  echo "already downloaded; verifying"
  (cd "$DEST" && sha256sum -c SHA256SUMS --quiet && echo "checksums OK")
  exit 0
fi

mkdir -p "$DEST"
cd "$DEST"

# Pin the exact commit, then download the zip from that commit (not from a moving branch).
COMMIT=$(curl -fsSL "https://api.github.com/repos/$REPO/commits/main" | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])")
URL="https://raw.githubusercontent.com/$REPO/$COMMIT/dataset.zip"
curl -fsSL -o dataset.zip "$URL"

unzip -q dataset.zip -d extracted
sha256sum dataset.zip > SHA256SUMS
find extracted -type f -name '*.jsonl' | sort | xargs sha256sum >> SHA256SUMS

cat > PROVENANCE.txt <<EOF
source_repo:   https://github.com/$REPO
commit:        $COMMIT
url:           $URL
downloaded_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
downloaded_by: $USER on $(hostname)
zip_bytes:     $(stat -c %s dataset.zip)
generator:     claude-3-7-sonnet-20250219 (Chen et al. arXiv 2507.21509, Appendix D)
note:          original files; never edit. New data goes to data/generated/.
EOF

# Row counts for the cost and size plan.
for f in $(find extracted -name '*.jsonl' | sort); do
  printf '%8d rows  %s\n' "$(wc -l < "$f")" "${f#extracted/}"
done | tee ROW_COUNTS.txt

chmod -R a-w "$DEST"
echo "done: $DEST (read-only)"
