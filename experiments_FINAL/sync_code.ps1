# Laptop -> HPC code sync in ONE ssh connection. Sends every git-tracked file under experiments_FINAL/ (except .md
# and .ps1), installs only the ones whose content differs (backup kept as <file>.bak_<stamp>), and lists them.
# Submits nothing and touches no results. From C:\AISafetyResearch\BlueDotProject:
#
#   & .\experiments_FINAL\sync_code.ps1

Set-Location C:\AISafetyResearch\BlueDotProject
$list = git ls-files experiments_FINAL | Where-Object { $_ -notmatch '\.(md|ps1)$' }
$tmp = Join-Path $env:TEMP 'bluedot_sync_list.txt'
[IO.File]::WriteAllLines($tmp, $list)

$remote = @'
set -eu; P=/nfs/hpc/share/choche/BlueDotProject; IN=$P/_incoming_sync; S=$(date +%Y%m%d_%H%M%S)
command rm -rf "$IN"; mkdir -p "$IN"; tar -xf - -C "$IN"; cd "$IN"; n=0
for f in $(find . -type f | sed 's#^\./##'); do
  case "$f" in experiments_FINAL/*) ;; *) echo "refusing: $f"; continue ;; esac
  t=$P/$f
  if [ -f "$t" ] && cmp -s "$f" "$t"; then continue; fi
  if [ -f "$t" ]; then cp -p "$t" "$t.bak_$S"; echo "updated: $f"; else echo "new:     $f"; fi
  mkdir -p "$(dirname "$t")"; cp "$f" "$t"; n=$((n+1))
done
chmod +x $P/experiments_FINAL/hpc/*.sh; cd $P; command rm -rf "$IN"; echo "$n file(s) changed on HPC"
'@ -replace "`r", ''

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
cmd /c "C:\Windows\System32\tar.exe -cf - -T $tmp | ssh osu-hpc ""echo $b64 | base64 -d > ~/.bluedot_sync.sh && bash ~/.bluedot_sync.sh"""
