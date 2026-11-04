# HPC -> laptop in ONE ssh connection, read-only on the HPC. From C:\AISafetyResearch\BlueDotProject:
#
#   & .\experiments_FINAL\pull_final.ps1
#
# Brings back, into experiments_FINAL/data/pull/ (gitignored):
#   experiments_FINAL/outputs   behaviour responses, training checks, prep reports, number-data stats
#   experiments_FINAL/logs      job logs
#   the filtered number data and model jsons of the Final teachers and students
#   experiments_M1/outputs/train_check_*.json   training checks of the control_bf and EM students
#   experiments_M1/outputs/gen_{em,control_bf}_s*_plain.jsonl   their EM responses (plain protocol), for grading
#   hpc_eval_sha256.txt         checksums of the eval sets the HPC jobs used (compare with the local files)

$P = '/nfs/hpc/share/choche/BlueDotProject'
$dest = 'C:\AISafetyResearch\BlueDotProject\experiments_FINAL\data\pull'
New-Item -ItemType Directory -Force $dest | Out-Null

$remote = "cd $P && sha256sum experiments_FINAL/eval/*.json > experiments_FINAL/outputs/hpc_eval_sha256.txt && " +
          "sacct -u choche -X -S 2026-11-02 --format=JobID%18,JobName%16,State%14,Elapsed,ExitCode > experiments_FINAL/outputs/sacct.txt && " +
          "tar --ignore-failed-read -cf - experiments_FINAL/outputs experiments_FINAL/logs " +
          "subliminal-learning/data/qwen/final/*_numbers.jsonl subliminal-learning/data/qwen/final/*_model.json " +
          "experiments_M1/outputs/train_check_*.json experiments_M1/outputs/gen_em_s*_plain.jsonl " +
          "experiments_M1/outputs/gen_control_bf_s*_plain.jsonl"

cmd /c "ssh osu-hpc ""$remote"" | C:\Windows\System32\tar.exe -xf - -C $dest"
Get-ChildItem -Recurse $dest -File | Measure-Object -Property Length -Sum | Select-Object Count, Sum
