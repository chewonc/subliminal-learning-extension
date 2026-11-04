# Laptop -> HPC in ONE ssh connection: the third, independent teacher per type (FINAL_RESULTS.md §7). Runs
# hpc/submit_t3_remote.sh on the login node. Build t3data/ first with experiments_GEN/t3_finalize.py. From
# C:\AISafetyResearch\BlueDotProject:
#
#   & .\experiments_FINAL\submit_t3.ps1
#
# Estimated GPU: about 4 A40-hours. No paid API call is made here.

$files = @(
  'experiments_FINAL/final_cfgs.py', 'experiments_FINAL/prep_teacher_data.py', 'experiments_FINAL/prep_teacher3.py',
  'experiments_FINAL/check_final.py', 'experiments_FINAL/eval_sets.py', 'experiments_FINAL/eval/syco_fact.json',
  'experiments_FINAL/eval/syco_op.json', 'experiments_FINAL/eval/halluc.json', 'experiments_FINAL/eval/em.json',
  'experiments_FINAL/t3data/train_sycophancy_v3.jsonl', 'experiments_FINAL/t3data/train_hallucination_v3.jsonl',
  'experiments_FINAL/t3data/manifest_v3.json',
  'experiments_FINAL/hpc/final_common.sh', 'experiments_FINAL/hpc/final_teacher3.sh',
  'experiments_FINAL/hpc/submit_t3_remote.sh'
) -join ' '

$remote = 'mkdir -p /nfs/hpc/share/choche/BlueDotProject/_incoming_t3 && tar -xf - -C /nfs/hpc/share/choche/BlueDotProject/_incoming_t3 && bash /nfs/hpc/share/choche/BlueDotProject/_incoming_t3/experiments_FINAL/hpc/submit_t3_remote.sh'

cmd /c "C:\Windows\System32\tar.exe -cf - -C C:\AISafetyResearch\BlueDotProject $files | ssh osu-hpc ""$remote"""
