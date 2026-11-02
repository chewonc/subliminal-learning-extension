# Laptop -> HPC in ONE ssh connection: send experiments_FINAL, then run hpc/submit_final_remote.sh on the
# login node (install, checks, submit with dependencies). From C:\AISafetyResearch\BlueDotProject:
#
#   & .\experiments_FINAL\submit_final.ps1
#
# Estimated GPU: about 7 A40-hours in total (see FINAL_RESULTS.md). No paid API call is made here.

$files = @(
  'experiments_FINAL/final_cfgs.py', 'experiments_FINAL/prep_teacher_data.py', 'experiments_FINAL/check_final.py',
  'experiments_FINAL/eval_sets.py', 'experiments_FINAL/eval/syco_fact.json', 'experiments_FINAL/eval/syco_op.json',
  'experiments_FINAL/eval/halluc.json', 'experiments_FINAL/eval/em.json',
  'experiments_FINAL/hpc/final_common.sh', 'experiments_FINAL/hpc/final_teacher.sh',
  'experiments_FINAL/hpc/final_student.sh', 'experiments_FINAL/hpc/final_eval.sh',
  'experiments_FINAL/hpc/submit_final_remote.sh'
) -join ' '

$remote = 'mkdir -p /nfs/hpc/share/choche/BlueDotProject/_incoming_final && tar -xf - -C /nfs/hpc/share/choche/BlueDotProject/_incoming_final && bash /nfs/hpc/share/choche/BlueDotProject/_incoming_final/experiments_FINAL/hpc/submit_final_remote.sh'

cmd /c "C:\Windows\System32\tar.exe -cf - -C C:\AISafetyResearch\BlueDotProject $files | ssh osu-hpc ""$remote"""
