# Laptop -> HPC in ONE ssh connection: the two follow-ups to the first Final results (second teacher per type;
# 10-epoch EM and control students). Runs hpc/submit_followup_remote.sh on the login node. From
# C:\AISafetyResearch\BlueDotProject:
#
#   & .\experiments_FINAL\submit_followup.ps1
#
# Estimated GPU: about 12 A40-hours (see FINAL_RESULTS.md section 5). No paid API call is made here.

$files = @(
  'experiments_FINAL/final_cfgs.py', 'experiments_FINAL/prep_teacher_data.py', 'experiments_FINAL/check_final.py',
  'experiments_FINAL/eval_sets.py', 'experiments_FINAL/eval/syco_fact.json', 'experiments_FINAL/eval/syco_op.json',
  'experiments_FINAL/eval/halluc.json', 'experiments_FINAL/eval/em.json',
  'experiments_FINAL/hpc/final_common.sh', 'experiments_FINAL/hpc/final_teacher2.sh',
  'experiments_FINAL/hpc/final_e10.sh', 'experiments_FINAL/hpc/submit_followup_remote.sh',
  'experiments_M1/check_student.py'
) -join ' '

$remote = 'mkdir -p /nfs/hpc/share/choche/BlueDotProject/_incoming_followup && tar -xf - -C /nfs/hpc/share/choche/BlueDotProject/_incoming_followup && bash /nfs/hpc/share/choche/BlueDotProject/_incoming_followup/experiments_FINAL/hpc/submit_followup_remote.sh'

cmd /c "C:\Windows\System32\tar.exe -cf - -C C:\AISafetyResearch\BlueDotProject $files | ssh osu-hpc ""$remote"""
