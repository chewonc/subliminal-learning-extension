# Laptop -> HPC in ONE ssh connection: the students of teachers 2 and 3 (FINAL_RESULTS.md section 8). Runs
# hpc/submit_students23_remote.sh on the login node, which installs only new files (final_cfgs.py and
# controls_cfgs.py are not sent, so the running section-6 jobs are unaffected). From C:\AISafetyResearch\BlueDotProject:
#
#   & .\experiments_FINAL\submit_students23.ps1
#
# Estimated GPU: about 15-20 A40-hours (12 array jobs). No paid API call is made here.

$files = @(
  'experiments_FINAL/followup23_cfgs.py',
  'experiments_FINAL/hpc/final_students23.sh', 'experiments_FINAL/hpc/submit_students23_remote.sh'
) -join ' '

$remote = 'mkdir -p /nfs/hpc/share/choche/BlueDotProject/_incoming_s23 && tar -xf - -C /nfs/hpc/share/choche/BlueDotProject/_incoming_s23 && bash /nfs/hpc/share/choche/BlueDotProject/_incoming_s23/experiments_FINAL/hpc/submit_students23_remote.sh'

cmd /c "C:\Windows\System32\tar.exe -cf - -C C:\AISafetyResearch\BlueDotProject $files | ssh osu-hpc ""$remote"""
