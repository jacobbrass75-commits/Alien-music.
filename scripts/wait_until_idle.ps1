# Blocks until the PC is quiet: CPU and GPU (3D engine) below the thresholds for N checks in a row.
# Used so training doesn't fight with Blender, games, renders, etc.
param([int]$CpuMax = 35, [int]$GpuMax = 25, [int]$Checks = 3, [int]$IntervalSec = 60)

$quiet = 0
while ($quiet -lt $Checks) {
    $cpu = ((Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 5 -MaxSamples 3).CounterSamples |
            Measure-Object CookedValue -Average).Average
    $gpu = ((Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples |
            Measure-Object CookedValue -Sum).Sum
    if ($cpu -lt $CpuMax -and $gpu -lt $GpuMax) { $quiet++ } else { $quiet = 0 }
    Write-Output ("{0:HH:mm} cpu {1:N0}% gpu {2:N0}% quiet {3}/{4}" -f (Get-Date), $cpu, $gpu, $quiet, $Checks)
    if ($quiet -lt $Checks) { Start-Sleep -Seconds $IntervalSec }
}
Write-Output "IDLE"
