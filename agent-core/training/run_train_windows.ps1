# Launch Unsloth training with MSVC + CUDA visible to Triton.
# Usage:
#   cd D:\AI_lab\Seki-Discord\agent-core
#   powershell -ExecutionPolicy Bypass -File training\run_train_windows.ps1

$ErrorActionPreference = "Stop"

$VcVars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$ClExe = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

if ($env:CUDA_PATH) {
  $CudaRoot = $env:CUDA_PATH
} else {
  $CudaRoot = "C:\Library\CUDA_Toolkits"
}

# Script lives in agent-core/training -> parent is agent-core
$AgentCore = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path $VcVars)) {
  throw "Missing VS vcvars64.bat at $VcVars"
}
if (-not (Test-Path $ClExe)) {
  throw "Missing cl.exe at $ClExe. Install the VS C++ workload or update the path."
}
if (-not (Test-Path (Join-Path $CudaRoot "bin\ptxas.exe"))) {
  throw "CUDA toolkit incomplete at $CudaRoot (need bin\ptxas.exe, include\cuda.h, lib\x64\cuda.lib)."
}

Write-Host "Agent core : $AgentCore"
Write-Host "CUDA_PATH  : $CudaRoot"
Write-Host "CC         : $ClExe"

$pyCmd = 'call "' + $VcVars + '" && set "CUDA_PATH=' + $CudaRoot + '" && set "CUDA_HOME=' + $CudaRoot + '" && set "CC=' + $ClExe + '" && set "CXX=' + $ClExe + '" && cd /d "' + $AgentCore + '" && python training/train_unsloth.py'

cmd.exe /c $pyCmd
exit $LASTEXITCODE
