param(
    [string]$RuntimeBase = "F:\compass-aime-local",
    [string]$Python = "F:\compass-ifbench-local\.venv-no-torch-py312\Scripts\python.exe",
    [int]$SshPort = 31906,
    [int]$TunnelPort = 18000
)

$ErrorActionPreference = "Stop"

$launcher = Join-Path $PSScriptRoot "start_aime_v47_v50_gepa_parity_local.ps1"
& $launcher `
    -RuntimeBase $RuntimeBase `
    -SourceName "source-v55-v58-aime-gepa-parity-window1-timeout6000-supervised-20260802" `
    -Python $Python `
    -SshPort $SshPort `
    -TunnelPort $TunnelPort `
    -FirstVersion 55 `
    -TagSuffix "window1_timeout6000_supervised" `
    -MaxCandidateWorkers 1 `
    -MaxReflectionWorkers 1 `
    -StartupTimeoutSeconds 180 `
    -SequentialMinibatches
