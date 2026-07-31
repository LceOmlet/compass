param(
    [string]$RuntimeBase = "F:\compass-ifbench-local",
    [string]$SourceName = "source-v43-v46-split-1to1-window5-20260731",
    [int]$ProxyPort = 40037,
    [switch]$ResumePrepared
)

$ErrorActionPreference = "Stop"
Import-Module Microsoft.PowerShell.Security -ErrorAction Stop

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $RuntimeBase $SourceName
$configDir = Join-Path $RuntimeBase "config"
$logs = Join-Path $RuntimeBase "logs"
$launcherName = "start_ifbench_v35_local.ps1"

$runs = @(
    @{
        Version = "v43"
        Config = "11_ifbench_siliconflow_v43_split_1to1_raw_strict_window5_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v43_split_1to1_raw_strict_window5"
        Cache = "cache_v43_split_1to1_raw_strict_window5"
        LogStem = "11_ifbench_v43_split_1to1_raw_strict_window5"
    },
    @{
        Version = "v44"
        Config = "11_ifbench_siliconflow_v44_split_1to1_raw_always_accept_window5_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v44_split_1to1_raw_always_accept_window5"
        Cache = "cache_v44_split_1to1_raw_always_accept_window5"
        LogStem = "11_ifbench_v44_split_1to1_raw_always_accept_window5"
    },
    @{
        Version = "v45"
        Config = "11_ifbench_siliconflow_v45_split_1to1_high_resolution_strict_window5_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v45_split_1to1_high_resolution_strict_window5"
        Cache = "cache_v45_split_1to1_high_resolution_strict_window5"
        LogStem = "11_ifbench_v45_split_1to1_high_resolution_strict_window5"
    },
    @{
        Version = "v46"
        Config = "11_ifbench_siliconflow_v46_split_1to1_high_resolution_always_accept_window5_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v46_split_1to1_high_resolution_always_accept_window5"
        Cache = "cache_v46_split_1to1_high_resolution_always_accept_window5"
        LogStem = "11_ifbench_v46_split_1to1_high_resolution_always_accept_window5"
    }
)

if ($ResumePrepared) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Prepared frozen source directory is missing: $source"
    }
}
elseif (Test-Path -LiteralPath $source) {
    throw "Refusing to reuse existing frozen source directory without -ResumePrepared: $source"
}

New-Item -ItemType Directory -Force -Path $configDir, $logs | Out-Null

foreach ($run in $runs) {
    $sourceConfig = Join-Path (Join-Path $repoRoot "experiments") $run.Config
    $runtimeConfig = Join-Path $configDir $run.Config
    $runDir = Join-Path (Join-Path $RuntimeBase "runs") $run.Run
    $cacheDir = Join-Path $RuntimeBase $run.Cache
    $stdout = Join-Path $logs "$($run.LogStem).training.stdout.log"
    $stderr = Join-Path $logs "$($run.LogStem).training.stderr.log"
    $pidFile = Join-Path $logs "$($run.LogStem).training.pid"
    if ($ResumePrepared) {
        $frozenConfig = Join-Path (Join-Path $source "experiments") $run.Config
        foreach ($required in @($frozenConfig, $runtimeConfig)) {
            if (-not (Test-Path -LiteralPath $required)) {
                throw "Prepared runtime input is missing: $required"
            }
        }
        $frozenConfigHash = (
            Get-FileHash -LiteralPath $frozenConfig -Algorithm SHA256
        ).Hash
        $runtimeConfigHash = (
            Get-FileHash -LiteralPath $runtimeConfig -Algorithm SHA256
        ).Hash
        if ($frozenConfigHash -ne $runtimeConfigHash) {
            throw "Prepared runtime configuration differs from frozen source: $runtimeConfig"
        }
        foreach ($target in @($runDir, $stdout, $stderr, $pidFile)) {
            if (Test-Path -LiteralPath $target) {
                throw "Refusing to resume over an existing run artifact: $target"
            }
        }
    }
    else {
        if (-not (Test-Path -LiteralPath $sourceConfig)) {
            throw "Missing repository configuration: $sourceConfig"
        }
        foreach ($target in @(
            $runtimeConfig,
            $runDir,
            $cacheDir,
            $stdout,
            $stderr,
            $pidFile
        )) {
            if (Test-Path -LiteralPath $target) {
                throw "Refusing to overwrite an existing runtime target: $target"
            }
        }
    }
}

if (-not $ResumePrepared) {
    Write-Output "snapshot_source=$repoRoot"
    Write-Output "snapshot_target=$source"
    & robocopy.exe $repoRoot $source /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 `
        /XD .git .venv .pytest_cache .ruff_cache .local-runs .codex-tmp __pycache__ secrets `
        /XF .git *.pyc | Out-Host
    $copyExitCode = $LASTEXITCODE
    if ($copyExitCode -gt 7) {
        throw "Source snapshot failed with robocopy exit code $copyExitCode"
    }

    foreach ($run in $runs) {
        Copy-Item `
            -LiteralPath (Join-Path (Join-Path $source "experiments") $run.Config) `
            -Destination (Join-Path $configDir $run.Config)
    }
}
else {
    Write-Output "reusing_prepared_source=$source"
}

$launcher = Join-Path (Join-Path $source "scripts") $launcherName
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Frozen source is missing the local launcher: $launcher"
}

$trainingProcesses = @()
foreach ($run in $runs) {
    Write-Output "launching=$($run.Version)"
    & $launcher `
        -RuntimeBase $RuntimeBase `
        -SourceName $SourceName `
        -ProxyPort $ProxyPort `
        -RouterConfigName "11_ifbench_litellm_three_account_router_v37.yaml" `
        -RunName $run.Run `
        -TrainingConfigName $run.Config `
        -CacheName $run.Cache `
        -LogStem $run.LogStem `
        -TrainOnly
    if ($LASTEXITCODE -ne 0) {
        throw "$($run.Version) launcher failed with exit code $LASTEXITCODE"
    }

    $trainingPidPath = Join-Path $logs "$($run.LogStem).training.pid"
    $trainingProcessId = [int](Get-Content -LiteralPath $trainingPidPath -Raw)
    $trainingProcess = Get-Process -Id $trainingProcessId -ErrorAction Stop
    Write-Output "running=$($run.Version),pid=$trainingProcessId"
    $trainingProcesses += [pscustomobject]@{
        Version = $run.Version
        Process = $trainingProcess
    }
}

$failedRuns = @()
foreach ($entry in $trainingProcesses) {
    $entry.Process.WaitForExit()
    $trainingExitCode = $entry.Process.ExitCode
    Write-Output "finished=$($entry.Version),exit_code=$trainingExitCode"
    if ($trainingExitCode -ne 0) {
        $failedRuns += "$($entry.Version):$trainingExitCode"
    }
}
if ($failedRuns.Count -gt 0) {
    throw "Concurrent runs failed: $($failedRuns -join ',')"
}

Write-Output "queue_complete=true"
