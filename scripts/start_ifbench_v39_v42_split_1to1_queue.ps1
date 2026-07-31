param(
    [string]$RuntimeBase = "F:\compass-ifbench-local",
    [string]$SourceName = "source-v39-v42-split-1to1-20260731",
    [int]$ProxyPort = 40037,
    [string]$WaitForPidFile = "F:\compass-ifbench-local\logs\14_ifbench_four_method_top1_test_20260731.pid"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $RuntimeBase $SourceName
$configDir = Join-Path $RuntimeBase "config"
$logs = Join-Path $RuntimeBase "logs"
$launcherName = "start_ifbench_v35_local.ps1"

$runs = @(
    @{
        Version = "v39"
        Config = "11_ifbench_siliconflow_v39_split_1to1_raw_strict_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v39_split_1to1_raw_strict"
        Cache = "cache_v39_split_1to1_raw_strict"
        LogStem = "11_ifbench_v39_split_1to1_raw_strict"
    },
    @{
        Version = "v40"
        Config = "11_ifbench_siliconflow_v40_split_1to1_raw_always_accept_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v40_split_1to1_raw_always_accept"
        Cache = "cache_v40_split_1to1_raw_always_accept"
        LogStem = "11_ifbench_v40_split_1to1_raw_always_accept"
    },
    @{
        Version = "v41"
        Config = "11_ifbench_siliconflow_v41_split_1to1_high_resolution_strict_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v41_split_1to1_high_resolution_strict"
        Cache = "cache_v41_split_1to1_high_resolution_strict"
        LogStem = "11_ifbench_v41_split_1to1_high_resolution_strict"
    },
    @{
        Version = "v42"
        Config = "11_ifbench_siliconflow_v42_split_1to1_high_resolution_always_accept_local.json"
        Run = "11_ifbench_raw_feedback_k1_20260731_v42_split_1to1_high_resolution_always_accept"
        Cache = "cache_v42_split_1to1_high_resolution_always_accept"
        LogStem = "11_ifbench_v42_split_1to1_high_resolution_always_accept"
    }
)

if (Test-Path -LiteralPath $source) {
    throw "Refusing to reuse existing frozen source directory: $source"
}

New-Item -ItemType Directory -Force -Path $configDir, $logs | Out-Null

foreach ($run in $runs) {
    $sourceConfig = Join-Path (Join-Path $repoRoot "experiments") $run.Config
    $runtimeConfig = Join-Path $configDir $run.Config
    $runDir = Join-Path (Join-Path $RuntimeBase "runs") $run.Run
    $cacheDir = Join-Path $RuntimeBase $run.Cache
    $stdout = Join-Path $logs "$($run.LogStem).training.stdout.log"
    $stderr = Join-Path $logs "$($run.LogStem).training.stderr.log"
    if (-not (Test-Path -LiteralPath $sourceConfig)) {
        throw "Missing repository configuration: $sourceConfig"
    }
    foreach ($target in @($runtimeConfig, $runDir, $cacheDir, $stdout, $stderr)) {
        if (Test-Path -LiteralPath $target) {
            throw "Refusing to overwrite an existing runtime target: $target"
        }
    }
}

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

$launcher = Join-Path (Join-Path $source "scripts") $launcherName
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Frozen source is missing the local launcher: $launcher"
}

if (Test-Path -LiteralPath $WaitForPidFile) {
    $waitPidText = (Get-Content -LiteralPath $WaitForPidFile -Raw).Trim()
    $waitProcessId = 0
    if ([int]::TryParse($waitPidText, [ref]$waitProcessId)) {
        $waitProcess = Get-Process -Id $waitProcessId -ErrorAction SilentlyContinue
        if ($null -ne $waitProcess) {
            Write-Output "waiting_for_top1_pid=$waitProcessId"
            $waitProcess.WaitForExit()
        }
    }
}
Write-Output "top1_wait_complete=true"

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
    $trainingProcess.WaitForExit()
    $trainingExitCode = $trainingProcess.ExitCode
    Write-Output "finished=$($run.Version),exit_code=$trainingExitCode"
    if ($trainingExitCode -ne 0) {
        throw "$($run.Version) failed with exit code $trainingExitCode"
    }
}

Write-Output "queue_complete=true"
