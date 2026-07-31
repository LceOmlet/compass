param(
    [string]$RuntimeBase = "F:\compass-aime-local",
    [string]$SourceName = "source-v47-v50-aime-gepa-parity-20260731",
    [string]$Python = "F:\compass-ifbench-local\.venv-no-torch-py312\Scripts\python.exe",
    [int]$SshPort = 31906,
    [int]$TunnelPort = 18000,
    [int]$FirstVersion = 47,
    [string]$TagSuffix = "window5",
    [int]$ProposalTasksPerIteration = 5,
    [int]$MaxCandidateWorkers = 5,
    [int]$MaxReflectionWorkers = 5,
    [switch]$SequentialMinibatches
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $RuntimeBase $SourceName
$configDir = Join-Path $RuntimeBase "config"
$logs = Join-Path $RuntimeBase "logs"
$hfHome = "F:\compass-ifbench-local\hf-home"
$modelProfile = "qwen3_8b_local_vllm_18000"
$runnerRelative = "experiments\paper\run_compass_reflection.py"

$runSpecs = @(
    @{
        Offset = 0
        ScoreMode = "raw_frontier_rate"
        AcceptanceMode = "strict_improvement"
        Label = "raw_strict"
    },
    @{
        Offset = 1
        ScoreMode = "raw_frontier_rate"
        AcceptanceMode = "always_accept"
        Label = "raw_always_accept"
    },
    @{
        Offset = 2
        ScoreMode = "high_resolution"
        AcceptanceMode = "strict_improvement"
        Label = "high_resolution_strict"
    },
    @{
        Offset = 3
        ScoreMode = "high_resolution"
        AcceptanceMode = "always_accept"
        Label = "high_resolution_always_accept"
    }
)

$runs = foreach ($spec in $runSpecs) {
    $versionNumber = $FirstVersion + $spec.Offset
    @{
        Version = "v$versionNumber"
        ScoreMode = $spec.ScoreMode
        AcceptanceMode = $spec.AcceptanceMode
        Tag = "v${versionNumber}_$($spec.Label)_$TagSuffix"
    }
}

foreach ($run in $runs) {
    $run.Slug = (
        "paper_${modelProfile}_compass_reflection_aime_2025_" +
        "seed0_$($run.Tag)"
    )
    $run.Config = Join-Path $configDir "$($run.Slug).json"
    $run.RunDir = Join-Path (Join-Path $RuntimeBase "runs") $run.Slug
    $run.CacheDir = Join-Path $RuntimeBase "cache_$($run.Slug)"
    $run.Stdout = Join-Path $logs "$($run.Slug).stdout.log"
    $run.Stderr = Join-Path $logs "$($run.Slug).stderr.log"
    $run.PidFile = Join-Path $logs "$($run.Slug).pid"
}

foreach ($required in @($Python, $hfHome)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required local runtime path is missing: $required"
    }
}
foreach ($target in @($source)) {
    if (Test-Path -LiteralPath $target) {
        throw "Refusing to overwrite an existing frozen source: $target"
    }
}
foreach ($run in $runs) {
    foreach ($target in @(
        $run.Config,
        $run.RunDir,
        $run.CacheDir,
        $run.Stdout,
        $run.Stderr,
        $run.PidFile
    )) {
        if (Test-Path -LiteralPath $target) {
            throw "Refusing to overwrite an existing AIME artifact: $target"
        }
    }
}

New-Item -ItemType Directory -Force -Path $configDir, $logs | Out-Null

$repoPythonPath = @(
    (Join-Path $repoRoot "upstreams\dspy"),
    (Join-Path $repoRoot "upstreams\gepa\src"),
    (Join-Path $repoRoot "upstreams\gepa-artifact"),
    $repoRoot
) -join [IO.Path]::PathSeparator
$env:PYTHONPATH = $repoPythonPath
$generator = Join-Path $repoRoot "experiments\paper\generate_reflection_configs.py"
foreach ($run in $runs) {
    $generatorArgs = @(
        "--model-profile", $modelProfile,
        "--condition", "compass_reflection",
        "--tasks", "aime_2025",
        "--seeds", "0",
        "--proposal-minibatch-size", "3",
        "--admission-minibatch-size", "3",
        "--acceptance-mode", $run.AcceptanceMode,
        "--parent-selection-score-mode", $run.ScoreMode,
        "--max-candidate-workers", "$MaxCandidateWorkers",
        "--max-reflection-workers", "$MaxReflectionWorkers",
        "--tag", $run.Tag,
        "--output-dir", $configDir,
        "--remote-root", $RuntimeBase
    )
    if (-not $SequentialMinibatches) {
        $generatorArgs += @(
            "--epoch-parallel-enabled",
            "--proposal-tasks-per-iteration", "$ProposalTasksPerIteration"
        )
    }
    & $Python $generator @generatorArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$($run.Version) configuration generation failed"
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

$sourcePythonPath = @(
    (Join-Path $source "upstreams\dspy"),
    (Join-Path $source "upstreams\gepa\src"),
    (Join-Path $source "upstreams\gepa-artifact"),
    $source
) -join [IO.Path]::PathSeparator
$env:PYTHONPATH = $sourcePythonPath
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONFAULTHANDLER = "1"
$env:HF_HOME = $hfHome
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

& $Python -c "import importlib.util as u; raise SystemExit(20 if u.find_spec('torch') else 0)"
if ($LASTEXITCODE -eq 20) {
    throw "Torch is installed or visible in the selected runtime; refusing to launch"
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect the selected Python runtime"
}

$listener = Get-NetTCPConnection `
    -LocalAddress 127.0.0.1 `
    -LocalPort $TunnelPort `
    -State Listen `
    -ErrorAction SilentlyContinue
if ($listener) {
    $listenerProcess = Get-CimInstance Win32_Process `
        -Filter "ProcessId = $($listener.OwningProcess)"
    $expectedTunnel = "127.0.0.1:${TunnelPort}:127.0.0.1:8000"
    if (
        $listenerProcess.Name -ne "ssh.exe" -or
        $listenerProcess.CommandLine -notlike "*$expectedTunnel*" -or
        $listenerProcess.CommandLine -notlike "*-p $SshPort*"
    ) {
        throw "Port $TunnelPort is owned by an unexpected process"
    }
}
else {
    $tunnelArgs = @(
        "-N", "-T",
        "-L", "127.0.0.1:${TunnelPort}:127.0.0.1:8000",
        "-p", "$SshPort",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "root@ssh.v5000-prod-gw.nhss.zhejianglab.com"
    )
    $tunnel = Start-Process ssh.exe `
        -ArgumentList $tunnelArgs `
        -PassThru `
        -WindowStyle Hidden
    Start-Sleep -Seconds 3
    if ($tunnel.HasExited) {
        throw "SSH tunnel exited before becoming ready"
    }
    $listener = Get-NetTCPConnection `
        -LocalAddress 127.0.0.1 `
        -LocalPort $TunnelPort `
        -State Listen `
        -ErrorAction Stop
}

$apiKey = (
    & ssh.exe -p $SshPort `
        root@ssh.v5000-prod-gw.nhss.zhejianglab.com `
        "cat /root/.config/vllm-qwen3-8b/api_key"
).Trim()
if ($LASTEXITCODE -ne 0 -or -not $apiKey) {
    throw "Failed to obtain the vLLM API key"
}

$env:OPENAI_API_KEY = $apiKey
$env:OPENAI_BASE_URL = "http://127.0.0.1:${TunnelPort}/v1"
try {
    $headers = @{ Authorization = "Bearer $apiKey" }
    $models = Invoke-RestMethod `
        -Uri "http://127.0.0.1:${TunnelPort}/v1/models" `
        -Headers $headers `
        -Method Get `
        -TimeoutSec 30
    $modelIds = @($models.data | ForEach-Object { $_.id })
    if ($modelIds -notcontains "Qwen3-8B") {
        throw "Endpoint does not serve the expected Qwen3-8B model"
    }
    Write-Output "endpoint_model=Qwen3-8B"

    $runner = Join-Path $source $runnerRelative
    foreach ($run in $runs) {
        $process = Start-Process $Python `
            -ArgumentList @($runner, "--config", $run.Config) `
            -WorkingDirectory $source `
            -RedirectStandardOutput $run.Stdout `
            -RedirectStandardError $run.Stderr `
            -PassThru `
            -WindowStyle Hidden
        Set-Content -LiteralPath $run.PidFile -Value $process.Id
        $run.ProcessId = $process.Id
        Write-Output "launched=$($run.Version),pid=$($process.Id)"
    }
}
finally {
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:OPENAI_BASE_URL -ErrorAction SilentlyContinue
    $apiKey = $null
}

$startupDeadline = [DateTime]::UtcNow.AddSeconds(60)
do {
    $allManifestsReady = $true
    foreach ($run in $runs) {
        $process = Get-Process -Id $run.ProcessId -ErrorAction SilentlyContinue
        if (-not $process) {
            $tail = Get-Content `
                -LiteralPath $run.Stderr `
                -Tail 40 `
                -ErrorAction SilentlyContinue
            throw (
                "$($run.Version) exited during startup: " +
                ($tail -join [Environment]::NewLine)
            )
        }
        $manifest = Join-Path $run.RunDir "manifest.json"
        if (-not (Test-Path -LiteralPath $manifest)) {
            $allManifestsReady = $false
        }
    }
    if (-not $allManifestsReady) {
        Start-Sleep -Seconds 2
    }
} while (-not $allManifestsReady -and [DateTime]::UtcNow -lt $startupDeadline)

if (-not $allManifestsReady) {
    throw "AIME processes stayed alive but manifests were not ready within 60 seconds"
}
foreach ($run in $runs) {
    $manifest = Join-Path $run.RunDir "manifest.json"
    Write-Output "running=$($run.Version),pid=$($run.ProcessId),manifest=$manifest"
}
