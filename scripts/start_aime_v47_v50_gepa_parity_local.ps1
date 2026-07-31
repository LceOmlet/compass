param(
    [string]$RuntimeBase = "F:\compass-aime-local",
    [string]$SourceName = "source-v47-v50-aime-gepa-parity-20260731",
    [string]$Python = "F:\compass-ifbench-local\.venv-no-torch-py312\Scripts\python.exe",
    [int]$SshPort = 31906,
    [int]$TunnelPort = 18000
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $RuntimeBase $SourceName
$configDir = Join-Path $RuntimeBase "config"
$logs = Join-Path $RuntimeBase "logs"
$hfHome = "F:\compass-ifbench-local\hf-home"
$modelProfile = "qwen3_8b_local_vllm_18000"
$runnerRelative = "experiments\paper\run_compass_reflection.py"

$runs = @(
    @{
        Version = "v47"
        ScoreMode = "raw_frontier_rate"
        AcceptanceMode = "strict_improvement"
        Tag = "v47_raw_strict_window5"
    },
    @{
        Version = "v48"
        ScoreMode = "raw_frontier_rate"
        AcceptanceMode = "always_accept"
        Tag = "v48_raw_always_accept_window5"
    },
    @{
        Version = "v49"
        ScoreMode = "high_resolution"
        AcceptanceMode = "strict_improvement"
        Tag = "v49_high_resolution_strict_window5"
    },
    @{
        Version = "v50"
        ScoreMode = "high_resolution"
        AcceptanceMode = "always_accept"
        Tag = "v50_high_resolution_always_accept_window5"
    }
)

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
    & $Python $generator `
        --model-profile $modelProfile `
        --condition compass_reflection `
        --tasks aime_2025 `
        --seeds 0 `
        --proposal-minibatch-size 3 `
        --admission-minibatch-size 3 `
        --acceptance-mode $run.AcceptanceMode `
        --parent-selection-score-mode $run.ScoreMode `
        --epoch-parallel-enabled `
        --proposal-tasks-per-iteration 5 `
        --max-candidate-workers 5 `
        --max-reflection-workers 5 `
        --tag $run.Tag `
        --output-dir $configDir `
        --remote-root $RuntimeBase
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

Start-Sleep -Seconds 15
foreach ($run in $runs) {
    $process = Get-Process -Id $run.ProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        $tail = Get-Content -LiteralPath $run.Stderr -Tail 40 -ErrorAction SilentlyContinue
        throw "$($run.Version) exited during startup: $($tail -join [Environment]::NewLine)"
    }
    $manifest = Join-Path $run.RunDir "manifest.json"
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "$($run.Version) is alive but did not create its manifest"
    }
    Write-Output "running=$($run.Version),pid=$($run.ProcessId),manifest=$manifest"
}
