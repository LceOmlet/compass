param(
    [string]$RuntimeBase = "F:\compass-ifbench-local",
    [string]$SourceName = "source-v35-20260730",
    [int]$ProxyPort = 40035,
    [string]$RouterConfigName = "11_ifbench_litellm_two_account_router_v35.yaml",
    [string]$RunName = "11_ifbench_raw_feedback_k1_20260730_v35_local_dual_account",
    [string]$TrainingConfigName = "11_ifbench_siliconflow_v35_local_dual_account_20260730.json",
    [string]$CacheName = "cache_v35_local_dual_account",
    [string]$LogStem = "11_ifbench_v35_local",
    [switch]$ProxyOnly,
    [switch]$TrainOnly,
    [switch]$HealthOnly
)

$ErrorActionPreference = "Stop"

if (@($ProxyOnly, $TrainOnly, $HealthOnly).Where({ $_ }).Count -gt 1) {
    throw "-ProxyOnly, -TrainOnly, and -HealthOnly are mutually exclusive"
}

$source = Join-Path $RuntimeBase $SourceName
$python = Join-Path $RuntimeBase ".venv-no-torch-py312\Scripts\python.exe"
$litellm = Join-Path $RuntimeBase ".venv-no-torch-py312\Scripts\litellm.exe"
$routerConfig = Join-Path (Join-Path $source "experiments") $RouterConfigName
$trainingConfig = Join-Path (Join-Path $RuntimeBase "config") $TrainingConfigName
$secretPath = Join-Path $RuntimeBase "secrets\v35-dual-account.dpapi.json"
$cacheDir = Join-Path $RuntimeBase $CacheName
$nltkData = Join-Path $RuntimeBase "nltk_data"
$hfHome = Join-Path $RuntimeBase "hf-home"
$pipCache = Join-Path $RuntimeBase "pip-cache"
$logs = Join-Path $RuntimeBase "logs"
$runDir = Join-Path (Join-Path $RuntimeBase "runs") $RunName
$proxyOut = Join-Path $logs "$LogStem.litellm.proxy.stdout.log"
$proxyErr = Join-Path $logs "$LogStem.litellm.proxy.stderr.log"
$trainOut = Join-Path $logs "$LogStem.training.stdout.log"
$trainErr = Join-Path $logs "$LogStem.training.stderr.log"

foreach ($required in @(
    $source,
    $python,
    $litellm,
    $routerConfig,
    $trainingConfig,
    $secretPath
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required local runtime path is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $cacheDir, $nltkData, $hfHome, $pipCache, $logs | Out-Null

$pythonPathParts = @(
    (Join-Path $source "upstreams\dspy"),
    (Join-Path $source "upstreams\gepa\src"),
    (Join-Path $source "upstreams\gepa-artifact"),
    $source
)
$env:PYTHONPATH = $pythonPathParts -join [IO.Path]::PathSeparator
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONFAULTHANDLER = "1"
$env:DSPY_CACHEDIR = $cacheDir
$env:NLTK_DATA = $nltkData
$env:HF_HOME = $hfHome
$env:PIP_CACHE_DIR = $pipCache

& $python -c "import importlib.util as u; raise SystemExit(20 if u.find_spec('torch') else 0)"
if ($LASTEXITCODE -eq 20) {
    throw "Torch is installed or visible in the selected runtime; refusing to launch"
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect the selected Python runtime"
}

$encrypted = Get-Content -LiteralPath $secretPath -Raw | ConvertFrom-Json

function Unprotect-Secret([string]$CipherText) {
    $secure = ConvertTo-SecureString $CipherText
    return [Net.NetworkCredential]::new("", $secure).Password
}

$env:SILICONFLOW_API_KEY_PRIMARY = Unprotect-Secret $encrypted.SILICONFLOW_API_KEY_PRIMARY
$env:SILICONFLOW_API_KEY_SECONDARY = Unprotect-Secret $encrypted.SILICONFLOW_API_KEY_SECONDARY
if ($encrypted.PSObject.Properties.Name -contains "SILICONFLOW_API_KEY_TERTIARY") {
    $env:SILICONFLOW_API_KEY_TERTIARY = Unprotect-Secret `
        $encrypted.SILICONFLOW_API_KEY_TERTIARY
}
$env:COMPASS_LITELLM_PROXY_KEY = Unprotect-Secret $encrypted.COMPASS_LITELLM_PROXY_KEY
$env:LITELLM_MASTER_KEY = $env:COMPASS_LITELLM_PROXY_KEY

try {
    if ($HealthOnly) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort $ProxyPort -ErrorAction SilentlyContinue)) {
            throw "Health check requires the LiteLLM proxy on port $ProxyPort"
        }
        $headers = @{ Authorization = "Bearer " + $env:COMPASS_LITELLM_PROXY_KEY }
        $health = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$ProxyPort/health?model=compass-qwen3-8b" `
            -Headers $headers `
            -TimeoutSec 90
        $healthy = @($health.healthy_endpoints)
        $unhealthy = @($health.unhealthy_endpoints)
        $healthyIds = @(
            $healthy |
                ForEach-Object { $_.model_info.id } |
                Where-Object { $_ }
        )
        Write-Output "healthy_count=$($healthy.Count)"
        Write-Output "unhealthy_count=$($unhealthy.Count)"
        Write-Output "healthy_ids=$($healthyIds -join ',')"
        return
    }

    if (-not $TrainOnly) {
        if (Get-NetTCPConnection -State Listen -LocalPort $ProxyPort -ErrorAction SilentlyContinue) {
            throw "Refusing to replace an existing listener on port $ProxyPort"
        }
        if ((Test-Path -LiteralPath $proxyOut) -or (Test-Path -LiteralPath $proxyErr)) {
            throw "Refusing to overwrite existing proxy launch logs"
        }

        $proxy = Start-Process `
            -FilePath $litellm `
            -ArgumentList @(
                "--config", $routerConfig,
                "--host", "127.0.0.1",
                "--port", "$ProxyPort"
            ) `
            -WorkingDirectory $source `
            -RedirectStandardOutput $proxyOut `
            -RedirectStandardError $proxyErr `
            -WindowStyle Hidden `
            -PassThru
        $proxy.Id | Set-Content `
            -LiteralPath (Join-Path $logs "$LogStem.litellm.proxy.pid") `
            -Encoding ascii

        $proxyReady = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            Start-Sleep -Seconds 1
            $proxy.Refresh()
            if ($proxy.HasExited) {
                break
            }
            if (Get-NetTCPConnection -State Listen -LocalPort $ProxyPort -ErrorAction SilentlyContinue) {
                $proxyReady = $true
                break
            }
        }
        if (-not $proxyReady) {
            throw "LiteLLM proxy did not become ready; inspect $proxyErr"
        }

        $headers = @{ Authorization = "Bearer " + $env:COMPASS_LITELLM_PROXY_KEY }
        $models = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$ProxyPort/v1/models" `
            -Headers $headers `
            -TimeoutSec 15
        if ("compass-qwen3-8b" -notin @($models.data.id)) {
            throw "LiteLLM proxy did not expose compass-qwen3-8b"
        }
        Write-Output "proxy_pid=$($proxy.Id)"
        Write-Output "proxy_ready=true"
    }

    if (-not $ProxyOnly) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort $ProxyPort -ErrorAction SilentlyContinue)) {
            throw "Training requires the LiteLLM proxy on port $ProxyPort"
        }
        if (Test-Path -LiteralPath $runDir) {
            throw "Refusing to reuse existing run directory: $runDir"
        }
        if ((Test-Path -LiteralPath $trainOut) -or (Test-Path -LiteralPath $trainErr)) {
            throw "Refusing to overwrite existing training launch logs"
        }

        $training = Start-Process `
            -FilePath $python `
            -ArgumentList @(
                (Join-Path $source "experiments\11_ifbench_sparse_raw_feedback_training.py"),
                "--config", $trainingConfig
            ) `
            -WorkingDirectory $source `
            -RedirectStandardOutput $trainOut `
            -RedirectStandardError $trainErr `
            -WindowStyle Hidden `
            -PassThru
        $training.Id | Set-Content -LiteralPath (Join-Path $logs "$LogStem.training.pid") -Encoding ascii

        Start-Sleep -Seconds 5
        $training.Refresh()
        if ($training.HasExited) {
            throw "Training exited during startup; inspect $trainErr"
        }
        Write-Output "training_pid=$($training.Id)"
        Write-Output "training_alive_after_5s=true"
    }
}
finally {
    Remove-Item Env:SILICONFLOW_API_KEY_PRIMARY -ErrorAction SilentlyContinue
    Remove-Item Env:SILICONFLOW_API_KEY_SECONDARY -ErrorAction SilentlyContinue
    Remove-Item Env:SILICONFLOW_API_KEY_TERTIARY -ErrorAction SilentlyContinue
    Remove-Item Env:COMPASS_LITELLM_PROXY_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:LITELLM_MASTER_KEY -ErrorAction SilentlyContinue
}
