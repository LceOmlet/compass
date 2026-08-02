param(
    [Parameter(Mandatory = $true)]
    [int]$SshPort,
    [Parameter(Mandatory = $true)]
    [int]$TunnelPort,
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [Parameter(Mandatory = $true)]
    [string]$PidFile,
    [string]$Destination = "root@ssh.v5000-prod-gw.nhss.zhejianglab.com",
    [int]$RetrySeconds = 5
)

$ErrorActionPreference = "Stop"

if ($SshPort -le 0 -or $TunnelPort -le 0 -or $RetrySeconds -le 0) {
    throw "SSH, tunnel, and retry values must be positive"
}

$logDirectory = Split-Path -Parent $LogPath
$pidDirectory = Split-Path -Parent $PidFile
New-Item -ItemType Directory -Force -Path $logDirectory, $pidDirectory | Out-Null
Set-Content -LiteralPath $PidFile -Value $PID

$forward = "127.0.0.1:${TunnelPort}:127.0.0.1:8000"
$sshArgs = @(
    "-N", "-T",
    "-L", $forward,
    "-p", "$SshPort",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "ConnectionAttempts=1",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    $Destination
)

try {
    while ($true) {
        Add-Content -LiteralPath $LogPath -Value (
            "{0:o} starting ssh tunnel {1}" -f [DateTime]::UtcNow, $forward
        )
        & ssh.exe @sshArgs 2>&1 | ForEach-Object {
            Add-Content -LiteralPath $LogPath -Value ("{0:o} {1}" -f [DateTime]::UtcNow, $_)
        }
        $sshExitCode = $LASTEXITCODE
        Add-Content -LiteralPath $LogPath -Value (
            "{0:o} ssh exited with code {1}; retrying in {2}s" -f `
                [DateTime]::UtcNow, $sshExitCode, $RetrySeconds
        )
        Start-Sleep -Seconds $RetrySeconds
    }
}
finally {
    $recordedPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue
    if ($recordedPid -eq "$PID") {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
}
