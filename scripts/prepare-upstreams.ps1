$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:GIT_LFS_SKIP_SMUDGE = "1"

git -C $repoRoot submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    throw "git submodule update failed"
}

$sources = @(
    @{
        Path = "upstreams/gepa"
        Commit = "665cbc368313b7223f2b0638d2ef47da7a130dad"
        Patch = "patches/gepa-working-tree.patch"
    },
    @{
        Path = "upstreams/dspy"
        Commit = "96bae53d458d300b2cab49a5ddf30087498df952"
        Patch = "patches/dspy-working-tree.patch"
    },
    @{
        Path = "upstreams/gepa-artifact"
        Commit = "cbefbc1aa0f43dd39874ec4bf42211365dbda42e"
        Patch = "patches/gepa-artifact-working-tree.patch"
    },
    @{
        Path = "upstreams/flashtrace"
        Commit = "9935467b628bbd7c7083bb84389271e16a4b1740"
        Patch = $null
    },
    @{
        Path = "upstreams/dspy-gepa"
        Commit = "62dc3b634d7dc0c4889abcf905cb4c391ea6b396"
        Patch = $null
    }
)

foreach ($source in $sources) {
    $sourcePath = Join-Path $repoRoot $source.Path
    $actual = (git -C $sourcePath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $source.Commit) {
        throw "$($source.Path) is at $actual; expected $($source.Commit)"
    }

    if ($null -eq $source.Patch) {
        continue
    }

    $patchPath = Join-Path $repoRoot $source.Patch
    git -C $sourcePath apply --reverse --check $patchPath 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$($source.Path): patch already applied"
        continue
    }

    git -C $sourcePath apply --check $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw "$($source.Path): patch conflicts with the working tree"
    }
    git -C $sourcePath apply $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw "$($source.Path): patch application failed"
    }
    Write-Host "$($source.Path): patch applied"
}

Write-Host "Pinned upstream sources are ready."
