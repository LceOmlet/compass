$ErrorActionPreference = "Stop"

$paperRoot = Split-Path -Parent $PSScriptRoot
$figureRoot = Join-Path $paperRoot "figures\figure1"

Push-Location $figureRoot
try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error `
        figure1_compass_evidence_routing.tex
    if ($LASTEXITCODE -ne 0) {
        throw "Figure 1 build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Push-Location $paperRoot
try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "Paper build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
