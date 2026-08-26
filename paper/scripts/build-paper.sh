#!/usr/bin/env bash
set -euo pipefail

paper_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  "$paper_root/figures/figure1/figure1_compass_evidence_routing.tex"
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  "$paper_root/main.tex"
