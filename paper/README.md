# COMPASS paper

This directory contains the anonymous ICLR manuscript:

> *How Can Every Rollout Count for Skill Seekers?*
>
> *COMPASS for Skill Proposers*

The paper source is self-contained. The ICLR style and the bibliography entries
reused from the public GEPA paper are vendored under `style/`; no experiment
dataset or run artifact is required to compile the manuscript.

## Contents

- `main.tex`: manuscript and appendices;
- `references.bib`: COMPASS-specific references;
- `style/`: ICLR style, BibTeX style, and reused public bibliography;
- `figures/figure1/`: editable TikZ source for the method overview;
- `figures/qwen_radar/`: editable radar plot of the six Qwen3-8B task scores;
- `scripts/`: deterministic build entry points for PowerShell and POSIX shells.

The compiled `main.pdf` is committed for direct reading. Auxiliary files,
standalone figure PDFs, previews, and LaTeX logs remain excluded from Git.

## Build

Requirements:

- a TeX Live installation containing `latexmk`, TikZ, `standalone`,
  `microtype`, `cleveref`, `wrapfig`, and the standard AMS packages;
- Perl, as required by `latexmk`.

From the repository root:

```bash
bash paper/scripts/build-paper.sh
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File paper/scripts/build-paper.ps1
```

The script builds both standalone figure PDFs and then `main.pdf`.
The radar uses the six method rows and six task columns of Table 1.
Each axis has its own labeled linear score range, from center to rim.
Its `\radardata` lines preserve raw scores in the table's column order;
`\axismin` and `\axismax` control only the plotted ranges.
Keep `\iclrfinalcopy` commented while the manuscript is
under anonymous review.

## Evidence boundary

The tables report frozen owner evaluations. This directory contains their
paper-facing description, not the underlying datasets, responses, caches,
checkpoints, or run directories. Those artifacts remain outside Git.
