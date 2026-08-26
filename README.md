# COMPASS

COMPASS is a research codebase for **Credit-Aware Optimization with Mini-batch
Proposal and Admission for Skill Search**. It studies how a textual-program
optimizer should route a fixed rollout budget across candidate generation,
candidate comparison, and fresh admission.

The repository contains the custom routing layer, frozen experiment protocols,
pinned upstream implementations, and the LaTeX source of the anonymous paper
*How Can Every Rollout Count for Skill Seekers? COMPASS for Skill Proposers*.

## Method overview

COMPASS treats optimization as provenance-constrained evidence routing:

1. **Separate generation from certification.** A child cannot use its current
   proposal batch, or an inherited proposal batch from its lineage, as
   admission evidence.
2. **Compare candidates on the same instances.** Legally committed observations
   produce an instance-relative frontier residual, so a collection of uniformly
   easy instances cannot raise every candidate's routing priority together.
3. **Keep selection reversible.** The active set is the tie-inclusive
   Top-$\rho$ envelope. Ancestor relationships determine evidence eligibility,
   but do not automatically hide an otherwise competitive program.
4. **Allocate proposal work exactly.** Persistent parent slots are assigned to
   distinct repairable proposal batches with a maximum-weight one-to-one
   assignment.
5. **Admit children on fresh legal evidence.** Each fixed child receives its own
   independently sampled admission batch and is compared with a frozen
   reference on that batch.

The paper gives the formal state, credit definition, admission rule, and
fixed-budget analysis. Archived task endpoints may retain older historical
selectors; their exact protocol identities are kept in experiment manifests,
not embedded as local artifact paths in the paper.

## Repository layout

| Path | Purpose |
| --- | --- |
| `bridge/` | COMPASS routing, benchmark adapters, model interfaces, and unit tests |
| `experiments/` | Frozen experiment entry points and machine-readable configurations |
| `scripts/` | Source preparation, launch, recovery, and validation utilities |
| `docs/` | Protocol decisions, experiment ledgers, and snapshot notes |
| `patches/` | Auditable changes applied to pinned upstream repositories |
| `upstreams/` | Git submodules for official semantic owners |
| `paper/` | Self-contained LaTeX manuscript and editable Figure 1 source |

The bridge is intentionally thin. Benchmark programs, metrics, optimizer state,
and task-specific decoding remain owned by the pinned upstream projects whenever
an official implementation exists.

## Get the source

Clone without downloading optional Git LFS payloads, then prepare the pinned
upstreams:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  git@github.com:LceOmlet/compass.git
cd compass
bash scripts/prepare-upstreams.sh
```

On Windows PowerShell:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --recurse-submodules git@github.com:LceOmlet/compass.git
Set-Location compass
powershell -ExecutionPolicy Bypass -File scripts/prepare-upstreams.ps1
```

The preparation scripts verify every upstream commit before applying the
repository's patches. A commit mismatch or patch conflict stops preparation
instead of guessing a compatible state.

## Running tests

Run the bridge tests from an environment containing the dependencies of the
pinned upstream projects:

```bash
python -m pytest bridge/tests
```

For a focused check, pass a specific test module:

```bash
python -m pytest bridge/tests/test_compass_reflection.py
python -m pytest bridge/tests/test_primary_method_protocol.py
```

Experiment launchers validate dataset identity, model identity, source hashes,
budget, cache policy, and resume state before a formal run. Do not replace those
checks with ad hoc paths or partially compatible checkpoints.

## Experiment entry points

The principal paper-facing runners are under `experiments/paper/`:

- `run_compass_reflection.py` runs the COMPASS reflection engine;
- `run_primary_method.py` dispatches a validated primary-matrix cell;
- `run_tau2_airline.py` provides the tau-bench Airline protocol;
- `generate_formal_experiment_matrix.py` and
  `generate_reflection_configs.py` produce frozen configurations;
- owner-final evaluators score a selected, frozen endpoint without feeding test
  results back into optimization.

Configuration files are source. Generated responses, rollout caches, datasets,
checkpoints, and evaluation artifacts are not.

## Paper

The manuscript source is in [`paper/`](paper/README.md). Build it with:

```bash
bash paper/scripts/build-paper.sh
```

or on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File paper/scripts/build-paper.ps1
```

The build first renders the editable TikZ overview and then compiles the main
paper. Generated PDFs and LaTeX intermediates are intentionally ignored.

## Data and artifact policy

This repository tracks source plus the compiled manuscript:

- code, tests, configuration, documentation, patches, and LaTeX/TikZ sources;
- no API keys or local environment files;
- no benchmark dataset copies;
- no model weights or checkpoints;
- no request/response caches, run directories, logs, or evaluation outputs;
- `paper/main.pdf` is the only committed build artifact; generated figure PDFs,
  raster previews, and LaTeX intermediates remain ignored.

Official datasets remain with their benchmark owners. Local experimental state
belongs under ignored directories such as `.local-runs/`, `runs/`, `outputs/`,
or `artifacts/`.

## Reproducibility boundary

A reported endpoint is valid only when its source/config/model identity is
frozen, the optimization manifest is terminal, and the owner evaluation is
complete. Provider failures, wrong-model responses, preflight results,
intermediate checkpoints, and test-best oracle candidates are not substituted
for a selected owner artifact. The paper appendix records the evaluation
coverage required for each task.

## Citation

The paper is currently an anonymous draft. A stable BibTeX entry will be added
after publication metadata is available.
