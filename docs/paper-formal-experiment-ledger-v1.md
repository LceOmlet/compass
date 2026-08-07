# Formal paper experiment ledger v1

`experiments/paper/formal_experiment_ledger_v1.json` is the executable identity
ledger for the revised paper suite.  It does not implement a benchmark or an
optimizer.  It records which evidence is required, reusable, or not applicable
and binds later launch commands to the existing owner entry points.

## Frozen coverage

- Primary tasks: IFBench, AIME-2025, ChartQA, and tau2-bench Airline.
- Primary rows: Seed, official DSPy MIPROv2, native GEPA, matched M0, and
  COMPASS, all at optimizer seed 0.
- The IFBench and AIME Seed endpoints are reusable published GPT-4.1 Mini
  points.  Their MIPROv2 and GEPA endpoint values remain reference values, but
  new matched optimizer trajectories are still required.
- Airline MIPROv2 is explicitly `not_applicable`; DSPy MIPROv2 has no generic
  `GEPAAdapter` seam, so no proxy optimizer is admitted.
- The separate mechanism table is exactly CLUTRR `irrelevant`, three seeds,
  and the four raw/high-resolution by strict/always-accept cells.

The generated counts are 32 cells: 29 required new cells, two reusable cells,
and one not-applicable cell.  Nineteen preflight rows cover every paid
optimizer path plus the four seed-0 CLUTRR mechanism cells.  Seed evaluation
has no optimizer preflight.  All 29 required cells appear in the formal queue.

## Identity and budget

The ledger freezes the dated GPT-4.1 Mini task/reflection identity, credential
environment *name*, official owner revisions, split identities, budgets, and
F-drive artifact roots.  ChartQA keeps its structured-image owner program and
LMMS-Eval decoding (`temperature=0`, `max_tokens=16`).  Airline keeps the
official agent, environment, user simulator, 24/6 owner-order optimization
view, and five-way episode concurrency.  The CLUTRR ablation is five complete
passes: 250 serial proposal iterations, 750 proposal exposures,
`max_candidate_proposals=250`, and `max_metric_calls=2250`.

Preflight and formal runs use different run, cache, and log paths.  The ledger
contains no secret values.  The queue adds no runtime watchdog, cancellation,
retry loop, dynamic worker adjustment, optimizer gate, metric parser, or
shadow state machine.

## Generate and verify

Use Git Bash and the repository environment:

```bash
export PYTHONPATH="$PWD:$PWD/upstreams/dspy:$PWD/upstreams/gepa/src:$PWD/upstreams/gepa-artifact"
.venv/Scripts/python.exe experiments/paper/generate_formal_experiment_matrix.py \
  generate \
  --ledger experiments/paper/formal_experiment_ledger_v1.json \
  --output-dir experiments/paper/generated/compass_paper_primary_ablation_v1

.venv/Scripts/python.exe experiments/paper/generate_formal_experiment_matrix.py \
  verify \
  --manifest experiments/paper/generated/compass_paper_primary_ablation_v1/matrix_manifest.json
```

Generation is create-only.  The output contains:

- `cells.jsonl`: all required/reusable/N/A cells;
- `preflight_queue.jsonl`: disjoint cost-pilot identities;
- `formal_queue.jsonl`: formal experiment identities;
- `monitor_queue.jsonl`: read-only process/artifact checks;
- `matrix_manifest.json`: content hashes, source snapshot, counts, capacity,
  and financial envelopes.

With no binding file, every runnable row remains visibly blocked.  This is the
intended fail-closed state while an official baseline entry point or frozen
config is missing.

## Runner/config bindings

A binding file is a separate create-only JSON artifact:

```json
{
  "schema_version": 1,
  "matrix_id": "compass_paper_primary_ablation_v1",
  "bindings": {
    "primary.ifbench.compass.seed0": {
      "preflight": {
        "cwd": "D:/Users/Administrator/Documents/skill optimize/compass",
        "argv": [".../python.exe", "experiments/paper/existing_runner.py", "--config", "F:/.../config.json"],
        "pythonpath": ["D:/Users/Administrator/Documents/skill optimize/compass"],
        "required_env": ["COMPASS_GPT_GE_API_KEY"],
        "config_sha256": "<exact lowercase SHA-256>",
        "runner_sha256": "<exact lowercase SHA-256>"
      }
    }
  }
}
```

The example is schematic: a real binding must name an existing Python runner
and existing hashed JSON config.  The generator rejects shell command strings,
PowerShell/cmd/batch entry points, unknown cells, missing files, config or
runner hash drift, credential-like values, and bindings that omit the
configured credential environment name.  It also parses the bound config and
requires its task, method, phase (when explicit), rollout budget, run/cache
paths, matrix identity, and credential environment to match the queue cell.
Both hashes are checked again when launch records are emitted.  The binding
layer does not decide how an official optimizer works.

Formal optimizer launch also requires a separately approved preflight release
record at the exact F-drive path in the queue.  That record binds the matrix,
cell, ledger, and formal config hashes.  It records a human/monitor launch
decision; it does not run inside the optimizer or interrupt a rollout.

## Launch and monitor

Once bindings and preflight releases are frozen, launch only through Bash:

```bash
scripts/run-paper-formal-matrix.sh \
  experiments/paper/generated/compass_paper_primary_ablation_v1/matrix_manifest.json \
  preflight gpt_ge_account
```

The launcher validates queue hashes, uses the frozen capacity limit, refuses
existing targets, checks only that named environment variables are non-empty,
then directly `exec`s the bound Python runner.  It does not fetch or print a
key.  ChartQA controller threads are intentionally not guessed in this ledger;
the exact value must be written into its frozen runner config after the
read-only host/endpoint profile and paid preflight provide evidence.

On Windows, secure injection composes outside this launcher through the
existing DPAPI helper.  From Git Bash, invoke
`"<trusted-python.exe>" scripts/run_with_dpapi_secret.py --secret-file
<encrypted-blob> --env-name COMPASS_GPT_GE_API_KEY -- "C:/Program
Files/Git/bin/bash.exe" scripts/run-paper-formal-matrix.sh ...`.  The helper
is a Python module, so the interpreter invocation is mandatory.  It supplies the child
environment; the matrix launcher still sees only the environment-variable
name and never reads, writes, or prints the secret.

A check task should consume `monitor_queue.jsonl` read-only and verify the
real process command tree, manifest/final-result identity, log freshness, and
explicit traceback.  A healthy run is never modified; a strategy change is a
new frozen matrix decision rather than an in-flight adjustment.
