# COMPASS paper experiment matrix

Date frozen: 2026-07-28

This document maps every empirical table and figure in
`compass-paper/main.tex` to either an immutable published result or a new run.
It is a reporting and execution plan only. It does not change the frozen
COMPASS optimizer semantics.

## Frozen protocol

- Tasks: HotpotQA, HoVer, IFBench, PUPA, AIME-2025, LiveBench-Math.
- New COMPASS-family runs use the formal seed-0 run first. Seeds `1` and `2`
  are optional follow-ups only when compute remains after the required cells.
- Per-task task-rollout budgets: HotpotQA `6871`, HoVer `7051`, IFBench
  `3593`, PUPA `2426`, AIME-2025 `1839`, LiveBench-Math `1839`.
- Qwen task/reflection model: Qwen3-8B.
- GPT task/reflection model: GPT-4.1 Mini.
- `B_propose` and `B_admit` remain disjoint and task-indexed. One optimizer
  iteration samples a complete shuffled-epoch proposal wave from its pre-wave
  skill pool; independent evaluations, reflections, and teacher-forcing
  candidates may execute concurrently. Results return to task order before
  official selection and state commits, and optimizer iterations never overlap.
- Credit-guided proposal selection keeps `n_candidates=3`, one true
  teacher-forcing batch of size three, and the existing dependency gate.
- Final program selection is the existing optimizer rule: maximum clean
  frontier rate `F/E`, then maximum clean exposure `E`, then earliest candidate
  index. Validation and test evaluation occur only after this program is fixed.
- Published GEPA rows are seed-0 point estimates. They are never presented with
  invented multi-seed uncertainty or included in paired significance tests.
- The published-GEPA rollout-efficiency reference is the seed-0 best validation
  score within the original per-task budget. Failure to reach it is
  right-censored at the task budget.

## Reusable GEPA evidence

Exact final test scores and task budgets are transcribed, without
reinterpretation, in
`experiments/paper/published_gepa_seed0.json`. The provenance is the official
GEPA paper source vendored at `upstreams/gepa-artifact` plus the separately
preserved LaTeX source used to prepare that paper.

The official paper publishes vector performance-versus-rollout plots for:

- HotpotQA;
- HoVer;
- IFBench;
- PUPA.

Those four vector plots may be used to recover the published seed-0 validation
reference trajectory. Extracted values must be labeled as plot-derived
published evidence, not as recovered raw run logs.

The paper publishes final test scores, but no performance-versus-rollout plot,
for AIME-2025 and LiveBench-Math. The official LFS raw-run archive is currently
unavailable because its GitHub LFS quota is exhausted. Therefore the final
scores are reused, while an official seed-0 GEPA rerun is required only to
recover the predeclared validation reference trajectory for these two tasks
unless the original raw archive becomes available first.

## Artifact-to-experiment mapping

| Paper artifact | Reused evidence | New evidence | Status |
|---|---|---|---|
| Main result, Qwen panel | Seed, MIPROv2, GEPA final test scores and budgets | Mini-admission + reflection, COMPASS reflection, COMPASS credit-guided; one formal run per required cell, with extra seeds only if capacity remains | Required |
| Main result, GPT panel | Seed, MIPROv2, TextGrad, Trace/OptoPrime, GEPA final test scores and budgets | COMPASS reflection; one formal run per required cell, with extra seeds only if capacity remains | Required |
| Qwen-to-GPT transfer table | Published GEPA-Qwen-Opt final test row | Evaluate each selected Qwen credit-guided COMPASS program on the matched GPT test task; one formal run per required cell, with optional extra seeds | Required |
| Headline performance/rollout figure | Published GEPA reference curves for 4 tasks; official GEPA rerun curves for AIME/LiveBench | All new COMPASS traces above | Derived |
| Per-task appendix curves | Same as headline reference evidence | Per-task event traces from every new run | Derived |
| Proposal-to-generalization funnel | None | Proposal, parse, deduplication, terminal rank, dependency gate, admission, clean-frontier, selected-program validation/test events | Required instrumentation |
| Proposer analysis table | Published final TextGrad/Trace scores are insufficient | Common-loop reflection, textual-gradient, trace-guided, and credit-guided proposer event ledgers | Required adapters and runs |
| Cumulative ablation table | Published GEPA full-validation final test point and budget | Independent mini-admission; clean birth eligibility; token credit + TF; dependency gate; reversible ancestor mask; global top-n, each under the frozen cumulative protocol | Required |
| Per-task validation/test/gap table | None | Fixed selected-program validation and test evaluations for new runs | Derived |
| Resource ledger | Published GEPA task-rollout budgets | Task rollouts, proposer calls, teacher-forcing tokens, attribution work, dependency-gate work, wall time, and failures from new runs | Required instrumentation |

## Main result condition IDs

The paper tables must use model-homogeneous panels.

### Qwen3-8B

| Condition ID | Source | Seeds |
|---|---|---|
| `qwen_seed` | Published GEPA paper | Published seed 0 |
| `qwen_miprov2` | Published GEPA paper | Published seed 0 |
| `qwen_gepa` | Published GEPA paper | Published seed 0 |
| `qwen_mini_admission_reflection` | New | 0 first; 1 and 2 only if capacity remains |
| `qwen_compass_reflection` | New | 0 first; 1 and 2 only if capacity remains |
| `qwen_compass_credit` | New | 0 first; 1 and 2 only if capacity remains |

### GPT-4.1 Mini

| Condition ID | Source | Seeds |
|---|---|---|
| `gpt_seed` | Published GEPA paper | Published seed 0 |
| `gpt_miprov2` | Published GEPA paper | Published seed 0 |
| `gpt_textgrad` | Published GEPA paper | Published seed 0 |
| `gpt_trace_optoprime` | Published GEPA paper | Published seed 0 |
| `gpt_gepa` | Published GEPA paper | Published seed 0 |
| `gpt_compass_reflection` | New | 0 first; 1 and 2 only if capacity remains |

### Transfer

| Condition ID | Optimizer model | Evaluation model | Source |
|---|---|---|---|
| `qwen_compass_credit_to_qwen` | Qwen3-8B | Qwen3-8B | New main runs |
| `gpt_compass_reflection_to_gpt` | GPT-4.1 Mini | GPT-4.1 Mini | New main runs |
| `qwen_compass_credit_to_gpt` | Qwen3-8B | GPT-4.1 Mini | New held-out evaluation only |

## Run record requirements

Every new run must write a machine-readable manifest before model execution.
The existing official GEPA callbacks, `run_log.json`, `run_log.txt`, and
`gepa_state.bin` remain the runtime evidence owners; no parallel audit
framework is introduced. Together these records must expose:

- source commit and dirty-submodule patch hashes;
- benchmark registry identity, official program class, predictor names, metric,
  feedback-map provenance, and exact split fingerprints;
- optimizer seed, model identities, cache identity, concurrency, and budget;
- ordered `B_propose` and `B_admit` IDs and their disjointness check;
- proposal count and `duplicate_count`, format failures, terminal-ranking
  inputs/results, dependency distance and gate outcome;
- parent/admission rewards, accept/reject result, clean `F/E`, lineage-active
  and selection-active sets, top-n boundary ties, and fixed-`X` hits;
- final selected candidate index and its validation/test scores;
- task rollouts, proposer calls/tokens, teacher-forcing tokens, attribution and
  dependency costs, wall time, retry counts, and terminal exception state.

Raw observations excluded by fixed `X` remain observable in the raw execution
record but must not enter clean `F`, `E`, frontier ownership, or reference
ownership.

## Hard reporting rules

- Never fill a paper placeholder from a test-selected oracle. In particular,
  `experiments/13_ifbench_snapshot_oracle_eval.py` is not a paper selection
  path.
- Never label vector-plot extraction as an original run log.
- Never attach three-seed confidence intervals to a published seed-0 baseline.
- Do not infer a failure reason for a proposal that did not enter terminal
  ranking unless a saved event explicitly records it.
- A missing or censored rollout-to-reference result remains missing/censored;
  it is not replaced with an invented threshold or extrapolation.
- Paper language about “fresh” evidence must describe the implemented
  fixed-exclusion protocol and its remaining dependence. It must not claim an
  untouched iid reservoir that the experiment does not use.
