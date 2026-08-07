# COMPASS paper experiment matrix

Date frozen: 2026-07-28

## Superseding primary-suite decision (2026-08-06)

This section supersedes the six-task primary-suite scope below wherever the two
conflict.  The remaining sections are retained as historical planning context
until the revised budgets, seeds, and exact benchmark revisions are frozen.

- The primary suite contains four capability domains: IFBench for instruction
  following, AIME-2025 for mathematical reasoning, ChartQA for multimodal chart
  reasoning, and the official `airline` domain of tau2-bench for stateful tool
  use.  The local two-example `tau2_bench` compatibility fixture is a smoke test
  and is not admissible paper evidence.
- The primary suite uses one model-homogeneous GPT-4.1 Mini panel for both task
  execution and reflection.  Existing Qwen3-8B IFBench and AIME artifacts may
  appear only as secondary open-model evidence; they do not create a second
  required primary matrix.  The exact dated GPT-4.1 Mini alias and decoding
  fingerprint must be frozen before formal launch.
- The primary OpenAI-compatible route is `https://api.gpt.ge/v1`, configured
  through model profile `gpt_4_1_mini_gpt_ge` and credential environment name
  `COMPASS_GPT_GE_API_KEY`; no credential value is stored in the repository.
  A 2026-08-07 probe resolved both the rolling and dated names and verified that
  the dated request returns `gpt-4.1-mini-2025-04-14`.  Exact-parameter text and
  structured-image smoke requests both completed normally; the representative
  low-detail ChartQA image request used 299 prompt and 8 completion tokens.
  These probes establish route capability only and are not benchmark evidence.
- Task execution and proposal/reflection use the same dated model identity,
  `gpt-4.1-mini-2025-04-14`, while benchmark-owned task decoding is preserved.
  ChartQA task rollout, validation, and test calls use its pinned LMMS-Eval
  generation contract (`temperature=0`, `max_new_tokens=16`,
  `do_sample=False`), mapped through the owner program's official DSPy
  `Predict.update_config` seam to
  `temperature=0` and `max_tokens=16`; its single-prediction owner program
  supplies one output.  ChartQA proposal/reflection calls retain the paper
  profile (`temperature=1.0`, output cap `16384`).  Unsupported
  `do_sample` is recorded as owner provenance rather than forwarded to the
  OpenAI-compatible API.  Each benchmark--method run owns
  a distinct DSPy disk and memory cache namespace, activated through DSPy's
  official `configure_cache` interface before its LM is created; cache entries
  are never reused across methods, pilots, or formal runs.  A shared-route TPM
  or 429 wait remains inside the same logical API call and does not create a
  second logical task rollout.  Genuine generation timeout or overlength
  remains the owning method's ordinary failure outcome.
- Each task is reported with five method rows: unoptimized Seed, official DSPy
  MIPROv2, native official GEPA, matched sparse-commit M0, and full COMPASS.
  The main-table M0 and COMPASS rows both use `StrictImprovement` admission.
  `AlwaysAccept` is a mechanism ablation only and must not create additional
  main-table rows.  For tau2-bench Airline, Seed uses the official tau2 runner,
  while GEPA, M0, and COMPASS use the same native `GEPAAdapter` over the
  official tau2 agent/environment loop.  The Airline MIPROv2 cell remains
  blocked because DSPy MIPROv2 has no generic `GEPAAdapter` seam; it must not be
  replaced by a proxy predictor or a locally reimplemented optimizer.
- Published GPT-4.1 Mini IFBench and AIME-2025 Seed/MIPROv2/GEPA endpoints may
  be reused only when model, program, evaluator, split, and decoding protocol
  match exactly.
  They remain provenance-bearing reproduction references rather than
  substitutes for the new matched trajectories: MIPROv2, native GEPA, M0, and
  COMPASS are rerun on all four primary benchmarks so endpoint utility,
  rollout efficiency, resource use, and skill yield come from one protocol.
  Seed is evaluation-only rather than an optimizer run.
- TextGrad and Trace/OptoPrime are restricted to the IFBench proposer analysis.
  They are not silently reimplemented for ChartQA or tau2-bench Airline and do
  not replace the matched M0 allocation control.
- Required new primary-table runs use optimizer seed `0`.  Seeds `1` and `2`
  are optional only after every required primary cell has completed and
  resources remain; a seed-0 result is reported as a point estimate without
  invented uncertainty.  The separately frozen CLUTRR ablation keeps its three
  required seeds.
- HiTab and CLUTRR are not primary-table tasks.  Existing artifacts, if valid,
  may be reported only as secondary mechanism or external-validity evidence.
- Mechanism ablations should be run on one separately frozen lightweight
  benchmark rather than multiplying the four primary-suite experiments.  Its
  task revision, split, cells, seeds, and smaller rollout budget must be stated
  explicitly; an ablation result cannot substitute for a primary-table result.
  The frozen ablation benchmark is the owner-defined CLUTRR `irrelevant`
  variant with one fixed split and optimizer seeds `0`, `1`, and `2`.  Other
  CLUTRR variants and HiTab may remain secondary artifacts but are not required
  cells in the revised ablation table.  Its optimization budget is five
  complete passes over the frozen 150-instance paper-lite training split.  With
  `proposal_minibatch_size=3`, this is exactly 250 serial proposal iterations
  and 750 proposal-instance exposures.  Each ablation cell therefore freezes
  `max_candidate_proposals=250`, `max_metric_calls=2250`, and
  `epoch_parallel_enabled=false`; the metric-call cap is the existing
  nine-calls-per-proposal worst-case accounting, not a requirement to add
  filler evaluations.  Actual optimization task rollouts are reported because
  reference reuse and failed proposals can make them smaller than the common
  cap.
- MIPROv2, native GEPA, M0, and COMPASS are matched within each task by the
  number of optimization task rollouts, not by proposal count or wall time.  A
  complete tau2-bench Airline interaction episode is one task rollout
  regardless of its internal number of model turns or tool calls.  Internal LM
  calls, tool calls, tokens, and wall time remain separately reported resource
  quantities rather than alternative optimization-budget units.
- Optimization budgets are assigned per benchmark rather than globally across
  the suite.  The frozen budgets are IFBench `3593`, AIME-2025 `1839`, and
  ChartQA `1152` task rollouts per optimizer method.  The frozen tau2-bench
  Airline budget is `600` complete training episodes per optimizer method.
  Under the frozen tau2-bench `v1.0.1` task split, Airline has 30
  training tasks and 20 test tasks, so this provides 20 equivalent
  full-training-set passes.  Seed has no
  optimization budget; held-out validation and test calls are reported
  separately and do not consume these optimizer caps.
- ChartQA uses a deterministic balanced optimization view of the pinned owner
  splits: train contains 150 questions (75 human and 75 augmented), validation
  contains 300 questions (150 human and 150 augmented), and final evaluation
  uses all 2,500 official test questions (1,250 human and 1,250 augmented) in
  owner order.  The view is rebuilt from the pinned vis-nlp/ChartQA annotation
  files by a versioned selection rule, and its builder, source hashes, selected
  owner positions, images, manifest, and loaded split fingerprints are frozen
  before any paid run.  It is reported as a balanced low-budget optimization
  view, not as an official `ChartQA-Lite` split.  Test data never participates
  in proposal, admission, validation, candidate selection, or cache reuse.
- The authorized financial envelope is USD `400` for the four-benchmark
  primary matrix, USD `110` for the frozen CLUTRR ablation, and USD `70`
  reserved for explicit final-evaluation failure recovery and unanticipated
  metered overhead.  These are hard ceilings rather than spending targets.
  Pilot spending belongs to the corresponding primary or ablation envelope;
  the reserve cannot be used to start optional seeds or enlarge an optimizer
  budget without a new decision.
- Current planning estimates, derived from existing IFBench/AIME token records,
  the ChartQA multimodal smoke test, and historical official Airline
  trajectories, are USD `154--304` for the primary matrix and USD `12--30`
  for the CLUTRR ablation.  These estimates are not paper results.  Every
  formal run reports actual provider usage, logical task rollouts, lower-level
  LM calls, proposal/reflection calls, tokens, retries, and dollars.
- Before a formal cell starts, a disjoint cost preflight must exercise the
  frozen official program and method-owned path without contributing optimizer
  state or cache entries to the formal run.  Minimum pilot coverage is 96
  logical rollouts per IFBench optimizer, 60 per AIME optimizer, 60 per
  ChartQA optimizer, and one complete 30-episode Airline training pass per
  optimizer; the CLUTRR pilot uses 60 task rollouts in each seed-0 mechanism
  cell.  IFBench must cover its two official stages; ChartQA must retain
  structured image inputs; Airline must meter both agent and user-simulator
  calls.  Each API-owning method phase must execute at least once, with GEPA,
  M0, and COMPASS covering at least eight genuine proposal events and MIPROv2
  completing initialization and at least two official trials.
- The preflight ledger records the resolved dated model alias, endpoint route,
  benchmark/method/phase, logical rollout and proposal identity, input/output/
  cached/image/reasoning tokens when reported, billed and formula cost,
  latency, finish reason, rate-limit waiting, retry identity, and terminal
  result.  A formal cell is released only when its task/config/split/model/
  price fingerprints are frozen, all API-owning phases are represented, and
  the observed mean cost plus a 25-percent planning margin keeps the complete
  primary or ablation projection inside its own envelope.  This is a launch
  decision only: no runtime watchdog, proactive cancellation, dynamic worker
  change, or new optimizer gate is introduced.
- A submitted logical rollout that reaches a genuine generation timeout or
  generation-length failure consumes one rollout from the optimization budget.
  TPM rate-limit keepalive remains part of the same logical rollout and is not
  counted again.  Final held-out reporting may recover explicit execution
  failures to obtain a complete score vector, but those retries never alter the
  optimized state or candidate choice.
- Formal launches place run outputs, caches, checkpoints, logs, and temporary
  benchmark artifacts on the F: drive.  Before launch, the frozen run manifest
  records endpoint/model capabilities, context limits, worker counts, CPU
  thread settings, expected host-memory use, and task/config fingerprints.
  Concurrency is selected from a read-only resource/profile check and then
  frozen before the formal run.  Within a task, MIPROv2, native GEPA, M0, and
  COMPASS use the same frozen concurrency.  A single global worker count is not
  imposed across benchmarks with different model and interaction costs.  No
  runtime watchdog, proactive interruption, resource gate, or new
  training-control path is added; the purpose is to avoid interrupting logical
  rollouts, not to terminate them when utilization rises.

The remaining task versions and split fingerprints must be frozen before
launching new runs.  For tau2-bench, frozen tag `v1.0.1` resolves to
commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`; its bundled Airline split is
30 train, 20 test, and 50 base tasks.  The release notes' “27 Airline task
fixes” describes the number of corrected tasks, not the task-set size.  Tau2
does not define a validation partition inside its 30-task train split.  Before
formal optimization, every method config must therefore freeze an explicit,
disjoint proposal/validation partition whose union is exactly those 30 tasks;
the runner rejects an absent, overlapping, reordered, or incomplete view.

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
| Cumulative ablation table | Published GEPA full-validation final test point and budget | Independent mini-admission; clean birth eligibility; token credit + TF; dependency gate; legacy global-score and current common-exposure ancestor masks; global top-n, each under the frozen cumulative protocol | Required |
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
