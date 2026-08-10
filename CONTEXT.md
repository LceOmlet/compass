# IFBench Skill Evolution

This context defines the evaluation batches and concurrency boundaries used while evolving IFBench skills.

## Language

**Proposal batch (`B_propose`)**:
The epoch-shuffled training-instance batch whose parent observations supply reflection evidence.
_Avoid_: mini train val, validation batch

**Admission batch (`B_admit`)**:
The training-instance batch on which the final proposed child is compared with its bound references for admission.
_Avoid_: mini train val, validation batch

**Training-time admission pool**:
The optimization-time instance universe from which admission instances are selected. On the DCI path, the exact historical frontier IDs/evidence may already define the full eligible DCI admission set during proposal; only the nominee's new rollout/outcome is post-nomination. Completed admission trajectories may become later proposal evidence. Any dataset placed in this pool is not a held-out reporting set.
_Avoid_: held-out validation set, final test set, globally unseen reservoir

**Held-out reporting set**:
The independent evaluation set that never enters proposal, DCI interaction, admission, parent selection, frontier state, or optimization history and is used only for final reporting or explicitly hidden audits.
_Avoid_: admission pool, `B_admit`, optimization validation

**Rollout bottleneck seed**:
A provenance-bearing event in one logical rollout where an attempted necessary intermediate goal first becomes observably blocked, violated, or diverted without later recovery. Every propose instance contributes its seed rollout to its own proposer evidence with probability one, even when that rollout does not become the highest-reward observation. A current `B_propose` batch can contribute new seeds, but it is never the universe in which a subproblem is defined or counted.
_Avoid_: whole-batch cluster, low-score label, proven causal bottleneck

**Subproblem definition**:
A natural-language capability or constraint region stated by the free-text result of one seed DCI interaction with the full frozen rollout corpus. The orchestrator commits every independent seed DCI as a new immutable identity; historical records may remain visible to the model as optional context, but identity and merge decisions are not model tasks. Its rollout members come exclusively from the paired DCI selection workspace rather than parsing the definition.
_Avoid_: current-batch cluster, batch pairwise clustering, rollout/span count, new admission gate, held-out split

**Subproblem record**:
The immutable creation record for one independent seed DCI discovery: its free-text definition, frozen selection-workspace member identities, unique-instance and success/failure counts, seed identity, and corpus/frontier snapshot identity. New DCI discoveries create new records rather than revising or overwriting old ones.
_Avoid_: definition revision, in-place update, semantic deduplication, mutable summary

**Subproblem origin anchor**:
The frozen unique-instance membership paired with a subproblem's one immutable creation record. It remains available as stable DataIds, metadata, and read-only raw rollout files, never becomes a historical union, and is not concatenated into the initial prompt. It preserves the evidence behind that discovery but is not used to merge later discoveries.
_Avoid_: moving anchor, historical union, concatenated rollout prompt, generated rollout summary

**DCI raw failure count**:
For one newly discovered subproblem and one frozen clean-frontier snapshot, the number of unique instance IDs selected into its unresolved/failure member pool. Tied frontier owners and multiple matching spans remain one instance vote. The experiment uses this count directly, without a failure-rate calculation.
_Avoid_: rollout hit count, span count, failure rate, target-mass product

**DCI selection workspace**:
The isolated per-seed set of rollout identities explicitly selected by DCI as members of the subproblem found in that interaction. Only this set supplies additional rollout sampling and the DCI admission scope; search hits, inspected candidates, and counterexamples are not members merely because DCI accessed them. Free text is not parsed to infer identities, and overlap with another workspace does not by itself authorize merging their subproblem identities.
_Avoid_: all tool hits, parsed final answer, JSON result schema, current execution batch, owner object, automatic semantic merge

**DCI rollout corpus snapshot**:
The frozen set of previously committed training rollouts visible to one seed interaction; the current seed is supplied separately and held-out reporting data is absent. A task-instance view may group records by `DataId` only when it preserves every field and ordering relation from the existing serialized logical-rollout payload and keeps every record accessible.
_Avoid_: current seed rollout, held-out set, one-row-per-instance table, generated summary, ranked or filtered corpus view

**Subproblem instance**:
A unique eligible training `DataId` whose clean-frontier evidence faces the same concrete requirement or observable bottleneck as the seed, either as an unresolved example or a successful contrast that crosses it. Multiple rollout records, frontier owners, text spans, or search hits for that `DataId` still count as one instance.
_Avoid_: rollout record, `rollout_ref`, text span, grep hit, same-topic task

**DCI execution batch**:
A resource batch of `W` seed-instance evolution events executed concurrently from one frozen pre-batch skill/frontier/corpus snapshot; batches themselves execute serially. Same-batch events cannot see one another's children, DCI and admission retain their full scopes, and new subproblem ID/record append plus GEPA state commit follow canonical instance order. No same-batch semantic reconciliation is performed.
_Avoid_: proposal minibatch, dataset split, DCI search scope, admission scope

**DCI proposal minibatch**:
The evidence package for one seed's skill-evolution operation, distinct from the outer seed-instance execution batch. The seed rollout and subproblem definition are always present. When both evidence strata are nonempty, each remaining slot independently chooses the complete-success or unresolved/failure stratum with probability `0.5`, then samples a unique instance uniformly from that stratum. If one stratum has no usable instance, every remaining slot uses the other stratum. If neither has a usable instance, only that seed event skips proposal/admission and increments the explicit no-evidence skip counter; other seed events continue.
_Avoid_: outer execution batch, fixed one-success/one-failure composition, DCI search scope, admission set

**DCI proposal/admission evidence ratio**:
When the subproblem has fewer than three eligible distinct admission instances, sample its eligible members without replacement and fill the remainder uniformly without replacement from the global legal admission pool. The fill pool still excludes the current proposal instance and all recursively inherited proposal instances. Fill instances remain non-members of the subproblem and do not alter subproblem metadata.
The cardinality ratio between one proposal evidence package—one fresh seed plus reused subproblem-history rollouts—and the distinct subproblem instances selected for fresh candidate admission. With three proposal-evidence instances, `1:1` means three admission instances even though the fresh-rollout cost ratio is `1:3`.
_Avoid_: fresh-rollout ratio, train/validation split ratio, whole-subproblem admission

**DCI cold-start epoch**:
One complete pass in which the initial skill is evaluated once on every proposal-eligible training instance before DCI retrieval, skill proposal, or admission is enabled. These are ordinary fresh rollouts: they enter the append-only corpus and count through the official budget/frontier/F/E owners. DCI evolution begins only after the complete epoch boundary.
_Avoid_: first parallel batch, unbudgeted corpus seeding, partially enabled DCI warm-up

**Fresh-rollout budget**:
The optimization budget counts only newly executed instance rollouts, individually at their execution owner. A fresh seed rollout counts even if its event later skips for lack of DCI evidence; historical-corpus reads, DCI retrieval over stored rollouts, cache reuse, proposal events, and admission instances that were never executed do not count. Fresh admission rollouts count individually when executed.
_Avoid_: seed-event count, DCI-query count, planned batch size, cache hit count

**Whole-epoch proposal wave**:
One ordered optimizer iteration whose proposal/admission tasks cover one shuffled training epoch and may execute concurrently from the same pre-wave skill pool.
_Avoid_: concurrent optimizer iterations, intra-batch-only concurrency

**Proposal-lineage exclusion**:
For a skill, the task IDs in its own proposal batch and every transitive ancestor's proposal batch; these IDs are excluded from its clean frontier evidence. Admission batches are not part of this exclusion.
_Avoid_: admission exclusion, direct-parent exclusion

**Raw frontier rate (`F/E`)**:
The share of a skill's clean exposures on which it is an official instance-frontier owner, with every frontier membership counted in full.
_Avoid_: shared frontier rate, unique coverage

**Program candidate (`skill`)**:
The complete mapping of instruction components identified by one GEPA program index. It is the unit eligible for proposal-parent scheduling and admission; component selection inside the candidate remains owned by the official proposer.
_Avoid_: instruction component, isolated prompt field, partial candidate

**Proposal-active skill set (`A_t`)**:
The non-masked program candidates retained by the tie-inclusive lexicographic Top-k boundary before proposal minibatches are allocated. Boundary ties remain members of the same active set.
_Avoid_: all candidate-pool members, masked ancestor set, final-incumbent set

**Joint proposal matching**:
Maximum-weight rectangular matching between the proposal-active skill set and the current uncovered proposal minibatches using their joint scheduling scores. One matching pass uses each skill and minibatch at most once; when minibatches outnumber skills, further passes reuse the full skill set on only the remaining minibatches, without a separate per-skill capacity gate. Each pass first maximizes total scheduling score, then minimizes the sum of the selected skills' previously completed joint-arm observation counts. One completed `(skill, proposal minibatch)` task with an observed `y` adds one count; a task without a complete official admission comparison adds none. Any remaining optimum is the lexicographically smallest assignment sequence after sorting pairs by epoch minibatch position and then program index. Scores are never perturbed, and solver return order is not a tie-break.
_Avoid_: proportional parent sampling, per-skill capacity parameter, executor concurrency limit, epsilon tie-breaking

**Joint LinUCB scheduler**:
The single deterministic proposal scheduler over `(skill, proposal minibatch)` arms. Its context is exactly `x=(1,H,N,U)`, its shared ridge state starts at `V_0=I_4` and `q_0=0`, and its score is exactly `x^T V_t^{-1}q_t + sqrt(x^T V_t^{-1}x)` (`beta_t=1`). One whole proposal wave freezes its active set, observations, `V_t`, and `q_t`, constructs the complete matching before execution, and then aggregates all completed admission observations in canonical assignment order into `V_{t+1}` and `q_{t+1}`. It delegates proposal generation, independent admission, acceptance, stopping, official testing, and final-candidate selection to their existing owners. The scheduler is implemented through the official joint `SamplingStrategy.sample_tasks(...)` seam and has no second policy layer.
_Avoid_: per-skill bandit, tuned beta schedule, probability sampling, extra exploration rule, shadow proposal loop

**Minimal scheduler boundary**:
Only active-set construction, exact context calculation, joint LinUCB scoring, repeated rectangular matching, sufficient-statistic updates, and their checkpoint/trace representation belong to the scheduler. Executor concurrency is not a scheduling constraint, and the scheduler adds no capacity gate, IPS estimator, fallback selector, retry wrapper, budget pre-check, admission variant, or final-selection rule.
_Avoid_: defensive policy additions, speculative gates, duplicated owner behavior

**Signed admission gain (`y`)**:
The unnormalized arithmetic-mean reward difference on the official independent admission batch between a completed child and the history-frozen per-instance bound references already used by official admission. It is recorded with its sign, without positive-part clipping and without division by rollout cost. Every completed admission outcome updates the shared LinUCB sufficient statistics with this value, whether the child is accepted or rejected.
_Avoid_: accepted-only feedback, rejected-as-zero feedback, cost-normalized gain, extra parent evaluation, held-out test reward

**Unobserved scheduler outcome**:
A scheduled arm that does not produce a complete official admission comparison has no observed `y` and therefore makes no LinUCB update. Execution, reflection, or transport failure is never encoded as zero or as a negative method reward; a later completed execution supplies the sole update for that arm.
_Avoid_: failure-as-zero feedback, failure penalty, partial-outcome update

**Lexicographic high-resolution selection (`high_resolution_lexicographic`)**:
Proposal-parent ordering by the exact tuple `(F/E, C/F)`, where `F/E` is the clean frontier hit rate and `C/F` is average shared credit conditional on a frontier hit. The second coordinate refines only exact hit-rate ties. Global top-N is tie inclusive on the complete tuple and the retained set is sampled uniformly, because a tuple has no proportional scalar weight. The historical `high_resolution` label and missing mode fields remain the legacy scalar `C/E` behavior for clean reruns and result interpretation; lexicographic mode requires explicit opt-in and a distinct run identity.
_Avoid_: weighted sum, product score, epsilon scalarization, task-difficulty weighting

**Observed active set (`O_i`)**:
The proposal-active program candidates with a finite committed training observation on instance `i`. A missing observation is unknown evidence, never a zero reward.
_Avoid_: complete active set, zero-filled score set, held-out result set

**Observed repairable gap (`H`)**:
The proposal-batch average positive deficit of one program candidate relative to the best observed active candidate. An unobserved candidate-instance pair contributes no repairable-gap value and remains represented by observation incompleteness.
_Avoid_: proven inability, imputed-zero deficit, admission gain

**Observed residual gap (`N`)**:
The proposal-batch average shortfall of the best observed active candidate from the run's frozen official `perfect_score`. It means that all active candidates remain short of that target only when active-set observations are complete.
_Avoid_: proven universal failure under incomplete observations, held-out error

**Active-set observation incompleteness (`U`)**:
The share of proposal-batch instances for which at least one proposal-active candidate lacks a finite committed training observation. It may be positive together with observed repairable and residual gaps.
_Avoid_: failure score, abstention, incomparable lineage relation

**Common-exposure ancestor mask**:
A reversible proposal-parent eligibility rule that compares a strict descendant with an ancestor only on clean instance IDs exposed to both. Lexicographic high-resolution mode compares `(F/E, C/F)` on that shared domain; the descendant masks the ancestor when its complete tuple is no lower. With no shared clean exposure, the ancestor remains lineage-active, while explicit raw `F/E` and legacy scalar `C/E` modes retain their historical comparisons.
_Avoid_: global-score ancestor mask, age penalty, confidence gate, optimism-corrected score

**Final proposal candidate**:
The single proposed child evaluated on the admission batch. Terminal-analysis modes obtain it after their configured ranking and dependency gate; raw-feedback mode asks the official proposer for exactly one child directly.
_Avoid_: admission candidate set, candidate batch

**Teacher-forced proposal selection**:
A three-candidate proposal mode in which one true tensor batch establishes candidate order before dependency gating. Batch rows retain candidate identity, while low-precision batch-shape effects are allowed to change likelihood values and ranking relative to batch-one execution.
_Avoid_: admission ranking

**Single-candidate proposal selection**:
A teacher-forcing-free proposal mode in which the official proposer produces one candidate and dependency gating determines whether it becomes the final proposal candidate.
_Avoid_: unranked multi-candidate selection

**Raw-feedback single-candidate proposal**:
A no-local-model mode in which the official DSPy proposer consumes its ordinary reflective dataset and produces exactly one candidate. It performs neither FlashTrace credit nor dependency gating; admission still uses the independent admission batch and strict improvement.
_Avoid_: disabled admission, local-model proposal

**Logical rollout**:
One LM trajectory defined by an unchanged prompt, message history, and rollout identity. Re-attempts after TPM rate limiting remain part of the same rollout.
_Avoid_: replacement trajectory, new rollout

**TPM keepalive**:
Indefinite waiting and re-attempting of a logical rollout after TPM rate limiting. It is not a rollout failure and has no cumulative waiting deadline.
_Avoid_: bounded retry, failed rollout

**Dual-account rollout routing**:
Use either of two transport accounts for one logical rollout. Normal calls use the official LiteLLM `least-busy` strategy. A TPM rate limit may make exactly one official same-model-group retry so the other account can serve the unchanged prompt, message history, and rollout identity; other error classes do not receive this retry. If both accounts are TPM-limited, control returns to the logical-rollout caller's TPM keepalive loop.
_Avoid_: replacement trajectory, account-sticky rollout

**True rollout failure**:
A generation request that genuinely times out or a rollout that genuinely exceeds its configured generation bound. TPM rate limiting is not a true rollout failure.
_Avoid_: TPM rate-limit event
