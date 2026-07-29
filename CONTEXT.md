# IFBench Skill Evolution

This context defines the evaluation batches and concurrency boundaries used while evolving IFBench skills.

## Language

**Proposal batch (`B_propose`)**:
The epoch-shuffled training-instance batch whose parent observations supply reflection evidence.
_Avoid_: mini train val, validation batch

**Admission batch (`B_admit`)**:
The training-instance batch on which the final proposed child is compared with its bound references for admission.
_Avoid_: mini train val, validation batch

**Whole-epoch proposal wave**:
One ordered optimizer iteration whose proposal/admission tasks cover one shuffled training epoch and may execute concurrently from the same pre-wave skill pool.
_Avoid_: concurrent optimizer iterations, intra-batch-only concurrency

**Proposal-lineage exclusion**:
For a skill, the task IDs in its own proposal batch and every transitive ancestor's proposal batch; these IDs are excluded from its clean frontier evidence. Admission batches are not part of this exclusion.
_Avoid_: admission exclusion, direct-parent exclusion

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
