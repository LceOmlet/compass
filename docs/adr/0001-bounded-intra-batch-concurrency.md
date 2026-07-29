---
status: superseded by ADR-0002
---

# Bound concurrency to one IFBench evaluation batch

IFBench optimizer iterations remain ordered because parent selection, frontier evidence, and admission depend on the preceding state transition. Within one iteration, each three-instance proposal or admission batch is evaluated concurrently; teacher-forced selection scores three candidates in one GPU tensor batch, while disabling teacher forcing restores the official single-candidate proposal and retains dependency gating. Only the final candidate reaches admission, and a batch-three OOM fails the run instead of silently changing its experimental condition.

Candidate index is bound to tensor row before the forward and preserved when scores are returned. The production decision uses the ranking produced by that batch-three forward. It does not require numerical or ranking equivalence with three batch-one forwards: the pinned MetaX Qwen3 BF16 runtime was observed to use batch-shape-dependent reductions even for identical unpadded rows. This is accepted execution semantics, not an instance-ID reorder, and no post-hoc tolerance or correction rule is introduced.
