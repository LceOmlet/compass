---
status: accepted
---

# Use whole-epoch proposal waves

IFBench uses one ordered optimizer iteration to sample the proposal/admission tasks for a complete shuffled training epoch from one pre-wave skill pool. Independent evaluations and reflections may run concurrently, but the official GEPA engine restores task order before selection and state commits; different optimizer iterations never overlap. This supersedes batch-local-only concurrency because it left independent epoch work serialized without adding any required state dependency.
