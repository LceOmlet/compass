---
status: accepted
---

# Use joint LinUCB proposal matching

The revised primary COMPASS proposal scheduler first obtains the non-masked,
tie-inclusive lexicographic Top-k program candidates and the current epoch's
uncovered proposal minibatches.  It scores every `(program, minibatch)` arm
with one shared LinUCB model using exactly `x=(1,H,N,U)`, `V_0=I_4`, `q_0=0`,
and `beta=1`, then allocates minibatches by deterministic repeated rectangular
Hungarian matching.  A pass maximizes total UCB score, minimizes the sum of
the selected programs' completed joint-arm observation counts on an exact
tie, and finally chooses the lexicographically smallest assignment by epoch
minibatch position and program index.  There is no proportional sampling,
per-program capacity, epsilon perturbation, or second exploration rule.
The completed-count tie-break is frozen at wave start across repeated passes;
scheduled but not-yet-completed tasks do not change it.

One wave freezes the active set, observations, and LinUCB state before any
execution.  A completed official independent-admission comparison contributes
the signed, unnormalized mean reward difference between the child and the
history-frozen bound references; negative and rejected outcomes remain valid
feedback, while an incomplete comparison produces no update.  The whole
wave's sufficient-statistic update is committed in canonical assignment order
to the official GEPA checkpoint.

The implementation extends GEPA's existing `SamplingStrategy` lifecycle with
default-inert task metadata, completed-admission feedback, and a dedicated
sampling-strategy state bag.  Proposal generation, admission, acceptance,
budget stopping, official testing, and final-candidate selection remain owned
by their existing implementations.  Existing configurations retain their
historical scheduler unless they explicitly opt into the new method under a
new run identity.
