---
status: superseded
superseded_by: 0013-use-lexicographic-high-resolution-parent-selection
---

# Default to high-resolution proposal-parent selection

Proposal-parent selection defaults to high-resolution scoring. Each clean
instance contributes one selection-credit unit, shared equally among every
official clean frontier owner for that instance. A skill's accumulated credit
is divided by its unchanged official clean exposure count.

This global score is the input to tie-inclusive top-5 and proportional parent
sampling. In high-resolution mode, the reversible ancestor mask applies the
same credit convention on each ancestor-descendant pair's common clean
exposure domain, as recorded in ADR 0005. Official candidate identity,
frontier ownership, clean exposure, raw `F/E`, admission, reference ownership,
and final-program selection remain unchanged. Raw `F/E` selection remains
available only as an explicit comparison mode with its legacy global-score
mask.

Changing this selector changes the experiment, so high-resolution runs start
from clean state rather than resuming checkpoints produced by raw `F/E`
selection.
