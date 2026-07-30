---
status: accepted
---

# Default to high-resolution proposal-parent selection

Proposal-parent selection defaults to high-resolution scoring. Each clean
instance contributes one selection-credit unit, shared equally among every
official clean frontier owner for that instance. A skill's accumulated credit
is divided by its unchanged official clean exposure count.

This score is only the input to the existing reversible ancestor mask,
tie-inclusive top-5, and proportional parent sampling. Official candidate
identity, frontier ownership, clean exposure, raw `F/E`, admission, reference
ownership, and final-program selection remain unchanged. Raw `F/E` selection
remains available only as an explicit comparison mode.

Changing this selector changes the experiment, so high-resolution runs start
from clean state rather than resuming checkpoints produced by raw `F/E`
selection.
