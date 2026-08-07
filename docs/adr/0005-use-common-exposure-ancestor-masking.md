---
status: accepted
---

# Use common-exposure ancestor masking

High-resolution reversible ancestor masking compares each strict ancestor-descendant pair only on instance IDs that are clean exposures for both, using official shared frontier credit; a no-worse descendant masks its ancestor, while zero common exposure establishes no masking relation. This removes the seed and other broad-exposure ancestors' structural advantage from unchallenged instances without changing global parent scores, top-N, proportional sampling, frontier state, admission, or the explicit legacy raw-`F/E` comparison mode, at the accepted cost that small common domains retain sampling variance.
