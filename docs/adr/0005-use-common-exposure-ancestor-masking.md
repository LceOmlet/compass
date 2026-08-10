---
status: accepted
amended_by: 0013-use-lexicographic-high-resolution-parent-selection
---

# Use common-exposure ancestor masking

Reversible ancestor masking compares each strict ancestor--descendant pair only
on instance IDs that are clean exposures for both; a no-worse descendant masks
its ancestor, while zero common exposure establishes no masking relation.  The
legacy scalar `high_resolution` mode compares official shared credit per common
exposure and retains its historical proportional sampling.  The revised
`high_resolution_lexicographic` mode compares `(F/E, C/F)` on that same common
domain and samples uniformly after lexicographic top-N.  Neither mode changes
official frontier state, admission, or the explicit legacy raw-`F/E` path.
The common-domain restriction removes broad-exposure ancestors' advantage from
unchallenged instances, at the accepted cost that small common domains retain
sampling variance.
