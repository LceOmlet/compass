---
status: accepted
---

# Use lexicographic high-resolution proposal-parent selection

The revised primary COMPASS proposal-parent selector uses the exact
lexicographic key

`(clean frontier hit rate, conditional tie resolution) = (F/E, C/F)`.

Here `F` is the unshared clean-frontier count, `E` is clean exposure, and `C`
shares one credit unit equally among all official owners of each clean
frontier. The second coordinate is defined only for frontier-active skills;
skills with `F = 0` remain ineligible. The selector first compares `F/E` and
uses `C/F` only when the hit rates tie.

The previous scalar score satisfies `C/E = (F/E)(C/F)`. Multiplication lets a
lower frontier hit rate outrank a higher one solely because the former has
less crowded frontier ownership. That makes the resolution term a tradeoff
against the primary signal instead of a refinement of it. Lexicographic
ordering preserves the two roles without a scale coefficient, epsilon, or
other hidden scalarization.

The common-exposure ancestor mask computes the same two-coordinate key on the
pair's common clean domain. Global top-N uses the complete key and retains all
exact boundary ties. Because an ordinal tuple does not define cardinal
weights, the active set is sampled uniformly. Rank weights and tuple-to-float
encodings are intentionally excluded.

The existing `high_resolution` mode retains its historical scalar `C/E`
semantics for clean legacy reruns and result interpretation. Missing mode
fields continue to resolve to that legacy behavior. The new mode is named
`high_resolution_lexicographic`; experiments using it must opt in explicitly
and require a clean, distinct run identity. Official frontier ownership, state
updates, admission references, and owner-final incumbent selection remain
unchanged.
