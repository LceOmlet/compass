---
status: accepted
---

# Recursively exclude proposal-lineage evidence

New COMPASS states exclude a skill's own `B_propose` IDs and every transitive ancestor's `B_propose` IDs from that skill's clean frontier evidence; `B_admit` remains eligible. The new state schema rejects custom checkpoints created with the legacy direct-parent rule so a run cannot silently change algorithms after resume. The existing v26 experiment instead finishes from its original checkpoint under its frozen legacy code and configuration.
