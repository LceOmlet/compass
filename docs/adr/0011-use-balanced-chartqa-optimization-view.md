# Use a balanced, reproducible ChartQA optimization view

The ChartQA main-table runs use 75 human and 75 augmented owner examples for training, 150 human and 150 augmented owner examples for validation, and the complete 2,500-example official test split.  We replace the earlier human-only first-N snapshot because it mismatched the test composition and lacked a repository-owned reconstruction path; the new versioned builder freezes source identities, selected owner positions, images, manifest, and loaded split fingerprints without changing the owner prompt or metric.
