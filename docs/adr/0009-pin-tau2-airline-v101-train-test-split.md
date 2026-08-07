# Pin tau2-bench Airline to v1.0.1 and its disjoint train/test split

The paper's tool-use benchmark uses official tau2-bench tag `v1.0.1` at commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`, with all 30 bundled Airline `train` tasks for optimization and the 20 bundled `test` tasks for final evaluation.  This avoids mutable task revisions and train/test leakage, preserves the 600-episode budget as exactly 20 equivalent training-set passes, and excludes both the 50-task `base` evaluation split and the local two-example compatibility fixture.
