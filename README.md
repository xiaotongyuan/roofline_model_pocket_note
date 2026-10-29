# Roofline Model Pocket Note — Calculator

This repo holds the **companion calculator** (C++17) for the *LLM Inference Roofline — A Pocket Note*. The note itself (the formulas, the recipes, the explanations) is sold separately — link to be added here once published.

The calculator computes the model-level roofline for LLM inference: weights vs KV cache, decode vs prefill, and the TP/PP/EP/DP parallel paths. It is a single C++ file so you can read the whole thing.

For **how to run it and what options exist**, see [`calculator/README.md`](calculator/README.md).

License: MIT — see [`LICENSE`](LICENSE).
