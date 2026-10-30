# Roofline Model Pocket Note — Calculator

This repo holds the **companion calculator** (C++17) for Pocket Notes **1 (model)**
and **2 (operator)**. The notes are sold separately:

- Book 1 — *LLM Inference Roofline*: [Amazon](https://www.amazon.com/dp/B0HPSHMFZ7) (ASIN `B0HPSHMFZ7`)
- Book 2 — operator (by shape): not published yet

Note 3 (layer-accumulated model) uses a **separate** program, not this calculator.

The program is a single C++ file. `model` is the Book 1 closed-form roofline
(weights vs KV, prefill vs decode, TP/PP/EP/DP). `operator` is the Book 2
per-shape roofline (one kernel vs the hardware ridge point).

For **how to run it and what options exist**, see [`calculator/README.md`](calculator/README.md).

License: MIT — see [`LICENSE`](LICENSE).
