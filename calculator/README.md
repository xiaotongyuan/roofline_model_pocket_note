# Calculator — LLM Inference Roofline (model level)

A single-file C++17 program for Pocket Notes **1 (model)** and **2 (operator)**. Note 3 (layer-accumulated) uses a different program.

This README tells you **how to run it, what options exist, and how to customize a run**. It does **not** explain the formulas — those live in the pocket note.

> **Book 1 (ebook):** [LLM Inference Roofline on Amazon](https://www.amazon.com/dp/B0HPSHMFZ7) (ASIN `B0HPSHMFZ7`). Book 2 is not published yet.

---

## Build

Requires CMake 3.20+ and a C++17 compiler.

```powershell
# from the repo root
cmake -S calculator -B calculator/build
cmake --build calculator/build --config Release
```

On Linux/macOS:

```bash
cmake -S calculator -B calculator/build
cmake --build calculator/build
```

The binary is `calculator/build/tool1_roofline` (Linux/macOS) or `calculator\build\tool1_roofline.exe` (Windows). With the Visual Studio generator on Windows it lands in `calculator\build\Release\` instead — adjust the path if so.

---

## Commands

```text
tool1_roofline <command>

  model                run one model case (uses the defaults in main(), see below)
  operator             run one operator case (uses the defaults in main())
  export-model [file]  export the model test matrix to CSV
  export-operator [dir] export the operator test matrix to CSV (split into parts)
  run-model-tests      run the built-in checks against the pocket-note numbers
```

### Run the tests (do this first)

```powershell
.\calculator\build\tool1_roofline.exe run-model-tests
```

Expected (22/22). Recipe 1 is `A1`–`A2c`; comm is `E2`/`E3`; then prefill, PP, DP, MLA, and dies:

```text
A1 expected=484 actual=483.767 delta%=0.0480459 => PASS
A2 expected=0.393 actual=0.393435 delta%=0.110803 => PASS
A2b expected=10178 actual=10166.9 delta%=0.109532 => PASS
A2c expected=15.4 actual=15.3754 delta%=0.1599 => PASS
E2 expected=2.16 actual=2.15541 delta%=0.212675 => PASS
E2b expected=1 actual=1 delta%=0 => PASS
E2c expected=453 actual=452.802 delta%=0.0437407 => PASS
E3 expected=2.16 actual=2.15541 delta%=0.212675 => PASS
E3b expected=8278 actual=8278.7 delta%=0.00849449 => PASS
E3c expected=1 actual=1 delta%=0 => PASS
A1b expected=169.3 actual=169.319 delta%=0.0109924 => PASS
A1c expected=5.91 actual=5.90603 delta%=0.0672534 => PASS
P1 expected=984 actual=984.024 delta%=0.00244147 => PASS
P2 expected=1 actual=1 delta%=0 => PASS
PP1 expected=0.25 actual=0.25 delta%=0 => PASS
PP2 expected=8.268 actual=8.26844 delta%=0.00526676 => PASS
PP3 expected=484 actual=483.767 delta%=0.0480459 => PASS
PP4 expected=169.3 actual=169.319 delta%=0.0109924 => PASS
D1 expected=5.91 actual=5.90603 delta%=0.0672534 => PASS
D2 expected=677.3 actual=677.274 delta%=0.00377369 => PASS
M1 expected=0.0002186 actual=0.000218624 delta%=0.010979 => PASS
DIES1 expected=70312.5 actual=70312.5 delta%=0 => PASS
Model tests pass 22/22
```

### One model case

```powershell
.\calculator\build\tool1_roofline.exe model
```

Output (with the built-in defaults — Llama-8B on H20, decode, B=1, C=4096):

```text
Model: Llama-8B on H20
BytesPerToken_GB=8.26844, TPS_comp_final=9250, TPS_mem=483.767, TPS_roof=483.767
Bottleneck=mem, TPS_Effective=169.319, TPS_Cluster=169.319, Latency_ms=5.90603
```

Those intermediate values are Recipe 1 Case A in the pocket note.

### Export CSV

```powershell
.\calculator\build\tool1_roofline.exe export-model model_matrix.csv
.\calculator\build\tool1_roofline.exe export-operator .
```

---

## Built-in models

Defined in `BuildModelDb()` in `src/main.cpp`. Fields: total params (B), active params (B), layers, hidden, Q heads, KV heads, MLA, MoE, experts (total / active), default bytes-per-param.

| Model | Total (B) | Active (B) | Layers | Hidden | Q heads | KV heads | MLA | MoE | Experts (total/active) | Default BPE |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|
| `Llama-8B` | 8 | 8 | 32 | 4096 | 32 | 8 | no | no | 0 / 0 | 1 (FP8) |
| `Llama-70B` | 70 | 70 | 80 | 8192 | 64 | 8 | no | no | 0 / 0 | 1 |
| `Llama-405B` | 405 | 405 | 128 | 16384 | 128 | 8 | no | no | 0 / 0 | 1 |
| `Qwen2.5-72B` | 72 | 72 | 80 | 8192 | 64 | 8 | no | no | 0 / 0 | 1 |
| `Mixtral-8x7B` | 47 | 13 | 32 | 4096 | 32 | 8 | no | yes | 8 / 2 | 1 |
| `DeepSeek-V3` | 671 | 37 | 61 | 7168 | 128 | 16 | yes (factor 0.25) | yes | 256 / 8 | 1 |

For MoE models, **active params** is the routed-expert parameter count (what the compute roofline uses); **total params** is for VRAM / capacity. `DeepSeek-V3` uses MLA latent KV (the `MLA_factor = 0.25` term), which is a different KV formula from GQA — see the pocket note.

---

## Built-in hardware (AI cards)

Defined in `BuildHardwareDb()` in `src/main.cpp`. Key fields shown below (full list in the `Hardware` struct):

| Card | FP8 mat (TFLOPS) | BF16 mat (TFLOPS) | HBM BW (GB/s) | VRAM (GB) | NVLink (GB/s) | dies | η_comp | η_mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `H20` | 148 | 74 | 4000 | 96 | 900 | 1 | 0.45 | 0.35 |
| `B200` | 2250 | 1125 | 8000 | 192 | 1800 | 2 | 0.45 | 0.35 |
| `GB300` | 2500 | 1250 | 8500 | 192 | 1800 | 2 | 0.45 | 0.35 |
| `Custom` | 307 | 153 | 4000 | 144 | 800 | 2 | 0.45 | 0.70 |

Other fields per card: `cards_per_node`, `fp8_vector`, `bf16_vector`, `fp8_sparse`, `bf16_sparse`, `sfu`, `sfu_eff`, `d2d_bw`, `rdma_bw`, `pcie_bw`. The `dies` field divides per-card `TF` and `BW` to get per-die values (H20 is a single-die package, so `dies = 1` and it has no effect). On a dual-die card (B200, GB300), set `TP >= dies` if you want to feed the whole package.

**The `η_comp` / `η_mem` values are placeholders, not measurements** — replace them with your own for your hardware (see the pocket note §1).

---

## Built-in operators (the `operator` command)

Defined in `BuildOperatorDb()`. The operator profiler is a separate, finer-grained view; this note/calculator's main path is the `model` command.

| Category | Operators |
|---|---|
| Compute-Intensive (Matrix) | `GEMM_QK`, `GEMM_AV`, `GEMM_FFN_Up`, `GEMM_FFN_Down`, `GEMM_Gate`, `GEMM_Output` |
| Memory-Bound (SFU) | `SiLU`, `GELU`, `RMSNorm`, `LayerNorm`, `Softmax` |
| Memory-Bound (Vector) | `ReLU`, `RoPE`, `KV_Cache_Read`, `KV_Cache_Write` |
| Sparse | `MoE_Router`, `MoE_Sparse_GEMM`, `Sparse_Attention` |

---

## How to customize a run

**There is no config file and no CLI flag for inputs yet** — this is intentional, so the program stays one readable file. To run a custom case, edit the defaults and rebuild. There are two places to edit:

- The **defaults** live in the `ModelInput` / `OperatorInput` struct definitions near the top of `src/main.cpp`. Change those to change what `model` / `operator` run by default.
- Or **add explicit assignments** in the `if (cmd == "model")` / `if (cmd == "operator")` block in `main()`, which override the defaults for that run.

### Model input fields (`ModelInput`)

| Field | Default | Meaning |
|---|---|---|
| `hardware` | `"H20"` | any name from `BuildHardwareDb()` |
| `model` | `"Llama-8B"` | any name from `BuildModelDb()` |
| `stage` | `"Decode"` | `"Decode"` or `"Prefill"` |
| `bytes_per_param` | `1.0` | FP8 = 1, BF16 = 2 (overridden by the model's `default_bpp` in the `model` block) |
| `batch` | `1` | batch **per replica** (not cluster-wide) |
| `context` | `4096` | decode: KV depth `C`; prefill: prompt length `S` (same field, two meanings) |
| `tp` | `1` | tensor parallel |
| `pp` | `1` | pipeline parallel |
| `dp` | `1` | data parallel (scales cluster throughput, not one user's latency) |
| `ep` | `1` | expert parallel (MoE only) |
| `nodes` | `1` | number of nodes (for cluster card count) |
| `micro_batch` | `1` | prefill PP: micro-batch **size** (count = ceil(`B × S / size`)) |
| `pp_placement` | `"Intra-node"` | `"Intra-node"` or `"Inter-node"` |

Example — to run Llama-70B on H20, decode, TP=8, B=256, edit the `model` block:

```cpp
if (cmd == "model") {
    ModelInput in;
    in.model = "Llama-70B";
    in.tp = 8;
    in.batch = 256;
    in.context = 4096;
    in.bytes_per_param = model_db.at(in.model).default_bpp;
    const ModelOutput o = CalculateModel(in, hw_db, model_db);
    ...
}
```

Rebuild and run `model`. The `context` field is `C` for decode and `S` for prefill (the program stores both in one field — see the pocket note §2d).

### Operator input fields (`OperatorInput`)

| Field | Default | Meaning |
|---|---|---|
| `hardware` | `"H20"` | from `BuildHardwareDb()` |
| `precision` | `"FP8"` | `"FP8"` or `"BF16"` |
| `operator_name` | `"GEMM_QK"` | from `BuildOperatorDb()` |
| `sparsity_pct` | `0` | 2:4 sparse ratio (0 = dense) |
| `b` | `1` | batch |
| `s` | `4096` | sequence |
| `h` | `4096` | hidden |
| `n` | `32` | Q heads |
| `d` | `128` | head dim |
| `i` | `11008` | FFN intermediate |
| `v` | `128256` | vocab |
| `l` | `32` | layers |
| `e` | `256` | MoE experts total |
| `k` | `8` | top-K active experts |
| `w` | `1024` | SWA window (0 = full attention) |
| `tp` | `1` | tensor parallel |
| `dp` | `1` | data parallel |

Edit the `operator` block the same way.

### Add your own model

Edit `BuildModelDb()` and add one line, following the `Model` struct order: `name, total_params_b, active_params_b, layers, hidden, q_heads, kv_heads, use_mla, mla_kv_factor, moe, experts_total, experts_active, default_bpp`.

```cpp
{"MyModel", {"MyModel", 13, 13, 40, 5120, 40, 8, false, 1, false, 0, 0, 1}},
```

Then set `in.model = "MyModel";` in the `model` block.

### Add your own card

Edit `BuildHardwareDb()` and add one line, following the `Hardware` struct order: `name, cards_per_node, fp8_matrix, bf16_matrix, fp8_vector, bf16_vector, fp8_sparse, bf16_sparse, sfu, sfu_eff, hbm_bw, mem_eff, d2d_bw, nvlink_bw, rdma_bw, pcie_bw, vram, comp_eff, dies_per_card`.

```cpp
{"MyCard", {"MyCard", 8, 1000, 500, 200, 100, 2000, 1000, 50, 0.85, 6000, 0.35, 0, 1200, 600, 128, 160, 0.45, 2}},
```

Then set `in.hardware = "MyCard";`.

### Change the efficiencies

The `η_comp` / `η_mem` are stored per card in `BuildHardwareDb()` (the `comp_eff` and `mem_eff` fields). Edit them there for an existing card, or set them when you add a new card. They are placeholders — replace with your own measured or assumed values.

---

## What this calculator does and does not do

**Does:**
- Model-level roofline (weights + KV cache) for decode and prefill.
- TP / PP / EP / DP parallelism, with communication time folded into the compute path. PP shards **capacity** and cuts decode compute (`pp_eff = 1/PP`); it does not reduce decode BytesPerToken.
- Bottleneck label (memory / compute / comm-suspect) and per-replica + cluster throughput.
- An operator profiler (`operator` command) for finer-grained per-operator views.

**Does not:**
- Operand-level analysis (cacheline waste, burst length, broadcast). That is a later note.
- Kernel fusion, FlashAttention, compiler schedules.
- Match `nvidia-smi` on a given night. The efficiencies are placeholders, not measurements.

---

## License

MIT — see [`../LICENSE`](../LICENSE).
