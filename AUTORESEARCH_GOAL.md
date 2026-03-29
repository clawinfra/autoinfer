# AUTORESEARCH_GOAL.md — Self-Direction Rules

## Mission
Maximize inference throughput (tok/s) for large MoE models on consumer hardware
through systematic, autonomous experimentation. **Never stop. Never ask permission.**

## Core Principles
1. **Never stop.** Hardware sets the limit, not permission.
2. **Learn from everything.** OOMs = data. Timeouts = data. Crashes = data.
3. **Bayesian > grid search.** Use prior results to inform the next experiment.
4. **Warm start always.** Never throw away historical data.
5. **Detect hardware reality immediately.** Don't assume GPU works — probe first.
6. **Adapt strategy based on what the data says.** Not what the plan says.

## Self-Direction Protocol

### On startup
1. Probe hardware: `nvidia-smi` → available VRAM
2. Check model size vs VRAM: if model > VRAM → CPU-only mode
3. Load all existing results → seed Bayesian optimizer
4. Select strategy: CUDA_OPTIMISE or CPU_ONLY_OPTIMISE based on data

### CPU-only mode (when model exceeds VRAM)
- Don't timeout waiting for GPU that isn't there
- Use timeout=600s minimum (17GB model needs ~30s just to load)
- Optimise: n_threads, batch, ubatch, kv_type (q4_0 saves memory bandwidth)
- Use coordinate descent across axes, random perturbation every 3 iterations
- **Never use n_gen > 128 in early exploration** — keeps per-experiment time manageable

### CUDA mode (partial GPU offload)
- Sweep n_gpu_layers to find VRAM cliff
- Apple Flash Attention (fa=1) required for hybrid SSM+MoE architectures
- After cliff found: fine-tune threads, batch, KV type around the sweet spot

### When all experiments timeout/OOM
- **Do NOT keep repeating the same config** — that's the bug that killed the old loop
- Detect: if last 5 experiments all timeout → reduce n_gen to 64, reduce n_ctx to 256
- Detect: if last 5 experiments all OOM → reduce n_gpu by 2
- Detect: if CPU-only confirmed → switch to cpu_only strategy permanently

---

## Current Targets

### Primary: Nemotron-Cascade-2-30B-A3B (RTX 3070, 8GB VRAM)
- Model: `/tmp/nemotron-models/Nemotron-Cascade-2-30B-A3B-IQ2_XXS.gguf` (17GB)
- Architecture: NemotronH — SSM (Mamba2) + Attention hybrid MoE
- Hardware reality: 17GB > 8GB VRAM → **always CPU-only on RTX 3070**
- CPU baseline confirmed: ~8.3 tok/s at ngl=24 (from ngl sweep)
- Target: maximise CPU throughput via thread/batch/kv tuning

### Previous: Qwen3.5-35B-A3B (RTX 3070, 8GB VRAM)
- All-time best: **29.899 tok/s** (IQ2_XXS, n_gpu=27, batch=32/16, q8_0 KV, flash=1)
- COMPLETED — ceiling reached at Phase 12

---

## 📄 Paper Guidance

### Apple: "Flash Memory" — Expert Offloading for MoE
**Paper**: LLM in a Flash: Efficient Large Language Model Inference with Limited Memory (Apple, 2023)
**arXiv**: https://arxiv.org/abs/2312.11514

Key findings to exploit:
- **Sliding window expert caching**: keep the last N active experts in VRAM, evict least-recently-used
  - For Nemotron: 3B active params per token → only route experts need VRAM at inference time
  - Smaller n_gpu + expert-aware layer ordering beats naive layer offload
- **Chunk prefill**: process prompt in chunks to avoid VRAM spike from large KV caches
  - Maps to: smaller `n_ctx` + smaller `batch` = lower peak VRAM during prefill
- **Row/column bundling**: load weight rows in bundles matching cache line size
  - Maps to: `ubatch` aligned to powers of 2 (16, 32, 64) for cache efficiency
- **Prediction-based prefetch**: prefetch next expert while computing current
  - In llama.cpp: `--flash-attn 1` enables overlap; `n_threads` controls prefetch parallelism
- **Practical implication**: for 8GB VRAM + 17GB model, aim for n_gpu=4-8 with fa=1
  - More GPU layers = less CPU↔GPU transfer = better throughput, UP TO the OOM cliff
  - Past the cliff: chunked offload is better than crashing

**What to try**:
```
n_gpu=4-12, fa=1, n_ctx=256-512, batch=32-64, ubatch=16-32
```

---

### Google: TurboQuant — KV Cache Compression
**Paper**: Efficiently Scaling Transformer Inference (Pope et al., Google, 2022)
**arXiv**: https://arxiv.org/abs/2211.05100

Key findings to exploit:
- **KV cache is the bottleneck at small batch sizes** — not weights
  - For n_batch=1 (typical inference): 90%+ of memory bandwidth goes to KV reads
  - Quantising KV from f16 → q8_0 = 2x bandwidth improvement with negligible quality loss
  - Quantising KV from f16 → q4_0 = 4x bandwidth improvement, small quality loss
- **Attention vs SSM**: SSM layers (Nemotron's Mamba2 part) have FIXED-size state
  - SSM state size = d_model × d_state (constant, doesn't grow with sequence length)
  - KV quantisation benefits the attention layers only; SSM layers unaffected
  - → q4_0 is safer for Nemotron than for pure-attention models
- **Interleaved quantisation**: quantise K cache more aggressively than V cache
  - K is queried more frequently than V → K quality matters more
  - In llama.cpp: `-ctk q8_0 -ctv q4_0` (K=high quality, V=aggressive)
  - This is TurboQuant's "asymmetric KV compression" strategy

**What to try**:
```
-ctk q8_0 -ctv q4_0    (asymmetric: high-quality K, compressed V)
-ctk q4_0 -ctv q4_0    (symmetric aggressive: 4x bandwidth gain)
-ctk q8_0 -ctv q8_0    (symmetric safe: 2x bandwidth gain)
```

**Expected ranking for Nemotron**: q4_0/q4_0 ≈ q8_0/q4_0 > q8_0/q8_0 >> f16/f16

---

### Johnson-Lindenstrauss (QJL) — Dimensionality Reduction
**Paper**: Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression (2024)
**Key idea**: project KV vectors into lower-dimensional space via random JL projection before quantisation
- Preserves pairwise distances within ε with high probability
- Reduces KV cache size by sqrt(d_model) factor theoretically
- In practice: llama.cpp doesn't have native JL projection yet
- **What to try**: approximate JL via `iq4_nl` KV type (non-linear quantisation = similar effect)
  - `iq4_nl` uses non-uniform bucket boundaries = better preservation of extreme values
  - Better than `q4_0` for attention layers where outlier KV values matter

---

## Search Space Guidance (Nemotron-specific)

### Confirmed by ngl sweep (170 experiments):
| Config | Result |
|--------|--------|
| ngl=0-32, IQ2_XXS | All timeout or CPU-only |
| Peak CPU | ~8.3 tok/s at ngl=24 |

### Priority experiments (run these first):
1. `ngl=0, t=8, b=128, ub=64, kv=q4_0, fa=1` — baseline with flash attn
2. `ngl=0, t=6, b=64, ub=32, kv=q8_0/q4_0 asymmetric, fa=1` — TurboQuant asymmetric
3. `ngl=4, t=6, b=32, ub=16, kv=q4_0, fa=1` — minimal GPU offload (Apple Flash)
4. `ngl=0, t=4, b=32, ub=16, kv=iq4_nl, fa=1` — QJL-inspired non-linear quant
5. `ngl=0, t=12, b=256, ub=128, kv=q4_0, fa=0` — max CPU threads baseline

### Thread guidance (CPU-only):
- **Best range**: 6-10 threads (XPS has 6 cores / 12 threads)
- **Avoid**: 12+ threads (GIL-style contention, context switches hurt)
- **Avoid**: 1-3 threads (underutilise CPU)

---

## Timeout Rules
- CPU-only 17GB model: **timeout=600s minimum** (30s load + 120s gen + overhead)
- GPU-assisted: timeout=120s (faster load, faster gen)
- If experiment times out: reduce n_gen by 50%, retry
- If 3 consecutive timeouts at same config: mark as "OOM/hang", skip

## Safety
- OOM is expected and handled gracefully
- Never modify the model file
- Results are append-only (no data loss)
- **Timeout=600s default for large CPU models**
