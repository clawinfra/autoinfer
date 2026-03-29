# AUTORESEARCH_GOAL.md — Self-Direction Rules

## Mission
**Maximise usable intelligence per second on consumer hardware.**

That means two axes — not one:
1. **tok/s** — raw throughput (how fast)
2. **n_ctx** — context window (how much the model can see)

A 30 tok/s model with 512 ctx is worse than a 25 tok/s model with 4096 ctx for real tasks.
**The goal is to push BOTH simultaneously. Never stop. Never ask permission.**

### Qwen3.5 legacy target (COMPLETED)
- Best: **29.899 tok/s @ ctx=512** (IQ2_XXS, n_gpu=27, batch=32/16, q8_0 KV, flash=1)
- Ceiling reached at Phase 12. ctx=512 was never the bottleneck — we just didn't push it.

### New primary goal: Qwen3.5 @ higher context
**Push ctx beyond 512 while preserving tok/s ≥ 25.**

KV cache budget (8GB VRAM, 27 GPU layers = ~7.5GB used, ~700MB left):
| ctx  | q8_0 KV | q4_0 KV | fits in 700MB? |
|------|---------|---------|----------------|
| 512  | 67MB    | 34MB    | ✅ trivial      |
| 1024 | 134MB   | 67MB    | ✅ easy         |
| 2048 | 268MB   | 134MB   | ✅ comfortable  |
| 4096 | 537MB   | 268MB   | ✅ tight/fine   |
| 8192 | 1074MB  | 537MB   | ⚠️ needs fewer GPU layers |

**Target sweet spot: ctx=4096, q4_0 KV (268MB) — 8× the old window, fits easily.**

With Flash Attention (Apple paper), long context is O(N) memory not O(N²) — so 4096 ctx
should cost less than expected. This is the key insight to exploit.

### Secondary: Nemotron-Cascade-2-30B-A3B (CPU-only baseline)
- Model: 17GB IQ2_XXS, always CPU-only on RTX 3070
- CPU baseline: ~8.3 tok/s
- Same ctx expansion goal: push ctx=2048+ while maintaining max CPU throughput

---

## Core Principles
1. **Never stop.** Hardware sets the limit, not permission.
2. **Optimise quality-adjusted throughput** = tok/s × log2(ctx/512). Bigger window = more valuable.
3. **Bayesian > grid search.** Use prior results to inform the next experiment.
4. **Warm start always.** Never throw away historical data.
5. **Detect hardware reality immediately.** Probe first, assume nothing.
6. **Adapt strategy based on what the data says.** Not what the plan says.
7. **Paper guidance is a prior, not a rule.** Verify with experiments.

---

## Self-Direction Protocol

### On startup
1. Probe hardware: `nvidia-smi` → available VRAM
2. Check model size vs VRAM → select strategy
3. Load all existing results → seed Bayesian optimizer with quality-adjusted score
4. Enqueue priority seeds from paper guidance first
5. Then let Bayesian TPE drive exploration

### Quality-adjusted score (use this as the objective, not raw tok/s)
```python
score = tok_s * math.log2(max(n_ctx, 512) / 512)
# Examples:
# 29.9 tok/s @ ctx=512  → score = 29.9 × 1.0 = 29.9
# 27.0 tok/s @ ctx=1024 → score = 27.0 × 2.0 = 54.0  ← BETTER
# 24.0 tok/s @ ctx=4096 → score = 24.0 × 3.0 = 72.0  ← MUCH BETTER
# 20.0 tok/s @ ctx=8192 → score = 20.0 × 4.0 = 80.0  ← FRONTIER
```

### CUDA mode (Qwen3.5, partial GPU offload)
Priority order:
1. **ctx expansion first**: fix n_gpu=27, threads=8, batch=32/16, vary ctx=1024→4096 with q4_0 KV
2. **Find OOM cliff for ctx**: binary search between ctx=4096 and ctx=8192 with q4_0 KV
3. **If OOM at ctx=4096 q8_0**: switch to q4_0 (TurboQuant asymmetric: q8_0 K + q4_0 V)
4. **After ctx ceiling found**: fine-tune batch/threads/KV around that ctx
5. **Then push n_gpu higher** if VRAM permits at the new ctx size

### CPU-only mode (Nemotron, always CPU)
- timeout=600s minimum (17GB model needs ~30s just to load)
- Priority: threads=8, batch=32/16 (Qwen transfer), q8_0 KV, fa=1
- Then ctx expansion: 512→1024→2048 (KV is CPU RAM, not VRAM — much more headroom)
- Nemotron has 16GB RAM available → ctx=32768 is theoretically possible CPU-side

### When all experiments timeout/OOM
- Last 5 timeout → reduce n_gen to 64, reduce n_ctx by 50%
- Last 5 OOM → reduce n_gpu by 4 OR switch kv_type to q4_0
- CPU-only confirmed → switch strategy, never try ngl>0 again

---

## 📄 Paper Guidance — How to Use It

### Apple: "LLM in a Flash" — Expert Offloading + Flash Attention
**arXiv**: https://arxiv.org/abs/2312.11514

**Key insight for ctx expansion**:
- Flash Attention = O(N) memory complexity (vs naive O(N²))
- With `fa=1` (forced), going from ctx=512→4096 costs 8× tokens but only ~2× memory
- **Always use fa=1.** Every experiment. No exceptions.
- Chunk prefill: smaller `batch` during prefill = lower peak VRAM spike at long ctx
  → At ctx=4096, try batch=16-32 to avoid prefill OOM

**Sliding window for MoE**: for Nemotron SSM+MoE hybrid:
- SSM state is fixed-size (doesn't grow with ctx) — ctx expansion is cheap
- Attention layers (interleaved) do grow — but with fa=1, manageable
- → Nemotron can likely handle ctx=4096+ in CPU RAM without issues

**What to try**:
```
ctx=1024-4096, fa=1, batch=16-32 (smaller batch at longer ctx to control prefill VRAM)
```

---

### Google: "Efficiently Scaling Transformer Inference" — TurboQuant KV
**arXiv**: https://arxiv.org/abs/2211.05100

**Key insight for ctx expansion**:
- KV cache grows linearly with ctx — this IS the bottleneck
- q4_0 KV = 4× bandwidth gain vs f16, negligible quality loss on Nemotron (SSM buffers most context)
- **Asymmetric KV (TurboQuant)**: K-cache needs higher precision than V-cache
  - K is used for dot-product attention (affects routing) → keep at q8_0
  - V is used for weighted sum (more forgiving) → compress to q4_0
  - `-ctk q8_0 -ctv q4_0`: best quality/bandwidth tradeoff
- At ctx=4096 with asymmetric KV: only 268+134=402MB vs 537MB symmetric q8_0

**Priority experiments**:
```
ctx=4096, ctk=q8_0, ctv=q4_0, fa=1   ← TurboQuant asymmetric (FIRST TARGET)
ctx=4096, ctk=q4_0, ctv=q4_0, fa=1   ← symmetric aggressive
ctx=2048, ctk=q8_0, ctv=q4_0, fa=1   ← safe 2× jump first
ctx=8192, ctk=q4_0, ctv=q4_0, fa=1   ← frontier (536MB KV)
```

---

### QJL / PolarQuant — Non-linear KV quantisation
**Key idea**: iq4_nl uses non-uniform bucket boundaries, better for outlier-heavy KV vectors
- Safer than q4_0 at long ctx where outlier KV values compound
- Use as fallback if q4_0 quality degrades at ctx=4096+

```
ctx=4096, ctk=iq4_nl, ctv=iq4_nl, fa=1  ← non-linear fallback
```

---

## Priority Seed Queue (run in this order)

### Qwen3.5 seeds (GPU-resident, ctx expansion focus)
```python
# Seed 1: Qwen best config + 2× ctx — minimal risk, TurboQuant asymmetric
{"n_gpu": 27, "n_ctx": 1024, "batch": 32, "ubatch": 16, "n_threads": 8,
 "n_gen": 128, "kv_type": "q8_0", "kv_type_v": "q4_0", "flash_attn": True}

# Seed 2: 4× ctx with TurboQuant asymmetric (the main target)
{"n_gpu": 27, "n_ctx": 2048, "batch": 32, "ubatch": 16, "n_threads": 8,
 "n_gen": 128, "kv_type": "q8_0", "kv_type_v": "q4_0", "flash_attn": True}

# Seed 3: 8× ctx — TurboQuant asymmetric, smaller batch for prefill control
{"n_gpu": 27, "n_ctx": 4096, "batch": 16, "ubatch": 16, "n_threads": 8,
 "n_gen": 128, "kv_type": "q8_0", "kv_type_v": "q4_0", "flash_attn": True}

# Seed 4: 8× ctx — symmetric q4_0 (4× BW gain, SSM buffers quality loss)
{"n_gpu": 27, "n_ctx": 4096, "batch": 16, "ubatch": 16, "n_threads": 8,
 "n_gen": 128, "kv_type": "q4_0", "kv_type_v": "q4_0", "flash_attn": True}

# Seed 5: frontier — ctx=8192 with q4_0 (536MB KV, may need n_gpu drop)
{"n_gpu": 24, "n_ctx": 8192, "batch": 16, "ubatch": 8, "n_threads": 8,
 "n_gen": 64, "kv_type": "q4_0", "kv_type_v": "q4_0", "flash_attn": True}
```

### Nemotron seeds (CPU-only, ctx + throughput)
```python
# Seed 1: Qwen best-config transfer (n_gpu=0)
{"n_gpu": 0, "n_ctx": 512, "batch": 32, "ubatch": 16, "n_threads": 8,
 "n_gen": 64, "kv_type": "q8_0", "flash_attn": True}

# Seed 2: CPU ctx expansion — 2× ctx (CPU RAM, no VRAM cost)
{"n_gpu": 0, "n_ctx": 2048, "batch": 32, "ubatch": 16, "n_threads": 8,
 "n_gen": 64, "kv_type": "q8_0", "kv_type_v": "q4_0", "flash_attn": True}

# Seed 3: TurboQuant asymmetric on CPU
{"n_gpu": 0, "n_ctx": 1024, "batch": 32, "ubatch": 16, "n_threads": 8,
 "n_gen": 64, "kv_type": "q8_0", "kv_type_v": "q4_0", "flash_attn": True}
```

---

## Search Space (Bayesian optimizer bounds)
```python
SEARCH_SPACE = {
    "n_gpu":     {"type": "int",        "low": 0,   "high": 32},
    "n_ctx":     {"type": "categorical","choices": [512, 1024, 2048, 4096, 8192]},
    "batch":     {"type": "int",        "low": 8,   "high": 256},
    "ubatch":    {"type": "int",        "low": 8,   "high": 128},
    "n_threads": {"type": "int",        "low": 4,   "high": 12},
    "n_gen":     {"type": "categorical","choices": [64, 128, 256]},
    "kv_type":   {"type": "categorical","choices": ["q4_0", "iq4_nl", "q8_0"]},
    "kv_type_v": {"type": "categorical","choices": ["q4_0", "q8_0", "same"]},
    "flash_attn":{"type": "categorical","choices": [True]},   # ALWAYS on
}
```

---

## Timeout Rules
- Qwen3.5 (GPU): timeout=300s
- Nemotron (CPU-only): timeout=600s
- OOM detected → result=0, logged, never retried with same config
- 3 consecutive timeouts → halve n_gen and n_ctx

## Safety
- Never modify the model file
- Results append-only
- Always flash_attn=True (Apple paper)
