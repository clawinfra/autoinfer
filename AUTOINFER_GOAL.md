# AutoInfer Research Methodology

## Problem Statement

Local LLM inference optimization is currently a manual, non-reproducible process:

1. **Hardware-specific**: Optimal configurations for one GPU don't transfer to another
2. **Model-specific**: Different quantizations (Q3_K_M vs IQ2_M vs IQ3_S) behave differently
3. **Wrong metric**: The community optimizes for raw tok/s, ignoring quality degradation
4. **Non-reproducible**: Ad-hoc tuning produces results that can't be compared or replicated

## Core Insight

**The right metric is quality-adjusted throughput**, not raw speed.

```
quality_adj_throughput = tok_s × quality_score
```

Where `quality_score` is normalized perplexity relative to a baseline quantization:

```
quality_score = baseline_perplexity / measured_perplexity
```

This means:
- A model at **21.6 tok/s** with quality_score **0.55** → quality_adj = **11.88**
- A model at **12.3 tok/s** with quality_score **1.00** → quality_adj = **12.30**

The "slower" model actually delivers **more useful output per second**.

## Approach: Bayesian Optimization over Quality-Adjusted Throughput

### Why Bayesian (not grid search)?

The parameter space has 7+ dimensions with complex interactions:
- GPU layers × batch size → VRAM pressure
- KV cache type × flash attention → compatibility constraints
- Thread count × batch size → CPU saturation

Grid search over this space requires O(n^7) evaluations. Bayesian optimization (TPE) typically finds near-optimal configs in 30-50 trials by learning which regions of the space are promising.

### Methodology

1. **Hardware Profiling**
   - Auto-detect GPU (nvidia-smi), RAM (/proc/meminfo), CPU cores (lscpu)
   - Measure storage bandwidth (dd benchmark)
   - Determine VRAM-based constraints on GPU layer offloading

2. **Parameter Space Definition**
   - Hardware-aware bounds (max GPU layers from VRAM)
   - Known-good seeds from prior experiments
   - Constraint enforcement (n_ubatch ≤ n_batch)

3. **Bayesian Search (Optuna TPE)**
   - Warm start from legacy experiment data
   - Objective: maximize quality_adj_throughput
   - Soft penalty for configs below quality threshold
   - Graceful handling of OOM/crash (return -inf)

4. **Pareto Analysis**
   - Track full Pareto frontier in (quality, speed) space
   - Allow users to query: "fastest config at quality ≥ 0.95"
   - Persistent TSV logging for reproducibility

## Findings So Far

### Qwen3.5-35B-A3B on RTX 3090 + 3080 + 2070 SUPER (42GB total VRAM, 16GB RAM)

From 700+ experiments across phases 4-11:

| Quantization | Best tok/s | Approx. Quality | Quality-Adj |
|-------------|-----------|-----------------|-------------|
| Q3_K_M (14.7GB) | 12.331 | 1.000 (baseline) | 12.331 |
| IQ2_M (10.5GB) | ~21.6* | TBD (needs ppl measurement) | TBD |
| IQ3_S (12.3GB) | TBD | TBD | TBD |

*IQ2_M speed estimated from prior experiments; 75% faster due to smaller model size and more layers fitting in VRAM.

### Key Configuration Discoveries

**Best Q3_K_M config** (phase 10, verified):
```
n_gpu=17, n_batch=252, n_ubatch=94, type_k=q8_0, type_v=q8_0, flash_attn=1
→ 12.331 tok/s, 7751MB VRAM
```

**Critical findings:**
- Flash attention is essential (10-15% improvement)
- q8_0 KV cache ≥ f16 KV cache performance (surprising — quantized KV is faster AND saves VRAM)
- n_gpu=17 is the sweet spot for this hardware (16 = OOM on context, 18 = model load failure)
- Batch size 252 with ubatch 94 outperforms standard power-of-2 sizes

## Perplexity Measurement Strategy

Three approaches, in order of preference:

1. **llama-perplexity** binary (most accurate, needs llama.cpp build)
2. **llama-cpp-python** (Python bindings, approximate but portable)
3. **Fixed-corpus generation analysis** (compare output quality on standardized prompts)

The baseline corpus is a 512-token passage covering multiple domains (history, science, literature) to provide a representative sample for perplexity evaluation.

## Future Work

- **Multi-model support**: Automatically compare quantizations of the same base model
- **Apple Silicon**: Adapt profiler for Metal/ANE detection
- **Jetson/Edge**: Support NVIDIA Jetson Nano, Xavier for edge deployment
- **Context length scaling**: How does optimal config change with n_ctx?
- **Continuous optimization**: Background daemon that re-optimizes when hardware changes
- **Community benchmarks**: Standardized hardware profiles for reproducible comparisons

## Data Sources

All findings are derived from experiments in the `qwen35-moe-offload` research repo:
- `results_phase4.tsv` through `results_phase11.tsv` (724 rows)
- Rust bench binary for precise tok/s measurement
- Three model quantizations: Q3_K_M, IQ2_M, IQ3_S

## Citation

If you use AutoInfer in your research:

```bibtex
@software{autoinfer2026,
  title={AutoInfer: Universal Hardware-Adaptive LLM Inference Optimizer},
  author={ClawInfra},
  year={2026},
  url={https://github.com/clawinfra/autoinfer}
}
```
