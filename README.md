# AutoInfer

> Hardware-adaptive MoE inference optimizer — find the fastest llama.cpp config for *your* GPU.

**770 experiments. 29.9 tok/s. RTX 3070 8GB. $500 GPU.**

AutoInfer automatically finds the optimal [llama.cpp](https://github.com/ggerganov/llama.cpp) configuration for your hardware — maximizing quality-adjusted throughput, not just raw speed.

## Install

```bash
# From GitHub
pip install git+https://github.com/clawinfra/autoinfer.git

# Or clone and install locally
git clone https://github.com/clawinfra/autoinfer.git
cd autoinfer && pip install .

# With uv (recommended)
uv pip install git+https://github.com/clawinfra/autoinfer.git
```

## Quick Start

```bash
# 1. Auto-detect your hardware (GPU, RAM, CPU)
autoinfer profile

# 2. Run optimization on a model
autoinfer optimize \
  --model models/Qwen3.5-35B-A3B-IQ2_XXS.gguf \
  --bench llama-bench \
  --trials 50

# 3. Analyze existing experiment data
autoinfer analyze results.tsv

# 4. Run autonomous research loop (advanced)
autoinfer loop --model models/Qwen3.5-35B-A3B-IQ2_XXS.gguf --max-experiments 100
```

## Hardware Requirements

| Hardware | Minimum | Recommended |
|----------|---------|-------------|
| GPU | RTX 3060 6GB / M1 | RTX 3070+ 8GB / M1 Pro |
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8+ cores |
| Storage | SSD | NVMe SSD |

## Pre-Optimized Hardware Configs

AutoInfer ships with pre-optimized configs in `configs/` for common hardware:

| Hardware | Quant | GPU Layers | Est. tok/s | Config |
|----------|-------|-----------|-----------|--------|
| **RTX 3070 8GB** | IQ2_XXS | 27 | 27-30 | [`rtx3070_8gb.json`](configs/rtx3070_8gb.json) |
| **RTX 3080 10GB** | IQ2_M | 32 | 35-40 | [`rtx3080_10gb.json`](configs/rtx3080_10gb.json) |
| **RTX 3090 24GB** | Q3_K_M | 64 | 45-55 | [`rtx3090_24gb.json`](configs/rtx3090_24gb.json) |
| **RTX 4070 Ti 12GB** | IQ3_S | 40 | 40-50 | [`rtx4070ti_12gb.json`](configs/rtx4070ti_12gb.json) |
| **Apple M1 Pro 16GB** | IQ2_M | 32 | 15-20 | [`m1_pro_16gb.json`](configs/m1_pro_16gb.json) |
| **Apple M2 Max 32GB** | IQ3_S | 64 | 25-35 | [`m2_max_32gb.json`](configs/m2_max_32gb.json) |
| **CPU Only 32GB** | IQ2_XXS | 0 | 3-5 | [`cpu_only_32gb.json`](configs/cpu_only_32gb.json) |

All configs target **Qwen3.5-35B-A3B** (35B params, 3B active — MoE architecture).

## Apple Silicon

AutoInfer supports Apple Silicon via llama.cpp's Metal backend:

- **Unified memory** — all GPU layers use the same RAM pool, no PCIe bottleneck
- **Metal acceleration** — llama.cpp compiles with Metal support by default on macOS
- **Memory bandwidth** — M1 Pro (200GB/s), M2 Max (400GB/s) — the real throughput limiter
- Use `autoinfer profile` to auto-detect your Apple Silicon config

```bash
# On macOS with Apple Silicon
autoinfer profile
# darwin | Apple M1 Pro (16.0GB unified) | RAM 16.0GB | 10 cores
```

## The Key Insight

### Quality-Adjusted Throughput

```
effective_throughput = tok/s × quality_score
```

A model at 30 tok/s with severe quality degradation may be *worse* than 20 tok/s with good quality. AutoInfer finds the **Pareto frontier**: maximum throughput at acceptable quality.

### Quantization Scaling Law

From our 770-experiment research on Qwen3.5-35B-A3B:

> **Each GB freed from model weights produces 4.25× the proportional throughput increase** (α=4.25 power exponent)

This means aggressive quantization (IQ2_XXS) on VRAM-limited GPUs delivers super-linear speed gains — not just from fitting more layers, but from improved cache utilization and reduced memory bandwidth pressure.

| Quant | Model Size | Quality Score | Throughput (RTX 3070) |
|-------|-----------|---------------|----------------------|
| Q3_K_M | 15.2 GB | 0.97 | ~8 tok/s (CPU offload) |
| IQ3_S | 12.9 GB | 0.93 | ~15 tok/s |
| IQ2_M | 10.1 GB | 0.85 | ~22 tok/s |
| IQ2_XXS | 8.1 GB | 0.78 | **~30 tok/s** |

## How It Works

```
Hardware Profiling → Parameter Space → Bayesian Optimization → Pareto Analysis
        ↓                    ↓                    ↓                    ↓
   GPU/RAM/CPU         n_gpu_layers          Optuna TPE          Quality × Speed
   auto-detect         batch_size            50+ trials          frontier curves
                       KV cache type
                       flash_attn
                       thread count
```

1. **Profile** your hardware (GPU VRAM, RAM, CPU cores, storage speed)
2. **Define** the parameter space with hardware-aware constraints
3. **Search** via Bayesian optimization (TPE sampler) for optimal configs
4. **Evaluate** each config: measure tok/s and quality
5. **Rank** by quality-adjusted throughput on the Pareto frontier

### Parameter Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `n_gpu_layers` | 0 – max | GPU-offloaded layers (VRAM-constrained) |
| `batch_size` | 16 – 128 | Batch size for prompt processing |
| `ubatch_size` | 8 – 64 | Micro-batch size |
| `n_threads` | 2 – 8 | CPU threads (GIL-friendly) |
| `kv_cache_type` | q4_0, iq4_nl, q8_0 | KV cache quantization |
| `flash_attn` | on/off | Flash attention toggle |

## Architecture

```
autoinfer/
  cli.py          — Command-line interface
  profiler.py     — Hardware detection (GPU, RAM, CPU, storage)
  params.py       — Parameter space + hardware-aware constraints
  evaluator.py    — Speed benchmarks + perplexity measurement
  executor.py     — Bench binary execution + output parsing
  optimizer.py    — Bayesian optimization via Optuna TPE
  loop.py         — Autonomous research loop
  reporter.py     — Progress reporting
  results.py      — Pareto frontier + TSV tracking
```

## Research Data

The `scripts/` directory contains our full research dataset:
- **`autoresearch_results.json`** — 770 experiments with full parameters and results
- **`AUTORESEARCH_REPORT.md`** — Detailed analysis and findings

## Development

```bash
git clone https://github.com/clawinfra/autoinfer.git
cd autoinfer
pip install -e ".[dev]"
pytest
```

## License

MIT — [ClawInfra](https://github.com/clawinfra)
