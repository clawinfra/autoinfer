# AutoInfer

> Universal hardware-adaptive LLM inference optimizer — quality-adjusted throughput maximization

## The Problem

Manual LLM inference tuning is:
- **Non-reproducible** — results vary by hardware, model quantization, and driver version
- **Hardware-specific** — optimal config for RTX 3090 ≠ RTX 4090 ≠ Apple M4
- **Speed-obsessed** — tok/s alone is the wrong metric when quality degrades

## The Key Insight

**Quality-adjusted throughput** = tok/s × quality_score

A model running at 21.6 tok/s with severe perplexity degradation (IQ2_M) may deliver *worse effective output* than the same model at 12.3 tok/s with good quality (Q3_K_M). AutoInfer finds the **Pareto frontier**: maximum throughput at acceptable quality.

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
4. **Evaluate** each config: measure tok/s AND perplexity
5. **Rank** by quality-adjusted throughput on the Pareto frontier

## Installation

```bash
# From source
git clone https://github.com/clawinfra/autoinfer.git
cd autoinfer
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

## Quick Start

### Profile your hardware
```bash
autoinfer profile
# linux | RTX 3090 (24.0GB) | RAM 16.0GB | 6 cores | Storage 0 MB/s

autoinfer profile --json --storage
```

### Optimize a model
```bash
autoinfer optimize \
  --model models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  --bench ./target/release/bench \
  --corpus benchmarks/wikitext_sample.txt \
  --trials 50 \
  --target-quality 0.95 \
  --output results.tsv
```

### Analyze existing experiments
```bash
autoinfer analyze results_phase9.tsv results_phase10.tsv results_phase11.tsv
```

## Architecture

```
autoinfer/
  profiler.py     — Hardware detection (GPU, RAM, CPU, storage)
  params.py       — Parameter space + hardware-aware constraints
  evaluator.py    — Speed benchmarks + perplexity measurement
  optimizer.py    — Bayesian optimization via Optuna TPE
  results.py      — Pareto frontier + TSV tracking
  cli.py          — Command-line interface
```

### Evaluator Backends

AutoInfer supports multiple evaluation backends (auto-detected):

| Backend | Speed | Perplexity | Notes |
|---------|-------|------------|-------|
| Custom Rust bench | ✅ | ❌ | Fastest, from qwen35-moe-offload |
| llama-perplexity | ❌ | ✅ | llama.cpp built-in |
| llama-cli | ✅ | ✅ | General purpose |
| llama-cpp-python | ✅ | ✅ | Pure Python fallback |

### Parameter Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `n_gpu` | 0 – max_layers | GPU-offloaded layers (VRAM-constrained) |
| `n_batch` | 64 – 512 | Batch size for prompt processing |
| `n_ubatch` | 32 – 256 | Micro-batch size |
| `n_threads` | 1 – cpu_cores | CPU threads |
| `type_k` | f16, q4_0, q8_0 | KV cache key quantization |
| `type_v` | f16, q4_0, q8_0 | KV cache value quantization |
| `flash_attn` | 0, 1 | Flash attention toggle |

## Research Methodology

See [AUTOINFER_GOAL.md](AUTOINFER_GOAL.md) for the full research methodology and findings.

## Prior Art

Built on data from 700+ experiments on Qwen3.5-35B-A3B across Q3_K_M, IQ2_M, and IQ3_S quantizations. AutoInfer generalizes these findings into a reusable framework.

## License

MIT
