# AUTORESEARCH_GOAL.md — Self-Direction Rules

## Mission
Maximize inference throughput (tok/s) for large MoE models on consumer hardware
through systematic, autonomous experimentation.

## Principles
1. **Never stop.** Hardware sets the limit, not permission.
2. **Learn from everything.** OOMs are data. Timeouts are data. Crashes are data.
3. **Bayesian > grid search.** Use prior results to inform the next experiment.
4. **Warm start always.** Never throw away historical data.
5. **Report only bests.** Don't spam — signal when something meaningful happens.

## Current Target
- Model: Qwen3.5-35B-A3B (MoE, 3B active parameters)
- Quantization: Q3_K_M (primary), IQ2_XXS, IQ2_M, IQ3_S, IQ4_XS (secondary)
- Hardware: Multi-GPU consumer setup (3090 + 3080 + 2070S)
- Metric: tok/s generation speed
- Current best: ~27.6 tok/s (Phase 12, IQ2_XXS quantization)

## Search Strategy
- Explore GPU layer allocation aggressively (most impactful parameter)
- Balance batch/ubatch for memory bandwidth
- KV cache type interacts with flash attention
- Thread count has diminishing returns past ~11-12
- Context length trades off with batch size for VRAM

## Safety
- OOM is expected and handled gracefully
- Timeout after 300s kills hung processes
- Never modify the model file
- Results are append-only (no data loss)
