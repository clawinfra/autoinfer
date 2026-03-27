#!/usr/bin/env python3
"""Establish quality baselines by running evaluations on specified models.

Usage:
    uv run python benchmarks/run_baseline.py \
        --models models/Q3_K_M.gguf,models/IQ2_M.gguf \
        --bench ./target/release/bench \
        --corpus benchmarks/wikitext_sample.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoinfer.evaluator import EvalConfig, evaluate
from autoinfer.profiler import profile_hardware
from autoinfer.results import ResultsTracker, ParetoFrontier


# Best known configs from prior experiments
KNOWN_CONFIGS = {
    "phase10_best": {
        "n_gpu": 17,
        "n_batch": 252,
        "n_ubatch": 94,
        "n_threads": 4,
        "type_k": 8,   # q8_0
        "type_v": 8,   # q8_0
        "flash_attn": 1,
    },
    "phase11_best": {
        "n_gpu": 17,
        "n_batch": 252,
        "n_ubatch": 94,
        "n_threads": 4,
        "type_k": 8,
        "type_v": 8,
        "flash_attn": 1,
    },
    "conservative": {
        "n_gpu": 15,
        "n_batch": 256,
        "n_ubatch": 128,
        "n_threads": 4,
        "type_k": 1,   # f16
        "type_v": 1,   # f16
        "flash_attn": 1,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--models", required=True, help="Comma-separated model paths")
    parser.add_argument("--configs", default="phase10_best", help="Comma-separated config names")
    parser.add_argument("--bench", default="", help="Path to bench binary")
    parser.add_argument("--corpus", default="benchmarks/wikitext_sample.txt", help="Eval corpus")
    parser.add_argument("--perplexity-binary", default="", help="llama-perplexity path")
    parser.add_argument("--ld-library-path", default="", help="LD_LIBRARY_PATH")
    parser.add_argument("--output", default="baseline_results.tsv", help="Output TSV")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    config_names = [c.strip() for c in args.configs.split(",")]

    # Profile hardware
    hw = profile_hardware()
    print(f"Hardware: {hw.summary()}")

    # Setup evaluator
    eval_config = EvalConfig(
        bench_binary=args.bench,
        perplexity_binary=args.perplexity_binary,
        corpus_path=args.corpus,
        ld_library_path=args.ld_library_path,
    )

    # Run baselines
    tracker = ResultsTracker(args.output)
    results_by_model = {}

    for model_path in models:
        model_name = os.path.basename(model_path)
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if not os.path.isfile(model_path):
            print(f"  SKIP: file not found")
            continue

        model_results = []
        for config_name in config_names:
            config = KNOWN_CONFIGS.get(config_name, KNOWN_CONFIGS["phase10_best"])
            print(f"\n  Config: {config_name}")
            print(f"  Params: {json.dumps(config, indent=4)}")

            has_ppl = bool(eval_config.perplexity_binary)
            result = evaluate(
                model_path=model_path,
                config=config,
                eval_config=eval_config,
                measure_perplexity=has_ppl,
            )

            exp_id = tracker.record(result)
            model_results.append(result)

            print(f"  Status: {result.status}")
            if result.status == "ok":
                print(f"  tok/s: {result.tok_s:.3f}")
                print(f"  Perplexity: {result.perplexity:.4f}")
                print(f"  Quality score: {result.quality_score:.4f}")
                print(f"  Quality-adj throughput: {result.quality_adj_throughput:.3f}")
                print(f"  VRAM: {result.vram_mb} MB")
            else:
                print(f"  Notes: {result.notes}")

        results_by_model[model_name] = model_results

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")

    for model_name, results in results_by_model.items():
        ok_results = [r for r in results if r.status == "ok"]
        if ok_results:
            best = max(ok_results, key=lambda r: r.tok_s)
            print(f"\n  {model_name}:")
            print(f"    Best tok/s: {best.tok_s:.3f}")
            print(f"    Perplexity: {best.perplexity}")
            print(f"    Quality-adj: {best.quality_adj_throughput:.3f}")

    if args.output:
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
