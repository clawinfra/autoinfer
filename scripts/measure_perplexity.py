#!/usr/bin/env python3
"""Standalone perplexity measurement via llama.cpp or llama-cpp-python.

Usage:
    uv run python scripts/measure_perplexity.py \
        --model models/Q3_K_M.gguf \
        --corpus benchmarks/wikitext_sample.txt

Falls back to llama-cpp-python if llama-perplexity binary is not available.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys


def measure_via_binary(
    model_path: str,
    corpus_path: str,
    binary_path: str = "llama-perplexity",
    n_ctx: int = 512,
    n_gpu_layers: int = 0,
    ld_library_path: str = "",
) -> float | None:
    """Measure perplexity using llama-perplexity binary."""
    cmd = [
        binary_path,
        "--model", model_path,
        "--file", corpus_path,
        "--ctx-size", str(n_ctx),
        "--n-gpu-layers", str(n_gpu_layers),
    ]

    env = os.environ.copy()
    if ld_library_path:
        env["LD_LIBRARY_PATH"] = ld_library_path

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        output = r.stdout + "\n" + r.stderr

        m = re.search(r"(?:perplexity|ppl)\s*[=:]\s*([\d.]+)", output, re.IGNORECASE)
        if m:
            return float(m.group(1))

        m = re.search(r"([\d.]+)\s*\+/-\s*[\d.]+", output)
        if m:
            return float(m.group(1))

    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"Binary measurement failed: {e}", file=sys.stderr)

    return None


def measure_via_python(
    model_path: str,
    corpus_path: str,
    n_ctx: int = 512,
    n_gpu_layers: int = 0,
) -> float | None:
    """Approximate perplexity using llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not available", file=sys.stderr)
        return None

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        with open(corpus_path) as f:
            text = f.read()

        # Tokenize
        tokens = llm.tokenize(text.encode("utf-8"))
        if len(tokens) < 2:
            return None

        # Truncate to context size
        tokens = tokens[:n_ctx]

        # Compute perplexity via log-likelihood
        total_log_prob = 0.0
        n_tokens = 0

        # Process in chunks
        chunk_size = min(512, n_ctx)
        for start in range(0, len(tokens) - 1, chunk_size):
            chunk = tokens[start : start + chunk_size + 1]
            if len(chunk) < 2:
                break

            llm.reset()
            llm.eval(chunk[:-1])

            # Get logits for last position
            # This is approximate — real perplexity needs all positions
            scores = llm.scores
            if scores is not None and len(scores) > 0:
                import numpy as np
                logits = np.array(scores[-1])
                log_probs = logits - np.log(np.sum(np.exp(logits)))
                target = chunk[-1]
                if target < len(log_probs):
                    total_log_prob += log_probs[target]
                    n_tokens += 1

        if n_tokens == 0:
            return None

        avg_neg_log_prob = -total_log_prob / n_tokens
        return math.exp(avg_neg_log_prob)

    except Exception as e:
        print(f"Python measurement failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Measure model perplexity")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--corpus", required=True, help="Eval corpus path")
    parser.add_argument("--binary", default="", help="llama-perplexity path")
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--ld-library-path", default="")
    args = parser.parse_args()

    # Try binary first, then Python fallback
    ppl = None

    if args.binary:
        print(f"Trying binary: {args.binary}")
        ppl = measure_via_binary(
            args.model, args.corpus, args.binary,
            args.n_ctx, args.n_gpu_layers, args.ld_library_path,
        )

    if ppl is None:
        print("Trying Python (llama-cpp-python)...")
        ppl = measure_via_python(
            args.model, args.corpus, args.n_ctx, args.n_gpu_layers,
        )

    if ppl is not None:
        print(f"\nPerplexity: {ppl:.4f}")
        return 0
    else:
        print("\nFailed to measure perplexity with any method.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
