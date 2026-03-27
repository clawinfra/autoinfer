"""Evaluator — run inference benchmarks and measure quality.

Supports multiple backends:
  1. Custom Rust bench binary (preferred for tok/s)
  2. llama-perplexity (for perplexity measurement)
  3. llama-cli / llama-bench (fallback)
  4. llama-cpp-python (pure Python fallback)
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoinfer.params import KV_TYPE_MAP


@dataclass
class EvalResult:
    """Result of a single evaluation run."""

    tok_s: float = 0.0  # tokens per second (generation)
    perplexity: float = float("inf")  # lower = better
    quality_score: float = 0.0  # normalized: 1.0 = baseline quality
    quality_adj_throughput: float = 0.0  # tok_s × quality_score
    vram_mb: int = 0
    config: dict = field(default_factory=dict)
    status: str = "pending"  # ok, oom, crash, timeout, error
    wall_time_s: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def failed(cls, config: dict, status: str, notes: str = "") -> "EvalResult":
        """Create a failed result."""
        return cls(config=config, status=status, notes=notes)


@dataclass
class EvalConfig:
    """Configuration for the evaluator."""

    bench_binary: str = ""  # path to bench binary
    perplexity_binary: str = ""  # path to llama-perplexity
    llama_cli: str = ""  # path to llama-cli
    ld_library_path: str = ""  # for CUDA libs
    corpus_path: str = ""  # path to evaluation corpus
    n_ctx: int = 512  # context length
    n_predict: int = 256  # tokens to generate for speed test
    baseline_perplexity: float = 0.0  # Q3_K_M baseline (0 = not set)
    timeout_s: int = 120  # per-eval timeout
    workdir: str = ""  # working directory for bench


def _find_binary(names: list[str], search_paths: list[str]) -> str:
    """Find a binary by name in common locations."""
    for name in names:
        # Check PATH
        found = shutil.which(name)
        if found:
            return found
        # Check search paths
        for sp in search_paths:
            candidate = os.path.join(sp, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return ""


def auto_detect_config(workdir: str = "") -> EvalConfig:
    """Auto-detect available binaries and create config."""
    search_paths = [
        "/tmp/llama-cpp-build",
        "/usr/local/bin",
        os.path.expanduser("~/.local/bin"),
    ]
    if workdir:
        search_paths.insert(0, os.path.join(workdir, "target", "release"))

    return EvalConfig(
        bench_binary=_find_binary(["bench"], search_paths),
        perplexity_binary=_find_binary(
            ["llama-perplexity", "perplexity"], search_paths
        ),
        llama_cli=_find_binary(["llama-cli", "main"], search_paths),
        ld_library_path=os.environ.get("LD_LIBRARY_PATH", ""),
        workdir=workdir or os.getcwd(),
    )


def _run_bench(
    model_path: str,
    config: dict,
    eval_config: EvalConfig,
) -> tuple[float, int, str]:
    """Run bench binary, return (tok_s, vram_mb, notes).

    The bench binary is expected to output lines like:
        tok/s: 12.331
        VRAM: 7751 MB
    or JSON output.
    """
    if not eval_config.bench_binary:
        return 0.0, 0, "no bench binary"

    cmd = [eval_config.bench_binary]
    cmd.extend(["--model", model_path])
    cmd.extend(["--n-gpu-layers", str(config.get("n_gpu", 0))])
    cmd.extend(["--batch-size", str(config.get("n_batch", 256))])
    cmd.extend(["--ubatch-size", str(config.get("n_ubatch", 128))])
    cmd.extend(["--threads", str(config.get("n_threads", 4))])
    cmd.extend(["--ctx-size", str(eval_config.n_ctx)])
    cmd.extend(["--n-predict", str(eval_config.n_predict)])

    if config.get("flash_attn", 0):
        cmd.append("--flash-attn")

    kv_k = KV_TYPE_MAP.get(config.get("type_k", 8), "q8_0")
    kv_v = KV_TYPE_MAP.get(config.get("type_v", 8), "q8_0")
    cmd.extend(["--cache-type-k", kv_k])
    cmd.extend(["--cache-type-v", kv_v])

    env = os.environ.copy()
    if eval_config.ld_library_path:
        env["LD_LIBRARY_PATH"] = eval_config.ld_library_path

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=eval_config.timeout_s,
            cwd=eval_config.workdir,
            env=env,
        )

        output = r.stdout + "\n" + r.stderr

        # Parse tok/s — look for common patterns
        tok_s = 0.0
        vram_mb = 0

        # Pattern: "eval time = ... ms / ... tokens (X.XX ms per token, Y.YY tokens per second)"
        m = re.search(r"([\d.]+)\s*tokens?\s*per\s*second", output, re.IGNORECASE)
        if m:
            tok_s = float(m.group(1))

        # Pattern: "internal=12.331tok/s"
        m = re.search(r"internal=([\d.]+)tok/s", output)
        if m:
            tok_s = float(m.group(1))

        # Pattern: "tok/s: 12.331"
        m = re.search(r"tok/s[:\s]+([\d.]+)", output, re.IGNORECASE)
        if m and tok_s == 0.0:
            tok_s = float(m.group(1))

        # VRAM from nvidia-smi or bench output
        m = re.search(r"VRAM[:\s]+([\d]+)\s*MB", output, re.IGNORECASE)
        if m:
            vram_mb = int(m.group(1))

        # Check for OOM
        if "out of memory" in output.lower() or "oom" in output.lower():
            return 0.0, 0, "oom"

        if r.returncode != 0 and tok_s == 0.0:
            return 0.0, 0, f"crash (rc={r.returncode})"

        notes = f"tok/s={tok_s:.3f}, vram={vram_mb}MB"
        return tok_s, vram_mb, notes

    except subprocess.TimeoutExpired:
        return 0.0, 0, "timeout"
    except OSError as e:
        return 0.0, 0, f"error: {e}"


def _run_perplexity(
    model_path: str,
    config: dict,
    eval_config: EvalConfig,
) -> tuple[float, str]:
    """Measure perplexity using llama-perplexity binary.

    Returns (perplexity, notes). perplexity=inf on failure.
    """
    if not eval_config.perplexity_binary:
        return float("inf"), "no perplexity binary"

    if not eval_config.corpus_path or not os.path.isfile(eval_config.corpus_path):
        return float("inf"), "no corpus file"

    cmd = [eval_config.perplexity_binary]
    cmd.extend(["--model", model_path])
    cmd.extend(["--file", eval_config.corpus_path])
    cmd.extend(["--n-gpu-layers", str(config.get("n_gpu", 0))])
    cmd.extend(["--threads", str(config.get("n_threads", 4))])
    cmd.extend(["--ctx-size", str(eval_config.n_ctx)])

    kv_k = KV_TYPE_MAP.get(config.get("type_k", 8), "q8_0")
    cmd.extend(["--cache-type-k", kv_k])

    env = os.environ.copy()
    if eval_config.ld_library_path:
        env["LD_LIBRARY_PATH"] = eval_config.ld_library_path

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=eval_config.timeout_s * 3,  # perplexity takes longer
            env=env,
        )

        output = r.stdout + "\n" + r.stderr

        # Pattern: "Final estimate: PPL = 6.1234"
        m = re.search(
            r"(?:perplexity|ppl)\s*[=:]\s*([\d.]+)", output, re.IGNORECASE
        )
        if m:
            ppl = float(m.group(1))
            return ppl, f"ppl={ppl:.4f}"

        # Pattern: "6.1234 +/- 0.1234"
        m = re.search(r"([\d.]+)\s*\+/-\s*[\d.]+", output)
        if m:
            ppl = float(m.group(1))
            return ppl, f"ppl={ppl:.4f}"

        return float("inf"), "could not parse perplexity"

    except subprocess.TimeoutExpired:
        return float("inf"), "perplexity timeout"
    except OSError as e:
        return float("inf"), f"perplexity error: {e}"


def compute_quality_score(
    perplexity: float,
    baseline_perplexity: float,
) -> float:
    """Compute quality score normalized to baseline.

    A quality score of 1.0 means identical perplexity to baseline.
    Higher perplexity (worse quality) → score < 1.0.
    Lower perplexity (better quality) → score > 1.0.

    Uses inverse ratio with log smoothing to handle large differences:
        score = baseline_ppl / measured_ppl  (clamped to [0, 2])

    If baseline is not set, returns 1.0 (neutral).
    """
    if baseline_perplexity <= 0 or math.isinf(baseline_perplexity):
        return 1.0
    if perplexity <= 0 or math.isinf(perplexity):
        return 0.0

    # Simple ratio: baseline/measured
    # If measured == baseline → 1.0
    # If measured > baseline (worse) → < 1.0
    # If measured < baseline (better) → > 1.0
    score = baseline_perplexity / perplexity

    # Clamp to reasonable range
    return max(0.0, min(score, 2.0))


def evaluate(
    model_path: str,
    config: dict,
    eval_config: Optional[EvalConfig] = None,
    measure_perplexity: bool = True,
) -> EvalResult:
    """Run a complete evaluation: benchmark speed + measure quality.

    Args:
        model_path: Path to GGUF model file.
        config: Inference parameters (n_gpu, n_batch, etc.).
        eval_config: Evaluator configuration (auto-detected if None).
        measure_perplexity: Whether to run perplexity measurement.

    Returns:
        EvalResult with all metrics populated.
    """
    if eval_config is None:
        eval_config = auto_detect_config()

    start = time.monotonic()

    # Step 1: Speed benchmark
    tok_s, vram_mb, speed_notes = _run_bench(model_path, config, eval_config)

    if "oom" in speed_notes:
        return EvalResult.failed(config, "oom", speed_notes)
    if "crash" in speed_notes:
        return EvalResult.failed(config, "crash", speed_notes)
    if "timeout" in speed_notes:
        return EvalResult.failed(config, "timeout", speed_notes)
    if tok_s <= 0:
        return EvalResult.failed(config, "error", speed_notes)

    # Step 2: Perplexity (optional)
    perplexity = float("inf")
    ppl_notes = "skipped"
    if measure_perplexity:
        perplexity, ppl_notes = _run_perplexity(model_path, config, eval_config)

    # Step 3: Quality score
    quality_score = compute_quality_score(perplexity, eval_config.baseline_perplexity)

    # Step 4: Quality-adjusted throughput
    quality_adj = tok_s * quality_score

    wall_time = time.monotonic() - start

    return EvalResult(
        tok_s=tok_s,
        perplexity=perplexity,
        quality_score=quality_score,
        quality_adj_throughput=quality_adj,
        vram_mb=vram_mb,
        config=config,
        status="ok",
        wall_time_s=wall_time,
        notes=f"{speed_notes} | {ppl_notes}",
    )
