"""Executes the Rust bench binary and parses output.

Wraps the qwen35-moe-offload bench binary, handling environment setup,
argument mapping, output parsing, and failure modes (OOM, timeout, crash).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("autoinfer.executor")

# Defaults — overridable via env vars or CLI args
BENCH_BINARY = os.environ.get(
    "AUTOINFER_BENCH", "/tmp/qwen35-moe-offload/target/release/bench"
)
LLAMA_LIB = os.environ.get("LLAMA_LIB", "/tmp/llama-cpp-build")
CUDA_LIB = os.environ.get("CUDA_LIB", "/usr/local/lib/ollama/cuda_v12")
MODEL_PATH = os.environ.get(
    "AUTOINFER_MODEL",
    "/tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf",
)

# KV type name → integer code for the bench binary
KV_NAME_TO_INT = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
    "iq4_nl": 20,
}


@dataclass
class ExperimentResult:
    """Result from a single bench execution."""

    tok_s: float = -1.0
    vram_mb: int = 0
    wall_time_s: float = 0.0
    status: str = "unknown"  # ok, oom, crash, timeout, error
    notes: str = ""
    raw_output: str = ""

    @property
    def success(self) -> bool:
        return self.status == "ok" and self.tok_s > 0


def _build_env(llama_lib: str = LLAMA_LIB, cuda_lib: str = CUDA_LIB) -> dict:
    """Build environment with correct library paths."""
    env = os.environ.copy()
    ld_paths = [llama_lib, cuda_lib]
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        ld_paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld_paths if p)
    return env


def _parse_tok_s(output: str) -> float:
    """Parse tok/s from bench output. Returns -1.0 if not found."""
    # Pattern 1: "internal=12.331tok/s"
    m = re.search(r"internal=([\d.]+)\s*tok/s", output)
    if m:
        return float(m.group(1))

    # Pattern 2: "wall=12.331tok/s"
    m = re.search(r"wall=([\d.]+)\s*tok/s", output)
    if m:
        return float(m.group(1))

    # Pattern 3: "X.XX tokens per second"
    m = re.search(r"([\d.]+)\s*tokens?\s*per\s*second", output, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern 4: "tok/s: X.XX" or "tok/s = X.XX"
    m = re.search(r"tok/s\s*[=:]\s*([\d.]+)", output, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern 5: generic "X.XX tok/s"
    m = re.search(r"([\d.]+)\s*tok/s", output)
    if m:
        return float(m.group(1))

    return -1.0


def _parse_vram(output: str) -> int:
    """Parse VRAM usage from bench output."""
    m = re.search(r"VRAM[:\s]+([\d]+)\s*MB", output, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"vram[=:]([\d]+)", output)
    if m:
        return int(m.group(1))
    return 0


def run_experiment(
    params: dict,
    bench_binary: str = BENCH_BINARY,
    model_path: str = MODEL_PATH,
    llama_lib: str = LLAMA_LIB,
    cuda_lib: str = CUDA_LIB,
    timeout: int = 300,
) -> ExperimentResult:
    """Run bench binary with params, return ExperimentResult.

    Args:
        params: Dict with keys: n_gpu, n_ctx, batch, ubatch, kv_type,
                n_threads, n_gen, flash_attn, op_offload, no_mmap, etc.
        bench_binary: Path to the bench binary.
        model_path: Path to the GGUF model.
        llama_lib: Path to llama.cpp shared libs.
        cuda_lib: Path to CUDA runtime libs.
        timeout: Max seconds before killing the process.

    Returns:
        ExperimentResult with tok/s, status, timing info.
    """
    start = time.monotonic()

    # Keep kv_type as string name for llama-bench CLI (e.g. "q8_0", "f16")
    kv_type = params.get("kv_type", "q8_0")
    if isinstance(kv_type, str):
        type_k = kv_type  # pass directly e.g. "q8_0"
    else:
        # reverse map int→name
        _INT_TO_KV = {v: k for k, v in KV_NAME_TO_INT.items()}
        type_k = _INT_TO_KV.get(int(kv_type), "q8_0")
    type_v = type_k

    flash_attn_val = params.get("flash_attn", True)
    if isinstance(flash_attn_val, bool):
        flash_int = 1 if flash_attn_val else 0
    else:
        flash_int = int(flash_attn_val)

    cmd = [
        bench_binary,
        "-m", model_path,
        "-ngl", str(params.get("n_gpu", 16)),
        "-p", str(params.get("n_ctx", 512)),   # n-prompt (prefill tokens)
        "-b", str(params.get("batch", 252)),    # batch-size
        "-ub", str(params.get("ubatch", 94)),   # ubatch-size
        "-ctk", str(type_k),
        "-ctv", str(type_v),
        "-fa", str(flash_int),
        "-t", str(params.get("n_threads", 11)),
        "-n", str(params.get("n_gen", 264)),
        "-nopo", "0",  # enable op offload (no-op-offload=0)
    ]

    if params.get("no_mmap", False):
        cmd.extend(["-mmp", "0"])

    env = _build_env(llama_lib, cuda_lib)

    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        output = result.stdout + "\n" + result.stderr
        wall_time = time.monotonic() - start

        # Check for OOM
        output_lower = output.lower()
        if "out of memory" in output_lower or "failed to create context" in output_lower:
            return ExperimentResult(
                tok_s=-1.0,
                wall_time_s=wall_time,
                status="oom",
                notes="OOM during model loading or inference",
                raw_output=output[:500],
            )

        # Parse tok/s
        tok_s = _parse_tok_s(output)
        vram_mb = _parse_vram(output)

        if tok_s <= 0:
            if result.returncode != 0:
                return ExperimentResult(
                    tok_s=-1.0,
                    wall_time_s=wall_time,
                    status="crash",
                    notes=f"rc={result.returncode}, no tok/s parsed",
                    raw_output=output[:500],
                )
            return ExperimentResult(
                tok_s=-1.0,
                wall_time_s=wall_time,
                status="error",
                notes="no tok/s found in output",
                raw_output=output[:500],
            )

        return ExperimentResult(
            tok_s=tok_s,
            vram_mb=vram_mb,
            wall_time_s=wall_time,
            status="ok",
            notes=f"tok/s={tok_s:.3f}, vram={vram_mb}MB, wall={wall_time:.1f}s",
            raw_output=output[:500],
        )

    except subprocess.TimeoutExpired:
        return ExperimentResult(
            tok_s=-1.0,
            wall_time_s=time.monotonic() - start,
            status="timeout",
            notes=f"Timed out after {timeout}s (likely OOM or hang)",
        )
    except FileNotFoundError:
        return ExperimentResult(
            tok_s=-1.0,
            wall_time_s=0.0,
            status="error",
            notes=f"Bench binary not found: {bench_binary}",
        )
    except Exception as e:
        return ExperimentResult(
            tok_s=-1.0,
            wall_time_s=time.monotonic() - start,
            status="error",
            notes=f"Exception: {type(e).__name__}: {e}",
        )
