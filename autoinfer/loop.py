"""AutoInfer autonomous research loop.

Runs forever: observe → hypothesize → experiment → learn → repeat.
Never stops. Never asks permission. Hardware sets the limit.

The loop:
1. Loads all existing TSV results on startup (warm start)
2. Seeds an Optuna TPE optimizer with historical data
3. Proposes next configuration via Bayesian optimization
4. Executes the Rust bench binary with proposed config
5. Parses tok/s result, appends to results store
6. Feeds result back into optimizer
7. Immediately proposes next experiment
8. Logs progress every 10 experiments
9. Reports new all-time bests
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import optuna
from optuna.samplers import TPESampler

from autoinfer.executor import ExperimentResult, run_experiment
from autoinfer.reporter import report_completion, report_new_best, report_progress

# Silence Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("autoinfer.loop")


# ── Search space definition ──────────────────────────────────────────

SEARCH_SPACE = {
    # Nemotron-Cascade-2-30B guidance (2026-03-29):
    # - Model is 17GB IQ2_XXS on 8GB VRAM → only 4-10 GPU layers fit; rest CPU-offloaded
    # - Apple Flash Attention (fa=1) REQUIRED: SSM+MoE hybrid needs memory-efficient attn
    # - PolarQuant / GIL KV cache: q4_0 and iq4_nl are the sweet spots for 8GB
    #   (q8_0 wastes KV VRAM; f16 OOMs; iq4_nl = Google QJL-inspired non-linear quant)
    # - GIL (GPU-CPU Interleaved Loading): keep n_threads=4-8, avoid 12+ (GIL contention)
    # - Conservative ngl: 4-12 layers only — anything higher OOMs on 8GB
    # - Small batch + ubatch: large batches = OOM; 32/16 or 64/32 are safe starting points
    # - n_ctx=512 max (larger = OOM on 8GB with 17GB model)
    "n_gpu": {"type": "int", "low": 0, "high": 12},          # Nemotron 17GB is CPU-only; 0=pure CPU, small ngl still tested
    "n_ctx": {"type": "categorical", "choices": [256, 512]},  # keep small — OOM risk
    "batch": {"type": "int", "low": 16, "high": 128},         # conservative batch sizes
    "ubatch": {"type": "int", "low": 8, "high": 64},          # ubatch ≤ batch always
    "n_threads": {"type": "int", "low": 2, "high": 8},        # GIL-friendly: 4-8 threads optimal
    "n_gen": {"type": "int", "low": 64, "high": 256},         # shorter gen = faster timeout detection
    "kv_type": {"type": "categorical", "choices": ["q4_0", "iq4_nl", "q8_0"]},  # PolarQuant: iq4_nl = non-linear quant; TurboQuant: q4_0=4x BW gain
    "kv_type_v": {"type": "categorical", "choices": ["q4_0", "q8_0", "same"]},  # TurboQuant asymmetric: K≠V compression; "same" = use kv_type for both
    "flash_attn": {"type": "categorical", "choices": [True]},  # FORCE fa=1: Apple Flash Attn required for SSM/MoE
}

# ── Nemotron priority seed configs ──────────────────────────────────
# Based on Apple Flash Attn + PolarQuant + GIL thread guidance.
# These are enqueued FIRST before any Bayesian proposals.
NEMOTRON_PRIORITY_SEEDS = [
    # ── Qwen3.5 transfer seeds ──────────────────────────────────────────────
    # Qwen3.5 all-time best: 29.899 tok/s @ n_gpu=27, batch=32/16, threads=8,
    # q8_0 KV, flash=1, op_offload=1.  Nemotron is CPU-only (17GB > 8GB VRAM)
    # so n_gpu is dropped to 0, but batch/threads/kv/flash transfer directly.
    # This is Bayesian transfer learning: Qwen posterior → Nemotron prior.
    #
    # Seed 1: direct Qwen best-config transfer (CPU-only, same batch/threads/KV)
    {"n_gpu": 0, "n_ctx": 512, "batch": 32, "ubatch": 16, "n_threads": 8,
     "n_gen": 64, "kv_type": "q8_0", "flash_attn": True},
    # Seed 2: Qwen best batch with 12 threads (Nemotron is denser — may need more)
    {"n_gpu": 0, "n_ctx": 256, "batch": 32, "ubatch": 16, "n_threads": 12,
     "n_gen": 64, "kv_type": "q8_0", "flash_attn": True},
    # Seed 3: larger batch from Qwen phase 10 (batch=252/94 scaled down for CPU)
    {"n_gpu": 0, "n_ctx": 256, "batch": 128, "ubatch": 64, "n_threads": 8,
     "n_gen": 64, "kv_type": "q4_0", "flash_attn": True},
    # ── Nemotron-specific seeds ─────────────────────────────────────────────
    # Seed 4: iq4_nl PolarQuant KV — non-linear quant, best quality/BW ratio
    {"n_gpu": 0, "n_ctx": 512, "batch": 64, "ubatch": 32, "n_threads": 8,
     "n_gen": 64, "kv_type": "iq4_nl", "flash_attn": True},
    # Seed 5: TurboQuant asymmetric — q8_0 K + q4_0 V (Google paper guidance)
    {"n_gpu": 0, "n_ctx": 256, "batch": 32, "ubatch": 16, "n_threads": 8,
     "n_gen": 64, "kv_type": "q8_0", "kv_type_v": "q4_0", "flash_attn": True},
]


# ── TSV column definitions ───────────────────────────────────────────

LOOP_TSV_COLUMNS = [
    "exp_id", "tok_s", "vram_mb", "status", "wall_time_s",
    "n_gpu", "n_ctx", "batch", "ubatch", "kv_type",
    "flash_attn", "n_threads", "n_gen", "notes",
]


# ── Legacy TSV loaders ───────────────────────────────────────────────

@dataclass
class LegacyResult:
    """A result loaded from legacy TSV files."""
    tok_s: float
    params: dict
    status: str = "ok"


def _load_legacy_phase456(path: str) -> list[LegacyResult]:
    """Load phase 4/5/6 format: exp, tok_per_sec, ..., type_k, type_v, flash_attn, n_gpu_layers, n_batch, n_ubatch, n_threads"""
    results = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                tok_s = float(row.get("tok_per_sec") or row.get("tok_s") or 0)
                status = row.get("status", "ok")
                if tok_s <= 0 or status not in ("ok", "keep", "baseline"):
                    continue

                # Map kv type names
                type_k = row.get("type_k") or row.get("kv_type_k") or "q8_0"
                flash_raw = row.get("flash_attn", "False")
                flash = flash_raw in ("True", "1", "true", True)

                params = {
                    "n_gpu": int(row.get("n_gpu_layers") or row.get("n_gpu") or 16),
                    "n_ctx": int(row.get("n_ctx") or 512),
                    "batch": int(row.get("n_batch") or 252),
                    "ubatch": int(row.get("n_ubatch") or 94),
                    "kv_type": type_k,
                    "flash_attn": flash,
                    "n_threads": int(row.get("n_threads") or 11),
                    "n_gen": int(row.get("gen_tokens") or row.get("gen") or row.get("n_gen") or 264),
                }
                results.append(LegacyResult(tok_s=tok_s, params=params))
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
    return results


def _load_legacy_phase789(path: str) -> list[LegacyResult]:
    """Load phase 7/8/9 format with explicit exp_id, tok_s columns."""
    results = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                tok_s = float(row.get("tok_s") or 0)
                status = row.get("status", "ok")
                if tok_s <= 0 or status != "ok":
                    continue

                kv_k = row.get("kv_type_k") or row.get("type_k") or "q8_0"
                flash_raw = row.get("flash_attn", "True")
                flash = flash_raw in ("True", "1", "true", True)

                params = {
                    "n_gpu": int(row.get("n_gpu") or 16),
                    "n_ctx": int(row.get("n_ctx") or 512),
                    "batch": int(row.get("n_batch") or 252),
                    "ubatch": int(row.get("n_ubatch") or 94),
                    "kv_type": kv_k,
                    "flash_attn": flash,
                    "n_threads": int(row.get("n_threads") or 11),
                    "n_gen": int(row.get("gen_tokens") or row.get("gen") or row.get("n_gen") or 264),
                }
                results.append(LegacyResult(tok_s=tok_s, params=params))
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
    return results


def _load_legacy_phase1012(path: str) -> list[LegacyResult]:
    """Load phase 10/11/12.

    Phase 12 has a mismatched header — the header claims 'model_file' etc.
    but the data rows follow the phase10 schema (exp_id, tok_s, vram_mb, ...).
    We detect this by checking if 'model_file' column contains a float.
    """
    results = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            headers = reader.fieldnames or []

            # Detect phase 12 shifted format
            is_phase12_shifted = "model_file" in headers

            for row in reader:
                try:
                    if is_phase12_shifted:
                        # Phase 12: header is wrong. Data rows are:
                        # exp_id, tok_s, vram_mb, n_ctx, kv_type_k, kv_type_v,
                        # flash_attn, n_gpu, n_batch, n_ubatch, label, status, notes
                        # But DictReader maps: model_file→tok_s, n_gpu→vram_mb, etc.
                        tok_s = float(row.get("model_file") or 0)
                        status_val = row.get("n_gen") or row.get("status") or "ok"
                        if tok_s <= 0 or status_val != "ok":
                            continue

                        # Parse from notes field which has the real details
                        notes = row.get("tok_s") or row.get("notes") or ""
                        # Notes format: "n_gpu=26, batch=32/16, q8_0 KV, n_ctx=256, ..."
                        import re
                        n_gpu_m = re.search(r"n_gpu=(\d+)", notes)
                        batch_m = re.search(r"batch=(\d+)/(\d+)", notes)
                        ctx_m = re.search(r"n_ctx=(\d+)", notes)

                        params = {
                            "n_gpu": int(n_gpu_m.group(1)) if n_gpu_m else 16,
                            "n_ctx": int(ctx_m.group(1)) if ctx_m else 512,
                            "batch": int(batch_m.group(1)) if batch_m else 252,
                            "ubatch": int(batch_m.group(2)) if batch_m else 94,
                            "kv_type": row.get("batch_size") or "q8_0",  # shifted: batch_size col = kv_type_k
                            "flash_attn": True,  # all phase 12 used flash
                            "n_threads": 11,
                            "n_gen": 200,  # phase 12 default
                        }
                    else:
                        # Standard phase 10/11 format
                        tok_s = float(row.get("tok_s") or 0)
                        status_val = row.get("status", "ok")
                        if tok_s <= 0 or status_val != "ok":
                            continue

                        kv_k = row.get("kv_type_k") or row.get("type_k") or "q8_0"
                        flash_raw = row.get("flash_attn", "True")
                        flash = flash_raw in ("True", "1", "true", True)

                        params = {
                            "n_gpu": int(row.get("n_gpu") or 16),
                            "n_ctx": int(row.get("n_ctx") or 512),
                            "batch": int(row.get("n_batch") or 252),
                            "ubatch": int(row.get("n_ubatch") or 94),
                            "kv_type": kv_k,
                            "flash_attn": flash,
                            "n_threads": int(row.get("n_threads") or 11),
                            "n_gen": 264,
                        }

                    results.append(LegacyResult(tok_s=tok_s, params=params))
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
    return results


def load_all_legacy(paths: list[str]) -> list[LegacyResult]:
    """Load results from all legacy TSV files, auto-detecting format."""
    all_results = []
    for path in paths:
        basename = os.path.basename(path).lower()
        if "phase4" in basename or "phase5" in basename or "phase6" in basename:
            results = _load_legacy_phase456(path)
        elif "phase7" in basename or "phase8" in basename or "phase9" in basename:
            results = _load_legacy_phase789(path)
        else:
            # Phase 10, 11, 12 or unknown — try the general loader
            results = _load_legacy_phase1012(path)

        if results:
            logger.info(f"Loaded {len(results)} results from {path}")
        all_results.extend(results)

    logger.info(f"Total legacy results loaded: {len(all_results)}")
    return all_results


# ── Optuna warm-start ────────────────────────────────────────────────

def _suggest_params(trial: optuna.Trial) -> dict:
    """Suggest parameters from the search space."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])

    # Constraint: ubatch ≤ batch
    if params["ubatch"] > params["batch"]:
        params["ubatch"] = params["batch"]

    return params


def _warm_start_study(
    study: optuna.Study,
    legacy_results: list[LegacyResult],
    max_seeds: int = 5,
) -> int:
    """Seed the Optuna study with top legacy results.

    Only seeds a small number (default 5) to let the optimizer explore
    freely. Uses diverse configs — dedup by n_gpu to avoid all seeds
    hitting the same GPU count (which may OOM for different model sizes).

    Returns number of trials enqueued.
    """
    # Sort by tok/s descending
    top = sorted(legacy_results, key=lambda r: r.tok_s, reverse=True)
    enqueued = 0
    seen_n_gpu: set[int] = set()

    for r in top:
        if enqueued >= max_seeds:
            break
        try:
            params = {}
            for name, spec in SEARCH_SPACE.items():
                val = r.params.get(name)
                if val is None:
                    continue
                if spec["type"] == "int":
                    val = int(val)
                    val = max(spec["low"], min(spec["high"], val))
                elif spec["type"] == "categorical":
                    if val not in spec["choices"]:
                        continue
                params[name] = val

            # Must have at least the key params
            if "n_gpu" not in params or "batch" not in params:
                continue

            # Dedup by n_gpu to get diverse GPU allocations
            n_gpu = params["n_gpu"]
            if n_gpu in seen_n_gpu:
                continue
            seen_n_gpu.add(n_gpu)

            study.enqueue_trial(params)
            enqueued += 1
        except (ValueError, KeyError, TypeError):
            continue

    return enqueued


# ── TSV output ───────────────────────────────────────────────────────

def _init_tsv(path: str) -> None:
    """Write TSV header if file doesn't exist."""
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("\t".join(LOOP_TSV_COLUMNS) + "\n")


def _append_tsv(path: str, exp_id: int, params: dict, result: ExperimentResult) -> None:
    """Append one result row to the TSV file."""
    row = {
        "exp_id": exp_id,
        "tok_s": f"{result.tok_s:.3f}" if result.tok_s > 0 else "0.000",
        "vram_mb": result.vram_mb,
        "status": result.status,
        "wall_time_s": f"{result.wall_time_s:.1f}",
        "n_gpu": params.get("n_gpu", ""),
        "n_ctx": params.get("n_ctx", ""),
        "batch": params.get("batch", ""),
        "ubatch": params.get("ubatch", ""),
        "kv_type": params.get("kv_type", ""),
        "flash_attn": params.get("flash_attn", ""),
        "n_threads": params.get("n_threads", ""),
        "n_gen": params.get("n_gen", ""),
        "notes": result.notes,
    }
    with open(path, "a") as f:
        f.write("\t".join(str(row.get(c, "")) for c in LOOP_TSV_COLUMNS) + "\n")


# ── Main loop ────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    """Configuration for the autonomous research loop."""
    bench_binary: str = ""
    model_path: str = ""
    warmup_paths: list[str] = field(default_factory=list)
    output_path: str = ""
    max_experiments: int = 0  # 0 = infinite
    report_interval: int = 10
    seed: int = 42


def run_loop(config: LoopConfig) -> dict:
    """Run the autonomous research loop.

    Returns a summary dict with results.
    """
    start_time = time.monotonic()

    # ── 1. Load legacy data ──
    legacy = load_all_legacy(config.warmup_paths)
    best_tok_s = max((r.tok_s for r in legacy), default=0.0)
    best_params: dict = {}
    if legacy:
        best_result = max(legacy, key=lambda r: r.tok_s)
        best_params = best_result.params.copy()
    logger.info(f"Historical best: {best_tok_s:.3f} tok/s")

    # ── 2. Create Optuna study ──
    sampler = TPESampler(
        seed=config.seed,
        n_startup_trials=min(10, max(5, len(legacy) // 10)),
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="autoinfer_loop",
    )

    # ── Nemotron priority seeds: enqueue FIRST before any Bayesian proposals ──
    # Apple Flash Attn (forced), PolarQuant/iq4_nl KV, GIL thread counts, conservative ngl
    for seed_params in NEMOTRON_PRIORITY_SEEDS:
        try:
            study.enqueue_trial(seed_params)
        except Exception:
            pass
    logger.info(f"Enqueued {len(NEMOTRON_PRIORITY_SEEDS)} Nemotron priority seeds (Flash+PolarQuant+GIL)")

    # Warm-start with top legacy results (after priority seeds so they run first)
    n_seeded = _warm_start_study(study, legacy)
    logger.info(f"Seeded study with {n_seeded} top legacy configurations")

    # ── 3. Init output TSV ──
    if config.output_path:
        _init_tsv(config.output_path)

    # ── 4. Run the loop ──
    exp_count = 0
    new_bests = 0
    failures = 0
    recent_tok_s: list[float] = []
    results_log: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal exp_count, best_tok_s, best_params, new_bests, failures, recent_tok_s

        params = _suggest_params(trial)
        exp_count += 1

        logger.debug(f"Experiment #{exp_count}: {params}")

        # Run the experiment — 600s timeout for large CPU-only models (17GB needs ~30s load)
        result = run_experiment(
            params=params,
            bench_binary=config.bench_binary,
            model_path=config.model_path,
            timeout=600,
        )

        # Record to TSV
        if config.output_path:
            _append_tsv(config.output_path, exp_count, params, result)

        # Track result
        entry = {
            "exp_id": exp_count,
            "tok_s": result.tok_s,
            "status": result.status,
            "params": params.copy(),
        }
        results_log.append(entry)

        if not result.success:
            failures += 1
            logger.debug(f"  → FAILED: {result.status} ({result.notes})")

            # ── Self-direction: detect failure patterns and adapt ──
            recent_statuses = [e["status"] for e in results_log[-5:]] if results_log else []
            consecutive_timeouts = sum(1 for s in recent_statuses if s == "timeout")
            consecutive_ooms = sum(1 for s in recent_statuses if s == "oom")

            if consecutive_timeouts >= 3:
                # Too many timeouts: shrink experiment to fit within timeout budget
                if params.get("n_gen", 128) > 64:
                    logger.warning(f"[self-direct] {consecutive_timeouts} consecutive timeouts — "
                                   f"enqueuing reduced n_gen=64, n_ctx=256 variant")
                    reduced = params.copy()
                    reduced["n_gen"] = 64
                    reduced["n_ctx"] = 256
                    try:
                        study.enqueue_trial(reduced)
                    except Exception:
                        pass
                elif params.get("n_gpu", 0) > 0:
                    # Still timing out with small gen? Drop to CPU-only
                    logger.warning(f"[self-direct] Persistent timeouts — forcing n_gpu=0 (CPU-only)")
                    cpu_params = params.copy()
                    cpu_params["n_gpu"] = 0
                    cpu_params["n_gen"] = 64
                    cpu_params["n_ctx"] = 256
                    try:
                        study.enqueue_trial(cpu_params)
                    except Exception:
                        pass

            if consecutive_ooms >= 3:
                # OOM: reduce GPU layers
                current_ngl = params.get("n_gpu", 0)
                if current_ngl > 2:
                    reduced_ngl = max(0, current_ngl - 4)
                    logger.warning(f"[self-direct] {consecutive_ooms} consecutive OOMs — "
                                   f"enqueuing n_gpu={reduced_ngl}")
                    oom_params = params.copy()
                    oom_params["n_gpu"] = reduced_ngl
                    try:
                        study.enqueue_trial(oom_params)
                    except Exception:
                        pass

            return float("-inf")

        recent_tok_s.append(result.tok_s)
        if len(recent_tok_s) > 10:
            recent_tok_s = recent_tok_s[-10:]

        # Check for new best
        if result.tok_s > best_tok_s:
            prev_best = best_tok_s
            best_tok_s = result.tok_s
            best_params = params.copy()
            new_bests += 1
            report_new_best(exp_count, result.tok_s, prev_best, params)

        logger.info(f"  → {result.tok_s:.3f} tok/s ({result.status})")

        # Periodic progress report
        if exp_count % config.report_interval == 0:
            avg = sum(recent_tok_s) / len(recent_tok_s) if recent_tok_s else 0
            report_progress(exp_count, exp_count, best_tok_s, best_params, avg, failures)

        return result.tok_s

    # Determine trial count
    n_trials = config.max_experiments if config.max_experiments > 0 else 999_999_999

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except KeyboardInterrupt:
        logger.info("Loop interrupted by user")

    # ── 5. Report completion ──
    duration = time.monotonic() - start_time
    report_completion(
        total_experiments=exp_count,
        best_tok_s=best_tok_s,
        best_params=best_params,
        new_bests_found=new_bests,
        duration_s=duration,
        output_path=config.output_path,
    )

    return {
        "total_experiments": exp_count,
        "best_tok_s": best_tok_s,
        "best_params": best_params,
        "new_bests": new_bests,
        "failures": failures,
        "duration_s": duration,
        "results": results_log[:50],  # First 50 for reporting
    }
