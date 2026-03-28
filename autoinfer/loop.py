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
    "n_gpu": {"type": "int", "low": 12, "high": 28},
    "n_ctx": {"type": "categorical", "choices": [512, 1024, 2048]},
    "batch": {"type": "int", "low": 64, "high": 512},
    "ubatch": {"type": "int", "low": 32, "high": 256},
    "n_threads": {"type": "int", "low": 4, "high": 16},
    "n_gen": {"type": "int", "low": 128, "high": 512},
    "kv_type": {"type": "categorical", "choices": ["q8_0", "q4_0", "f16"]},
    "flash_attn": {"type": "categorical", "choices": [True, False]},
}


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
    """Load phase 10/11/12. Phase 12 has shifted columns (model_file=tok_s)."""
    results = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            headers = reader.fieldnames or []
            for row in reader:
                # Phase 12 check: if 'model_file' col has a float, it's tok_s
                if "model_file" in headers:
                    try:
                        tok_s = float(row.get("model_file") or 0)
                    except (ValueError, TypeError):
                        tok_s = float(row.get("tok_s") or 0)
                else:
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
                    "batch": int(row.get("n_batch") or row.get("batch_size") or 252),
                    "ubatch": int(row.get("n_ubatch") or row.get("ubatch_size") or 94),
                    "kv_type": kv_k,
                    "flash_attn": flash,
                    "n_threads": int(row.get("n_threads") or 11),
                    "n_gen": int(row.get("n_gen") or 264),
                }
                results.append(LegacyResult(tok_s=tok_s, params=params))
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
    max_seeds: int = 20,
) -> int:
    """Seed the Optuna study with top legacy results.

    Returns number of trials enqueued.
    """
    # Sort by tok/s descending, take top N
    top = sorted(legacy_results, key=lambda r: r.tok_s, reverse=True)[:max_seeds]
    enqueued = 0

    for r in top:
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
            if "n_gpu" in params and "batch" in params:
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

    # Warm-start with top legacy results
    n_seeded = _warm_start_study(study, legacy)
    logger.info(f"Seeded study with {n_seeded} top configurations")

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

        # Run the experiment
        result = run_experiment(
            params=params,
            bench_binary=config.bench_binary,
            model_path=config.model_path,
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
