"""Bayesian optimizer over inference parameter space.

Uses Optuna's TPE (Tree-structured Parzen Estimator) for efficient
exploration of the configuration space, maximizing quality-adjusted
throughput.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import optuna
from optuna.samplers import TPESampler

from autoinfer.evaluator import EvalConfig, EvalResult, evaluate
from autoinfer.params import ParamSpace, estimate_max_gpu_layers
from autoinfer.profiler import HardwareProfile
from autoinfer.results import ParetoFrontier, ResultsTracker, load_legacy_tsv

# Silence Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("autoinfer.optimizer")


@dataclass
class OptimizeResult:
    """Result of an optimization run."""

    best_config: dict = field(default_factory=dict)
    best_quality_adj: float = 0.0
    best_tok_s: float = 0.0
    best_quality_score: float = 0.0
    n_trials: int = 0
    n_ok: int = 0
    n_failed: int = 0
    frontier: Optional[ParetoFrontier] = None
    tracker: Optional[ResultsTracker] = None

    def summary(self) -> str:
        """Human-readable optimization summary."""
        lines = [
            f"Optimization complete: {self.n_trials} trials ({self.n_ok} ok, {self.n_failed} failed)",
            f"Best config: {json.dumps(self.best_config, indent=2)}",
            f"Best quality-adjusted throughput: {self.best_quality_adj:.2f}",
            f"  tok/s: {self.best_tok_s:.2f}",
            f"  quality: {self.best_quality_score:.3f}",
        ]
        if self.frontier:
            lines.append("")
            lines.append(self.frontier.summary())
        return "\n".join(lines)


def _create_objective(
    model_path: str,
    param_space: ParamSpace,
    eval_config: EvalConfig,
    tracker: ResultsTracker,
    target_quality: float,
    measure_perplexity: bool,
) -> Callable:
    """Create the objective function for Optuna."""

    def objective(trial: optuna.Trial) -> float:
        # Sample config from parameter space
        config = param_space.suggest(trial)

        # Run evaluation
        result = evaluate(
            model_path=model_path,
            config=config,
            eval_config=eval_config,
            measure_perplexity=measure_perplexity,
        )

        # Record result
        exp_id = tracker.record(result)

        if result.status != "ok":
            logger.debug(f"Trial {trial.number} failed: {result.status} ({result.notes})")
            return float("-inf")

        # Penalize below quality threshold
        if result.quality_score < target_quality:
            logger.debug(
                f"Trial {trial.number}: quality {result.quality_score:.3f} "
                f"< threshold {target_quality}"
            )
            # Soft penalty: proportional to how far below threshold
            penalty = (target_quality - result.quality_score) * 100
            return result.quality_adj_throughput - penalty

        logger.info(
            f"Trial {trial.number}: {result.tok_s:.2f} tok/s × "
            f"{result.quality_score:.3f} = {result.quality_adj_throughput:.2f}"
        )
        return result.quality_adj_throughput

    return objective


def optimize(
    model_path: str,
    hardware: HardwareProfile,
    eval_config: Optional[EvalConfig] = None,
    target_quality: float = 0.95,
    n_trials: int = 50,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    measure_perplexity: bool = True,
    warmup_paths: Optional[list[str]] = None,
) -> OptimizeResult:
    """Run Bayesian optimization to find the best inference config.

    Args:
        model_path: Path to the GGUF model.
        hardware: Hardware profile.
        eval_config: Evaluator config (auto-detected if None).
        target_quality: Minimum acceptable quality score (0-1).
        n_trials: Number of optimization trials.
        output_path: Path to save results TSV.
        seed: Random seed for reproducibility.
        measure_perplexity: Whether to measure perplexity.
        warmup_paths: Paths to legacy TSV files to seed the optimizer.

    Returns:
        OptimizeResult with best configuration and frontier.
    """
    if eval_config is None:
        from autoinfer.evaluator import auto_detect_config
        eval_config = auto_detect_config()

    # Create parameter space
    param_space = ParamSpace.default(hardware, model_path)
    logger.info(f"Parameter space:\n{param_space.summary()}")

    # Create results tracker
    tracker = ResultsTracker(output_path)

    # Load warmup data from legacy experiments
    if warmup_paths:
        for path in warmup_paths:
            legacy = load_legacy_tsv(path)
            logger.info(f"Loaded {len(legacy)} legacy results from {path}")
            for r in legacy:
                tracker.frontier.add(r)

    # Create Optuna study
    sampler = TPESampler(seed=seed, n_startup_trials=min(10, n_trials // 3))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"autoinfer_{os.path.basename(model_path)}",
    )

    # Seed study with known good configs from warmup data
    if warmup_paths:
        _seed_study_from_legacy(study, param_space, warmup_paths)

    # Create and run objective
    objective = _create_objective(
        model_path=model_path,
        param_space=param_space,
        eval_config=eval_config,
        tracker=tracker,
        target_quality=target_quality,
        measure_perplexity=measure_perplexity,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Build result
    best_trial = study.best_trial
    n_ok = sum(1 for r in tracker.results if r.status == "ok")

    return OptimizeResult(
        best_config=best_trial.params,
        best_quality_adj=best_trial.value if best_trial.value != float("-inf") else 0.0,
        best_tok_s=_extract_tok_s_from_tracker(tracker, best_trial.number),
        best_quality_score=_extract_quality_from_tracker(tracker, best_trial.number),
        n_trials=len(study.trials),
        n_ok=n_ok,
        n_failed=len(study.trials) - n_ok,
        frontier=tracker.frontier,
        tracker=tracker,
    )


def _seed_study_from_legacy(
    study: optuna.Study,
    param_space: ParamSpace,
    warmup_paths: list[str],
) -> None:
    """Add known good configs from legacy data as initial trials."""
    for path in warmup_paths:
        results = load_legacy_tsv(path)
        # Take top 5 by tok/s
        top = sorted(results, key=lambda r: r.tok_s, reverse=True)[:5]
        for r in top:
            try:
                # Map config to parameter names
                params = {}
                for p in param_space.params:
                    val = r.config.get(p.name)
                    if val is not None:
                        if p.type == "categorical" and val in (p.choices or []):
                            params[p.name] = val
                        elif p.type == "int":
                            params[p.name] = max(int(p.low or 0), min(int(p.high or 999), int(val)))
                        elif p.type == "float":
                            params[p.name] = max(p.low or 0, min(p.high or 999, float(val)))

                if params:
                    study.enqueue_trial(params)
            except (ValueError, KeyError):
                continue


def _extract_tok_s_from_tracker(tracker: ResultsTracker, trial_number: int) -> float:
    """Extract tok/s from tracker for a given trial number."""
    if trial_number < len(tracker.results):
        return tracker.results[trial_number].tok_s
    return 0.0


def _extract_quality_from_tracker(tracker: ResultsTracker, trial_number: int) -> float:
    """Extract quality score from tracker for a given trial number."""
    if trial_number < len(tracker.results):
        return tracker.results[trial_number].quality_score
    return 0.0


def optimize_from_existing(
    results_paths: list[str],
    target_quality: float = 0.95,
) -> OptimizeResult:
    """Analyze existing experiment data without running new experiments.

    Useful for finding the Pareto frontier from prior runs.
    """
    all_results = []
    for path in results_paths:
        all_results.extend(load_legacy_tsv(path))

    frontier = ParetoFrontier.from_results(all_results)
    best = frontier.best()

    return OptimizeResult(
        best_config=best.config if best else {},
        best_quality_adj=best.quality_adj_throughput if best else 0.0,
        best_tok_s=best.tok_s if best else 0.0,
        best_quality_score=best.quality_score if best else 0.0,
        n_trials=len(all_results),
        n_ok=len(all_results),
        n_failed=0,
        frontier=frontier,
    )
