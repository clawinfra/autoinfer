"""CLI interface for AutoInfer.

Usage:
    autoinfer profile                          # Show hardware profile
    autoinfer optimize --model <path>          # Run optimization
    autoinfer analyze --results <path> [...]   # Analyze existing data
    autoinfer loop --model <path>              # Run autonomous research loop
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from autoinfer.evaluator import EvalConfig, auto_detect_config
from autoinfer.optimizer import optimize, optimize_from_existing
from autoinfer.profiler import profile_hardware
from autoinfer.results import load_legacy_tsv, ParetoFrontier


def cmd_profile(args: argparse.Namespace) -> int:
    """Show hardware profile."""
    hw = profile_hardware(measure_storage=args.storage)
    print(hw.summary())
    if args.json:
        import dataclasses
        d = dataclasses.asdict(hw)
        print(json.dumps(d, indent=2))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run optimization."""
    from autoinfer.profiler import profile_hardware

    print(f"Model: {args.model}")
    print("Profiling hardware...")
    hw = profile_hardware(measure_storage=False)
    print(f"Hardware: {hw.summary()}")

    eval_config = auto_detect_config(workdir=args.workdir or "")
    if args.bench:
        eval_config.bench_binary = args.bench
    if args.corpus:
        eval_config.corpus_path = args.corpus
    if args.baseline_ppl:
        eval_config.baseline_perplexity = args.baseline_ppl
    if args.ld_library_path:
        eval_config.ld_library_path = args.ld_library_path

    warmup = args.warmup.split(",") if args.warmup else None

    print(f"\nRunning {args.trials} optimization trials...")
    print(f"Target quality: {args.target_quality}")
    if warmup:
        print(f"Warmup data: {warmup}")

    result = optimize(
        model_path=args.model,
        hardware=hw,
        eval_config=eval_config,
        target_quality=args.target_quality,
        n_trials=args.trials,
        output_path=args.output,
        seed=args.seed,
        measure_perplexity=not args.skip_perplexity,
        warmup_paths=warmup,
    )

    print("\n" + result.summary())

    if args.output:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    """Run the autonomous research loop."""
    import glob as glob_mod
    from autoinfer.loop import LoopConfig, run_loop
    from autoinfer.executor import BENCH_BINARY, MODEL_PATH

    bench = args.bench or BENCH_BINARY
    model = args.model or MODEL_PATH

    # Expand globs in results paths
    warmup_paths = []
    for pattern in (args.results or []):
        expanded = glob_mod.glob(pattern)
        warmup_paths.extend(expanded if expanded else [pattern])

    # Default warmup: try to find phase files
    if not warmup_paths:
        default_pattern = "/tmp/qwen35-moe-offload/results_phase*.tsv"
        warmup_paths = sorted(glob_mod.glob(default_pattern))

    output = args.output or "/tmp/qwen35-moe-offload/results_autoinfer.tsv"

    print(f"AutoInfer Autonomous Research Loop")
    print(f"  Bench:    {bench}")
    print(f"  Model:    {model}")
    print(f"  Warmup:   {len(warmup_paths)} files")
    print(f"  Output:   {output}")
    print(f"  Max exp:  {args.max_experiments or '∞'}")
    print()

    config = LoopConfig(
        bench_binary=bench,
        model_path=model,
        warmup_paths=warmup_paths,
        output_path=output,
        max_experiments=args.max_experiments,
        report_interval=args.report_interval,
        seed=args.seed,
    )

    summary = run_loop(config)

    # Print final summary
    print(f"\nFinal summary:")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Best tok/s: {summary['best_tok_s']:.3f}")
    print(f"  New bests: {summary['new_bests']}")
    print(f"  Failures: {summary['failures']}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze existing experiment data."""
    print(f"Analyzing {len(args.results)} result files...")

    all_results = []
    for path in args.results:
        results = load_legacy_tsv(path)
        print(f"  {path}: {len(results)} valid results")
        all_results.extend(results)

    if not all_results:
        print("No valid results found.")
        return 1

    print(f"\nTotal: {len(all_results)} results")

    # Build Pareto frontier
    frontier = ParetoFrontier.from_results(all_results)
    print(f"\n{frontier.summary()}")

    # Top configs by raw tok/s
    top_speed = sorted(all_results, key=lambda r: r.tok_s, reverse=True)[:5]
    print("\nTop 5 by raw tok/s:")
    for i, r in enumerate(top_speed, 1):
        gpu = r.config.get("n_gpu", "?")
        batch = r.config.get("n_batch", "?")
        flash = "flash" if r.config.get("flash_attn") else "noflash"
        print(f"  #{i}: {r.tok_s:.2f} tok/s | gpu={gpu} batch={batch} {flash} | vram={r.vram_mb}MB")

    # Quality at different thresholds
    if args.target_quality:
        best_at_quality = frontier.best_at_quality(args.target_quality)
        if best_at_quality:
            print(f"\nBest at quality ≥ {args.target_quality}:")
            print(f"  {best_at_quality.tok_s:.2f} tok/s × {best_at_quality.quality_score:.3f}")
        else:
            print(f"\nNo results meet quality ≥ {args.target_quality}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="autoinfer",
        description="Universal hardware-adaptive LLM inference optimizer",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Profile command
    p_profile = subparsers.add_parser("profile", help="Show hardware profile")
    p_profile.add_argument("--storage", action="store_true", help="Measure storage speed")
    p_profile.add_argument("--json", action="store_true", help="Output as JSON")

    # Optimize command
    p_opt = subparsers.add_parser("optimize", help="Run optimization")
    p_opt.add_argument("--model", required=True, help="Path to GGUF model")
    p_opt.add_argument("--trials", type=int, default=50, help="Number of trials")
    p_opt.add_argument("--target-quality", type=float, default=0.95, help="Min quality score")
    p_opt.add_argument("--output", help="Output TSV path")
    p_opt.add_argument("--bench", help="Path to bench binary")
    p_opt.add_argument("--corpus", help="Path to eval corpus")
    p_opt.add_argument("--baseline-ppl", type=float, help="Baseline perplexity")
    p_opt.add_argument("--ld-library-path", help="LD_LIBRARY_PATH for CUDA")
    p_opt.add_argument("--workdir", help="Working directory")
    p_opt.add_argument("--warmup", help="Comma-separated legacy TSV paths")
    p_opt.add_argument("--seed", type=int, help="Random seed")
    p_opt.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity")

    # Analyze command
    p_analyze = subparsers.add_parser("analyze", help="Analyze existing data")
    p_analyze.add_argument("results", nargs="+", help="TSV result files")
    p_analyze.add_argument("--target-quality", type=float, default=0.95, help="Quality threshold")

    # Loop command
    p_loop = subparsers.add_parser("loop", help="Run autonomous research loop")
    p_loop.add_argument("--model", help="Path to GGUF model")
    p_loop.add_argument("--bench", help="Path to bench binary")
    p_loop.add_argument(
        "--results", nargs="*", default=[],
        help="Legacy TSV result files to warm-start from"
    )
    p_loop.add_argument("--output", help="Output TSV path for new results")
    p_loop.add_argument(
        "--max-experiments", type=int, default=0,
        help="Max experiments to run (0 = infinite)"
    )
    p_loop.add_argument(
        "--report-interval", type=int, default=10,
        help="Report progress every N experiments"
    )
    p_loop.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    if args.command == "profile":
        return cmd_profile(args)
    elif args.command == "optimize":
        return cmd_optimize(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "loop":
        return cmd_loop(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
