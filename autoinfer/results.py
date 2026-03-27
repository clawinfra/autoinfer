"""Results tracking, TSV logging, and Pareto frontier analysis."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from autoinfer.evaluator import EvalResult


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""

    tok_s: float
    perplexity: float
    quality_score: float
    quality_adj_throughput: float
    config: dict

    def dominates(self, other: "ParetoPoint") -> bool:
        """Check if this point Pareto-dominates another.

        A point dominates if it's better in quality-adjusted throughput
        AND has equal or better quality score.
        """
        return (
            self.quality_adj_throughput >= other.quality_adj_throughput
            and self.quality_score >= other.quality_score
            and (
                self.quality_adj_throughput > other.quality_adj_throughput
                or self.quality_score > other.quality_score
            )
        )


class ParetoFrontier:
    """Computes and maintains the Pareto frontier of evaluation results.

    The frontier contains all non-dominated points in the
    (quality_score, tok_s) space.
    """

    def __init__(self) -> None:
        self.points: list[ParetoPoint] = []

    def add(self, result: EvalResult) -> bool:
        """Add a result and update the frontier.

        Returns True if the point is on the new frontier.
        """
        new_point = ParetoPoint(
            tok_s=result.tok_s,
            perplexity=result.perplexity,
            quality_score=result.quality_score,
            quality_adj_throughput=result.quality_adj_throughput,
            config=result.config,
        )

        # Check if new point is dominated by any existing point
        for existing in self.points:
            if existing.dominates(new_point):
                return False

        # Remove points dominated by new point
        self.points = [p for p in self.points if not new_point.dominates(p)]
        self.points.append(new_point)

        # Sort by quality_adj_throughput descending
        self.points.sort(key=lambda p: p.quality_adj_throughput, reverse=True)
        return True

    def best(self) -> Optional[ParetoPoint]:
        """Return the point with highest quality-adjusted throughput."""
        return self.points[0] if self.points else None

    def best_at_quality(self, min_quality: float) -> Optional[ParetoPoint]:
        """Return fastest point that meets a minimum quality threshold."""
        candidates = [p for p in self.points if p.quality_score >= min_quality]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.tok_s)

    def summary(self) -> str:
        """Human-readable frontier summary."""
        if not self.points:
            return "Pareto frontier: empty"

        lines = [f"Pareto frontier ({len(self.points)} points):"]
        for i, p in enumerate(self.points):
            lines.append(
                f"  #{i + 1}: {p.tok_s:.2f} tok/s × {p.quality_score:.3f} quality "
                f"= {p.quality_adj_throughput:.2f} quality-adj | "
                f"ppl={p.perplexity:.2f}"
            )
        return "\n".join(lines)

    @classmethod
    def from_results(cls, results: list[EvalResult]) -> "ParetoFrontier":
        """Build frontier from a list of results."""
        frontier = cls()
        for r in results:
            if r.status == "ok" and r.tok_s > 0:
                frontier.add(r)
        return frontier


# TSV column names
TSV_COLUMNS = [
    "exp_id",
    "tok_s",
    "perplexity",
    "quality_score",
    "quality_adj",
    "vram_mb",
    "n_gpu",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "type_k",
    "type_v",
    "flash_attn",
    "status",
    "wall_time_s",
    "notes",
]


class ResultsTracker:
    """Track and persist evaluation results."""

    def __init__(self, output_path: Optional[str] = None) -> None:
        self.results: list[EvalResult] = []
        self.output_path = output_path
        self._next_id = 1
        self.frontier = ParetoFrontier()

        # Load existing results if file exists
        if output_path and os.path.isfile(output_path):
            self._load_existing()

    def _load_existing(self) -> None:
        """Load results from existing TSV file."""
        if not self.output_path:
            return
        try:
            with open(self.output_path, "r") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    try:
                        result = EvalResult(
                            tok_s=float(row.get("tok_s", 0)),
                            perplexity=float(row.get("perplexity", "inf")),
                            quality_score=float(row.get("quality_score", 0)),
                            quality_adj_throughput=float(row.get("quality_adj", 0)),
                            vram_mb=int(row.get("vram_mb", 0)),
                            config={
                                "n_gpu": int(row.get("n_gpu", 0)),
                                "n_batch": int(row.get("n_batch", 256)),
                                "n_ubatch": int(row.get("n_ubatch", 128)),
                                "n_threads": int(row.get("n_threads", 4)),
                                "type_k": int(row.get("type_k", 8)),
                                "type_v": int(row.get("type_v", 8)),
                                "flash_attn": int(row.get("flash_attn", 0)),
                            },
                            status=row.get("status", "ok"),
                            wall_time_s=float(row.get("wall_time_s", 0)),
                            notes=row.get("notes", ""),
                        )
                        self.results.append(result)
                        if result.status == "ok":
                            self.frontier.add(result)
                        self._next_id += 1
                    except (ValueError, KeyError):
                        self._next_id += 1
                        continue
        except (OSError, csv.Error):
            pass

    def record(self, result: EvalResult) -> int:
        """Record a result and append to TSV file.

        Returns the experiment ID.
        """
        exp_id = self._next_id
        self._next_id += 1
        self.results.append(result)

        if result.status == "ok":
            self.frontier.add(result)

        # Append to TSV
        if self.output_path:
            self._append_tsv(exp_id, result)

        return exp_id

    def _append_tsv(self, exp_id: int, result: EvalResult) -> None:
        """Append a single result to the TSV file."""
        if not self.output_path:
            return

        file_exists = os.path.isfile(self.output_path)
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        with open(self.output_path, "a", newline="") as f:
            if not file_exists:
                f.write("\t".join(TSV_COLUMNS) + "\n")

            row = self._result_to_row(exp_id, result)
            f.write("\t".join(str(row.get(c, "")) for c in TSV_COLUMNS) + "\n")

    def _result_to_row(self, exp_id: int, result: EvalResult) -> dict:
        """Convert EvalResult to TSV row dict."""
        cfg = result.config
        ppl = result.perplexity if result.perplexity != float("inf") else "inf"
        return {
            "exp_id": exp_id,
            "tok_s": f"{result.tok_s:.3f}",
            "perplexity": ppl if isinstance(ppl, str) else f"{ppl:.4f}",
            "quality_score": f"{result.quality_score:.4f}",
            "quality_adj": f"{result.quality_adj_throughput:.3f}",
            "vram_mb": result.vram_mb,
            "n_gpu": cfg.get("n_gpu", 0),
            "n_batch": cfg.get("n_batch", 256),
            "n_ubatch": cfg.get("n_ubatch", 128),
            "n_threads": cfg.get("n_threads", 4),
            "type_k": cfg.get("type_k", 8),
            "type_v": cfg.get("type_v", 8),
            "flash_attn": cfg.get("flash_attn", 0),
            "status": result.status,
            "wall_time_s": f"{result.wall_time_s:.1f}",
            "notes": result.notes,
        }

    def to_tsv(self) -> str:
        """Export all results as a TSV string."""
        buf = io.StringIO()
        buf.write("\t".join(TSV_COLUMNS) + "\n")
        for i, result in enumerate(self.results, 1):
            row = self._result_to_row(i, result)
            buf.write("\t".join(str(row.get(c, "")) for c in TSV_COLUMNS) + "\n")
        return buf.getvalue()

    def summary(self) -> str:
        """Human-readable summary."""
        total = len(self.results)
        ok = sum(1 for r in self.results if r.status == "ok")
        failed = total - ok
        best = self.frontier.best()

        lines = [
            f"Results: {total} total ({ok} ok, {failed} failed)",
        ]

        if best:
            lines.append(
                f"Best: {best.tok_s:.2f} tok/s × {best.quality_score:.3f} "
                f"= {best.quality_adj_throughput:.2f} quality-adj"
            )

        lines.append(self.frontier.summary())
        return "\n".join(lines)


def load_legacy_tsv(path: str) -> list[EvalResult]:
    """Load results from legacy TSV format (phase9/10/11 files).

    Handles the format from qwen35-moe-offload experiments.
    """
    results = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    tok_s_raw = row.get("tok_s")
                    tok_s = float(tok_s_raw) if tok_s_raw is not None else 0.0
                    vram_raw = row.get("vram_mb")
                    vram_mb = int(vram_raw) if vram_raw is not None else 0
                    status = row.get("status", "ok") or "ok"

                    if tok_s <= 0 or status != "ok":
                        continue

                    # Map legacy fields
                    config = {
                        "n_gpu": int(row.get("n_gpu", 0)),
                        "n_batch": int(row.get("n_batch", 256)),
                        "n_ubatch": int(row.get("n_ubatch", 128)),
                    }

                    # Parse kv_type_k/v
                    kv_k = row.get("kv_type_k", "q8_0")
                    kv_v = row.get("kv_type_v", "q8_0")
                    kv_map = {"f16": 1, "q4_0": 4, "q8_0": 8, "f32": 0}
                    config["type_k"] = kv_map.get(kv_k, 8)
                    config["type_v"] = kv_map.get(kv_v, 8)

                    flash = row.get("flash_attn", "False")
                    config["flash_attn"] = 1 if flash in ("True", "1", "true") else 0

                    result = EvalResult(
                        tok_s=tok_s,
                        vram_mb=vram_mb,
                        config=config,
                        status=status,
                        notes=row.get("notes", ""),
                    )
                    results.append(result)
                except (ValueError, KeyError):
                    continue
    except (OSError, csv.Error):
        pass

    return results
