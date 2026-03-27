"""Tests for results tracking and Pareto frontier."""

from __future__ import annotations

import os
import tempfile

import pytest

from autoinfer.evaluator import EvalResult
from autoinfer.results import (
    ParetoFrontier,
    ParetoPoint,
    ResultsTracker,
    TSV_COLUMNS,
    load_legacy_tsv,
)


class TestParetoPoint:
    def test_dominates_better_in_both(self):
        a = ParetoPoint(tok_s=15.0, perplexity=5.0, quality_score=1.0,
                         quality_adj_throughput=15.0, config={})
        b = ParetoPoint(tok_s=10.0, perplexity=6.0, quality_score=0.8,
                         quality_adj_throughput=8.0, config={})
        assert a.dominates(b) is True
        assert b.dominates(a) is False

    def test_no_domination_tradeoff(self):
        # a is faster but lower quality
        a = ParetoPoint(tok_s=20.0, perplexity=8.0, quality_score=0.7,
                         quality_adj_throughput=14.0, config={})
        # b is slower but higher quality
        b = ParetoPoint(tok_s=10.0, perplexity=4.0, quality_score=1.2,
                         quality_adj_throughput=12.0, config={})
        assert a.dominates(b) is False
        assert b.dominates(a) is False

    def test_equal_not_dominating(self):
        a = ParetoPoint(tok_s=10.0, perplexity=5.0, quality_score=1.0,
                         quality_adj_throughput=10.0, config={})
        b = ParetoPoint(tok_s=10.0, perplexity=5.0, quality_score=1.0,
                         quality_adj_throughput=10.0, config={})
        assert a.dominates(b) is False


class TestParetoFrontier:
    def test_empty_frontier(self):
        f = ParetoFrontier()
        assert f.best() is None
        assert f.best_at_quality(0.5) is None
        assert "empty" in f.summary()

    def test_single_point(self):
        f = ParetoFrontier()
        r = EvalResult(tok_s=12.0, perplexity=5.5, quality_score=1.0,
                        quality_adj_throughput=12.0, config={"n_gpu": 17})
        on_frontier = f.add(r)
        assert on_frontier is True
        assert f.best().tok_s == 12.0
        assert len(f.points) == 1

    def test_dominated_point_rejected(self):
        f = ParetoFrontier()
        good = EvalResult(tok_s=15.0, perplexity=5.0, quality_score=1.0,
                           quality_adj_throughput=15.0, config={})
        bad = EvalResult(tok_s=10.0, perplexity=6.0, quality_score=0.8,
                          quality_adj_throughput=8.0, config={})
        f.add(good)
        on_frontier = f.add(bad)
        assert on_frontier is False
        assert len(f.points) == 1

    def test_new_dominant_removes_old(self):
        f = ParetoFrontier()
        old = EvalResult(tok_s=10.0, perplexity=6.0, quality_score=0.8,
                          quality_adj_throughput=8.0, config={})
        new = EvalResult(tok_s=15.0, perplexity=5.0, quality_score=1.0,
                          quality_adj_throughput=15.0, config={})
        f.add(old)
        assert len(f.points) == 1
        f.add(new)
        assert len(f.points) == 1
        assert f.best().tok_s == 15.0

    def test_pareto_tradeoff_keeps_both(self):
        f = ParetoFrontier()
        # Fast but lower quality
        fast = EvalResult(tok_s=20.0, perplexity=8.0, quality_score=0.7,
                           quality_adj_throughput=14.0, config={})
        # Slow but higher quality
        quality = EvalResult(tok_s=10.0, perplexity=4.0, quality_score=1.2,
                              quality_adj_throughput=12.0, config={})
        f.add(fast)
        f.add(quality)
        assert len(f.points) == 2

    def test_best_at_quality(self):
        f = ParetoFrontier()
        fast = EvalResult(tok_s=20.0, perplexity=8.0, quality_score=0.7,
                           quality_adj_throughput=14.0, config={})
        quality = EvalResult(tok_s=10.0, perplexity=4.0, quality_score=1.2,
                              quality_adj_throughput=12.0, config={})
        f.add(fast)
        f.add(quality)

        # At quality >= 1.0, only the slower one qualifies
        best = f.best_at_quality(1.0)
        assert best is not None
        assert best.tok_s == 10.0

        # At quality >= 0.5, both qualify, pick faster
        best = f.best_at_quality(0.5)
        assert best is not None
        assert best.tok_s == 20.0

    def test_from_results(self):
        results = [
            EvalResult(tok_s=15.0, quality_score=1.0, quality_adj_throughput=15.0,
                        config={}, status="ok"),
            EvalResult(tok_s=5.0, quality_score=0.5, quality_adj_throughput=2.5,
                        config={}, status="ok"),
            EvalResult(tok_s=0.0, config={}, status="oom"),
        ]
        f = ParetoFrontier.from_results(results)
        assert len(f.points) >= 1
        assert f.best().tok_s == 15.0

    def test_summary_format(self):
        f = ParetoFrontier()
        r = EvalResult(tok_s=12.3, perplexity=5.5, quality_score=0.95,
                        quality_adj_throughput=11.685, config={})
        f.add(r)
        s = f.summary()
        assert "12.30" in s
        assert "0.950" in s


class TestResultsTracker:
    def test_record_and_count(self):
        tracker = ResultsTracker()
        r1 = EvalResult(tok_s=10.0, config={"n_gpu": 15}, status="ok")
        r2 = EvalResult(tok_s=0.0, config={"n_gpu": 20}, status="oom")
        id1 = tracker.record(r1)
        id2 = tracker.record(r2)
        assert id1 == 1
        assert id2 == 2
        assert len(tracker.results) == 2

    def test_tsv_output(self):
        path = tempfile.mktemp(suffix=".tsv")
        # Ensure file does NOT exist yet so tracker creates it with header
        if os.path.exists(path):
            os.unlink(path)

        try:
            tracker = ResultsTracker(path)
            r = EvalResult(
                tok_s=12.331,
                perplexity=5.5,
                quality_score=0.95,
                quality_adj_throughput=11.714,
                vram_mb=7751,
                config={
                    "n_gpu": 17, "n_batch": 252, "n_ubatch": 94,
                    "n_threads": 4, "type_k": 8, "type_v": 8, "flash_attn": 1,
                },
                status="ok",
                wall_time_s=25.3,
                notes="test run",
            )
            tracker.record(r)

            with open(path) as f:
                content = f.read()

            assert "exp_id" in content  # header
            assert "12.331" in content
            assert "7751" in content
        finally:
            os.unlink(path)

    def test_to_tsv_string(self):
        tracker = ResultsTracker()
        r = EvalResult(tok_s=10.0, config={"n_gpu": 15}, status="ok")
        tracker.record(r)
        tsv = tracker.to_tsv()
        assert "exp_id" in tsv
        assert "10.000" in tsv

    def test_summary(self):
        tracker = ResultsTracker()
        tracker.record(EvalResult(tok_s=10.0, quality_score=1.0,
                                   quality_adj_throughput=10.0, config={}, status="ok"))
        tracker.record(EvalResult(config={}, status="oom"))
        s = tracker.summary()
        assert "2 total" in s
        assert "1 ok" in s
        assert "1 failed" in s

    def test_load_existing_tsv(self):
        """Test loading from an existing TSV file."""
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
            f.write("\t".join(TSV_COLUMNS) + "\n")
            f.write("1\t12.331\t5.5\t0.95\t11.714\t7751\t17\t252\t94\t4\t8\t8\t1\tok\t25.3\ttest\n")
            path = f.name

        try:
            tracker = ResultsTracker(path)
            assert len(tracker.results) == 1
            assert tracker.results[0].tok_s == pytest.approx(12.331)
            assert tracker._next_id == 2
        finally:
            os.unlink(path)


class TestLoadLegacyTsv:
    def test_load_phase10_format(self):
        """Test loading legacy TSV format from qwen35-moe-offload."""
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
            f.write("exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n")
            f.write("1\t0.000\t0\t512\tq8_0\tq8_0\tFalse\t16\t252\t94\tp10_noflash\toom\tFailed\n")
            f.write("2\t12.331\t7751\t512\tq8_0\tq8_0\tTrue\t17\t252\t94\tp10_ngpu17\tok\tgood run\n")
            f.write("3\t12.066\t7751\t512\tf16\tf16\tTrue\t17\t252\t94\tp10_f16kv\tok\ttest\n")
            path = f.name

        try:
            results = load_legacy_tsv(path)
            assert len(results) == 2  # OOM filtered out
            assert results[0].tok_s == pytest.approx(12.331)
            assert results[0].config["n_gpu"] == 17
            assert results[0].config["type_k"] == 8  # q8_0
            assert results[0].config["flash_attn"] == 1

            assert results[1].config["type_k"] == 1  # f16
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        results = load_legacy_tsv("/nonexistent/file.tsv")
        assert results == []
