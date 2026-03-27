"""Tests for evaluator module."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from autoinfer.evaluator import (
    EvalConfig,
    EvalResult,
    auto_detect_config,
    compute_quality_score,
    evaluate,
    _run_bench,
    _run_perplexity,
)


class TestEvalResult:
    def test_default_values(self):
        r = EvalResult()
        assert r.tok_s == 0.0
        assert r.perplexity == float("inf")
        assert r.status == "pending"

    def test_failed_factory(self):
        r = EvalResult.failed({"n_gpu": 17}, "oom", "out of memory")
        assert r.status == "oom"
        assert r.config == {"n_gpu": 17}
        assert r.notes == "out of memory"
        assert r.tok_s == 0.0

    def test_to_dict(self):
        r = EvalResult(tok_s=12.0, config={"n_gpu": 17}, status="ok")
        d = r.to_dict()
        assert d["tok_s"] == 12.0
        assert d["config"]["n_gpu"] == 17


class TestComputeQualityScore:
    def test_same_as_baseline(self):
        score = compute_quality_score(5.5, 5.5)
        assert score == pytest.approx(1.0)

    def test_worse_than_baseline(self):
        score = compute_quality_score(11.0, 5.5)
        assert score == pytest.approx(0.5)

    def test_better_than_baseline(self):
        score = compute_quality_score(2.75, 5.5)
        assert score == pytest.approx(2.0)  # clamped

    def test_infinite_perplexity(self):
        score = compute_quality_score(float("inf"), 5.5)
        assert score == 0.0

    def test_no_baseline(self):
        score = compute_quality_score(5.5, 0.0)
        assert score == 1.0

    def test_zero_perplexity(self):
        score = compute_quality_score(0.0, 5.5)
        assert score == 0.0


class TestRunBench:
    def test_parse_tok_s_internal(self):
        bench_output = (
            "Loading model...\n"
            "wall=12.214tok/s internal=12.331tok/s, gen=264, t_eval=21327.8ms\n"
            "VRAM: 7751 MB\n"
        )
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(
                returncode=0,
                stdout=bench_output,
                stderr="",
            )
            config = EvalConfig(bench_binary="/fake/bench")
            tok_s, vram, notes = _run_bench("/model.gguf", {"n_gpu": 17}, config)
            assert tok_s == pytest.approx(12.331)
            assert vram == 7751

    def test_parse_tokens_per_second(self):
        bench_output = "eval time = 21327.80 ms / 264 tokens (80.79 ms per token, 12.38 tokens per second)"
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(
                returncode=0,
                stdout=bench_output,
                stderr="",
            )
            config = EvalConfig(bench_binary="/fake/bench")
            tok_s, vram, notes = _run_bench("/model.gguf", {}, config)
            assert tok_s == pytest.approx(12.38)

    def test_oom_detection(self):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(
                returncode=1,
                stdout="Failed to create context (Out of Memory)\n",
                stderr="",
            )
            config = EvalConfig(bench_binary="/fake/bench")
            tok_s, vram, notes = _run_bench("/model.gguf", {}, config)
            assert tok_s == 0.0
            assert "oom" in notes

    def test_no_binary(self):
        config = EvalConfig(bench_binary="")
        tok_s, vram, notes = _run_bench("/model.gguf", {}, config)
        assert tok_s == 0.0
        assert "no bench binary" in notes

    def test_timeout(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("bench", 120)):
            config = EvalConfig(bench_binary="/fake/bench")
            tok_s, vram, notes = _run_bench("/model.gguf", {}, config)
            assert tok_s == 0.0
            assert "timeout" in notes


class TestRunPerplexity:
    def test_parse_ppl_output(self):
        ppl_output = "Final estimate: PPL = 6.1234 +/- 0.1234"
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(
                returncode=0,
                stdout=ppl_output,
                stderr="",
            )
            config = EvalConfig(
                perplexity_binary="/fake/llama-perplexity",
                corpus_path="/fake/corpus.txt",
            )
            with patch("os.path.isfile", return_value=True):
                ppl, notes = _run_perplexity("/model.gguf", {}, config)
                assert ppl == pytest.approx(6.1234)

    def test_no_binary(self):
        config = EvalConfig(perplexity_binary="")
        ppl, notes = _run_perplexity("/model.gguf", {}, config)
        assert ppl == float("inf")
        assert "no perplexity binary" in notes

    def test_no_corpus(self):
        config = EvalConfig(
            perplexity_binary="/fake/llama-perplexity",
            corpus_path="",
        )
        ppl, notes = _run_perplexity("/model.gguf", {}, config)
        assert ppl == float("inf")


class TestEvaluate:
    @patch("autoinfer.evaluator._run_bench")
    @patch("autoinfer.evaluator._run_perplexity")
    def test_full_evaluation(self, mock_ppl, mock_bench):
        mock_bench.return_value = (12.331, 7751, "tok/s=12.331")
        mock_ppl.return_value = (5.5, "ppl=5.5")

        config = {"n_gpu": 17, "n_batch": 252}
        eval_config = EvalConfig(
            bench_binary="/fake/bench",
            perplexity_binary="/fake/ppl",
            corpus_path="/fake/corpus.txt",
            baseline_perplexity=5.5,
        )

        result = evaluate("/model.gguf", config, eval_config)
        assert result.status == "ok"
        assert result.tok_s == pytest.approx(12.331)
        assert result.perplexity == pytest.approx(5.5)
        assert result.quality_score == pytest.approx(1.0)
        assert result.quality_adj_throughput == pytest.approx(12.331)

    @patch("autoinfer.evaluator._run_bench")
    def test_oom_handling(self, mock_bench):
        mock_bench.return_value = (0.0, 0, "oom")
        config = {"n_gpu": 20}
        eval_config = EvalConfig(bench_binary="/fake/bench")

        result = evaluate("/model.gguf", config, eval_config, measure_perplexity=False)
        assert result.status == "oom"

    @patch("autoinfer.evaluator._run_bench")
    def test_skip_perplexity(self, mock_bench):
        mock_bench.return_value = (10.0, 5000, "ok")
        config = {"n_gpu": 15}
        eval_config = EvalConfig(bench_binary="/fake/bench")

        result = evaluate("/model.gguf", config, eval_config, measure_perplexity=False)
        assert result.status == "ok"
        assert result.perplexity == float("inf")
        assert result.quality_score == 1.0  # no baseline → neutral


class TestAutoDetectConfig:
    def test_returns_eval_config(self):
        config = auto_detect_config()
        assert isinstance(config, EvalConfig)
