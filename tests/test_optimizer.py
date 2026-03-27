"""Tests for Bayesian optimizer."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from autoinfer.evaluator import EvalConfig, EvalResult
from autoinfer.optimizer import (
    OptimizeResult,
    optimize,
    optimize_from_existing,
)
from autoinfer.profiler import HardwareProfile


def _make_hardware() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="RTX 3090",
        vram_gb=24.0,
        total_vram_gb=24.0,
        ram_gb=16.0,
        cpu_cores=6,
        platform="linux",
        gpu_count=1,
    )


def _make_eval_config() -> EvalConfig:
    return EvalConfig(
        bench_binary="/fake/bench",
        timeout_s=10,
    )


class TestOptimizeResult:
    def test_summary(self):
        r = OptimizeResult(
            best_config={"n_gpu": 17, "n_batch": 252},
            best_quality_adj=12.0,
            best_tok_s=12.331,
            best_quality_score=0.97,
            n_trials=10,
            n_ok=8,
            n_failed=2,
        )
        s = r.summary()
        assert "10 trials" in s
        assert "8 ok" in s
        assert "12.33" in s


class TestOptimize:
    @patch("autoinfer.optimizer.evaluate")
    def test_basic_optimization(self, mock_evaluate):
        """Test that optimizer runs and converges with mock evaluator."""
        call_count = [0]

        def mock_eval(model_path, config, eval_config, measure_perplexity):
            call_count[0] += 1
            # Simulate: more GPU layers = faster, but cap at some point
            n_gpu = config.get("n_gpu", 0)
            tok_s = 5.0 + n_gpu * 0.5
            if n_gpu > 20:
                return EvalResult.failed(config, "oom")
            return EvalResult(
                tok_s=tok_s,
                perplexity=5.5,
                quality_score=1.0,
                quality_adj_throughput=tok_s,
                vram_mb=n_gpu * 400,
                config=config,
                status="ok",
            )

        mock_evaluate.side_effect = mock_eval

        with patch("autoinfer.params.os.path.getsize", return_value=10 * 1024**3):
            result = optimize(
                model_path="/fake/model.gguf",
                hardware=_make_hardware(),
                eval_config=_make_eval_config(),
                n_trials=15,
                seed=42,
                measure_perplexity=False,
            )

            assert result.n_trials == 15
            assert result.n_ok > 0
            assert result.best_quality_adj > 0
            assert call_count[0] == 15

    @patch("autoinfer.optimizer.evaluate")
    def test_quality_threshold(self, mock_evaluate):
        """Test that low-quality results are penalized."""
        def mock_eval(model_path, config, eval_config, measure_perplexity):
            n_gpu = config.get("n_gpu", 0)
            # High GPU = fast but bad quality
            tok_s = 5.0 + n_gpu * 1.0
            quality = max(0.5, 1.0 - n_gpu * 0.03)
            return EvalResult(
                tok_s=tok_s,
                perplexity=5.5 / quality,
                quality_score=quality,
                quality_adj_throughput=tok_s * quality,
                config=config,
                status="ok",
            )

        mock_evaluate.side_effect = mock_eval

        with patch("autoinfer.params.os.path.getsize", return_value=5 * 1024**3):
            result = optimize(
                model_path="/fake/model.gguf",
                hardware=_make_hardware(),
                eval_config=_make_eval_config(),
                target_quality=0.9,
                n_trials=10,
                seed=42,
                measure_perplexity=False,
            )

            assert result.n_trials == 10
            # Best should balance speed and quality
            assert result.best_quality_adj > 0


class TestOptimizeFromExisting:
    def test_analyze_legacy_data(self):
        """Test Pareto frontier extraction from legacy TSV."""
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
            f.write("exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n")
            f.write("1\t12.331\t7751\t512\tq8_0\tq8_0\tTrue\t17\t252\t94\tbest\tok\tgood\n")
            f.write("2\t10.500\t6000\t512\tf16\tf16\tTrue\t15\t256\t128\talt\tok\talt\n")
            f.write("3\t8.200\t5000\t512\tq4_0\tq4_0\tFalse\t12\t128\t64\tslow\tok\tslow\n")
            path = f.name

        try:
            result = optimize_from_existing([path])
            assert result.n_trials == 3
            assert result.best_tok_s == pytest.approx(12.331)
            assert result.frontier is not None
            assert len(result.frontier.points) >= 1
        finally:
            import os
            os.unlink(path)

    def test_empty_data(self):
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
            f.write("exp_id\ttok_s\tvram_mb\tstatus\n")
            path = f.name

        try:
            result = optimize_from_existing([path])
            assert result.n_trials == 0
        finally:
            import os
            os.unlink(path)
