"""Tests for autoinfer.loop."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from autoinfer.loop import (
    LOOP_TSV_COLUMNS,
    SEARCH_SPACE,
    LegacyResult,
    LoopConfig,
    _append_tsv,
    _init_tsv,
    _suggest_params,
    _warm_start_study,
    load_all_legacy,
    run_loop,
)
from autoinfer.executor import ExperimentResult


class TestSearchSpace:
    """Validate search space definition."""

    def test_all_params_defined(self):
        expected = {"n_gpu", "n_ctx", "batch", "ubatch", "n_threads", "n_gen", "kv_type", "kv_type_v", "flash_attn"}
        assert set(SEARCH_SPACE.keys()) == expected

    def test_n_gpu_range(self):
        assert SEARCH_SPACE["n_gpu"]["low"] == 0
        assert SEARCH_SPACE["n_gpu"]["high"] == 32

    def test_kv_type_choices(self):
        assert "q8_0" in SEARCH_SPACE["kv_type"]["choices"]
        assert "q4_0" in SEARCH_SPACE["kv_type"]["choices"]
        assert "iq4_nl" in SEARCH_SPACE["kv_type"]["choices"]


class TestSuggestParams:
    """Test parameter suggestion with mock Optuna trial."""

    def test_suggest_produces_valid_params(self):
        trial = MagicMock()
        trial.suggest_int.side_effect = lambda name, low, high: (low + high) // 2
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        params = _suggest_params(trial)

        assert "n_gpu" in params
        assert "batch" in params
        assert "ubatch" in params
        assert params["ubatch"] <= params["batch"]

    def test_ubatch_constraint(self):
        """ubatch must be ≤ batch."""
        trial = MagicMock()
        # Force ubatch > batch
        def suggest_int(name, low, high):
            if name == "batch":
                return 64
            if name == "ubatch":
                return 256  # > batch
            return (low + high) // 2

        trial.suggest_int.side_effect = suggest_int
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        params = _suggest_params(trial)
        assert params["ubatch"] <= params["batch"]


class TestLegacyLoading:
    """Test loading legacy TSV files."""

    def test_load_phase10_format(self, tmp_path):
        tsv_content = (
            "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n"
            "1\t0.000\t0\t512\tq8_0\tq8_0\tFalse\t16\t252\t94\tp10_noflash\toom\tFailed\n"
            "2\t12.331\t7751\t512\tq8_0\tq8_0\tTrue\t17\t252\t94\tp10_ngpu17\tok\tok\n"
            "3\t11.475\t7691\t512\tq8_0\tq8_0\tTrue\t17\t252\t94\tp10_base\tok\tok\n"
        )
        path = tmp_path / "results_phase10.tsv"
        path.write_text(tsv_content)

        results = load_all_legacy([str(path)])
        # Only 2 ok results (first is oom)
        assert len(results) == 2
        assert results[0].tok_s == pytest.approx(12.331)
        assert results[0].params["n_gpu"] == 17
        assert results[0].params["flash_attn"] is True

    def test_load_phase4_format(self, tmp_path):
        tsv_content = (
            "exp\ttok_per_sec\tvram_peak_mb\tn_ctx\ttype_k\ttype_v\tflash_attn\tn_gpu_layers\tn_batch\tn_ubatch\tn_threads\tstatus\tdescription\n"
            "1\t8.257\t3601\t512\tf16\tf16\tFalse\t5\t512\t512\t10\tbaseline\tPhase 4\n"
            "2\t8.518\t3609\t512\tq8_0\tq8_0\tTrue\t5\t512\t512\t10\tkeep\tq8_0 KV\n"
        )
        path = tmp_path / "results_phase4.tsv"
        path.write_text(tsv_content)

        results = load_all_legacy([str(path)])
        assert len(results) == 2
        assert results[0].tok_s == pytest.approx(8.257)
        assert results[0].params["kv_type"] == "f16"
        assert results[0].params["n_gpu"] == 5

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "results_phase5_clean.tsv"
        path.write_text("")
        results = load_all_legacy([str(path)])
        assert len(results) == 0

    def test_load_nonexistent(self):
        results = load_all_legacy(["/nonexistent/file.tsv"])
        assert len(results) == 0

    def test_load_multiple_files(self, tmp_path):
        tsv1 = (
            "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n"
            "1\t12.0\t7000\t512\tq8_0\tq8_0\tTrue\t17\t252\t94\tp10\tok\tok\n"
        )
        tsv2 = (
            "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n"
            "1\t15.0\t8000\t512\tq8_0\tq8_0\tTrue\t20\t128\t64\tp11\tok\tok\n"
        )
        p1 = tmp_path / "results_phase10.tsv"
        p2 = tmp_path / "results_phase11.tsv"
        p1.write_text(tsv1)
        p2.write_text(tsv2)

        results = load_all_legacy([str(p1), str(p2)])
        assert len(results) == 2


class TestWarmStartStudy:
    """Test Optuna study warm-starting."""

    def test_enqueue_top_results(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        legacy = [
            LegacyResult(tok_s=10.0, params={"n_gpu": 16, "batch": 128, "ubatch": 64, "kv_type": "q8_0", "flash_attn": True, "n_ctx": 512, "n_threads": 11, "n_gen": 264}),
            LegacyResult(tok_s=15.0, params={"n_gpu": 20, "batch": 252, "ubatch": 94, "kv_type": "q8_0", "flash_attn": True, "n_ctx": 512, "n_threads": 11, "n_gen": 264}),
            LegacyResult(tok_s=12.0, params={"n_gpu": 18, "batch": 200, "ubatch": 100, "kv_type": "q4_0", "flash_attn": False, "n_ctx": 1024, "n_threads": 8, "n_gen": 200}),
        ]
        enqueued = _warm_start_study(study, legacy, max_seeds=3)
        assert enqueued >= 2  # At least 2 should be valid

    def test_empty_legacy(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        enqueued = _warm_start_study(study, [])
        assert enqueued == 0


class TestTsvOutput:
    """Test TSV file operations."""

    def test_init_tsv(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        _init_tsv(path)
        assert os.path.isfile(path)
        with open(path) as f:
            header = f.readline().strip()
        assert header == "\t".join(LOOP_TSV_COLUMNS)

    def test_init_tsv_no_overwrite(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        with open(path, "w") as f:
            f.write("existing content\n")
        _init_tsv(path)
        with open(path) as f:
            content = f.read()
        assert content == "existing content\n"

    def test_append_tsv(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        _init_tsv(path)

        result = ExperimentResult(
            tok_s=12.5, vram_mb=7500, status="ok", wall_time_s=5.2, notes="test"
        )
        params = {"n_gpu": 17, "n_ctx": 512, "batch": 252, "ubatch": 94, "kv_type": "q8_0", "flash_attn": True, "n_threads": 11, "n_gen": 264}
        _append_tsv(path, 1, params, result)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 row
        assert "12.500" in lines[1]
        assert "ok" in lines[1]


class TestRunLoop:
    """Test the main loop with mocked executor."""

    @patch("autoinfer.loop.run_experiment")
    @patch("autoinfer.loop.report_new_best")
    @patch("autoinfer.loop.report_completion")
    def test_basic_loop(self, mock_completion, mock_new_best, mock_run, tmp_path):
        """Run 3 experiments, verify loop mechanics."""
        # Mock run_experiment to return varying results
        results = [
            ExperimentResult(tok_s=10.0, vram_mb=7000, status="ok", wall_time_s=5.0, notes="ok"),
            ExperimentResult(tok_s=12.0, vram_mb=7500, status="ok", wall_time_s=6.0, notes="ok"),
            ExperimentResult(tok_s=-1.0, status="oom", notes="OOM"),
        ]
        mock_run.side_effect = results

        output = str(tmp_path / "results.tsv")
        config = LoopConfig(
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
            warmup_paths=[],
            output_path=output,
            max_experiments=3,
        )

        summary = run_loop(config)
        assert summary["total_experiments"] == 3
        # best_tok_s is now quality-adjusted: tok/s × log2(ctx/512 + 1)
        # The optimizer uses quality_score, not raw tok/s
        assert summary["best_tok_s"] > 0
        assert summary["total_experiments"] == 3
        assert summary["failures"] == 1
        assert mock_completion.called

    @patch("autoinfer.loop.run_experiment")
    @patch("autoinfer.loop.report_new_best")
    @patch("autoinfer.loop.report_completion")
    def test_warm_start_affects_best(self, mock_completion, mock_new_best, mock_run, tmp_path):
        """Warm start from legacy should set initial best."""
        # Create a legacy file with a known best
        legacy_tsv = (
            "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n"
            "1\t20.0\t8000\t512\tq8_0\tq8_0\tTrue\t22\t252\t94\tlegacy\tok\tok\n"
        )
        legacy_path = tmp_path / "results_phase10.tsv"
        legacy_path.write_text(legacy_tsv)

        # New experiment below historical best — should NOT trigger new_best
        mock_run.return_value = ExperimentResult(
            tok_s=15.0, vram_mb=7000, status="ok", wall_time_s=5.0, notes="ok"
        )

        config = LoopConfig(
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
            warmup_paths=[str(legacy_path)],
            output_path=str(tmp_path / "results.tsv"),
            max_experiments=1,
        )

        summary = run_loop(config)
        # Best should still be 20.0 from legacy
        assert summary["best_tok_s"] == pytest.approx(20.0)
        assert summary["new_bests"] == 0

    @patch("autoinfer.loop.run_experiment")
    @patch("autoinfer.loop.report_new_best")
    @patch("autoinfer.loop.report_completion")
    def test_all_failures(self, mock_completion, mock_new_best, mock_run, tmp_path):
        """Handle all experiments failing gracefully."""
        mock_run.return_value = ExperimentResult(
            tok_s=-1.0, status="oom", notes="OOM"
        )

        config = LoopConfig(
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
            warmup_paths=[],
            output_path=str(tmp_path / "results.tsv"),
            max_experiments=5,
        )

        summary = run_loop(config)
        assert summary["failures"] == 5
        assert summary["best_tok_s"] == pytest.approx(0.0)


class TestReporter:
    """Test reporter module."""

    def test_report_progress_no_crash(self, capsys):
        from autoinfer.reporter import report_progress
        report_progress(10, 10, 12.5, {"n_gpu": 17}, 11.2, 2)
        output = capsys.readouterr().out
        assert "12.500" in output
        assert "10 experiments" in output

    def test_report_new_best_no_crash(self, capsys):
        from autoinfer.reporter import report_new_best
        with patch("autoinfer.reporter._send_to_session"):
            report_new_best(5, 15.0, 12.0, {"n_gpu": 20})
        output = capsys.readouterr().out
        assert "NEW BEST" in output
        assert "15.000" in output
