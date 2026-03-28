"""Tests for autoinfer.executor."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from autoinfer.executor import (
    ExperimentResult,
    KV_NAME_TO_INT,
    _build_env,
    _parse_tok_s,
    _parse_vram,
    run_experiment,
)


class TestParseTokS:
    """Test tok/s parsing from bench output."""

    def test_internal_pattern(self):
        output = "internal=12.331tok/s"
        assert _parse_tok_s(output) == pytest.approx(12.331)

    def test_wall_pattern(self):
        output = "wall=11.378tok/s"
        assert _parse_tok_s(output) == pytest.approx(11.378)

    def test_tokens_per_second(self):
        output = "eval time = 21327.8 ms / 264 tokens (80.79 ms per token, 12.38 tokens per second)"
        assert _parse_tok_s(output) == pytest.approx(12.38)

    def test_tok_s_colon(self):
        output = "tok/s: 15.123"
        assert _parse_tok_s(output) == pytest.approx(15.123)

    def test_generic_tok_s(self):
        output = "Result: 27.559 tok/s"
        assert _parse_tok_s(output) == pytest.approx(27.559)

    def test_no_match(self):
        assert _parse_tok_s("no useful output here") == -1.0

    def test_empty(self):
        assert _parse_tok_s("") == -1.0

    def test_multiline_prefers_internal(self):
        output = "wall=11.0tok/s internal=12.5tok/s gen=264"
        assert _parse_tok_s(output) == pytest.approx(12.5)

    def test_full_bench_output(self):
        output = (
            "n_gpu=17, batch=252/94, q8_0 KV, n_ctx=512, flash=1 — "
            "wall=11.378tok/s internal=11.475tok/s, gen=264, t_eval=22918.6ms"
        )
        assert _parse_tok_s(output) == pytest.approx(11.475)


class TestParseVram:
    """Test VRAM parsing."""

    def test_vram_mb(self):
        assert _parse_vram("VRAM: 7751 MB") == 7751

    def test_vram_lowercase(self):
        assert _parse_vram("vram=8192") == 8192

    def test_no_vram(self):
        assert _parse_vram("nothing here") == 0


class TestBuildEnv:
    """Test environment building."""

    def test_combines_paths(self):
        env = _build_env("/path/to/llama", "/path/to/cuda")
        assert "/path/to/llama" in env["LD_LIBRARY_PATH"]
        assert "/path/to/cuda" in env["LD_LIBRARY_PATH"]

    def test_empty_paths(self):
        env = _build_env("", "")
        # Should still work without error
        assert "LD_LIBRARY_PATH" in env


class TestKvNameToInt:
    """Test KV type mapping."""

    def test_known_types(self):
        assert KV_NAME_TO_INT["q8_0"] == 8
        assert KV_NAME_TO_INT["q4_0"] == 2
        assert KV_NAME_TO_INT["f16"] == 1
        assert KV_NAME_TO_INT["f32"] == 0


class TestExperimentResult:
    """Test ExperimentResult dataclass."""

    def test_success(self):
        r = ExperimentResult(tok_s=12.0, status="ok")
        assert r.success is True

    def test_failure_oom(self):
        r = ExperimentResult(tok_s=-1.0, status="oom")
        assert r.success is False

    def test_failure_zero_tok_s(self):
        r = ExperimentResult(tok_s=0.0, status="ok")
        assert r.success is False


class TestRunExperiment:
    """Test run_experiment with mocked subprocess."""

    @patch("autoinfer.executor.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="wall=10.5tok/s internal=11.2tok/s, gen=264, VRAM: 7500 MB",
            stderr="",
            returncode=0,
        )
        result = run_experiment(
            params={"n_gpu": 16, "batch": 252, "ubatch": 94},
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
        )
        assert result.success
        assert result.tok_s == pytest.approx(11.2)
        assert result.status == "ok"

    @patch("autoinfer.executor.subprocess.run")
    def test_oom(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="Failed to create context (OOM?)",
            returncode=1,
        )
        result = run_experiment(
            params={"n_gpu": 28},
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
        )
        assert not result.success
        assert result.status == "oom"

    @patch("autoinfer.executor.subprocess.run")
    def test_crash(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="segfault",
            stderr="",
            returncode=139,
        )
        result = run_experiment(
            params={"n_gpu": 16},
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
        )
        assert not result.success
        assert result.status == "crash"

    @patch("autoinfer.executor.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bench", timeout=300)
        result = run_experiment(
            params={"n_gpu": 16},
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
            timeout=300,
        )
        assert not result.success
        assert result.status == "timeout"

    @patch("autoinfer.executor.subprocess.run")
    def test_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("not found")
        result = run_experiment(
            params={"n_gpu": 16},
            bench_binary="/nonexistent/bench",
            model_path="/fake/model.gguf",
        )
        assert not result.success
        assert result.status == "error"
        assert "not found" in result.notes

    @patch("autoinfer.executor.subprocess.run")
    def test_no_tok_s_in_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="model loaded successfully\ndone",
            stderr="",
            returncode=0,
        )
        result = run_experiment(
            params={"n_gpu": 16},
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
        )
        assert not result.success
        assert result.status == "error"

    @patch("autoinfer.executor.subprocess.run")
    def test_kv_type_string(self, mock_run):
        """Verify kv_type string is converted to integer for bench CLI."""
        mock_run.return_value = MagicMock(
            stdout="internal=10.0tok/s",
            stderr="",
            returncode=0,
        )
        run_experiment(
            params={"n_gpu": 16, "kv_type": "q4_0", "batch": 128, "ubatch": 64},
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
        )
        # Check the command that was called
        cmd = mock_run.call_args[0][0]
        # --type-k should be "2" (q4_0 → 2)
        type_k_idx = cmd.index("--type-k") + 1
        assert cmd[type_k_idx] == "2"

    @patch("autoinfer.executor.subprocess.run")
    def test_flash_attn_bool(self, mock_run):
        """Verify flash_attn bool is converted to int."""
        mock_run.return_value = MagicMock(
            stdout="internal=10.0tok/s",
            stderr="",
            returncode=0,
        )
        run_experiment(
            params={"n_gpu": 16, "flash_attn": False, "batch": 128, "ubatch": 64},
            bench_binary="/fake/bench",
            model_path="/fake/model.gguf",
        )
        cmd = mock_run.call_args[0][0]
        flash_idx = cmd.index("--flash-attn") + 1
        assert cmd[flash_idx] == "0"
