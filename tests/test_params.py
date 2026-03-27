"""Tests for parameter space definitions."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from autoinfer.params import (
    KV_TYPE_MAP,
    KV_TYPE_REVERSE,
    ParamRange,
    ParamSpace,
    estimate_max_gpu_layers,
    estimate_model_layers,
)
from autoinfer.profiler import HardwareProfile


class TestKVTypeMap:
    def test_common_types(self):
        assert KV_TYPE_MAP[1] == "f16"
        assert KV_TYPE_MAP[4] == "q4_0"
        assert KV_TYPE_MAP[8] == "q8_0"

    def test_reverse_map(self):
        assert KV_TYPE_REVERSE["f16"] == 1
        assert KV_TYPE_REVERSE["q4_0"] == 4
        assert KV_TYPE_REVERSE["q8_0"] == 8


class TestParamRange:
    def test_int_validation(self):
        pr = ParamRange(name="n_gpu", type="int", low=0, high=32)
        assert pr.validate_value(0) is True
        assert pr.validate_value(16) is True
        assert pr.validate_value(32) is True
        assert pr.validate_value(-1) is False
        assert pr.validate_value(33) is False

    def test_categorical_validation(self):
        pr = ParamRange(name="n_batch", type="categorical", choices=[128, 256, 512])
        assert pr.validate_value(128) is True
        assert pr.validate_value(256) is True
        assert pr.validate_value(64) is False

    def test_float_validation(self):
        pr = ParamRange(name="lr", type="float", low=0.0, high=1.0)
        assert pr.validate_value(0.5) is True
        assert pr.validate_value(0.0) is True
        assert pr.validate_value(1.1) is False


class TestParamSpace:
    def setup_method(self):
        self.hardware = HardwareProfile(
            gpu_name="RTX 3090",
            vram_gb=24.0,
            total_vram_gb=24.0,
            ram_gb=16.0,
            cpu_cores=6,
            platform="linux",
            gpu_count=1,
        )

    def test_default_creates_all_params(self):
        # Create a small temp file and mock os.path.getsize for the model size
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"x" * 1024)  # small file
            model_path = f.name

        try:
            with patch("autoinfer.params.os.path.getsize", return_value=10 * 1024**3):
                ps = ParamSpace.default(self.hardware, model_path)
                param_names = [p.name for p in ps.params]
                assert "n_gpu" in param_names
                assert "n_batch" in param_names
                assert "n_ubatch" in param_names
                assert "n_threads" in param_names
                assert "type_k" in param_names
                assert "flash_attn" in param_names
                assert len(ps.constraints) > 0
        finally:
            os.unlink(model_path)

    def test_suggest_with_mock_trial(self):
        ps = ParamSpace(params=[
            ParamRange(name="n_gpu", type="int", low=0, high=20),
            ParamRange(name="n_batch", type="categorical", choices=[128, 256]),
            ParamRange(name="flash_attn", type="categorical", choices=[0, 1]),
        ])

        trial = MagicMock()
        trial.suggest_int.return_value = 10
        trial.suggest_categorical.side_effect = [256, 1]

        config = ps.suggest(trial)
        assert config["n_gpu"] == 10
        assert config["n_batch"] == 256
        assert config["flash_attn"] == 1

    def test_constraint_ubatch_le_batch(self):
        ps = ParamSpace()
        config = {"n_batch": 128, "n_ubatch": 256}
        fixed = ps._apply_constraints(config)
        assert fixed["n_ubatch"] == 128

    def test_summary(self):
        ps = ParamSpace(
            params=[
                ParamRange(name="n_gpu", type="int", low=0, high=32),
                ParamRange(name="n_batch", type="categorical", choices=[128, 256]),
            ],
            constraints=["n_ubatch ≤ n_batch"],
        )
        s = ps.summary()
        assert "n_gpu" in s
        assert "n_batch" in s
        assert "n_ubatch ≤ n_batch" in s


class TestEstimateMaxGpuLayers:
    def test_basic_estimation(self):
        hw = HardwareProfile(
            vram_gb=24.0,
            total_vram_gb=24.0,
        )
        with patch("autoinfer.params.os.path.getsize", return_value=10 * 1024**3):
            layers = estimate_max_gpu_layers(hw, "/fake/model.gguf")
            # 23GB usable / (10GB / 64 layers) ≈ 147, capped at 64
            assert layers == 64

    def test_limited_vram(self):
        hw = HardwareProfile(
            vram_gb=4.0,
            total_vram_gb=4.0,
        )
        with patch("autoinfer.params.os.path.getsize", return_value=20 * 1024**3):
            layers = estimate_max_gpu_layers(hw, "/fake/model.gguf")
            # 3GB usable / (20GB / 64) = 3 / 0.3125 ≈ 9
            assert layers < 20
            assert layers > 0

    def test_no_gpu(self):
        hw = HardwareProfile(vram_gb=0.0, total_vram_gb=0.0)
        layers = estimate_max_gpu_layers(hw, "/nonexistent.gguf")
        assert layers == 0

    def test_nonexistent_model(self):
        hw = HardwareProfile(vram_gb=24.0, total_vram_gb=24.0)
        # Should fall back to 10GB default
        layers = estimate_max_gpu_layers(hw, "/nonexistent.gguf")
        assert layers > 0


class TestEstimateModelLayers:
    def test_known_sizes(self):
        assert estimate_model_layers("Qwen-7B-Q4.gguf") == 32
        assert estimate_model_layers("llama-13b-chat.gguf") == 40
        assert estimate_model_layers("Qwen3.5-35B-A3B-Q3_K_M.gguf") == 64
        assert estimate_model_layers("llama-70b.gguf") == 80

    def test_unknown_model(self):
        assert estimate_model_layers("custom_model.gguf") == 64  # default
