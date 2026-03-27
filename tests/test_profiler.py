"""Tests for hardware profiler."""

from __future__ import annotations

import platform
from unittest.mock import MagicMock, mock_open, patch

import pytest

from autoinfer.profiler import (
    HardwareProfile,
    _detect_cpu_cores,
    _detect_gpu,
    _detect_ram,
    _detect_ram_linux,
    _detect_storage_speed,
    _run,
    profile_hardware,
)


class TestHardwareProfile:
    def test_default_values(self):
        hp = HardwareProfile()
        assert hp.gpu_name == "none"
        assert hp.vram_gb == 0.0
        assert hp.ram_gb == 0.0
        assert hp.cpu_cores == 1
        assert hp.platform == "unknown"
        assert hp.gpu_count == 0
        assert hp.gpu_details == []

    def test_summary_no_gpu(self):
        hp = HardwareProfile(ram_gb=16.0, cpu_cores=8, platform="linux")
        s = hp.summary()
        assert "none" in s
        assert "16.0GB" in s
        assert "8 cores" in s

    def test_summary_single_gpu(self):
        hp = HardwareProfile(
            gpu_name="RTX 3090",
            vram_gb=24.0,
            ram_gb=32.0,
            cpu_cores=16,
            nvme_read_mbps=3000.0,
            platform="linux",
            gpu_count=1,
            total_vram_gb=24.0,
        )
        s = hp.summary()
        assert "RTX 3090" in s
        assert "24.0GB" in s
        assert "3000 MB/s" in s

    def test_summary_multi_gpu(self):
        hp = HardwareProfile(
            gpu_name="RTX 3090",
            vram_gb=24.0,
            ram_gb=16.0,
            cpu_cores=6,
            platform="linux",
            gpu_count=3,
            total_vram_gb=42.0,
            gpu_details=[
                {"name": "RTX 3090", "vram_gb": 24.0},
                {"name": "RTX 3080", "vram_gb": 10.0},
                {"name": "RTX 2070", "vram_gb": 8.0},
            ],
        )
        s = hp.summary()
        assert "3x GPUs" in s
        assert "42.0GB total" in s


class TestRun:
    def test_run_success(self):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0, stdout="hello\n")
            result = _run(["echo", "hello"])
            assert result == "hello"

    def test_run_failure(self):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=1, stdout="")
            result = _run(["false"])
            assert result is None

    def test_run_file_not_found(self):
        result = _run(["/nonexistent/binary"])
        assert result is None

    def test_run_timeout(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            result = _run(["sleep", "999"])
            assert result is None


class TestDetectGpu:
    def test_nvidia_smi_output(self):
        nvidia_output = "NVIDIA GeForce RTX 3090, 24576\nNVIDIA GeForce RTX 3080, 10240"
        with patch("autoinfer.profiler._run", return_value=nvidia_output):
            name, vram, count, total, details = _detect_gpu()
            assert name == "NVIDIA GeForce RTX 3090"
            assert vram == pytest.approx(24.0, abs=0.1)
            assert count == 2
            assert total == pytest.approx(34.0, abs=0.1)
            assert len(details) == 2

    def test_no_gpu(self):
        with patch("autoinfer.profiler._run", return_value=None):
            name, vram, count, total, details = _detect_gpu()
            assert name == "none"
            assert vram == 0.0
            assert count == 0

    def test_single_gpu(self):
        nvidia_output = "NVIDIA GeForce RTX 4090, 24576"
        with patch("autoinfer.profiler._run", return_value=nvidia_output):
            name, vram, count, total, details = _detect_gpu()
            assert name == "NVIDIA GeForce RTX 4090"
            assert count == 1
            assert len(details) == 1

    def test_malformed_output(self):
        with patch("autoinfer.profiler._run", return_value="garbage data"):
            name, vram, count, total, details = _detect_gpu()
            assert name == "none"
            assert count == 0


class TestDetectRam:
    def test_linux_meminfo(self):
        meminfo = "MemTotal:       16384000 kB\nMemFree:        8000000 kB\n"
        with patch("platform.system", return_value="Linux"):
            with patch("builtins.open", mock_open(read_data=meminfo)):
                ram = _detect_ram_linux()
                assert ram == pytest.approx(15.625, abs=0.1)

    def test_linux_meminfo_missing(self):
        with patch("builtins.open", side_effect=OSError("no file")):
            ram = _detect_ram_linux()
            assert ram == 0.0


class TestDetectCpuCores:
    def test_lscpu_output(self):
        lscpu_output = (
            "Architecture:          x86_64\n"
            "CPU(s):                12\n"
            "Thread(s) per core:    2\n"
            "Core(s) per socket:    6\n"
            "Socket(s):             1\n"
        )
        with patch("platform.system", return_value="Linux"):
            with patch("autoinfer.profiler._run", return_value=lscpu_output):
                cores = _detect_cpu_cores()
                assert cores == 6

    def test_multi_socket(self):
        lscpu_output = (
            "Core(s) per socket:    8\n"
            "Socket(s):             2\n"
        )
        with patch("platform.system", return_value="Linux"):
            with patch("autoinfer.profiler._run", return_value=lscpu_output):
                cores = _detect_cpu_cores()
                assert cores == 16


class TestDetectStorageSpeed:
    def test_dd_output_mb(self):
        dd_stderr = "268435456 bytes (256 MB) copied, 0.5 s, 512 MB/s"
        with patch("shutil.which", return_value="/usr/bin/dd"):
            with patch("subprocess.run") as mock:
                mock.return_value = MagicMock(returncode=0, stderr=dd_stderr)
                with patch("os.unlink"):
                    speed = _detect_storage_speed()
                    assert speed == pytest.approx(512.0)

    def test_dd_output_gb(self):
        dd_stderr = "268435456 bytes copied, 0.1 s, 2.5 GB/s"
        with patch("shutil.which", return_value="/usr/bin/dd"):
            with patch("subprocess.run") as mock:
                mock.return_value = MagicMock(returncode=0, stderr=dd_stderr)
                with patch("os.unlink"):
                    speed = _detect_storage_speed()
                    assert speed == pytest.approx(2560.0)

    def test_no_dd(self):
        with patch("shutil.which", return_value=None):
            speed = _detect_storage_speed()
            assert speed == 0.0


class TestProfileHardware:
    def test_profile_integrates(self):
        """Test that profile_hardware returns a valid HardwareProfile."""
        with patch("autoinfer.profiler._detect_gpu", return_value=("RTX 3090", 24.0, 1, 24.0, [{"name": "RTX 3090", "vram_gb": 24.0}])):
            with patch("autoinfer.profiler._detect_ram", return_value=16.0):
                with patch("autoinfer.profiler._detect_cpu_cores", return_value=6):
                    hp = profile_hardware(measure_storage=False)
                    assert hp.gpu_name == "RTX 3090"
                    assert hp.ram_gb == 16.0
                    assert hp.cpu_cores == 6
                    assert hp.nvme_read_mbps == 0.0  # not measured
