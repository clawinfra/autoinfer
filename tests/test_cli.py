"""Tests for CLI interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoinfer.cli import main, cmd_profile, cmd_analyze


class TestMain:
    def test_no_command_prints_help(self, capsys):
        ret = main([])
        assert ret == 1

    def test_profile_command(self):
        with patch("autoinfer.cli.profile_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(
                summary=lambda: "linux | none | RAM 16.0GB | 6 cores | Storage 0 MB/s"
            )
            ret = main(["profile"])
            assert ret == 0

    def test_profile_with_json(self, capsys):
        from autoinfer.profiler import HardwareProfile
        with patch("autoinfer.cli.profile_hardware") as mock_hw:
            mock_hw.return_value = HardwareProfile(
                gpu_name="none", vram_gb=0.0, ram_gb=16.0,
                cpu_cores=6, platform="linux",
            )
            ret = main(["profile", "--json"])
            assert ret == 0
            captured = capsys.readouterr()
            assert "linux" in captured.out

    def test_analyze_missing_file(self, capsys):
        ret = main(["analyze", "/nonexistent/file.tsv"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "No valid results" in captured.out

    def test_optimize_requires_model(self):
        with pytest.raises(SystemExit):
            main(["optimize"])


class TestCmdAnalyze:
    def test_analyze_with_data(self, tmp_path, capsys):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n"
            "1\t12.331\t7751\t512\tq8_0\tq8_0\tTrue\t17\t252\t94\tbest\tok\tgood\n"
            "2\t10.500\t6000\t512\tf16\tf16\tTrue\t15\t256\t128\talt\tok\talt\n"
        )

        import argparse
        args = argparse.Namespace(results=[str(tsv)], target_quality=0.95)
        ret = cmd_analyze(args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "12.33" in captured.out
        assert "Pareto" in captured.out

    def test_analyze_with_quality_filter(self, tmp_path, capsys):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes\n"
            "1\t12.331\t7751\t512\tq8_0\tq8_0\tTrue\t17\t252\t94\tbest\tok\tgood\n"
        )

        import argparse
        args = argparse.Namespace(results=[str(tsv)], target_quality=0.95)
        ret = cmd_analyze(args)
        assert ret == 0
