"""Hardware profiler — detect GPU, VRAM, RAM, CPU cores, storage speed."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    gpu_name: str = "none"
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    cpu_cores: int = 1
    nvme_read_mbps: float = 0.0
    platform: str = "unknown"
    gpu_count: int = 0
    total_vram_gb: float = 0.0
    gpu_details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary for logging."""
        gpu_info = f"{self.gpu_name} ({self.vram_gb:.1f}GB)"
        if self.gpu_count > 1:
            gpu_info = f"{self.gpu_count}x GPUs ({self.total_vram_gb:.1f}GB total)"
        return (
            f"{self.platform} | {gpu_info} | "
            f"RAM {self.ram_gb:.1f}GB | {self.cpu_cores} cores | "
            f"Storage {self.nvme_read_mbps:.0f} MB/s"
        )


def _run(cmd: list[str], timeout: int = 10) -> Optional[str]:
    """Run command, return stdout or None on failure."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _detect_gpu() -> tuple[str, float, int, float, list[dict]]:
    """Detect GPU via nvidia-smi. Returns (name, vram_gb, count, total_vram, details)."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return "none", 0.0, 0, 0.0, []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            try:
                vram_mb = float(parts[1])
            except ValueError:
                vram_mb = 0.0
            gpus.append({"name": name, "vram_gb": vram_mb / 1024.0})

    if not gpus:
        return "none", 0.0, 0, 0.0, []

    total_vram = sum(g["vram_gb"] for g in gpus)
    # Primary GPU = largest VRAM
    primary = max(gpus, key=lambda g: g["vram_gb"])
    return primary["name"], primary["vram_gb"], len(gpus), total_vram, gpus


def _detect_ram_linux() -> float:
    """Read total RAM from /proc/meminfo in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(re.findall(r"\d+", line)[0])
                    return kb / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        pass
    return 0.0


def _detect_ram_darwin() -> float:
    """Read total RAM on macOS."""
    out = _run(["sysctl", "-n", "hw.memsize"])
    if out:
        try:
            return int(out) / (1024**3)
        except ValueError:
            pass
    return 0.0


def _detect_ram() -> float:
    """Detect total system RAM in GB."""
    sys = platform.system().lower()
    if sys == "linux":
        return _detect_ram_linux()
    elif sys == "darwin":
        return _detect_ram_darwin()
    # Windows fallback
    out = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory", "/format:value"])
    if out:
        m = re.search(r"TotalPhysicalMemory=(\d+)", out)
        if m:
            return int(m.group(1)) / (1024**3)
    return 0.0


def _detect_cpu_cores() -> int:
    """Detect physical CPU cores (not hyperthreads)."""
    # Try os.cpu_count() as baseline (includes hyperthreads)
    count = os.cpu_count() or 1

    sys = platform.system().lower()
    if sys == "linux":
        out = _run(["lscpu"])
        if out:
            # Look for "Core(s) per socket" × "Socket(s)"
            cores_per = 1
            sockets = 1
            for line in out.splitlines():
                if "Core(s) per socket:" in line:
                    m = re.search(r"(\d+)", line.split(":")[-1])
                    if m:
                        cores_per = int(m.group(1))
                if "Socket(s):" in line:
                    m = re.search(r"(\d+)", line.split(":")[-1])
                    if m:
                        sockets = int(m.group(1))
            physical = cores_per * sockets
            if physical > 0:
                return physical
    elif sys == "darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out:
            try:
                return int(out)
            except ValueError:
                pass

    return count


def _detect_storage_speed() -> float:
    """Estimate storage read speed in MB/s using dd."""
    # Try to read from a tmpfs or root device
    dd_path = shutil.which("dd")
    if not dd_path:
        return 0.0

    # Use /dev/zero as fallback (measures memory speed, not disk)
    # For real disk: try to find the root block device
    test_file = "/tmp/.autoinfer_speed_test"
    try:
        # Write a 256MB test file
        r = subprocess.run(
            [
                "dd",
                "if=/dev/zero",
                f"of={test_file}",
                "bs=1M",
                "count=256",
                "conv=fdatasync",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stderr = r.stderr
        # Parse "XXX MB/s" or "XXX GB/s"
        m = re.search(r"([\d.]+)\s*(GB|MB|kB)/s", stderr)
        if m:
            speed = float(m.group(1))
            unit = m.group(2)
            if unit == "GB":
                speed *= 1024
            elif unit == "kB":
                speed /= 1024
            return speed
    except (subprocess.TimeoutExpired, OSError):
        pass
    finally:
        try:
            os.unlink(test_file)
        except OSError:
            pass

    return 0.0


def profile_hardware(measure_storage: bool = False) -> HardwareProfile:
    """Profile the current hardware.

    Args:
        measure_storage: If True, run a dd benchmark for storage speed.
                        Disabled by default as it takes ~5s.

    Returns:
        HardwareProfile with detected capabilities.
    """
    gpu_name, vram_gb, gpu_count, total_vram, gpu_details = _detect_gpu()
    ram_gb = _detect_ram()
    cpu_cores = _detect_cpu_cores()
    nvme_speed = _detect_storage_speed() if measure_storage else 0.0
    plat = platform.system().lower()

    return HardwareProfile(
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        nvme_read_mbps=nvme_speed,
        platform=plat,
        gpu_count=gpu_count,
        total_vram_gb=total_vram,
        gpu_details=gpu_details,
    )
