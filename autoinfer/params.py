"""Parameter space definitions and hardware-aware constraints."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from autoinfer.profiler import HardwareProfile


# KV cache type mapping: llama.cpp integer codes
KV_TYPE_MAP = {
    0: "f32",
    1: "f16",
    4: "q4_0",
    7: "q5_0",
    8: "q8_0",
    2: "q4_1",
    3: "q5_1",
}

KV_TYPE_REVERSE = {v: k for k, v in KV_TYPE_MAP.items()}


@dataclass
class ParamRange:
    """Definition of a single parameter's search range."""

    name: str
    type: str  # "int", "categorical", "float"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[list] = None
    log: bool = False

    def validate_value(self, value: Any) -> bool:
        """Check if a value is within this parameter's range."""
        if self.type == "categorical":
            return value in (self.choices or [])
        if self.type == "int":
            return (self.low or 0) <= value <= (self.high or float("inf"))
        if self.type == "float":
            return (self.low or 0.0) <= value <= (self.high or float("inf"))
        return True


@dataclass
class ParamSpace:
    """Complete parameter space for optimization."""

    params: list[ParamRange] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)  # human-readable

    @classmethod
    def default(cls, hardware: HardwareProfile, model_path: str) -> "ParamSpace":
        """Create the default parameter space, constrained by hardware."""
        max_gpu = estimate_max_gpu_layers(hardware, model_path)
        max_threads = max(hardware.cpu_cores, 4)

        params = [
            ParamRange(
                name="n_gpu",
                type="int",
                low=0,
                high=max_gpu,
            ),
            ParamRange(
                name="n_batch",
                type="categorical",
                choices=[64, 128, 192, 252, 256, 320, 512],
            ),
            ParamRange(
                name="n_ubatch",
                type="categorical",
                choices=[32, 64, 94, 96, 128, 256],
            ),
            ParamRange(
                name="n_threads",
                type="int",
                low=1,
                high=max_threads,
            ),
            ParamRange(
                name="type_k",
                type="categorical",
                choices=[1, 4, 8],  # f16, q4_0, q8_0
            ),
            ParamRange(
                name="type_v",
                type="categorical",
                choices=[1, 4, 8],  # f16, q4_0, q8_0
            ),
            ParamRange(
                name="flash_attn",
                type="categorical",
                choices=[0, 1],
            ),
        ]

        constraints = [
            f"n_gpu ∈ [0, {max_gpu}] (VRAM-limited)",
            f"n_threads ∈ [1, {max_threads}] (CPU cores)",
            "n_ubatch ≤ n_batch",
            "flash_attn requires type_k ∈ {f16, q8_0}",
        ]

        return cls(params=params, constraints=constraints)

    def suggest(self, trial: Any) -> dict:
        """Suggest a configuration from an Optuna trial."""
        config = {}
        for p in self.params:
            if p.type == "int":
                config[p.name] = trial.suggest_int(
                    p.name, int(p.low or 0), int(p.high or 32)
                )
            elif p.type == "categorical":
                config[p.name] = trial.suggest_categorical(p.name, p.choices)
            elif p.type == "float":
                config[p.name] = trial.suggest_float(
                    p.name, p.low or 0.0, p.high or 1.0, log=p.log
                )
        # Enforce constraints
        config = self._apply_constraints(config)
        return config

    def _apply_constraints(self, config: dict) -> dict:
        """Enforce hard constraints on a configuration."""
        # n_ubatch must be ≤ n_batch
        if config.get("n_ubatch", 0) > config.get("n_batch", 512):
            config["n_ubatch"] = config["n_batch"]

        # flash_attn with q4_0 KV is unsupported in some builds
        # Keep it but the evaluator will handle failures gracefully
        return config

    def summary(self) -> str:
        """Human-readable summary of the parameter space."""
        lines = ["Parameter Space:"]
        for p in self.params:
            if p.type == "categorical":
                lines.append(f"  {p.name}: {p.choices}")
            else:
                lines.append(f"  {p.name}: [{p.low}, {p.high}]")
        lines.append("\nConstraints:")
        for c in self.constraints:
            lines.append(f"  - {c}")
        return "\n".join(lines)


def estimate_max_gpu_layers(
    hardware: HardwareProfile,
    model_path: str,
    headroom_gb: float = 1.0,
    default_n_layers: int = 64,
) -> int:
    """Estimate maximum GPU layers that fit in VRAM.

    Args:
        hardware: Detected hardware profile.
        model_path: Path to the GGUF model file.
        headroom_gb: VRAM to reserve for KV cache + overhead.
        default_n_layers: Assumed number of layers in the model.

    Returns:
        Estimated max layers that fit in VRAM (capped at default_n_layers).
    """
    if hardware.vram_gb <= 0:
        return 0

    try:
        model_size_gb = os.path.getsize(model_path) / (1024**3)
    except OSError:
        model_size_gb = 10.0  # conservative default

    if model_size_gb <= 0:
        return 0

    usable_vram = hardware.total_vram_gb - headroom_gb
    if usable_vram <= 0:
        return 0

    # Each layer ≈ model_size / n_layers
    layer_size_gb = model_size_gb / default_n_layers
    max_layers = int(usable_vram / layer_size_gb)

    return min(max_layers, default_n_layers)


def estimate_model_layers(model_path: str) -> int:
    """Try to guess the number of layers from model filename or metadata.

    Falls back to 64 (common for MoE models like Qwen3.5-35B-A3B).
    """
    basename = os.path.basename(model_path).lower()

    # Common layer counts by model size
    layer_hints = {
        "7b": 32,
        "8b": 32,
        "13b": 40,
        "14b": 40,
        "30b": 60,
        "33b": 60,
        "34b": 60,
        "35b": 64,  # Qwen3.5-35B-A3B
        "65b": 80,
        "70b": 80,
    }

    for hint, layers in layer_hints.items():
        if hint in basename:
            return layers

    return 64  # safe default
