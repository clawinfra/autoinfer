"""AutoInfer — Universal hardware-adaptive LLM inference optimizer.

Finds the optimal inference configuration for any GGUF model on any hardware
by maximizing quality-adjusted throughput: tok/s × quality_score.
"""

__version__ = "0.1.0"

from autoinfer.profiler import HardwareProfile, profile_hardware
from autoinfer.evaluator import EvalResult, evaluate
from autoinfer.optimizer import optimize
from autoinfer.results import ResultsTracker, ParetoFrontier
from autoinfer.params import ParamSpace, estimate_max_gpu_layers

__all__ = [
    "HardwareProfile",
    "profile_hardware",
    "EvalResult",
    "evaluate",
    "optimize",
    "ResultsTracker",
    "ParetoFrontier",
    "ParamSpace",
    "estimate_max_gpu_layers",
]
