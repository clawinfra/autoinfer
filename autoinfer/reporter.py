"""Progress reporter for the autonomous research loop.

Reports milestones (new bests, progress intervals) to stdout and
optionally to the main agent session via OpenClaw sessions API.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger("autoinfer.reporter")


def _format_params(params: dict) -> str:
    """Format params dict into a compact string."""
    parts = []
    for k in ["n_gpu", "n_ctx", "batch", "ubatch", "kv_type", "flash_attn", "n_threads", "n_gen"]:
        if k in params:
            v = params[k]
            if k == "flash_attn":
                v = "✓" if v else "✗"
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def report_progress(
    experiment_num: int,
    total_run: int,
    best_tok_s: float,
    best_params: dict,
    recent_avg: float,
    recent_failures: int,
) -> None:
    """Report periodic progress to stdout."""
    print(
        f"\n{'='*60}\n"
        f"[AutoInfer] Progress: {experiment_num} experiments run\n"
        f"  Best so far:  {best_tok_s:.3f} tok/s\n"
        f"  Best config:  {_format_params(best_params)}\n"
        f"  Last 10 avg:  {recent_avg:.3f} tok/s\n"
        f"  Failures:     {recent_failures}/{total_run} ({100*recent_failures/max(total_run,1):.0f}%)\n"
        f"{'='*60}"
    )


def report_new_best(
    experiment_num: int,
    tok_s: float,
    prev_best: float,
    params: dict,
    session_key: str = "agent:main:main",
) -> None:
    """Report a new all-time best to stdout and optionally to main session."""
    improvement = tok_s - prev_best
    pct = (improvement / max(prev_best, 0.001)) * 100

    msg = (
        f"🏆 [AutoInfer] NEW BEST: {tok_s:.3f} tok/s "
        f"(+{improvement:.3f}, +{pct:.1f}%) at experiment #{experiment_num}\n"
        f"   Config: {_format_params(params)}"
    )
    print(msg)

    # Try to send to main session
    _send_to_session(session_key, msg)


def report_completion(
    total_experiments: int,
    best_tok_s: float,
    best_params: dict,
    new_bests_found: int,
    duration_s: float,
    output_path: Optional[str] = None,
) -> None:
    """Report loop completion."""
    hours = duration_s / 3600
    mins = duration_s / 60

    msg = (
        f"\n{'='*60}\n"
        f"🏁 [AutoInfer] Loop complete\n"
        f"  Total experiments: {total_experiments}\n"
        f"  Duration: {mins:.1f}min ({hours:.2f}h)\n"
        f"  Best: {best_tok_s:.3f} tok/s\n"
        f"  Config: {_format_params(best_params)}\n"
        f"  New bests found: {new_bests_found}\n"
    )
    if output_path:
        msg += f"  Results file: {output_path}\n"
    msg += f"{'='*60}"
    print(msg)

    _send_to_session("agent:main:main", msg)


def _send_to_session(session_key: str, message: str) -> None:
    """Try to send a message to an OpenClaw session. Fails silently."""
    try:
        cmd = ["openclaw", "sessions", "send", "--to", session_key, "--message", message]
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        logger.debug(f"Could not send to session {session_key} (non-fatal)")
