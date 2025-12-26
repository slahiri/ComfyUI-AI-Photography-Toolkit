"""
Centralized logging utility with timestamps for performance analysis.

Usage:
    from ..core.log import log, log_start, log_end

    log("ImageAnalysis", "Starting analysis...")
    start = log_start("ImageAnalysis", "Florence Pass 1")
    # ... do work ...
    log_end("ImageAnalysis", "Florence Pass 1", start)
"""

import time
from datetime import datetime
from typing import Optional

# Global verbose flag
_verbose = True


def set_verbose(verbose: bool) -> None:
    """Enable or disable logging."""
    global _verbose
    _verbose = verbose


def _timestamp() -> str:
    """Get current timestamp in HH:MM:SS.mmm format."""
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def log(module: str, message: str, force: bool = False) -> None:
    """
    Log a message with timestamp.

    Args:
        module: Module name (e.g., "ImageAnalysis", "Synthesis")
        message: Log message
        force: Print even if verbose is False
    """
    if _verbose or force:
        print(f"[{_timestamp()}] [SID-{module}] {message}")


def log_start(module: str, step: str) -> float:
    """
    Log the start of a step and return start time.

    Args:
        module: Module name
        step: Step name (e.g., "Florence Pass 1", "Loading model")

    Returns:
        Start time for use with log_end()
    """
    log(module, f"{step}...")
    return time.perf_counter()


def log_end(module: str, step: str, start_time: float, extra: str = "") -> float:
    """
    Log the end of a step with duration.

    Args:
        module: Module name
        step: Step name
        start_time: Start time from log_start()
        extra: Optional extra info to append

    Returns:
        Duration in seconds
    """
    duration = time.perf_counter() - start_time
    msg = f"{step} complete ({duration:.2f}s)"
    if extra:
        msg += f" - {extra}"
    log(module, msg)
    return duration


def log_error(module: str, message: str) -> None:
    """Log an error (always prints, ignores verbose flag)."""
    print(f"[{_timestamp()}] [SID-{module}] ERROR: {message}")


def log_warn(module: str, message: str) -> None:
    """Log a warning (always prints, ignores verbose flag)."""
    print(f"[{_timestamp()}] [SID-{module}] WARN: {message}")
