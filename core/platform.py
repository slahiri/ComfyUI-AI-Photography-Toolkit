"""Platform detection, memory management, and state isolation."""

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Generator

import torch


class DeviceType(Enum):
    """Supported device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True)
class SystemInfo:
    """System capabilities."""
    device_type: DeviceType
    device_name: str
    vram_gb: float
    supports_quantization: bool

    @property
    def available_quants(self) -> list[str]:
        """Available quantization options."""
        if self.supports_quantization:
            return ["Q4", "Q8", "F16"]
        return ["F16"]


@lru_cache(maxsize=1)
def get_system_info() -> SystemInfo:
    """Detect system capabilities. Cached."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return SystemInfo(
            device_type=DeviceType.CUDA,
            device_name=props.name,
            vram_gb=round(props.total_memory / (1024**3), 1),
            supports_quantization=True,
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return SystemInfo(
            device_type=DeviceType.MPS,
            device_name="Apple Silicon",
            vram_gb=0,
            supports_quantization=False,
        )

    return SystemInfo(
        device_type=DeviceType.CPU,
        device_name="CPU",
        vram_gb=0,
        supports_quantization=False,
    )


def get_device() -> str:
    """Get torch device string."""
    return get_system_info().device_type.value


# =============================================================================
# Memory Management
# =============================================================================

def cleanup_memory() -> None:
    """Aggressive memory cleanup. Call after model unload."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_vram_usage() -> tuple[float, float]:
    """Get current VRAM usage (used_gb, total_gb). Returns (0,0) if no CUDA."""
    if not torch.cuda.is_available():
        return (0.0, 0.0)

    used = torch.cuda.memory_allocated() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return (round(used, 2), round(total, 2))


# =============================================================================
# State Isolation - Protect other nodes from our changes
# =============================================================================

@dataclass
class TorchState:
    """Snapshot of torch global state."""
    grad_enabled: bool
    inference_mode: bool
    default_dtype: torch.dtype
    cudnn_benchmark: bool
    cudnn_deterministic: bool


def _capture_torch_state() -> TorchState:
    """Capture current torch global state."""
    return TorchState(
        grad_enabled=torch.is_grad_enabled(),
        inference_mode=torch.is_inference_mode_enabled(),
        default_dtype=torch.get_default_dtype(),
        cudnn_benchmark=torch.backends.cudnn.benchmark if torch.cuda.is_available() else False,
        cudnn_deterministic=torch.backends.cudnn.deterministic if torch.cuda.is_available() else False,
    )


def _restore_torch_state(state: TorchState) -> None:
    """Restore torch global state."""
    torch.set_grad_enabled(state.grad_enabled)
    torch.set_default_dtype(state.default_dtype)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = state.cudnn_benchmark
        torch.backends.cudnn.deterministic = state.cudnn_deterministic


@contextmanager
def isolated_execution() -> Generator[None, None, None]:
    """
    Context manager for isolated model execution.

    Captures torch state before execution and restores it after,
    ensuring our VLM inference doesn't affect KSamplers or other nodes.

    Usage:
        with isolated_execution():
            model.generate(image, prompt)
    """
    # Capture state before
    original_state = _capture_torch_state()

    try:
        # Set optimal state for VLM inference
        torch.set_grad_enabled(False)
        yield
    finally:
        # Always restore original state
        _restore_torch_state(original_state)

        # Clean up any leftover tensors
        cleanup_memory()
