# -*- coding: utf-8 -*-
"""
System Information Gathering

Collects system information for debug evaluation context.
"""

import platform
import sys
from typing import Dict, Any


def get_system_info() -> Dict[str, Any]:
    """
    Gather comprehensive system information.

    Returns:
        Dictionary with system details
    """
    info = {
        "platform": sys.platform,
        "os_version": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
    }

    # PyTorch and CUDA info
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
                "vram_available_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3), 1),
            }

            # Try to get driver version
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info["gpu"]["driver_version"] = result.stdout.strip()
            except:
                pass
        else:
            info["gpu"] = None
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
        info["gpu"] = None

    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        info["ram_total_gb"] = None
        info["ram_available_gb"] = None

    # ComfyUI version (if available)
    try:
        # Try to get ComfyUI version from various sources
        import importlib.metadata
        try:
            info["comfyui_version"] = importlib.metadata.version("comfyui")
        except:
            info["comfyui_version"] = "unknown"
    except:
        info["comfyui_version"] = "unknown"

    return info


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with VRAM info in GB
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                "cached_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
                "available_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3), 2),
            }
    except:
        pass

    return {
        "total_gb": 0,
        "allocated_gb": 0,
        "cached_gb": 0,
        "available_gb": 0,
    }
