# -*- coding: utf-8 -*-
"""
SID Photography Toolkit v5.0

AI-powered image captioning for ComfyUI.
Clean, modular architecture with plug-and-play models.

Author: Siddhartha Lahiri
License: MIT
"""

__version__ = "5.0.0"

# Import nodes
from .nodes import SID_ZImagePromptGenerator, SID_TaggerConfig

# V1 API registration
NODE_CLASS_MAPPINGS = {
    "SID_ZImagePromptGenerator": SID_ZImagePromptGenerator,
    "SID_TaggerConfig": SID_TaggerConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_ZImagePromptGenerator": "SID Z-Image Prompt Generator",
    "SID_TaggerConfig": "SID Tagger Config",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Startup message
def _print_startup():
    try:
        from .core.platform import get_system_info
        info = get_system_info()
        print(f"\n[SID-Toolkit] v{__version__} loaded")
        print(f"[SID-Toolkit] Device: {info.device_name}")
        print(f"[SID-Toolkit] Quantization: {', '.join(info.available_quants)}\n")
    except Exception as e:
        print(f"\n[SID-Toolkit] v{__version__} loaded (device detection failed: {e})\n")

_print_startup()
