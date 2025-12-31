"""
Prompt generation modes.

Each mode implements a different strategy for generating prompts:
- Override: Pass-through user prompts
- Florence: VLM-based description
- Quick: Single-shot LLM (no taggers)
- Standard: Tag-driven template LLM (single pass)
- Detailed: Multi-pass LLM with full enhancements (4 passes: 3 content + 1 optimization)
- Extreme: 7-pass LLM with maximum detail (6 section passes + 1 optimization)
"""

from .base import BaseMode
from .override import OverrideMode
from .florence import FlorenceMode
from .quick import QuickMode
from .standard import StandardMode
from .detailed import DetailedMode
from .extreme import ExtremeMode

__all__ = [
    "BaseMode",
    "OverrideMode",
    "FlorenceMode",
    "QuickMode",
    "StandardMode",
    "DetailedMode",
    "ExtremeMode",
]
