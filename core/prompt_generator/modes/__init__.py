"""
Prompt generation modes.

Each mode implements a different strategy for generating prompts:
- Override: Pass-through user prompts
- Florence: VLM-based description
- Quick: Single-shot LLM (no taggers)
- Standard: Tag-driven template LLM (single pass)
- Detailed: Multi-pass LLM with full enhancements (3 passes)
- Extreme: 6-pass LLM with maximum detail (one pass per section)
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
