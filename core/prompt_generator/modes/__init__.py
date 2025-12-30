"""
Prompt generation modes.

Each mode implements a different strategy for generating prompts:
- Override: Pass-through user prompts
- Florence: VLM-based description
- Quick: Single-shot LLM (no taggers)
- Standard: Tag-driven template LLM (single pass)
- Detailed: Multi-pass LLM with full enhancements (3 passes)
"""

from .base import BaseMode
from .override import OverrideMode
from .florence import FlorenceMode
from .quick import QuickMode
from .standard import StandardMode
from .detailed import DetailedMode

__all__ = [
    "BaseMode",
    "OverrideMode",
    "FlorenceMode",
    "QuickMode",
    "StandardMode",
    "DetailedMode",
]
