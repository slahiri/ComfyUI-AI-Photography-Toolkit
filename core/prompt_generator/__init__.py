"""
Prompt Generator Module - Stage 1A Tagger Engine.

Implements the "Tri-Stage Captioning Synthesizer" architecture.
Stage 1A: Multi-model tagging with automatic human/non-human routing.

Reuses all existing taggers from core/taggers/.
"""

from .engine import PromptGeneratorEngine
from .formatter import PromptFormatter

__all__ = [
    "PromptGeneratorEngine",
    "PromptFormatter",
]
