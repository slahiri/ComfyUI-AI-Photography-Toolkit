"""
SID_PromptGenerator - Core prompt generation module.

This module implements a multi-mode prompt generation system with:
- Override mode: Pass-through user prompts
- Florence mode: VLM-based description when no LLM provided
- Quick mode: Single-shot LLM call (no taggers)
- Standard mode: Tag-driven template-based LLM call
- Detailed mode: Multi-pass LLM (2-3 passes)
- Extreme mode: Component-by-component LLM (10+ passes)
"""

from .types import (
    AnalysisMode,
    SubjectType,
    NSFWLevel,
    ShotType,
    GeneratorResult,
    Decisions,
)

__all__ = [
    "AnalysisMode",
    "SubjectType",
    "NSFWLevel",
    "ShotType",
    "GeneratorResult",
    "Decisions",
]
