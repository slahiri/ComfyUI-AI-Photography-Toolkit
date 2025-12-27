"""Assembler module for generating natural language prompts.

This module handles Phase 4 of the pipeline:
- Rule-based assembly (templates, phrase mappings, grammar rules)
- Converts classified tokens into coherent prompts
"""

from .base import (
    BaseAssembler,
    AssemblerConfig,
    AssembledPrompt,
    CategorySection,
    PromptStyle,
)

from .rules import (
    Gender,
    detect_gender,
    get_pronouns,
    clean_token_for_output,
    join_tokens_natural,
    build_subject_phrase,
    CATEGORY_TEMPLATES,
    SUBJECT_DETAIL_TEMPLATES,
)

from .standard import (
    StandardAssembler,
    assemble_prompt,
)


__all__ = [
    # Base classes
    "BaseAssembler",
    "AssemblerConfig",
    "AssembledPrompt",
    "CategorySection",
    "PromptStyle",
    # Rules
    "Gender",
    "detect_gender",
    "get_pronouns",
    "clean_token_for_output",
    "join_tokens_natural",
    "build_subject_phrase",
    "CATEGORY_TEMPLATES",
    "SUBJECT_DETAIL_TEMPLATES",
    # Standard assembler
    "StandardAssembler",
    "assemble_prompt",
]
