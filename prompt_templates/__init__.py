"""
Prompt Templates Module

SOLID-compliant prompt template system for different LLM providers.
Each provider can have optimized prompts for best results.
"""

from .base_prompt_template import BasePromptTemplate
from .template_factory import get_prompt_template_for_provider, PromptTemplateType

__all__ = [
    "BasePromptTemplate",
    "get_prompt_template_for_provider",
    "PromptTemplateType",
]
