"""
Prompt Template Factory

Factory for getting the appropriate prompt template based on provider.
Follows Open/Closed and Dependency Inversion principles.
"""

from enum import Enum
from typing import Optional

from .base_prompt_template import BasePromptTemplate
from .claude_template import ClaudePromptTemplate


class PromptTemplateType(Enum):
    """Available prompt template types."""
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"
    DEFAULT = "default"


# Singleton instances for each template type
_template_instances: dict = {}


def _get_template_instance(template_type: PromptTemplateType) -> BasePromptTemplate:
    """Get or create a singleton template instance."""
    if template_type not in _template_instances:
        # Claude template works well for all providers
        _template_instances[template_type] = ClaudePromptTemplate()
    return _template_instances[template_type]


def get_prompt_template_for_provider(
    provider: str,
    model: Optional[str] = None,
    template_override: Optional[PromptTemplateType] = None,
) -> BasePromptTemplate:
    """
    Get the appropriate prompt template for a provider.

    This factory enables:
    - Open/Closed: Add new templates without modifying existing code
    - Dependency Inversion: Callers depend on BasePromptTemplate abstraction

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "qwenvl", "ollama")
        model: Optional model name for fine-grained selection
        template_override: Force a specific template type

    Returns:
        BasePromptTemplate instance optimized for the provider
    """
    # Allow explicit override
    if template_override:
        return _get_template_instance(template_override)

    # All providers use the Claude template - it works well universally
    return _get_template_instance(PromptTemplateType.CLAUDE)


def register_template(
    template_type: PromptTemplateType,
    template_instance: BasePromptTemplate
) -> None:
    """
    Register a custom template instance.

    This allows extending the system with custom templates without
    modifying the factory code (Open/Closed principle).

    Args:
        template_type: The template type to register
        template_instance: The template instance to use
    """
    _template_instances[template_type] = template_instance
