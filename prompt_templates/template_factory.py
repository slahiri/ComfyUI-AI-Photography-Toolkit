"""
Prompt Template Factory

Factory for getting the appropriate prompt template based on provider.
Follows Open/Closed and Dependency Inversion principles.
"""

from enum import Enum
from typing import Optional

from .base_prompt_template import BasePromptTemplate
from .claude_template import ClaudePromptTemplate
from .local_gguf_template import LocalGGUFPromptTemplate


class PromptTemplateType(Enum):
    """Available prompt template types."""
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL_GGUF = "local_gguf"
    DEFAULT = "default"


# Singleton instances for each template type
_template_instances: dict = {}


def _get_template_instance(template_type: PromptTemplateType) -> BasePromptTemplate:
    """Get or create a singleton template instance."""
    if template_type not in _template_instances:
        if template_type == PromptTemplateType.LOCAL_GGUF:
            _template_instances[template_type] = LocalGGUFPromptTemplate()
        else:
            # Claude template works well for Claude, OpenAI, Grok, etc.
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
        provider: Provider name (e.g., "anthropic", "openai", "gguf", "ollama")
        model: Optional model name for fine-grained selection
        template_override: Force a specific template type

    Returns:
        BasePromptTemplate instance optimized for the provider
    """
    # Allow explicit override
    if template_override:
        return _get_template_instance(template_override)

    provider_lower = provider.lower()

    # Map providers to their optimal templates
    if provider_lower == "gguf":
        return _get_template_instance(PromptTemplateType.LOCAL_GGUF)

    elif provider_lower == "qwenvl":
        # QwenVL models work best with simpler prompts like GGUF
        return _get_template_instance(PromptTemplateType.LOCAL_GGUF)

    elif provider_lower == "ollama":
        # Ollama can run various models - check model name for hints
        if model:
            model_lower = model.lower()
            # Small local models benefit from simpler prompts
            if any(x in model_lower for x in ["llava", "moondream", "bakllava", "minicpm"]):
                return _get_template_instance(PromptTemplateType.LOCAL_GGUF)
        # Default Ollama to Claude-style (works for larger models)
        return _get_template_instance(PromptTemplateType.CLAUDE)

    elif provider_lower in ["anthropic", "claude"]:
        return _get_template_instance(PromptTemplateType.CLAUDE)

    elif provider_lower in ["openai", "grok"]:
        # OpenAI and Grok work well with Claude-style prompts
        return _get_template_instance(PromptTemplateType.CLAUDE)

    else:
        # Default to Claude template for unknown providers
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
