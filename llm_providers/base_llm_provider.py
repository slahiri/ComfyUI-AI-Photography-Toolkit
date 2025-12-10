"""
Base LLM Provider Class

Base class for all LLM provider nodes.
Provides common functionality and interface for provider implementations.

Note: We don't use ABC (Abstract Base Class) because it conflicts with
ComfyUI's internal class locking mechanism for V3 API nodes.
"""

from typing import List, Dict, Any, Optional, Tuple
from .llm_model_type import LLMModelConfig


class BaseLLMProvider:
    """
    Base class for LLM provider nodes.

    All provider nodes (Anthropic, OpenAI, Ollama, etc.) should inherit
    from this class to ensure consistent interface.
    """

    # Override in subclasses
    PROVIDER_NAME: str = "base"
    CATEGORY: str = "SID Photography Toolkit/LLM Providers"

    @classmethod
    def get_models(cls) -> List[str]:
        """
        Return list of available models for this provider.
        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_models()")

    @classmethod
    def get_default_model(cls) -> str:
        """
        Return the default model for this provider.
        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_default_model()")

    @classmethod
    def get_default_url(cls) -> str:
        """
        Return the default API URL for this provider.
        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_default_url()")

    @classmethod
    def requires_api_key(cls) -> bool:
        """
        Return whether this provider requires an API key.
        Override in subclasses if needed (e.g., Ollama returns False).
        """
        return True

    @classmethod
    def get_api_key_url(cls) -> str:
        """
        Return URL where users can get an API key.
        Override in subclasses.
        """
        return ""

    @classmethod
    def supports_vision(cls, model: str) -> bool:
        """
        Return whether the specified model supports vision/image input.
        Override in subclasses if needed.
        """
        return True

    @classmethod
    def supports_system_prompt(cls, model: str) -> bool:
        """
        Return whether the specified model supports system prompts.
        Override in subclasses if needed.
        """
        return True

    @classmethod
    def supports_reasoning(cls, model: str) -> bool:
        """
        Return whether the specified model supports extended thinking/reasoning mode.
        Override in subclasses with MODEL_METADATA dict.
        """
        return False

    @classmethod
    def validate_api_key(cls, api_key: str) -> Tuple[bool, str]:
        """
        Validate the API key format (not actual authentication).
        Returns (is_valid, error_message).
        Override in subclasses for provider-specific validation.
        """
        if cls.requires_api_key() and not api_key.strip():
            return False, f"API key is required for {cls.PROVIDER_NAME}"
        return True, ""

    @classmethod
    def create_config(
        cls,
        model: str,
        api_key: str = "",
        api_url: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMModelConfig:
        """
        Create an LLMModelConfig for this provider.
        Can be overridden in subclasses for custom behavior.
        """
        return LLMModelConfig(
            provider=cls.PROVIDER_NAME,
            model=model,
            api_key=api_key,
            api_url=api_url or cls.get_default_url(),
            max_tokens=max_tokens,
            temperature=temperature,
            supports_vision=cls.supports_vision(model),
            supports_system_prompt=cls.supports_system_prompt(model),
            supports_reasoning=cls.supports_reasoning(model),
            extra_params=extra_params or {},
        )
