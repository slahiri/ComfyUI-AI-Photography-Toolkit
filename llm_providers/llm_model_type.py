# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - LLM Model Type

Custom type for passing LLM configuration between nodes.
This enables modular provider selection in ComfyUI workflows.

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass
class LLMModelConfig:
    """
    Configuration object for LLM providers.

    This is passed between provider nodes and consumer nodes
    (like SID_ZImagePromptGenerator) to configure the LLM.
    """

    # Provider identification
    provider: str                    # "anthropic", "openai", "ollama", etc.
    model: str                       # Vision model identifier (e.g., "claude-sonnet-4-5-20250929")
    text_model: str = ""             # Text model identifier (if different from vision model)

    # Authentication
    api_key: str = ""               # API key (empty for local providers like Ollama)
    api_url: str = ""               # Base URL for API calls

    # Generation parameters
    max_tokens: int = 1024          # Maximum output tokens
    temperature: float = 0.7         # Creativity (0.0-1.0)

    # Analysis mode - set by LLM provider node
    analysis_mode: str = "detailed"  # "standard" (1 call) or "detailed" (3-4 calls)

    # Capabilities
    supports_vision: bool = True     # Whether model supports image input
    supports_system_prompt: bool = True  # Whether model supports system prompts
    supports_reasoning: bool = False  # Whether model supports extended thinking/reasoning mode

    # Provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.provider:
            raise ValueError("Provider name is required")
        if not self.model:
            raise ValueError("Model name is required")
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "text_model": self.text_model,
            "api_key": self.api_key,
            "api_url": self.api_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "analysis_mode": self.analysis_mode,
            "supports_vision": self.supports_vision,
            "supports_system_prompt": self.supports_system_prompt,
            "supports_reasoning": self.supports_reasoning,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMModelConfig":
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            text_model=data.get("text_model", ""),
            api_key=data.get("api_key", ""),
            api_url=data.get("api_url", ""),
            max_tokens=data.get("max_tokens", 1024),
            temperature=data.get("temperature", 0.7),
            analysis_mode=data.get("analysis_mode", "standard"),
            supports_vision=data.get("supports_vision", True),
            supports_system_prompt=data.get("supports_system_prompt", True),
            supports_reasoning=data.get("supports_reasoning", False),
            extra_params=data.get("extra_params", {}),
        )

    def __repr__(self) -> str:
        return f"LLMModelConfig(provider='{self.provider}', model='{self.model}')"


# Type alias for ComfyUI type system
LLM_MODEL = LLMModelConfig
