"""
LLM Provider Models for ComfyUI-AI-Photography-Toolkit.

This module contains modular LLM provider nodes that can be connected
to SID_ZImagePromptGenerator for flexible provider selection.

Nodes:
- SID_LLM_API: Unified cloud LLM provider (Anthropic, OpenAI, Gemini, Grok, etc.)
- SID_LLM_Local: Local vision models (Florence-2, Moondream2, SmolVLM, Phi-3.5, QwenVL)
"""

from .llm_model_type import LLMModelConfig, LLM_MODEL
from .base_llm_provider import BaseLLMProvider
from .sid_llm_api import SID_LLM_API
from .sid_llm_local import SID_LLM_Local, LocalModelClient, LOCAL_MODELS

__all__ = [
    # Types
    "LLMModelConfig",
    "LLM_MODEL",
    # Base class
    "BaseLLMProvider",
    # Provider nodes (only 2 nodes)
    "SID_LLM_API",      # Unified cloud LLM provider
    "SID_LLM_Local",    # Local vision models (Florence-2, Moondream2, SmolVLM, Phi-3.5, QwenVL)
    # Local model utilities
    "LocalModelClient",
    "LOCAL_MODELS",
]
