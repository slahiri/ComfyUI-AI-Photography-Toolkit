"""
SID_OpenAI_Compatible_LLM Node

OpenAI-compatible LLM provider node for ComfyUI.
Supports OpenAI, Azure OpenAI, Together AI, Anyscale, LM Studio, and any OpenAI-compatible API.
Outputs an LLM_MODEL configuration that can be connected to SID_ZImagePromptGenerator.
"""

from typing import List
from comfy_api.latest import io as comfy_io
from .llm_model_type import LLMModelConfig
from .base_llm_provider import BaseLLMProvider

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


class SID_OpenAI_Compatible_LLM(comfy_io.ComfyNode, BaseLLMProvider):
    """
    OpenAI-compatible LLM Provider.

    Supports OpenAI and any OpenAI-compatible API including:
    - OpenAI (official)
    - Azure OpenAI
    - Together AI
    - Anyscale
    - Fireworks AI
    - LM Studio (local)
    - Ollama (OpenAI-compatible mode)
    - Any other OpenAI-compatible endpoint

    Connect the output to SID_ZImagePromptGenerator's llm_model input.
    """

    PROVIDER_NAME = "openai"

    # Common OpenAI and compatible models with vision support
    MODELS = [
        # OpenAI models
        "gpt-4o",                # GPT-4o (latest, multimodal)
        "gpt-4o-mini",           # GPT-4o Mini (fast/cheap)
        "gpt-4-turbo",           # GPT-4 Turbo with vision
        # Together AI models
        "meta-llama/Llama-Vision-Free",
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        # Use custom_model for other providers
    ]

    @classmethod
    def get_models(cls) -> List[str]:
        return cls.MODELS

    @classmethod
    def get_default_model(cls) -> str:
        return "gpt-4o"

    @classmethod
    def get_default_url(cls) -> str:
        return "https://api.openai.com/v1"

    @classmethod
    def get_api_key_url(cls) -> str:
        return "https://platform.openai.com/api-keys"

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""
        return comfy_io.Schema(
            node_id="SID_OpenAI_Compatible_LLM",
            display_name="SID OpenAI Compatible",
            category="SID Photography Toolkit/LLM Providers",
            description="OpenAI-compatible LLM provider. Supports OpenAI, Azure, Together AI, LM Studio, etc.",
            inputs=[
                comfy_io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="API key for your provider (OpenAI, Together AI, etc.)"
                ),
                comfy_io.String.Input(
                    "api_url",
                    default="https://api.openai.com/v1",
                    multiline=False,
                    tooltip="API endpoint URL. Examples: https://api.openai.com/v1 (OpenAI), https://api.together.xyz/v1 (Together), http://localhost:1234/v1 (LM Studio)"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=cls.MODELS,
                    default=cls.get_default_model(),
                    tooltip="Select a preset model or use custom_model field below"
                ),
                comfy_io.String.Input(
                    "custom_model",
                    default="",
                    multiline=False,
                    tooltip="Custom model name (overrides dropdown if not empty). Use for models not in the list."
                ),
                comfy_io.Int.Input(
                    "max_tokens",
                    default=300,
                    min=50,
                    max=4096,
                    step=50,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Maximum tokens in response"
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity level (0=focused, 2=very creative)"
                ),
            ],
            outputs=[
                LLM_MODEL_Type.Output(
                    "llm_model",
                    display_name="LLM_MODEL",
                    tooltip="LLM configuration to connect to SID_ZImagePromptGenerator"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        api_key: str,
        api_url: str,
        model: str,
        custom_model: str,
        max_tokens: int,
        temperature: float,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""

        # Use custom model if provided, otherwise use dropdown selection
        actual_model = custom_model.strip() if custom_model.strip() else model

        # Ensure model name starts with gpt- for OpenAI-compatible routing
        # If it doesn't, we'll prepend gpt- internally for routing purposes
        if not actual_model.startswith("gpt-"):
            # Store original model name but mark as OpenAI-compatible
            routing_model = f"gpt-{actual_model}"
        else:
            routing_model = actual_model

        # Validate API key (allow empty for local endpoints like LM Studio)
        api_url_clean = api_url.strip() if api_url.strip() else cls.get_default_url()
        is_local = "localhost" in api_url_clean or "127.0.0.1" in api_url_clean

        if not is_local:
            is_valid, error_msg = cls.validate_api_key(api_key)
            if not is_valid:
                print(f"[SID_OpenAI_Compatible] Warning: {error_msg}")

        # Create configuration - use original model name for API call
        config = LLMModelConfig(
            provider=cls.PROVIDER_NAME,
            model=actual_model,  # Use actual model name for API
            api_key=api_key.strip(),
            api_url=api_url_clean,
            max_tokens=max_tokens,
            temperature=temperature,
            supports_vision=True,
            supports_system_prompt=True,
            extra_params={"routing_model": routing_model},  # For internal routing
        )

        print(f"[SID_OpenAI_Compatible] Configured: {actual_model} @ {api_url_clean}")
        print(f"  max_tokens={max_tokens}, temp={temperature}")

        return comfy_io.NodeOutput(config)
