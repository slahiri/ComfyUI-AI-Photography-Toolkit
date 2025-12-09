"""
SID_Anthropic_LLM Node

Anthropic Claude LLM provider node for ComfyUI.
Outputs an LLM_MODEL configuration that can be connected to SID_ZImagePromptGenerator.
"""

from typing import List
from comfy_api.latest import io as comfy_io
from .llm_model_type import LLMModelConfig
from .base_llm_provider import BaseLLMProvider


class SID_Anthropic_LLM(comfy_io.ComfyNode, BaseLLMProvider):
    """
    Anthropic Claude LLM Provider.

    Configures Claude models for use with SID Photography Toolkit nodes.
    Connect the output to SID_ZImagePromptGenerator's llm_model input.
    """

    PROVIDER_NAME = "anthropic"

    # Available Claude models with vision support
    MODELS = [
        "claude-sonnet-4-5-20250929",    # Latest Sonnet 4.5
        "claude-haiku-4-5-20251001",     # Latest Haiku 4.5 (fast/cheap)
        "claude-opus-4-1-20250805",      # Opus 4.1 (most capable)
        "claude-3-5-sonnet-20241022",    # Claude 3.5 Sonnet
        "claude-3-5-haiku-20241022",     # Claude 3.5 Haiku
    ]

    @classmethod
    def get_models(cls) -> List[str]:
        return cls.MODELS

    @classmethod
    def get_default_model(cls) -> str:
        return "claude-sonnet-4-5-20250929"

    @classmethod
    def get_default_url(cls) -> str:
        return "https://api.anthropic.com"

    @classmethod
    def get_api_key_url(cls) -> str:
        return "https://console.anthropic.com/"

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""
        return comfy_io.Schema(
            node_id="SID_Anthropic_LLM",
            display_name="SID Anthropic LLM",
            category="SID Photography Toolkit/LLM Providers",
            description="Anthropic Claude LLM provider. Connect to SID_ZImagePromptGenerator.",
            inputs=[
                comfy_io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="Anthropic API key (get from console.anthropic.com)"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=cls.MODELS,
                    default=cls.get_default_model(),
                    tooltip="Claude model to use. Sonnet=balanced, Haiku=fast/cheap, Opus=most capable"
                ),
                comfy_io.Int.Input(
                    "max_tokens",
                    default=1024,
                    min=50,
                    max=4096,
                    step=50,
                    tooltip="Maximum tokens in response"
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity level (0=focused, 1=creative)"
                ),
            ],
            outputs=[
                comfy_io.Custom.Output(
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
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""

        # Validate API key
        is_valid, error_msg = cls.validate_api_key(api_key)
        if not is_valid:
            print(f"[SID_Anthropic_LLM] Warning: {error_msg}")

        # Create configuration
        config = cls.create_config(
            model=model,
            api_key=api_key.strip(),
            api_url=cls.get_default_url(),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        print(f"[SID_Anthropic_LLM] Configured: {model} (max_tokens={max_tokens}, temp={temperature})")

        return comfy_io.NodeOutput(config)
