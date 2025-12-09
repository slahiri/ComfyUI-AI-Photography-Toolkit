"""
SID_Grok_LLM Node

xAI Grok LLM provider node for ComfyUI.
Outputs an LLM_MODEL configuration that can be connected to SID_ZImagePromptGenerator.
"""

from typing import List
from comfy_api.latest import io as comfy_io
from .llm_model_type import LLMModelConfig
from .base_llm_provider import BaseLLMProvider

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


class SID_Grok_LLM(comfy_io.ComfyNode, BaseLLMProvider):
    """
    xAI Grok LLM Provider.

    Configures Grok models for use with SID Photography Toolkit nodes.
    Connect the output to SID_ZImagePromptGenerator's llm_model input.
    """

    PROVIDER_NAME = "grok"

    # Available Grok models with vision support
    MODELS = [
        "grok-2-vision-1212",    # Grok 2 Vision (latest)
        "grok-vision-beta",      # Grok Vision Beta
    ]

    @classmethod
    def get_models(cls) -> List[str]:
        return cls.MODELS

    @classmethod
    def get_default_model(cls) -> str:
        return "grok-2-vision-1212"

    @classmethod
    def get_default_url(cls) -> str:
        return "https://api.x.ai/v1"

    @classmethod
    def get_api_key_url(cls) -> str:
        return "https://console.x.ai/"

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""
        return comfy_io.Schema(
            node_id="SID_Grok_LLM",
            display_name="SID Grok LLM",
            category="SID Photography Toolkit/LLM Providers",
            description="xAI Grok LLM provider. Connect to SID_ZImagePromptGenerator.",
            inputs=[
                comfy_io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="xAI API key (get from console.x.ai)"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=cls.MODELS,
                    default=cls.get_default_model(),
                    tooltip="Grok model to use. grok-2-vision=latest"
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
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""

        # Validate API key
        is_valid, error_msg = cls.validate_api_key(api_key)
        if not is_valid:
            print(f"[SID_Grok_LLM] Warning: {error_msg}")

        # Create configuration
        config = cls.create_config(
            model=model,
            api_key=api_key.strip(),
            api_url=cls.get_default_url(),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        print(f"[SID_Grok_LLM] Configured: {model} (max_tokens={max_tokens}, temp={temperature})")

        return comfy_io.NodeOutput(config)
