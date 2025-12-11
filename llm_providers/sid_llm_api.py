"""
SID_LLM_API Node - Unified Cloud LLM Provider

Single node for all cloud-based and local LLM providers.
Only vision-capable models are listed (required for image analysis).

Cloud Providers (API key required):
- Anthropic (Claude - all models support vision)
- OpenAI (GPT-4o, GPT-4.1 - vision models)
- Google Gemini (all models support vision)
- xAI Grok (grok-2-vision)
- Mistral AI (Pixtral vision models)

Free/Freemium Cloud Providers:
- Groq (free tier - Llama 4, Llama 3.2 vision)
- Together AI (free tier - Llama Vision, Qwen2.5-VL)
- OpenRouter (aggregator - free vision models)
- Fireworks AI (Llama 4, Llama 3.2 vision, Qwen VL)
- HuggingFace Inference (Llama 3.2 vision, Qwen VL)

Local Providers (no API key):
- Ollama (dynamic detection)
- LM Studio (dynamic detection)
- Custom OpenAI-compatible endpoints

For local vision models with VRAM management, use SID_LLM_Local instead.
"""

from typing import List, Dict, Any, Tuple
import requests
from comfy_api.latest import io as comfy_io
from .llm_model_type import LLMModelConfig

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


# =============================================================================
# Dynamic Model Detection
# =============================================================================

# Dynamic model detection disabled - users should use custom_model field
# def get_ollama_models(timeout: float = 2.0) -> List[str]:
#     """Query Ollama API for installed models."""
#     ...
# def get_lmstudio_models(timeout: float = 2.0) -> List[str]:
#     """Query LM Studio API for loaded models."""
#     ...


def is_likely_vision_model(model_name: str) -> bool:
    """
    Check if a model name suggests vision capability.
    Used to warn users when selecting non-vision models for image analysis.
    """
    model_lower = model_name.lower()
    vision_keywords = [
        "vision", "vlm", "vl", "visual",
        "llava", "bakllava", "pixtral", "qwenvl", "qwen-vl", "qwen2-vl", "qwen2.5-vl",
        "moondream", "florence", "phi-3-vision", "phi3-vision", "smolvlm",
        "llama-3.2-11b", "llama-3.2-90b",  # Llama 3.2 vision models
        "llama3.2-vision", "llama3.2:11b", "llama3.2:90b",
        "llama-4", "llama4",  # Llama 4 models are multimodal
        "gpt-4o", "gpt-4-turbo", "gpt-4.1",  # OpenAI vision models
        "claude",  # All Claude models support vision
        "gemini",  # Gemini supports vision
        "grok-vision", "grok-2-vision", "grok-4",  # Grok vision models
    ]
    return any(kw in model_lower for kw in vision_keywords)


# =============================================================================
# Provider Configurations
# =============================================================================

PROVIDERS = {
    # =========================================================================
    # Major Cloud Providers (API key required)
    # =========================================================================
    "Anthropic": {
        "api_url": "https://api.anthropic.com",
        "api_key_url": "https://console.anthropic.com/",
        "provider_name": "anthropic",
        "requires_key": True,
        "is_local": False,
        "models": [
            # All Claude models support vision
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
        ],
        "default_model": "claude-sonnet-4-5-20250929",
    },
    "OpenAI": {
        "api_url": "https://api.openai.com/v1",
        "api_key_url": "https://platform.openai.com/api-keys",
        "provider_name": "openai",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Vision-capable models only
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "default_model": "gpt-4o",
    },
    "Google Gemini": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_url": "https://aistudio.google.com/app/apikey",
        "provider_name": "gemini",
        "requires_key": True,
        "is_local": False,
        "models": [
            # All Gemini models support vision
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ],
        "default_model": "gemini-2.5-flash",
    },
    "xAI Grok": {
        "api_url": "https://api.x.ai/v1",
        "api_key_url": "https://console.x.ai/",
        "provider_name": "grok",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Vision-capable models only
            "grok-2-vision-1212",
        ],
        "default_model": "grok-2-vision-1212",
    },
    "Mistral AI": {
        "api_url": "https://api.mistral.ai/v1",
        "api_key_url": "https://console.mistral.ai/api-keys/",
        "provider_name": "mistral",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Pixtral (vision models only)
            "pixtral-large-latest",
            "pixtral-12b-2409",
        ],
        "default_model": "pixtral-large-latest",
    },
    # DeepSeek removed - no vision models available

    # =========================================================================
    # Free/Freemium Cloud Providers
    # =========================================================================
    "Groq (Free Tier)": {
        "api_url": "https://api.groq.com/openai/v1",
        "api_key_url": "https://console.groq.com/keys",
        "provider_name": "groq",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Vision-capable models only (Llama 4 multimodal)
            # Note: Llama 3.2 vision preview models have been decommissioned
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ],
        "default_model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "Together AI (Free Tier)": {
        "api_url": "https://api.together.xyz/v1",
        "api_key_url": "https://api.together.xyz/settings/api-keys",
        "provider_name": "together",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Vision-capable models only
            "meta-llama/Llama-Vision-Free",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        ],
        "default_model": "meta-llama/Llama-Vision-Free",
    },
    "OpenRouter": {
        "api_url": "https://openrouter.ai/api/v1",
        "api_key_url": "https://openrouter.ai/keys",
        "provider_name": "openrouter",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Vision-capable free models (from /v1/models API Dec 2025)
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "google/gemini-2.0-flash-exp:free",
            "google/gemma-3-27b-it:free",
            "google/gemma-3-12b-it:free",
            "amazon/nova-2-lite-v1:free",
        ],
        "default_model": "nvidia/nemotron-nano-12b-v2-vl:free",
    },
    "Fireworks AI": {
        "api_url": "https://api.fireworks.ai/inference/v1",
        "api_key_url": "https://fireworks.ai/api-keys",
        "provider_name": "fireworks",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Vision-capable models only
            # Llama 4 multimodal (latest)
            "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "accounts/fireworks/models/llama4-scout-instruct-basic",
            # Llama 3.2 vision
            "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
            "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
            # Qwen vision
            "accounts/fireworks/models/qwen2p5-vl-32b-instruct",
            # Phi vision
            "accounts/fireworks/models/phi-3-vision-128k-instruct",
        ],
        "default_model": "accounts/fireworks/models/llama4-maverick-instruct-basic",
    },
    # Cerebras removed - no vision models available (text-only: llama3.1-70b, llama3.1-8b, llama-3.3-70b)
    "HuggingFace Inference": {
        "api_url": "https://api-inference.huggingface.co/v1",
        "api_key_url": "https://huggingface.co/settings/tokens",
        "provider_name": "huggingface",
        "requires_key": True,
        "is_local": False,
        "models": [
            # Vision-capable models only
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
        ],
        "default_model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    },

    # =========================================================================
    # Local Providers (no API key required)
    # =========================================================================
    "Ollama (Local)": {
        "api_url": "http://localhost:11434/v1",
        "api_key_url": "",
        "provider_name": "ollama",
        "requires_key": False,
        "is_local": True,
        "models": "dynamic",  # Auto-detected from Ollama API, use custom_model for manual entry
        "fallback_models": [],  # No fallback - use custom_model field instead
        "default_model": "use custom_model field",
    },
    "LM Studio (Local)": {
        "api_url": "http://localhost:1234/v1",
        "api_key_url": "",
        "provider_name": "lmstudio",
        "requires_key": False,
        "is_local": True,
        "models": "dynamic",  # Auto-detected from LM Studio API, use custom_model for manual entry
        "fallback_models": [],  # No fallback - use custom_model field instead
        "default_model": "use custom_model field",
    },
    "Custom OpenAI-Compatible": {
        "api_url": "http://localhost:8080/v1",
        "api_key_url": "",
        "provider_name": "openai_compatible",
        "requires_key": False,
        "is_local": True,
        "models": [
            "custom (use custom_model field)",
        ],
        "default_model": "custom (use custom_model field)",
    },
}

# Model metadata for all providers
MODEL_METADATA = {
    # =========================================================================
    # Anthropic models
    # =========================================================================
    "claude-sonnet-4-5-20250929": {"supports_reasoning": True, "max_output_tokens": 64000},
    "claude-haiku-4-5-20251001": {"supports_reasoning": False, "max_output_tokens": 64000},
    "claude-opus-4-1-20250805": {"supports_reasoning": True, "max_output_tokens": 64000},
    "claude-3-5-sonnet-20241022": {"supports_reasoning": True, "max_output_tokens": 8192},
    "claude-3-5-haiku-20241022": {"supports_reasoning": False, "max_output_tokens": 8192},

    # =========================================================================
    # OpenAI models
    # =========================================================================
    "gpt-4o": {"supports_reasoning": False, "max_output_tokens": 16384},
    "gpt-4o-mini": {"supports_reasoning": False, "max_output_tokens": 16384},
    "gpt-4.1": {"supports_reasoning": False, "max_output_tokens": 32768},
    "gpt-4.1-mini": {"supports_reasoning": False, "max_output_tokens": 32768},
    "gpt-4.1-nano": {"supports_reasoning": False, "max_output_tokens": 16384},
    "gpt-4-turbo": {"supports_reasoning": False, "max_output_tokens": 4096},
    "o1": {"supports_reasoning": True, "max_output_tokens": 100000},
    "o1-mini": {"supports_reasoning": True, "max_output_tokens": 65536},
    "o1-preview": {"supports_reasoning": True, "max_output_tokens": 32768},
    "o3": {"supports_reasoning": True, "max_output_tokens": 100000},
    "o3-mini": {"supports_reasoning": True, "max_output_tokens": 65536},
    "o4-mini": {"supports_reasoning": True, "max_output_tokens": 100000},

    # =========================================================================
    # Gemini models
    # =========================================================================
    "gemini-2.0-flash": {"supports_reasoning": False, "max_output_tokens": 8192},
    "gemini-2.0-flash-lite": {"supports_reasoning": False, "max_output_tokens": 8192},
    "gemini-1.5-pro": {"supports_reasoning": False, "max_output_tokens": 8192},
    "gemini-1.5-flash": {"supports_reasoning": False, "max_output_tokens": 8192},
    "gemini-1.5-flash-8b": {"supports_reasoning": False, "max_output_tokens": 8192},
    "gemini-2.5-pro-preview-06-05": {"supports_reasoning": True, "max_output_tokens": 65536},
    "gemini-2.5-flash-preview-05-20": {"supports_reasoning": True, "max_output_tokens": 65536},

    # =========================================================================
    # Grok models
    # =========================================================================
    "grok-2-vision-1212": {"supports_reasoning": False, "max_output_tokens": 32768},
    "grok-vision-beta": {"supports_reasoning": False, "max_output_tokens": 8192},
    "grok-3": {"supports_reasoning": True, "max_output_tokens": 131072},
    "grok-3-mini": {"supports_reasoning": True, "max_output_tokens": 131072},

    # =========================================================================
    # Mistral models
    # =========================================================================
    "pixtral-large-latest": {"supports_reasoning": False, "max_output_tokens": 16384},
    "pixtral-12b-2409": {"supports_reasoning": False, "max_output_tokens": 8192},
    "mistral-large-latest": {"supports_reasoning": False, "max_output_tokens": 16384},
    "mistral-medium-latest": {"supports_reasoning": False, "max_output_tokens": 8192},
    "mistral-small-latest": {"supports_reasoning": False, "max_output_tokens": 8192},
    "codestral-latest": {"supports_reasoning": False, "max_output_tokens": 16384},
    "open-mistral-nemo": {"supports_reasoning": False, "max_output_tokens": 8192},

    # =========================================================================
    # DeepSeek models
    # =========================================================================
    "deepseek-chat": {"supports_reasoning": False, "max_output_tokens": 8192},
    "deepseek-reasoner": {"supports_reasoning": True, "max_output_tokens": 16384},

    # =========================================================================
    # Groq models (free tier, very fast)
    # =========================================================================
    "llama-3.3-70b-versatile": {"supports_reasoning": False, "max_output_tokens": 32768},
    "llama-3.1-70b-versatile": {"supports_reasoning": False, "max_output_tokens": 32768},
    "llama-3.1-8b-instant": {"supports_reasoning": False, "max_output_tokens": 8192},
    "llama-3.2-90b-vision-preview": {"supports_reasoning": False, "max_output_tokens": 8192},
    "llama-3.2-11b-vision-preview": {"supports_reasoning": False, "max_output_tokens": 8192},
    "llama-3.2-3b-preview": {"supports_reasoning": False, "max_output_tokens": 8192},
    "llama-3.2-1b-preview": {"supports_reasoning": False, "max_output_tokens": 8192},
    "mixtral-8x7b-32768": {"supports_reasoning": False, "max_output_tokens": 32768},
    "gemma2-9b-it": {"supports_reasoning": False, "max_output_tokens": 8192},

    # =========================================================================
    # Together AI models
    # =========================================================================
    "meta-llama/Llama-Vision-Free": {"supports_reasoning": False, "max_output_tokens": 4096},
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": {"supports_reasoning": False, "max_output_tokens": 4096},
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": {"supports_reasoning": False, "max_output_tokens": 4096},
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"supports_reasoning": False, "max_output_tokens": 4096},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"supports_reasoning": False, "max_output_tokens": 4096},
    "Qwen/Qwen2.5-72B-Instruct-Turbo": {"supports_reasoning": False, "max_output_tokens": 4096},
    "Qwen/QwQ-32B-Preview": {"supports_reasoning": True, "max_output_tokens": 16384},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"supports_reasoning": False, "max_output_tokens": 4096},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free": {"supports_reasoning": True, "max_output_tokens": 8192},

    # =========================================================================
    # OpenRouter models (free tier)
    # =========================================================================
    "meta-llama/llama-3.2-90b-vision-instruct:free": {"supports_reasoning": False, "max_output_tokens": 4096},
    "meta-llama/llama-3.2-11b-vision-instruct:free": {"supports_reasoning": False, "max_output_tokens": 4096},
    "qwen/qwen-2-vl-7b-instruct:free": {"supports_reasoning": False, "max_output_tokens": 4096},
    "google/gemini-2.0-flash-exp:free": {"supports_reasoning": False, "max_output_tokens": 8192},
    "deepseek/deepseek-r1:free": {"supports_reasoning": True, "max_output_tokens": 8192},
    "deepseek/deepseek-chat:free": {"supports_reasoning": False, "max_output_tokens": 8192},
    "meta-llama/llama-3.3-70b-instruct:free": {"supports_reasoning": False, "max_output_tokens": 4096},
    "microsoft/phi-3-medium-128k-instruct:free": {"supports_reasoning": False, "max_output_tokens": 4096},
    "mistralai/mistral-7b-instruct:free": {"supports_reasoning": False, "max_output_tokens": 4096},

    # =========================================================================
    # Cerebras models (free tier, very fast)
    # =========================================================================
    "llama3.1-70b": {"supports_reasoning": False, "max_output_tokens": 8192},
    "llama3.1-8b": {"supports_reasoning": False, "max_output_tokens": 8192},
    "llama-3.3-70b": {"supports_reasoning": False, "max_output_tokens": 8192},

    # =========================================================================
    # Default for unknown/local models
    # =========================================================================
    "_default": {"supports_reasoning": False, "max_output_tokens": 4096},
}


def get_provider_models(provider_name: str, config: Dict) -> List[str]:
    """Get models for a specific provider."""
    models = config.get("models", [])

    # For local providers with "dynamic", return empty list
    # Users should use the custom_model field to specify their model
    if models == "dynamic":
        return []

    return models if isinstance(models, list) else []


def get_all_models() -> List[str]:
    """
    Get all models from all providers with provider prefix.

    Models are organized with folder-like grouping:
    - Models with "/" in their name are grouped by prefix (e.g., meta-llama/, Qwen/)
    - Within each provider, models are sorted with grouped models together
    - This creates a visual hierarchy in the dropdown
    """
    all_models = []

    for provider_name, config in PROVIDERS.items():
        models = get_provider_models(provider_name, config)

        # Separate models into groups based on their prefix
        # Models with "/" get grouped, others stay as-is
        grouped = {}  # prefix -> list of models
        ungrouped = []  # models without "/"

        for model in models:
            if "/" in model:
                # Extract prefix (everything before the first /)
                prefix = model.split("/")[0]
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(model)
            else:
                ungrouped.append(model)

        # Build the sorted list: ungrouped first, then grouped by prefix
        provider_models = []

        # Add ungrouped models first (they don't have subfolders)
        provider_models.extend(ungrouped)

        # Add grouped models, sorted by prefix
        for prefix in sorted(grouped.keys()):
            prefix_models = grouped[prefix]
            # Sort models within each group
            prefix_models.sort()
            provider_models.extend(prefix_models)

        # Add provider prefix to all models
        for model in provider_models:
            all_models.append(f"[{provider_name}] {model}")

    return all_models


def parse_model_selection(model_str: str) -> Tuple[str, str]:
    """Parse model string to get provider and model name."""
    if model_str.startswith("["):
        # Format: [Provider] model_name
        end_bracket = model_str.find("]")
        if end_bracket > 0:
            provider = model_str[1:end_bracket]
            model = model_str[end_bracket + 2:]  # Skip "] "
            return provider, model
    return "OpenAI", model_str


def get_model_metadata(model: str) -> Dict[str, Any]:
    """Get metadata for a model."""
    return MODEL_METADATA.get(model, MODEL_METADATA["_default"])


class SID_LLM_API(comfy_io.ComfyNode):
    """
    Unified LLM Provider for Cloud and Local APIs.
    Only vision-capable models are listed (required for image analysis).

    Cloud Providers (API key required):
    - Anthropic (Claude - all models support vision)
    - OpenAI (GPT-4o, GPT-4.1 - vision models)
    - Google Gemini (all models support vision)
    - xAI Grok (grok-2-vision)
    - Mistral AI (Pixtral vision models)

    Free/Freemium Cloud Providers:
    - Groq (free tier - Llama 4, Llama 3.2 vision)
    - Together AI (free tier - Llama Vision, Qwen2.5-VL)
    - OpenRouter (aggregator - free vision models)
    - Fireworks AI (Llama 4, Llama 3.2 vision, Qwen VL)
    - HuggingFace Inference (Llama 3.2 vision, Qwen VL)

    Local Providers (no API key):
    - Ollama (localhost:11434 - dynamic detection)
    - LM Studio (localhost:1234 - dynamic detection)
    - Custom OpenAI-compatible endpoints

    For local vision models with VRAM management, use SID_LLM_Local instead.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""
        all_models = get_all_models()
        provider_list = list(PROVIDERS.keys())

        # Max tokens presets
        max_tokens_options = [
            "Low (512)",
            "Medium (2048)",
            "High (8192)",
            "Very High (Model Max)",
            "Custom",
        ]

        return comfy_io.Schema(
            node_id="SID_LLM_API",
            display_name="SID LLM API",
            category="SID Photography Toolkit/LLM Providers",
            description="Unified LLM provider: Cloud (Anthropic, OpenAI, Gemini, Groq, etc.) + Local (Ollama, LM Studio)",
            inputs=[
                # Provider selection
                comfy_io.Combo.Input(
                    "provider",
                    options=provider_list,
                    default="Anthropic",
                    tooltip="Select LLM provider"
                ),

                # API Key
                comfy_io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="API key for your provider (not needed for local providers)"
                ),

                # Model selection - all models from all providers
                comfy_io.Combo.Input(
                    "model",
                    options=all_models,
                    default="[Anthropic] claude-sonnet-4-5-20250929",
                    tooltip="Select model (provider prefix shows which API will be used)"
                ),

                # Custom model override
                comfy_io.String.Input(
                    "custom_model",
                    default="",
                    multiline=False,
                    tooltip="Override model name (for custom/unlisted models)"
                ),

                # Custom API URL
                comfy_io.String.Input(
                    "api_url",
                    default="",
                    multiline=False,
                    tooltip="Custom API URL (leave empty to use provider default)"
                ),

                # Temperature
                comfy_io.Float.Input(
                    "temperature",
                    default=0.3,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity level (0=deterministic, 0.3=balanced, 1+=creative)"
                ),

                # Max tokens preset
                comfy_io.Combo.Input(
                    "max_tokens_preset",
                    options=max_tokens_options,
                    default="Medium (2048)",
                    tooltip="Output length: Low=512, Medium=2048, High=8192, Very High=Model Max"
                ),

                # Custom max tokens (only used when preset is "Custom")
                comfy_io.Int.Input(
                    "custom_max_tokens",
                    default=4096,
                    min=128,
                    max=200000,
                    tooltip="Custom max tokens (only used when preset is 'Custom')"
                ),

                # Reasoning toggle
                comfy_io.Boolean.Input(
                    "enable_reasoning",
                    default=True,
                    display_name="Enable Reasoning",
                    tooltip="Enable extended thinking for supported models (Claude 4.5, o1, o3, DeepSeek R1, etc.)"
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
        provider: str,
        api_key: str,
        model: str,
        custom_model: str,
        api_url: str,
        temperature: float,
        max_tokens_preset: str,
        custom_max_tokens: int,
        enable_reasoning: bool,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""
        try:
            # Parse model selection
            model_provider, model_name = parse_model_selection(model)

            # Use custom model if provided
            if custom_model.strip():
                model_name = custom_model.strip()

            # Get provider config
            provider_config = PROVIDERS.get(provider)
            if not provider_config:
                raise ValueError(f"Unknown provider: {provider}")

            provider_name = provider_config["provider_name"]

            # Determine API URL
            if api_url.strip():
                actual_url = api_url.strip()
            else:
                actual_url = provider_config.get("api_url", "")

            # Use default URLs for local providers if not specified
            if not actual_url:
                if provider_name == "lmstudio":
                    actual_url = "http://localhost:1234/v1"
                    print(f"[SID_LLM_API] Using default LM Studio URL: {actual_url}")
                elif provider_name == "ollama":
                    actual_url = "http://localhost:11434/v1"
                    print(f"[SID_LLM_API] Using default Ollama URL: {actual_url}")
                else:
                    raise ValueError(f"API URL is required for {provider}")

            # Auto-fix common URL mistakes for local providers
            # LM Studio and Ollama require /v1 suffix for OpenAI-compatible API
            if provider_name in ("lmstudio", "ollama", "openai_compatible"):
                # Remove trailing slash first
                actual_url = actual_url.rstrip("/")
                # Add /v1 if missing
                if not actual_url.endswith("/v1"):
                    actual_url = f"{actual_url}/v1"
                    print(f"[SID_LLM_API] Auto-corrected URL to: {actual_url}")

            # Check if provider requires API key
            requires_key = provider_config.get("requires_key", True)
            is_local = provider_config.get("is_local", False)

            # Validate API key - RAISE ERROR if required but missing
            if requires_key and not api_key.strip():
                api_key_url = provider_config.get("api_key_url", "")
                error_msg = f"API key is required for {provider}."
                if api_key_url:
                    error_msg += f"\n\nGet your API key at:\n{api_key_url}"
                raise ValueError(error_msg)

            # Get model metadata
            metadata = get_model_metadata(model_name)
            model_supports_reasoning = metadata["supports_reasoning"]
            model_max_tokens = metadata["max_output_tokens"]

            # Resolve max_tokens from preset
            max_tokens_map = {
                "Low (512)": 512,
                "Medium (2048)": 2048,
                "High (8192)": 8192,
                "Very High (Model Max)": model_max_tokens,
                "Custom": custom_max_tokens,
            }
            max_tokens = max_tokens_map.get(max_tokens_preset, 2048)

            # Cap max_tokens to model's maximum
            if max_tokens > model_max_tokens:
                print(f"[SID_LLM_API] Note: Capping max_tokens from {max_tokens} to model max {model_max_tokens}")
                max_tokens = model_max_tokens

            # Validate max_tokens
            if max_tokens < 1:
                raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

            # Validate temperature
            if temperature < 0 or temperature > 2:
                raise ValueError(f"Temperature must be between 0 and 2, got {temperature}")

            # Determine if reasoning should be enabled
            # Auto-disable reasoning for models that don't support it
            actual_reasoning = model_supports_reasoning and enable_reasoning

            # Inform user if reasoning was requested but not supported
            if enable_reasoning and not model_supports_reasoning:
                print(f"[SID_LLM_API] Note: Reasoning disabled - {model_name} does not support it")

            # Create configuration
            config = LLMModelConfig(
                provider=provider_name,
                model=model_name,
                api_key=api_key.strip(),
                api_url=actual_url,
                max_tokens=max_tokens,
                temperature=temperature,
                supports_vision=True,
                supports_system_prompt=True,
                supports_reasoning=actual_reasoning,
                extra_params={
                    "model_max_output_tokens": model_max_tokens,
                    "original_provider": provider,
                    "is_local": is_local,
                    "requires_key": requires_key,
                    "reasoning_supported": model_supports_reasoning,
                },
            )

            provider_type = "Local" if is_local else "Cloud"
            reasoning_status = "ON" if actual_reasoning else ("OFF (not supported)" if not model_supports_reasoning else "OFF")
            print(f"[SID_LLM_API] Configured: {provider} ({provider_type})")
            print(f"  Model: {model_name}")
            print(f"  URL: {actual_url}")
            print(f"  max_tokens={max_tokens} ({max_tokens_preset}), temp={temperature}")
            print(f"  reasoning={reasoning_status}")

            # Warn if local provider model may not support vision (but allow it)
            if is_local and not is_likely_vision_model(model_name):
                print(f"  ⚠️ WARNING: '{model_name}' may not support vision/image input.")
                print(f"     For image analysis, consider using a vision model like 'llava', 'llama3.2-vision', etc.")
                print(f"     Proceeding anyway - if this model does support vision, you can ignore this warning.")

            return comfy_io.NodeOutput(config)

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Catch any unknown errors and raise with context
            raise RuntimeError(f"[SID_LLM_API] Unexpected error: {type(e).__name__}: {str(e)}") from e
