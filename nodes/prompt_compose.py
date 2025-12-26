"""SID_PromptCompose node for prompt composition from metadata."""

import json
from typing import Tuple

from ..core.compose import ComposePipeline, ComposeConfig
from ..core.log import log, log_start, log_end, log_error


# =============================================================================
# Model Configuration - Single Dropdown
# =============================================================================

# Models for Enhance with AI mode - TEXT-ONLY models (not VLMs)
ENHANCE_MODELS = [
    # Local Qwen2.5 Text-Only (NOT VL/Vision models)
    "[Local] Qwen2.5-0.5B (~0.5GB)",   # Fastest, basic quality
    "[Local] Qwen2.5-1.5B (~1.5GB)",   # Good balance
    "[Local] Qwen2.5-3B (~2.5GB)",     # Better quality
    "[Local] Qwen2.5-7B (~5GB)",       # Best local quality
    # Google Gemini
    "[Gemini] gemini-2.0-flash",
    "[Gemini] gemini-1.5-flash",
    "[Gemini] gemini-1.5-pro",
    # OpenAI
    "[OpenAI] gpt-4o",
    "[OpenAI] gpt-4o-mini",
    # Anthropic
    "[Anthropic] claude-sonnet-4-20250514",
    "[Anthropic] claude-3-5-haiku-20241022",
]

# Map display name to (provider, model_id/config_name)
MODEL_MAP = {
    # Local Qwen2.5 Text-Only
    "[Local] Qwen2.5-0.5B (~0.5GB)": ("local", "qwen25_text_0.5b"),
    "[Local] Qwen2.5-1.5B (~1.5GB)": ("local", "qwen25_text_1.5b"),
    "[Local] Qwen2.5-3B (~2.5GB)": ("local", "qwen25_text_3b"),
    "[Local] Qwen2.5-7B (~5GB)": ("local", "qwen25_text_7b"),
    # Gemini
    "[Gemini] gemini-2.0-flash": ("gemini", "gemini-2.0-flash-exp"),
    "[Gemini] gemini-1.5-flash": ("gemini", "gemini-1.5-flash"),
    "[Gemini] gemini-1.5-pro": ("gemini", "gemini-1.5-pro"),
    # OpenAI
    "[OpenAI] gpt-4o": ("openai", "gpt-4o"),
    "[OpenAI] gpt-4o-mini": ("openai", "gpt-4o-mini"),
    # Anthropic
    "[Anthropic] claude-sonnet-4-20250514": ("anthropic", "claude-sonnet-4-20250514"),
    "[Anthropic] claude-3-5-haiku-20241022": ("anthropic", "claude-3-5-haiku-20241022"),
}

OUTPUT_STYLES = ["z-image"]
COMPOSE_MODES = ["Standard", "Enhance with AI"]
CAPTION_SOURCES = [
    "Auto",                    # Best available (VLM > Florence)
    "Synthesis (VLM)",         # Use VLM description from Synthesis
    "Analysis (Florence)",     # Use Florence caption from Analysis
    "Tags Only",               # Build from tags, no caption base
]


class SID_PromptCompose:
    """
    Compose natural language prompts from image analysis metadata.

    Converts multi-model tagging metadata into structured prompts
    optimized for specific image generation models (Z-Image, Flux, etc.).

    Modes:
    - Standard: NLP + Florence integration (deterministic)
    - Enhance with AI: AI-based synthesis (best quality)
    """

    CATEGORY = "SID Nodes"
    FUNCTION = "compose"
    RETURN_TYPES = ("IMAGE", "SID_METADATA", "STRING")
    RETURN_NAMES = ("image", "metadata", "prompt")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image (passed through to output)"
                }),
                "metadata": ("SID_METADATA", {
                    "tooltip": "Analysis metadata from SID_ImageAnalysis"
                }),
                "output_style": (OUTPUT_STYLES, {
                    "default": "z-image",
                    "tooltip": "Target prompt format (z-image for Flux/Z-Image models)"
                }),
                "mode": (COMPOSE_MODES, {
                    "default": "Standard",
                    "tooltip": "Standard: NLP+Florence | Enhance with AI: AI synthesis"
                }),
                "caption_source": (CAPTION_SOURCES, {
                    "default": "Auto",
                    "tooltip": "Caption base: Auto (best available) | Synthesis (VLM) | Analysis (Florence) | Tags Only"
                }),
                "enhance_model": (ENHANCE_MODELS, {
                    "default": ENHANCE_MODELS[0],
                    "tooltip": "AI model for Enhance with AI mode (ignored in Standard)"
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for cloud providers (Gemini/OpenAI/Anthropic)"
                }),
                "min_confidence": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Minimum tag confidence to include"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "AI sampling temperature (Enhance with AI mode only)"
                }),
                "release_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release local AI model after use"
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed logging with analytics"
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token for gated local models"
                }),
            },
        }

    def compose(
        self,
        image,
        metadata,
        output_style: str = "z-image",
        mode: str = "Standard",
        caption_source: str = "Auto",
        enhance_model: str = None,
        api_key: str = "",
        min_confidence: float = 0.1,
        temperature: float = 0.1,
        release_vram: bool = True,
        verbose: bool = False,
        hf_token: str = "",
    ) -> Tuple:
        """Compose prompt from metadata."""

        # Parse model selection
        enhance_model = enhance_model or ENHANCE_MODELS[0]
        provider, model_id = MODEL_MAP.get(enhance_model, ("local", "qwen25_text_1.5b"))

        # Use hf_token for local models, api_key for cloud providers
        hf_token_val = hf_token if provider == "local" else None
        cloud_api_key = api_key if provider != "local" else None

        # Normalize mode for internal use
        mode_internal = "enhance_ai" if "enhance" in mode.lower() else "standard"

        log("Compose", f"Mode: {mode} | Style: {output_style} | Caption: {caption_source}", force=True)
        if mode_internal == "enhance_ai":
            log("Compose", f"Model: {enhance_model}", force=True)

        # Parse metadata if string
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                log_error("Compose", "Failed to parse metadata JSON")
                metadata_dict = {}
        else:
            metadata_dict = metadata

        # Build config
        config = ComposeConfig(
            min_confidence=min_confidence,
            caption_source=caption_source,
            temperature=temperature,
        )

        # Create pipeline
        pipeline = ComposePipeline(
            style=output_style,
            mode=mode_internal,
            caption_source=caption_source,
            provider=provider,
            model=model_id,
            api_key=cloud_api_key,
            hf_token=hf_token_val,
            config=config,
        )

        # Generate
        try:
            result = pipeline.generate(metadata_dict)
            prompt = result.prompt

            # Log analytics if verbose
            if verbose and result.analytics:
                analytics = result.analytics.to_dict()
                log("Compose", f"Analytics: {json.dumps(analytics['summary'], indent=2)}", force=True)

                for source, stats in analytics.get('by_source', {}).items():
                    log("Compose", f"  {source}: {stats['included']}/{stats['total']} tags included", force=True)

            log("Compose", f"Generated {result.word_count} words", force=True)

            # Add analytics to metadata
            if result.analytics:
                metadata_dict["compose_analytics"] = result.analytics.to_dict()

        except Exception as e:
            log_error("Compose", f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            prompt = f"Error: {e}"

        finally:
            # Cleanup if Enhance with AI mode and release_vram is enabled
            if mode_internal == "enhance_ai" and release_vram:
                pipeline.unload()

        # Return metadata as JSON string
        metadata_out = json.dumps(metadata_dict, indent=2, ensure_ascii=False)

        return (image, metadata_out, prompt)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SID_PromptCompose": SID_PromptCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptCompose": "SID Prompt Compose",
}
