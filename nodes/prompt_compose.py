"""SID_PromptCompose node for prompt composition from metadata."""

import json
from typing import Tuple

from ..core.compose.compose_pipeline import (
    ComposePipeline,
    PipelineConfig,
    ComposeMode,
    PromptStyle,
)
from ..core.log import log, log_start, log_end, log_error


# =============================================================================
# Model Configuration
# =============================================================================

# Models for LLM mode - TEXT-ONLY models (not VLMs)
LLM_MODELS = [
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

# Mode options
COMPOSE_MODES = ["NLP (Fast)", "LLM (Enhanced)"]

# Prompt output styles
PROMPT_STYLES = ["Natural", "Tags", "Hybrid", "Structured"]

# Style map
STYLE_MAP = {
    "Natural": PromptStyle.NATURAL,
    "Tags": PromptStyle.TAGS,
    "Hybrid": PromptStyle.HYBRID,
    "Structured": PromptStyle.STRUCTURED,
}


class SID_PromptCompose:
    """
    Compose natural language prompts from image analysis metadata.

    Uses a multi-phase pipeline:
    1. Tokenization - Extract tokens from all metadata sources
    2. Classification - Categorize tokens into 9 canonical categories
    3. Assembly - Build structured prompts from classified tokens
    4. Validation - Check quality and auto-correct issues

    Modes:
    - NLP (Fast): Rule-based processing, deterministic, ~50ms
    - LLM (Enhanced): AI-enhanced with local or cloud LLMs, ~1-3s
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
                "mode": (COMPOSE_MODES, {
                    "default": "NLP (Fast)",
                    "tooltip": "NLP: Fast rule-based | LLM: AI-enhanced quality"
                }),
                "prompt_style": (PROMPT_STYLES, {
                    "default": "Structured",
                    "tooltip": "Natural: Prose | Tags: Comma-separated | Hybrid: Sentence + tags | Structured: Category-labeled lines"
                }),
                "llm_model": (LLM_MODELS, {
                    "default": LLM_MODELS[0],
                    "tooltip": "AI model for LLM mode (ignored in NLP mode)"
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for cloud providers (Gemini/OpenAI/Anthropic)"
                }),
                "min_confidence": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Minimum token confidence to include"
                }),
                "max_tokens_per_category": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Maximum tokens per category in output (higher = more detail)"
                }),
                "use_embeddings": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use semantic embeddings for classification (requires sentence-transformers)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "LLM sampling temperature (LLM mode only)"
                }),
                "release_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release local AI model after use"
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed logging with analytics"
                }),
            },
        }

    def compose(
        self,
        image,
        metadata,
        mode: str = "NLP (Fast)",
        prompt_style: str = "Natural",
        llm_model: str = None,
        api_key: str = "",
        min_confidence: float = 0.3,
        max_tokens_per_category: int = 5,
        use_embeddings: bool = False,
        temperature: float = 0.7,
        release_vram: bool = True,
        verbose: bool = False,
    ) -> Tuple:
        """Compose prompt from metadata using the new pipeline."""

        # Parse mode
        compose_mode = ComposeMode.LLM if "LLM" in mode else ComposeMode.NLP

        # Parse style
        style = STYLE_MAP.get(prompt_style, PromptStyle.NATURAL)

        # Parse LLM model selection
        llm_model = llm_model or LLM_MODELS[0]
        provider, model_id = MODEL_MAP.get(llm_model, ("local", "qwen25_text_1.5b"))

        # API key handling
        llm_api_key = api_key if provider != "local" and api_key else None

        log("Compose", f"Mode: {mode} | Style: {prompt_style}", force=True)
        if compose_mode == ComposeMode.LLM:
            log("Compose", f"LLM: {llm_model}", force=True)

        # Parse metadata if string
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                log_error("Compose", "Failed to parse metadata JSON")
                metadata_dict = {}
        else:
            metadata_dict = metadata if isinstance(metadata, dict) else {}

        # Build pipeline config
        config = PipelineConfig(
            mode=compose_mode,
            min_confidence=min_confidence,
            prompt_style=style,
            max_tokens_per_category=max_tokens_per_category,
            use_embeddings=use_embeddings,
            llm_provider=provider,
            llm_model=model_id,
            llm_api_key=llm_api_key,
            llm_temperature=temperature,
        )

        # Create and run pipeline
        pipeline = ComposePipeline(config)

        try:
            start = log_start("Compose", f"Generating ({mode})")
            result = pipeline.compose(metadata_dict)
            prompt = result.prompt

            log_end("Compose", f"Generated", start, f"{result.word_count} words")

            # Log analytics if verbose
            if verbose:
                log("Compose", f"  Tokens extracted: {result.tokens_extracted}", force=True)
                log("Compose", f"  Classification: {result.classification_stats.get('processed_percent', 0)}%", force=True)
                log("Compose", f"  Quality score: {result.quality_score:.2f}", force=True)
                log("Compose", f"  Categories: {', '.join(result.categories_used)}", force=True)
                if result.validation_issues > 0:
                    log("Compose", f"  Validation issues: {result.validation_issues}", force=True)

            # Add pipeline stats to metadata
            metadata_dict["compose_stats"] = {
                "mode": compose_mode.value,
                "style": prompt_style,
                "word_count": result.word_count,
                "tokens_extracted": result.tokens_extracted,
                "quality_score": result.quality_score,
                "categories_used": result.categories_used,
                "classification": result.classification_stats,
            }

        except Exception as e:
            log_error("Compose", f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            prompt = f"Error: {e}"

        finally:
            # Cleanup if LLM mode and release_vram is enabled
            if compose_mode == ComposeMode.LLM and release_vram:
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
