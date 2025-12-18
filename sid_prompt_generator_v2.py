# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - SID Z-Image Prompt Generator V2

ComfyUI wrapper for the standalone prompt generator.
All core logic is in prompt_generator_core.py for standalone usage.

Author: Siddhartha Lahiri
License: MIT
"""

import gc
import numpy as np
from PIL import Image
from typing import Dict, Any

from comfy_api.latest import io

# Local imports
from .llm_providers import LLMModelConfig
from .llm_providers.sid_llm_api import LLM_MODEL_Type
from .prompt_generator_core import (
    PromptGenerator,
    LLMConfig,
)


# =============================================================================
# Helper Functions
# =============================================================================

def tensor_to_pil(tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image."""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)


def pil_to_tensor(pil_image: Image.Image):
    """Convert PIL Image to ComfyUI tensor."""
    import torch
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)


# =============================================================================
# Main Node Class
# =============================================================================

class SID_ZImagePromptGeneratorV2(io.ComfyNode):
    """
    SID Z-Image Prompt Generator V2

    ComfyUI node wrapper for fast, reliable prompt generation.
    - CV detection (YOLO + MediaPipe) for human detection
    - Vision LLM for detailed descriptions
    - Z-Image optimization always on
    - Optional negative and caption generation

    For standalone usage (batch processing, training data):
        from prompt_generator_core import PromptGenerator
        generator = PromptGenerator(provider="ollama", model="llava")
        result = generator.process_image("image.jpg")
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SID_ZImagePromptGeneratorV2",
            display_name="SID Z-Image Prompt Generator",
            category="SID Photography Toolkit",
            description="Fast, reliable prompt generation with CV detection + Vision LLM",
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image to analyze"
                ),
                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="Connect SID_LLM_API or SID_LLM_Local node"
                ),
                io.Combo.Input(
                    "prompt_style",
                    options=["Expanded", "Tags"],
                    default="Expanded",
                    tooltip="Expanded: Natural flowing sentences. Tags: Comma-separated booru-style tags (better for anime/illustration models)"
                ),
                io.Int.Input(
                    "prompt_length",
                    default=400,
                    min=50,
                    max=1000,
                    tooltip="Target word count (50-1000)"
                ),
                io.Boolean.Input(
                    "generate_negative",
                    default=False,
                    display_name="Generate Negative",
                    tooltip="Generate negative prompt"
                ),
                io.Boolean.Input(
                    "generate_caption",
                    default=False,
                    display_name="Generate Caption",
                    tooltip="Generate image caption"
                ),
                io.Boolean.Input(
                    "nsfw_mode",
                    default=False,
                    display_name="NSFW Mode",
                    tooltip="Enable detailed NSFW feature description (sideboob, underboob, cleavage, exposed skin, etc.)"
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    tooltip="Seed for reproducibility (0 = random)"
                ),
                io.String.Input(
                    "prompt_override",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Optional: Skip generation and use this prompt directly"
                ),
                io.String.Input(
                    "prompt_enhance",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Optional: Additional text to append to the generated prompt (e.g., clothing changes, nudity, style modifications)"
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="prompt"),
                io.String.Output("negative", display_name="negative"),
                io.String.Output("caption", display_name="caption"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        llm_model: LLMModelConfig,
        prompt_style: str,
        prompt_length: int,
        generate_negative: bool,
        generate_caption: bool,
        nsfw_mode: bool,
        seed: int,
        prompt_override: str = "",
        prompt_enhance: str = "",
    ):
        """Execute prompt generation using core module."""
        import random

        # Handle prompt_override - skip generation entirely if provided
        if prompt_override and prompt_override.strip():
            print("[SID Prompt Generator] Using prompt override - skipping generation")
            final_prompt = prompt_override.strip()
            # Still apply enhancement to override if provided
            if prompt_enhance and prompt_enhance.strip():
                final_prompt = final_prompt + " " + prompt_enhance.strip()
                print(f"[SID Prompt Generator] Applied enhancement to override")
            return io.NodeOutput(final_prompt, "", "")

        # Set seed for reproducibility
        if seed == 0:
            seed = random.randint(1, 2147483647)
        random.seed(seed)

        # Clear VRAM
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except:
            pass

        # Convert ComfyUI tensor to PIL Image
        pil_image = tensor_to_pil(image)

        # Get analysis_mode, temperature, and reasoning from LLM config
        analysis_mode = getattr(llm_model, 'analysis_mode', 'standard')
        temperature = getattr(llm_model, 'temperature', 0.7)
        supports_reasoning = getattr(llm_model, 'supports_reasoning', False)

        # Create generator with LLM config from node connection
        generator = PromptGenerator(
            provider=llm_model.provider,
            model=llm_model.model,
            text_model=llm_model.text_model or "",
            api_key=llm_model.api_key or "",
            api_url=llm_model.api_url or "",
            temperature=temperature,
            analysis_mode=analysis_mode,
            enable_reasoning=supports_reasoning,
            prompt_style=prompt_style.lower(),  # "expanded" or "tags"
            prompt_length=prompt_length,
            generate_negative=generate_negative,
            generate_caption=generate_caption,
            nsfw_mode=nsfw_mode,
            verbose=True,
            extra_params=llm_model.extra_params
        )

        # Process image using core module (no user_guidance - removed)
        result = generator.process_image(pil_image)

        # Print metadata and debug log to console
        reasoning_str = "ON" if supports_reasoning else "OFF"
        nsfw_str = "ON" if nsfw_mode else "OFF"
        print(f"\n[Seed: {seed}, Temperature: {temperature}, Mode: {analysis_mode}, Style: {prompt_style}, Reasoning: {reasoning_str}, NSFW: {nsfw_str}]")
        print(result.get_metadata_str())

        # Apply prompt enhancement if provided
        final_prompt = result.prompt
        if prompt_enhance and prompt_enhance.strip():
            final_prompt = final_prompt + " " + prompt_enhance.strip()
            print(f"[SID Prompt Generator] Applied enhancement: {prompt_enhance.strip()[:50]}...")

        return io.NodeOutput(final_prompt, result.negative, result.caption)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SID_ZImagePromptGeneratorV2": SID_ZImagePromptGeneratorV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_ZImagePromptGeneratorV2": "SID Z-Image Prompt Generator",
}
