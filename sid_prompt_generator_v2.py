# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - SID Z-Image Prompt Generator V2

ComfyUI wrapper for the standalone prompt generator.
All core logic is in prompt_generator_core.py for standalone usage.

Author: Siddhartha Lahiri
License: MIT
"""

import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Dict, Any

from comfy_api.latest import io

# Results storage directory
GENERATION_RESULTS_DIR = Path(__file__).parent / "generation_results"

# Local imports
from .llm_providers import LLMModelConfig
from .llm_providers.sid_llm_api import LLM_MODEL_Type
from .prompt_generator_core import (
    PromptGenerator,
    LLMConfig,
    load_templates,
    get_template_names,
    get_template_by_name,
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


def format_emphasis(text: str, strength: float = 1.3) -> str:
    """
    Format enhancement text with emphasis syntax and repetition.

    Input: "nude, red hair, blue eyes"
    Output: "(nude:1.3), nude, (red hair:1.3), red hair, (blue eyes:1.3), blue eyes"

    This provides double emphasis:
    1. Weight syntax (keyword:strength) for SD/SDXL/ComfyUI
    2. Repetition for natural language models (Z-Image, Flux)
    """
    if not text or not text.strip():
        return ""

    # Split by comma and clean up
    keywords = [k.strip() for k in text.split(",") if k.strip()]

    if not keywords:
        return ""

    # Format each keyword with emphasis + repetition
    formatted_parts = []
    for keyword in keywords:
        # Add weighted version and plain repetition
        formatted_parts.append(f"({keyword}:{strength})")
        formatted_parts.append(keyword)

    return ", ".join(formatted_parts)


def save_generation_result(
    pil_image: Image.Image,
    prompt: str,
    negative: str,
    caption: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Save generation result to disk for review.

    Args:
        pil_image: Source image
        prompt: Generated prompt
        negative: Negative prompt
        caption: Caption
        metadata: Generation metadata (settings, timing, etc.)

    Returns:
        Session ID (folder name)
    """
    GENERATION_RESULTS_DIR.mkdir(exist_ok=True)

    # Create session folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"gen_{timestamp}"
    session_dir = GENERATION_RESULTS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save source image
    try:
        image_path = session_dir / "source.jpg"
        pil_image.save(image_path, "JPEG", quality=90)
    except Exception as e:
        print(f"[SID Prompt Generator] Warning: Could not save image: {e}")

    # Save prompt
    prompt_path = session_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    # Save negative prompt if exists
    if negative:
        negative_path = session_dir / "negative.txt"
        negative_path.write_text(negative, encoding="utf-8")

    # Save caption if exists
    if caption:
        caption_path = session_dir / "caption.txt"
        caption_path.write_text(caption, encoding="utf-8")

    # Save metadata
    metadata_path = session_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[SID Prompt Generator] Results saved to: {session_dir}")
    return session_id


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
                    options=["Template", "Verbose", "Tags"],
                    default="Verbose",
                    tooltip="Template: Use preset prompt templates. Verbose: Natural flowing sentences. Tags: Comma-separated booru-style tags"
                ),
                io.Combo.Input(
                    "template",
                    options=get_template_names() or ["Detailed"],
                    default="Detailed",
                    tooltip="Select prompt template (only used when prompt_style is Template). Add templates to config/templates.toml and restart ComfyUI"
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
                    "prompt_enhance",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Optional: Keywords to emphasize (comma-separated). Integrated by LLM + added with emphasis syntax at end"
                ),
                io.Float.Input(
                    "emphasis_strength",
                    default=1.3,
                    min=1.0,
                    max=2.0,
                    step=0.1,
                    round=0.1,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Emphasis weight for prompt_enhance keywords (1.0=normal, 1.3=default, 2.0=max)"
                ),
                io.String.Input(
                    "prompt_override",
                    optional=True,
                    tooltip="Optional input: Skip generation and use this prompt directly (connect from another node)"
                ),
                io.Boolean.Input(
                    "store_results",
                    default=True,
                    display_name="Store Results",
                    tooltip="Save prompt, image, and metadata locally. View at /sid/generation-results"
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
        template: str,
        prompt_length: int,
        generate_negative: bool,
        generate_caption: bool,
        nsfw_mode: bool,
        seed: int,
        prompt_enhance: str = "",
        emphasis_strength: float = 1.3,
        prompt_override: str = None,
        store_results: bool = False,
    ):
        """Execute prompt generation using core module."""
        import random
        import time
        start_time = time.time()

        # Handle prompt_override - skip generation entirely if provided
        if prompt_override is not None and prompt_override.strip():
            print("[SID Prompt Generator] Using prompt override - skipping generation")
            final_prompt = prompt_override.strip()
            # Still apply emphasis layer to override if provided
            if prompt_enhance and prompt_enhance.strip():
                emphasis_text = format_emphasis(prompt_enhance.strip(), emphasis_strength)
                final_prompt = final_prompt + ", " + emphasis_text
                print(f"[SID Prompt Generator] Applied emphasis to override: {emphasis_text[:60]}...")
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

        # Map prompt_style to internal style
        # Verbose -> verbose (natural sentences)
        # Tags -> tags (comma-separated)
        # Template -> template (uses selected template's system prompt)
        internal_style = prompt_style.lower()
        if internal_style == "verbose":
            internal_style = "verbose"  # Keep as verbose

        # Get template info if using Template style
        template_prompt = None
        if prompt_style.lower() == "template":
            template_data = get_template_by_name(template)
            if template_data:
                template_prompt = template_data.get("system", "")
                print(f"[SID Prompt Generator] Using template: {template}")
            else:
                print(f"[SID Prompt Generator] Template not found: {template}, falling back to Verbose")
                internal_style = "verbose"

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
            prompt_style=internal_style,
            prompt_length=prompt_length,
            generate_negative=generate_negative,
            generate_caption=generate_caption,
            nsfw_mode=nsfw_mode,
            verbose=True,
            extra_params=llm_model.extra_params,
            template_prompt=template_prompt
        )

        # Process image using core module
        # For Standard/Detailed modes, prompt_enhance is integrated in synthesis
        # For Quick mode, it will be appended after generation
        enhance_text = prompt_enhance.strip() if prompt_enhance else ""
        result = generator.process_image(pil_image, user_guidance=enhance_text)

        # Print metadata and debug log to console
        reasoning_str = "ON" if supports_reasoning else "OFF"
        nsfw_str = "ON" if nsfw_mode else "OFF"
        print(f"\n[Seed: {seed}, Temperature: {temperature}, Mode: {analysis_mode}, Style: {prompt_style}, Reasoning: {reasoning_str}, NSFW: {nsfw_str}]")
        print(result.get_metadata_str())

        # Apply emphasis layer for ALL modes
        # This adds (keyword:weight) syntax + repetition for extra emphasis
        # LLM already integrated enhancement naturally, this is an additional emphasis layer
        final_prompt = result.prompt
        if enhance_text:
            emphasis_text = format_emphasis(enhance_text, emphasis_strength)
            final_prompt = final_prompt + ", " + emphasis_text
            print(f"[SID Prompt Generator] Applied emphasis layer: {emphasis_text[:60]}...")

        # Store results if enabled
        if store_results:
            generation_time = time.time() - start_time
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "seed": seed,
                "prompt_style": prompt_style,
                "template": template if prompt_style.lower() == "template" else None,
                "prompt_length": prompt_length,
                "generate_negative": generate_negative,
                "generate_caption": generate_caption,
                "nsfw_mode": nsfw_mode,
                "prompt_enhance": enhance_text,
                "emphasis_strength": emphasis_strength,
                "model_config": {
                    "provider": llm_model.provider,
                    "model": llm_model.model,
                    "analysis_mode": analysis_mode,
                    "temperature": temperature,
                },
                "timing": {
                    "total_seconds": round(generation_time, 2)
                }
            }
            save_generation_result(pil_image, final_prompt, result.negative, result.caption, metadata)

        return io.NodeOutput(final_prompt, result.negative, result.caption)


# =============================================================================
# Prompt Template Node
# =============================================================================

class SID_PromptTemplate(io.ComfyNode):
    """
    SID Prompt Template Node

    Select a template from the dropdown to populate the prompt field.
    Edit the prompt as needed, then connect to other nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Get template names and create a mapping
        template_names = get_template_names() or ["Default"]

        return io.Schema(
            node_id="SID_PromptTemplate",
            display_name="SID Prompt Template",
            category="SID Photography Toolkit",
            description="Select and edit prompt templates",
            inputs=[
                io.Combo.Input(
                    "template",
                    options=template_names,
                    default=template_names[0] if template_names else "Default",
                    tooltip="Select a prompt template to load"
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Edit the prompt template here. Changes are used as output."
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="prompt"),
            ],
        )

    @classmethod
    def execute(cls, template: str, prompt: str):
        """Return the prompt text."""
        # If prompt is empty, load from template
        if not prompt or not prompt.strip():
            template_data = get_template_by_name(template)
            if template_data:
                prompt = template_data.get("system", "")

        return io.NodeOutput(prompt)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SID_ZImagePromptGeneratorV2": SID_ZImagePromptGeneratorV2,
    "SID_PromptTemplate": SID_PromptTemplate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_ZImagePromptGeneratorV2": "SID Z-Image Prompt Generator",
    "SID_PromptTemplate": "SID Prompt Template",
}
