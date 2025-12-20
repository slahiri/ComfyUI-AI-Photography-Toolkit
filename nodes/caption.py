"""SID Caption node for ComfyUI."""

import time
import numpy as np
from PIL import Image

from ..core.models import ModelFactory
from ..core.models.base import CaptionMode, GenerationConfig
from ..core.output import clean_caption


def _tensor_to_pil(tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image."""
    # ComfyUI tensors are (B, H, W, C) in 0-1 range
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch

    # Convert to numpy and scale to 0-255
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(np_image, mode="RGB")


# Model name to factory key mapping
MODEL_MAP = {
    # Fast tier (small models, ~1-8GB VRAM)
    "[Fast] BLIP-2": "blip2",
    "[Fast] Florence-2-base-PromptGen": "florence_base",
    "[Fast] Florence-2-large-PromptGen": "florence",
    "[Fast] Qwen2-VL-2B-Caption": "qwen2_2b",
    "[Fast] Qwen2.5-VL-3B-Caption": "qwen25_3b",
    "[Fast] Qwen3-VL-2B-Abliterated": "qwen3_2b",
    "[Fast] Qwen3-VL-4B-Abliterated": "qwen3_4b",

    # Balanced tier (large models, ~8-17GB VRAM)
    "[Balanced] JoyCaption-Alpha-Two": "joycaption_alpha",
    "[Balanced] JoyCaption-Beta-One": "joycaption",
    "[Balanced] MiniCPM-V-2.6": "minicpm_v26",
    "[Balanced] Qwen2-VL-7B-Captioner": "qwen2_7b",
    "[Balanced] Qwen2.5-VL-7B-Captioner": "qwen25_7b_captioner",
    "[Balanced] Qwen2.5-VL-7B-Caption": "qwen25_7b_caption",
    "[Balanced] Qwen3-VL-8B-Caption": "qwen3_8b",
}

# Sorted model list (Fast first, then Balanced)
ALL_MODELS = [
    # Fast tier
    "[Fast] BLIP-2",
    "[Fast] Florence-2-base-PromptGen",
    "[Fast] Florence-2-large-PromptGen",
    "[Fast] Qwen2-VL-2B-Caption",
    "[Fast] Qwen2.5-VL-3B-Caption",
    "[Fast] Qwen3-VL-2B-Abliterated",
    "[Fast] Qwen3-VL-4B-Abliterated",
    # Balanced tier
    "[Balanced] JoyCaption-Alpha-Two",
    "[Balanced] JoyCaption-Beta-One",
    "[Balanced] MiniCPM-V-2.6",
    "[Balanced] Qwen2-VL-7B-Captioner",
    "[Balanced] Qwen2.5-VL-7B-Captioner",
    "[Balanced] Qwen2.5-VL-7B-Caption",
    "[Balanced] Qwen3-VL-8B-Caption",
]


class SID_Caption:
    """
    Image captioning node with model selection.

    Generates captions optimized for image generation prompts.
    Supports multiple VLM models via plug-and-play architecture.
    """

    # Track last used model/precision for cleanup on switch
    _last_model_key: str | None = None
    _last_precision: str | None = None

    # Caption modes
    MODES = ["detailed", "short", "tags", "analyze"]

    # Precision modes
    PRECISION = ["Auto", "FP16", "FP8", "4-Bit"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (ALL_MODELS, {"default": "[Fast] Florence-2-large-PromptGen"}),
                "mode": (cls.MODES, {"default": "detailed"}),
                "precision": (cls.PRECISION, {"default": "Auto"}),
                "max_tokens": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 32}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1, "display": "slider"}),
            },
            "optional": {
                "clean_output": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": False}),
                "release_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "SID Photography Toolkit"

    def generate_caption(
        self,
        image,
        model: str = "[Fast] Florence-2-large-PromptGen",
        mode: str = "detailed",
        precision: str = "Auto",
        max_tokens: int = 256,
        num_beams: int = 1,
        temperature: float = 0.7,
        clean_output: bool = True,
        verbose: bool = False,
        release_model: bool = True,
    ) -> tuple[str]:
        """
        Generate caption for input image.

        Args:
            image: ComfyUI image tensor
            model: Model to use for captioning (prefixed with tier)
            mode: Caption mode (detailed/short/tags/analyze)
            precision: Model precision (Auto/FP16/FP8/4-Bit)
            max_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            clean_output: Whether to clean the output
            verbose: Whether to log details to console
            release_model: Whether to release model after use (frees VRAM for KSampler)

        Returns:
            Tuple containing caption string
        """
        start_time = time.time()

        # Convert tensor to PIL
        pil_image = _tensor_to_pil(image)

        # Get model key from display name
        model_key = MODEL_MAP.get(model, "florence")

        # Release previous model if switching to a different model or precision
        model_changed = SID_Caption._last_model_key and SID_Caption._last_model_key != model_key
        precision_changed = SID_Caption._last_precision and SID_Caption._last_precision != precision

        if SID_Caption._last_model_key and (model_changed or precision_changed):
            if verbose:
                reason = "model" if model_changed else "precision"
                print(f"[SID-Caption] Releasing previous model ({SID_Caption._last_model_key}, {SID_Caption._last_precision}) - {reason} changed")
            ModelFactory.release(SID_Caption._last_model_key)

        if verbose:
            print(f"[SID-Caption] Model: {model} ({model_key})")
            print(f"[SID-Caption] Mode: {mode}, Precision: {precision}")
            print(f"[SID-Caption] Config: max_tokens={max_tokens}, num_beams={num_beams}, temperature={temperature}")
            print(f"[SID-Caption] Image size: {pil_image.size}")

        # Get model instance (cached by factory)
        vlm_model = ModelFactory.get(model_key, precision=precision)

        # Create generation config
        gen_config = GenerationConfig(
            max_tokens=max_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=temperature > 0.1,
        )

        # Map string mode to enum
        caption_mode = CaptionMode(mode)

        if verbose:
            print(f"[SID-Caption] Generating caption...")

        # Generate caption
        caption = vlm_model.generate(pil_image, mode=caption_mode, config=gen_config)

        # Clean if requested
        if clean_output:
            caption = clean_caption(caption)

        elapsed = time.time() - start_time

        if verbose:
            print(f"[SID-Caption] Generation time: {elapsed:.2f}s")
            print(f"[SID-Caption] Caption length: {len(caption)} chars")
            print(f"[SID-Caption] Output: {caption[:200]}{'...' if len(caption) > 200 else ''}")

        # Track current model/precision for cleanup on next switch
        SID_Caption._last_model_key = model_key
        SID_Caption._last_precision = precision

        # Release model to free VRAM for other nodes (e.g., KSampler)
        if release_model:
            if verbose:
                print(f"[SID-Caption] Releasing model to free VRAM...")
            ModelFactory.release(model_key)
            SID_Caption._last_model_key = None  # Clear tracking since model is released
            SID_Caption._last_precision = None
            if verbose:
                print(f"[SID-Caption] Model released")

        return (caption,)
