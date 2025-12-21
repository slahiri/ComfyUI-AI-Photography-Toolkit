"""SID Z-Image Prompt Generator node for ComfyUI."""

import time
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

import folder_paths

from ..core.models import ModelFactory
from ..core.models.base import CaptionMode, GenerationConfig
from ..core.output import clean_caption
from ..core.config import get_model_config
from ..core import download as download_module
from ..core.models import florence as florence_module
from ..core.models import qwen as qwen_module
from ..core.models import qwen3 as qwen3_module
from ..core.models import joycaption as joycaption_module
from ..core.models import minicpm as minicpm_module
from ..core.tagging_pipeline import TaggingPipeline


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


class SID_ZImagePromptGenerator:
    """
    Z-Image Prompt Generator node with model selection.

    Generates prompts optimized for image generation (SD, Flux, etc.).
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
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1, "display": "slider"}),
            },
            "optional": {
                "tag_config": ("TAG_CONFIG", {
                    "tooltip": "Tag configuration from SID Tag Configurator. If connected, runs tagging pipeline before VLM."
                }),
                "clean_output": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": False}),
                "release_model": ("BOOLEAN", {"default": True}),
                "force_redownload": ("BOOLEAN", {"default": False, "tooltip": "Delete and re-download model files (use if download was corrupted)"}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated models (e.g., MiniCPM). Get from https://huggingface.co/settings/tokens"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "tags")
    FUNCTION = "generate_prompt"
    CATEGORY = "SID Photography Toolkit"

    def _delete_model_files(self, model_key: str, verbose: bool = False) -> None:
        """Delete local model files to force re-download."""
        # First release the model from cache
        ModelFactory.release(model_key)

        # Get config to find model path
        try:
            config = get_model_config(model_key)
            model_id = config.get("model_id", "")
            subfolder = config.get("download", {}).get("subfolder", "LLM")

            if model_id:
                model_name = model_id.split("/")[-1]
                local_path = Path(folder_paths.models_dir) / subfolder / model_name

                if local_path.exists():
                    if verbose:
                        print(f"[SID-PromptGen] Deleting model files: {local_path}")
                    shutil.rmtree(local_path)
                    print(f"[SID-PromptGen] Deleted {model_name}, will re-download on next run")
                else:
                    if verbose:
                        print(f"[SID-PromptGen] Model not found locally: {local_path}")
        except Exception as e:
            print(f"[SID-PromptGen] Warning: Could not delete model files: {e}")

    def generate_prompt(
        self,
        image,
        model: str = "[Fast] Florence-2-large-PromptGen",
        mode: str = "detailed",
        precision: str = "Auto",
        max_tokens: int = 512,
        num_beams: int = 1,
        temperature: float = 0.7,
        tag_config: Optional[dict] = None,
        clean_output: bool = True,
        verbose: bool = False,
        release_model: bool = True,
        force_redownload: bool = False,
        hf_token: str = "",
    ) -> tuple[str, str]:
        """
        Generate prompt for input image.

        Args:
            image: ComfyUI image tensor
            model: Model to use for prompt generation (prefixed with tier)
            mode: Generation mode (detailed/short/tags/analyze)
            precision: Model precision (Auto/FP16/FP8/4-Bit)
            max_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            tag_config: Optional tag configuration from SID_TagConfigurator
            clean_output: Whether to clean the output
            verbose: Whether to log details to console
            release_model: Whether to release model after use (frees VRAM for KSampler)
            force_redownload: Whether to delete and re-download model files
            hf_token: HuggingFace token for gated models

        Returns:
            Tuple containing (prompt, tags)
        """
        start_time = time.time()

        # Set verbose mode and HF token for all modules
        download_module.set_verbose(verbose)
        download_module.set_hf_token(hf_token)
        florence_module.set_verbose(verbose)
        qwen_module.set_verbose(verbose)
        qwen3_module.set_verbose(verbose)
        joycaption_module.set_verbose(verbose)
        minicpm_module.set_verbose(verbose)

        # Convert tensor to PIL
        pil_image = _tensor_to_pil(image)

        # Run tagging pipeline if configured
        detected_tags = ""
        if tag_config is not None:
            if verbose:
                print(f"[SID-PromptGen] Running tagging pipeline...")

            # Override verbose in tag_config
            tag_config["verbose"] = verbose

            pipeline = TaggingPipeline(tag_config)
            tag_result = pipeline.run(pil_image)
            detected_tags = tag_result.tags_string

            if verbose:
                print(f"[SID-PromptGen] Detected {len(tag_result.tags_list)} tags")
                if detected_tags:
                    print(f"[SID-PromptGen] Tags: {detected_tags[:150]}...")

            # Unload taggers to free VRAM for VLM
            pipeline.unload_all()

        # Get model key from display name
        model_key = MODEL_MAP.get(model, "florence")

        # Handle force re-download
        if force_redownload:
            self._delete_model_files(model_key, verbose)

        # Release previous model if switching to a different model or precision
        model_changed = SID_ZImagePromptGenerator._last_model_key and SID_ZImagePromptGenerator._last_model_key != model_key
        precision_changed = SID_ZImagePromptGenerator._last_precision and SID_ZImagePromptGenerator._last_precision != precision

        if SID_ZImagePromptGenerator._last_model_key and (model_changed or precision_changed):
            if verbose:
                reason = "model" if model_changed else "precision"
                print(f"[SID-PromptGen] Releasing previous model ({SID_ZImagePromptGenerator._last_model_key}, {SID_ZImagePromptGenerator._last_precision}) - {reason} changed")
            ModelFactory.release(SID_ZImagePromptGenerator._last_model_key)

        if verbose:
            print(f"[SID-PromptGen] Model: {model} ({model_key})")
            print(f"[SID-PromptGen] Mode: {mode}, Precision: {precision}")
            print(f"[SID-PromptGen] Config: max_tokens={max_tokens}, num_beams={num_beams}, temperature={temperature}")
            print(f"[SID-PromptGen] Image size: {pil_image.size}")

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
            print(f"[SID-PromptGen] Generating prompt...")

        # Generate prompt
        prompt = vlm_model.generate(pil_image, mode=caption_mode, config=gen_config)

        # Clean if requested
        if clean_output:
            prompt = clean_caption(prompt)

        elapsed = time.time() - start_time

        if verbose:
            print(f"[SID-PromptGen] Generation time: {elapsed:.2f}s")
            print(f"[SID-PromptGen] Prompt length: {len(prompt)} chars")
            print(f"[SID-PromptGen] Output: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        # Track current model/precision for cleanup on next switch
        SID_ZImagePromptGenerator._last_model_key = model_key
        SID_ZImagePromptGenerator._last_precision = precision

        # Release model to free VRAM for other nodes (e.g., KSampler)
        if release_model:
            if verbose:
                print(f"[SID-PromptGen] Releasing model to free VRAM...")
            ModelFactory.release(model_key)
            SID_ZImagePromptGenerator._last_model_key = None  # Clear tracking since model is released
            SID_ZImagePromptGenerator._last_precision = None
            if verbose:
                print(f"[SID-PromptGen] Model released")

        return (prompt, detected_tags)
