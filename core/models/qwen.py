"""Qwen2-VL and Qwen2.5-VL model implementation."""

import math
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .base import BaseCaptionModel, CaptionMode, GenerationConfig, get_dtype, get_quantization_config
from ..config import get_model_config, get_prompt
from ..download import download_model
from ..platform import isolated_execution

# ComfyUI imports
import comfy.model_management as mm

# Global verbose flag
_verbose = False


def set_verbose(verbose: bool) -> None:
    """Set verbose logging flag."""
    global _verbose
    _verbose = verbose


def _log(message: str) -> None:
    """Print message if verbose is enabled."""
    if _verbose:
        print(f"[SID-Qwen] {message}")


def _get_qwen_model_class(model_type: str):
    """Get the appropriate Qwen model class based on model type."""
    if model_type == "qwen2":
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    else:  # qwen25 or default
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


def _resize_for_qwen(image: Image.Image, max_pixels: int = 1003520) -> Image.Image:
    """
    Resize image to fit within max_pixels while maintaining aspect ratio.
    Qwen requires dimensions divisible by 28.
    """
    width, height = image.size
    current_pixels = width * height

    if current_pixels <= max_pixels:
        # Just ensure divisible by 28
        factor = 28
        new_width = max(factor, round(width / factor) * factor)
        new_height = max(factor, round(height / factor) * factor)
    else:
        # Scale down to fit max_pixels
        scale = math.sqrt(max_pixels / current_pixels)
        factor = 28
        new_width = max(factor, int(width * scale // factor) * factor)
        new_height = max(factor, int(height * scale // factor) * factor)

    if (new_width, new_height) != (width, height):
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


class QwenVLModel(BaseCaptionModel):
    """
    Qwen2-VL and Qwen2.5-VL model for image captioning.

    Supports Qwen2-VL and Qwen2.5-VL variants (2B, 3B, 7B).
    Configuration loaded from config files.
    """

    CONFIG_NAME = "qwen"  # Default, can be overridden

    def __init__(self, model_id: Optional[str] = None, precision: str = "auto", config_name: Optional[str] = None):
        # Use provided config_name or default
        self._config_name = config_name or self.CONFIG_NAME

        # Load all config from file
        self._config = get_model_config(self._config_name)

        # Use provided model_id or fall back to config
        default_model = self._config.get("model_id")
        super().__init__(model_id or default_model, precision=precision)

        # Get model type for selecting correct transformers class
        self._model_type = self._config.get("model_type", "qwen25")

        # Get dtype from config (used for auto/fp16 modes)
        dtype_str = self._config.get("dtype", "bfloat16")
        self._dtype = get_dtype(dtype_str)

        # Override dtype based on precision
        if precision == "fp16":
            self._dtype = torch.float16

    def _get_local_path(self) -> Path:
        """Get local model path, downloading if needed."""
        return download_model(self.model_id, self._config)

    def load(self) -> None:
        """Load Qwen2.5-VL model and processor."""
        if self.is_loaded:
            return

        # Get local path (downloads if needed)
        local_path = self._get_local_path()

        # Use ComfyUI's model management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Get quantization config if needed
        quant_config = get_quantization_config(self.precision)

        model_name = self._config.get("name", "Qwen-VL")
        precision_label = self.precision.upper() if self.precision != "auto" else "BF16"
        print(f"[SID-Toolkit] Loading {model_name} from {local_path} ({precision_label})")

        from transformers import AutoProcessor

        # Get the correct model class for this Qwen variant
        QwenModelClass = _get_qwen_model_class(self._model_type)

        # Load model with appropriate config
        if quant_config is not None:
            # Quantized loading - load directly to GPU
            self._model = QwenModelClass.from_pretrained(
                local_path,
                quantization_config=quant_config,
                device_map="auto",
            )
            self._quantized = True
        else:
            # Standard loading - load to CPU first
            self._model = QwenModelClass.from_pretrained(
                local_path,
                torch_dtype=self._dtype,
                device_map="cpu",
            )
            self._quantized = False

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            local_path,
            use_fast=True,
        )

        self._device = device
        self._offload_device = offload_device

        print(f"[SID-Toolkit] {model_name} loaded successfully ({precision_label})")

    def generate(
        self,
        image: Image.Image,
        mode: CaptionMode = CaptionMode.DETAILED,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate caption using Qwen2.5-VL."""
        _log(f"Generate called with mode: {mode.value}")

        if not self.is_loaded:
            _log("Model not loaded, loading now...")
            self.load()

        # Use provided config or fall back to defaults from file
        if config is None:
            gen_config = self._config.get("generation", {})
            config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 256),
                do_sample=gen_config.get("do_sample", False),
                temperature=gen_config.get("temperature", 0.7),
            )

        _log(f"Generation config: max_tokens={config.max_tokens}, do_sample={config.do_sample}, temp={config.temperature}")

        # Get prompt from config file
        prompt = get_prompt(self._config_name, mode.value)
        _log(f"Prompt: '{prompt}'")

        # Resize image for Qwen
        image = _resize_for_qwen(image.convert("RGB"))

        with isolated_execution():
            # Move model to compute device
            self._move_to_device()

            # Build messages in Qwen format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process inputs
            inputs = self._processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.target_device)

            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature if config.do_sample else None,
                )

            # Move model back to offload device
            self._move_to_offload()

            # Trim input tokens and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return output_text.strip()

    def unload(self) -> None:
        """Release model from memory."""
        self._cleanup()
