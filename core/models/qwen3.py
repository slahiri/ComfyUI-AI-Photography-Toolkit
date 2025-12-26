"""Qwen3-VL model implementation."""

import math
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .base import BaseCaptionModel, CaptionMode, GenerationConfig, get_dtype, get_quantization_config
from ..config import get_model_config, get_prompt
from ..download import download_model
from ..platform import isolated_execution
from ..log import log, log_start, log_end

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
        log("Qwen3", message)


def _resize_for_qwen3(image: Image.Image, max_pixels: int = 1003520) -> Image.Image:
    """
    Resize image to fit within max_pixels while maintaining aspect ratio.
    Qwen3 requires dimensions divisible by 28.
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


class Qwen3VLModel(BaseCaptionModel):
    """
    Qwen3-VL model for image captioning.

    Supports Qwen3-VL variants (2B, 4B, 8B).
    Configuration loaded from config files.
    """

    CONFIG_NAME = "qwen3_2b_abliterated"  # Default

    def __init__(self, model_id: Optional[str] = None, precision: str = "auto", config_name: Optional[str] = None):
        # Use provided config_name or default
        self._config_name = config_name or self.CONFIG_NAME

        # Load all config from file
        self._config = get_model_config(self._config_name)

        # Use provided model_id or fall back to config
        default_model = self._config.get("model_id")
        super().__init__(model_id or default_model, precision=precision)

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
        """Load Qwen3-VL model and processor."""
        if self.is_loaded:
            return

        # Get local path (downloads if needed)
        local_path = self._get_local_path()

        # Use ComfyUI's model management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Force aggressive VRAM cleanup before loading
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Soft unload other ComfyUI models to free VRAM
            mm.soft_empty_cache()

        # Get quantization config if needed
        quant_config = get_quantization_config(self.precision)

        model_name = self._config.get("name", "Qwen3-VL")
        precision_label = self.precision.upper() if self.precision != "auto" else "BF16"
        load_start = log_start("Qwen3", f"Loading {model_name} ({precision_label})")

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # Load model with appropriate config
        if quant_config is not None:
            # Quantized loading - force all on GPU (no CPU offload)
            # Use device_map with explicit GPU to avoid CPU/GPU split issues
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_path,
                quantization_config=quant_config,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
            self._quantized = True
        else:
            # Standard loading - load to CPU first
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
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

        log_end("Qwen3", f"{model_name} loaded", load_start)

    def generate(
        self,
        image: Image.Image,
        mode: CaptionMode = CaptionMode.DETAILED,
        config: Optional[GenerationConfig] = None,
        custom_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate caption using Qwen3-VL.

        Args:
            image: PIL Image to caption
            mode: Caption generation mode (used if custom_prompt not provided)
            config: Optional generation configuration
            custom_prompt: Optional custom prompt (overrides mode-based prompt)
            system_prompt: Optional system prompt for setting model behavior
        """
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

        # Use custom prompt or get from config file
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = get_prompt(self._config_name, mode.value)
        _log(f"Prompt: '{prompt[:100]}...'")

        # Resize image for Qwen3
        image = _resize_for_qwen3(image.convert("RGB"))

        with isolated_execution():
            # Move model to compute device
            self._move_to_device()

            # Build messages in Qwen3 format
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })

            # Add user message with image and prompt
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            })

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

            # Generate with full sampling parameters
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": config.max_tokens,
                    "do_sample": config.do_sample,
                }
                if config.do_sample:
                    gen_kwargs["temperature"] = config.temperature
                    gen_kwargs["top_p"] = config.top_p
                    if config.top_k is not None:
                        gen_kwargs["top_k"] = config.top_k

                generated_ids = self._model.generate(**inputs, **gen_kwargs)

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

    def generate_text_only(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text-only response (no image).

        Used for synthesis tasks where we want to use the LLM
        capabilities of the VLM without vision input.

        Args:
            prompt: Text prompt
            config: Optional generation configuration

        Returns:
            Generated text
        """
        _log(f"Generate text-only called")

        if not self.is_loaded:
            _log("Model not loaded, loading now...")
            self.load()

        # Use provided config or fall back to defaults
        if config is None:
            gen_config = self._config.get("generation", {})
            config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 256),
                do_sample=gen_config.get("do_sample", False),
                temperature=gen_config.get("temperature", 0.7),
            )

        _log(f"Generation config: max_tokens={config.max_tokens}, do_sample={config.do_sample}, temp={config.temperature}")

        with isolated_execution():
            # Move model to compute device
            self._move_to_device()

            # Build messages in Qwen3 format (text-only, no image)
            messages = [
                {
                    "role": "user",
                    "content": [
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

            # Process inputs (no images)
            inputs = self._processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(self.target_device)

            # Generate with full sampling parameters
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": config.max_tokens,
                    "do_sample": config.do_sample,
                }
                if config.do_sample:
                    gen_kwargs["temperature"] = config.temperature
                    gen_kwargs["top_p"] = config.top_p
                    if config.top_k is not None:
                        gen_kwargs["top_k"] = config.top_k

                generated_ids = self._model.generate(**inputs, **gen_kwargs)

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
