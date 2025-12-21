"""MiniCPM-V model implementation."""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .base import BaseCaptionModel, CaptionMode, GenerationConfig, get_dtype, get_quantization_config
from ..config import get_model_config, get_prompt
from ..download import download_model
from ..platform import isolated_execution, cleanup_memory

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
        print(f"[SID-MiniCPM] {message}")


class MiniCPMVModel(BaseCaptionModel):
    """
    MiniCPM-V model for image captioning.

    High-performance multimodal model from OpenBMB.
    Configuration loaded from config files.
    """

    CONFIG_NAME = "minicpm"

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

        # Store tokenizer separately
        self._tokenizer = None

    def _get_local_path(self) -> Path:
        """Get local model path, downloading if needed."""
        return download_model(self.model_id, self._config)

    def load(self) -> None:
        """Load MiniCPM-V model and tokenizer."""
        if self.is_loaded:
            _log("Model already loaded, skipping")
            return

        _log(f"Loading model: {self.model_id}")
        _log(f"Precision: {self.precision}, dtype: {self._dtype}")

        # Clean up memory before loading large model
        cleanup_memory(aggressive=True)

        # Get local path (downloads if needed)
        local_path = self._get_local_path()
        _log(f"Model path: {local_path}")

        # Use ComfyUI's model management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        _log(f"Device: {device}, Offload device: {offload_device}")

        # Get quantization config if needed
        quant_config = get_quantization_config(self.precision)
        _log(f"Quantization config: {quant_config}")

        model_name = self._config.get("name", "MiniCPM-V")
        precision_label = self.precision.upper() if self.precision != "auto" else "BF16"
        print(f"[SID-Toolkit] Loading {model_name} from {local_path} ({precision_label})")

        from transformers import AutoModel, AutoTokenizer

        # Load model with appropriate config
        # MiniCPM-V requires loading directly to GPU to avoid CUDA allocator issues
        if quant_config is not None:
            # Quantized loading - load directly to GPU
            _log("Loading with quantization config...")
            self._model = AutoModel.from_pretrained(
                local_path,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self._quantized = True
        else:
            # MiniCPM-V works best when loaded directly to GPU
            # Loading to CPU then moving causes CUDA allocator issues
            _log("Loading directly to GPU...")
            self._model = AutoModel.from_pretrained(
                local_path,
                torch_dtype=self._dtype,
                device_map="cuda",
                trust_remote_code=True,
            )
            self._quantized = False

        # Set to eval mode
        self._model.eval()

        # Load tokenizer
        _log("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
        )

        self._device = device
        self._offload_device = offload_device

        # Sync CUDA to ensure model is fully loaded
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(f"[SID-Toolkit] {model_name} loaded successfully ({precision_label})")

    def generate(
        self,
        image: Image.Image,
        mode: CaptionMode = CaptionMode.DETAILED,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate caption using MiniCPM-V."""
        _log(f"Generate called with mode: {mode.value}")

        if not self.is_loaded:
            _log("Model not loaded, loading now...")
            self.load()

        # Use provided config or fall back to defaults from file
        if config is None:
            gen_config = self._config.get("generation", {})
            config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 512),
                do_sample=gen_config.get("do_sample", True),
                temperature=gen_config.get("temperature", 0.7),
            )

        _log(f"Generation config: max_tokens={config.max_tokens}, do_sample={config.do_sample}, temp={config.temperature}")

        # Get prompt from config file
        prompt = get_prompt(self._config_name, mode.value)
        _log(f"Prompt: '{prompt}'")

        # Ensure RGB
        image = image.convert("RGB")
        _log(f"Image size: {image.size}, mode: {image.mode}")

        # Clear cache before inference to avoid allocator issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with isolated_execution():
            # MiniCPM-V is already on GPU (device_map="cuda"), no need to move
            # Build message in MiniCPM-V format
            msgs = [{"role": "user", "content": [image, prompt]}]
            _log("Starting generation...")

            # Generate using the model's chat method
            with torch.no_grad():
                output_text = self._model.chat(
                    image=None,  # Image is passed in msgs
                    msgs=msgs,
                    tokenizer=self._tokenizer,
                    max_new_tokens=config.max_tokens,
                    sampling=config.do_sample,
                    temperature=config.temperature if config.do_sample else None,
                )

            _log(f"Generation complete, output length: {len(output_text)}")

        return output_text.strip()

    def unload(self) -> None:
        """Release model from memory."""
        _log("Unloading model...")
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # MiniCPM is loaded with device_map, need special handling
        if self._model is not None:
            # Clear any cached states
            if hasattr(self._model, 'reset'):
                try:
                    self._model.reset()
                except Exception:
                    pass
            del self._model
            self._model = None

        self._quantized = False
        cleanup_memory(aggressive=True)
        _log("Model unloaded")
