"""MiniCPM-V model implementation."""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .base import BaseCaptionModel, CaptionMode, GenerationConfig, get_dtype, get_quantization_config
from ..config import get_model_config, get_prompt
from ..platform import isolated_execution

# ComfyUI imports
import comfy.model_management as mm
import folder_paths


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
        download_config = self._config.get("download", {})
        subfolder = download_config.get("subfolder", "LLM")
        use_symlinks = download_config.get("use_symlinks", False)

        model_name = self.model_id.split("/")[-1]
        local_path = Path(folder_paths.models_dir) / subfolder / model_name

        if local_path.exists():
            return local_path

        print(f"[SID-Toolkit] Downloading {self.model_id} to {local_path}...")

        from huggingface_hub import snapshot_download

        local_path.parent.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.model_id,
            local_dir=local_path,
            local_dir_use_symlinks=use_symlinks,
        )

        print(f"[SID-Toolkit] Download complete: {model_name}")
        return local_path

    def load(self) -> None:
        """Load MiniCPM-V model and tokenizer."""
        if self.is_loaded:
            return

        # Get local path (downloads if needed)
        local_path = self._get_local_path()

        # Use ComfyUI's model management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Get quantization config if needed
        quant_config = get_quantization_config(self.precision)

        model_name = self._config.get("name", "MiniCPM-V")
        precision_label = self.precision.upper() if self.precision != "auto" else "BF16"
        print(f"[SID-Toolkit] Loading {model_name} from {local_path} ({precision_label})")

        from transformers import AutoModel, AutoTokenizer

        # Load model with appropriate config
        if quant_config is not None:
            # Quantized loading - load directly to GPU
            self._model = AutoModel.from_pretrained(
                local_path,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self._quantized = True
        else:
            # Standard loading - load to CPU first
            self._model = AutoModel.from_pretrained(
                local_path,
                torch_dtype=self._dtype,
                device_map="cpu",
                trust_remote_code=True,
            )
            self._quantized = False

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
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
        """Generate caption using MiniCPM-V."""
        if not self.is_loaded:
            self.load()

        # Use provided config or fall back to defaults from file
        if config is None:
            gen_config = self._config.get("generation", {})
            config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 512),
                do_sample=gen_config.get("do_sample", True),
                temperature=gen_config.get("temperature", 0.7),
            )

        # Get prompt from config file
        prompt = get_prompt(self._config_name, mode.value)

        # Ensure RGB
        image = image.convert("RGB")

        with isolated_execution():
            # Move model to compute device
            self._move_to_device()

            # Build message in MiniCPM-V format
            msgs = [{"role": "user", "content": [image, prompt]}]

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

            # Move model back to offload device
            self._move_to_offload()

        return output_text.strip()

    def unload(self) -> None:
        """Release model from memory."""
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._cleanup()
