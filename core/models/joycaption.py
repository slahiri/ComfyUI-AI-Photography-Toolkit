"""JoyCaption Beta One model implementation."""

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


class JoyCaptionModel(BaseCaptionModel):
    """
    JoyCaption model for image captioning.

    Based on LLaVA architecture with Llama language model.
    Configuration loaded from config files.
    """

    CONFIG_NAME = "joycaption"

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
        """Load JoyCaption model and processor."""
        if self.is_loaded:
            return

        # Get local path (downloads if needed)
        local_path = self._get_local_path()

        # Use ComfyUI's model management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Get quantization config if needed
        quant_config = get_quantization_config(self.precision)

        precision_label = self.precision.upper() if self.precision != "auto" else "BF16"
        print(f"[SID-Toolkit] Loading JoyCaption Beta One from {local_path} ({precision_label})")

        from transformers import LlavaForConditionalGeneration, AutoProcessor

        # Load model with appropriate config
        if quant_config is not None:
            # Quantized loading - load directly to GPU
            self._model = LlavaForConditionalGeneration.from_pretrained(
                local_path,
                quantization_config=quant_config,
                device_map="auto",
            )
            self._quantized = True
        else:
            # Standard loading - load to CPU first
            self._model = LlavaForConditionalGeneration.from_pretrained(
                local_path,
                torch_dtype=self._dtype,
                device_map="cpu",
            )
            self._quantized = False

        # Load processor
        self._processor = AutoProcessor.from_pretrained(local_path)

        self._device = device
        self._offload_device = offload_device

        print(f"[SID-Toolkit] JoyCaption Beta One loaded successfully ({precision_label})")

    def generate(
        self,
        image: Image.Image,
        mode: CaptionMode = CaptionMode.DETAILED,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate caption using JoyCaption."""
        if not self.is_loaded:
            self.load()

        # Use provided config or fall back to defaults from file
        if config is None:
            gen_config = self._config.get("generation", {})
            config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 256),
                do_sample=gen_config.get("do_sample", True),
                temperature=gen_config.get("temperature", 0.6),
            )

        # Get prompt from config file
        prompt = get_prompt(self._config_name, mode.value)

        # Ensure RGB
        image = image.convert("RGB")

        with isolated_execution():
            # Move model to compute device
            self._move_to_device()

            # Build conversation in LLaVA format
            conversation = [
                {
                    "role": "user",
                    "content": f"<image>\n{prompt}",
                },
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process inputs
            inputs = self._processor(
                text=text,
                images=image,
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
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]

            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return output_text.strip()

    def unload(self) -> None:
        """Release model from memory."""
        self._cleanup()
