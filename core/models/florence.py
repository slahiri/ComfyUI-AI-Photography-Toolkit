"""Florence-2 PromptGen model implementation."""

import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_imports

from .base import BaseCaptionModel, CaptionMode, GenerationConfig
from ..config import get_model_config, get_prompt
from ..platform import cleanup_memory, isolated_execution

# ComfyUI imports
import comfy.model_management as mm
import folder_paths

_MODELS_DIR = Path(folder_paths.models_dir) / "LLM"

# Check if we need to use bundled Florence model (transformers >= 4.51.0)
_USE_BUNDLED_FLORENCE = transformers.__version__ >= "4.51.0"


# Workaround for unnecessary flash_attn requirement in Florence-2
def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py"):
        try:
            imports.remove("flash_attn")
        except ValueError:
            pass  # flash_attn not in imports
    return imports


def _get_florence_class():
    """Get Florence2 model class from kijai's bundled implementation."""
    import importlib.util

    kijai_path = Path(folder_paths.base_path) / "custom_nodes" / "comfyui-florence2"
    package_name = "comfyui_florence2_bundled"

    # Load configuration module first (required by modeling module)
    config_spec = importlib.util.spec_from_file_location(
        f"{package_name}.configuration_florence2",
        kijai_path / "configuration_florence2.py",
        submodule_search_locations=[str(kijai_path)]
    )
    config_module = importlib.util.module_from_spec(config_spec)
    config_module.__package__ = package_name
    sys.modules[f"{package_name}.configuration_florence2"] = config_module
    # Also register without package prefix for relative imports
    sys.modules["configuration_florence2"] = config_module
    config_spec.loader.exec_module(config_module)

    # Load modeling module with proper package context
    model_spec = importlib.util.spec_from_file_location(
        f"{package_name}.modeling_florence2",
        kijai_path / "modeling_florence2.py",
        submodule_search_locations=[str(kijai_path)]
    )
    model_module = importlib.util.module_from_spec(model_spec)
    model_module.__package__ = package_name
    sys.modules[f"{package_name}.modeling_florence2"] = model_module
    model_spec.loader.exec_module(model_module)

    return model_module.Florence2ForConditionalGeneration


def _load_florence_model(local_path: Path, dtype: torch.dtype, trust_remote: bool):
    """Load Florence model with appropriate method based on transformers version."""
    if _USE_BUNDLED_FLORENCE:
        # Use kijai's bundled Florence2 implementation for newer transformers
        # This version properly inherits from GenerationMixin
        try:
            kijai_path = Path(folder_paths.base_path) / "custom_nodes" / "comfyui-florence2"
            if kijai_path.exists():
                Florence2ForConditionalGeneration = _get_florence_class()

                print(f"[SID-Toolkit] Using bundled Florence2 (transformers {transformers.__version__})")
                return Florence2ForConditionalGeneration.from_pretrained(
                    local_path,
                    torch_dtype=dtype,
                    trust_remote_code=trust_remote,
                )
        except Exception as e:
            print(f"[SID-Toolkit] Bundled Florence2 failed: {e}, falling back to AutoModel")

    # Fallback: Use AutoModelForCausalLM with flash_attn workaround
    from transformers import AutoModelForCausalLM

    print(f"[SID-Toolkit] Using AutoModelForCausalLM (transformers {transformers.__version__})")
    with patch(
        "transformers.dynamic_module_utils.get_imports",
        _fixed_get_imports
    ):
        return AutoModelForCausalLM.from_pretrained(
            local_path,
            attn_implementation="eager",
            torch_dtype=dtype,
            trust_remote_code=trust_remote,
        )


def _get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


class FlorenceModel(BaseCaptionModel):
    """
    Florence-2 PromptGen model for image captioning.

    Optimized for generating prompts for image generation models.
    All configuration loaded from config/florence.json
    """

    CONFIG_NAME = "florence"

    def __init__(self, model_id: Optional[str] = None):
        # Load all config from file
        self._config = get_model_config(self.CONFIG_NAME)

        # Use provided model_id or fall back to config
        default_model = self._config.get("model_id")
        super().__init__(model_id or default_model)

        # Get dtype from config
        dtype_str = self._config.get("dtype", "float16")
        self._dtype = _get_dtype(dtype_str)

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
        """Load Florence-2 model and processor."""
        if self.is_loaded:
            return

        # Get local path (downloads if needed)
        local_path = self._get_local_path()

        # Use ComfyUI's model management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Get model loading config
        trust_remote = self._config.get("trust_remote_code", True)

        print(f"[SID-Toolkit] Loading Florence-2 (transformers {transformers.__version__})")

        # Load model with appropriate method based on transformers version
        self._model = _load_florence_model(
            local_path, self._dtype, trust_remote
        ).to(offload_device)

        self._processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=trust_remote,
        )

        self._device = device
        self._offload_device = offload_device

    def generate(
        self,
        image: Image.Image,
        mode: CaptionMode = CaptionMode.DETAILED,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate caption using Florence-2."""
        if not self.is_loaded:
            self.load()

        # Get generation config from file or use provided
        gen_config = self._config.get("generation", {})
        config = config or GenerationConfig(
            max_tokens=gen_config.get("max_tokens", 1024),
            num_beams=gen_config.get("num_beams", 4),
            do_sample=gen_config.get("do_sample", False),
        )

        # Get prompt from config file
        prompt = get_prompt(self.CONFIG_NAME, mode.value)

        with isolated_execution():
            # Move model to compute device
            self._model.to(self._device)

            # Prepare inputs
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                do_rescale=False,
            ).to(self._dtype).to(self._device)

            # Generate (use_cache=False required for bundled Florence2 model)
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=config.max_tokens,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                early_stopping=gen_config.get("early_stopping", False),
                use_cache=False,
            )

            # Move model back to offload device
            self._model.to(self._offload_device)

            # Decode
            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]

            # Post-process
            result = self._processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height),
            )

        return result.get(prompt, "")

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        cleanup_memory()
