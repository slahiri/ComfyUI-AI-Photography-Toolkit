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

from .base import BaseCaptionModel, CaptionMode, GenerationConfig, get_dtype, get_quantization_config
from ..config import get_model_config, get_prompt
from ..download import download_model
from ..platform import isolated_execution

# ComfyUI imports
import comfy.model_management as mm
import folder_paths

# Global verbose flag
_verbose = False


def set_verbose(verbose: bool) -> None:
    """Set verbose logging flag."""
    global _verbose
    _verbose = verbose


def _log(message: str) -> None:
    """Print message if verbose is enabled."""
    if _verbose:
        print(f"[SID-Florence] {message}")

_MODELS_DIR = Path(folder_paths.models_dir) / "LLM"

# Check if we need to use bundled Florence model (transformers >= 4.50.0)
# After 4.50, PreTrainedModel no longer inherits from GenerationMixin
def _parse_version(version_str: str) -> tuple:
    """Parse version string to tuple for comparison."""
    try:
        parts = version_str.split(".")[:3]
        return tuple(int(p.split("+")[0].split("a")[0].split("b")[0].split("rc")[0]) for p in parts)
    except (ValueError, IndexError):
        return (0, 0, 0)

_USE_BUNDLED_FLORENCE = _parse_version(transformers.__version__) >= (4, 50, 0)


# Workaround for unnecessary flash_attn requirement in Florence-2
def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py"):
        try:
            imports.remove("flash_attn")
        except ValueError:
            pass  # flash_attn not in imports
    return imports


def _setup_florence_modules():
    """
    Set up Florence2 modules from kijai's bundled implementation.
    Also registers a mock transformers.models.florence2 to satisfy dynamic imports.
    """
    import importlib.util
    from types import ModuleType

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

    # Create mock transformers.models.florence2 module to satisfy dynamic imports
    # This is needed because AutoProcessor tries to import from transformers.models.florence2
    if "transformers.models.florence2" not in sys.modules:
        mock_florence2 = ModuleType("transformers.models.florence2")
        mock_florence2.__file__ = str(kijai_path / "modeling_florence2.py")
        mock_florence2.__path__ = [str(kijai_path)]

        # Copy all exports from kijai's modules
        for name in dir(config_module):
            if not name.startswith("_"):
                setattr(mock_florence2, name, getattr(config_module, name))
        for name in dir(model_module):
            if not name.startswith("_"):
                setattr(mock_florence2, name, getattr(model_module, name))

        # Register the mock module
        sys.modules["transformers.models.florence2"] = mock_florence2

        # Also ensure transformers.models knows about it
        import transformers.models
        if not hasattr(transformers.models, "florence2"):
            transformers.models.florence2 = mock_florence2

    return model_module.Florence2ForConditionalGeneration


def _get_florence_class():
    """Get Florence2 model class from kijai's bundled implementation."""
    return _setup_florence_modules()


def _load_florence_model(local_path: Path, dtype: torch.dtype, trust_remote: bool):
    """Load Florence model - requires kijai's comfyui-florence2 for transformers >= 4.50."""

    # For transformers >= 4.50, we MUST use kijai's bundled implementation
    # because the model's bundled code doesn't inherit from GenerationMixin
    if _USE_BUNDLED_FLORENCE:
        kijai_path = Path(folder_paths.base_path) / "custom_nodes" / "comfyui-florence2"
        if not kijai_path.exists():
            raise RuntimeError(
                "Florence-2 requires comfyui-florence2 for transformers >= 4.50. "
                "Please install it from: https://github.com/kijai/ComfyUI-Florence2 "
                "(git clone into custom_nodes folder) OR downgrade transformers: "
                "pip install transformers==4.44.2"
            )

        Florence2ForConditionalGeneration = _get_florence_class()
        print(f"[SID-Toolkit] Using kijai's Florence2 (transformers {transformers.__version__})")

        with patch(
            "transformers.dynamic_module_utils.get_imports",
            _fixed_get_imports
        ):
            return Florence2ForConditionalGeneration.from_pretrained(
                local_path,
                torch_dtype=dtype,
                trust_remote_code=trust_remote,
            )

    # For older transformers, use AutoModelForCausalLM
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


class FlorenceModel(BaseCaptionModel):
    """
    Florence-2 PromptGen model for image captioning.

    Optimized for generating prompts for image generation models.
    Configuration loaded from config files.
    """

    CONFIG_NAME = "florence"

    def __init__(self, model_id: Optional[str] = None, precision: str = "auto", config_name: Optional[str] = None):
        # Use provided config_name or default
        self._config_name = config_name or self.CONFIG_NAME

        # Load all config from file
        self._config = get_model_config(self._config_name)

        # Use provided model_id or fall back to config
        default_model = self._config.get("model_id")
        super().__init__(model_id or default_model, precision=precision)

        # Get dtype from config (used for auto/fp16 modes)
        dtype_str = self._config.get("dtype", "float16")
        self._dtype = get_dtype(dtype_str)

        # Override dtype based on precision
        if precision == "fp16":
            self._dtype = torch.float16

    def _get_local_path(self) -> Path:
        """Get local model path, downloading if needed."""
        return download_model(self.model_id, self._config)

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

        # Get quantization config if needed
        quant_config = get_quantization_config(self.precision)

        precision_label = self.precision.upper() if self.precision != "auto" else "FP16"
        print(f"[SID-Toolkit] Loading Florence-2 ({precision_label}, transformers {transformers.__version__})")

        # Load model - Florence-2 doesn't support quantization well due to custom model code
        if quant_config is not None:
            print(f"[SID-Toolkit] Warning: Florence-2 doesn't support {self.precision} quantization, using FP16")

        # Standard loading for Florence-2 (already small ~1.5GB model)
        self._model = _load_florence_model(
            local_path, self._dtype, trust_remote
        ).to(offload_device)
        self._quantized = False

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=trust_remote,
        )

        self._device = device
        self._offload_device = offload_device

        print(f"[SID-Toolkit] Florence-2 loaded successfully ({precision_label})")

    def generate(
        self,
        image: Image.Image,
        mode: CaptionMode = CaptionMode.DETAILED,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate caption using Florence-2."""
        _log(f"Generate called with mode: {mode.value}")

        if not self.is_loaded:
            _log("Model not loaded, loading now...")
            self.load()

        # Use provided config or fall back to defaults from file
        if config is None:
            gen_config = self._config.get("generation", {})
            config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 1024),
                num_beams=gen_config.get("num_beams", 4),
                do_sample=gen_config.get("do_sample", False),
            )

        _log(f"Generation config: max_tokens={config.max_tokens}, num_beams={config.num_beams}, do_sample={config.do_sample}")

        # Get prompt from config file
        prompt = get_prompt(self._config_name, mode.value)
        _log(f"Prompt/Task: '{prompt}'")

        with isolated_execution():
            # Move model to compute device
            self._move_to_device()

            # Prepare inputs
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                do_rescale=False,
            ).to(self._dtype).to(self.target_device)

            # Generate (use_cache=False required for bundled Florence2 model)
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=config.max_tokens,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                temperature=config.temperature if config.do_sample else None,
                early_stopping=False,
                use_cache=False,
            )

            # Move model back to offload device
            self._move_to_offload()

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
        self._cleanup()
