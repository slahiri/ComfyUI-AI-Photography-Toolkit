"""
Florence mode - VLM-based image description.

When no LLM model is provided, this mode uses Florence-2 to
generate a detailed description of the image.

Compatible with transformers >= 4.50 via class inheritance patching.
"""

import time
from typing import Any, Optional
from unittest.mock import patch

import numpy as np
from PIL import Image

from .base import BaseMode
from ..types import GeneratorResult

# Store original get_imports before patching
from transformers.dynamic_module_utils import get_imports as _original_get_imports


def _fixed_get_imports(filename):
    """Workaround for unnecessary flash_attn requirement in Florence-2."""
    imports = _original_get_imports(filename)
    if str(filename).endswith("modeling_florence2.py"):
        try:
            imports.remove("flash_attn")
        except ValueError:
            pass
    return imports


class FlorenceMode(BaseMode):
    """
    Florence mode implementation.

    Uses Florence-2 VLM to generate detailed image descriptions
    when no external LLM is available.
    """

    # Default Florence model configuration
    DEFAULT_MODEL_ID = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
    TASK = "<MORE_DETAILED_CAPTION>"

    # Singleton model instance - keyed by model_id
    _models = {}  # model_id -> model
    _processors = {}  # model_id -> processor
    _current_model_id = None
    _device = None

    @property
    def name(self) -> str:
        return "florence"

    @property
    def requires_llm(self) -> bool:
        return False

    @property
    def requires_taggers(self) -> bool:
        return False

    def execute(
        self,
        image: Any = None,
        model_id: str = None,
        hf_token: str = "",
        **kwargs
    ) -> GeneratorResult:
        """
        Execute Florence mode.

        Args:
            image: Image tensor from ComfyUI
            model_id: HuggingFace model ID to use (defaults to DEFAULT_MODEL_ID)
            hf_token: HuggingFace token for model access
            **kwargs: Ignored

        Returns:
            GeneratorResult with Florence-generated description
        """
        if image is None:
            return GeneratorResult(
                prompt="[No image provided]",
                metadata={
                    "mode": "florence",
                    "error": "No image provided",
                }
            )

        # Use provided model_id or default
        model_id = model_id or self.DEFAULT_MODEL_ID
        start_time = time.time()

        try:
            # Convert ComfyUI image tensor to PIL
            pil_image = self._tensor_to_pil(image)

            # Load model if not already loaded (or different model requested)
            self._ensure_model_loaded(model_id=model_id, hf_token=hf_token)

            # Run Florence caption
            caption = self._run_caption(model_id, pil_image)

            inference_time = int((time.time() - start_time) * 1000)

            return GeneratorResult(
                prompt=caption,
                metadata={
                    "mode": "florence",
                    "source": "florence_description",
                    "model": model_id,
                    "task": self.TASK,
                    "processing": {
                        "taggers_executed": False,
                        "florence_executed": True,
                        "llm_executed": False,
                    },
                    "timing": {
                        "florence_ms": inference_time,
                    }
                }
            )

        except Exception as e:
            error_msg = str(e)
            print(f"[Florence] Error: {error_msg}")
            import traceback
            traceback.print_exc()

            return GeneratorResult(
                prompt=f"[Florence error: {error_msg}]",
                metadata={
                    "mode": "florence",
                    "model": model_id,
                    "error": error_msg,
                    "processing": {
                        "taggers_executed": False,
                        "florence_executed": False,
                        "llm_executed": False,
                    }
                }
            )

    def _tensor_to_pil(self, image_tensor: Any) -> Image.Image:
        """Convert ComfyUI image tensor to PIL Image."""
        # ComfyUI images are [batch, height, width, channels] tensors
        if hasattr(image_tensor, 'cpu'):
            img_np = image_tensor[0].cpu().numpy()
        else:
            img_np = np.array(image_tensor[0])

        # Convert from 0-1 float to 0-255 uint8
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        return Image.fromarray(img_np)

    def _ensure_model_loaded(self, model_id: str, hf_token: str = ""):
        """Load Florence model if not already loaded or different model requested."""
        # Check if we already have this model loaded AND it has generate method
        if model_id in FlorenceMode._models and FlorenceMode._current_model_id == model_id:
            model = FlorenceMode._models[model_id]
            if hasattr(model, 'generate') and callable(getattr(model, 'generate', None)):
                return
            else:
                # Model is cached but broken, reload it
                print(f"[Florence] Cached model missing generate(), reloading...")
                self._unload_current_model()

        # If switching models, unload current to save VRAM
        if FlorenceMode._current_model_id and FlorenceMode._current_model_id != model_id:
            print(f"[Florence] Switching from {FlorenceMode._current_model_id} to {model_id}")
            self._unload_current_model()

        print(f"[Florence] Loading model: {model_id}")
        start_time = time.time()

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            # Determine device
            if torch.cuda.is_available():
                FlorenceMode._device = "cuda"
                dtype = torch.float16
            else:
                FlorenceMode._device = "cpu"
                dtype = torch.float32

            # Prepare token argument
            token_arg = hf_token if hf_token and hf_token.strip() else None

            # Load the model
            with patch(
                "transformers.dynamic_module_utils.get_imports",
                _fixed_get_imports
            ):
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    token=token_arg,
                    attn_implementation="eager",
                ).to(FlorenceMode._device)

            # Patch the actual model class AFTER loading
            # This is needed because trust_remote_code may load fresh classes
            from transformers import GenerationMixin, GenerationConfig

            def _add_generation_mixin(obj, main_model, name=""):
                """Add GenerationMixin to an object's class and set up generation_config."""
                cls = type(obj)
                patched = False

                if GenerationMixin not in cls.__mro__:
                    try:
                        cls.__bases__ = (GenerationMixin,) + cls.__bases__
                        print(f"[Florence] Patched {name} class: {cls.__name__}")
                        patched = True
                    except TypeError:
                        # Can't modify __bases__, try creating new class
                        new_cls = type(cls.__name__, (GenerationMixin,) + cls.__bases__, dict(cls.__dict__))
                        obj.__class__ = new_cls
                        print(f"[Florence] Replaced {name} class: {cls.__name__}")
                        patched = True

                # Ensure generation_config exists (required by GenerationMixin.generate)
                if not hasattr(obj, 'generation_config') or obj.generation_config is None:
                    # Build generation_config from model config
                    config = getattr(obj, 'config', None) or getattr(main_model, 'config', None)
                    gen_kwargs = {}

                    if config:
                        for key in ['bos_token_id', 'eos_token_id', 'pad_token_id',
                                    'decoder_start_token_id', 'vocab_size']:
                            val = getattr(config, key, None)
                            if val is not None:
                                gen_kwargs[key] = val

                    obj.generation_config = GenerationConfig(**gen_kwargs)
                    print(f"[Florence] Added generation_config to {name}")

                return patched

            # Patch the main model
            _add_generation_mixin(model, model, "model")

            # Patch language_model if it exists (this is where generate is called)
            if hasattr(model, 'language_model'):
                _add_generation_mixin(model.language_model, model, "language_model")

            # Step 5: Verify model has generate method
            if not hasattr(model, 'generate'):
                raise RuntimeError("Model loaded but missing generate() method")

            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=token_arg,
            )

            # Store in cache
            FlorenceMode._models[model_id] = model
            FlorenceMode._processors[model_id] = processor
            FlorenceMode._current_model_id = model_id

            load_time = time.time() - start_time
            print(f"[Florence] Model loaded in {load_time:.1f}s on {FlorenceMode._device}")

        except Exception as e:
            print(f"[Florence] Failed to load model: {e}")
            raise

    def _unload_current_model(self):
        """Unload current model to free memory."""
        if FlorenceMode._current_model_id:
            model_id = FlorenceMode._current_model_id
            if model_id in FlorenceMode._models:
                del FlorenceMode._models[model_id]
            if model_id in FlorenceMode._processors:
                del FlorenceMode._processors[model_id]
            FlorenceMode._current_model_id = None

            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def _run_caption(self, model_id: str, pil_image: Image.Image) -> str:
        """Run Florence caption on image."""
        import torch

        model = FlorenceMode._models[model_id]
        processor = FlorenceMode._processors[model_id]

        # Get model dtype for consistency
        model_dtype = next(model.parameters()).dtype

        # Prepare inputs
        inputs = processor(
            text=self.TASK,
            images=pil_image,
            return_tensors="pt",
        )

        # Move to device and convert to model dtype
        inputs = {
            k: v.to(device=FlorenceMode._device, dtype=model_dtype) if v.dtype.is_floating_point else v.to(FlorenceMode._device)
            for k, v in inputs.items()
        }

        # Generate
        print(f"[Florence] Generating with input_ids shape: {inputs['input_ids'].shape}, pixel_values shape: {inputs['pixel_values'].shape}")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                early_stopping=False,
                use_cache=False,
            )
        print(f"[Florence] Generated ids shape: {generated_ids.shape}")

        # Decode
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        print(f"[Florence] Raw output length: {len(generated_ids[0])} tokens")
        print(f"[Florence] Decoded text: '{generated_text[:200]}...' (len={len(generated_text)})")

        # Post-process: remove task prefix if present
        caption = generated_text.replace(self.TASK, "").strip()

        if not caption:
            print(f"[Florence] WARNING: Empty caption after stripping task prefix")

        return caption

    @classmethod
    def unload_model(cls):
        """Unload all Florence models to free memory."""
        for model_id in list(cls._models.keys()):
            del cls._models[model_id]
        for model_id in list(cls._processors.keys()):
            del cls._processors[model_id]

        cls._models = {}
        cls._processors = {}
        cls._current_model_id = None

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        print("[Florence] All models unloaded")
