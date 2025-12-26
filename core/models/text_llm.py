"""Text-only LLM model implementation for prompt composition."""

from pathlib import Path
from typing import Optional

import torch

from .base import get_dtype, get_quantization_config
from ..config import get_model_config
from ..download import download_model
from ..log import log, log_start, log_end

# ComfyUI imports
import comfy.model_management as mm


class TextLLMModel:
    """
    Text-only LLM for prompt composition.

    Uses Qwen2.5-Instruct models (text-only, no vision).
    Much lighter than VLM models for text generation tasks.
    """

    def __init__(self, config_name: str = "qwen25_text_1.5b", precision: str = "4bit"):
        """
        Initialize text LLM.

        Args:
            config_name: Config file name (e.g., "qwen25_text_1.5b")
            precision: Precision mode ("4bit", "8bit", "fp16", "auto")
        """
        self._config_name = config_name
        self._config = get_model_config(config_name)
        self.model_id = self._config.get("model_id")
        self.precision = precision

        # Get dtype from config
        dtype_str = self._config.get("dtype", "bfloat16")
        self._dtype = get_dtype(dtype_str)

        if precision == "fp16":
            self._dtype = torch.float16

        # Model state
        self._model = None
        self._tokenizer = None
        self._device = None
        self._offload_device = None
        self._quantized = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_local_path(self) -> Path:
        """Get local model path, downloading if needed."""
        return download_model(self.model_id, self._config)

    def load(self) -> None:
        """Load model and tokenizer."""
        if self.is_loaded:
            return

        local_path = self._get_local_path()

        # Use ComfyUI's model management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Force VRAM cleanup before loading
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mm.soft_empty_cache()

        # Get quantization config
        quant_config = get_quantization_config(self.precision)

        model_name = self._config.get("name", "Qwen2.5-Text")
        precision_label = self.precision.upper() if self.precision != "auto" else "BF16"
        load_start = log_start("TextLLM", f"Loading {model_name} ({precision_label})")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
        )

        # Load model
        if quant_config is not None:
            # Quantized loading
            self._model = AutoModelForCausalLM.from_pretrained(
                local_path,
                quantization_config=quant_config,
                device_map={"": device},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self._quantized = True
        else:
            # Standard loading - load to CPU first
            self._model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=self._dtype,
                device_map="cpu",
                trust_remote_code=True,
            )
            self._quantized = False

        self._device = device
        self._offload_device = offload_device

        log_end("TextLLM", f"{model_name} loaded", load_start)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        if not self.is_loaded:
            self.load()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Move model to compute device if not quantized
        if not self._quantized:
            self._model.to(self._device)

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
        ).to(self._device if self._quantized else self._model.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Move back to offload device if not quantized
        if not self._quantized:
            self._model.to(self._offload_device)

        # Decode - skip input tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
