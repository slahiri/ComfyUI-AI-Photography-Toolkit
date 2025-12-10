"""
SID_QwenVL_LLM Node

QwenVL vision-language model provider using HuggingFace transformers.
No llama-cpp-python required - uses transformers with BitsAndBytes quantization.
Supports Qwen2.5-VL and Qwen3-VL models.
"""

import os
import gc
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from comfy_api.latest import io as comfy_io
import folder_paths

from .llm_model_type import LLMModelConfig
from .base_llm_provider import BaseLLMProvider

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")

# Model download directory
QWENVL_DIR = os.path.join(folder_paths.models_dir, "LLM", "QwenVL")
os.makedirs(QWENVL_DIR, exist_ok=True)


@dataclass
class QwenVLModelInfo:
    """Information about a QwenVL model."""
    name: str
    repo_id: str
    vram_fp16: float  # GB
    vram_8bit: float  # GB
    vram_4bit: float  # GB
    is_quantized: bool = False  # Pre-quantized FP8 models
    max_output_tokens: int = 4096
    description: str = ""


# Available QwenVL models
QWENVL_MODELS: Dict[str, QwenVLModelInfo] = {
    # Qwen3-VL Series (Latest)
    "Qwen3-VL-2B-Instruct": QwenVLModelInfo(
        name="Qwen3-VL 2B (4GB VRAM) - Fast",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        description="Fast, good for 8GB GPUs"
    ),
    "Qwen3-VL-4B-Instruct": QwenVLModelInfo(
        name="Qwen3-VL 4B (6GB VRAM) - Recommended",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        description="Best balance of speed and quality"
    ),
    "Qwen3-VL-8B-Instruct": QwenVLModelInfo(
        name="Qwen3-VL 8B (12GB VRAM) - High Quality",
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        vram_fp16=12.0, vram_8bit=7.0, vram_4bit=4.5,
        description="High quality, needs more VRAM"
    ),
    # Qwen2.5-VL Series
    "Qwen2.5-VL-3B-Instruct": QwenVLModelInfo(
        name="Qwen2.5-VL 3B (6GB VRAM)",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        description="Stable, well-tested"
    ),
    "Qwen2.5-VL-7B-Instruct": QwenVLModelInfo(
        name="Qwen2.5-VL 7B (15GB VRAM) - Best Quality",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        vram_fp16=15.0, vram_8bit=8.5, vram_4bit=5.0,
        description="Best quality, needs 16GB+ VRAM"
    ),
}


class QwenVLClient:
    """
    Wrapper class that provides OpenAI-compatible interface for QwenVL models.
    Uses HuggingFace transformers with BitsAndBytes quantization.
    """

    # Class-level model cache for keep_model_loaded
    _cached_model = None
    _cached_processor = None
    _cached_tokenizer = None
    _cached_signature = None

    def __init__(
        self,
        model_name: str,
        quantization: str = "4-bit",
        device: str = "auto",
        keep_model_loaded: bool = True,
    ):
        """
        Initialize the QwenVL client.

        Args:
            model_name: Model key from QWENVL_MODELS
            quantization: "4-bit", "8-bit", or "None (FP16)"
            device: "auto", "cuda", "cpu", "mps"
            keep_model_loaded: Keep model in VRAM between calls
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.keep_model_loaded = keep_model_loaded

        # Will be loaded on first call
        self.model = None
        self.processor = None
        self.tokenizer = None

    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        missing = []

        try:
            import torch
        except ImportError:
            missing.append("torch")

        try:
            import transformers
        except ImportError:
            missing.append("transformers")

        try:
            import accelerate
        except ImportError:
            missing.append("accelerate")

        # bitsandbytes is optional (only needed for quantization on CUDA)
        if self.quantization != "None (FP16)":
            try:
                import bitsandbytes
            except ImportError:
                print("[QwenVLClient] Warning: bitsandbytes not installed, falling back to FP16")
                self.quantization = "None (FP16)"

        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}"
            )

    def _get_device_info(self) -> Dict[str, Any]:
        """Detect GPU type and available memory."""
        import torch

        info = {"type": "cpu", "available_gb": 0}

        if torch.cuda.is_available():
            info["type"] = "cuda"
            info["available_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["type"] = "mps"
            info["available_gb"] = 16  # Assume 16GB for Apple Silicon

        return info

    def _load_model(self):
        """Load model, processor, and tokenizer."""
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        from huggingface_hub import snapshot_download

        # Check if we can reuse cached model
        signature = (self.model_name, self.quantization, self.device)
        if (QwenVLClient._cached_model is not None and
            QwenVLClient._cached_signature == signature):
            print(f"[QwenVLClient] Reusing cached model: {self.model_name}")
            self.model = QwenVLClient._cached_model
            self.processor = QwenVLClient._cached_processor
            self.tokenizer = QwenVLClient._cached_tokenizer
            return

        model_info = QWENVL_MODELS.get(self.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"[QwenVLClient] Loading model: {model_info.name}")
        print(f"  Repo: {model_info.repo_id}")
        print(f"  Quantization: {self.quantization}")

        # Download model if needed
        model_path = os.path.join(QWENVL_DIR, self.model_name)
        if not os.path.exists(model_path):
            print(f"  Downloading model to {model_path}...")
            snapshot_download(
                repo_id=model_info.repo_id,
                local_dir=model_path,
                ignore_patterns=["*.md", ".git*"],
            )

        # Determine device
        device_info = self._get_device_info()
        if self.device == "auto":
            device = device_info["type"]
        else:
            device = self.device

        print(f"  Device: {device}")

        # Setup quantization config
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}

            if self.quantization == "4-bit":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    print("  Using 4-bit quantization (BitsAndBytes)")
                except ImportError:
                    load_kwargs["torch_dtype"] = torch.float16
                    print("  Fallback to FP16 (bitsandbytes not available)")

            elif self.quantization == "8-bit":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    print("  Using 8-bit quantization (BitsAndBytes)")
                except ImportError:
                    load_kwargs["torch_dtype"] = torch.float16
                    print("  Fallback to FP16 (bitsandbytes not available)")

            else:  # FP16
                load_kwargs["torch_dtype"] = torch.float16
                print("  Using FP16 precision")

        elif device == "mps":
            load_kwargs["device_map"] = "mps"
            load_kwargs["torch_dtype"] = torch.float32  # MPS works better with float32
            print("  Using Apple Metal (MPS)")

        else:  # CPU
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            print("  Using CPU")

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            **load_kwargs
        )
        self.model.eval()

        # Enable KV cache
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Cache for reuse
        if self.keep_model_loaded:
            QwenVLClient._cached_model = self.model
            QwenVLClient._cached_processor = self.processor
            QwenVLClient._cached_tokenizer = self.tokenizer
            QwenVLClient._cached_signature = signature

        print(f"[QwenVLClient] Model loaded successfully")

    def clear(self):
        """Unload model from memory."""
        if not self.keep_model_loaded:
            self.model = None
            self.processor = None
            self.tokenizer = None

            QwenVLClient._cached_model = None
            QwenVLClient._cached_processor = None
            QwenVLClient._cached_tokenizer = None
            QwenVLClient._cached_signature = None

            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            print("[QwenVLClient] Model unloaded")

    @property
    def chat(self):
        """OpenAI-style chaining: client.chat.completions.create()"""
        return self

    @property
    def completions(self):
        """OpenAI-style chaining: client.chat.completions.create()"""
        return self

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.6,
        **kwargs
    ) -> "QwenVLResponse":
        """
        Create a chat completion (OpenAI-compatible interface).

        Args:
            model: Model name (ignored, uses loaded model)
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            OpenAI-compatible response object
        """
        import torch
        import time
        from PIL import Image
        import base64
        import io

        self._check_dependencies()

        if self.model is None:
            self._load_model()

        start_time = time.time()

        # Convert messages to QwenVL format
        conversation = []
        images = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                conversation.append({"role": role, "content": [{"type": "text", "text": content}]})
            elif isinstance(content, list):
                conv_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            conv_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Extract base64 image
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)

                            if url.startswith("data:image"):
                                # Parse base64
                                header, b64data = url.split(",", 1)
                                img_bytes = base64.b64decode(b64data)
                                img = Image.open(io.BytesIO(img_bytes))
                                images.append(img)
                                conv_content.append({"type": "image", "image": img})
                    elif isinstance(item, str):
                        conv_content.append({"type": "text", "text": item})

                conversation.append({"role": role, "content": conv_content})

        # Apply chat template
        chat_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        processed = self.processor(
            text=chat_text,
            images=images if images else None,
            return_tensors="pt"
        )

        # Move to model device
        model_device = next(self.model.parameters()).device
        model_inputs = {
            k: v.to(model_device) if torch.is_tensor(v) else v
            for k, v in processed.items()
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        input_len = model_inputs["input_ids"].shape[1]
        response_ids = outputs[0][input_len:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        elapsed = time.time() - start_time
        print(f"[QwenVLClient] Generated in {elapsed:.1f}s")

        return QwenVLResponse(response_text.strip())


class QwenVLResponse:
    """OpenAI-compatible response wrapper."""

    def __init__(self, content: str):
        self.choices = [QwenVLChoice(content)]


class QwenVLChoice:
    """OpenAI-compatible choice wrapper."""

    def __init__(self, content: str):
        self.message = QwenVLMessage(content)


class QwenVLMessage:
    """OpenAI-compatible message wrapper."""

    def __init__(self, content: str):
        self.content = content


class SID_QwenVL_LLM(comfy_io.ComfyNode, BaseLLMProvider):
    """
    QwenVL Local Vision Model Provider.

    Uses HuggingFace transformers with BitsAndBytes quantization.
    No llama-cpp-python required - easier installation than GGUF.

    Supported models:
    - Qwen3-VL 2B/4B/8B (latest generation)
    - Qwen2.5-VL 3B/7B (stable, well-tested)
    """

    PROVIDER_NAME = "qwenvl"

    @classmethod
    def get_models(cls) -> List[str]:
        return list(QWENVL_MODELS.keys())

    @classmethod
    def get_default_model(cls) -> str:
        return "Qwen3-VL-4B-Instruct"

    @classmethod
    def get_default_url(cls) -> str:
        return ""  # Local model

    @classmethod
    def requires_api_key(cls) -> bool:
        return False

    @classmethod
    def supports_reasoning(cls, model: str) -> bool:
        return False  # Local models don't support reasoning mode

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""

        model_options = list(QWENVL_MODELS.keys())

        quantization_options = [
            "4-bit (VRAM-friendly)",
            "8-bit (Balanced)",
            "None (FP16)",
        ]

        model_info_text = (
            "QwenVL Local Vision Models - No API needed, no llama-cpp-python required.\n\n"
            "Uses HuggingFace transformers with BitsAndBytes quantization.\n"
            "Easier installation than GGUF models.\n\n"
            "Models:\n"
            "- Qwen3-VL 2B: 4GB VRAM, Fast\n"
            "- Qwen3-VL 4B: 6GB VRAM, Recommended\n"
            "- Qwen3-VL 8B: 12GB VRAM, High quality\n"
            "- Qwen2.5-VL 3B: 6GB VRAM, Stable\n"
            "- Qwen2.5-VL 7B: 15GB VRAM, Best quality"
        )

        return comfy_io.Schema(
            node_id="SID_QwenVL_LLM",
            display_name="SID QwenVL LLM",
            category="SID Photography Toolkit/LLM Providers",
            description=model_info_text,
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=model_options,
                    default=cls.get_default_model(),
                    tooltip="Select QwenVL model. Qwen3-VL-4B recommended for most users."
                ),
                comfy_io.Combo.Input(
                    "quantization",
                    options=quantization_options,
                    default="4-bit (VRAM-friendly)",
                    tooltip="4-bit: lowest VRAM, 8-bit: balanced, FP16: best quality but most VRAM"
                ),
                comfy_io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu", "mps"],
                    default="auto",
                    tooltip="Device to run on. Auto detects best option."
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity level (0=focused, 1=creative). 0.6 recommended."
                ),
                comfy_io.Boolean.Input(
                    "keep_model_loaded",
                    default=True,
                    display_name="Keep Model Loaded",
                    tooltip="Keep model in VRAM between runs. Faster but uses memory."
                ),
            ],
            outputs=[
                LLM_MODEL_Type.Output(
                    "llm_model",
                    display_name="LLM_MODEL",
                    tooltip="LLM configuration to connect to SID_ZImagePromptGenerator"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: str,
        quantization: str,
        device: str,
        temperature: float,
        keep_model_loaded: bool,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""

        # Get model info
        model_info = QWENVL_MODELS.get(model)
        if not model_info:
            raise ValueError(f"Unknown model: {model}")

        # Parse quantization
        quant_map = {
            "4-bit (VRAM-friendly)": "4-bit",
            "8-bit (Balanced)": "8-bit",
            "None (FP16)": "None (FP16)",
        }
        quant = quant_map.get(quantization, "4-bit")

        # Create configuration
        config = LLMModelConfig(
            provider=cls.PROVIDER_NAME,
            model=model,
            api_key="",  # No API key needed
            api_url="",  # Local model
            max_tokens=model_info.max_output_tokens,
            temperature=temperature,
            supports_vision=True,
            supports_system_prompt=True,
            supports_reasoning=False,
            extra_params={
                "quantization": quant,
                "device": device,
                "keep_model_loaded": keep_model_loaded,
                "repo_id": model_info.repo_id,
            },
        )

        print(f"[SID_QwenVL_LLM] Configured: {model}")
        print(f"  Quantization: {quant}, Device: {device}")
        print(f"  Keep loaded: {keep_model_loaded}, Temperature: {temperature}")

        return comfy_io.NodeOutput(config)
