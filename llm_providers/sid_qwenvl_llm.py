"""
SID_QwenVL_LLM Node

QwenVL vision-language model provider using HuggingFace transformers.
No llama-cpp-python required - uses transformers with BitsAndBytes quantization.

Supports Qwen3-VL models:
- Instruct variants: 2B, 4B, 8B, 32B (standard instruction-following)
- Thinking variants: 2B, 4B, 8B (reasoning/analysis models)

Features:
- VRAM auto-downgrade with 20% safety margin
- Attention mode selection (flash_attention_2, sdpa)
- Generation parameters (num_beams, repetition_penalty, top_p)
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

# Attention modes
ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa", "eager"]


@dataclass
class QwenVLModelInfo:
    """Information about a QwenVL model."""
    name: str
    repo_id: str
    vram_fp16: float  # GB
    vram_8bit: float  # GB
    vram_4bit: float  # GB
    is_fp8: bool = False  # Pre-quantized FP8 models
    is_thinking: bool = False  # Thinking/reasoning models
    max_output_tokens: int = 4096
    description: str = ""


# Available QwenVL models - comprehensive list matching QwenVL node
QWENVL_MODELS: Dict[str, QwenVLModelInfo] = {
    # =========================================================================
    # Qwen3-VL Series (Latest) - Instruct
    # =========================================================================
    "Qwen3-VL-2B-Instruct": QwenVLModelInfo(
        name="Qwen3-VL 2B Instruct (4GB) - Fast",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        description="Fast, good for 8GB GPUs"
    ),
    "Qwen3-VL-4B-Instruct": QwenVLModelInfo(
        name="Qwen3-VL 4B Instruct (6GB) - Recommended",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        description="Best balance of speed and quality"
    ),
    "Qwen3-VL-8B-Instruct": QwenVLModelInfo(
        name="Qwen3-VL 8B Instruct (12GB) - High Quality",
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        vram_fp16=12.0, vram_8bit=7.0, vram_4bit=4.5,
        description="High quality, needs 12GB+ VRAM"
    ),
    "Qwen3-VL-32B-Instruct": QwenVLModelInfo(
        name="Qwen3-VL 32B Instruct (28GB) - Best",
        repo_id="Qwen/Qwen3-VL-32B-Instruct",
        vram_fp16=28.0, vram_8bit=14.0, vram_4bit=8.5,
        description="Best quality, needs 24GB+ VRAM"
    ),

    # =========================================================================
    # Qwen3-VL Series - Thinking (Reasoning)
    # =========================================================================
    "Qwen3-VL-2B-Thinking": QwenVLModelInfo(
        name="Qwen3-VL 2B Thinking (4GB) - Reasoning",
        repo_id="Qwen/Qwen3-VL-2B-Thinking",
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        is_thinking=True,
        description="Fast reasoning model"
    ),
    "Qwen3-VL-4B-Thinking": QwenVLModelInfo(
        name="Qwen3-VL 4B Thinking (6GB) - Reasoning",
        repo_id="Qwen/Qwen3-VL-4B-Thinking",
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        is_thinking=True,
        description="Balanced reasoning model"
    ),
    "Qwen3-VL-8B-Thinking": QwenVLModelInfo(
        name="Qwen3-VL 8B Thinking (12GB) - Reasoning",
        repo_id="Qwen/Qwen3-VL-8B-Thinking",
        vram_fp16=12.0, vram_8bit=7.0, vram_4bit=4.5,
        is_thinking=True,
        description="High quality reasoning"
    ),

}


def get_available_vram_gb() -> float:
    """Get available VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get free memory
            free_mem = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            return (free_mem - allocated) / (1024**3)
    except Exception:
        pass
    return 0.0


def get_total_vram_gb() -> float:
    """Get total VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


class QwenVLClient:
    """
    Wrapper class that provides OpenAI-compatible interface for QwenVL models.
    Uses HuggingFace transformers with BitsAndBytes quantization.

    Features image embedding caching - when the same image is analyzed multiple times
    (e.g., component mode with 8+ calls per image), the processed image tensors are
    cached and reused, significantly reducing processing time.
    """

    # Class-level model cache for keep_model_loaded
    _cached_model = None
    _cached_processor = None
    _cached_tokenizer = None
    _cached_signature = None

    # Class-level image cache (hash -> processed tensors)
    _image_cache = {}
    _image_cache_max_size = 3  # Keep last 3 images cached

    def __init__(
        self,
        model_name: str,
        quantization: str = "4-bit",
        device: str = "auto",
        attention_mode: str = "auto",
        keep_model_loaded: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_beams: int = 1,
    ):
        """
        Initialize the QwenVL client.

        Args:
            model_name: Model key from QWENVL_MODELS
            quantization: "4-bit", "8-bit", or "None (FP16)"
            device: "auto", "cuda", "cpu", "mps"
            attention_mode: "auto", "flash_attention_2", "sdpa", "eager"
            keep_model_loaded: Keep model in VRAM between calls
            top_p: Nucleus sampling threshold (0.0-1.0)
            repetition_penalty: Penalty for repeated tokens (0.5-2.0)
            num_beams: Beam search width (1=sampling, >1=beam search)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.attention_mode = attention_mode
        self.keep_model_loaded = keep_model_loaded
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams

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
        if self.quantization not in ["None (FP16)", "FP8"]:
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

        info = {"type": "cpu", "available_gb": 0, "total_gb": 0}

        if torch.cuda.is_available():
            info["type"] = "cuda"
            props = torch.cuda.get_device_properties(0)
            info["total_gb"] = props.total_memory / (1024**3)
            info["available_gb"] = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["type"] = "mps"
            info["total_gb"] = 16  # Assume 16GB for Apple Silicon
            info["available_gb"] = 16

        return info

    def _auto_downgrade_quantization(self, model_info: QwenVLModelInfo, device_info: Dict) -> str:
        """
        Automatically downgrade quantization if insufficient VRAM.
        Uses 20% safety margin like QwenVL node.
        """
        if device_info["type"] not in ["cuda", "mps"]:
            return self.quantization  # No auto-downgrade for CPU

        available = device_info["available_gb"]
        safety_margin = 1.2  # 20% safety margin

        # Check each quantization level
        if self.quantization == "None (FP16)":
            needed = model_info.vram_fp16 * safety_margin
            if needed > available:
                print(f"[QwenVLClient] FP16 needs {needed:.1f}GB but only {available:.1f}GB available")
                print(f"[QwenVLClient] Auto-downgrading to 8-bit")
                self.quantization = "8-bit"

        if self.quantization == "8-bit":
            needed = model_info.vram_8bit * safety_margin
            if needed > available:
                print(f"[QwenVLClient] 8-bit needs {needed:.1f}GB but only {available:.1f}GB available")
                print(f"[QwenVLClient] Auto-downgrading to 4-bit")
                self.quantization = "4-bit"

        if self.quantization == "4-bit":
            needed = model_info.vram_4bit * safety_margin
            if needed > available:
                print(f"[QwenVLClient] WARNING: 4-bit needs {needed:.1f}GB but only {available:.1f}GB available")
                print(f"[QwenVLClient] May run out of memory!")

        return self.quantization

    def _load_model(self):
        """Load model, processor, and tokenizer."""
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        from huggingface_hub import snapshot_download

        # Check if we can reuse cached model
        signature = (self.model_name, self.quantization, self.device, self.attention_mode)
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

        print(f"  Device: {device} (Total: {device_info['total_gb']:.1f}GB, Available: {device_info['available_gb']:.1f}GB)")

        # Auto-downgrade quantization if needed (skip for FP8 models)
        if not model_info.is_fp8:
            self._auto_downgrade_quantization(model_info, device_info)

        print(f"  Quantization: {self.quantization}")

        # Setup quantization config
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Attention mode
        if self.attention_mode != "auto":
            load_kwargs["attn_implementation"] = self.attention_mode
            print(f"  Attention: {self.attention_mode}")
        else:
            # Try flash_attention_2, fallback to sdpa
            # Must check both import AND package metadata (transformers checks version)
            flash_available = False
            try:
                import flash_attn
                import importlib.metadata
                # This is what transformers checks - must succeed
                importlib.metadata.version("flash_attn")
                flash_available = True
            except (ImportError, importlib.metadata.PackageNotFoundError):
                pass

            if flash_available:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"  Attention: flash_attention_2 (auto-detected)")
            else:
                load_kwargs["attn_implementation"] = "sdpa"
                print(f"  Attention: sdpa (flash_attn not available)")

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}

            # Skip quantization for FP8 models (already quantized)
            if model_info.is_fp8:
                load_kwargs["torch_dtype"] = torch.float16
                print("  Using pre-quantized FP8 model")

            elif self.quantization == "4-bit":
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

    @classmethod
    def _hash_image(cls, b64_data: str) -> str:
        """Create a hash for image data (for caching)."""
        import hashlib
        # Use first 2000 chars for speed (enough to identify unique images)
        return hashlib.md5(b64_data[:2000].encode()).hexdigest()

    @classmethod
    def _get_cached_image(cls, image_hash: str) -> Optional[Dict]:
        """Get cached processed image tensors."""
        return cls._image_cache.get(image_hash)

    @classmethod
    def _cache_image(cls, image_hash: str, processed: Dict):
        """Cache processed image tensors."""
        # LRU-style: remove oldest if at capacity
        if len(cls._image_cache) >= cls._image_cache_max_size:
            oldest = next(iter(cls._image_cache))
            del cls._image_cache[oldest]

        cls._image_cache[image_hash] = processed
        print(f"[QwenVLClient] Image cached (cache size: {len(cls._image_cache)})")

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

        # Convert messages to QwenVL format with image caching
        conversation = []
        images = []
        image_hash = None  # For caching

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

                                # Generate hash for caching
                                image_hash = self._hash_image(b64data)

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

        # Process inputs with image caching
        model_device = next(self.model.parameters()).device
        cached_tensors = self._get_cached_image(image_hash) if image_hash else None

        if cached_tensors and images:
            # Cache HIT - reuse processed image tensors
            print(f"[QwenVLClient] Image cache HIT")

            # Process text only (faster)
            processed = self.processor(
                text=chat_text,
                images=images,  # Still needed for correct input_ids
                return_tensors="pt"
            )

            # Replace with cached image tensors (already on device)
            model_inputs = {}
            for k, v in processed.items():
                if k in cached_tensors:
                    # Use cached tensor (already on device)
                    model_inputs[k] = cached_tensors[k]
                elif torch.is_tensor(v):
                    model_inputs[k] = v.to(model_device)
                else:
                    model_inputs[k] = v
        else:
            # Cache MISS - process and cache
            if image_hash and images:
                print(f"[QwenVLClient] Image cache MISS - processing")

            processed = self.processor(
                text=chat_text,
                images=images if images else None,
                return_tensors="pt"
            )

            # Move to model device
            model_inputs = {
                k: v.to(model_device) if torch.is_tensor(v) else v
                for k, v in processed.items()
            }

            # Cache image tensors for reuse
            if image_hash and images:
                cache_data = {}
                for k in ["pixel_values", "image_grid_thw"]:
                    if k in model_inputs:
                        cache_data[k] = model_inputs[k]
                if cache_data:
                    self._cache_image(image_hash, cache_data)

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Beam search vs sampling
        if self.num_beams > 1:
            # Beam search mode - disables temperature/top_p
            gen_kwargs["num_beams"] = self.num_beams
            gen_kwargs["do_sample"] = False
        else:
            # Sampling mode
            gen_kwargs["do_sample"] = temperature > 0
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = self.top_p

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                **gen_kwargs
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
    No llama-cpp-python required - easy installation.

    Features:
    - Auto VRAM management with downgrade (20% safety margin)
    - Attention mode selection (flash_attention_2, sdpa)
    - Generation parameters (num_beams, repetition_penalty, top_p)
    - Thinking/Reasoning model variants for Agentic mode

    Supported models (7 total):
    - Qwen3-VL 2B/4B/8B/32B Instruct (standard)
    - Qwen3-VL 2B/4B/8B Thinking (reasoning)
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
        """Thinking models support reasoning."""
        model_info = QWENVL_MODELS.get(model)
        return model_info.is_thinking if model_info else False

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""

        model_options = list(QWENVL_MODELS.keys())

        quantization_options = [
            "4-bit (VRAM-friendly)",
            "8-bit (Balanced)",
            "None (FP16)",
        ]

        # Get VRAM info for display
        total_vram = get_total_vram_gb()
        vram_info = f" [GPU: {total_vram:.0f}GB]" if total_vram > 0 else " [No GPU]"

        model_info_text = (
            f"QwenVL Local Vision Models{vram_info}\n\n"
            "No API needed, no llama-cpp-python required.\n"
            "Uses HuggingFace transformers + BitsAndBytes.\n\n"
            "Instruct Models (standard):\n"
            "- 2B: 1.5-4GB, 4B: 2-6GB (Recommended)\n"
            "- 8B: 4.5-12GB, 32B: 8.5-28GB\n\n"
            "Thinking Models (reasoning, for Agentic mode):\n"
            "- 2B: 1.5-4GB, 4B: 2-6GB, 8B: 4.5-12GB"
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
                    tooltip="Select QwenVL model. Qwen3-VL-4B-Instruct recommended. Thinking models provide better analysis."
                ),
                comfy_io.Combo.Input(
                    "quantization",
                    options=quantization_options,
                    default="4-bit (VRAM-friendly)",
                    tooltip="4-bit: lowest VRAM (auto-downgrades if needed), 8-bit: balanced, FP16: best quality"
                ),
                comfy_io.Combo.Input(
                    "attention_mode",
                    options=ATTENTION_MODES,
                    default="auto",
                    tooltip="auto: tries flash_attention_2 then sdpa. Override for debugging."
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
                    tooltip="Sampling randomness. 0.2-0.4=focused, 0.7+=creative. Disabled if num_beams>1."
                ),
                comfy_io.Float.Input(
                    "top_p",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    round=0.05,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Nucleus sampling cutoff. 0.9-0.95 allows variety. Disabled if num_beams>1."
                ),
                comfy_io.Float.Input(
                    "repetition_penalty",
                    default=1.2,
                    min=0.5,
                    max=2.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Penalty for repeated phrases. >1 reduces repetition, 1.1-1.3 recommended."
                ),
                comfy_io.Int.Input(
                    "num_beams",
                    default=1,
                    min=1,
                    max=8,
                    tooltip="Beam search width. 1=sampling mode, >1=beam search (disables temperature/top_p, more stable)."
                ),
                comfy_io.Int.Input(
                    "max_tokens",
                    default=512,
                    min=512,
                    max=4096,
                    step=128,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Maximum tokens to generate. Higher=longer output but slower."
                ),
                comfy_io.Boolean.Input(
                    "keep_model_loaded",
                    default=True,
                    display_name="Keep Model Loaded",
                    tooltip="Keep model in VRAM between runs. Faster but uses memory."
                ),
                comfy_io.Boolean.Input(
                    "enable_reasoning",
                    default=False,
                    display_name="Enable Reasoning (Agentic)",
                    tooltip="Enable Agentic/reasoning mode in Advanced V2 generator. Only works with Thinking models."
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
        attention_mode: str,
        device: str,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        num_beams: int,
        max_tokens: int,
        keep_model_loaded: bool,
        enable_reasoning: bool,
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

        # For FP8 models, override quantization
        if model_info.is_fp8:
            quant = "FP8"

        # Reasoning requires both: Thinking model + user enabled
        supports_reasoning = model_info.is_thinking and enable_reasoning

        # Create configuration
        config = LLMModelConfig(
            provider=cls.PROVIDER_NAME,
            model=model,
            api_key="",  # No API key needed
            api_url="",  # Local model
            max_tokens=max_tokens,
            temperature=temperature,
            supports_vision=True,
            supports_system_prompt=True,
            supports_reasoning=supports_reasoning,
            extra_params={
                "quantization": quant,
                "device": device,
                "attention_mode": attention_mode,
                "keep_model_loaded": keep_model_loaded,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams,
                "repo_id": model_info.repo_id,
                "is_thinking": model_info.is_thinking,
                "is_fp8": model_info.is_fp8,
            },
        )

        # Log configuration
        print(f"[SID_QwenVL_LLM] Configured: {model}")
        print(f"  Quantization: {quant}, Device: {device}, Attention: {attention_mode}")
        print(f"  Temperature: {temperature}, Top-P: {top_p}, Rep.Penalty: {repetition_penalty}")
        print(f"  Num Beams: {num_beams}, Max Tokens: {max_tokens}")
        if model_info.is_thinking:
            if enable_reasoning:
                print(f"  [Thinking Model - Reasoning ENABLED]")
            else:
                print(f"  [Thinking Model - Reasoning DISABLED]")
        else:
            if enable_reasoning:
                print(f"  [Instruct Model - Reasoning toggle ignored (not a Thinking model)]")

        return comfy_io.NodeOutput(config)
