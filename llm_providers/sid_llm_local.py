"""
SID_LLM_Local Node

Unified local vision-language model provider for ComfyUI.
Supports multiple model families with automatic VRAM management.

Supported Model Families:
- QwenVL: Qwen3-VL, Qwen2.5-VL (2B-32B)
- Florence-2: Microsoft Florence-2 (0.2B-0.7B) - Fast captioning
- Moondream2: Lightweight VLM (1.8B) - Efficient
- SmolVLM: HuggingFace SmolVLM (0.25B-2B) - Ultra-efficient
- Phi-3.5-Vision: Microsoft Phi-3.5 (4.2B) - High quality

Features:
- VRAM auto-downgrade with safety margin
- Multiple quantization options
- Model caching for faster inference
- Image caching for repeated analyses
"""

import os
import gc
import sys
import hashlib
import threading
import time as time_module
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from comfy_api.latest import io as comfy_io
import folder_paths

from .llm_model_type import LLMModelConfig
from .base_llm_provider import BaseLLMProvider

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


# =============================================================================
# Speed Optimizations - Global Setup
# =============================================================================

def setup_cuda_optimizations():
    """
    Configure CUDA for optimal inference speed.
    Call once at module load.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Enable cuDNN autotuner for faster convolutions
            torch.backends.cudnn.benchmark = True
            # Allow TF32 on Ampere+ GPUs for faster matrix ops
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Disable gradient computation globally for inference
            torch.set_grad_enabled(False)
            print("[LocalModelClient] CUDA optimizations enabled: cudnn.benchmark, TF32")
    except Exception as e:
        print(f"[LocalModelClient] CUDA optimization setup warning: {e}")


def clear_memory():
    """
    Aggressively clear GPU and system memory before loading models.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def get_optimal_dtype(device: str) -> "torch.dtype":
    """
    Get the optimal dtype for the current GPU.
    - BF16 for Ampere+ (compute capability >= 8.0)
    - FP16 for older GPUs
    """
    import torch

    if device != "cuda":
        return torch.float32

    try:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere or newer
            return torch.bfloat16
        else:
            return torch.float16
    except Exception:
        return torch.float16


class InferenceProgressSpinner:
    """
    A console progress spinner that shows activity during model inference.
    Shows elapsed time and a spinning animation.
    """

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Generating"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _spinner_loop(self):
        """Background thread that displays the spinner."""
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time_module.time() - self._start_time
            spinner = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
            # Print progress on same line
            sys.stdout.write(f"\r[LocalModelClient] {spinner} {self.message}... {elapsed:.1f}s")
            sys.stdout.flush()
            idx += 1
            time_module.sleep(0.1)

    def start(self):
        """Start the spinner."""
        self._start_time = time_module.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the spinner and print completion."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        elapsed = time_module.time() - self._start_time if self._start_time else 0
        # Clear the spinner line and print done
        sys.stdout.write(f"\r[LocalModelClient] ✓ {self.message} complete ({elapsed:.1f}s)        \n")
        sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# Run CUDA optimizations on module load
setup_cuda_optimizations()

# Model download directory
LLM_LOCAL_DIR = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(LLM_LOCAL_DIR, exist_ok=True)

# Attention modes
ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa", "eager"]


class ModelFamily(Enum):
    """Supported model families."""
    QWENVL = "qwenvl"
    FLORENCE2 = "florence2"
    MOONDREAM2 = "moondream2"
    SMOLVLM = "smolvlm"
    PHI35_VISION = "phi35_vision"


@dataclass
class LocalModelInfo:
    """Information about a local vision model."""
    name: str
    repo_id: str
    family: ModelFamily
    vram_fp16: float  # GB
    vram_8bit: float  # GB
    vram_4bit: float  # GB
    is_fp8: bool = False
    is_thinking: bool = False
    max_output_tokens: int = 4096
    description: str = ""
    model_class: str = ""  # HuggingFace model class


# =============================================================================
# Model Registry - All supported local models
# =============================================================================

LOCAL_MODELS: Dict[str, LocalModelInfo] = {
    # =========================================================================
    # QwenVL Series - Full Featured VLM (Listed first as recommended)
    # =========================================================================
    "Qwen3-VL-2B-Instruct": LocalModelInfo(
        name="Qwen3-VL 2B | 4K tokens | 1.5GB VRAM (Fast)",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        max_output_tokens=4096,
        description="Fast, good for 8GB GPUs",
        model_class="AutoModelForVision2Seq"
    ),
    "Qwen3-VL-4B-Instruct": LocalModelInfo(
        name="Qwen3-VL 4B | 4K tokens | 2GB VRAM (Recommended)",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        max_output_tokens=4096,
        description="Best balance of speed and quality",
        model_class="AutoModelForVision2Seq"
    ),
    "Qwen3-VL-8B-Instruct": LocalModelInfo(
        name="Qwen3-VL 8B | 4K tokens | 4.5GB VRAM (Quality)",
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=12.0, vram_8bit=7.0, vram_4bit=4.5,
        max_output_tokens=4096,
        description="High quality, needs 12GB+ VRAM",
        model_class="AutoModelForVision2Seq"
    ),
    "Qwen3-VL-2B-Thinking": LocalModelInfo(
        name="Qwen3-VL 2B Thinking | 4K tokens | 1.5GB VRAM (Reasoning)",
        repo_id="Qwen/Qwen3-VL-2B-Thinking",
        family=ModelFamily.QWENVL,
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        max_output_tokens=4096,
        is_thinking=True,
        description="Fast reasoning model",
        model_class="AutoModelForVision2Seq"
    ),
    "Qwen3-VL-4B-Thinking": LocalModelInfo(
        name="Qwen3-VL 4B Thinking | 4K tokens | 2GB VRAM (Reasoning)",
        repo_id="Qwen/Qwen3-VL-4B-Thinking",
        family=ModelFamily.QWENVL,
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        max_output_tokens=4096,
        is_thinking=True,
        description="Balanced reasoning model",
        model_class="AutoModelForVision2Seq"
    ),
    "Qwen2.5-VL-3B-Instruct": LocalModelInfo(
        name="Qwen2.5-VL 3B | 4K tokens | 2GB VRAM",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=5.0, vram_8bit=3.0, vram_4bit=2.0,
        max_output_tokens=4096,
        description="Previous gen, compact model",
        model_class="AutoModelForVision2Seq"
    ),
    "Qwen2.5-VL-7B-Instruct": LocalModelInfo(
        name="Qwen2.5-VL 7B | 4K tokens | 4GB VRAM",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=10.0, vram_8bit=6.0, vram_4bit=4.0,
        max_output_tokens=4096,
        description="Previous gen, stable quality",
        model_class="AutoModelForVision2Seq"
    ),

    # =========================================================================
    # Florence-2 Series (Microsoft) - Ultra Fast Captioning
    # =========================================================================
    "Florence-2-base": LocalModelInfo(
        name="Florence-2 Base | 1K tokens | 0.6GB VRAM (Ultra Fast)",
        repo_id="microsoft/Florence-2-base",
        family=ModelFamily.FLORENCE2,
        vram_fp16=1.5, vram_8bit=1.0, vram_4bit=0.6,
        max_output_tokens=1024,
        description="Ultra-fast captioning, 0.23B params",
        model_class="AutoModelForCausalLM"
    ),
    "Florence-2-large": LocalModelInfo(
        name="Florence-2 Large | 1K tokens | 1.2GB VRAM (Fast)",
        repo_id="microsoft/Florence-2-large",
        family=ModelFamily.FLORENCE2,
        vram_fp16=3.0, vram_8bit=2.0, vram_4bit=1.2,
        max_output_tokens=1024,
        description="Better quality captioning, 0.77B params",
        model_class="AutoModelForCausalLM"
    ),
    "Florence-2-base-ft": LocalModelInfo(
        name="Florence-2 Base FT | 1K tokens | 0.6GB VRAM",
        repo_id="microsoft/Florence-2-base-ft",
        family=ModelFamily.FLORENCE2,
        vram_fp16=1.5, vram_8bit=1.0, vram_4bit=0.6,
        max_output_tokens=1024,
        description="Fine-tuned for downstream tasks",
        model_class="AutoModelForCausalLM"
    ),
    "Florence-2-large-ft": LocalModelInfo(
        name="Florence-2 Large FT | 1K tokens | 1.2GB VRAM",
        repo_id="microsoft/Florence-2-large-ft",
        family=ModelFamily.FLORENCE2,
        vram_fp16=3.0, vram_8bit=2.0, vram_4bit=1.2,
        max_output_tokens=1024,
        description="Fine-tuned, best Florence quality",
        model_class="AutoModelForCausalLM"
    ),

    # =========================================================================
    # Moondream2 - Efficient VLM
    # =========================================================================
    "Moondream2": LocalModelInfo(
        name="Moondream2 | 2K tokens | 1.5GB VRAM (Efficient)",
        repo_id="vikhyatk/moondream2",
        family=ModelFamily.MOONDREAM2,
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        max_output_tokens=2048,
        description="Efficient VLM, runs on CPU too",
        model_class="AutoModelForCausalLM"
    ),

    # =========================================================================
    # SmolVLM Series (HuggingFace) - Ultra Efficient
    # =========================================================================
    "SmolVLM-256M": LocalModelInfo(
        name="SmolVLM 256M | 1K tokens | 0.3GB VRAM (Tiny)",
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        family=ModelFamily.SMOLVLM,
        vram_fp16=0.8, vram_8bit=0.5, vram_4bit=0.3,
        max_output_tokens=1024,
        description="Tiniest VLM, edge/mobile ready",
        model_class="AutoModelForVision2Seq"
    ),
    "SmolVLM-500M": LocalModelInfo(
        name="SmolVLM 500M | 1K tokens | 0.6GB VRAM (Small)",
        repo_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        family=ModelFamily.SMOLVLM,
        vram_fp16=1.5, vram_8bit=1.0, vram_4bit=0.6,
        max_output_tokens=1024,
        description="Very small, good balance",
        model_class="AutoModelForVision2Seq"
    ),
    "SmolVLM-2B": LocalModelInfo(
        name="SmolVLM 2B | 2K tokens | 2GB VRAM",
        repo_id="HuggingFaceTB/SmolVLM-Instruct",
        family=ModelFamily.SMOLVLM,
        vram_fp16=4.5, vram_8bit=3.0, vram_4bit=2.0,
        max_output_tokens=2048,
        description="Best SmolVLM quality",
        model_class="AutoModelForVision2Seq"
    ),

    # =========================================================================
    # Phi-3.5-Vision (Microsoft) - High Quality
    # =========================================================================
    "Phi-3.5-Vision": LocalModelInfo(
        name="Phi-3.5 Vision | 4K tokens | 3GB VRAM (Quality)",
        repo_id="microsoft/Phi-3.5-vision-instruct",
        family=ModelFamily.PHI35_VISION,
        vram_fp16=8.5, vram_8bit=5.0, vram_4bit=3.0,
        max_output_tokens=4096,
        description="High quality vision reasoning",
        model_class="AutoModelForCausalLM"
    ),
}


def get_available_vram_gb() -> float:
    """Get available VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
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


# =============================================================================
# Unified Local Model Client
# =============================================================================

class LocalModelClient:
    """
    Unified client for local vision-language models.
    Provides OpenAI-compatible interface for all model families.

    Speed Optimizations:
    - Model caching (keeps model loaded between runs)
    - Image caching with hash-based deduplication
    - BF16/FP16 auto-selection based on GPU capability
    - CUDA optimizations (cudnn.benchmark, TF32)
    - KV cache enabled for all models
    - Flash Attention 2 / SDPA auto-selection
    - Image resizing for faster processing
    """

    # Class-level model cache
    _cached_model = None
    _cached_processor = None
    _cached_tokenizer = None
    _cached_signature = None
    _image_cache = {}
    _image_cache_max_size = 5
    _compiled_generate = None  # For torch.compile caching

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
        use_torch_compile: bool = False,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.attention_mode = attention_mode
        self.keep_model_loaded = keep_model_loaded
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.use_torch_compile = use_torch_compile

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_info = LOCAL_MODELS.get(model_name)

    def _get_device_info(self) -> Dict[str, Any]:
        """Detect GPU type and available memory with detailed logging."""
        import torch

        info = {"type": "cpu", "available_gb": 0, "total_gb": 0, "gpu_name": "N/A"}

        print("[LocalModelClient] Detecting device...")

        # Check CUDA
        if torch.cuda.is_available():
            info["type"] = "cuda"
            props = torch.cuda.get_device_properties(0)
            info["total_gb"] = props.total_memory / (1024**3)
            info["available_gb"] = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            info["gpu_name"] = props.name
            info["compute_capability"] = f"{props.major}.{props.minor}"
            print(f"  CUDA available: {info['gpu_name']}")
            print(f"  Compute capability: {info['compute_capability']}")
            print(f"  VRAM: {info['total_gb']:.1f}GB total, {info['available_gb']:.1f}GB free")

            # Check bitsandbytes for quantization
            try:
                import bitsandbytes
                print(f"  bitsandbytes: installed (v{bitsandbytes.__version__})")
            except ImportError:
                print("  WARNING: bitsandbytes not installed - 4-bit/8-bit quantization unavailable")
                print("  Install with: pip install bitsandbytes")

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["type"] = "mps"
            info["total_gb"] = 16  # Estimate
            info["available_gb"] = 16
            info["gpu_name"] = "Apple Silicon (MPS)"
            print(f"  MPS available: Apple Silicon GPU")

        else:
            print("  WARNING: No GPU detected - running on CPU (will be slow)")
            print("  For CUDA: ensure NVIDIA drivers and pytorch+cuda are installed")
            print("  For MPS: ensure macOS 12.3+ with Apple Silicon")

        return info

    def _auto_downgrade_quantization(self, model_info: LocalModelInfo, device_info: Dict) -> str:
        """Automatically downgrade quantization if insufficient VRAM."""
        if device_info["type"] not in ["cuda", "mps"]:
            return self.quantization

        available = device_info["available_gb"]
        safety_margin = 1.2

        if self.quantization == "None (FP16)":
            needed = model_info.vram_fp16 * safety_margin
            if needed > available:
                print(f"[LocalModelClient] FP16 needs {needed:.1f}GB, auto-downgrading to 8-bit")
                self.quantization = "8-bit"

        if self.quantization == "8-bit":
            needed = model_info.vram_8bit * safety_margin
            if needed > available:
                print(f"[LocalModelClient] 8-bit needs {needed:.1f}GB, auto-downgrading to 4-bit")
                self.quantization = "4-bit"

        return self.quantization

    def _load_model(self):
        """Load model, processor, and tokenizer with speed optimizations."""
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        from huggingface_hub import snapshot_download

        # Check cache
        signature = (self.model_name, self.quantization, self.device, self.attention_mode)
        if (LocalModelClient._cached_model is not None and
            LocalModelClient._cached_signature == signature):
            print(f"[LocalModelClient] Reusing cached model: {self.model_name}")
            self.model = LocalModelClient._cached_model
            self.processor = LocalModelClient._cached_processor
            self.tokenizer = LocalModelClient._cached_tokenizer
            return

        # Clear memory before loading new model
        if LocalModelClient._cached_model is not None:
            print(f"[LocalModelClient] Unloading previous model...")
            LocalModelClient._cached_model = None
            LocalModelClient._cached_processor = None
            LocalModelClient._cached_tokenizer = None
            LocalModelClient._cached_signature = None
            LocalModelClient._compiled_generate = None
            clear_memory()

        if not self.model_info:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"[LocalModelClient] Loading model: {self.model_info.name}")
        print(f"  Family: {self.model_info.family.value}")
        print(f"  Repo: {self.model_info.repo_id}")

        # Download model
        family_dir = os.path.join(LLM_LOCAL_DIR, self.model_info.family.value)
        os.makedirs(family_dir, exist_ok=True)
        model_path = os.path.join(family_dir, self.model_name)

        if not os.path.exists(model_path):
            print(f"  Downloading model to {model_path}...")
            snapshot_download(
                repo_id=self.model_info.repo_id,
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

        # Auto-downgrade quantization
        if not self.model_info.is_fp8:
            self._auto_downgrade_quantization(self.model_info, device_info)

        print(f"  Quantization: {self.quantization}")

        # Load based on model family
        if self.model_info.family == ModelFamily.FLORENCE2:
            self._load_florence2(model_path, device)
        elif self.model_info.family == ModelFamily.MOONDREAM2:
            self._load_moondream2(model_path, device)
        elif self.model_info.family == ModelFamily.SMOLVLM:
            self._load_smolvlm(model_path, device)
        elif self.model_info.family == ModelFamily.PHI35_VISION:
            self._load_phi35_vision(model_path, device)
        elif self.model_info.family == ModelFamily.QWENVL:
            self._load_qwenvl(model_path, device)

        # Cache for reuse
        if self.keep_model_loaded:
            LocalModelClient._cached_model = self.model
            LocalModelClient._cached_processor = self.processor
            LocalModelClient._cached_tokenizer = self.tokenizer
            LocalModelClient._cached_signature = signature

        print(f"[LocalModelClient] Model loaded successfully")

    def _get_quantization_config(self, device: str):
        """
        Get BitsAndBytes quantization config with optimal dtype.
        Uses BF16 on Ampere+ GPUs for better performance.
        """
        import torch

        # Get optimal dtype for this GPU
        optimal_dtype = get_optimal_dtype(device)

        if device != "cuda" or self.quantization == "None (FP16)":
            dtype_name = "BF16" if optimal_dtype == torch.bfloat16 else "FP16"
            print(f"  Using {dtype_name} (no quantization)")
            return None, optimal_dtype

        try:
            from transformers import BitsAndBytesConfig

            if self.quantization == "4-bit":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=optimal_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ), None
            elif self.quantization == "8-bit":
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False,
                ), None
        except ImportError:
            print("  bitsandbytes not available, using optimal dtype")

        return None, optimal_dtype

    def _load_florence2(self, model_path: str, device: str):
        """Load Florence-2 model with speed optimizations."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        # Enable KV cache for faster generation
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def _load_moondream2(self, model_path: str, device: str):
        """Load Moondream2 model with speed optimizations."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        # Enable KV cache for faster generation
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = self.model  # Moondream has built-in image processing

    def _load_smolvlm(self, model_path: str, device: str):
        """Load SmolVLM model with speed optimizations."""
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Add attention implementation for SmolVLM
        if device == "cuda":
            try:
                # Check both import AND package metadata (transformers requires metadata)
                import flash_attn
                import importlib.metadata
                importlib.metadata.version("flash_attn")  # This will raise if metadata missing
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                else:
                    load_kwargs["attn_implementation"] = "sdpa"
            except (ImportError, importlib.metadata.PackageNotFoundError):
                load_kwargs["attn_implementation"] = "sdpa"

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.float16

        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        # Enable KV cache for faster generation
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def _load_phi35_vision(self, model_path: str, device: str):
        """Load Phi-3.5-Vision model with speed optimizations."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "_attn_implementation": "eager",  # Phi requires eager attention
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        # Enable KV cache for faster generation
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def _load_qwenvl(self, model_path: str, device: str):
        """Load QwenVL model with speed optimizations."""
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }

        # Select optimal attention implementation
        actual_attention_mode = self.attention_mode
        if self.attention_mode != "auto":
            load_kwargs["attn_implementation"] = self.attention_mode
            print(f"  Using {self.attention_mode} (user-specified)")
        elif device == "cuda":
            try:
                # Check both import AND package metadata (transformers requires metadata)
                import flash_attn
                import importlib.metadata
                importlib.metadata.version("flash_attn")  # This will raise if metadata missing
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    actual_attention_mode = "flash_attention_2"
                    print("  Using Flash Attention 2 (auto-detected)")
                else:
                    load_kwargs["attn_implementation"] = "sdpa"
                    actual_attention_mode = "sdpa"
                    print("  Using SDPA (GPU compute capability < 8.0)")
            except (ImportError, importlib.metadata.PackageNotFoundError):
                load_kwargs["attn_implementation"] = "sdpa"
                actual_attention_mode = "sdpa"
                print("  Using SDPA (flash_attn not properly installed)")

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.float16

        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        # Enable KV cache for faster generation
        self.model.config.use_cache = True

        # Apply torch.compile for faster inference (CUDA + Torch 2.1+ only)
        if self.use_torch_compile and device == "cuda":
            try:
                torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
                if torch_version >= (2, 1):
                    print("  Applying torch.compile (mode=reduce-overhead)...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("  torch.compile applied successfully (first run may be slower)")
                else:
                    print(f"  torch.compile skipped: requires Torch 2.1+, found {torch.__version__}")
            except Exception as e:
                print(f"  torch.compile failed (continuing without): {e}")

        # Adaptive resolution for QwenVL (optimal range: 480x480 to 2560x2560)
        # Qwen2.5-VL uses patch_size=28, Qwen3-VL uses patch_size=14
        # Resolution rounded to nearest multiple of patch_size
        is_qwen3 = "Qwen3" in self.model_name
        patch_size = 14 if is_qwen3 else 28

        # Min: ~200K pixels (good for speed), Max: ~1M pixels (good for detail)
        # These values align with Qwen docs: 480x480 to 1280x1280 optimal
        min_pixels = 256 * patch_size * patch_size   # ~100K-200K pixels
        max_pixels = 1280 * patch_size * patch_size  # ~500K-1M pixels

        print(f"  Resolution limits: {min_pixels//1000}K - {max_pixels//1000}K pixels (patch_size={patch_size})")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.6,
        min_response_words: int = 20,
        max_retries: int = 2,
        **kwargs
    ) -> "LocalModelResponse":
        """Create a chat completion (OpenAI-compatible interface)."""
        import torch
        import time
        from PIL import Image
        import base64
        import io

        if self.model is None:
            self._load_model()

        start_time = time.time()

        # Extract image and text from messages
        images = []
        text_prompt = ""

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_prompt += content + "\n"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_prompt += item.get("text", "") + "\n"
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                            if url.startswith("data:image"):
                                _, b64data = url.split(",", 1)
                                img_bytes = base64.b64decode(b64data)
                                img = Image.open(io.BytesIO(img_bytes))
                                images.append(img)

        # Generate with retry logic for short responses
        response_text = ""
        retry_count = 0
        current_temp = temperature

        while retry_count <= max_retries:
            gen_start = time.time()

            # Generate based on model family
            if self.model_info.family == ModelFamily.FLORENCE2:
                response_text = self._generate_florence2(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.MOONDREAM2:
                response_text = self._generate_moondream2(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.SMOLVLM:
                response_text = self._generate_smolvlm(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.PHI35_VISION:
                response_text = self._generate_phi35_vision(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.QWENVL:
                response_text = self._generate_qwenvl(messages, images, max_tokens, current_temp)
            else:
                raise ValueError(f"Unknown model family: {self.model_info.family}")

            gen_elapsed = time.time() - gen_start

            # Validate response length
            word_count = len(response_text.split())
            if word_count >= min_response_words:
                break

            # Response too short, retry with adjusted temperature
            retry_count += 1
            if retry_count <= max_retries:
                current_temp = min(1.0, temperature + 0.2 * retry_count)
                print(f"[LocalModelClient] Response too short ({word_count} words), retrying {retry_count}/{max_retries} with temp={current_temp:.1f}")

        elapsed = time.time() - start_time
        word_count = len(response_text.split())
        tokens_approx = len(response_text) // 4  # Rough token estimate

        # Quality metrics
        print(f"[LocalModelClient] Generated in {elapsed:.1f}s")
        print(f"  Output: {word_count} words, ~{tokens_approx} tokens")
        if elapsed > 0:
            print(f"  Speed: ~{tokens_approx/elapsed:.1f} tokens/sec")
        if retry_count > 0:
            print(f"  Retries: {retry_count}")

        return LocalModelResponse(response_text.strip())

    def _generate_florence2(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Florence-2."""
        import torch

        # Florence-2 uses task prompts like <DETAILED_CAPTION>
        task_prompt = "<MORE_DETAILED_CAPTION>"

        if images:
            inputs = self.processor(text=task_prompt, images=images[0], return_tensors="pt")
        else:
            inputs = self.processor(text=task_prompt, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with InferenceProgressSpinner("Generating (Florence-2)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def _generate_moondream2(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Moondream2."""
        import torch

        if images:
            # Moondream has built-in image encoding
            enc_image = self.model.encode_image(images[0])
            with InferenceProgressSpinner("Generating (Moondream2)"):
                with torch.inference_mode():
                    response = self.model.answer_question(enc_image, prompt.strip(), self.tokenizer)
            return response
        else:
            return "No image provided"

    def _generate_smolvlm(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with SmolVLM."""
        import torch

        messages = [{"role": "user", "content": []}]
        if images:
            messages[0]["content"].append({"type": "image"})
        messages[0]["content"].append({"type": "text", "text": prompt.strip()})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        # Get input length to extract only generated tokens later
        input_len = inputs["input_ids"].shape[1]

        with InferenceProgressSpinner("Generating (SmolVLM)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        # Only decode the NEW tokens (exclude the input prompt)
        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_phi35_vision(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Phi-3.5-Vision."""
        import torch

        # Compatibility fix for newer transformers versions
        # Phi-3.5-Vision's custom code expects attributes that newer transformers removed
        try:
            from transformers.cache_utils import DynamicCache
            # seen_tokens was renamed/removed
            if not hasattr(DynamicCache, 'seen_tokens'):
                @property
                def seen_tokens_compat(self):
                    return self.get_seq_length()
                DynamicCache.seen_tokens = seen_tokens_compat

            # get_max_length was removed in newer transformers
            if not hasattr(DynamicCache, 'get_max_length'):
                def get_max_length_compat(self):
                    # Return None as default (no max length limit)
                    # Or return the max cache length if set
                    return getattr(self, '_max_cache_len', None)
                DynamicCache.get_max_length = get_max_length_compat

            # get_usable_length was removed in newer transformers
            if not hasattr(DynamicCache, 'get_usable_length'):
                def get_usable_length_compat(self, new_seq_length: int = 0, layer_idx: int = 0):
                    # Return current sequence length for the given layer (all cached tokens are usable)
                    return self.get_seq_length(layer_idx)
                DynamicCache.get_usable_length = get_usable_length_compat
        except ImportError:
            pass

        messages = [{"role": "user", "content": ""}]
        if images:
            messages[0]["content"] = f"<|image_1|>\n{prompt.strip()}"
        else:
            messages[0]["content"] = prompt.strip()

        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=images if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        # Get input length to extract only generated tokens later
        input_len = inputs["input_ids"].shape[1]

        with InferenceProgressSpinner("Generating (Phi-3.5-Vision)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

        # Only decode the NEW tokens (exclude the input prompt)
        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_qwenvl(self, messages: List, images: List, max_tokens: int, temperature: float) -> str:
        """Generate with QwenVL."""
        import torch

        # Build QwenVL conversation format
        conversation = []
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
                            if images:
                                conv_content.append({"type": "image", "image": images[0]})
                conversation.append({"role": role, "content": conv_content})

        chat_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        processed = self.processor(
            text=chat_text,
            images=images if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in processed.items()}

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.num_beams > 1:
            gen_kwargs["num_beams"] = self.num_beams
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = temperature > 0
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = self.top_p

        with InferenceProgressSpinner("Generating (QwenVL)"):
            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, **gen_kwargs)

        input_len = model_inputs["input_ids"].shape[1]
        response_ids = outputs[0][input_len:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)


class LocalModelResponse:
    """OpenAI-compatible response wrapper."""
    def __init__(self, content: str):
        self.choices = [LocalModelChoice(content)]


class LocalModelChoice:
    """OpenAI-compatible choice wrapper."""
    def __init__(self, content: str):
        self.message = LocalModelMessage(content)


class LocalModelMessage:
    """OpenAI-compatible message wrapper."""
    def __init__(self, content: str):
        self.content = content


# =============================================================================
# ComfyUI Node
# =============================================================================

class SID_LLM_Local(comfy_io.ComfyNode, BaseLLMProvider):
    """
    Local Vision Language Model Provider.

    Supports multiple model families:
    - Florence-2: Ultra-fast captioning (0.2-0.8B)
    - Moondream2: Efficient VLM (1.8B)
    - SmolVLM: Tiny models (0.25-2B)
    - Phi-3.5-Vision: High quality (4.2B)
    - QwenVL: Full-featured VLM (2-8B)

    No API needed, runs locally with automatic VRAM management.
    """

    PROVIDER_NAME = "local"

    @classmethod
    def get_models(cls) -> List[str]:
        return list(LOCAL_MODELS.keys())

    @classmethod
    def get_default_model(cls) -> str:
        return "Qwen3-VL-2B-Instruct"

    @classmethod
    def get_default_url(cls) -> str:
        return ""

    @classmethod
    def requires_api_key(cls) -> bool:
        return False

    @classmethod
    def supports_reasoning(cls, model: str) -> bool:
        model_info = LOCAL_MODELS.get(model)
        return model_info.is_thinking if model_info else False

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""

        # Build model options with display names
        model_options = []
        for model_id, model_info in LOCAL_MODELS.items():
            model_options.append(model_id)

        quantization_options = [
            "Auto (Detect Best)",
            "4-bit (VRAM-friendly)",
            "8-bit (Balanced)",
            "None (FP16)",
        ]

        max_tokens_options = [
            "Low (512)",
            "Medium (1024)",
            "High (2048)",
            "Very High (Model Max)",
            "Custom",
        ]

        total_vram = get_total_vram_gb()
        vram_info = f" [GPU: {total_vram:.0f}GB]" if total_vram > 0 else " [No GPU]"

        return comfy_io.Schema(
            node_id="SID_LLM_Local",
            display_name="SID LLM Local",
            category="SID Photography Toolkit/LLM Providers",
            description=f"Local Vision Models{vram_info} - No API needed",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=model_options,
                    default=cls.get_default_model(),
                    tooltip="Select model: Name | Max Tokens | VRAM (4-bit)"
                ),
                comfy_io.Combo.Input(
                    "quantization",
                    options=quantization_options,
                    default="Auto (Detect Best)",
                    tooltip="Auto: detects best based on VRAM, 4-bit: lowest VRAM, FP16: best quality"
                ),
                comfy_io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu", "mps"],
                    default="auto",
                    tooltip="Device to run on (auto recommended)"
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity (0=deterministic, 0.3=balanced, 1=creative)"
                ),
                comfy_io.Combo.Input(
                    "max_tokens_preset",
                    options=max_tokens_options,
                    default="Medium (1024)",
                    tooltip="Output length: Low=512, Medium=1024, High=2048, Very High=Model Max"
                ),
                comfy_io.Int.Input(
                    "custom_max_tokens",
                    default=1024,
                    min=128,
                    max=4096,
                    tooltip="Custom max tokens (only used when preset is 'Custom')"
                ),
                comfy_io.Boolean.Input(
                    "enable_reasoning",
                    default=True,
                    display_name="Enable Reasoning",
                    tooltip="Enable reasoning for Thinking models (Qwen3-VL-*-Thinking)"
                ),
                comfy_io.Boolean.Input(
                    "keep_model_loaded",
                    default=True,
                    display_name="Keep Model Loaded",
                    tooltip="Keep model in VRAM between runs (faster repeat inference)"
                ),
                comfy_io.Combo.Input(
                    "attention_mode",
                    options=ATTENTION_MODES,
                    default="auto",
                    tooltip="Attention implementation: auto (recommended), flash_attention_2 (Ampere+), sdpa, eager"
                ),
                comfy_io.Float.Input(
                    "repetition_penalty",
                    default=1.2,
                    min=0.8,
                    max=2.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Penalize repeated tokens (1.0=off, 1.2=recommended, 2.0=strong)"
                ),
                comfy_io.Float.Input(
                    "top_p",
                    default=0.9,
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    round=0.05,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Nucleus sampling (0.9=recommended, lower=more focused)"
                ),
                comfy_io.Boolean.Input(
                    "use_torch_compile",
                    default=False,
                    display_name="Use Torch Compile",
                    tooltip="Enable torch.compile for faster inference (CUDA + Torch 2.1+ only, first run slower)"
                ),
            ],
            outputs=[
                LLM_MODEL_Type.Output(
                    "llm_model",
                    display_name="LLM_MODEL",
                    tooltip="LLM configuration to connect to prompt generator"
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
        max_tokens_preset: str,
        custom_max_tokens: int,
        enable_reasoning: bool,
        keep_model_loaded: bool,
        attention_mode: str,
        repetition_penalty: float,
        top_p: float,
        use_torch_compile: bool,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""
        try:
            # Validate model
            model_info = LOCAL_MODELS.get(model)
            if not model_info:
                raise ValueError(f"[SID_LLM_Local] Unknown model: {model}")

            # Handle quantization (including Auto mode)
            if quantization == "Auto (Detect Best)":
                available_vram = get_available_vram_gb()
                if available_vram <= 0:
                    # No GPU detected, use 4-bit (can fallback to CPU)
                    quant = "4-bit"
                    print(f"[SID_LLM_Local] Auto-quantization: No GPU, using 4-bit")
                else:
                    # Select based on available VRAM with safety margin
                    safety_margin = 1.2
                    if model_info.vram_fp16 * safety_margin <= available_vram:
                        quant = "None (FP16)"
                        print(f"[SID_LLM_Local] Auto-quantization: FP16 (VRAM: {available_vram:.1f}GB available)")
                    elif model_info.vram_8bit * safety_margin <= available_vram:
                        quant = "8-bit"
                        print(f"[SID_LLM_Local] Auto-quantization: 8-bit (VRAM: {available_vram:.1f}GB available)")
                    else:
                        quant = "4-bit"
                        print(f"[SID_LLM_Local] Auto-quantization: 4-bit (VRAM: {available_vram:.1f}GB available)")
            else:
                quant_map = {
                    "4-bit (VRAM-friendly)": "4-bit",
                    "8-bit (Balanced)": "8-bit",
                    "None (FP16)": "None (FP16)",
                }
                quant = quant_map.get(quantization, "4-bit")

            # Resolve max_tokens from preset
            model_max_tokens = model_info.max_output_tokens
            max_tokens_map = {
                "Low (512)": 512,
                "Medium (1024)": 1024,
                "High (2048)": 2048,
                "Very High (Model Max)": model_max_tokens,
                "Custom": custom_max_tokens,
            }
            max_tokens = max_tokens_map.get(max_tokens_preset, 1024)

            # Validate custom_max_tokens doesn't exceed model max
            if max_tokens > model_max_tokens:
                print(f"[SID_LLM_Local] Warning: Requested {max_tokens} tokens exceeds model max ({model_max_tokens}), capping")
                max_tokens = model_max_tokens

            # Handle reasoning - only enable for Thinking models
            reasoning_enabled = enable_reasoning and model_info.is_thinking
            if enable_reasoning and not model_info.is_thinking:
                print(f"[SID_LLM_Local] Note: Reasoning disabled - {model} is not a Thinking model")

            config = LLMModelConfig(
                provider=cls.PROVIDER_NAME,
                model=model,
                api_key="",
                api_url="",
                max_tokens=max_tokens,
                temperature=temperature,
                supports_vision=True,
                supports_system_prompt=True,
                supports_reasoning=reasoning_enabled,
                extra_params={
                    "quantization": quant,
                    "device": device,
                    "attention_mode": attention_mode,
                    "keep_model_loaded": keep_model_loaded,
                    "repo_id": model_info.repo_id,
                    "family": model_info.family.value,
                    "is_thinking": model_info.is_thinking,
                    "enable_reasoning": reasoning_enabled,
                    "repetition_penalty": repetition_penalty,
                    "top_p": top_p,
                    "use_torch_compile": use_torch_compile,
                },
            )

            print(f"[SID_LLM_Local] Configured: {model}")
            print(f"  Family: {model_info.family.value}, VRAM: {model_info.vram_4bit:.1f}-{model_info.vram_fp16:.1f}GB")
            print(f"  Quantization: {quant}, Device: {device}, Max Tokens: {max_tokens}")
            if model_info.is_thinking:
                print(f"  Reasoning: {'Enabled' if reasoning_enabled else 'Disabled'}")

            return comfy_io.NodeOutput(config)

        except ValueError:
            # Re-raise validation errors as-is for ComfyUI error modal
            raise
        except Exception as e:
            # Catch unexpected errors and raise with context
            raise RuntimeError(f"[SID_LLM_Local] Unexpected error: {type(e).__name__}: {str(e)}") from e
