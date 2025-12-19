# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Local LLM Provider Node

Unified local vision-language model provider for ComfyUI.
Supports multiple model families with automatic VRAM management.

Supported Vision Model Families:
- QwenVL: Qwen3-VL, Qwen2.5-VL (2B-32B) - Best overall quality

Text-Only Models (for prompt generation):
- Qwen3 Text: Qwen3 text models (0.6B-8B)
- Qwen2.5 Text: Qwen2.5 text models (1.5B-7B)
- Llama 3.2 Text: Llama 3.2 text (1B-3B)
- Phi-3.5 Mini: Microsoft Phi-3.5-mini
- Mistral 7B: Mistral text model
- Gemma 2: Google Gemma 2 (2B)

Features:
- VRAM auto-downgrade with safety margin
- Multiple quantization options (FP16, 8-bit, 4-bit)
- Model caching for faster inference
- Image caching for repeated analyses
- Anti-hallucination guardrails for scene detection

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
"""

import os
import gc
import sys
import json
import hashlib
import threading
import time as time_module
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from comfy_api.latest import io as comfy_io
import folder_paths
import comfy.model_management

from .llm_model_type import LLMModelConfig


from .base_llm_provider import BaseLLMProvider


def check_interrupted():
    """Check if processing was interrupted by user and raise exception if so."""
    comfy.model_management.throw_exception_if_processing_interrupted()

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


# =============================================================================
# Speed Optimizations - Global Setup
# =============================================================================

def setup_cuda_optimizations():
    """
    DISABLED: Global CUDA optimizations removed.

    We no longer set global CUDA settings because they interfere with ComfyUI:

    1. torch.backends.cudnn.benchmark = True
       - Caches convolution algorithms for specific input sizes
       - Causes re-benchmarking overhead when switching between our models and KSampler
       - Results in slower diffusion sampling after local model inference

    2. torch.backends.cuda.matmul.allow_tf32 = True
       - Changes precision globally, may affect other nodes' outputs

    3. torch.set_grad_enabled(False)
       - Breaks VAE decode and other nodes that need gradients

    Instead, optimizations should be applied ONLY within our inference context
    and restored afterward. ComfyUI manages its own CUDA settings.
    """
    # Intentionally empty - do not modify global PyTorch/CUDA state
    pass


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
    # Text-only models
    QWEN_TEXT = "qwen_text"
    QWEN3_TEXT = "qwen3_text"  # Qwen3 text models (latest, optimized for prompts)
    LLAMA_TEXT = "llama_text"
    PHI_TEXT = "phi_text"
    MISTRAL_TEXT = "mistral_text"
    GEMMA_TEXT = "gemma_text"


class ModelType(Enum):
    """Model capability type."""
    VISION = "vision"   # Vision-only (requires images)
    TEXT = "text"       # Text-only (no image support)
    BOTH = "both"       # Supports both vision and text


@dataclass
class LocalModelInfo:
    """Information about a local model."""
    name: str
    repo_id: str
    family: ModelFamily
    vram_fp16: float  # GB
    vram_8bit: float  # GB
    vram_4bit: float  # GB
    model_type: ModelType = ModelType.VISION  # Vision, Text, or Both
    is_fp8: bool = False
    is_thinking: bool = False
    max_output_tokens: int = 4096
    description: str = ""
    model_class: str = ""  # HuggingFace model class
    target_image_size: int = 384  # Target image size for Standard mode (pixels)


# =============================================================================
# Model Registry - All supported local models
# =============================================================================

LOCAL_MODELS: Dict[str, LocalModelInfo] = {
    # =========================================================================
    # QwenVL Series - Full Featured VLM (Vision)
    # =========================================================================
    "Qwen3-VL-2B-Instruct": LocalModelInfo(
        name="Qwen3-VL 2B (Vision) | 4K | 1.5GB",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Fast vision model",
        model_class="AutoModelForVision2Seq",
        target_image_size=320
    ),
    "Qwen3-VL-4B-Instruct": LocalModelInfo(
        name="Qwen3-VL 4B (Vision) | 4K | 2GB [Recommended]",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Best balance for vision",
        model_class="AutoModelForVision2Seq",
        target_image_size=384
    ),
    "Qwen3-VL-8B-Instruct": LocalModelInfo(
        name="Qwen3-VL 8B (Vision) | 4K | 4.5GB",
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=12.0, vram_8bit=7.0, vram_4bit=4.5,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="High quality vision",
        model_class="AutoModelForVision2Seq",
        target_image_size=448
    ),
    "Qwen3-VL-2B-Thinking": LocalModelInfo(
        name="Qwen3-VL 2B Thinking (Vision) | 4K | 1.5GB",
        repo_id="Qwen/Qwen3-VL-2B-Thinking",
        family=ModelFamily.QWENVL,
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        model_type=ModelType.VISION,
        is_thinking=True,
        max_output_tokens=4096,
        description="Fast reasoning vision",
        model_class="AutoModelForVision2Seq",
        target_image_size=320
    ),
    "Qwen3-VL-4B-Thinking": LocalModelInfo(
        name="Qwen3-VL 4B Thinking (Vision) | 4K | 2GB",
        repo_id="Qwen/Qwen3-VL-4B-Thinking",
        family=ModelFamily.QWENVL,
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        model_type=ModelType.VISION,
        is_thinking=True,
        max_output_tokens=4096,
        description="Balanced reasoning vision",
        model_class="AutoModelForVision2Seq",
        target_image_size=384
    ),
    "Qwen3-VL-8B-Thinking": LocalModelInfo(
        name="Qwen3-VL 8B Thinking (Vision) | 4K | 4.5GB",
        repo_id="Qwen/Qwen3-VL-8B-Thinking",
        family=ModelFamily.QWENVL,
        vram_fp16=12.0, vram_8bit=7.0, vram_4bit=4.5,
        model_type=ModelType.VISION,
        is_thinking=True,
        max_output_tokens=4096,
        description="Best reasoning vision",
        model_class="AutoModelForVision2Seq",
        target_image_size=448
    ),
    "Qwen2.5-VL-3B-Instruct": LocalModelInfo(
        name="Qwen2.5-VL 3B (Vision) | 4K | 2GB",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=5.0, vram_8bit=3.0, vram_4bit=2.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Compact vision model",
        model_class="AutoModelForVision2Seq",
        target_image_size=352
    ),
    "Qwen2.5-VL-7B-Instruct": LocalModelInfo(
        name="Qwen2.5-VL 7B (Vision) | 4K | 4GB",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=10.0, vram_8bit=6.0, vram_4bit=4.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Stable quality vision",
        model_class="AutoModelForVision2Seq",
        target_image_size=416
    ),

    # QwenVL FP8 Pre-quantized Models
    "Qwen3-VL-2B-Instruct-FP8": LocalModelInfo(
        name="Qwen3-VL 2B FP8 (Vision) | 4K | 2.5GB",
        repo_id="Qwen/Qwen3-VL-2B-Instruct-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=2.5, vram_8bit=2.5, vram_4bit=2.5,
        model_type=ModelType.VISION,
        is_fp8=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8",
        model_class="AutoModelForVision2Seq",
        target_image_size=320
    ),
    "Qwen3-VL-2B-Thinking-FP8": LocalModelInfo(
        name="Qwen3-VL 2B Thinking FP8 (Vision) | 4K | 2.5GB",
        repo_id="Qwen/Qwen3-VL-2B-Thinking-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=2.5, vram_8bit=2.5, vram_4bit=2.5,
        model_type=ModelType.VISION,
        is_fp8=True,
        is_thinking=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8 reasoning",
        model_class="AutoModelForVision2Seq",
        target_image_size=320
    ),
    "Qwen3-VL-4B-Instruct-FP8": LocalModelInfo(
        name="Qwen3-VL 4B FP8 (Vision) | 4K | 2.5GB",
        repo_id="Qwen/Qwen3-VL-4B-Instruct-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=2.5, vram_8bit=2.5, vram_4bit=2.5,
        model_type=ModelType.VISION,
        is_fp8=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8",
        model_class="AutoModelForVision2Seq",
        target_image_size=384
    ),
    "Qwen3-VL-4B-Thinking-FP8": LocalModelInfo(
        name="Qwen3-VL 4B Thinking FP8 (Vision) | 4K | 2.5GB",
        repo_id="Qwen/Qwen3-VL-4B-Thinking-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=2.5, vram_8bit=2.5, vram_4bit=2.5,
        model_type=ModelType.VISION,
        is_fp8=True,
        is_thinking=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8 reasoning",
        model_class="AutoModelForVision2Seq",
        target_image_size=384
    ),
    "Qwen3-VL-8B-Instruct-FP8": LocalModelInfo(
        name="Qwen3-VL 8B FP8 (Vision) | 4K | 7.5GB",
        repo_id="Qwen/Qwen3-VL-8B-Instruct-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=7.5, vram_8bit=7.5, vram_4bit=7.5,
        model_type=ModelType.VISION,
        is_fp8=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8",
        model_class="AutoModelForVision2Seq",
        target_image_size=448
    ),
    "Qwen3-VL-8B-Thinking-FP8": LocalModelInfo(
        name="Qwen3-VL 8B Thinking FP8 (Vision) | 4K | 7.5GB",
        repo_id="Qwen/Qwen3-VL-8B-Thinking-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=7.5, vram_8bit=7.5, vram_4bit=7.5,
        model_type=ModelType.VISION,
        is_fp8=True,
        is_thinking=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8 reasoning",
        model_class="AutoModelForVision2Seq",
        target_image_size=448
    ),

    # QwenVL 32B Models (Large)
    "Qwen3-VL-32B-Instruct": LocalModelInfo(
        name="Qwen3-VL 32B (Vision) | 4K | 8.5GB",
        repo_id="Qwen/Qwen3-VL-32B-Instruct",
        family=ModelFamily.QWENVL,
        vram_fp16=28.0, vram_8bit=14.0, vram_4bit=8.5,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Largest Qwen3-VL",
        model_class="AutoModelForVision2Seq",
        target_image_size=512
    ),
    "Qwen3-VL-32B-Thinking": LocalModelInfo(
        name="Qwen3-VL 32B Thinking (Vision) | 4K | 8.5GB",
        repo_id="Qwen/Qwen3-VL-32B-Thinking",
        family=ModelFamily.QWENVL,
        vram_fp16=28.0, vram_8bit=14.0, vram_4bit=8.5,
        model_type=ModelType.VISION,
        is_thinking=True,
        max_output_tokens=4096,
        description="Best reasoning vision",
        model_class="AutoModelForVision2Seq",
        target_image_size=512
    ),
    "Qwen3-VL-32B-Instruct-FP8": LocalModelInfo(
        name="Qwen3-VL 32B FP8 (Vision) | 4K | 24GB",
        repo_id="Qwen/Qwen3-VL-32B-Instruct-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=24.0, vram_8bit=24.0, vram_4bit=24.0,
        model_type=ModelType.VISION,
        is_fp8=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8 large",
        model_class="AutoModelForVision2Seq",
        target_image_size=512
    ),
    "Qwen3-VL-32B-Thinking-FP8": LocalModelInfo(
        name="Qwen3-VL 32B Thinking FP8 (Vision) | 4K | 24GB",
        repo_id="Qwen/Qwen3-VL-32B-Thinking-FP8",
        family=ModelFamily.QWENVL,
        vram_fp16=24.0, vram_8bit=24.0, vram_4bit=24.0,
        model_type=ModelType.VISION,
        is_fp8=True,
        is_thinking=True,
        max_output_tokens=4096,
        description="Pre-quantized FP8 reasoning large",
        model_class="AutoModelForVision2Seq",
        target_image_size=512
    ),

    # =========================================================================
    # TEXT-ONLY MODELS - For Prompt Enhancement
    # =========================================================================

    # Qwen2.5 Text Models - Excellent for prompt enhancement
    "Qwen2.5-1.5B-Instruct": LocalModelInfo(
        name="Qwen2.5 1.5B (Text) | 32K | 1GB [Fast]",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        family=ModelFamily.QWEN_TEXT,
        vram_fp16=3.0, vram_8bit=2.0, vram_4bit=1.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Ultra-fast text model",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen2.5-3B-Instruct": LocalModelInfo(
        name="Qwen2.5 3B (Text) | 32K | 2GB [Recommended]",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        family=ModelFamily.QWEN_TEXT,
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Best balance for text",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen2.5-7B-Instruct": LocalModelInfo(
        name="Qwen2.5 7B (Text) | 32K | 4GB [Quality]",
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        family=ModelFamily.QWEN_TEXT,
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=4.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="High quality text",
        model_class="AutoModelForCausalLM"
    ),

    # Qwen3 Text Models - Latest generation, optimized for prompt generation
    # Note: Qwen3 base models are instruction-tuned (no separate -Instruct versions)
    "Qwen3-0.6B": LocalModelInfo(
        name="Qwen3 0.6B (Text) | 32K | 0.5GB [Ultra-Fast]",
        repo_id="Qwen/Qwen3-0.6B",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=1.2, vram_8bit=0.8, vram_4bit=0.5,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Ultra-fast prompt generation",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen3-1.7B": LocalModelInfo(
        name="Qwen3 1.7B (Text) | 32K | 1.4GB [Fast]",
        repo_id="Qwen/Qwen3-1.7B",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=3.5, vram_8bit=2.2, vram_4bit=1.4,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Fast prompt generation",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen3-4B": LocalModelInfo(
        name="Qwen3 4B (Text) | 32K | 3GB [Balanced]",
        repo_id="Qwen/Qwen3-4B",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=8.0, vram_8bit=5.0, vram_4bit=3.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Balanced prompt generation",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen3-8B": LocalModelInfo(
        name="Qwen3 8B (Text) | 32K | 6GB [Quality]",
        repo_id="Qwen/Qwen3-8B",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=16.0, vram_8bit=10.0, vram_4bit=6.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="High quality prompts",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen3-4B-Instruct-2507": LocalModelInfo(
        name="Qwen3 4B 2507 (Text) | 32K | 3GB [Latest]",
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=8.0, vram_8bit=5.0, vram_4bit=3.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Latest Qwen3 (Aug 2025)",
        model_class="AutoModelForCausalLM"
    ),

    # Llama 3.2 Text Models
    "Llama-3.2-1B-Instruct": LocalModelInfo(
        name="Llama 3.2 1B (Text) | 128K | 0.8GB",
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        family=ModelFamily.LLAMA_TEXT,
        vram_fp16=2.0, vram_8bit=1.2, vram_4bit=0.8,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Smallest Llama, very fast",
        model_class="AutoModelForCausalLM"
    ),
    "Llama-3.2-3B-Instruct": LocalModelInfo(
        name="Llama 3.2 3B (Text) | 128K | 2GB",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        family=ModelFamily.LLAMA_TEXT,
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Fast with huge context",
        model_class="AutoModelForCausalLM"
    ),

    # Phi-3.5 Mini (Text only)
    "Phi-3.5-mini-instruct": LocalModelInfo(
        name="Phi-3.5 Mini (Text) | 128K | 2.5GB",
        repo_id="microsoft/Phi-3.5-mini-instruct",
        family=ModelFamily.PHI_TEXT,
        vram_fp16=7.5, vram_8bit=4.5, vram_4bit=2.5,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Strong reasoning",
        model_class="AutoModelForCausalLM"
    ),

    # Mistral 7B (Text only)
    "Mistral-7B-Instruct-v0.3": LocalModelInfo(
        name="Mistral 7B (Text) | 32K | 4GB [Creative]",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        family=ModelFamily.MISTRAL_TEXT,
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=4.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Creative writing",
        model_class="AutoModelForCausalLM"
    ),

    # Gemma 2 (Text only)
    "Gemma-2-2B-it": LocalModelInfo(
        name="Gemma 2 2B (Text) | 8K | 1.5GB",
        repo_id="google/gemma-2-2b-it",
        family=ModelFamily.GEMMA_TEXT,
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        model_type=ModelType.TEXT,
        max_output_tokens=2048,
        description="Efficient text model",
        model_class="AutoModelForCausalLM"
    ),
}


# =============================================================================
# Custom Models Loading
# =============================================================================

NODE_DIR = Path(__file__).parent.parent  # ComfyUI-AI-Photography-Toolkit directory
CUSTOM_MODELS_PATH = NODE_DIR / "custom_models.json"


def load_custom_models():
    """
    Load custom models from custom_models.json if it exists.
    Users can define their own HuggingFace models by copying
    custom_models_example.json to custom_models.json and modifying it.
    """
    global LOCAL_MODELS

    if not CUSTOM_MODELS_PATH.exists():
        return

    try:
        with open(CUSTOM_MODELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both "hf_models" and "models" keys for flexibility
        models = data.get("hf_models", {}) or data.get("models", {})

        if not models:
            return

        count = 0
        for model_name, model_info in models.items():
            # Skip comments and metadata
            if model_name.startswith("_"):
                continue

            # Parse family
            family_str = model_info.get("family", "qwenvl").lower()
            family_map = {
                "qwenvl": ModelFamily.QWENVL,
                "qwen_text": ModelFamily.QWEN_TEXT,
                "qwen3_text": ModelFamily.QWEN3_TEXT,
                "llama_text": ModelFamily.LLAMA_TEXT,
                "phi_text": ModelFamily.PHI_TEXT,
                "mistral_text": ModelFamily.MISTRAL_TEXT,
                "gemma_text": ModelFamily.GEMMA_TEXT,
            }
            family = family_map.get(family_str, ModelFamily.QWENVL)

            # Parse model type
            model_type_str = model_info.get("model_type", "vision").lower()
            model_type_map = {
                "vision": ModelType.VISION,
                "text": ModelType.TEXT,
                "both": ModelType.BOTH,
            }
            model_type = model_type_map.get(model_type_str, ModelType.VISION)

            # Parse VRAM requirements
            vram = model_info.get("vram_requirement", {})

            # Create LocalModelInfo
            LOCAL_MODELS[model_name] = LocalModelInfo(
                name=f"{model_name} (Custom)",
                repo_id=model_info.get("repo_id", ""),
                family=family,
                vram_fp16=vram.get("full", 10.0),
                vram_8bit=vram.get("8bit", 6.0),
                vram_4bit=vram.get("4bit", 4.0),
                model_type=model_type,
                is_fp8=model_info.get("is_fp8", False),
                is_thinking=model_info.get("is_thinking", False),
                max_output_tokens=model_info.get("max_output_tokens", 4096),
                description=model_info.get("description", "Custom model"),
                model_class=model_info.get("model_class", "AutoModelForVision2Seq"),
            )
            count += 1

        if count > 0:
            print(f"[SID_LLM_Local] Loaded {count} custom model(s) from custom_models.json")

    except Exception as e:
        print(f"[SID_LLM_Local] Warning: Failed to load custom_models.json: {e}")


# Load custom models on module import
load_custom_models()


def free_comfyui_memory(required_gb: float = 0) -> float:
    """
    Free ComfyUI's cached models to make VRAM available.

    Args:
        required_gb: Amount of VRAM needed (will try to free this much)

    Returns:
        Available VRAM in GB after cleanup
    """
    import torch

    try:
        # First check if we already have enough
        if torch.cuda.is_available():
            current_free = get_available_vram_gb()
            if current_free >= required_gb:
                return current_free

        # Ask ComfyUI to unload models
        print(f"[SID_LLM_Local] Freeing VRAM for local LLM (need {required_gb:.1f}GB)...")

        # Soft empty cache first (unloads models to CPU if possible)
        comfy.model_management.soft_empty_cache()

        # Force garbage collection
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Check again
        if torch.cuda.is_available():
            current_free = get_available_vram_gb()
            if current_free >= required_gb:
                print(f"[SID_LLM_Local] Freed VRAM: {current_free:.1f}GB available")
                return current_free

        # If still not enough, try harder - unload all models
        try:
            comfy.model_management.unload_all_models()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[SID_LLM_Local] Warning: Could not unload all models: {e}")

        if torch.cuda.is_available():
            current_free = get_available_vram_gb()
            print(f"[SID_LLM_Local] After full cleanup: {current_free:.1f}GB available")
            return current_free

    except Exception as e:
        print(f"[SID_LLM_Local] Warning: Memory cleanup failed: {e}")

    return get_available_vram_gb()


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


def enforce_memory(model_info: "LocalModelInfo", quantization: str, device: str = "auto") -> str:
    """
    Enforce memory requirements and auto-downgrade quantization if needed.

    Args:
        model_info: LocalModelInfo with VRAM requirements
        quantization: Requested quantization ("4-bit", "8-bit", "None (FP16)")
        device: Target device ("auto", "cuda", "cpu", "mps")

    Returns:
        Potentially downgraded quantization string

    Raises:
        RuntimeError: If even 4-bit quantization won't fit in available memory
    """
    import torch

    # FP8 pre-quantized models don't need further quantization
    if model_info.is_fp8:
        return quantization

    # Determine actual device
    if device == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            actual_device = "mps"
        else:
            actual_device = "cpu"
    else:
        actual_device = device

    # CPU doesn't have VRAM constraints (but is slow)
    if actual_device == "cpu":
        return quantization

    # Safety margin (require 20% extra)
    safety_margin = 1.2

    # Get VRAM requirements based on quantization
    vram_map = {
        "None (FP16)": model_info.vram_fp16,
        "8-bit": model_info.vram_8bit,
        "4-bit": model_info.vram_4bit,
    }

    current_quant = quantization
    needed = vram_map.get(current_quant, model_info.vram_4bit) * safety_margin

    # Get initial available memory
    if actual_device == "cuda":
        available = get_available_vram_gb()
    elif actual_device == "mps":
        try:
            import psutil
            available = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available = 16.0
    else:
        available = 0

    # If not enough memory, try to free ComfyUI's cached models first
    if needed > available and actual_device == "cuda":
        available = free_comfyui_memory(needed)

    # Auto-downgrade if needed
    if needed > available:
        if current_quant == "None (FP16)":
            current_quant = "8-bit"
            needed = model_info.vram_8bit * safety_margin

    if needed > available:
        if current_quant == "8-bit":
            current_quant = "4-bit"
            needed = model_info.vram_4bit * safety_margin

    # Final check - try one more cleanup if still not enough
    if needed > available and actual_device == "cuda":
        available = free_comfyui_memory(needed)

    if needed > available:
        raise RuntimeError(
            f"Insufficient memory for model '{model_info.name}'.\n"
            f"Required: {needed:.1f}GB (4-bit), Available: {available:.1f}GB\n"
            f"Try closing other applications or using a smaller model."
        )

    return current_quant


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

    # Session management - tracks whether to unload model at end of node execution
    _session_active = False
    _session_keep_loaded = False  # True if any client in this session requested keep_model_loaded

    @classmethod
    def start_session(cls):
        """Start a new session. Call at beginning of node execution."""
        cls._session_active = True
        cls._session_keep_loaded = False
        print("[LocalModel] Session started")

    @classmethod
    def end_session(cls):
        """End the session and unload model if it was loaded with keep_model_loaded=True.

        Call at end of node execution to cleanup models that were kept loaded
        for the duration of the session (across loop iterations).
        """
        if cls._session_active:
            cls._session_active = False
            if cls._session_keep_loaded and cls._cached_model is not None:
                print("[LocalModel] Session ended - unloading session-cached model")
                cls.unload_model()
            else:
                print("[LocalModel] Session ended - no model to unload")
            cls._session_keep_loaded = False

    @classmethod
    def unload_model(cls):
        """Force unload the cached model to free VRAM for other nodes."""
        if cls._cached_model is not None:
            print("[LocalModel] Force unloading model from VRAM...")
            # Delete model references explicitly
            try:
                # Move model to CPU first to free VRAM faster
                if hasattr(cls._cached_model, 'to'):
                    cls._cached_model.to('cpu')
            except:
                pass
            try:
                del cls._cached_model
            except:
                pass
            try:
                del cls._cached_processor
            except:
                pass
            try:
                del cls._cached_tokenizer
            except:
                pass
            cls._cached_model = None
            cls._cached_processor = None
            cls._cached_tokenizer = None
            cls._cached_signature = None
            cls._compiled_generate = None
            cls._image_cache.clear()
            # Force immediate garbage collection - run multiple times
            gc.collect()
            gc.collect()
            gc.collect()
            # Aggressively clear memory
            clear_memory()
            print("[LocalModel] Model unloaded successfully")
            return True
        return False

    def __init__(
        self,
        model_name: str,
        quantization: str = "4-bit",
        device: str = "auto",
        attention_mode: str = "auto",
        keep_model_loaded: bool = True,
        temperature: float = 0.7,
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
        self.temperature = temperature  # Store user's temperature setting
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.use_torch_compile = use_torch_compile

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_info = LOCAL_MODELS.get(model_name)

    def _get_device_info(self) -> Dict[str, Any]:
        """Detect GPU type and available memory."""
        import torch

        info = {"type": "cpu", "available_gb": 0, "total_gb": 0, "gpu_name": "N/A"}

        # Check CUDA
        if torch.cuda.is_available():
            info["type"] = "cuda"
            props = torch.cuda.get_device_properties(0)
            info["total_gb"] = props.total_memory / (1024**3)
            info["available_gb"] = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            info["gpu_name"] = props.name
            info["compute_capability"] = f"{props.major}.{props.minor}"

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["type"] = "mps"
            info["total_gb"] = 16  # Estimate
            info["available_gb"] = 16
            info["gpu_name"] = "Apple Silicon (MPS)"

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
                self.quantization = "8-bit"

        if self.quantization == "8-bit":
            needed = model_info.vram_8bit * safety_margin
            if needed > available:
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
            self.model = LocalModelClient._cached_model
            self.processor = LocalModelClient._cached_processor
            self.tokenizer = LocalModelClient._cached_tokenizer
            return

        # Clear memory before loading new model - CRITICAL for preventing memory leaks
        if LocalModelClient._cached_model is not None:
            print("[LocalModel] Unloading previous model before loading new one...")
            # Delete references explicitly
            try:
                del LocalModelClient._cached_model
            except:
                pass
            try:
                del LocalModelClient._cached_processor
            except:
                pass
            try:
                del LocalModelClient._cached_tokenizer
            except:
                pass
            LocalModelClient._cached_model = None
            LocalModelClient._cached_processor = None
            LocalModelClient._cached_tokenizer = None
            LocalModelClient._cached_signature = None
            LocalModelClient._compiled_generate = None
            LocalModelClient._image_cache.clear()
            # Force immediate garbage collection
            gc.collect()
            gc.collect()  # Run twice to catch cyclic references
            clear_memory()

        if not self.model_info:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"[LocalModel] Loading {self.model_name}...")

        # Download model
        family_dir = os.path.join(LLM_LOCAL_DIR, self.model_info.family.value)
        os.makedirs(family_dir, exist_ok=True)
        model_path = os.path.join(family_dir, self.model_name)

        # Check if model files actually exist (not just empty folder from failed download)
        def has_model_files(path):
            if not os.path.exists(path):
                return False
            for f in os.listdir(path):
                if f.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                    return True
            return False

        if not has_model_files(model_path):
            # Remove empty/incomplete folder if it exists
            if os.path.exists(model_path):
                import shutil
                print(f"[LocalModel] Removing incomplete download at {model_path}")
                shutil.rmtree(model_path)

            print(f"[LocalModel] Downloading from {self.model_info.repo_id}...")
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

        # Auto-downgrade quantization
        if not self.model_info.is_fp8:
            self._auto_downgrade_quantization(self.model_info, device_info)

        # Load based on model family
        if self.model_info.family == ModelFamily.QWENVL:
            self._load_qwenvl(model_path, device)
        elif self.model_info.family in [ModelFamily.QWEN_TEXT, ModelFamily.QWEN3_TEXT,
                                         ModelFamily.LLAMA_TEXT, ModelFamily.PHI_TEXT,
                                         ModelFamily.MISTRAL_TEXT, ModelFamily.GEMMA_TEXT]:
            self._load_text_model(model_path, device)

        # Cache for reuse
        if self.keep_model_loaded:
            LocalModelClient._cached_model = self.model
            LocalModelClient._cached_processor = self.processor
            LocalModelClient._cached_tokenizer = self.tokenizer
            LocalModelClient._cached_signature = signature

        print(f"[LocalModel] Ready ({device.upper()}, {self.quantization}, temp={self.temperature})")

    def _get_quantization_config(self, device: str):
        """
        Get BitsAndBytes quantization config with optimal dtype.
        Uses BF16 on Ampere+ GPUs for better performance.
        """
        import torch

        # Get optimal dtype for this GPU
        optimal_dtype = get_optimal_dtype(device)

        if device != "cuda" or self.quantization == "None (FP16)":
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

    def _clear_transformers_cache(self, model_name: str):
        """
        Clear corrupted transformers_modules cache for a specific model.
        This helps recover from incomplete downloads or cache corruption.
        Also clears Python's sys.modules cache to prevent stale imports.
        """
        import shutil
        import sys
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
        if cache_dir.exists():
            # Find and remove directories matching the model name
            for item in cache_dir.iterdir():
                # Match variations like "Moondream2", "moondream2", "vikhyatk_moondream2"
                if model_name.lower() in item.name.lower():
                    try:
                        shutil.rmtree(item)
                        print(f"[LocalModelClient] Cleared corrupted cache: {item.name}")
                    except Exception as e:
                        print(f"[LocalModelClient] Warning: Could not clear cache {item.name}: {e}")

        # Clear any related entries from sys.modules to prevent stale imports
        modules_to_remove = [
            key for key in sys.modules.keys()
            if model_name.lower() in key.lower() or "transformers_modules" in key
        ]
        for mod in modules_to_remove:
            try:
                del sys.modules[mod]
            except Exception:
                pass
        if modules_to_remove:
            print(f"[LocalModelClient] Cleared {len(modules_to_remove)} cached Python modules")

    def _load_qwenvl(self, model_path: str, device: str):
        """Load QwenVL model with speed optimizations."""
        import importlib.metadata
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

        # Check if this is an FP8 pre-quantized model
        is_fp8_model = "FP8" in self.model_name or "fp8" in model_path.lower()

        # FP8 models don't need additional quantization
        if is_fp8_model:
            quant_config, dtype = None, None
            print(f"[LocalModel] FP8 pre-quantized model detected, skipping quantization")
        else:
            quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }

        # FP8 models need different loading strategy to avoid meta tensor issues
        if is_fp8_model:
            # Don't use low_cpu_mem_usage with FP8 - causes meta tensor errors
            load_kwargs["low_cpu_mem_usage"] = False
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        # Select optimal attention implementation
        actual_attention_mode = self.attention_mode
        if self.attention_mode != "auto":
            load_kwargs["attn_implementation"] = self.attention_mode
        elif device == "cuda":
            try:
                # Check both import AND package metadata (transformers requires metadata)
                import flash_attn
                importlib.metadata.version("flash_attn")  # This will raise if metadata missing
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    actual_attention_mode = "flash_attention_2"
                else:
                    load_kwargs["attn_implementation"] = "sdpa"
                    actual_attention_mode = "sdpa"
            except (ImportError, importlib.metadata.PackageNotFoundError):
                load_kwargs["attn_implementation"] = "sdpa"
                actual_attention_mode = "sdpa"

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            elif not is_fp8_model:
                # Only set dtype for non-FP8 models
                load_kwargs["dtype"] = dtype or torch.float16

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


        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _load_text_model(self, model_path: str, device: str):
        """Load text-only model (Qwen2.5, Llama, Phi, Mistral, Gemma)."""
        import importlib.metadata
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Select optimal attention implementation
        if device == "cuda":
            try:
                import flash_attn
                importlib.metadata.version("flash_attn")
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
                load_kwargs["dtype"] = dtype or torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        # Enable KV cache for faster generation
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = None  # Text models don't need a processor

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text-only response (for prompt enhancement)."""
        # Check for interrupt before text generation
        check_interrupted()

        import torch

        if self.model is None:
            self._load_model()

        # Check if this is a text model
        if self.model_info.model_type == ModelType.VISION:
            raise ValueError(
                f"Model '{self.model_name}' is a vision-only model. "
                "Please use a (Text) or (Both) model for text generation."
            )

        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for tokenizers without chat template
            text = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Generate
        with InferenceProgressSpinner("Generating text"):
            with torch.inference_mode():
                # Build generation kwargs - only include sampling params when sampling
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": self.repetition_penalty,
                }

                if temperature > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = self.top_p
                else:
                    gen_kwargs["do_sample"] = False

                outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        generated = outputs[0][input_len:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Cleanup
        self._cleanup_after_generation()

        return result.strip()

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

        try:
            if self.model is None:
                self._load_model()
        except Exception as e:
            error_msg = str(e)
            # Handle common loading errors with helpful messages
            if "403" in error_msg or "gated" in error_msg.lower():
                return LocalModelResponse(f"[ERROR] Model access denied. This is a gated model requiring HuggingFace approval. Please use a different Qwen3-VL model.")
            elif "no file named" in error_msg.lower() or "safetensors" in error_msg.lower():
                return LocalModelResponse(f"[ERROR] Model download incomplete. Please enable 'Clear Local Cache' and try again, or use a different model.")
            else:
                print(f"[LocalModel] Load error: {error_msg}")
                return LocalModelResponse(f"[ERROR] Failed to load model: {error_msg[:200]}")

        start_time = time.time()

        # Extract image and text from messages
        images = []
        text_prompt = ""

        try:
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
        except Exception as e:
            print(f"[LocalModel] Message parsing error: {e}")
            return LocalModelResponse(f"[ERROR] Failed to parse input: {str(e)[:100]}")

        # Generate with retry logic for short responses
        response_text = ""
        retry_count = 0
        current_temp = temperature

        try:
            while retry_count <= max_retries:
                gen_start = time.time()

                # Generate based on model family
                if self.model_info.family == ModelFamily.QWENVL:
                    response_text = self._generate_qwenvl(messages, images, max_tokens, current_temp)
                else:
                    return LocalModelResponse(f"[ERROR] Unknown model family: {self.model_info.family}")

                gen_elapsed = time.time() - gen_start

                # Validate response length
                word_count = len(response_text.split())
                if word_count >= min_response_words:
                    break

                # Response too short, retry with adjusted temperature
                retry_count += 1
                if retry_count <= max_retries:
                    current_temp = min(1.0, temperature + 0.2 * retry_count)

        except Exception as e:
            error_msg = str(e)
            print(f"[LocalModel] Generation error: {error_msg}")

            # Always cleanup after error
            self._cleanup_after_generation()

            # Handle specific error types with helpful messages
            if "CUDA" in error_msg or "index out of bounds" in error_msg or "scatter" in error_msg.lower():
                # Unload the model to recover from CUDA errors
                LocalModelClient.unload_model()
                return LocalModelResponse(
                    f"[ERROR] CUDA error - this model has compatibility issues with your PyTorch version. "
                    f"Please try a different Qwen3-VL model. Error: {error_msg[:100]}"
                )
            elif "out of memory" in error_msg.lower() or "OOM" in error_msg:
                LocalModelClient.unload_model()
                return LocalModelResponse(
                    f"[ERROR] Out of GPU memory. Try a smaller model or enable 4-bit quantization."
                )
            elif "NoneType" in error_msg:
                return LocalModelResponse(
                    f"[ERROR] Model initialization error - tokenizer or processor not loaded correctly. "
                    f"Please try a different Qwen3-VL model."
                )
            else:
                return LocalModelResponse(f"[ERROR] Generation failed: {error_msg[:200]}")

        elapsed = time.time() - start_time
        word_count = len(response_text.split())

        # CRITICAL: Clean up GPU memory after generation to prevent VAE decode hangs
        # This ensures CUDA operations are complete and memory is freed for subsequent nodes
        self._cleanup_after_generation()

        return LocalModelResponse(response_text.strip())

    def _cleanup_after_generation(self):
        """
        Clean up GPU state and memory after generation.

        This is critical to prevent ComfyUI's subsequent nodes (like VAE decode)
        from getting stuck due to:
        - Pending CUDA operations
        - Fragmented GPU memory
        - Disabled gradient computation
        - Unreleased tensor references

        Session-based model management:
        - keep_model_loaded=True: Model stays cached for the session (multiple loop iterations),
          unloads when session ends (node execution completes)
        - keep_model_loaded=False: Model unloads after each inference call
        """
        import torch

        try:
            # CRITICAL: Ensure gradient computation is enabled for other nodes
            # Some transformers models or generation code may disable gradients
            # and VAE/other nodes may need them enabled
            torch.set_grad_enabled(True)

            # Session-based model management
            if self.keep_model_loaded:
                # Mark that this session has a model to unload at session end
                LocalModelClient._session_keep_loaded = True
                # Model stays cached - will be unloaded when end_session() is called
            else:
                # Unload immediately after this inference
                self._unload_model_instance()

            # Clear image cache to prevent memory buildup
            if len(LocalModelClient._image_cache) > LocalModelClient._image_cache_max_size:
                LocalModelClient._image_cache.clear()

            if torch.cuda.is_available():
                # Synchronize to ensure all CUDA operations are complete
                torch.cuda.synchronize()

                # Clear CUDA cache to free up memory for VAE
                torch.cuda.empty_cache()

            # Force garbage collection to release any dangling references
            gc.collect()
            gc.collect()

        except Exception as e:
            print(f"[LocalModel] Cleanup warning: {e}")

    def _unload_model_instance(self):
        """Unload model from this instance to free VRAM."""
        import torch

        try:
            print("[LocalModel] Unloading model instance from VRAM...")

            # Move model to CPU first before deletion (helps with VRAM release)
            if hasattr(self, 'model') and self.model is not None:
                try:
                    if hasattr(self.model, 'to'):
                        self.model.to('cpu')
                except:
                    pass
                try:
                    del self.model
                except:
                    pass
                self.model = None

            if hasattr(self, 'processor') and self.processor is not None:
                try:
                    del self.processor
                except:
                    pass
                self.processor = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                try:
                    del self.tokenizer
                except:
                    pass
                self.tokenizer = None

            # Clear class-level cache as well - move to CPU first
            if LocalModelClient._cached_model is not None:
                try:
                    if hasattr(LocalModelClient._cached_model, 'to'):
                        LocalModelClient._cached_model.to('cpu')
                except:
                    pass
                try:
                    del LocalModelClient._cached_model
                except:
                    pass

            if LocalModelClient._cached_processor is not None:
                try:
                    del LocalModelClient._cached_processor
                except:
                    pass

            if LocalModelClient._cached_tokenizer is not None:
                try:
                    del LocalModelClient._cached_tokenizer
                except:
                    pass

            LocalModelClient._cached_model = None
            LocalModelClient._cached_processor = None
            LocalModelClient._cached_tokenizer = None
            LocalModelClient._cached_signature = None
            LocalModelClient._compiled_generate = None

            # Clear image cache
            LocalModelClient._image_cache.clear()

            # Force garbage collection - multiple passes for thorough cleanup
            gc.collect()
            gc.collect()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print("[LocalModel] Model unloaded from VRAM")

        except Exception as e:
            print(f"[LocalModel] Warning: Error during model unload: {e}")


    def _generate_qwenvl(self, messages: List, images: List, max_tokens: int, temperature: float) -> str:
        """Generate with QwenVL."""
        # Check for interrupt before QwenVL generation
        check_interrupted()

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
        raw_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Clean thinking output for Thinking models
        # Note: Non-thinking models get cleaned later in zimage_clean() via strip_reasoning_output()
        if self.model_info.is_thinking:
            return self._clean_thinking_output(raw_output)
        return raw_output

    def _clean_thinking_output(self, text: str) -> str:
        """
        Clean output from Thinking models by removing chain-of-thought reasoning.

        Qwen3-VL-*-Thinking models output their reasoning before the actual answer.
        This can be wrapped in <think>...</think> tags or appear as plain text.
        """
        import re

        if not text:
            return text

        original_text = text

        # Method 1: Check for <think>...</think> pattern
        # First try to get content AFTER </think> tag
        think_end_match = re.search(r'</think>\s*(.+)', text, re.DOTALL | re.IGNORECASE)
        if think_end_match:
            after_think = think_end_match.group(1).strip()
            if after_think and len(after_think) > 20:
                text = after_think
            else:
                # If nothing useful after </think>, extract from inside the tags
                think_content_match = re.search(r'<think>([\s\S]*?)</think>', text, re.IGNORECASE)
                if think_content_match:
                    text = think_content_match.group(1).strip()
        else:
            # No </think> tag - just remove any <think> tags if present
            text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE).strip()

        # If we still have no text, return original
        if not text.strip():
            return original_text

        # Common reasoning patterns to skip
        reasoning_markers = [
            r'^(?:Let me|Let\'s|We are given|First,|Step \d|Steps:)',
            r'^(?:I need to|I\'ll|I will|Now,|So,|Therefore|However)',
            r'^(?:Looking at|Analyzing|Checking|Based on)',
            r'^(?:The user|The original|The prompt|Given)',
            r'^(?:Alright|Okay|OK,)',
        ]

        # Check if output still looks like reasoning
        first_100_chars = text[:100].strip()
        is_reasoning = any(re.match(pattern, first_100_chars, re.IGNORECASE) for pattern in reasoning_markers)

        if is_reasoning:
            # Try to find the actual answer after reasoning

            # Look for JSON block (common in agentic mode)
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
            if json_match:
                return json_match.group(1)

            # Look for raw JSON object with prompt_description
            json_match = re.search(r'(\{[\s\S]*"prompt_description"[\s\S]*?\})', text)
            if json_match:
                return json_match.group(1)

            # Look for any JSON object
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text)
            if json_match:
                try:
                    import json
                    json.loads(json_match.group(1))  # Validate it's valid JSON
                    return json_match.group(1)
                except:
                    pass

            # Look for text after common delimiters
            delimiters = [
                r'(?:Output|Result|Answer|Modified prompt|Final prompt)[:\s]*\n?(.+)',
                r'(?:Here is|Here\'s)[^:]*:\s*\n?(.+)',
                r'(?:The final|Final)[^:]*:\s*\n?(.+)',
            ]
            for delimiter in delimiters:
                match = re.search(delimiter, text, re.IGNORECASE | re.DOTALL)
                if match:
                    result = match.group(1).strip()
                    # Make sure we got something meaningful (not just more reasoning)
                    if len(result) > 50 and not any(re.match(p, result[:50], re.IGNORECASE) for p in reasoning_markers):
                        return result

            # Last resort: Take the last substantial paragraph
            # Split by double newlines and take the last non-empty block
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if paragraphs:
                for para in reversed(paragraphs):
                    # Skip short paragraphs (likely conclusions like "Done." or "That's it.")
                    if len(para) > 100:
                        return para

            # If nothing worked, return the cleaned text as-is (better than nothing)
            return text.strip()

        # If not reasoning pattern, return cleaned text
        return text.strip() if text.strip() else original_text


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

    Supports QwenVL model family:
    - Qwen3-VL: Full-featured VLM (2-32B)
    - Qwen2.5-VL: Vision models (3-7B)

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
    def _clear_all_local_cache(cls, model_name: str):
        """Clear all cached files for a model and force re-download."""
        import shutil

        model_info = LOCAL_MODELS.get(model_name)
        if not model_info:
            print(f"[SID_LLM_Local] Unknown model: {model_name}")
            return

        repo_id = model_info.repo_id
        # Convert repo_id to cache folder name (e.g., "Qwen/Qwen3-VL-2B-Instruct" -> "models--Qwen--Qwen3-VL-2B-Instruct")
        cache_folder_name = f"models--{repo_id.replace('/', '--')}"

        # HuggingFace cache locations
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        hub_cache = os.path.join(hf_cache, "hub", cache_folder_name)

        # Local LLM cache
        llm_cache = os.path.join(folder_paths.models_dir, "LLM", model_name)

        # Transformers modules cache (for models with trust_remote_code)
        modules_cache = os.path.join(hf_cache, "modules", "transformers_modules")

        cleared = []

        # Clear HuggingFace hub cache
        if os.path.exists(hub_cache):
            try:
                shutil.rmtree(hub_cache)
                cleared.append(f"HuggingFace hub: {cache_folder_name}")
            except Exception as e:
                print(f"[SID_LLM_Local] Warning: Could not clear {hub_cache}: {e}")

        # Clear local LLM cache
        if os.path.exists(llm_cache):
            try:
                shutil.rmtree(llm_cache)
                cleared.append(f"Local LLM: {model_name}")
            except Exception as e:
                print(f"[SID_LLM_Local] Warning: Could not clear {llm_cache}: {e}")

        # Clear transformers modules cache for this model
        if os.path.exists(modules_cache):
            # Get model name variants to match folder names
            model_variants = [
                model_name,
                model_name.lower(),
                repo_id.split("/")[-1],  # e.g., "Qwen3-VL-2B-Instruct"
                repo_id.split("/")[-1].replace("-", "_"),
                repo_id.split("/")[-1].replace("-", "_hyphen_"),
            ]
            for folder in os.listdir(modules_cache):
                folder_lower = folder.lower()
                if any(variant.lower() in folder_lower for variant in model_variants):
                    folder_path = os.path.join(modules_cache, folder)
                    try:
                        shutil.rmtree(folder_path)
                        cleared.append(f"Transformers modules: {folder}")
                    except Exception as e:
                        print(f"[SID_LLM_Local] Warning: Could not clear {folder_path}: {e}")

        # Also unload from memory
        LocalModelClient.unload_model()

        if cleared:
            print(f"[SID_LLM_Local] Cleared cache for {model_name}:")
            for item in cleared:
                print(f"  ✓ {item}")
            print(f"[SID_LLM_Local] Model will be re-downloaded on next use.")
        else:
            print(f"[SID_LLM_Local] No cache found for {model_name}")

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""

        # Build separate model lists for vision and text
        vision_models = []
        text_models = []
        for model_id, model_info in LOCAL_MODELS.items():
            if model_info.model_type == ModelType.VISION:
                vision_models.append(model_id)
            elif model_info.model_type == ModelType.TEXT:
                text_models.append(model_id)
            else:  # BOTH - add to vision list
                vision_models.append(model_id)

        # Ensure we have at least one option in each list
        if not text_models:
            text_models = ["(No text models available)"]

        quantization_options = [
            "Auto (Detect Best)",
            "4-bit (VRAM-friendly)",
            "8-bit (Balanced)",
            "None (FP16)",
        ]

        max_tokens_options = [
            "Low (512)",
            "Medium (2048)",
            "High (4096)",
            "Max (Model Capacity)",
        ]

        total_vram = get_total_vram_gb()
        vram_info = f" [GPU: {total_vram:.0f}GB]" if total_vram > 0 else " [No GPU]"

        return comfy_io.Schema(
            node_id="SID_LLM_Local",
            display_name="SID LLM Local",
            category="SID Photography Toolkit/LLM Providers",
            description=f"Local Transformers Models{vram_info}",
            inputs=[
                # Vision Models
                comfy_io.Combo.Input(
                    "vision_model",
                    options=vision_models,
                    default=cls.get_default_model(),
                    display_name="Vision Model",
                    tooltip="Vision model for image analysis"
                ),
                # Text Models
                comfy_io.Combo.Input(
                    "text_model",
                    options=text_models,
                    default="Qwen3-0.6B" if "Qwen3-0.6B" in text_models else text_models[0] if text_models else "",
                    display_name="Text Model",
                    tooltip="Text model (future use)"
                ),
                # Settings
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
                    default="Medium (2048)",
                    display_name="Max Tokens",
                    tooltip="Output length: Low=512, Medium=2048, High=4096, Max=Model's full capacity"
                ),
                comfy_io.Boolean.Input(
                    "keep_model_loaded",
                    default=False,
                    display_name="Keep Model Loaded",
                    tooltip="When checked: model stays loaded across loop iterations within one execution, unloads when node completes. When unchecked: model loads/unloads on each loop."
                ),
                comfy_io.Combo.Input(
                    "attention_mode",
                    options=ATTENTION_MODES,
                    default="auto",
                    display_name="Attention Mode",
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
                    display_name="Repetition Penalty",
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
                    display_name="Top P",
                    tooltip="Nucleus sampling (0.9=recommended, lower=more focused)"
                ),
                comfy_io.Int.Input(
                    "num_beams",
                    default=1,
                    min=1,
                    max=8,
                    display_name="Num Beams",
                    tooltip="Beam search width (1=sampling, >1=beam search for stable outputs)"
                ),
                comfy_io.Boolean.Input(
                    "use_torch_compile",
                    default=False,
                    display_name="Torch Compile",
                    tooltip="Enable torch.compile for faster inference (CUDA + Torch 2.1+ only)"
                ),
                comfy_io.Boolean.Input(
                    "clear_local_cache",
                    default=False,
                    display_name="Clear Local Cache",
                    tooltip="Clear all cached model files and re-download from HuggingFace. Use if model is corrupted or outdated."
                ),
                comfy_io.String.Input(
                    "hf_token",
                    default="",
                    display_name="HuggingFace Token",
                    tooltip="HuggingFace token for gated models (PaliGemma, etc). Get token from huggingface.co/settings/tokens"
                ),
                # Analysis mode - Standard or Detailed
                comfy_io.Combo.Input(
                    "analysis_mode",
                    options=["Standard", "Detailed"],
                    default="Detailed",
                    tooltip="Standard: Single-pass (1 call). Detailed: Comprehensive multi-aspect analysis (3-4 calls)"
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
        vision_model: str,
        text_model: str,
        quantization: str,
        device: str,
        temperature: float,
        max_tokens_preset: str,
        keep_model_loaded: bool,
        attention_mode: str,
        repetition_penalty: float,
        top_p: float,
        num_beams: int,
        use_torch_compile: bool,
        clear_local_cache: bool,
        hf_token: str,
        analysis_mode: str,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""
        try:
            # Clear cache if requested
            if clear_local_cache:
                cls._clear_all_local_cache(vision_model)

            # Set HuggingFace token for gated models
            if hf_token and hf_token.strip():
                os.environ["HF_TOKEN"] = hf_token.strip()
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()  # Legacy env var
                print("[SID_LLM_Local] HuggingFace token set for gated model access")

            # Use vision_model (primary use case)
            model = vision_model

            # Validate model
            model_info = LOCAL_MODELS.get(model)
            if not model_info:
                raise ValueError(f"[SID_LLM_Local] Unknown model: {model}")

            # Handle quantization (including Auto mode)
            if quantization == "Auto (Detect Best)":
                available_vram = get_available_vram_gb()
                if available_vram <= 0:
                    quant = "4-bit"
                else:
                    # Select based on available VRAM with safety margin
                    safety_margin = 1.2
                    if model_info.vram_fp16 * safety_margin <= available_vram:
                        quant = "None (FP16)"
                    elif model_info.vram_8bit * safety_margin <= available_vram:
                        quant = "8-bit"
                    else:
                        quant = "4-bit"
            else:
                quant_map = {
                    "4-bit (VRAM-friendly)": "4-bit",
                    "8-bit (Balanced)": "8-bit",
                    "None (FP16)": "None (FP16)",
                }
                quant = quant_map.get(quantization, "4-bit")

            # Enforce memory requirements (may downgrade or raise error)
            quant = enforce_memory(model_info, quant, device)

            # Resolve max_tokens from preset
            model_max_tokens = model_info.max_output_tokens
            max_tokens_map = {
                "Low (512)": 512,
                "Medium (2048)": 2048,
                "High (4096)": 4096,
                "Max (Model Capacity)": model_max_tokens,
            }
            max_tokens = max_tokens_map.get(max_tokens_preset, 2048)

            # Cap to model's maximum if preset exceeds it
            if max_tokens > model_max_tokens:
                max_tokens = model_max_tokens

            # Reasoning is permanently disabled for local models
            # Local models output chain-of-thought mixed with results, breaking JSON parsing
            reasoning_enabled = False

            # Determine model capabilities based on model_type
            supports_vision_cap = model_info.model_type in [ModelType.VISION, ModelType.BOTH]
            supports_text_cap = model_info.model_type in [ModelType.TEXT, ModelType.BOTH]

            config = LLMModelConfig(
                provider=cls.PROVIDER_NAME,
                model=model,
                text_model=text_model,
                api_key="",
                api_url="",
                max_tokens=max_tokens,
                temperature=temperature,
                analysis_mode=analysis_mode.lower(),
                supports_vision=supports_vision_cap,
                supports_system_prompt=True,
                supports_reasoning=reasoning_enabled,
                extra_params={
                    "quantization": quant,
                    "device": device,
                    "attention_mode": attention_mode,
                    "keep_model_loaded": keep_model_loaded,
                    "repo_id": model_info.repo_id,
                    "family": model_info.family.value,
                    "model_type": model_info.model_type.value,
                    "is_thinking": model_info.is_thinking,
                    "is_fp8": model_info.is_fp8,
                    "enable_reasoning": reasoning_enabled,
                    "repetition_penalty": repetition_penalty,
                    "top_p": top_p,
                    "num_beams": num_beams,
                    "use_torch_compile": use_torch_compile,
                    "supports_text": supports_text_cap,
                    "supports_vision": supports_vision_cap,
                    "target_image_size": model_info.target_image_size,
                },
            )

            print(f"[SID_LLM_Local] {model} ({quant})")
            print(f"[SID_LLM_Local] max_tokens={max_tokens}, text_model={text_model}")

            return comfy_io.NodeOutput(config)

        except ValueError:
            # Re-raise validation errors as-is for ComfyUI error modal
            raise
        except Exception as e:
            # Catch unexpected errors and raise with context
            raise RuntimeError(f"[SID_LLM_Local] Unexpected error: {type(e).__name__}: {str(e)}") from e


# =============================================================================
# SID_LLM_Local_API - For Ollama, LM Studio, OpenAI Compatible
# =============================================================================

class SID_LLM_Local_API(comfy_io.ComfyNode):
    """
    Local API Provider for Ollama and LM Studio endpoints.

    Providers:
    - Ollama (localhost:11434)
    - LM Studio (localhost:1234)

    No API key required. Runs against local API servers.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""
        provider_options = ["Ollama", "LM Studio"]

        return comfy_io.Schema(
            node_id="SID_LLM_Local_API",
            display_name="SID LLM Local API",
            category="SID Photography Toolkit/LLM Providers",
            description="Ollama, LM Studio endpoints",
            inputs=[
                comfy_io.Combo.Input(
                    "provider",
                    options=provider_options,
                    default="Ollama",
                    tooltip="Select local API provider"
                ),
                comfy_io.String.Input(
                    "vision_model",
                    default="llava:latest",
                    display_name="Vision Model",
                    tooltip="Vision model for image analysis (e.g., 'llava:latest', 'llama3.2-vision', 'qwen2-vl')"
                ),
                comfy_io.String.Input(
                    "text_model",
                    default="llama3.2:latest",
                    display_name="Text Model",
                    tooltip="Text model for prompt enhancement (e.g., 'llama3.2:latest', 'qwen2.5:latest')"
                ),
                comfy_io.String.Input(
                    "api_url",
                    default="",
                    display_name="API URL (Optional)",
                    tooltip="Custom API URL. Leave empty for default (Ollama: localhost:11434, LM Studio: localhost:1234)"
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.3,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity (0=deterministic, 0.3=balanced, 1+=creative)"
                ),
                # Analysis mode - Standard or Detailed
                comfy_io.Combo.Input(
                    "analysis_mode",
                    options=["Standard", "Detailed"],
                    default="Detailed",
                    tooltip="Standard: Single-pass (1 call). Detailed: Comprehensive multi-aspect analysis (3-4 calls)"
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
        provider: str,
        vision_model: str,
        text_model: str,
        api_url: str,
        temperature: float,
        analysis_mode: str,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""
        try:
            if not vision_model.strip():
                raise ValueError("[SID_LLM_Local_API] Please enter a vision model name")

            vision_model_name = vision_model.strip()
            text_model_name = text_model.strip() if text_model.strip() else vision_model_name

            # Determine provider name and default URL
            if provider == "Ollama":
                provider_name = "ollama"
                default_url = "http://localhost:11434/v1"
            else:  # LM Studio
                provider_name = "lmstudio"
                default_url = "http://localhost:1234/v1"

            # Use custom URL or default
            actual_url = api_url.strip() if api_url.strip() else default_url

            # Ensure URL ends with /v1
            actual_url = actual_url.rstrip("/")
            if not actual_url.endswith("/v1"):
                actual_url = f"{actual_url}/v1"

            # Create configuration (use reasonable default for local APIs)
            config = LLMModelConfig(
                provider=provider_name,
                model=vision_model_name,
                text_model=text_model_name,
                api_key="",
                api_url=actual_url,
                max_tokens=4096,  # Default for local API providers
                temperature=temperature,
                analysis_mode=analysis_mode.lower(),
                supports_vision=True,
                supports_system_prompt=True,
                supports_reasoning=False,
                extra_params={
                    "target_image_size": 512,
                },
            )

            print(f"[SID_LLM_Local_API] {provider}")
            print(f"  Vision: {vision_model_name}")
            print(f"  Text: {text_model_name}")
            print(f"  URL: {actual_url}")

            return comfy_io.NodeOutput(config)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"[SID_LLM_Local_API] Unexpected error: {type(e).__name__}: {str(e)}") from e
