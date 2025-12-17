# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Local LLM Provider Node

Unified local vision-language model provider for ComfyUI.
Supports multiple model families with automatic VRAM management.

Supported Vision Model Families:
- QwenVL: Qwen3-VL, Qwen2.5-VL (2B-32B) - Best overall quality
- LLaVA: LLaVA 1.5/1.6 (7B-13B) - Popular VLM
- InternVL2: InternVL2 (2B-8B) - OpenGVLab
- MiniCPM-V: MiniCPM-V 2.6 - Efficient multi-modal
- PaliGemma: Google PaliGemma (3B) - Google VLM
- Llama-Vision: Llama 3.2 Vision (11B) - Meta VLM
- Pixtral: Mistral Pixtral (12B) - Mistral VLM
- Molmo: Allen AI Molmo (7B) - Allen AI VLM
- Idefics2: HuggingFace Idefics2 (8B) - HuggingFace VLM
- Florence-2: Microsoft Florence-2 (0.2B-0.7B) - Fast captioning
- Moondream2: Lightweight VLM (1.8B) - Efficient
- SmolVLM: HuggingFace SmolVLM (0.25B-2B) - Ultra-efficient
- Phi-3.5-Vision: Microsoft Phi-3.5 (4.2B) - High quality

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
    FLORENCE2 = "florence2"
    MOONDREAM2 = "moondream2"
    SMOLVLM = "smolvlm"
    PHI35_VISION = "phi35_vision"
    LLAVA = "llava"  # LLaVA vision-language models
    INTERNVL = "internvl"  # InternVL2 series
    MINICPM = "minicpm"  # MiniCPM-V series
    PALIGEMMA = "paligemma"  # Google PaliGemma
    LLAMA_VISION = "llama_vision"  # Llama 3.2 Vision
    PIXTRAL = "pixtral"  # Mistral Pixtral
    MOLMO = "molmo"  # Allen AI Molmo
    IDEFICS = "idefics"  # HuggingFace Idefics
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
    # Florence-2 Series (Microsoft) - Vision Only
    # =========================================================================
    "Florence-2-base": LocalModelInfo(
        name="Florence-2 Base (Vision) | 1K | 0.6GB",
        repo_id="microsoft/Florence-2-base",
        family=ModelFamily.FLORENCE2,
        vram_fp16=1.5, vram_8bit=1.0, vram_4bit=0.6,
        model_type=ModelType.VISION,
        max_output_tokens=1024,
        description="Ultra-fast captioning",
        model_class="AutoModelForCausalLM",
        target_image_size=256
    ),
    "Florence-2-large": LocalModelInfo(
        name="Florence-2 Large (Vision) | 1K | 1.2GB",
        repo_id="microsoft/Florence-2-large",
        family=ModelFamily.FLORENCE2,
        vram_fp16=3.0, vram_8bit=2.0, vram_4bit=1.2,
        model_type=ModelType.VISION,
        max_output_tokens=1024,
        description="Better captioning quality",
        model_class="AutoModelForCausalLM",
        target_image_size=320
    ),
    "Florence-2-base-ft": LocalModelInfo(
        name="Florence-2 Base FT (Vision) | 1K | 0.6GB",
        repo_id="microsoft/Florence-2-base-ft",
        family=ModelFamily.FLORENCE2,
        vram_fp16=1.5, vram_8bit=1.0, vram_4bit=0.6,
        model_type=ModelType.VISION,
        max_output_tokens=1024,
        description="Fine-tuned for tasks",
        model_class="AutoModelForCausalLM",
        target_image_size=256
    ),
    "Florence-2-large-ft": LocalModelInfo(
        name="Florence-2 Large FT (Vision) | 1K | 1.2GB",
        repo_id="microsoft/Florence-2-large-ft",
        family=ModelFamily.FLORENCE2,
        vram_fp16=3.0, vram_8bit=2.0, vram_4bit=1.2,
        model_type=ModelType.VISION,
        max_output_tokens=1024,
        description="Best Florence quality",
        model_class="AutoModelForCausalLM",
        target_image_size=320
    ),

    # =========================================================================
    # Moondream2 - Vision Only
    # =========================================================================
    "Moondream2": LocalModelInfo(
        name="Moondream2 (Vision) | 2K | 1.5GB",
        repo_id="vikhyatk/moondream2",
        family=ModelFamily.MOONDREAM2,
        vram_fp16=4.0, vram_8bit=2.5, vram_4bit=1.5,
        model_type=ModelType.VISION,
        max_output_tokens=2048,
        description="Efficient VLM",
        model_class="AutoModelForCausalLM",
        target_image_size=320
    ),

    # =========================================================================
    # LLaVA Series (HuggingFace) - Vision Only
    # =========================================================================
    "LLaVA-1.5-7B": LocalModelInfo(
        name="LLaVA 1.5 7B (Vision) | 4K | 7GB",
        repo_id="llava-hf/llava-1.5-7b-hf",
        family=ModelFamily.LLAVA,
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=5.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="LLaVA 1.5 - Good quality",
        model_class="LlavaForConditionalGeneration",
        target_image_size=384
    ),
    "LLaVA-1.5-13B": LocalModelInfo(
        name="LLaVA 1.5 13B (Vision) | 4K | 13GB",
        repo_id="llava-hf/llava-1.5-13b-hf",
        family=ModelFamily.LLAVA,
        vram_fp16=26.0, vram_8bit=14.0, vram_4bit=8.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="LLaVA 1.5 - Best quality",
        model_class="LlavaForConditionalGeneration",
        target_image_size=448
    ),
    "LLaVA-1.6-Mistral-7B": LocalModelInfo(
        name="LLaVA 1.6 Mistral 7B (Vision) | 4K | 7GB",
        repo_id="llava-hf/llava-v1.6-mistral-7b-hf",
        family=ModelFamily.LLAVA,
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=5.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="LLaVA 1.6 with Mistral backbone",
        model_class="LlavaNextForConditionalGeneration",
        target_image_size=384
    ),
    "LLaVA-1.6-Vicuna-7B": LocalModelInfo(
        name="LLaVA 1.6 Vicuna 7B (Vision) | 4K | 7GB",
        repo_id="llava-hf/llava-v1.6-vicuna-7b-hf",
        family=ModelFamily.LLAVA,
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=5.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="LLaVA 1.6 with Vicuna backbone",
        model_class="LlavaNextForConditionalGeneration",
        target_image_size=384
    ),
    "LLaVA-1.6-Vicuna-13B": LocalModelInfo(
        name="LLaVA 1.6 Vicuna 13B (Vision) | 4K | 13GB",
        repo_id="llava-hf/llava-v1.6-vicuna-13b-hf",
        family=ModelFamily.LLAVA,
        vram_fp16=26.0, vram_8bit=14.0, vram_4bit=8.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="LLaVA 1.6 Vicuna - Best quality",
        model_class="LlavaNextForConditionalGeneration",
        target_image_size=448
    ),

    # =========================================================================
    # InternVL2 Series (OpenGVLab) - Vision Only
    # =========================================================================
    "InternVL2-2B": LocalModelInfo(
        name="InternVL2 2B (Vision) | 4K | 2GB",
        repo_id="OpenGVLab/InternVL2-2B",
        family=ModelFamily.INTERNVL,
        vram_fp16=4.5, vram_8bit=3.0, vram_4bit=2.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Fast InternVL2",
        model_class="AutoModel",
        target_image_size=320
    ),
    "InternVL2-4B": LocalModelInfo(
        name="InternVL2 4B (Vision) | 4K | 4GB",
        repo_id="OpenGVLab/InternVL2-4B",
        family=ModelFamily.INTERNVL,
        vram_fp16=8.0, vram_8bit=5.0, vram_4bit=3.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Balanced InternVL2",
        model_class="AutoModel",
        target_image_size=384
    ),
    "InternVL2-8B": LocalModelInfo(
        name="InternVL2 8B (Vision) | 4K | 8GB",
        repo_id="OpenGVLab/InternVL2-8B",
        family=ModelFamily.INTERNVL,
        vram_fp16=16.0, vram_8bit=10.0, vram_4bit=6.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="High quality InternVL2",
        model_class="AutoModel",
        target_image_size=448
    ),

    # =========================================================================
    # MiniCPM-V Series (OpenBMB) - Vision Only
    # =========================================================================
    "MiniCPM-V-2.6": LocalModelInfo(
        name="MiniCPM-V 2.6 (Vision) | 4K | 8GB",
        repo_id="openbmb/MiniCPM-V-2_6",
        family=ModelFamily.MINICPM,
        vram_fp16=16.0, vram_8bit=10.0, vram_4bit=6.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Efficient multi-modal model",
        model_class="AutoModel",
        target_image_size=384
    ),

    # =========================================================================
    # PaliGemma Series (Google) - Vision Only
    # =========================================================================
    "PaliGemma-3B": LocalModelInfo(
        name="PaliGemma 3B (Vision) | 4K | 3GB",
        repo_id="google/paligemma-3b-mix-448",
        family=ModelFamily.PALIGEMMA,
        vram_fp16=7.0, vram_8bit=4.5, vram_4bit=3.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Google PaliGemma 3B",
        model_class="PaliGemmaForConditionalGeneration",
        target_image_size=448
    ),
    "PaliGemma2-3B": LocalModelInfo(
        name="PaliGemma2 3B (Vision) | 4K | 3GB",
        repo_id="google/paligemma2-3b-pt-448",
        family=ModelFamily.PALIGEMMA,
        vram_fp16=7.0, vram_8bit=4.5, vram_4bit=3.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Google PaliGemma2 3B",
        model_class="PaliGemmaForConditionalGeneration",
        target_image_size=448
    ),

    # =========================================================================
    # Llama 3.2 Vision (Meta) - Vision Only
    # =========================================================================
    "Llama-3.2-11B-Vision": LocalModelInfo(
        name="Llama 3.2 11B Vision (Vision) | 4K | 11GB",
        repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        family=ModelFamily.LLAMA_VISION,
        vram_fp16=22.0, vram_8bit=13.0, vram_4bit=8.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Meta Llama 3.2 Vision",
        model_class="MllamaForConditionalGeneration",
        target_image_size=448
    ),

    # =========================================================================
    # Pixtral (Mistral AI) - Vision Only
    # =========================================================================
    "Pixtral-12B": LocalModelInfo(
        name="Pixtral 12B (Vision) | 4K | 12GB",
        repo_id="mistralai/Pixtral-12B-2409",
        family=ModelFamily.PIXTRAL,
        vram_fp16=24.0, vram_8bit=14.0, vram_4bit=9.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Mistral Pixtral 12B",
        model_class="LlavaForConditionalGeneration",
        target_image_size=448
    ),

    # =========================================================================
    # Molmo (Allen AI) - Vision Only
    # =========================================================================
    "Molmo-7B-D": LocalModelInfo(
        name="Molmo 7B-D (Vision) | 4K | 7GB",
        repo_id="allenai/Molmo-7B-D-0924",
        family=ModelFamily.MOLMO,
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=5.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Allen AI Molmo 7B",
        model_class="AutoModelForCausalLM",
        target_image_size=384
    ),
    "Molmo-7B-O": LocalModelInfo(
        name="Molmo 7B-O (Vision) | 4K | 7GB",
        repo_id="allenai/Molmo-7B-O-0924",
        family=ModelFamily.MOLMO,
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=5.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="Allen AI Molmo 7B OpenAI-style",
        model_class="AutoModelForCausalLM",
        target_image_size=384
    ),

    # =========================================================================
    # Idefics2 (HuggingFace) - Vision Only
    # =========================================================================
    "Idefics2-8B": LocalModelInfo(
        name="Idefics2 8B (Vision) | 4K | 8GB",
        repo_id="HuggingFaceM4/idefics2-8b",
        family=ModelFamily.IDEFICS,
        vram_fp16=16.0, vram_8bit=10.0, vram_4bit=6.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="HuggingFace Idefics2 8B",
        model_class="Idefics2ForConditionalGeneration",
        target_image_size=416
    ),
    "Idefics2-8B-Chatty": LocalModelInfo(
        name="Idefics2 8B Chatty (Vision) | 4K | 8GB",
        repo_id="HuggingFaceM4/idefics2-8b-chatty",
        family=ModelFamily.IDEFICS,
        vram_fp16=16.0, vram_8bit=10.0, vram_4bit=6.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="HuggingFace Idefics2 8B Chatty",
        model_class="Idefics2ForConditionalGeneration",
        target_image_size=416
    ),

    # =========================================================================
    # SmolVLM Series (HuggingFace) - Vision Only
    # =========================================================================
    "SmolVLM-256M": LocalModelInfo(
        name="SmolVLM 256M (Vision) | 1K | 0.3GB",
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        family=ModelFamily.SMOLVLM,
        vram_fp16=0.8, vram_8bit=0.5, vram_4bit=0.3,
        model_type=ModelType.VISION,
        max_output_tokens=1024,
        description="Tiniest VLM",
        model_class="AutoModelForVision2Seq",
        target_image_size=224
    ),
    "SmolVLM-500M": LocalModelInfo(
        name="SmolVLM 500M (Vision) | 1K | 0.6GB",
        repo_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        family=ModelFamily.SMOLVLM,
        vram_fp16=1.5, vram_8bit=1.0, vram_4bit=0.6,
        model_type=ModelType.VISION,
        max_output_tokens=1024,
        description="Very small VLM",
        model_class="AutoModelForVision2Seq",
        target_image_size=256
    ),
    "SmolVLM-2B": LocalModelInfo(
        name="SmolVLM 2B (Vision) | 2K | 2GB",
        repo_id="HuggingFaceTB/SmolVLM-Instruct",
        family=ModelFamily.SMOLVLM,
        vram_fp16=4.5, vram_8bit=3.0, vram_4bit=2.0,
        model_type=ModelType.VISION,
        max_output_tokens=2048,
        description="Best SmolVLM",
        model_class="AutoModelForVision2Seq",
        target_image_size=320
    ),

    # =========================================================================
    # Phi-3.5-Vision (Microsoft) - Vision Only
    # =========================================================================
    "Phi-3.5-Vision": LocalModelInfo(
        name="Phi-3.5 Vision (Vision) | 4K | 3GB",
        repo_id="microsoft/Phi-3.5-vision-instruct",
        family=ModelFamily.PHI35_VISION,
        vram_fp16=8.5, vram_8bit=5.0, vram_4bit=3.0,
        model_type=ModelType.VISION,
        max_output_tokens=4096,
        description="High quality vision",
        model_class="AutoModelForCausalLM",
        target_image_size=384
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
    "Qwen3-0.6B-Instruct": LocalModelInfo(
        name="Qwen3 0.6B (Text) | 32K | 0.5GB [Ultra-Fast]",
        repo_id="Qwen/Qwen3-0.6B-Instruct",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=1.2, vram_8bit=0.8, vram_4bit=0.5,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Ultra-fast prompt generation",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen3-1.7B-Instruct": LocalModelInfo(
        name="Qwen3 1.7B (Text) | 32K | 1.4GB [Fast]",
        repo_id="Qwen/Qwen3-1.7B-Instruct",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=3.5, vram_8bit=2.2, vram_4bit=1.4,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Fast prompt generation",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen3-4B-Instruct": LocalModelInfo(
        name="Qwen3 4B (Text) | 32K | 3GB [Balanced]",
        repo_id="Qwen/Qwen3-4B-Instruct",
        family=ModelFamily.QWEN3_TEXT,
        vram_fp16=8.0, vram_8bit=5.0, vram_4bit=3.0,
        model_type=ModelType.TEXT,
        max_output_tokens=4096,
        description="Balanced prompt generation",
        model_class="AutoModelForCausalLM"
    ),
    "Qwen3-8B-Instruct": LocalModelInfo(
        name="Qwen3 8B (Text) | 32K | 6GB [Quality]",
        repo_id="Qwen/Qwen3-8B-Instruct",
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
                "florence2": ModelFamily.FLORENCE2,
                "moondream2": ModelFamily.MOONDREAM2,
                "smolvlm": ModelFamily.SMOLVLM,
                "phi35_vision": ModelFamily.PHI35_VISION,
                "llava": ModelFamily.LLAVA,
                "internvl": ModelFamily.INTERNVL,
                "minicpm": ModelFamily.MINICPM,
                "paligemma": ModelFamily.PALIGEMMA,
                "llama_vision": ModelFamily.LLAMA_VISION,
                "pixtral": ModelFamily.PIXTRAL,
                "molmo": ModelFamily.MOLMO,
                "idefics": ModelFamily.IDEFICS,
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

    @classmethod
    def unload_model(cls):
        """Force unload the cached model to free VRAM for other nodes."""
        if cls._cached_model is not None:
            # Delete model references
            del cls._cached_model
            cls._cached_model = None
            cls._cached_processor = None
            cls._cached_tokenizer = None
            cls._cached_signature = None
            cls._compiled_generate = None
            cls._image_cache.clear()
            # Aggressively clear memory
            clear_memory()
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

        # Clear memory before loading new model
        if LocalModelClient._cached_model is not None:
            LocalModelClient._cached_model = None
            LocalModelClient._cached_processor = None
            LocalModelClient._cached_tokenizer = None
            LocalModelClient._cached_signature = None
            LocalModelClient._compiled_generate = None
            clear_memory()

        if not self.model_info:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"[LocalModel] Loading {self.model_name}...")

        # Download model
        family_dir = os.path.join(LLM_LOCAL_DIR, self.model_info.family.value)
        os.makedirs(family_dir, exist_ok=True)
        model_path = os.path.join(family_dir, self.model_name)

        if not os.path.exists(model_path):
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
        if self.model_info.family == ModelFamily.FLORENCE2:
            self._load_florence2(model_path, device)
        elif self.model_info.family == ModelFamily.MOONDREAM2:
            self._load_moondream2(model_path, device)
        elif self.model_info.family == ModelFamily.SMOLVLM:
            self._load_smolvlm(model_path, device)
        elif self.model_info.family == ModelFamily.PHI35_VISION:
            self._load_phi35_vision(model_path, device)
        elif self.model_info.family == ModelFamily.LLAVA:
            self._load_llava(model_path, device)
        elif self.model_info.family == ModelFamily.INTERNVL:
            self._load_internvl(model_path, device)
        elif self.model_info.family == ModelFamily.MINICPM:
            self._load_minicpm(model_path, device)
        elif self.model_info.family == ModelFamily.PALIGEMMA:
            self._load_paligemma(model_path, device)
        elif self.model_info.family == ModelFamily.LLAMA_VISION:
            self._load_llama_vision(model_path, device)
        elif self.model_info.family == ModelFamily.PIXTRAL:
            self._load_pixtral(model_path, device)
        elif self.model_info.family == ModelFamily.MOLMO:
            self._load_molmo(model_path, device)
        elif self.model_info.family == ModelFamily.IDEFICS:
            self._load_idefics(model_path, device)
        elif self.model_info.family == ModelFamily.QWENVL:
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

    def _load_florence2(self, model_path: str, device: str):
        """Load Florence-2 model with speed optimizations."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            # Force eager attention - Florence-2's custom code doesn't define _supports_sdpa
            # which newer transformers (4.49+) checks for
            "attn_implementation": "eager",
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

    def _load_moondream2(self, model_path: str, device: str):
        """Load Moondream2 model with speed optimizations."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Note: Moondream2's vision encoder doesn't support bitsandbytes quantization
        # (causes Float vs Half dtype mismatch in layer_norm). Always use FP16/BF16.
        _, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            # Skip quantization for Moondream2 - use BF16 if supported, else FP16
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere+ supports BF16
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = torch.float16

        # Try loading, if cache is corrupted clear it and retry from HuggingFace
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except FileNotFoundError as e:
            if "transformers_modules" in str(e):
                print(f"[LocalModelClient] Detected corrupted cache, clearing and retrying from HuggingFace...")
                self._clear_transformers_cache("moondream")
                self._clear_transformers_cache("Moondream")
                # Use repo name with force_download to re-fetch modules
                retry_kwargs = {**load_kwargs, "force_download": True}
                self.model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", **retry_kwargs)
            else:
                raise
        self.model.eval()

        # Enable KV cache for faster generation
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = self.model  # Moondream has built-in image processing

    def _load_llava(self, model_path: str, device: str):
        """Load LLaVA model with quantization support."""
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        # Get quantization config
        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype

        # Determine model class based on version (1.5 vs 1.6)
        model_class_name = self.model_info.model_class
        if model_class_name == "LlavaNextForConditionalGeneration":
            from transformers import LlavaNextForConditionalGeneration
            ModelClass = LlavaNextForConditionalGeneration
        else:
            from transformers import LlavaForConditionalGeneration
            ModelClass = LlavaForConditionalGeneration

        self.model = ModelClass.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        # Enable KV cache
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def _load_internvl(self, model_path: str, device: str):
        """Load InternVL2 model with quantization support."""
        import torch
        from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.bfloat16

        self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

    def _load_minicpm(self, model_path: str, device: str):
        """Load MiniCPM-V model with quantization support."""
        import torch
        from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.bfloat16

        self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

    def _load_paligemma(self, model_path: str, device: str):
        """Load PaliGemma model with quantization support."""
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.bfloat16

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

    def _load_llama_vision(self, model_path: str, device: str):
        """Load Llama 3.2 Vision model with quantization support."""
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.bfloat16

        self.model = MllamaForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

    def _load_pixtral(self, model_path: str, device: str):
        """Load Pixtral model with quantization support."""
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.bfloat16

        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

    def _load_molmo(self, model_path: str, device: str):
        """Load Molmo model with quantization support."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

    def _load_idefics(self, model_path: str, device: str):
        """Load Idefics2 model with quantization support."""
        import torch
        from transformers import AutoProcessor, Idefics2ForConditionalGeneration, BitsAndBytesConfig

        quant_config, dtype = self._get_quantization_config(device)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype or torch.bfloat16

        self.model = Idefics2ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

    def _load_smolvlm(self, model_path: str, device: str):
        """Load SmolVLM model with speed optimizations."""
        import importlib.metadata
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
        import importlib.metadata
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
                load_kwargs["torch_dtype"] = dtype or torch.float16

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
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

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
            elif self.model_info.family == ModelFamily.LLAVA:
                response_text = self._generate_llava(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.INTERNVL:
                response_text = self._generate_internvl(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.MINICPM:
                response_text = self._generate_minicpm(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.PALIGEMMA:
                response_text = self._generate_paligemma(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.LLAMA_VISION:
                response_text = self._generate_llama_vision(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.PIXTRAL:
                response_text = self._generate_pixtral(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.MOLMO:
                response_text = self._generate_molmo(images, text_prompt, max_tokens, current_temp)
            elif self.model_info.family == ModelFamily.IDEFICS:
                response_text = self._generate_idefics(images, text_prompt, max_tokens, current_temp)
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
        """
        import torch

        try:
            # CRITICAL: Ensure gradient computation is enabled for other nodes
            # Some transformers models or generation code may disable gradients
            # and VAE/other nodes may need them enabled
            torch.set_grad_enabled(True)

            # If not keeping model loaded, unload it to free VRAM
            if not self.keep_model_loaded:
                self._unload_model_instance()

            if torch.cuda.is_available():
                # Synchronize to ensure all CUDA operations are complete
                torch.cuda.synchronize()

                # Clear CUDA cache to free up memory for VAE
                torch.cuda.empty_cache()

            # Force garbage collection to release any dangling references
            gc.collect()

        except Exception:
            pass

    def _unload_model_instance(self):
        """Unload model from this instance to free VRAM."""
        import torch

        try:
            # Delete model references
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None

            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Clear class-level cache as well
            LocalModelClient._cached_model = None
            LocalModelClient._cached_processor = None
            LocalModelClient._cached_tokenizer = None
            LocalModelClient._cached_signature = None

            # Force garbage collection
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[LocalModel] Model unloaded from VRAM")

        except Exception as e:
            print(f"[LocalModel] Warning: Error during model unload: {e}")

    def _generate_florence2(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Florence-2."""
        # Check for interrupt before Florence-2 generation
        check_interrupted()

        import torch

        # DynamicCache compatibility fix for transformers 4.49+
        self._patch_dynamic_cache_compat()

        # Florence-2 uses task prompts like <DETAILED_CAPTION>
        task_prompt = "<MORE_DETAILED_CAPTION>"

        if images:
            inputs = self.processor(text=task_prompt, images=images[0], return_tensors="pt")
        else:
            inputs = self.processor(text=task_prompt, return_tensors="pt")

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        # Move inputs to device and cast float tensors to model dtype
        inputs = {
            k: v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device)
            for k, v in inputs.items()
        }

        with InferenceProgressSpinner("Generating (Florence-2)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=1,  # Use greedy decoding - beam search has cache issues with newer transformers
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def _generate_moondream2(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Moondream2."""
        # Check for interrupt before Moondream2 generation
        check_interrupted()

        import torch

        # DynamicCache compatibility fix for transformers 4.49+
        self._patch_dynamic_cache_compat()

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
        # Check for interrupt before SmolVLM generation
        check_interrupted()

        import torch

        # DynamicCache compatibility fix for transformers 4.49+
        self._patch_dynamic_cache_compat()

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

    def _generate_llava(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with LLaVA models (1.5 and 1.6/Next)."""
        # Check for interrupt before LLaVA generation
        check_interrupted()

        import torch

        # Build conversation format for LLaVA
        # LLaVA expects: "USER: <image>\n{prompt}\nASSISTANT:"
        if images:
            conversation = f"USER: <image>\n{prompt.strip()}\nASSISTANT:"
        else:
            conversation = f"USER: {prompt.strip()}\nASSISTANT:"

        # Process inputs
        inputs = self.processor(
            text=conversation,
            images=images[0] if images else None,  # LLaVA takes single image
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        # Get input length to extract only generated tokens
        input_len = inputs["input_ids"].shape[1]

        with InferenceProgressSpinner("Generating (LLaVA)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

        # Only decode the NEW tokens (exclude input prompt)
        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_internvl(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with InternVL2 models."""
        check_interrupted()

        import torch

        # InternVL2 uses chat-style format
        messages = [{"role": "user", "content": f"<image>\n{prompt.strip()}"}]

        # Process inputs
        inputs = self.processor(
            text=prompt.strip(),
            images=images[0] if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        with InferenceProgressSpinner("Generating (InternVL2)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        generated_tokens = outputs[0][input_len:] if input_len > 0 else outputs[0]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_minicpm(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with MiniCPM-V models."""
        check_interrupted()

        import torch

        # MiniCPM-V has its own chat method
        if hasattr(self.model, 'chat'):
            # Use the model's built-in chat method
            msgs = [{"role": "user", "content": prompt.strip()}]
            with InferenceProgressSpinner("Generating (MiniCPM-V)"):
                response = self.model.chat(
                    image=images[0] if images else None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=max_tokens,
                    sampling=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )
            return response
        else:
            # Fallback to standard generation
            inputs = self.processor(
                text=prompt.strip(),
                images=images[0] if images else None,
                return_tensors="pt"
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

            with InferenceProgressSpinner("Generating (MiniCPM-V)"):
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                    )

            generated_tokens = outputs[0][input_len:] if input_len > 0 else outputs[0]
            return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_paligemma(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with PaliGemma models."""
        check_interrupted()

        import torch

        # PaliGemma format
        inputs = self.processor(
            text=prompt.strip(),
            images=images[0] if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with InferenceProgressSpinner("Generating (PaliGemma)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_llama_vision(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Llama 3.2 Vision models."""
        check_interrupted()

        import torch

        # Llama 3.2 Vision uses chat template
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt.strip()}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=input_text,
            images=images[0] if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with InferenceProgressSpinner("Generating (Llama-Vision)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_pixtral(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Pixtral models (uses LLaVA-style format)."""
        check_interrupted()

        import torch

        # Pixtral uses LLaVA-style format
        if images:
            conversation = f"USER: <image>\n{prompt.strip()}\nASSISTANT:"
        else:
            conversation = f"USER: {prompt.strip()}\nASSISTANT:"

        inputs = self.processor(
            text=conversation,
            images=images[0] if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with InferenceProgressSpinner("Generating (Pixtral)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _generate_molmo(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Molmo models."""
        check_interrupted()

        import torch

        # Molmo uses its processor for image+text
        inputs = self.processor.process(
            images=images[0] if images else None,
            text=prompt.strip()
        )

        # Convert to tensors and move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device).unsqueeze(0) if torch.is_tensor(v) else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        with InferenceProgressSpinner("Generating (Molmo)"):
            with torch.inference_mode():
                outputs = self.model.generate_from_batch(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        # Molmo returns the generated text directly
        if isinstance(outputs, str):
            return outputs

        generated_tokens = outputs[0][input_len:] if input_len > 0 else outputs[0]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _generate_idefics(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Idefics2 models."""
        check_interrupted()

        import torch

        # Idefics2 uses chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt.strip()},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with InferenceProgressSpinner("Generating (Idefics2)"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                )

        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def _patch_dynamic_cache_compat(self):
        """
        Holistic compatibility fix for DynamicCache in newer transformers versions (4.49+).

        Phi-3.5-Vision's custom modeling code expects deprecated/removed DynamicCache methods.
        This patches the DynamicCache class to add compatibility shims for:
        - seen_tokens property (deprecated in favor of cache_position tracking)
        - get_max_length() method (removed in v4.49)
        - get_usable_length(new_seq_length, layer_idx) method (removed in v4.49)

        See: https://github.com/huggingface/transformers/issues/36071
        """
        try:
            from transformers import DynamicCache

            # Only patch if the methods are missing (avoid double-patching)
            needs_patching = False

            # Check if seen_tokens property exists
            if not hasattr(DynamicCache, 'seen_tokens'):
                needs_patching = True

                @property
                def seen_tokens_compat(self):
                    """Compatibility shim for deprecated seen_tokens property."""
                    if hasattr(self, '_seen_tokens'):
                        return self._seen_tokens
                    # Try get_seq_length method (newer transformers)
                    if hasattr(self, 'get_seq_length'):
                        try:
                            return self.get_seq_length(0)
                        except (IndexError, TypeError):
                            pass
                    # Calculate from key_cache (older transformers)
                    if hasattr(self, 'key_cache') and self.key_cache and len(self.key_cache) > 0:
                        if self.key_cache[0] is not None:
                            return self.key_cache[0].shape[-2]
                    return 0

                DynamicCache.seen_tokens = seen_tokens_compat

            # Check if get_max_length method exists
            if not hasattr(DynamicCache, 'get_max_length'):
                needs_patching = True

                def get_max_length_compat(self):
                    """Compatibility shim for removed get_max_length method."""
                    return None  # DynamicCache has no max length limit

                DynamicCache.get_max_length = get_max_length_compat

            # Check if get_usable_length method exists
            if not hasattr(DynamicCache, 'get_usable_length'):
                needs_patching = True

                def get_usable_length_compat(self, new_seq_length: int = 0, layer_idx: int = 0):
                    """
                    Compatibility shim for removed get_usable_length method.

                    Args:
                        new_seq_length: The new sequence length being added
                        layer_idx: The layer index (not used, but required for signature)

                    Returns:
                        The current sequence length in the cache
                    """
                    # Try get_seq_length method (newer transformers)
                    if hasattr(self, 'get_seq_length'):
                        try:
                            return self.get_seq_length(layer_idx)
                        except (IndexError, TypeError):
                            pass
                    # Return the current cache length for the specified layer (older transformers)
                    if hasattr(self, 'key_cache') and self.key_cache and len(self.key_cache) > layer_idx:
                        if self.key_cache[layer_idx] is not None:
                            return self.key_cache[layer_idx].shape[-2]
                    return 0

                DynamicCache.get_usable_length = get_usable_length_compat

        except ImportError:
            # transformers not installed or DynamicCache not available
            pass
        except Exception:
            pass

    def _generate_phi35_vision(self, images: List, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Phi-3.5-Vision."""
        # Check for interrupt before Phi-3.5-Vision generation
        check_interrupted()

        import torch

        # Holistic compatibility fix for newer transformers versions (4.49+)
        # Phi-3.5-Vision's custom code expects deprecated/removed DynamicCache methods
        # See: https://github.com/huggingface/transformers/issues/36071
        self._patch_dynamic_cache_compat()

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
                    default="Qwen3-0.6B-Instruct" if "Qwen3-0.6B-Instruct" in text_models else text_models[0] if text_models else "",
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
                    tooltip="Keep model in VRAM between runs (faster repeat inference). Disabled by default to free VRAM after execution."
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
                # Analysis mode - Local supports Quick/Standard only (no Deep)
                comfy_io.Combo.Input(
                    "analysis_mode",
                    options=["Quick", "Standard"],
                    default="Standard",
                    tooltip="Analysis depth: Quick (fast, 30-60 words), Standard (balanced, 80-150 words)"
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
        analysis_mode: str,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""
        try:
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
                # Analysis mode - Local API supports Quick/Standard only (no Deep)
                comfy_io.Combo.Input(
                    "analysis_mode",
                    options=["Quick", "Standard"],
                    default="Standard",
                    tooltip="Analysis depth: Quick (fast, 30-60 words), Standard (balanced, 80-150 words)"
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
