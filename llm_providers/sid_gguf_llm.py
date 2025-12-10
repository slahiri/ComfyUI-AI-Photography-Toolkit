"""
SID_GGUF_LLM Node

Local GGUF model provider for ComfyUI using llama-cpp-python.
Supports vision-language models like LLaVA, MiniCPM-V, and Moondream.
Models are downloaded to ComfyUI/models/LLM/GGUF/ folder.
"""

import os
import sys
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from comfy_api.latest import io as comfy_io
import folder_paths
import comfy.utils

from .llm_model_type import LLMModelConfig
from .base_llm_provider import BaseLLMProvider

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")

# Register LLM/GGUF folder with ComfyUI
LLM_GGUF_DIR = os.path.join(folder_paths.models_dir, "LLM", "GGUF")
os.makedirs(LLM_GGUF_DIR, exist_ok=True)


@dataclass
class GGUFModelInfo:
    """Information about a GGUF model with metadata for validation."""
    name: str
    filename: str
    url: str
    size_mb: int
    vram_gb: float
    description: str
    chat_format: str  # llava-1-5, llava-1-6, chatml, minicpm-v, pixtral, llama-3-vision, etc.
    # Model capability metadata
    context_length: int = 4096  # Maximum context length supported
    recommended_n_ctx: int = 4096  # Recommended n_ctx setting
    max_output_tokens: int = 2048  # Maximum output tokens
    max_image_resolution: str = "dynamic"  # Max supported resolution: "1024x1024", "1120x1120", "dynamic" (any size)
    supports_json: bool = True  # Can follow JSON output format
    supports_detailed_prompts: bool = True  # Can handle complex prompts
    supports_reasoning: bool = False  # Local GGUF models don't support reasoning mode
    quality_tier: str = "good"  # basic, good, excellent, best
    mmproj_filename: Optional[str] = None  # For models needing separate vision encoder
    mmproj_url: Optional[str] = None


# Available GGUF vision models with download URLs
# All models support 1024x1024+ images natively (no tiling/downscaling)
# LLaVA 1.5/1.6 removed - limited to 336x336/672x672 per tile
GGUF_MODELS: Dict[str, GGUFModelInfo] = {
    # ===== SMALL MODELS (~4-6GB VRAM) =====
    "qwen3-vl-2b-q4_k_m": GGUFModelInfo(
        name="Qwen3-VL 2B Q4 (4GB VRAM) - Fast",
        filename="Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        url="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        size_mb=1100,
        vram_gb=4.0,
        description="Fastest option, dynamic resolution, great for 8GB GPUs",
        chat_format="chatml",
        context_length=32768,
        recommended_n_ctx=8192,
        max_output_tokens=4096,
        max_image_resolution="dynamic",
        supports_json=True,
        supports_detailed_prompts=True,
        quality_tier="good",
        mmproj_filename="mmproj-Qwen3VL-2B-Instruct-F16.gguf",
        mmproj_url="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf",
    ),

    # ===== MEDIUM MODELS (~6-10GB VRAM) =====
    "qwen2.5-vl-7b-q4_k_m": GGUFModelInfo(
        name="Qwen2.5-VL 7B Q4 (6GB VRAM) - Recommended",
        filename="Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
        url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
        size_mb=5200,
        vram_gb=6.0,
        description="Excellent quality, dynamic resolution, best value",
        chat_format="chatml",
        context_length=32768,
        recommended_n_ctx=8192,
        max_output_tokens=4096,
        max_image_resolution="dynamic",
        supports_json=True,
        supports_detailed_prompts=True,
        quality_tier="excellent",
        mmproj_filename="Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
    ),

    "llama-3.2-vision-11b-q4_k_m": GGUFModelInfo(
        name="Llama 3.2 Vision 11B Q4 (8GB VRAM)",
        filename="Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf",
        url="https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf",
        size_mb=5960,
        vram_gb=8.0,
        description="Meta's latest, native 1120x1120 resolution",
        chat_format="llama-3-vision",
        context_length=131072,
        recommended_n_ctx=8192,
        max_output_tokens=4096,
        max_image_resolution="1120x1120",
        supports_json=True,
        supports_detailed_prompts=True,
        quality_tier="excellent",
        mmproj_filename="Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf",
        mmproj_url="https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf",
    ),

    "pixtral-12b-q4_k_m": GGUFModelInfo(
        name="Pixtral 12B Q4 (10GB VRAM)",
        filename="pixtral-12b-Q4_K_M.gguf",
        url="https://huggingface.co/ggml-org/pixtral-12b-GGUF/resolve/main/pixtral-12b-Q4_K_M.gguf",
        size_mb=7480,
        vram_gb=10.0,
        description="Mistral's vision model, native 1024x1024 resolution",
        chat_format="pixtral",
        context_length=131072,
        recommended_n_ctx=8192,
        max_output_tokens=4096,
        max_image_resolution="1024x1024",
        supports_json=True,
        supports_detailed_prompts=True,
        quality_tier="excellent",
        mmproj_filename="mmproj-pixtral-12b-f16.gguf",
        mmproj_url="https://huggingface.co/ggml-org/pixtral-12b-GGUF/resolve/main/mmproj-pixtral-12b-f16.gguf",
    ),

    "minicpm-v-2_6-q4_k_m": GGUFModelInfo(
        name="MiniCPM-V 2.6 (10GB VRAM)",
        filename="MiniCPM-V-2_6-Q4_K_M.gguf",
        url="https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q4_K_M.gguf",
        size_mb=4800,
        vram_gb=10.0,
        description="Excellent vision understanding, dynamic resolution",
        chat_format="minicpm-v-2.6",
        context_length=8192,
        recommended_n_ctx=8192,
        max_output_tokens=4096,
        max_image_resolution="dynamic",
        supports_json=True,
        supports_detailed_prompts=True,
        quality_tier="excellent",
        mmproj_filename="MiniCPM-V-2_6-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf",
    ),

    "qwen2.5-vl-7b-q8": GGUFModelInfo(
        name="Qwen2.5-VL 7B Q8 (10GB VRAM) - Best Quality",
        filename="Qwen2.5-VL-7B-Instruct-q8_0.gguf",
        url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-q8_0.gguf",
        size_mb=8100,
        vram_gb=10.0,
        description="Highest quality 7B model, dynamic resolution",
        chat_format="chatml",
        context_length=32768,
        recommended_n_ctx=8192,
        max_output_tokens=4096,
        max_image_resolution="dynamic",
        supports_json=True,
        supports_detailed_prompts=True,
        quality_tier="excellent",
        mmproj_filename="Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
    ),
}


class LocalGGUFClient:
    """
    Wrapper class that provides OpenAI-compatible interface for local GGUF models.
    This allows seamless integration with the existing _call_vision_llm routing.

    Features image encoding caching - when the same image is analyzed multiple times,
    the expensive CLIP encoding is cached and reused.
    """

    # Class-level image cache shared across instances
    _image_cache: Dict[str, Any] = {}
    _cache_max_size: int = 5  # Keep last 5 images cached

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        chat_format: str = "llava-1-5",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        verbose: bool = False,
    ):
        """
        Initialize the local GGUF client.

        Args:
            model_path: Path to the GGUF model file
            mmproj_path: Path to the multimodal projector (for vision models)
            chat_format: Chat format to use (llava-1-5, llava-1-6, chatml, minicpm-v-2.6, moondream2)
            n_ctx: Context length
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            verbose: Enable verbose output
        """
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler, MoondreamChatHandler
        except ImportError as e:
            import platform
            system = platform.system()

            error_msg = (
                "\n" + "=" * 70 + "\n"
                "ERROR: llama-cpp-python is not installed!\n"
                "=" * 70 + "\n\n"
                "This package requires platform-specific installation.\n\n"
            )

            if system == "Windows":
                error_msg += (
                    "For Windows with NVIDIA GPU (CUDA 12.1+):\n"
                    "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                    "For Windows CPU-only:\n"
                    "  pip install llama-cpp-python\n\n"
                )
            elif system == "Darwin":  # macOS
                error_msg += (
                    "For macOS with Apple Silicon (M1/M2/M3):\n"
                    "  CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python\n\n"
                    "For macOS Intel:\n"
                    "  pip install llama-cpp-python\n\n"
                )
            else:  # Linux
                error_msg += (
                    "For Linux with NVIDIA GPU (CUDA 12.1+):\n"
                    "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                    "For Linux with AMD GPU (ROCm):\n"
                    "  CMAKE_ARGS=\"-DGGML_HIPBLAS=on\" pip install llama-cpp-python\n\n"
                    "For Linux CPU-only:\n"
                    "  pip install llama-cpp-python\n\n"
                )

            error_msg += (
                "After installing, restart ComfyUI.\n"
                "=" * 70 + "\n"
            )

            print(error_msg)
            raise ImportError(error_msg) from e

        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.chat_format = chat_format
        self.verbose = verbose

        # Track last image hash for KV cache optimization
        self._last_image_hash: Optional[str] = None
        self._conversation_history: List[Dict[str, Any]] = []

        # Create appropriate chat handler for vision models
        chat_handler = None
        if mmproj_path and os.path.exists(mmproj_path):
            # All modern vision models use the Llava15ChatHandler with appropriate clip model
            if chat_format in ["llava-1-5", "llava-1-6", "minicpm-v-2.6", "chatml", "llama-3-vision", "pixtral"]:
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=verbose)
            elif chat_format == "moondream2":
                chat_handler = MoondreamChatHandler(clip_model_path=mmproj_path, verbose=verbose)

        self._chat_handler = chat_handler

        # Load the model
        self.llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            logits_all=True if chat_handler else False,
        )

        print(f"[LocalGGUFClient] Model loaded: {os.path.basename(model_path)}")
        if mmproj_path:
            print(f"[LocalGGUFClient] Vision encoder: {os.path.basename(mmproj_path)}")
            print(f"[LocalGGUFClient] Image caching: ENABLED (cache size: {self._cache_max_size})")

    @property
    def chat(self):
        """Return self to allow OpenAI-style chaining: client.chat.completions.create()"""
        return self

    @property
    def completions(self):
        """Return self to allow OpenAI-style chaining: client.chat.completions.create()"""
        return self

    def _extract_image_hash(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract and hash image data from messages for caching."""
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                        if url.startswith("data:image"):
                            # Hash the base64 data (first 1000 chars for speed)
                            return hashlib.md5(url[:1000].encode()).hexdigest()
        return None

    def _should_use_cache(self, current_hash: Optional[str]) -> bool:
        """Check if we can leverage KV cache for this image."""
        if current_hash is None:
            return False
        return current_hash == self._last_image_hash

    def reset_cache(self):
        """Reset the image cache (call when switching to a new image)."""
        self._last_image_hash = None
        self._conversation_history = []
        # Reset LLM's internal KV cache
        if hasattr(self.llm, 'reset'):
            self.llm.reset()
        print("[LocalGGUFClient] Cache reset")

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> "LocalGGUFResponse":
        """
        Create a chat completion (OpenAI-compatible interface).

        Features intelligent caching: when analyzing the same image multiple times,
        the image encoding is cached, reducing subsequent calls from ~70s to ~5s.

        Args:
            model: Model name (ignored, uses loaded model)
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            OpenAI-compatible response object
        """
        import time
        start_time = time.time()

        # Check if this is the same image as last call
        current_hash = self._extract_image_hash(messages)
        use_cache = self._should_use_cache(current_hash)

        if use_cache:
            print(f"[LocalGGUFClient] Image cache HIT - reusing encoded image")
        elif current_hash:
            print(f"[LocalGGUFClient] Image cache MISS - encoding new image")
            self._last_image_hash = current_hash

        # Run inference
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        elapsed = time.time() - start_time
        print(f"[LocalGGUFClient] Inference completed in {elapsed:.1f}s")

        return LocalGGUFResponse(response)

    def unload(self):
        """Unload the model from memory."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None

            # Force garbage collection
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            print("[LocalGGUFClient] Model unloaded")


class LocalGGUFResponse:
    """OpenAI-compatible response wrapper for local GGUF inference."""

    def __init__(self, response: dict):
        self._response = response
        self.choices = [LocalGGUFChoice(response)]


class LocalGGUFChoice:
    """OpenAI-compatible choice wrapper."""

    def __init__(self, response: dict):
        self.message = LocalGGUFMessage(response)


class LocalGGUFMessage:
    """OpenAI-compatible message wrapper."""

    def __init__(self, response: dict):
        # Extract content from llama-cpp response
        if "choices" in response and len(response["choices"]) > 0:
            self.content = response["choices"][0].get("message", {}).get("content", "")
        else:
            self.content = ""



def download_file_with_progress(url: str, dest_path: str, desc: str, hf_token: Optional[str] = None):
    """
    Download a file with ComfyUI progress bar integration.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress display
        hf_token: Optional HuggingFace token for private repos
    """
    import urllib.request
    import shutil

    temp_path = dest_path + ".tmp"

    try:
        # Build request with optional auth header
        request = urllib.request.Request(url)
        if hf_token and hf_token.strip():
            request.add_header("Authorization", f"Bearer {hf_token}")

        # Open URL and get file size
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get("content-length", 0))
            total_mb = total_size / (1024 * 1024) if total_size else 0

            print(f"[SID_GGUF_LLM] Downloading {desc}")
            print(f"  Size: {total_mb:.1f} MB")

            # Create ComfyUI progress bar
            pbar = None
            if total_size > 0:
                pbar = comfy.utils.ProgressBar(100)

            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if pbar and total_size > 0:
                        percent = int((downloaded / total_size) * 100)
                        pbar.update_absolute(percent, 100)
                        mb_done = downloaded / (1024 * 1024)
                        # Print progress to console using carriage return
                        sys.stdout.write(f"  [{percent:3d}%] {mb_done:.1f} / {total_mb:.1f} MB" + " " * 10 + chr(13))
                        sys.stdout.flush()

            print()  # New line after progress

        # Move temp file to final destination
        shutil.move(temp_path, dest_path)
        print(f"[SID_GGUF_LLM] Downloaded: {os.path.basename(dest_path)}")

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Download failed: {e}")


def download_model(model_info: GGUFModelInfo, hf_token: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Download a GGUF model and its vision encoder if needed.

    Args:
        model_info: GGUFModelInfo with download URLs
        hf_token: Optional HuggingFace token for private repos

    Returns:
        Tuple of (model_path, mmproj_path)
    """
    model_path = os.path.join(LLM_GGUF_DIR, model_info.filename)
    mmproj_path = None

    # Download main model
    if not os.path.exists(model_path):
        download_file_with_progress(
            model_info.url,
            model_path,
            f"{model_info.name} ({model_info.size_mb}MB)",
            hf_token
        )

    # Download vision encoder if needed
    if model_info.mmproj_filename and model_info.mmproj_url:
        mmproj_path = os.path.join(LLM_GGUF_DIR, model_info.mmproj_filename)

        if not os.path.exists(mmproj_path):
            download_file_with_progress(
                model_info.mmproj_url,
                mmproj_path,
                f"Vision encoder for {model_info.name}",
                hf_token
            )

    return model_path, mmproj_path


def get_available_models() -> List[str]:
    """Get list of available models (downloaded + downloadable)."""
    models = list(GGUF_MODELS.keys())

    # Also check for manually placed models in the folder
    if os.path.exists(LLM_GGUF_DIR):
        for f in os.listdir(LLM_GGUF_DIR):
            if f.endswith(".gguf") and not f.endswith("-mmproj-f16.gguf"):
                custom_name = f.replace(".gguf", "")
                if custom_name not in models:
                    models.append(f"custom:{custom_name}")

    return models



class SID_GGUF_LLM(comfy_io.ComfyNode, BaseLLMProvider):
    """
    Local GGUF LLM Provider.

    Runs vision-language models locally using llama-cpp-python.
    Models are stored in ComfyUI/models/LLM/GGUF/ folder.

    All models support 1024x1024+ images natively (no tiling/downscaling):
    - Qwen3-VL 2B: Fast, dynamic resolution (~4GB VRAM)
    - Qwen2.5-VL 7B: Best value, dynamic resolution (~6-10GB VRAM)
    - Llama 3.2 Vision 11B: Meta's latest, 1120x1120 native (~8GB VRAM)
    - Pixtral 12B: Mistral's vision model, 1024x1024 native (~10GB VRAM)
    - MiniCPM-V 2.6: Multilingual, dynamic resolution (~10GB VRAM)
    """

    PROVIDER_NAME = "gguf"

    @classmethod
    def get_models(cls) -> List[str]:
        return get_available_models()

    @classmethod
    def get_default_model(cls) -> str:
        return "qwen2.5-vl-7b-q4_k_m"

    @classmethod
    def supports_reasoning(cls, model: str) -> bool:
        """Local GGUF models don't support reasoning mode."""
        # Check if model is in GGUF_MODELS dict
        if model in GGUF_MODELS:
            return GGUF_MODELS[model].supports_reasoning
        return False  # Custom models default to no reasoning

    @classmethod
    def get_default_url(cls) -> str:
        return ""  # Local model, no URL

    @classmethod
    def requires_api_key(cls) -> bool:
        return False

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""

        model_options = list(GGUF_MODELS.keys())

        # User-friendly presets for memory/performance tradeoffs
        memory_presets = [
            "Auto (Recommended)",
            "Low VRAM (4-6GB)",
            "Medium VRAM (8-12GB)",
            "High VRAM (16GB+)",
            "CPU Only (No GPU)",
        ]

        # Build model info for description
        model_info_text = (
            "[DEPRECATED] Use SID_QwenVL_LLM instead - easier installation!\n\n"
            "Local GGUF vision models - Requires llama-cpp-python (platform-specific build).\n\n"
            "Models:\n"
            "- Qwen3-VL 2B Q4: 4GB VRAM, Dynamic res, Good (fast)\n"
            "- Qwen2.5-VL 7B Q4: 6GB VRAM, Dynamic res, Excellent (recommended)\n"
            "- Llama 3.2 Vision 11B: 8GB VRAM, 1120x1120, Excellent\n"
            "- Pixtral 12B Q4: 10GB VRAM, 1024x1024, Excellent\n"
            "- MiniCPM-V 2.6: 10GB VRAM, Dynamic res, Excellent\n"
            "- Qwen2.5-VL 7B Q8: 10GB VRAM, Dynamic res, Best quality"
        )

        model_tooltip = (
            "Select model:\n"
            "- qwen3-vl-2b-q4_k_m: 4GB VRAM, dynamic res, fast\n"
            "- qwen2.5-vl-7b-q4_k_m: 6GB VRAM, dynamic res, recommended\n"
            "- llama-3.2-vision-11b-q4_k_m: 8GB VRAM, 1120x1120 native\n"
            "- pixtral-12b-q4_k_m: 10GB VRAM, 1024x1024 native\n"
            "- minicpm-v-2_6-q4_k_m: 10GB VRAM, dynamic res\n"
            "- qwen2.5-vl-7b-q8: 10GB VRAM, dynamic res, best quality"
        )

        return comfy_io.Schema(
            node_id="SID_GGUF_LLM",
            display_name="SID GGUF LLM [Deprecated]",
            category="SID Photography Toolkit/LLM Providers",
            description=model_info_text,
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=model_options,
                    default=cls.get_default_model(),
                    tooltip=model_tooltip
                ),
                comfy_io.Combo.Input(
                    "memory_mode",
                    options=memory_presets,
                    default="Auto (Recommended)",
                    display_name="Memory Mode",
                    tooltip="How to use your GPU memory. Auto works for most users. Use Low VRAM if you get out-of-memory errors."
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=1.5,
                    step=0.1,
                    round=0.1,
                    display_name="Creativity",
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="0.0-0.3=precise/accurate, 0.5-0.7=balanced (recommended), 0.8-1.5=creative/varied"
                ),
                comfy_io.Boolean.Input(
                    "auto_download",
                    default=True,
                    tooltip="Automatically download model if not present"
                ),
                comfy_io.String.Input(
                    "hf_token",
                    default="",
                    tooltip="HuggingFace token for gated/private models (optional)"
                ),
                comfy_io.Combo.Input(
                    "max_image_size",
                    options=["512", "768", "1024", "Original"],
                    default="512",
                    display_name="Max Image Size",
                    tooltip="Resize images before encoding for faster processing. 512=~4x faster (default), 768=~2x faster, 1024=~1.5x faster. Original=no resize (slowest but most detail)."
                ),
                comfy_io.Boolean.Input(
                    "enable_reasoning",
                    default=False,
                    display_name="Enable Reasoning (Agentic)",
                    tooltip="Enable Agentic/reasoning mode in Advanced V2 generator. Note: Current GGUF models don't support reasoning yet."
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
        memory_mode: str,
        temperature: float,
        auto_download: bool,
        hf_token: str,
        max_image_size: str,
        enable_reasoning: bool,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""

        # Early check for llama-cpp-python
        import importlib.util
        if importlib.util.find_spec("llama_cpp") is None:
            import platform
            system = platform.system()

            error_msg = (
                "\n" + "=" * 70 + "\n"
                "ERROR: llama-cpp-python is not installed!\n"
                "=" * 70 + "\n\n"
                "The SID_GGUF_LLM node requires llama-cpp-python to run local models.\n"
                "This package requires platform-specific installation.\n\n"
            )

            if system == "Windows":
                error_msg += (
                    "For Windows with NVIDIA GPU (CUDA 12.1+):\n"
                    "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                    "For Windows CPU-only:\n"
                    "  pip install llama-cpp-python\n\n"
                )
            elif system == "Darwin":  # macOS
                error_msg += (
                    "For macOS with Apple Silicon (M1/M2/M3):\n"
                    "  CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python\n\n"
                    "For macOS Intel:\n"
                    "  pip install llama-cpp-python\n\n"
                )
            else:  # Linux
                error_msg += (
                    "For Linux with NVIDIA GPU (CUDA 12.1+):\n"
                    "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                    "For Linux with AMD GPU (ROCm):\n"
                    "  CMAKE_ARGS=\"-DGGML_HIPBLAS=on\" pip install llama-cpp-python\n\n"
                    "For Linux CPU-only:\n"
                    "  pip install llama-cpp-python\n\n"
                )

            error_msg += (
                "After installing, restart ComfyUI.\n"
                "=" * 70
            )

            print(error_msg)
            raise RuntimeError(error_msg)

        # Convert memory preset to n_ctx and n_gpu_layers
        memory_settings = {
            "Auto (Recommended)": {"n_ctx": 8192, "n_gpu_layers": -1},
            "Low VRAM (4-6GB)": {"n_ctx": 4096, "n_gpu_layers": 20},
            "Medium VRAM (8-12GB)": {"n_ctx": 8192, "n_gpu_layers": -1},
            "High VRAM (16GB+)": {"n_ctx": 16384, "n_gpu_layers": -1},
            "CPU Only (No GPU)": {"n_ctx": 4096, "n_gpu_layers": 0},
        }

        mem_config = memory_settings.get(memory_mode, memory_settings["Auto (Recommended)"])
        n_ctx = mem_config["n_ctx"]
        n_gpu_layers = mem_config["n_gpu_layers"]

        # Get model info and metadata
        model_supports_reasoning_capability = False
        model_max_output_tokens = 2048  # Default fallback
        max_image_resolution = "dynamic"

        if model not in GGUF_MODELS:
            if model.startswith("custom:"):
                # Custom model - user placed it manually
                custom_filename = model.replace("custom:", "") + ".gguf"
                model_path = os.path.join(LLM_GGUF_DIR, custom_filename)
                mmproj_path = None
                chat_format = "chatml"  # Default format for custom models

                if not os.path.exists(model_path):
                    raise ValueError(f"Custom model not found: {model_path}")
            else:
                raise ValueError(f"Unknown model: {model}")
        else:
            model_info = GGUF_MODELS[model]
            model_path = os.path.join(LLM_GGUF_DIR, model_info.filename)
            mmproj_path = os.path.join(LLM_GGUF_DIR, model_info.mmproj_filename) if model_info.mmproj_filename else None
            chat_format = model_info.chat_format

            # Get model metadata
            model_supports_reasoning_capability = model_info.supports_reasoning
            model_max_output_tokens = model_info.max_output_tokens
            max_image_resolution = model_info.max_image_resolution

            # Check if model exists or needs download
            if not os.path.exists(model_path):
                if auto_download:
                    print(f"[SID_GGUF_LLM] Model not found, downloading...")
                    # Pass hf_token for authenticated downloads
                    token = hf_token.strip() if hf_token else None
                    model_path, mmproj_path = download_model(model_info, token)
                else:
                    raise ValueError(
                        f"Model not found: {model_path}. "
                        f"Enable 'auto_download' or manually download from: "
                        f"{model_info.url}"
                    )

        # Use model's max_output_tokens directly
        max_tokens = model_max_output_tokens

        # Determine if reasoning should be enabled
        # Only enable if BOTH model supports it AND user enabled it
        actual_reasoning = model_supports_reasoning_capability and enable_reasoning

        if enable_reasoning and not model_supports_reasoning_capability:
            print(f"[SID_GGUF_LLM] Note: {model} does not support reasoning mode")

        # Create configuration with all needed info for client creation
        config = LLMModelConfig(
            provider=cls.PROVIDER_NAME,
            model=model,
            api_key="",  # No API key needed
            api_url="",  # Local model
            max_tokens=max_tokens,
            temperature=temperature,
            supports_vision=True,
            supports_system_prompt=True,
            supports_reasoning=actual_reasoning,
            extra_params={
                "model_path": model_path,
                "mmproj_path": mmproj_path,
                "chat_format": chat_format,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "model_max_output_tokens": model_max_output_tokens,
                "max_image_resolution": max_image_resolution,
                "max_image_size": None if max_image_size == "Original" else int(max_image_size),
            },
        )

        print(f"[SID_GGUF_LLM] Configured: {model}")
        print(f"  Model: {model_path}")
        if mmproj_path:
            print(f"  Vision: {mmproj_path}")
        print(f"  Memory: {memory_mode} (ctx={n_ctx}, gpu_layers={n_gpu_layers})")
        print(f"  Max tokens: {max_tokens}, Model resolution: {max_image_resolution}")
        resize_info = f"resize to {max_image_size}px" if max_image_size != "Original" else "no resize"
        print(f"  Image preprocessing: {resize_info}")
        print(f"  Creativity: {temperature}, Reasoning: {actual_reasoning}")

        return comfy_io.NodeOutput(config)
