"""
SID_GGUF_LLM Node

Local GGUF model provider for ComfyUI using llama-cpp-python.
Supports vision-language models like LLaVA, MiniCPM-V, and Moondream.
Models are downloaded to ComfyUI/models/LLM/GGUF/ folder.
"""

import os
import sys
from typing import List, Optional, Dict, Any
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
    """Information about a GGUF model."""
    name: str
    filename: str
    url: str
    size_mb: int
    vram_gb: float
    description: str
    chat_format: str  # llava-1-5, llava-1-6, chatml, minicpm-v, moondream2, etc.
    mmproj_filename: Optional[str] = None  # For models needing separate vision encoder
    mmproj_url: Optional[str] = None


# Available GGUF vision models with download URLs
GGUF_MODELS: Dict[str, GGUFModelInfo] = {
    # ===== SMALL MODELS (~4GB VRAM) =====
    "moondream2-q4_k_m": GGUFModelInfo(
        name="Moondream2 Q4_K_M",
        filename="moondream2-text-model-f16.gguf",
        url="https://huggingface.co/moondream/moondream2-gguf/resolve/main/moondream2-text-model-f16.gguf",
        size_mb=3600,
        vram_gb=4.0,
        description="Small & fast vision model, good for basic tasks",
        chat_format="moondream2",
        mmproj_filename="moondream2-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/moondream/moondream2-gguf/resolve/main/moondream2-mmproj-f16.gguf",
    ),

    # ===== MEDIUM MODELS (~8-12GB VRAM) =====
    "llava-v1.5-7b-q4_k_m": GGUFModelInfo(
        name="LLaVA 1.5 7B Q4_K_M",
        filename="llava-v1.5-7b-Q4_K_M.gguf",
        url="https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf",
        size_mb=4100,
        vram_gb=8.0,
        description="Balanced vision model, good quality",
        chat_format="llava-1-5",
        mmproj_filename="llava-v1.5-7b-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf",
    ),

    "llava-v1.6-mistral-7b-q4_k_m": GGUFModelInfo(
        name="LLaVA 1.6 Mistral 7B Q4_K_M",
        filename="llava-v1.6-mistral-7b.Q4_K_M.gguf",
        url="https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf",
        size_mb=4400,
        vram_gb=8.0,
        description="Fast LLaVA 1.6 on Mistral backbone, great balance",
        chat_format="llava-1-6",
        mmproj_filename="llava-v1.6-mistral-7b-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf",
    ),

    "qwen2.5-vl-7b-q4_k_m": GGUFModelInfo(
        name="Qwen2.5-VL 7B Q4_K_M",
        filename="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        size_mb=5200,
        vram_gb=6.0,
        description="Qwen vision model Q4, good quality at low VRAM",
        chat_format="chatml",
        mmproj_filename="Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
    ),

    "minicpm-v-2_6-q4_k_m": GGUFModelInfo(
        name="MiniCPM-V 2.6 Q4_K_M",
        filename="MiniCPM-V-2_6-Q4_K_M.gguf",
        url="https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q4_K_M.gguf",
        size_mb=4800,
        vram_gb=10.0,
        description="Excellent vision understanding, multilingual",
        chat_format="minicpm-v-2.6",
        mmproj_filename="MiniCPM-V-2_6-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf",
    ),

    "qwen2.5-vl-7b-q8": GGUFModelInfo(
        name="Qwen2.5-VL 7B Q8",
        filename="Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
        url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
        size_mb=8100,
        vram_gb=10.0,
        description="Qwen vision model Q8, excellent understanding",
        chat_format="chatml",
        mmproj_filename="Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
    ),

    # ===== LARGE MODELS (~16GB+ VRAM) =====
    "llava-v1.5-13b-q4_k_m": GGUFModelInfo(
        name="LLaVA 1.5 13B Q4_K_M",
        filename="llava-v1.5-13b-Q4_K_M.gguf",
        url="https://huggingface.co/mys/ggml_llava-v1.5-13b/resolve/main/ggml-model-q4_k.gguf",
        size_mb=7400,
        vram_gb=16.0,
        description="High quality vision model, needs more VRAM",
        chat_format="llava-1-5",
        mmproj_filename="llava-v1.5-13b-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/mys/ggml_llava-v1.5-13b/resolve/main/mmproj-model-f16.gguf",
    ),

    # ===== HIGH-END MODELS (~22GB+ VRAM) =====
    "llava-v1.6-34b-q4_k_m": GGUFModelInfo(
        name="LLaVA 1.6 34B Q4_K_M",
        filename="llava-v1.6-34b.Q4_K_M.gguf",
        url="https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/llava-v1.6-34b.Q4_K_M.gguf",
        size_mb=20700,
        vram_gb=22.0,
        description="Best quality open-source vision model, excellent for detailed prompts",
        chat_format="llava-1-6",
        mmproj_filename="llava-v1.6-34b-mmproj-f16.gguf",
        mmproj_url="https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/mmproj-model-f16.gguf",
    ),
}


class LocalGGUFClient:
    """
    Wrapper class that provides OpenAI-compatible interface for local GGUF models.
    This allows seamless integration with the existing _call_vision_llm routing.
    """

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
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.chat_format = chat_format

        # Create appropriate chat handler for vision models
        chat_handler = None
        if mmproj_path and os.path.exists(mmproj_path):
            if chat_format in ["llava-1-5", "llava-1-6", "minicpm-v-2.6", "chatml"]:
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=verbose)
            elif chat_format == "moondream2":
                chat_handler = MoondreamChatHandler(clip_model_path=mmproj_path, verbose=verbose)

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

    @property
    def chat(self):
        """Return self to allow OpenAI-style chaining: client.chat.completions.create()"""
        return self

    @property
    def completions(self):
        """Return self to allow OpenAI-style chaining: client.chat.completions.create()"""
        return self

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

        Args:
            model: Model name (ignored, uses loaded model)
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            OpenAI-compatible response object
        """
        # Run inference
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

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

    Supported models:
    - Moondream2: Small & fast (~4GB VRAM)
    - LLaVA 1.5/1.6: Various sizes from 7B to 34B
    - Qwen2.5-VL: Excellent vision understanding (~6-10GB VRAM)
    - MiniCPM-V 2.6: Multilingual vision (~10GB VRAM)
    """

    PROVIDER_NAME = "gguf"

    @classmethod
    def get_models(cls) -> List[str]:
        return get_available_models()

    @classmethod
    def get_default_model(cls) -> str:
        return "moondream2-q4_k_m"

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

        return comfy_io.Schema(
            node_id="SID_GGUF_LLM",
            display_name="SID GGUF LLM",
            category="SID Photography Toolkit/LLM Providers",
            description="Local GGUF vision model. No API needed. Models: Moondream, LLaVA, Qwen2.5-VL, MiniCPM-V",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=model_options,
                    default=cls.get_default_model(),
                    tooltip="Select model. Moondream=fast/small, LLaVA-7B=balanced, Qwen/MiniCPM=best quality"
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
                comfy_io.Int.Input(
                    "n_ctx",
                    default=4096,
                    min=512,
                    max=32768,
                    step=512,
                    tooltip="Context length (higher = more memory). Use 4096+ for larger models"
                ),
                comfy_io.Int.Input(
                    "n_gpu_layers",
                    default=-1,
                    min=-1,
                    max=100,
                    step=1,
                    tooltip="GPU layers (-1 = all on GPU, 0 = CPU only)"
                ),
                comfy_io.Int.Input(
                    "max_tokens",
                    default=300,
                    min=50,
                    max=2048,
                    step=50,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Maximum tokens in response"
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity level (0=focused, 2=creative)"
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
        auto_download: bool,
        hf_token: str,
        n_ctx: int,
        n_gpu_layers: int,
        max_tokens: int,
        temperature: float,
    ) -> comfy_io.NodeOutput:
        """Create and return the LLM model configuration."""

        # Get model info
        if model not in GGUF_MODELS:
            if model.startswith("custom:"):
                # Custom model - user placed it manually
                custom_filename = model.replace("custom:", "") + ".gguf"
                model_path = os.path.join(LLM_GGUF_DIR, custom_filename)
                mmproj_path = None
                chat_format = "llava-1-5"  # Default format for custom models

                if not os.path.exists(model_path):
                    raise ValueError(f"Custom model not found: {model_path}")
            else:
                raise ValueError(f"Unknown model: {model}")
        else:
            model_info = GGUF_MODELS[model]
            model_path = os.path.join(LLM_GGUF_DIR, model_info.filename)
            mmproj_path = os.path.join(LLM_GGUF_DIR, model_info.mmproj_filename) if model_info.mmproj_filename else None
            chat_format = model_info.chat_format

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
            extra_params={
                "model_path": model_path,
                "mmproj_path": mmproj_path,
                "chat_format": chat_format,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
            },
        )

        print(f"[SID_GGUF_LLM] Configured: {model}")
        print(f"  Model: {model_path}")
        if mmproj_path:
            print(f"  Vision: {mmproj_path}")
        print(f"  Context: {n_ctx}, GPU layers: {n_gpu_layers}")
        print(f"  max_tokens={max_tokens}, temp={temperature}")

        return comfy_io.NodeOutput(config)
