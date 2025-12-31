"""
ComfyUI-AI-Photography-Toolkit
LLM Provider nodes for ComfyUI - Cloud and Local model support.
All nodes are prefixed with SID_ for easy identification.

Author: Siddhartha Lahiri
Version: 4.2.1
"""

__version__ = "4.2.1"

import sys
import subprocess
import importlib.util

# Track installation status for welcome message
_dependency_status = {}
_v3_api_available = False


def check_and_install_dependencies():
    """
    Check and install required dependencies. Track status for logging.
    """
    global _dependency_status

    # Required dependencies (auto-installed)
    dependencies = {
        # LLM API dependencies
        "anthropic": {
            "import_name": "anthropic",
            "pip_spec": "anthropic>=0.39.0",
            "description": "Anthropic Claude API SDK",
        },
        "openai": {
            "import_name": "openai",
            "pip_spec": "openai>=1.0.0",
            "description": "OpenAI API SDK (also used for Grok, LM Studio, etc.)",
        },
        "pyyaml": {
            "import_name": "yaml",
            "pip_spec": "pyyaml>=6.0",
            "description": "YAML configuration parser",
        },
        "requests": {
            "import_name": "requests",
            "pip_spec": "requests>=2.28.0",
            "description": "HTTP library (for Ollama API)",
        },
        "typing_extensions": {
            "import_name": "typing_extensions",
            "pip_spec": "typing_extensions>=4.0.0",
            "description": "Extended typing support",
        },
        # Tagger dependencies - Core
        "onnxruntime": {
            "import_name": "onnxruntime",
            "pip_spec": "onnxruntime>=1.16.0",
            "description": "ONNX Runtime for WD14/PixAI taggers",
        },
        "huggingface_hub": {
            "import_name": "huggingface_hub",
            "pip_spec": "huggingface_hub>=0.20.0",
            "description": "HuggingFace Hub for model downloads",
        },
        "opencv-python": {
            "import_name": "cv2",
            "pip_spec": "opencv-python>=4.8.0",
            "description": "OpenCV for image processing and face detection",
        },
        # Tagger dependencies - ML Models
        "transformers": {
            "import_name": "transformers",
            "pip_spec": "transformers>=4.36.0",
            "description": "HuggingFace Transformers for CLIP, JoyTag, etc.",
        },
        "scikit-learn": {
            "import_name": "sklearn",
            "pip_spec": "scikit-learn>=1.3.0",
            "description": "Scikit-learn for color clustering",
        },
        "safetensors": {
            "import_name": "safetensors",
            "pip_spec": "safetensors>=0.4.0",
            "description": "Safe model weight format",
        },
        # NudeNet for body detection
        "nudenet": {
            "import_name": "nudenet",
            "pip_spec": "nudenet>=3.0.0",
            "description": "NudeNet for body part detection",
        },
        # MediaPipe for pose detection (lightweight alternative)
        "mediapipe": {
            "import_name": "mediapipe",
            "pip_spec": "mediapipe>=0.10.0",
            "description": "MediaPipe for pose detection",
        },
    }

    python_exe = sys.executable
    install_count = 0

    # Check and install required dependencies
    for pkg_name, pkg_info in dependencies.items():
        import_name = pkg_info["import_name"]
        pip_spec = pkg_info["pip_spec"]
        description = pkg_info["description"]

        if importlib.util.find_spec(import_name) is not None:
            _dependency_status[pkg_name] = "installed"
        else:
            # Try to install
            print(f"[SID Photography Toolkit] Installing {pkg_name}: {description}...")
            try:
                subprocess.check_call(
                    [python_exe, "-m", "pip", "install", pip_spec],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                _dependency_status[pkg_name] = "just_installed"
                install_count += 1
                print(f"[SID Photography Toolkit] {pkg_name} installed successfully")
            except subprocess.CalledProcessError as e:
                _dependency_status[pkg_name] = "failed"
                print(f"[SID Photography Toolkit] Failed to install {pkg_name}: {e}")

    if install_count > 0:
        print(f"[SID Photography Toolkit] Installed {install_count} new dependencies")


# Run dependency check on module load
print("[SID Photography Toolkit] Checking dependencies...")
check_and_install_dependencies()

# Try to import V3 API (may not be available in older ComfyUI versions)
try:
    from typing_extensions import override
    from comfy_api.latest import ComfyExtension, io
    _v3_api_available = True
except ImportError:
    _v3_api_available = False
    print("[SID Photography Toolkit] V3 API not available, using V1 fallback")

# Import new Prompt Generator (modular implementation) - V1 style
from .core.prompt_generator.generator import SID_PromptGenerator

# Import LLM model type for V1 wrappers
from .llm_providers.llm_model_type import LLMModelConfig

# V3 API imports (conditional)
if _v3_api_available:
    # Import LLM provider nodes (unified cloud API + local models)
    from .llm_providers.sid_llm_api import SID_LLM_API
    from .llm_providers.sid_llm_local import SID_LLM_Local

    # Import Z-Image Prompt Generator (unified node)
    from .sid_zimage_prompt_generator import SID_ZImagePromptGenerator

    class SIDPhotographyToolkitExtension(ComfyExtension):
        """
        Extension class for SID Photography Toolkit.
        Registers SID_ prefixed LLM nodes with ComfyUI.
        """

        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            """
            Return list of all nodes in this extension.
            """
            return [
                SID_LLM_API,              # Cloud LLM (Anthropic, OpenAI, Gemini, Grok, Ollama, LM Studio)
                SID_LLM_Local,            # Local models (Florence-2, Moondream2, SmolVLM, Phi-3.5, QwenVL)
                SID_ZImagePromptGenerator,  # Z-Image prompt generator (auto Single-Shot/Agentic)
            ]


# =============================================================================
# V1-Style LLM Node Wrappers (fallback for older ComfyUI versions)
# =============================================================================

class SID_LLM_API_V1:
    """
    V1-style wrapper for SID_LLM_API.
    Cloud LLM provider supporting Anthropic, OpenAI, Gemini, Grok, Groq, etc.
    """

    CATEGORY = "SID Photography Toolkit/LLM Providers"
    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("llm_model",)
    FUNCTION = "execute"

    # Provider configurations
    PROVIDERS = {
        "Anthropic": {
            "api_url": "https://api.anthropic.com",
            "models": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101"],
            "default": "claude-sonnet-4-5-20250929",
        },
        "OpenAI": {
            "api_url": "https://api.openai.com/v1",
            "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
            "default": "gpt-4o",
        },
        "Google Gemini": {
            "api_url": "https://generativelanguage.googleapis.com/v1beta",
            "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            "default": "gemini-2.5-flash",
        },
        "Groq (Free)": {
            "api_url": "https://api.groq.com/openai/v1",
            "models": ["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"],
            "default": "meta-llama/llama-4-scout-17b-16e-instruct",
        },
        "Together AI": {
            "api_url": "https://api.together.xyz/v1",
            "models": ["meta-llama/Llama-Vision-Free", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"],
            "default": "meta-llama/Llama-Vision-Free",
        },
        "OpenRouter": {
            "api_url": "https://openrouter.ai/api/v1",
            "models": ["google/gemini-2.0-flash-exp:free", "google/gemma-3-27b-it:free"],
            "default": "google/gemini-2.0-flash-exp:free",
        },
        "Ollama (Local)": {
            "api_url": "http://localhost:11434/v1",
            "models": ["(use custom_model)"],
            "default": "(use custom_model)",
        },
        "LM Studio (Local)": {
            "api_url": "http://localhost:1234/v1",
            "models": ["(use custom_model)"],
            "default": "(use custom_model)",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        providers = list(cls.PROVIDERS.keys())
        all_models = []
        for p, cfg in cls.PROVIDERS.items():
            for m in cfg["models"]:
                all_models.append(f"[{p}] {m}")

        return {
            "required": {
                "provider": (providers, {"default": "Anthropic"}),
                "model": (all_models, {"default": "[Anthropic] claude-sonnet-4-5-20250929"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "tooltip": "Override model name"}),
                "api_url": ("STRING", {"default": "", "tooltip": "Custom API URL"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 128, "max": 32000}),
            },
        }

    def execute(self, provider, model, api_key, custom_model="", api_url="", temperature=0.3, max_tokens=2048):
        # Parse model selection
        if model.startswith("["):
            end = model.find("]")
            model_provider = model[1:end]
            model_name = model[end + 2:]
        else:
            model_provider = provider
            model_name = model

        # Use custom model if provided
        if custom_model.strip():
            model_name = custom_model.strip()

        # Determine API URL
        if api_url.strip():
            actual_url = api_url.strip()
        else:
            actual_url = self.PROVIDERS.get(provider, {}).get("api_url", "")

        # Determine provider name for API calls
        provider_map = {
            "Anthropic": "anthropic",
            "OpenAI": "openai",
            "Google Gemini": "gemini",
            "Groq (Free)": "groq",
            "Together AI": "together",
            "OpenRouter": "openrouter",
            "Ollama (Local)": "ollama",
            "LM Studio (Local)": "lmstudio",
        }
        provider_name = provider_map.get(provider, "openai")

        # Check for reasoning support
        reasoning_models = ["claude-sonnet-4", "claude-opus-4", "o1", "o3", "deepseek-r1"]
        supports_reasoning = any(rm in model_name.lower() for rm in reasoning_models)

        config = LLMModelConfig(
            provider=provider_name,
            model=model_name,
            api_key=api_key.strip(),
            api_url=actual_url,
            max_tokens=max_tokens,
            temperature=temperature,
            supports_vision=True,
            supports_system_prompt=True,
            supports_reasoning=supports_reasoning,
        )

        print(f"[SID_LLM_API] Configured: {provider} - {model_name}")
        return (config,)


class SID_LLM_Local_V1:
    """
    V1-style wrapper for SID_LLM_Local.
    Local vision models with VRAM management.
    """

    CATEGORY = "SID Photography Toolkit/LLM Providers"
    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("llm_model",)
    FUNCTION = "execute"

    LOCAL_MODELS = [
        "Qwen3-VL-2B-Abliterated",
        "Qwen3-VL-2B-Instruct",
        "Qwen3-VL-4B-Abliterated",
        "Qwen3-VL-4B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Abliterated",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen3-VL-8B-Abliterated",
        "Qwen3-VL-8B-Instruct",
    ]

    MODEL_REPOS = {
        "Qwen3-VL-2B-Abliterated": "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated",
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen3-VL-4B-Abliterated": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
        "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Abliterated": "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated",
        "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen3-VL-8B-Abliterated": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.LOCAL_MODELS, {"default": "Qwen3-VL-4B-Abliterated"}),
            },
            "optional": {
                "quantization": (["Auto", "4-bit", "8-bit", "FP16"], {"default": "Auto"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 128, "max": 4096}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated models"}),
            },
        }

    def execute(self, model, quantization="Auto", device="auto", temperature=0.3,
                max_tokens=1024, keep_model_loaded=True, hf_token=""):

        # Map quantization
        quant_map = {"Auto": "4-bit", "4-bit": "4-bit", "8-bit": "8-bit", "FP16": "None (FP16)"}
        quant = quant_map.get(quantization, "4-bit")

        repo_id = self.MODEL_REPOS.get(model, "")

        config = LLMModelConfig(
            provider="local",
            model=model,
            api_key="",
            api_url="",
            max_tokens=max_tokens,
            temperature=temperature,
            supports_vision=True,
            supports_system_prompt=True,
            supports_reasoning=False,
            extra_params={
                "quantization": quant,
                "device": device,
                "attention_mode": "auto",
                "keep_model_loaded": keep_model_loaded,
                "repo_id": repo_id,
                "family": "qwenvl",
                "repetition_penalty": 1.2,
                "top_p": 0.9,
                "use_torch_compile": False,
                "hf_token": hf_token if hf_token else None,
            },
        )

        print(f"[SID_LLM_Local] Configured: {model} ({quant})")
        return (config,)


def print_welcome_message():
    """
    Print welcome message with dependency status and available nodes.
    """
    global _dependency_status

    # Header
    print("")
    print("=" * 65)
    print("  SID Photography Toolkit for ComfyUI")
    print(f"  Version: {__version__}")
    print("  Author: Siddhartha Lahiri")
    print("=" * 65)

    # Dependencies status
    print("")
    print("  Dependencies:")

    # Required dependencies
    required_pkgs = ["anthropic", "openai", "pyyaml", "requests", "typing_extensions"]
    pkg_descriptions = {
        "anthropic": "Anthropic Claude API",
        "openai": "OpenAI/Grok/LM Studio API",
        "pyyaml": "YAML config parser",
        "requests": "HTTP library (Ollama)",
        "typing_extensions": "Type hints",
    }

    for pkg in required_pkgs:
        status = _dependency_status.get(pkg, "unknown")
        desc = pkg_descriptions.get(pkg, pkg)
        if status == "installed":
            print(f"    [OK] {desc}")
        elif status == "just_installed":
            print(f"    [INSTALLED] {desc} (just installed)")
        elif status == "failed":
            print(f"    [FAILED] {desc} - please install manually")
        else:
            print(f"    [?] {desc}")

    # Available nodes
    print("")
    print("-" * 65)
    print("  Available Nodes:")
    print("")
    print("  LLM Providers:")
    print("    - SID_LLM_API   Cloud APIs (Anthropic, OpenAI, Gemini, Grok, Ollama, LM Studio)")
    print("    - SID_LLM_Local Local models (Florence-2, Moondream, SmolVLM, Phi-3.5, QwenVL)")
    print("")
    print("  Prompt Generators:")
    print("    - SID_ZImagePromptGenerator  Z-Image prompts (auto Single-Shot/Agentic)")
    print("    - SID_PromptGenerator        New modular prompt generator (4.2.1)")

    print("")
    print("=" * 65)
    print("")


# Import User Prompt node
from .core.prompt_generator.nodes.user_prompt import SID_UserPrompt

# Import Prompt Modifier node
from .core.prompt_generator.nodes.prompt_modifier import SID_PromptModifier

# V1-style node exports (always available, works with all ComfyUI versions)
NODE_CLASS_MAPPINGS = {
    "SID_PromptGenerator": SID_PromptGenerator,
    "SID_UserPrompt": SID_UserPrompt,
    "SID_PromptModifier": SID_PromptModifier,
    "SID_LLM_API": SID_LLM_API_V1,
    "SID_LLM_Local": SID_LLM_Local_V1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptGenerator": "SID Prompt Generator",
    "SID_UserPrompt": "SID User Prompt",
    "SID_PromptModifier": "SID Prompt Modifier",
    "SID_LLM_API": "SID LLM API",
    "SID_LLM_Local": "SID LLM Local",
}

# Web directory for JavaScript extensions (node colors, widget control)
WEB_DIRECTORY = "./web"


# V3 entrypoint (only if V3 API available)
if _v3_api_available:
    async def comfy_entrypoint() -> SIDPhotographyToolkitExtension:
        """
        ComfyUI calls this function to load the extension and its nodes.
        This is the entry point for the custom node package.
        """
        print_welcome_message()
        return SIDPhotographyToolkitExtension()
else:
    # Print welcome message for V1-only mode
    print_welcome_message()
