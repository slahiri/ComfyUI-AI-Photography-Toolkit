# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit

LLM Provider nodes for ComfyUI - Cloud and Local model support.
All nodes are prefixed with SID_ for easy identification.

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
Version: 4.3.0
"""

__version__ = "4.3.0"

import sys
import subprocess
import importlib.util
from pathlib import Path

# =============================================================================
# Load Settings (including debug_mode)
# =============================================================================

DEBUG_MODE = False
_settings = {}

def load_settings():
    """Load settings from config/settings.toml"""
    global DEBUG_MODE, _settings

    settings_path = Path(__file__).parent / "config" / "settings.toml"

    if settings_path.exists():
        try:
            # Python 3.11+ has tomllib built-in
            import tomllib
            with open(settings_path, "rb") as f:
                _settings = tomllib.load(f)
        except ImportError:
            # Fallback for Python < 3.11
            try:
                import tomli
                with open(settings_path, "rb") as f:
                    _settings = tomli.load(f)
            except ImportError:
                # Manual parse for debug_mode only
                with open(settings_path, "r") as f:
                    content = f.read()
                    if "debug_mode = true" in content.lower():
                        _settings = {"development": {"debug_mode": True}}

        DEBUG_MODE = _settings.get("development", {}).get("debug_mode", False)

load_settings()

# Track installation status for welcome message
_dependency_status = {}


def check_and_install_dependencies():
    """
    Check and install required dependencies. Track status for logging.
    """
    global _dependency_status

    # Required dependencies (auto-installed)
    dependencies = {
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
        "transformers": {
            "import_name": "transformers",
            "pip_spec": "transformers>=4.57.0",
            "description": "HuggingFace Transformers (for local models)",
        },
        "accelerate": {
            "import_name": "accelerate",
            "pip_spec": "accelerate>=0.33.0",
            "description": "HuggingFace Accelerate (for model loading)",
        },
        "psutil": {
            "import_name": "psutil",
            "pip_spec": "psutil>=5.9.0",
            "description": "System utilities (for memory detection)",
        },
        "ultralytics": {
            "import_name": "ultralytics",
            "pip_spec": "ultralytics>=8.0.0",
            "description": "YOLO models (for human detection)",
        },
        # MediaPipe removed - has protobuf compatibility issues on Windows
        # YOLO provides sufficient human detection
        "opencv": {
            "import_name": "cv2",
            "pip_spec": "opencv-python>=4.8.0",
            "description": "OpenCV (for image processing)",
        },
        "sklearn": {
            "import_name": "sklearn",
            "pip_spec": "scikit-learn>=1.3.0",
            "description": "Scikit-learn (for color clustering)",
        },
        "aiohttp": {
            "import_name": "aiohttp",
            "pip_spec": "aiohttp>=3.8.0",
            "description": "Async HTTP (for web routes)",
        },
        "tomlkit": {
            "import_name": "tomlkit",
            "pip_spec": "tomlkit>=0.12.0",
            "description": "TOML parser (preserves comments)",
        },
    }

    python_exe = sys.executable

    # Check and install required dependencies
    for pkg_name, pkg_info in dependencies.items():
        import_name = pkg_info["import_name"]
        pip_spec = pkg_info["pip_spec"]

        if importlib.util.find_spec(import_name) is not None:
            _dependency_status[pkg_name] = "installed"
        else:
            # Try to install
            try:
                subprocess.check_call(
                    [python_exe, "-m", "pip", "install", pip_spec],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                _dependency_status[pkg_name] = "just_installed"
            except subprocess.CalledProcessError:
                _dependency_status[pkg_name] = "failed"


# Run dependency check on module load
check_and_install_dependencies()

# Track model download status
_model_status = {}


def download_cv_models():
    """
    Pre-download CV models (YOLO, MediaPipe) at init time.
    This ensures models are ready when user needs them.
    """
    global _model_status

    print("\n[SID-Toolkit] Checking CV models...")

    # Download YOLOv8n model
    try:
        from ultralytics import YOLO
        import os

        # YOLO downloads to ~/.cache/ultralytics or current dir
        # Check if model exists, if not it will download
        print("[SID-Toolkit] Loading YOLO model (yolov8n.pt)...")
        model = YOLO("yolov8n.pt")
        _model_status["yolo"] = "ready"
        print("[SID-Toolkit] YOLO model ready")

        # Clean up to free memory
        del model
    except ImportError:
        _model_status["yolo"] = "missing_ultralytics"
        print("[SID-Toolkit] WARNING: ultralytics not installed, YOLO unavailable")
    except Exception as e:
        _model_status["yolo"] = f"error: {e}"
        print(f"[SID-Toolkit] WARNING: YOLO model download failed: {e}")

    # MediaPipe disabled due to protobuf compatibility issues on Windows
    # YOLO provides sufficient person detection
    _model_status["mediapipe"] = "disabled"

    print("[SID-Toolkit] CV model check complete\n")


# Download CV models at init
download_cv_models()

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Import LLM provider nodes
from .llm_providers.sid_llm_api import SID_LLM_API
from .llm_providers.sid_llm_local import SID_LLM_Local, SID_LLM_Local_API

# Import Z-Image Prompt Generator V2 (CV-based detection + simplified interface)
from .sid_prompt_generator_v2 import SID_ZImagePromptGeneratorV2 as SID_ZImagePromptGenerator

# Import Debug Agent (always available)
try:
    from .sid_prompt_debug import SID_PromptDebugAgent
except ImportError as e:
    SID_PromptDebugAgent = None
    print(f"[SID-Toolkit] Debug Agent not available: {e}")

# Import Prompt Editor routes
try:
    from server import PromptServer
    from .prompt_editor import setup_routes as setup_prompt_editor_routes
    _prompt_editor_available = True
except ImportError as e:
    _prompt_editor_available = False
    print(f"[SID-Toolkit] Prompt Editor not available: {e}")


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
        nodes = [
            SID_LLM_API,              # Cloud APIs (Anthropic, OpenAI, Gemini, Groq, etc.)
            SID_LLM_Local,            # Local transformers models (QwenVL, Florence, etc.)
            SID_LLM_Local_API,        # Local API providers (Ollama, LM Studio, OpenAI Compatible)
            SID_ZImagePromptGenerator,  # Z-Image prompt generator
        ]

        # Add debug node if available (debug_mode = true)
        if SID_PromptDebugAgent is not None:
            nodes.append(SID_PromptDebugAgent)

        return nodes


def print_welcome_message():
    """
    Print welcome message with dependency status and available nodes.
    """
    global _dependency_status, _model_status

    # Header
    print("")
    print("=" * 65)
    print("  SID Photography Toolkit for ComfyUI")
    print(f"  Version: {__version__}")
    print("  Author: Siddhartha Lahiri")
    print("=" * 65)

    # CV Models status
    print("")
    print("  CV Models:")
    yolo_status = _model_status.get("yolo", "unknown")
    print(f"    [{'OK' if yolo_status == 'ready' else 'X'}] YOLO v8-nano (human detection)")

    # Dependencies status
    print("")
    print("  Dependencies:")

    # Required dependencies
    required_pkgs = ["anthropic", "openai", "pyyaml", "requests", "typing_extensions",
                     "transformers", "accelerate", "psutil", "ultralytics",
                     "opencv", "sklearn", "aiohttp", "tomlkit"]
    pkg_descriptions = {
        "anthropic": "Anthropic Claude API",
        "openai": "OpenAI/Grok/LM Studio API",
        "pyyaml": "YAML config parser",
        "requests": "HTTP library (Ollama)",
        "typing_extensions": "Type hints",
        "transformers": "HuggingFace Transformers",
        "accelerate": "HuggingFace Accelerate",
        "psutil": "System memory detection",
        "ultralytics": "YOLO (human detection)",
        "opencv": "OpenCV (image processing)",
        "sklearn": "Scikit-learn (color analysis)",
        "aiohttp": "Async HTTP (web routes)",
        "tomlkit": "TOML parser (prompt editor)",
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
    print("  Prompt Tools:")
    print("    - SID_ZImagePromptGenerator  Z-Image prompts (CV detection + Vision LLM)")
    print("                                 Fast YOLO/MediaPipe detection, Z-Image optimized")

    # Debug/Testing tools
    if SID_PromptDebugAgent is not None:
        print("")
        print("  Testing Tools:")
        print("    - SID_PromptDebugAgent  Evaluate prompt quality with Claude Opus 4.5")
        print("                            Compares source/output, scores against best practices")

    # Web Tools
    if _prompt_editor_available:
        print("")
        print("  Web Tools:")
        print("    - Prompt Editor:      http://localhost:8188/sid/prompt-editor")
        print("                          Edit TOML prompt templates in browser")
        print("    - Generation Results: http://localhost:8188/sid/generation-results")
        print("                          Browse saved prompts and metadata")
        print("    - Debug Viewer:       http://localhost:8188/sid/debug-results")
        print("                          Browse prompt evaluation results")

    print("")
    print("  Standalone CLI (batch processing for training):")
    print("    python prompt_generator_core.py --batch ./images/ --output prompts.jsonl")
    print("")
    print("=" * 65)
    print("")


async def comfy_entrypoint() -> SIDPhotographyToolkitExtension:
    """
    ComfyUI calls this function to load the extension and its nodes.
    This is the entry point for the custom node package.
    """
    # Register Prompt Editor web routes
    if _prompt_editor_available:
        try:
            setup_prompt_editor_routes(PromptServer.instance.routes)
            print("[SID-Toolkit] Prompt Editor available at: http://localhost:8188/sid/prompt-editor")
        except Exception as e:
            print(f"[SID-Toolkit] Failed to register Prompt Editor routes: {e}")

    print_welcome_message()
    return SIDPhotographyToolkitExtension()
