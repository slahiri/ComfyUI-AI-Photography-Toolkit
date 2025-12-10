"""
ComfyUI-AI-Photography-Toolkit
A collection of AI-powered photography and image generation tools for ComfyUI.
All nodes are prefixed with SID_ for easy identification.

Author: Siddhartha Lahiri
Version: 4.0.0
"""

__version__ = "4.2.0"

import os
import sys
import subprocess
import importlib.util

# Track installation status for welcome message
_dependency_status = {}
_gpu_info = {}


def detect_gpu_info():
    """
    Detect GPU type and CUDA version for llama-cpp-python installation.
    Supports: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), CPU
    Works on Windows, Linux, and macOS.
    """
    global _gpu_info
    import platform

    system = platform.system()  # 'Windows', 'Linux', 'Darwin'
    machine = platform.machine()  # 'x86_64', 'arm64', 'AMD64'

    # Check for Apple Silicon (macOS with ARM)
    if system == "Darwin" and machine == "arm64":
        _gpu_info = {"type": "apple_metal", "system": "macOS", "chip": "Apple Silicon"}
        return _gpu_info

    # Try to detect via PyTorch (most reliable in ComfyUI environment)
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                major, minor = cuda_version.split(".")[:2]
                _gpu_info = {
                    "type": "nvidia",
                    "cuda_version": cuda_version,
                    "cuda_major": int(major),
                    "cuda_minor": int(minor),
                    "system": system,
                }
                return _gpu_info
        # Check for MPS (Apple Metal via PyTorch) - fallback for macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _gpu_info = {"type": "apple_metal", "system": "macOS", "chip": "Apple Silicon"}
            return _gpu_info
    except ImportError:
        pass

    # Try nvidia-smi as fallback (Windows/Linux)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            _gpu_info = {"type": "nvidia", "cuda_version": "12.2", "cuda_major": 12, "cuda_minor": 2, "system": system}
            return _gpu_info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for AMD ROCm (Linux only)
    if system == "Linux":
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                _gpu_info = {"type": "amd", "rocm": True, "system": system}
                return _gpu_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if os.environ.get("ROCM_HOME") or os.path.exists("/opt/rocm"):
            _gpu_info = {"type": "amd", "rocm": True, "system": system}
            return _gpu_info

    # Default to CPU
    _gpu_info = {"type": "cpu", "system": system}
    return _gpu_info


def install_llama_cpp_python():
    """
    Auto-install llama-cpp-python with the correct GPU support.
    Returns True if installation successful, False otherwise.
    """
    global _gpu_info

    if not _gpu_info:
        detect_gpu_info()

    gpu_type = _gpu_info.get("type", "cpu")
    python_exe = sys.executable

    print("")
    print("-" * 60)
    print("[SID_GGUF_LLM] Auto-installing llama-cpp-python...")
    print(f"  Detected GPU: {gpu_type.upper()}")
    sys.stdout.flush()

    try:
        if gpu_type == "nvidia":
            cuda_major = _gpu_info.get("cuda_major", 12)
            cuda_minor = _gpu_info.get("cuda_minor", 2)
            cuda_version = _gpu_info.get("cuda_version", "12.2")

            print(f"  CUDA Version: {cuda_version}")

            # Choose wheel based on CUDA version
            if cuda_major == 12 and cuda_minor < 2:
                wheel_url = "https://abetlen.github.io/llama-cpp-python/whl/cu121"
                print("  Using: CUDA 12.1 wheel")
            else:
                wheel_url = "https://abetlen.github.io/llama-cpp-python/whl/cu122"
                print("  Using: CUDA 12.2+ wheel")

            print("  Installing... (this may take a few minutes)")
            print("-" * 60)
            sys.stdout.flush()
            # Show pip output in real-time
            subprocess.check_call(
                [python_exe, "-m", "pip", "install", "llama-cpp-python",
                 "--extra-index-url", wheel_url, "--progress-bar", "on"]
            )

        elif gpu_type == "apple_metal":
            print("  Using: Apple Metal build (M1/M2/M3)")
            print("  Installing... (this may take a few minutes)")
            print("-" * 60)
            sys.stdout.flush()
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DGGML_METAL=on"
            subprocess.check_call(
                [python_exe, "-m", "pip", "install", "llama-cpp-python", "--progress-bar", "on"],
                env=env
            )

        elif gpu_type == "amd":
            print("  Using: AMD ROCm build")
            print("  Installing... (this may take several minutes to compile)")
            print("-" * 60)
            sys.stdout.flush()
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DGGML_HIPBLAS=on"
            subprocess.check_call(
                [python_exe, "-m", "pip", "install", "llama-cpp-python", "--progress-bar", "on"],
                env=env
            )

        else:  # CPU
            print("  Using: CPU-only build")
            print("  Installing...")
            print("-" * 60)
            sys.stdout.flush()
            subprocess.check_call(
                [python_exe, "-m", "pip", "install", "llama-cpp-python", "--progress-bar", "on"]
            )

        print("  [OK] llama-cpp-python installed successfully!")
        print("-" * 60)
        return True

    except subprocess.CalledProcessError as e:
        print(f"  [FAILED] Installation failed: {e}")
        print("  Please install manually - see README for instructions")
        print("-" * 60)
        return False


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
    }

    # Optional dependencies (NOT auto-installed, require manual setup)
    optional_dependencies = {
        "llama_cpp": {
            "import_name": "llama_cpp",
            "description": "Local GGUF model inference (requires GPU-specific build)",
            "install_instructions": [
                "NVIDIA CUDA 12.1: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121",
                "NVIDIA CUDA 12.2+: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122",
                "CPU only: pip install llama-cpp-python",
                "AMD ROCm: CMAKE_ARGS=\"-DGGML_HIPBLAS=on\" pip install llama-cpp-python",
            ],
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

    # Check optional dependencies - auto-install llama-cpp-python with GPU detection
    for pkg_name, pkg_info in optional_dependencies.items():
        import_name = pkg_info["import_name"]
        if importlib.util.find_spec(import_name) is not None:
            _dependency_status[pkg_name] = "installed"
        else:
            # Auto-install llama-cpp-python with correct GPU support
            if pkg_name == "llama_cpp":
                detect_gpu_info()
                if install_llama_cpp_python():
                    _dependency_status[pkg_name] = "just_installed"
                else:
                    _dependency_status[pkg_name] = "failed"
            else:
                _dependency_status[pkg_name] = "not_installed"


# Run dependency check on module load
check_and_install_dependencies()

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Import all node classes
from .sid_zimage_prompt_generator import SID_ZImagePromptGenerator
from .sid_zimage_prompt_generator_advanced import SID_ZImagePromptGenerator_Advanced
from .sid_zimage_prompt_generator_advanced_v2 import SID_ZImagePromptGenerator_Advanced_V2

# Import LLM provider nodes
from .llm_providers.sid_anthropic_llm import SID_Anthropic_LLM
from .llm_providers.sid_openai_compatible_llm import SID_OpenAI_Compatible_LLM
from .llm_providers.sid_grok_llm import SID_Grok_LLM
from .llm_providers.sid_gguf_llm import SID_GGUF_LLM


class SIDPhotographyToolkitExtension(ComfyExtension):
    """
    Extension class for SID Photography Toolkit.
    Registers all SID_ prefixed nodes with ComfyUI.
    """

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """
        Return list of all nodes in this extension.
        Add new nodes here as they are created.
        """
        return [
            # Main nodes
            SID_ZImagePromptGenerator,
            SID_ZImagePromptGenerator_Advanced,
            SID_ZImagePromptGenerator_Advanced_V2,
            # LLM Provider nodes
            SID_Anthropic_LLM,
            SID_OpenAI_Compatible_LLM,
            SID_Grok_LLM,
            SID_GGUF_LLM,
        ]


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

    # llama-cpp-python status (now auto-installed)
    print("")
    llama_status = _dependency_status.get("llama_cpp", "not_installed")
    gpu_type = _gpu_info.get("type", "cpu")
    gpu_desc = {
        "nvidia": f"NVIDIA CUDA {_gpu_info.get('cuda_version', '')}",
        "apple_metal": "Apple Metal (M1/M2/M3)",
        "amd": "AMD ROCm",
        "cpu": "CPU",
    }.get(gpu_type, "CPU")

    if llama_status == "installed":
        print(f"    [OK] llama-cpp-python ({gpu_desc})")
    elif llama_status == "just_installed":
        print(f"    [INSTALLED] llama-cpp-python ({gpu_desc})")
    else:
        print(f"    [FAILED] llama-cpp-python - auto-install failed")
        print("         Please install manually - see README")

    # Available nodes
    print("")
    print("-" * 65)
    print("  Available Nodes:")
    print("")
    print("  Prompt Generators:")
    print("    - SID_ZImagePromptGenerator             (built-in Anthropic LLM)")
    print("    - SID_ZImagePromptGenerator_Advanced    (external LLM provider)")
    print("    - SID_ZImagePromptGenerator_Advanced_V2 (component-based analysis)")
    print("")
    print("  LLM Providers (connect to Advanced node):")
    print("    - SID_Anthropic_LLM        Anthropic Claude (cloud)")
    print("    - SID_OpenAI_Compatible_LLM OpenAI/GPT/Together AI/LM Studio")
    print("    - SID_Grok_LLM             xAI Grok (cloud)")
    if llama_status in ["installed", "just_installed"]:
        print(f"    - SID_GGUF_LLM             Local GGUF models ({gpu_desc})")
    else:
        print("    - SID_GGUF_LLM             [install failed - see README]")

    print("")
    print("  GGUF models location: ComfyUI/models/LLM/GGUF/")
    print("=" * 65)
    print("")


async def comfy_entrypoint() -> SIDPhotographyToolkitExtension:
    """
    ComfyUI calls this function to load the extension and its nodes.
    This is the entry point for the custom node package.
    """
    print_welcome_message()
    return SIDPhotographyToolkitExtension()
