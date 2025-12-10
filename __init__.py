"""
ComfyUI-AI-Photography-Toolkit
LLM Provider nodes for ComfyUI - Cloud and Local model support.
All nodes are prefixed with SID_ for easy identification.

Author: Siddhartha Lahiri
Version: 4.2.0
"""

__version__ = "4.2.0"

import sys
import subprocess
import importlib.util

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

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

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
    print("  Prompt Generator:")
    print("    - SID_ZImagePromptGenerator  Z-Image prompts (auto Single-Shot/Agentic)")

    print("")
    print("=" * 65)
    print("")


async def comfy_entrypoint() -> SIDPhotographyToolkitExtension:
    """
    ComfyUI calls this function to load the extension and its nodes.
    This is the entry point for the custom node package.
    """
    print_welcome_message()
    return SIDPhotographyToolkitExtension()
