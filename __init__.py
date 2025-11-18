"""
ComfyUI-AI-Photography-Toolkit
A collection of AI-powered photography and image generation tools for ComfyUI.
All nodes are prefixed with SID_ for easy identification.

Author: Siddhartha Lahiri
Version: 1.3.4
"""

__version__ = "1.3.4"

import os
import sys
import subprocess
import importlib.util

# Auto-install dependencies
def install_dependencies():
    """
    Automatically install required dependencies if not already installed.
    """
    dependencies = {
        "anthropic": "anthropic>=0.39.0",
    }

    python_exe = sys.executable

    for package_name, package_spec in dependencies.items():
        # Check if package is already installed
        if importlib.util.find_spec(package_name) is None:
            print(f"\n{'='*60}")
            print(f"Installing {package_name} for SID Photography Toolkit...")
            print(f"{'='*60}")
            try:
                subprocess.check_call(
                    [python_exe, "-m", "pip", "install", package_spec],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"{package_name} installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to install {package_name}")
                print(f"Please manually install: pip install {package_spec}")
                print(f"Error: {e}")
            print(f"{'='*60}\n")

# Install dependencies on module load
install_dependencies()

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Import all node classes
from .sid_ai_prompt_generator import SID_AIPromptGenerator


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
            SID_AIPromptGenerator,
            # Future nodes will be added here
            # SID_ImageAnalyzer,
            # SID_StyleTransfer,
            # etc.
        ]


async def comfy_entrypoint() -> SIDPhotographyToolkitExtension:
    """
    ComfyUI calls this function to load the extension and its nodes.
    This is the entry point for the custom node package.
    """
    print("\n" + "="*60)
    print(f"Loading SID Photography Toolkit v{__version__} for ComfyUI")
    print("="*60)
    print("Nodes loaded:")
    print("  - SID_AIPromptGenerator: AI-powered prompt generation from images")
    print("="*60 + "\n")

    return SIDPhotographyToolkitExtension()
