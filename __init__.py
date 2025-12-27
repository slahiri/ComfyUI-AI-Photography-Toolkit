# -*- coding: utf-8 -*-
"""
SID Photography Toolkit v5.0

AI-powered image captioning for ComfyUI.
Clean, modular architecture with plug-and-play models.

Author: Siddhartha Lahiri
License: MIT
"""

__version__ = "5.0.0"

# Import nodes
from .nodes import (
    SID_ImageAnalysis,
    SID_PromptSynthesis,
    SID_PromptCompose,
)

# V1 API registration
NODE_CLASS_MAPPINGS = {
    "SID_ImageAnalysis": SID_ImageAnalysis,
    "SID_PromptSynthesis": SID_PromptSynthesis,
    "SID_PromptCompose": SID_PromptCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_ImageAnalysis": "SID Image Analysis",
    "SID_PromptSynthesis": "SID Prompt Synthesis",
    "SID_PromptCompose": "SID Prompt Compose",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Auto-install dependencies
def _install_dependencies():
    """Auto-install required dependencies at startup."""
    import subprocess
    import sys

    # Required packages: (import_name, pip_package)
    required = [
        ("yaml", "pyyaml"),
        ("onnxruntime", "onnxruntime-gpu"),
        ("cv2", "opencv-python"),
        ("timm", "timm"),
        ("huggingface_hub", "huggingface_hub"),
    ]

    # Optional packages
    optional = [
        ("nudenet", "nudenet"),
        ("pyiqa", "pyiqa"),
        ("ultralytics", "ultralytics"),
        ("imgutils", "dghs-imgutils>=0.19.0"),
    ]

    def is_installed(module_name):
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def install(package, quiet=True):
        print(f"[AI-Photo-Toolkit] Installing {package}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            if quiet:
                cmd.append("-q")
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Show output for debugging
                subprocess.check_call(cmd)
            return True
        except subprocess.CalledProcessError as e:
            # Try without GPU for onnxruntime
            if package == "onnxruntime-gpu":
                print(f"[AI-Photo-Toolkit] GPU version failed, trying CPU...")
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "onnxruntime", "-q"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except subprocess.CalledProcessError:
                    pass
            print(f"[AI-Photo-Toolkit] Failed to install {package}: {e}")
            return False
        except Exception as e:
            print(f"[AI-Photo-Toolkit] Error installing {package}: {e}")
            return False

    # Install required
    for module, package in required:
        if not is_installed(module):
            install(package)

    # Install optional (show output for debugging)
    for module, package in optional:
        if not is_installed(module):
            install(package, quiet=False)


# Startup
def _print_startup():
    print(f"\n[AI-Photo-Toolkit] v{__version__} loading...")

    # Auto-install missing deps
    _install_dependencies()

    # Check what's available now
    available = []
    missing = []

    checks = [
        ("onnxruntime", "WD14, JoyTag"),
        ("cv2", "Photography, Composition"),
        ("timm", "Fashion"),
        ("nudenet", "NudeNet"),
        ("pyiqa", "IQA"),
        ("ultralytics", "YOLOv8"),
        ("imgutils", "PixAI"),
    ]

    for module, taggers in checks:
        try:
            __import__(module)
            available.append(taggers)
        except ImportError:
            missing.append(taggers)

    print(f"[AI-Photo-Toolkit] v{__version__} ready!")
    print(f"[AI-Photo-Toolkit] Available: {', '.join(available)}")
    if missing:
        print(f"[AI-Photo-Toolkit] Unavailable: {', '.join(missing)}")
    print()

_print_startup()
