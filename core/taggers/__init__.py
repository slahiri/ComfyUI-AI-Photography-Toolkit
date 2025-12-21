"""
Tagging models for image analysis.

Dependencies are automatically installed on first run.
"""

import subprocess
import sys

# =============================================================================
# Auto-Install Dependencies
# =============================================================================

__version__ = "5.0.0"
__author__ = "SID Photography Toolkit"

# Package dependencies: (import_name, pip_package, description)
DEPENDENCIES = [
    # Core (required)
    ("onnxruntime", "onnxruntime-gpu", "WD14 tagger"),
    ("torch", "torch", "Deep learning framework"),
    ("torchvision", "torchvision", "Vision models"),
    ("huggingface_hub", "huggingface_hub", "Model downloads"),
    ("safetensors", "safetensors", "Model loading"),
    # Taggers
    ("timm", "timm", "JoyTag (5000+ tags)"),
    ("nudenet", "nudenet", "NudeNet v2 (NSFW)"),
    ("pyiqa", "pyiqa", "IQA (NIMA, MUSIQ, BRISQUE)"),
    ("ultralytics", "ultralytics", "YOLOv8 clothing"),
    ("mediapipe", "mediapipe", "MediaPipe pose"),
    ("cv2", "opencv-python", "Composition elements"),
    ("scipy", "scipy", "Saliency analysis"),
]

# Packages that need version checks/upgrades
VERSION_DEPENDENCIES = [
    # transformers needs 4.46+ for Florence2 and ijepa support
    ("transformers", "transformers>=4.46.0", "4.46.0", "Florence2, ijepa, FashionCLIP"),
]

# Special packages that need git install
GIT_DEPENDENCIES = [
    ("ram", "git+https://github.com/xinyu1205/recognize-anything.git", "RAM++ (6400+ tags)"),
]


def _install_package(package: str, description: str) -> bool:
    """Install a package using pip."""
    try:
        is_git = package.startswith("git+")
        print(f"  [INSTALLING] {package} - {description}")
        if is_git:
            print(f"  [INFO] Git package may take a few minutes...")

        # Use verbose mode for git packages to help debug issues
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"  [OK] {description} installed")
            return True
        else:
            print(f"  [FAILED] {description}")
            if result.stderr:
                # Print last few lines of error
                error_lines = result.stderr.strip().split("\n")[-5:]
                for line in error_lines:
                    print(f"    {line}")
            return False
    except Exception as e:
        print(f"  [FAILED] {package}: {e}")
        return False


def _get_package_version(package_name: str) -> str | None:
    """Get installed package version."""
    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        return None


def _version_tuple(version_str: str) -> tuple:
    """Convert version string to tuple for comparison."""
    try:
        return tuple(int(x) for x in version_str.split(".")[:3])
    except Exception:
        return (0, 0, 0)


def _check_and_install_dependencies():
    """Check for missing dependencies and install them."""
    missing = []
    upgrades = []
    installed = []
    failed = []

    print("")
    print("=" * 80)
    print(f"  SID AI Photography Toolkit v{__version__}")
    print("=" * 80)
    print("")
    print("  Checking dependencies...")
    print("")

    # Check standard packages
    for import_name, pip_package, description in DEPENDENCIES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, pip_package, description))

    # Check version dependencies (upgrade if too old)
    for import_name, pip_package, min_version, description in VERSION_DEPENDENCIES:
        try:
            __import__(import_name)
            current_version = _get_package_version(import_name)
            if current_version:
                if _version_tuple(current_version) < _version_tuple(min_version):
                    print(f"  [UPGRADE] {import_name} {current_version} -> {min_version}+ (required for {description})")
                    upgrades.append((import_name, pip_package, description))
        except ImportError:
            missing.append((import_name, pip_package, description))

    # Check git packages
    for import_name, git_url, description in GIT_DEPENDENCIES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, git_url, description))

    # Install missing packages
    if missing:
        print(f"  Found {len(missing)} missing packages. Installing...")
        print("")

        for import_name, package, description in missing:
            if _install_package(package, description):
                installed.append(import_name)
            else:
                failed.append((import_name, package))

        print("")

    # Upgrade outdated packages
    if upgrades:
        print(f"  Upgrading {len(upgrades)} packages...")
        print("")

        for import_name, package, description in upgrades:
            if _install_package(package, f"UPGRADE: {description}"):
                installed.append(f"{import_name} (upgraded)")
            else:
                failed.append((import_name, package))

        print("")

    # Final status
    _print_status_banner(installed, failed)


def _print_status_banner(installed: list, failed: list):
    """Print startup banner with status."""
    print("  Nodes:")
    print("  -------")
    print("    - SID_TagConfigurator      Configure all taggers and thresholds")
    print("    - SID_PromptGenerator      Generate prompts from tags")
    print("")
    print("  Taggers:")
    print("  ---------")
    print("    General:     WD14 | RAM++ (6400+) | JoyTag (5000+)")
    print("    NSFW:        NudeNet v2 (16 classes)")
    print("    Quality:     NIMA | MUSIQ | BRISQUE | CLIPIQA")
    print("    Composition: CADB (13 types) | Saliency | Elements")
    print("    Fashion:     YOLOv8 | YOLOS | FashionCLIP | SegFormer | Wargon")
    print("    Pose:        MediaPipe | MotionBERT | Sapiens")
    print("    Photo:       Color | Contrast | Sharpness | Lighting | Framing")
    print("")

    # Check current status of all dependencies
    available = []
    missing = []

    checks = [
        ("onnxruntime", "WD14"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("ram", "RAM++"),
        ("timm", "JoyTag"),
        ("nudenet", "NudeNet"),
        ("pyiqa", "IQA"),
        ("ultralytics", "YOLOv8"),
        ("mediapipe", "MediaPipe"),
        ("cv2", "OpenCV"),
    ]

    for module, name in checks:
        try:
            __import__(module)
            available.append(name)
        except ImportError:
            missing.append(name)

    print("  Dependencies:")
    print("  --------------")
    if available:
        print(f"    [OK] {', '.join(available)}")
    if missing:
        print(f"    [!!] Missing: {', '.join(missing)}")
    if installed:
        print(f"    [NEW] Just installed: {', '.join(installed)}")
    if failed:
        print(f"    [FAILED] Could not install: {', '.join([f[0] for f in failed])}")
    print("")
    print("  Output Format: JSON { tagger: { tags: [...], count, metadata } }")
    print("")
    print("=" * 80)
    print("")


# Run dependency check on import
_check_and_install_dependencies()

# =============================================================================
# Imports
# =============================================================================

from .base import BaseTagger, TagItem, TaggerResult
from .wd14 import WD14Tagger, WD14_MODELS, DEFAULT_MODEL as WD14_DEFAULT_MODEL
from .instant import ColorAnalyzer, BlurDetector, PhotographyTagger
from .ram_plus import RAMPlusTagger, RAMPlusTaggerHF
from .joytag import JoyTagTagger
from .nudenet import NudeNetTagger, NudeNetONNXTagger
from .iqa import IQATagger
from .composition import CADBCompositionTagger, CompositionElementsTagger
from .saliency import SaliencyTagger

# Lazy imports for fashion and pose to avoid loading heavy dependencies
# Use: from .taggers.fashion import FashionTagger
# Use: from .taggers.pose import PoseTagger

__all__ = [
    "BaseTagger",
    "TagItem",
    "TaggerResult",
    "WD14Tagger",
    "WD14_MODELS",
    "WD14_DEFAULT_MODEL",
    "RAMPlusTagger",
    "RAMPlusTaggerHF",
    "JoyTagTagger",
    "NudeNetTagger",
    "NudeNetONNXTagger",
    "IQATagger",
    "CADBCompositionTagger",
    "CompositionElementsTagger",
    "SaliencyTagger",
    "ColorAnalyzer",
    "BlurDetector",
    "PhotographyTagger",
]
