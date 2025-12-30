"""
Tagger integration module.

Provides a unified interface for running image taggers and analyzers.
Implements all taggers and analyzers for comprehensive image analysis.
"""

from .runner import TaggerRunner, get_runner

# Tagger exports
from .wd14 import run_wd14, unload_model as unload_wd14
from .pixai import run_pixai, unload_model as unload_pixai
from .joytag import run_joytag, unload_model as unload_joytag
from .nudenet import run_nudenet, unload_detector as unload_nudenet
from .pose import run_pose, unload_model as unload_pose

# Analyzer exports
from .shot_type import run_shot_type, unload_model as unload_shot_type
from .places365 import run_places365, unload_model as unload_places365
from .photography import run_photography
from .iqa import run_iqa, unload_models as unload_iqa
from .composition import run_composition
from .saliency import run_saliency, unload_model as unload_saliency
from .clip_camera import run_clip_camera, unload_model as unload_clip_camera
from .lighting import run_lighting
from .fashion_color import run_fashion_color
from .deepfashion import run_deepfashion, unload_model as unload_deepfashion
from .wildlife import run_wildlife, unload_model as unload_wildlife
from .landmarks import run_landmarks, unload_model as unload_landmarks
from .sky_weather import run_sky_weather, unload_model as unload_sky_weather

__all__ = [
    "TaggerRunner",
    "get_runner",
    # Taggers
    "run_wd14", "unload_wd14",
    "run_pixai", "unload_pixai",
    "run_joytag", "unload_joytag",
    "run_nudenet", "unload_nudenet",
    "run_pose", "unload_pose",
    # Analyzers
    "run_shot_type", "unload_shot_type",
    "run_places365", "unload_places365",
    "run_photography",
    "run_iqa", "unload_iqa",
    "run_composition",
    "run_saliency", "unload_saliency",
    "run_clip_camera", "unload_clip_camera",
    "run_lighting",
    "run_fashion_color",
    "run_deepfashion", "unload_deepfashion",
    "run_wildlife", "unload_wildlife",
    "run_landmarks", "unload_landmarks",
    "run_sky_weather", "unload_sky_weather",
]
