"""
Tagging models for image analysis.

Dependencies are managed via requirements.txt and installed by ComfyUI-Manager.
"""

from .base import BaseTagger, TagItem, TaggerResult
from .wd14 import WD14Tagger, WD14_MODELS, DEFAULT_MODEL as WD14_DEFAULT_MODEL
from .pixai import PixAITagger
from .instant import ColorAnalyzer, BlurDetector, PhotographyTagger
from .joytag import JoyTagTagger
from .nudenet import NudeNetTagger, NudeNetONNXTagger
from .iqa import IQATagger
from .composition import CADBCompositionTagger, CompositionElementsTagger
from .saliency import SaliencyTagger
from .camera import ShotTypeClassifier, CLIPCameraTagger
from .lighting import IntrinsicLightingTagger, ShadowDetector
from .bbox_analyzer import BBoxAnalyzer
from .places365 import Places365Tagger
from .clip_scene import CLIPSceneTagger
from .wildlife import WildlifeTagger
from .landmarks import LandmarkTagger
from .sky_weather import SkyWeatherTagger

__all__ = [
    "BaseTagger",
    "TagItem",
    "TaggerResult",
    "WD14Tagger",
    "WD14_MODELS",
    "WD14_DEFAULT_MODEL",
    "PixAITagger",
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
    "ShotTypeClassifier",
    "CLIPCameraTagger",
    "IntrinsicLightingTagger",
    "ShadowDetector",
    "BBoxAnalyzer",
    "Places365Tagger",
    "CLIPSceneTagger",
    "WildlifeTagger",
    "LandmarkTagger",
    "SkyWeatherTagger",
]
