"""
Tagging models for image analysis.

Dependencies are managed via requirements.txt and installed by ComfyUI-Manager.
"""

from .base import BaseTagger, TagItem, TaggerResult
from .wd14 import WD14Tagger, WD14_MODELS, DEFAULT_MODEL as WD14_DEFAULT_MODEL
from .instant import ColorAnalyzer, BlurDetector, PhotographyTagger
from .joytag import JoyTagTagger
from .nudenet import NudeNetTagger, NudeNetONNXTagger
from .iqa import IQATagger
from .composition import CADBCompositionTagger, CompositionElementsTagger
from .saliency import SaliencyTagger

__all__ = [
    "BaseTagger",
    "TagItem",
    "TaggerResult",
    "WD14Tagger",
    "WD14_MODELS",
    "WD14_DEFAULT_MODEL",
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
