"""Fashion and clothing detection taggers."""

from .fashion_tagger import FashionTagger
from .yolos_fashionpedia import YOLOSFashionpediaTagger
from .yolov8_clothing import YOLOv8ClothingTagger
from .fashion_clip import FashionCLIPTagger
from .segformer_clothes import SegFormerClothesTagger
from .wargon_classifier import WargonClothingTagger
from .deepfashion_attributes import (
    DeepFashionAttributeTagger,
    FashionColorAnalyzer,
)

__all__ = [
    "FashionTagger",
    "YOLOSFashionpediaTagger",
    "YOLOv8ClothingTagger",
    "FashionCLIPTagger",
    "SegFormerClothesTagger",
    "WargonClothingTagger",
    "DeepFashionAttributeTagger",
    "FashionColorAnalyzer",
]
