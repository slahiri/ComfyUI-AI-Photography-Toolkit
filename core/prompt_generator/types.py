"""
Core types for SID_PromptGenerator.

Defines enums and dataclasses used throughout the prompt generation system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class AnalysisMode(Enum):
    """Analysis depth modes for prompt generation."""
    QUICK = "Quick"
    STANDARD = "Standard"
    DETAILED = "Detailed"
    EXTREME = "Extreme"


class SubjectType(Enum):
    """Primary subject type detected in image."""
    WOMAN = "woman"
    MAN = "man"
    COUPLE = "couple"
    GROUP = "group"
    LANDSCAPE = "landscape"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    PRODUCT = "product"
    UNKNOWN = "unknown"


class NSFWLevel(Enum):
    """NSFW content level detected."""
    SAFE = "safe"
    SUGGESTIVE = "suggestive"
    EXPLICIT = "explicit"


class ShotType(Enum):
    """Camera shot type / framing."""
    ECU = "extreme close-up"      # Face detail only
    CU = "close-up"               # Head and shoulders
    MCU = "medium close-up"       # Chest up
    MS = "medium shot"            # Waist up
    MFS = "medium full shot"      # Thighs up (cowboy shot)
    FS = "full shot"              # Full body
    MLS = "medium long shot"      # Full body + some environment
    LS = "long shot"              # Full body, environment prominent
    ELS = "extreme long shot"     # Environment dominant
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "ShotType":
        """Convert string to ShotType enum."""
        mapping = {
            "ecu": cls.ECU,
            "cu": cls.CU,
            "mcu": cls.MCU,
            "ms": cls.MS,
            "mfs": cls.MFS,
            "fs": cls.FS,
            "mls": cls.MLS,
            "ls": cls.LS,
            "els": cls.ELS,
        }
        return mapping.get(value.lower(), cls.UNKNOWN)


class SceneType(Enum):
    """Scene/environment type."""
    STUDIO = "studio"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NATURE = "nature"
    URBAN = "urban"
    UNKNOWN = "unknown"


@dataclass
class Decisions:
    """
    Tag-based decisions that drive template selection and prompt composition.

    These are derived from tagger/analyzer outputs and determine which
    template sections to activate.
    """
    # Core decisions
    subject_type: SubjectType = SubjectType.UNKNOWN
    gender: Optional[str] = None
    nsfw_level: NSFWLevel = NSFWLevel.SAFE
    shot_type: ShotType = ShotType.UNKNOWN
    scene_type: Optional[str] = None
    quality_level: Optional[str] = None
    lighting_mood: Optional[str] = None
    template_sections: List[str] = field(default_factory=list)

    # Extended decisions from analyzers
    wildlife_detected: bool = False
    wildlife_type: Optional[str] = None
    landmark_detected: bool = False
    landmark_name: Optional[str] = None
    landmark_famous: bool = False
    sky_visible: bool = False
    weather_condition: Optional[str] = None
    time_of_day: Optional[str] = None
    is_golden_hour: bool = False
    is_dramatic_lighting: bool = False
    camera_angle: Optional[str] = None
    depth_of_field: Optional[str] = None
    composition_style: Optional[str] = None
    fashion_detected: bool = False
    has_strong_shadows: bool = False
    lighting_direction: Optional[str] = None
    lighting_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subject_type": self.subject_type.value,
            "gender": self.gender,
            "nsfw_level": self.nsfw_level.value,
            "shot_type": self.shot_type.value,
            "scene_type": self.scene_type,
            "quality_level": self.quality_level,
            "lighting_mood": self.lighting_mood,
            "template_sections": self.template_sections,
            # Extended
            "wildlife_detected": self.wildlife_detected,
            "wildlife_type": self.wildlife_type,
            "landmark_detected": self.landmark_detected,
            "landmark_name": self.landmark_name,
            "landmark_famous": self.landmark_famous,
            "sky_visible": self.sky_visible,
            "weather_condition": self.weather_condition,
            "time_of_day": self.time_of_day,
            "is_golden_hour": self.is_golden_hour,
            "is_dramatic_lighting": self.is_dramatic_lighting,
            "camera_angle": self.camera_angle,
            "depth_of_field": self.depth_of_field,
            "composition_style": self.composition_style,
            "fashion_detected": self.fashion_detected,
            "has_strong_shadows": self.has_strong_shadows,
            "lighting_direction": self.lighting_direction,
            "lighting_type": self.lighting_type,
        }


@dataclass
class GeneratorResult:
    """
    Result from a prompt generation mode.

    Contains the generated prompt and metadata about the generation process.
    """
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_output(self) -> tuple:
        """Convert to ComfyUI output tuple (prompt, metadata_json)."""
        import json
        return (self.prompt, json.dumps(self.metadata, indent=2))


@dataclass
class TaggerResults:
    """
    Combined results from all taggers and analyzers.

    Stores raw outputs from each tagger for decision-making.
    """
    # Taggers
    wd14: Dict[str, float] = field(default_factory=dict)
    pixai: Dict[str, float] = field(default_factory=dict)
    joytag: Dict[str, float] = field(default_factory=dict)
    nudenet: Dict[str, Any] = field(default_factory=dict)
    pose: Dict[str, Any] = field(default_factory=dict)

    # Analyzers
    photography: Dict[str, float] = field(default_factory=dict)
    iqa: Dict[str, float] = field(default_factory=dict)
    composition: Dict[str, Any] = field(default_factory=dict)
    shot_type: Dict[str, float] = field(default_factory=dict)
    places365: Dict[str, float] = field(default_factory=dict)
    clip_camera: Dict[str, Any] = field(default_factory=dict)
    saliency: Dict[str, Any] = field(default_factory=dict)
    lighting: Dict[str, Any] = field(default_factory=dict)
    fashion_color: List[Dict[str, Any]] = field(default_factory=list)
    deepfashion: Dict[str, Any] = field(default_factory=dict)
    wildlife: Dict[str, Any] = field(default_factory=dict)
    landmarks: Dict[str, Any] = field(default_factory=dict)
    sky_weather: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "wd14": self.wd14,
            "pixai": self.pixai,
            "joytag": self.joytag,
            "nudenet": self.nudenet,
            "pose": self.pose,
            "photography": self.photography,
            "iqa": self.iqa,
            "composition": self.composition,
            "shot_type": self.shot_type,
            "places365": self.places365,
            "clip_camera": self.clip_camera,
            "saliency": self.saliency,
            "lighting": self.lighting,
            "fashion_color": self.fashion_color,
            "deepfashion": self.deepfashion,
            "wildlife": self.wildlife,
            "landmarks": self.landmarks,
            "sky_weather": self.sky_weather,
        }

    def get_top_tags(self, source: str, threshold: float = 0.5, limit: int = 20) -> Dict[str, float]:
        """Get top tags from a specific tagger above threshold."""
        source_data = getattr(self, source, {})
        if not isinstance(source_data, dict):
            return {}
        filtered = {k: v for k, v in source_data.items() if v >= threshold}
        sorted_tags = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_tags[:limit])
