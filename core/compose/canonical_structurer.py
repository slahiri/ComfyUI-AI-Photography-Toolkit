"""Canonical Structurer - Scene Detection and Category Classification.

This module:
1. Detects scene types from tagger outputs (1 primary + optional secondary)
2. Classifies all tags into 6 canonical prompt categories
3. Generates structured JSON with natural language phrases
4. Produces high-resolution text for each category

Based on Z-Image canonical prompt structure:
Subject + Scene + Composition + Lighting + Style + Constraints
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import re


# =============================================================================
# SCENE TYPES
# =============================================================================

class SceneType(Enum):
    """Primary scene type categories."""
    PORTRAIT = "portrait"
    FASHION = "fashion"
    ACTION = "action"
    GROUP = "group"
    PRODUCT = "product"
    FOOD = "food"
    STILL_LIFE = "still_life"
    LANDSCAPE = "landscape"
    ARCHITECTURE = "architecture"
    STREET = "street"
    CINEMATIC = "cinematic"
    EDITORIAL = "editorial"
    ANIME = "anime"
    VINTAGE = "vintage"
    CYBERPUNK = "cyberpunk"
    TEXT_OVERLAY = "text_overlay"


class CanonicalCategory(Enum):
    """The 6 canonical prompt categories."""
    SUBJECT = "subject"
    SCENE = "scene"
    COMPOSITION = "composition"
    LIGHTING = "lighting"
    STYLE = "style"
    CONSTRAINTS = "constraints"


# =============================================================================
# SCENE-BASED CATEGORY RELEVANCE
# =============================================================================
# Maps each scene type to which canonical categories are relevant
# Categories not listed are excluded from output for that scene type

SCENE_CATEGORY_RELEVANCE: Dict[SceneType, Set[CanonicalCategory]] = {
    # PORTRAIT - focus on subject, lighting, composition, style
    SceneType.PORTRAIT: {
        CanonicalCategory.SUBJECT,      # Person details, expression, hair
        CanonicalCategory.COMPOSITION,  # Shot type, angle, framing
        CanonicalCategory.LIGHTING,     # Portrait lighting is crucial
        CanonicalCategory.STYLE,        # Aesthetic, mood
        CanonicalCategory.CONSTRAINTS,  # Quality
    },
    # FASHION - full body, outfit emphasis
    SceneType.FASHION: {
        CanonicalCategory.SUBJECT,      # Person + clothing details
        CanonicalCategory.SCENE,        # Background/studio
        CanonicalCategory.COMPOSITION,  # Full body framing
        CanonicalCategory.LIGHTING,     # Fashion lighting
        CanonicalCategory.STYLE,        # Fashion aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # ACTION - dynamic motion shots
    SceneType.ACTION: {
        CanonicalCategory.SUBJECT,      # Who/what is in action
        CanonicalCategory.SCENE,        # Environment
        CanonicalCategory.COMPOSITION,  # Dynamic angles
        CanonicalCategory.STYLE,        # Action aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # GROUP - multiple people
    SceneType.GROUP: {
        CanonicalCategory.SUBJECT,      # People details
        CanonicalCategory.SCENE,        # Setting
        CanonicalCategory.COMPOSITION,  # Group arrangement
        CanonicalCategory.LIGHTING,
        CanonicalCategory.STYLE,
        CanonicalCategory.CONSTRAINTS,
    },
    # PRODUCT - commercial product photography
    SceneType.PRODUCT: {
        CanonicalCategory.SUBJECT,      # Product details
        CanonicalCategory.SCENE,        # Clean background
        CanonicalCategory.COMPOSITION,  # Product framing
        CanonicalCategory.LIGHTING,     # Product lighting crucial
        CanonicalCategory.STYLE,        # Commercial style
        CanonicalCategory.CONSTRAINTS,
    },
    # FOOD - food photography
    SceneType.FOOD: {
        CanonicalCategory.SUBJECT,      # Food items
        CanonicalCategory.SCENE,        # Table setting, props
        CanonicalCategory.COMPOSITION,  # Food arrangement
        CanonicalCategory.LIGHTING,     # Food lighting
        CanonicalCategory.STYLE,        # Food photography style
        CanonicalCategory.CONSTRAINTS,
    },
    # STILL_LIFE - artistic object arrangements
    SceneType.STILL_LIFE: {
        CanonicalCategory.SUBJECT,      # Objects
        CanonicalCategory.SCENE,        # Background
        CanonicalCategory.COMPOSITION,  # Arrangement
        CanonicalCategory.LIGHTING,     # Artistic lighting
        CanonicalCategory.STYLE,        # Artistic style
        CanonicalCategory.CONSTRAINTS,
    },
    # LANDSCAPE - wide environmental shots (NO subject/fashion)
    SceneType.LANDSCAPE: {
        CanonicalCategory.SCENE,        # Environment is the focus
        CanonicalCategory.COMPOSITION,  # Landscape framing
        CanonicalCategory.LIGHTING,     # Natural lighting
        CanonicalCategory.STYLE,        # Landscape aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # ARCHITECTURE - buildings, interiors
    SceneType.ARCHITECTURE: {
        CanonicalCategory.SUBJECT,      # Building/structure details
        CanonicalCategory.SCENE,        # Location context
        CanonicalCategory.COMPOSITION,  # Architectural framing
        CanonicalCategory.LIGHTING,     # Architectural lighting
        CanonicalCategory.STYLE,        # Architectural style
        CanonicalCategory.CONSTRAINTS,
    },
    # STREET - urban photography
    SceneType.STREET: {
        CanonicalCategory.SUBJECT,      # People/objects in scene
        CanonicalCategory.SCENE,        # Urban environment
        CanonicalCategory.COMPOSITION,  # Street framing
        CanonicalCategory.LIGHTING,     # Natural/urban lighting
        CanonicalCategory.STYLE,        # Documentary style
        CanonicalCategory.CONSTRAINTS,
    },
    # CINEMATIC - film-style compositions
    SceneType.CINEMATIC: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,
        CanonicalCategory.COMPOSITION,  # Cinematic framing crucial
        CanonicalCategory.LIGHTING,     # Dramatic lighting
        CanonicalCategory.STYLE,        # Film aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # EDITORIAL - magazine/fashion editorial
    SceneType.EDITORIAL: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.LIGHTING,
        CanonicalCategory.STYLE,
        CanonicalCategory.CONSTRAINTS,
    },
    # ANIME - anime/illustration styles (different subject handling)
    SceneType.ANIME: {
        CanonicalCategory.SUBJECT,      # Character details
        CanonicalCategory.SCENE,        # Background
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.STYLE,        # Anime style crucial
        CanonicalCategory.CONSTRAINTS,
    },
    # VINTAGE - retro/film aesthetics
    SceneType.VINTAGE: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.LIGHTING,
        CanonicalCategory.STYLE,        # Vintage style crucial
        CanonicalCategory.CONSTRAINTS,
    },
    # CYBERPUNK - sci-fi/futuristic
    SceneType.CYBERPUNK: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,        # Futuristic environment
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.LIGHTING,     # Neon lighting
        CanonicalCategory.STYLE,        # Cyberpunk aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # TEXT_OVERLAY - images with text elements
    SceneType.TEXT_OVERLAY: {
        CanonicalCategory.SUBJECT,      # Text content
        CanonicalCategory.SCENE,        # Background
        CanonicalCategory.COMPOSITION,  # Text placement
        CanonicalCategory.STYLE,
        CanonicalCategory.CONSTRAINTS,
    },
}

# Default: all categories relevant
DEFAULT_RELEVANT_CATEGORIES = set(CanonicalCategory)


# =============================================================================
# SCENE DETECTION KEYWORDS
# =============================================================================

SCENE_KEYWORDS: Dict[SceneType, Dict[str, float]] = {
    SceneType.PORTRAIT: {
        # High confidence
        "portrait": 1.0, "headshot": 1.0, "close-up": 0.8, "face": 0.7,
        "1girl": 0.6, "1boy": 0.6, "solo": 0.5, "looking at viewer": 0.6,
        "upper body": 0.7, "head": 0.5, "selfie": 0.8,
        # Medium confidence
        "eyes": 0.4, "lips": 0.4, "smile": 0.3, "expression": 0.4,
    },
    SceneType.FASHION: {
        "fashion": 1.0, "full body": 0.8, "outfit": 0.9, "model": 0.7,
        "dress": 0.6, "clothing": 0.5, "pose": 0.5, "standing": 0.4,
        "elegant": 0.5, "stylish": 0.6, "haute couture": 1.0,
        "runway": 0.9, "designer": 0.7, "vogue": 0.8,
    },
    SceneType.ACTION: {
        "action": 1.0, "dynamic": 0.8, "motion": 0.9, "running": 0.8,
        "jumping": 0.8, "fighting": 0.7, "sports": 0.7, "movement": 0.7,
        "athletic": 0.6, "exercise": 0.6, "weightlifting": 0.7,
        "dancing": 0.7, "martial arts": 0.8,
    },
    SceneType.GROUP: {
        "group": 1.0, "multiple": 0.8, "crowd": 0.9, "people": 0.6,
        "2girls": 0.7, "2boys": 0.7, "3+": 0.8, "team": 0.7,
        "audience": 0.7, "6+boys": 0.9, "6+girls": 0.9,
        "multiple boys": 0.8, "multiple girls": 0.8,
    },
    SceneType.PRODUCT: {
        "product": 1.0, "commercial": 0.8, "advertisement": 0.8,
        "white background": 0.7, "studio": 0.5, "isolated": 0.6,
        "object": 0.4, "item": 0.4, "packshot": 1.0,
    },
    SceneType.FOOD: {
        "food": 1.0, "dish": 0.9, "meal": 0.9, "cuisine": 0.8,
        "plate": 0.6, "restaurant": 0.7, "cooking": 0.7, "delicious": 0.6,
        "appetizing": 0.7, "gourmet": 0.8, "recipe": 0.7,
    },
    SceneType.STILL_LIFE: {
        "still life": 1.0, "arrangement": 0.8, "objects": 0.5,
        "table": 0.4, "flowers": 0.5, "vase": 0.6, "fruit": 0.5,
        "composition": 0.3, "artistic": 0.4,
    },
    SceneType.LANDSCAPE: {
        "landscape": 1.0, "scenery": 0.9, "nature": 0.7, "outdoor": 0.6,
        "mountain": 0.8, "sky": 0.5, "horizon": 0.7, "vista": 0.9,
        "panorama": 0.9, "sunset": 0.6, "sunrise": 0.6, "beach": 0.6,
        "forest": 0.7, "ocean": 0.7, "field": 0.6,
    },
    SceneType.ARCHITECTURE: {
        "architecture": 1.0, "building": 0.8, "interior": 0.8,
        "exterior": 0.7, "room": 0.6, "house": 0.5, "structure": 0.6,
        "cathedral": 0.9, "skyscraper": 0.8, "modern architecture": 1.0,
    },
    SceneType.STREET: {
        "street": 0.9, "urban": 0.8, "city": 0.7, "downtown": 0.8,
        "sidewalk": 0.7, "alley": 0.7, "crosswalk": 0.7,
        "street photography": 1.0, "candid": 0.6, "documentary": 0.6,
    },
    SceneType.CINEMATIC: {
        "cinematic": 1.0, "film": 0.7, "movie": 0.7, "dramatic": 0.7,
        "noir": 0.9, "moody": 0.6, "atmospheric": 0.6,
        "anamorphic": 0.9, "widescreen": 0.7, "theatrical": 0.8,
    },
    SceneType.EDITORIAL: {
        "editorial": 1.0, "magazine": 0.9, "vogue": 0.9, "fashion editorial": 1.0,
        "high fashion": 0.9, "glamour": 0.7, "beauty shot": 0.8,
        "cover": 0.6, "spread": 0.7,
    },
    SceneType.ANIME: {
        "anime": 1.0, "manga": 0.9, "illustration": 0.7, "2d": 0.6,
        "cel-shaded": 0.9, "cartoon": 0.6, "animated": 0.7,
        "japanese animation": 1.0, "studio ghibli": 1.0, "pixiv": 0.8,
    },
    SceneType.VINTAGE: {
        "vintage": 1.0, "retro": 0.9, "old": 0.5, "classic": 0.6,
        "film grain": 0.8, "polaroid": 0.9, "analog": 0.8,
        "nostalgic": 0.7, "1970s": 0.9, "1980s": 0.9, "1950s": 0.9,
    },
    SceneType.CYBERPUNK: {
        "cyberpunk": 1.0, "futuristic": 0.8, "sci-fi": 0.7, "neon": 0.7,
        "dystopian": 0.9, "blade runner": 1.0, "holographic": 0.8,
        "tech": 0.5, "cyber": 0.9, "synthwave": 0.8,
    },
    SceneType.TEXT_OVERLAY: {
        "text": 0.7, "typography": 1.0, "sign": 0.6, "logo": 0.6,
        "lettering": 0.9, "words": 0.6, "neon sign": 0.8,
        "graffiti": 0.7, "poster": 0.7, "title": 0.6,
    },
}


# =============================================================================
# CANONICAL CATEGORY KEYWORDS
# =============================================================================

CATEGORY_KEYWORDS: Dict[CanonicalCategory, Dict[str, Set[str]]] = {
    CanonicalCategory.SUBJECT: {
        "person_type": {
            "1girl", "1boy", "woman", "man", "girl", "boy", "child", "elderly",
            "model", "person", "figure", "character", "subject", "solo",
            "couple", "group", "crowd", "people", "multiple",
        },
        "body_parts": {
            "face", "eyes", "lips", "hair", "nose", "ears", "hands", "arms",
            "legs", "body", "torso", "head", "shoulders", "neck", "chest",
            "back", "feet", "fingers", "nails", "skin",
        },
        "attributes": {
            "young", "old", "tall", "short", "thin", "muscular", "athletic",
            "beautiful", "handsome", "elegant", "stylish", "casual",
        },
        "hair": {
            "blonde", "brunette", "redhead", "black hair", "brown hair",
            "long hair", "short hair", "curly", "straight", "wavy",
            "ponytail", "braid", "bun", "bangs",
        },
        "expression": {
            "smile", "smiling", "laughing", "serious", "neutral", "happy",
            "sad", "angry", "surprised", "contemplative", "pensive",
        },
        "clothing": {
            "dress", "shirt", "pants", "skirt", "jacket", "coat", "suit",
            "jeans", "shorts", "blouse", "sweater", "hoodie", "t-shirt",
            "shoes", "boots", "sneakers", "heels", "accessories",
            "jewelry", "hat", "glasses", "watch", "bag",
            "sports bra", "sportswear", "athletic wear",
        },
        "pose": {
            "standing", "sitting", "lying", "walking", "running", "jumping",
            "dancing", "posing", "leaning", "crouching", "kneeling",
            "arms up", "arms crossed", "hands on hips",
        },
    },
    CanonicalCategory.SCENE: {
        "location_indoor": {
            "studio", "room", "interior", "indoor", "indoors", "inside",
            "bedroom", "living room", "kitchen", "bathroom", "office",
            "gym", "restaurant", "cafe", "bar", "club", "museum",
            "gallery", "theater", "stadium", "arena",
        },
        "location_outdoor": {
            "outdoor", "outdoors", "outside", "street", "park", "garden",
            "beach", "mountain", "forest", "field", "meadow", "desert",
            "ocean", "lake", "river", "city", "urban", "rural",
            "countryside", "rooftop", "balcony", "terrace",
        },
        "time": {
            "day", "night", "morning", "evening", "afternoon", "dusk",
            "dawn", "sunset", "sunrise", "golden hour", "blue hour",
            "midnight", "noon", "twilight",
        },
        "weather": {
            "sunny", "cloudy", "rainy", "snowy", "foggy", "misty",
            "stormy", "windy", "overcast", "clear", "hazy",
        },
        "atmosphere": {
            "peaceful", "calm", "serene", "energetic", "vibrant",
            "mysterious", "romantic", "dramatic", "tense", "relaxed",
            "cozy", "warm", "cold", "eerie", "magical",
        },
        "background": {
            "background", "backdrop", "scenery", "environment", "setting",
            "bokeh", "blurred background", "out of focus",
        },
    },
    CanonicalCategory.COMPOSITION: {
        "shot_type": {
            "close-up", "extreme close-up", "medium shot", "medium close-up",
            "full shot", "full body", "wide shot", "long shot",
            "extreme long shot", "establishing shot", "detail shot",
            "headshot", "portrait", "half body", "upper body",
        },
        "angle": {
            "eye level", "low angle", "high angle", "bird's eye",
            "worm's eye", "dutch angle", "tilted", "straight on",
            "front view", "side view", "back view", "3/4 view",
            "profile", "from above", "from below",
        },
        "framing": {
            "centered", "rule of thirds", "golden ratio", "symmetrical",
            "asymmetrical", "balanced", "off-center", "frame within frame",
            "leading lines", "diagonal", "horizontal", "vertical",
        },
        "focus": {
            "shallow depth of field", "deep focus", "selective focus",
            "bokeh", "sharp", "soft focus", "tack sharp", "in focus",
            "out of focus", "blurred",
        },
        "lens": {
            "wide angle", "telephoto", "macro", "fisheye", "tilt-shift",
            "35mm", "50mm", "85mm", "135mm", "24mm", "70-200mm",
            "prime lens", "zoom lens", "anamorphic",
        },
        "aspect": {
            "portrait orientation", "landscape orientation", "square",
            "16:9", "4:3", "2.39:1", "cinemascope", "widescreen",
            "vertical", "horizontal",
        },
    },
    CanonicalCategory.LIGHTING: {
        "direction": {
            "front lighting", "side lighting", "back lighting", "rim lighting",
            "top lighting", "under lighting", "ambient", "directional",
            "light from left", "light from right", "light from above",
        },
        "quality": {
            "soft light", "hard light", "diffused", "harsh", "gentle",
            "even lighting", "dramatic lighting", "flat lighting",
            "high contrast", "low contrast", "soft shadows", "hard shadows",
        },
        "type": {
            "natural light", "artificial light", "studio lighting",
            "window light", "sunlight", "moonlight", "candlelight",
            "neon", "fluorescent", "led", "flash", "strobe",
        },
        "color": {
            "warm light", "cool light", "golden", "blue", "orange",
            "white balance", "color temperature", "tinted",
        },
        "style": {
            "rembrandt", "butterfly", "split", "loop", "broad", "short",
            "paramount", "clamshell", "beauty lighting",
            "chiaroscuro", "low key", "high key",
        },
        "effects": {
            "volumetric", "god rays", "lens flare", "light leak",
            "specular", "highlights", "catchlights", "rim highlights",
            "glow", "halo", "backlit",
        },
    },
    CanonicalCategory.STYLE: {
        "medium": {
            "photograph", "photography", "photo", "digital", "film",
            "painting", "illustration", "drawing", "sketch", "render",
            "3d", "cgi", "artwork", "art",
        },
        "aesthetic": {
            "realistic", "photorealistic", "hyperrealistic", "stylized",
            "artistic", "cinematic", "editorial", "commercial",
            "fine art", "documentary", "candid", "glamour",
            "minimalist", "maximalist", "abstract",
        },
        "genre": {
            "portrait photography", "fashion photography", "street photography",
            "landscape photography", "product photography", "food photography",
            "architectural photography", "sports photography",
            "wildlife photography", "macro photography",
        },
        "film_reference": {
            "kodak", "fujifilm", "ilford", "portra", "ektar", "tri-x",
            "cinestill", "polaroid", "vintage film", "film grain",
        },
        "art_reference": {
            "renaissance", "baroque", "impressionist", "modern",
            "contemporary", "pop art", "surreal", "fantasy",
            "anime", "manga", "studio ghibli", "pixar",
        },
        "color_grade": {
            "vibrant", "muted", "desaturated", "saturated",
            "warm tones", "cool tones", "neutral tones",
            "teal and orange", "monochrome", "black and white",
            "sepia", "cross-processed",
        },
    },
    CanonicalCategory.CONSTRAINTS: {
        "quality": {
            "high quality", "8k", "4k", "hd", "ultra hd", "detailed",
            "highly detailed", "sharp", "crisp", "clean", "professional",
            "masterpiece", "best quality",
        },
        "technical": {
            "raw", "uncompressed", "high resolution", "noise-free",
            "sharp focus", "perfect focus", "accurate colors",
        },
        "artifacts": {
            "no watermark", "no logo", "no text", "no blur",
            "no noise", "no grain", "no distortion",
        },
        "safety": {
            "safe for work", "sfw", "appropriate", "clean",
            "family friendly", "non-offensive",
        },
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SceneDetectionResult:
    """Result of scene type detection."""
    primary: SceneType
    primary_confidence: float
    secondary: Optional[SceneType] = None
    secondary_confidence: float = 0.0
    all_scores: Dict[SceneType, float] = field(default_factory=dict)


@dataclass
class CategoryItem:
    """A single item classified into a category."""
    text: str
    confidence: float
    source: str  # Which tagger it came from
    subcategory: str  # e.g., "hair", "lighting_direction"


@dataclass
class CanonicalStructure:
    """The full canonical structure with all categories."""
    scene_type: SceneDetectionResult
    subject: List[CategoryItem] = field(default_factory=list)
    scene: List[CategoryItem] = field(default_factory=list)
    composition: List[CategoryItem] = field(default_factory=list)
    lighting: List[CategoryItem] = field(default_factory=list)
    style: List[CategoryItem] = field(default_factory=list)
    constraints: List[CategoryItem] = field(default_factory=list)

    # Uncategorized tags (above threshold, not matched to any category)
    uncategorized: Dict[str, float] = field(default_factory=dict)

    # Below threshold tags (filtered out by tag_filter)
    below_threshold: Dict[str, float] = field(default_factory=dict)

    # Generated text for each category
    subject_text: str = ""
    scene_text: str = ""
    composition_text: str = ""
    lighting_text: str = ""
    style_text: str = ""
    constraints_text: str = ""

    # Full combined prompt
    full_prompt: str = ""


# =============================================================================
# SCENE TYPE DETECTOR
# =============================================================================

class SceneTypeDetector:
    """Detects scene type from tagger outputs."""

    def __init__(self):
        self.keywords = SCENE_KEYWORDS

    def detect(
        self,
        tags: Dict[str, float],
        florence_caption: str = "",
        florence_description: str = "",
    ) -> SceneDetectionResult:
        """
        Detect scene type from tags and Florence captions.

        Args:
            tags: Dict of tag -> confidence from all taggers
            florence_caption: Florence detailed caption
            florence_description: Florence more detailed description

        Returns:
            SceneDetectionResult with primary and optional secondary scene type
        """
        scores: Dict[SceneType, float] = {st: 0.0 for st in SceneType}

        # Normalize tags to lowercase
        tags_lower = {k.lower(): v for k, v in tags.items()}

        # Check for wildlife/nature override indicators
        has_no_humans = tags_lower.get("no humans", 0) > 0.5
        has_animal = tags_lower.get("animal", 0) > 0.5 or tags_lower.get("animal focus", 0) > 0.5

        # Score based on tag presence
        for scene_type, keywords in self.keywords.items():
            for keyword, weight in keywords.items():
                keyword_lower = keyword.lower()
                # Exact match
                if keyword_lower in tags_lower:
                    scores[scene_type] += tags_lower[keyword_lower] * weight
                # Partial match (keyword in tag)
                else:
                    for tag, conf in tags_lower.items():
                        if keyword_lower in tag:
                            scores[scene_type] += conf * weight * 0.5

        # Boost from Florence captions
        combined_caption = f"{florence_caption} {florence_description}".lower()
        for scene_type, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword.lower() in combined_caption:
                    scores[scene_type] += 0.3

        # Wildlife/Nature override: if "no humans" + "animal", force LANDSCAPE over PORTRAIT
        if has_no_humans and has_animal:
            # Boost landscape significantly
            scores[SceneType.LANDSCAPE] += 3.0
            # Reduce human-focused scene types
            scores[SceneType.PORTRAIT] *= 0.1
            scores[SceneType.FASHION] *= 0.1
            scores[SceneType.GROUP] *= 0.1

        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_scores[0][0]
        primary_conf = sorted_scores[0][1]

        # Secondary only if significantly present
        secondary = None
        secondary_conf = 0.0
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.3:
            secondary = sorted_scores[1][0]
            secondary_conf = sorted_scores[1][1]

        return SceneDetectionResult(
            primary=primary,
            primary_confidence=primary_conf,
            secondary=secondary,
            secondary_confidence=secondary_conf,
            all_scores=scores,
        )


# =============================================================================
# CANONICAL CATEGORY CLASSIFIER
# =============================================================================

class CanonicalClassifier:
    """Classifies tags into canonical categories."""

    def __init__(self):
        self.category_keywords = CATEGORY_KEYWORDS
        self._build_reverse_index()

    def _build_reverse_index(self):
        """Build reverse index from keyword to (category, subcategory)."""
        self.reverse_index: Dict[str, Tuple[CanonicalCategory, str]] = {}
        for category, subcategories in self.category_keywords.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    key = keyword.lower()
                    # Prefer more specific matches
                    if key not in self.reverse_index:
                        self.reverse_index[key] = (category, subcategory)

    def classify_tag(
        self,
        tag: str,
        confidence: float,
        source: str,
    ) -> Optional[CategoryItem]:
        """
        Classify a single tag into a canonical category.

        Returns CategoryItem if classified, None if unclassified.
        """
        tag_lower = tag.lower().strip()

        # Exact match
        if tag_lower in self.reverse_index:
            category, subcategory = self.reverse_index[tag_lower]
            return CategoryItem(
                text=tag,
                confidence=confidence,
                source=source,
                subcategory=subcategory,
            ), category

        # Partial match - find best matching keyword
        best_match = None
        best_category = None
        best_subcategory = None

        for keyword, (category, subcategory) in self.reverse_index.items():
            if keyword in tag_lower or tag_lower in keyword:
                if best_match is None or len(keyword) > len(best_match):
                    best_match = keyword
                    best_category = category
                    best_subcategory = subcategory

        if best_match:
            return CategoryItem(
                text=tag,
                confidence=confidence,
                source=source,
                subcategory=best_subcategory,
            ), best_category

        return None, None

    def classify_all(
        self,
        metadata: Dict[str, Dict[str, float]],
        threshold: float = 0.0,
    ) -> Tuple[Dict[CanonicalCategory, List[CategoryItem]], Dict[str, float], Dict[str, float]]:
        """
        Classify all tags from metadata into canonical categories.

        Args:
            metadata: Dict of source -> {tag: confidence}
            threshold: Minimum confidence to include in categories (0-1)

        Returns:
            Tuple of:
                - Dict of category -> list of CategoryItems (above threshold, categorized)
                - Dict of uncategorized tags -> confidence (above threshold, not categorized)
                - Dict of below_threshold tags -> confidence (below threshold)
        """
        result = {cat: [] for cat in CanonicalCategory}
        uncategorized: Dict[str, float] = {}
        below_threshold: Dict[str, float] = {}

        for source, tags in metadata.items():
            if not isinstance(tags, dict):
                continue
            # Skip non-tag metadata
            if source.startswith("florence_") or source == "image_info":
                continue
            # Skip bbox_analysis and canonical (not tag sources)
            if source in ("bbox_analysis", "canonical", "session_id", "tag_filter"):
                continue

            for tag, confidence in tags.items():
                if not isinstance(confidence, (int, float)):
                    continue

                # Check threshold first
                if confidence < threshold:
                    # Below threshold - keep highest confidence if duplicate
                    if tag not in below_threshold or confidence > below_threshold[tag]:
                        below_threshold[tag] = confidence
                    continue

                # Above threshold - try to classify
                item, category = self.classify_tag(tag, confidence, source)
                if item and category:
                    result[category].append(item)
                else:
                    # Unclassified - keep highest confidence if duplicate
                    if tag not in uncategorized or confidence > uncategorized[tag]:
                        uncategorized[tag] = confidence

        # Sort each category by confidence
        for category in result:
            result[category].sort(key=lambda x: x.confidence, reverse=True)

        # Sort uncategorized by confidence
        uncategorized = dict(sorted(uncategorized.items(), key=lambda x: x[1], reverse=True))

        # Sort below_threshold by confidence
        below_threshold = dict(sorted(below_threshold.items(), key=lambda x: x[1], reverse=True))

        return result, uncategorized, below_threshold


# =============================================================================
# TEXT GENERATOR
# =============================================================================

class CanonicalTextGenerator:
    """Generates natural language text for each category."""

    def generate_subject_text(self, items: List[CategoryItem]) -> str:
        """Generate subject description text."""
        if not items:
            return ""

        # Group by subcategory
        groups = self._group_by_subcategory(items)

        parts = []

        # Person type first
        if "person_type" in groups:
            person = groups["person_type"][0].text
            parts.append(person)

        # Attributes
        if "attributes" in groups:
            attrs = [item.text for item in groups["attributes"][:3]]
            if attrs:
                parts.append(", ".join(attrs))

        # Hair
        if "hair" in groups:
            hair = [item.text for item in groups["hair"][:2]]
            if hair:
                parts.append(f"with {' '.join(hair)}")

        # Expression
        if "expression" in groups:
            expr = groups["expression"][0].text
            parts.append(f"with {expr}")

        # Clothing
        if "clothing" in groups:
            clothes = [item.text for item in groups["clothing"][:5]]
            if clothes:
                parts.append(f"wearing {', '.join(clothes)}")

        # Pose
        if "pose" in groups:
            pose = groups["pose"][0].text
            parts.append(pose)

        return ". ".join(parts) if parts else ""

    def generate_scene_text(self, items: List[CategoryItem]) -> str:
        """Generate scene/environment description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Location
        location = None
        if "location_indoor" in groups:
            location = groups["location_indoor"][0].text
        elif "location_outdoor" in groups:
            location = groups["location_outdoor"][0].text

        if location:
            parts.append(f"in {location}")

        # Time
        if "time" in groups:
            time = groups["time"][0].text
            parts.append(f"at {time}")

        # Weather
        if "weather" in groups:
            weather = groups["weather"][0].text
            parts.append(f"{weather} conditions")

        # Atmosphere
        if "atmosphere" in groups:
            atmos = [item.text for item in groups["atmosphere"][:2]]
            if atmos:
                parts.append(f"{', '.join(atmos)} atmosphere")

        # Background
        if "background" in groups:
            bg = groups["background"][0].text
            parts.append(bg)

        return ". ".join(parts) if parts else ""

    def generate_composition_text(self, items: List[CategoryItem]) -> str:
        """Generate composition description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Shot type
        if "shot_type" in groups:
            shot = groups["shot_type"][0].text
            parts.append(shot)

        # Angle
        if "angle" in groups:
            angle = groups["angle"][0].text
            parts.append(f"from {angle}")

        # Framing
        if "framing" in groups:
            framing = [item.text for item in groups["framing"][:2]]
            if framing:
                parts.append(", ".join(framing))

        # Lens
        if "lens" in groups:
            lens = groups["lens"][0].text
            parts.append(f"shot with {lens}")

        # Focus
        if "focus" in groups:
            focus = groups["focus"][0].text
            parts.append(focus)

        # Aspect
        if "aspect" in groups:
            aspect = groups["aspect"][0].text
            parts.append(aspect)

        return ". ".join(parts) if parts else ""

    def generate_lighting_text(self, items: List[CategoryItem]) -> str:
        """Generate lighting description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Direction
        if "direction" in groups:
            direction = [item.text for item in groups["direction"][:2]]
            parts.append(", ".join(direction))

        # Quality
        if "quality" in groups:
            quality = [item.text for item in groups["quality"][:2]]
            parts.append(", ".join(quality))

        # Type
        if "type" in groups:
            light_type = groups["type"][0].text
            parts.append(light_type)

        # Style
        if "style" in groups:
            style = groups["style"][0].text
            parts.append(f"{style} lighting")

        # Color
        if "color" in groups:
            color = groups["color"][0].text
            parts.append(color)

        # Effects
        if "effects" in groups:
            effects = [item.text for item in groups["effects"][:3]]
            if effects:
                parts.append(f"with {', '.join(effects)}")

        return ". ".join(parts) if parts else ""

    def generate_style_text(self, items: List[CategoryItem]) -> str:
        """Generate style description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Medium
        if "medium" in groups:
            medium = groups["medium"][0].text
            parts.append(medium)

        # Aesthetic
        if "aesthetic" in groups:
            aesthetic = [item.text for item in groups["aesthetic"][:2]]
            parts.append(", ".join(aesthetic))

        # Genre
        if "genre" in groups:
            genre = groups["genre"][0].text
            parts.append(genre)

        # Color grade
        if "color_grade" in groups:
            grade = [item.text for item in groups["color_grade"][:2]]
            parts.append(", ".join(grade))

        # Art reference
        if "art_reference" in groups:
            ref = groups["art_reference"][0].text
            parts.append(f"inspired by {ref}")

        # Film reference
        if "film_reference" in groups:
            film = groups["film_reference"][0].text
            parts.append(f"shot on {film}")

        return ". ".join(parts) if parts else ""

    def generate_constraints_text(self, items: List[CategoryItem]) -> str:
        """Generate constraints text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Quality
        if "quality" in groups:
            quality = [item.text for item in groups["quality"][:3]]
            parts.append(", ".join(quality))

        # Technical
        if "technical" in groups:
            tech = [item.text for item in groups["technical"][:2]]
            parts.append(", ".join(tech))

        return ". ".join(parts) if parts else ""

    def _group_by_subcategory(
        self,
        items: List[CategoryItem]
    ) -> Dict[str, List[CategoryItem]]:
        """Group items by subcategory."""
        groups: Dict[str, List[CategoryItem]] = {}
        for item in items:
            if item.subcategory not in groups:
                groups[item.subcategory] = []
            groups[item.subcategory].append(item)
        return groups


# =============================================================================
# MAIN STRUCTURER
# =============================================================================

class CanonicalStructurer:
    """
    Main class that orchestrates scene detection, classification, and text generation.
    """

    def __init__(self):
        self.detector = SceneTypeDetector()
        self.classifier = CanonicalClassifier()
        self.generator = CanonicalTextGenerator()

    def structure(self, metadata: Dict, threshold: float = 0.0) -> CanonicalStructure:
        """
        Process metadata into canonical structure.

        Args:
            metadata: Full metadata dict from tagging pipeline
            threshold: Minimum confidence for categorization (from tag_filter)

        Returns:
            CanonicalStructure with all categories populated
        """
        # Collect all tags with confidences (for scene detection - uses all tags)
        all_tags = {}
        for source, data in metadata.items():
            if isinstance(data, dict) and not source.startswith("florence_"):
                if source not in ("image_info", "bbox_analysis", "canonical", "session_id", "tag_filter"):
                    for tag, conf in data.items():
                        if isinstance(conf, (int, float)):
                            # Keep highest confidence if duplicate
                            if tag not in all_tags or conf > all_tags[tag]:
                                all_tags[tag] = conf

        # Get Florence captions
        florence_caption = metadata.get("florence_caption", "")
        florence_description = metadata.get("florence_description", "")

        # Detect scene type (uses all tags, not filtered)
        scene_result = self.detector.detect(
            all_tags,
            florence_caption,
            florence_description,
        )

        # Classify into categories (applies threshold)
        classified, uncategorized, below_threshold = self.classifier.classify_all(metadata, threshold)

        # Create structure
        structure = CanonicalStructure(
            scene_type=scene_result,
            subject=classified[CanonicalCategory.SUBJECT],
            scene=classified[CanonicalCategory.SCENE],
            composition=classified[CanonicalCategory.COMPOSITION],
            lighting=classified[CanonicalCategory.LIGHTING],
            style=classified[CanonicalCategory.STYLE],
            constraints=classified[CanonicalCategory.CONSTRAINTS],
            uncategorized=uncategorized,
            below_threshold=below_threshold,
        )

        # Generate text for each category
        structure.subject_text = self.generator.generate_subject_text(structure.subject)
        structure.scene_text = self.generator.generate_scene_text(structure.scene)
        structure.composition_text = self.generator.generate_composition_text(structure.composition)
        structure.lighting_text = self.generator.generate_lighting_text(structure.lighting)
        structure.style_text = self.generator.generate_style_text(structure.style)
        structure.constraints_text = self.generator.generate_constraints_text(structure.constraints)

        # Generate full prompt
        structure.full_prompt = self._combine_prompt(structure)

        return structure

    def _combine_prompt(self, structure: CanonicalStructure) -> str:
        """Combine all category texts into full prompt."""
        parts = []

        if structure.subject_text:
            parts.append(structure.subject_text)
        if structure.scene_text:
            parts.append(structure.scene_text)
        if structure.composition_text:
            parts.append(structure.composition_text)
        if structure.lighting_text:
            parts.append(structure.lighting_text)
        if structure.style_text:
            parts.append(structure.style_text)
        if structure.constraints_text:
            parts.append(structure.constraints_text)

        return ". ".join(parts)

    def _items_to_dict(self, items: List[CategoryItem]) -> Dict[str, float]:
        """Convert items to simple {tag: confidence} dict, keeping highest confidence for duplicates."""
        result = {}
        for item in items:
            tag = item.text
            conf = round(item.confidence, 3)
            if tag not in result or conf > result[tag]:
                result[tag] = conf
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def to_json(self, structure: CanonicalStructure) -> Dict:
        """
        Convert structure to JSON-serializable dict.

        Applies scene-based category filtering - categories not relevant
        to the detected scene type are excluded from output.
        """
        # Get relevant categories for this scene type
        scene_type = structure.scene_type.primary
        relevant = SCENE_CATEGORY_RELEVANCE.get(scene_type, DEFAULT_RELEVANT_CATEGORIES)

        result = {
            "scene_detection": {
                "primary": scene_type.value,
                "primary_confidence": round(structure.scene_type.primary_confidence, 3),
                "secondary": structure.scene_type.secondary.value if structure.scene_type.secondary else None,
                "secondary_confidence": round(structure.scene_type.secondary_confidence, 3),
            },
        }

        # Only include categories relevant to this scene type
        if CanonicalCategory.SUBJECT in relevant:
            result["subject"] = self._items_to_dict(structure.subject)

        if CanonicalCategory.SCENE in relevant:
            result["scene"] = self._items_to_dict(structure.scene)

        if CanonicalCategory.COMPOSITION in relevant:
            result["composition"] = self._items_to_dict(structure.composition)

        if CanonicalCategory.LIGHTING in relevant:
            result["lighting"] = self._items_to_dict(structure.lighting)

        if CanonicalCategory.STYLE in relevant:
            result["style"] = self._items_to_dict(structure.style)

        if CanonicalCategory.CONSTRAINTS in relevant:
            result["constraints"] = self._items_to_dict(structure.constraints)

        # Uncategorized and below_threshold are always included
        result["uncategorized"] = {
            tag: round(conf, 3) for tag, conf in structure.uncategorized.items()
        }
        result["below_threshold"] = {
            tag: round(conf, 3) for tag, conf in structure.below_threshold.items()
        }

        return result
