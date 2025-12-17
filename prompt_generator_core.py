# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Prompt Generator Core

Standalone prompt generation module that can run independently of ComfyUI.
Supports batch processing for training data generation.

Usage (standalone):
    from prompt_generator_core import PromptGenerator

    generator = PromptGenerator(
        provider="ollama",
        model="llava:latest",
        api_url="http://localhost:11434/v1"
    )

    result = generator.process_image("path/to/image.jpg")
    print(result["prompt"])

    # Or batch process:
    results = generator.process_batch("path/to/images/", output_file="prompts.jsonl")

Command line:
    python prompt_generator_core.py --image path/to/image.jpg --provider ollama --model llava
    python prompt_generator_core.py --batch path/to/images/ --output prompts.jsonl

Author: Siddhartha Lahiri
License: MIT
"""

import os
import io
import gc
import re
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

# =============================================================================
# Config Loader - Try to import, fallback to defaults if not available
# =============================================================================

_CONFIG_AVAILABLE = False
_config_loader = None

try:
    # Try relative import (when used as part of ComfyUI package)
    from . import config_loader as _config_loader
    _CONFIG_AVAILABLE = True
except ImportError:
    try:
        # Try absolute import (when run standalone from same directory)
        import config_loader as _config_loader
        _CONFIG_AVAILABLE = True
    except ImportError:
        _CONFIG_AVAILABLE = False
        print("[PromptGen] Config loader not available, using built-in defaults")

# =============================================================================
# Configuration
# =============================================================================

ANALYSIS_MODES = ["quick", "standard", "detailed"]

IMAGE_SIZES = {
    "local": {"cv": 640, "llm": 512, "quality": 80},
    "api": {"cv": 640, "llm": 1024, "quality": 85},
}

LOCAL_PROVIDERS = ["ollama", "lmstudio", "local"]
API_PROVIDERS = ["anthropic", "openai", "gemini", "groq", "together",
                 "openrouter", "fireworks", "cerebras", "mistral", "deepseek"]

# Z-Image cleanup patterns
ZIMAGE_META_TAGS = [
    r'\b8K\b', r'\b4K\b', r'\bHDR\b', r'\bUHD\b',
    r'\bmasterpiece\b', r'\bbest quality\b', r'\bhigh quality\b',
    r'\bultra detailed\b', r'\bhighly detailed\b', r'\bsuper detailed\b',
    r'\bprofessional photo\b', r'\baward winning\b', r'\bphoto realistic\b',
    r'\btrending on artstation\b', r'\bunreal engine\b', r'\boctane render\b',
]

ZIMAGE_ABSTRACT_WORDS = [
    r'\bbeautiful\b', r'\bstunning\b', r'\bgorgeous\b', r'\bbreakthtaking\b',
    r'\bamazing\b', r'\bperfect\b', r'\bincredible\b', r'\bwonderful\b',
    r'\bmagnificent\b', r'\bexquisite\b', r'\belegant\b',
]

# Comprehensive uniform and costume types for detection
UNIFORM_COSTUME_TYPES = """
RELIGIOUS:
- Nun habit types:
  * Roman Catholic nun (traditional black habit, white coif/wimple)
  * Modern Catholic nun (simplified habit, shorter veil)
  * Dominican nun (white habit with black veil)
  * Carmelite nun (brown habit)
  * Benedictine nun (black habit)
  * Stylized/sexy nun (crop top variant, short habit, exposed midriff)
- Priest vestments (black cassock, white collar)
- Monk robes (brown Franciscan, black Benedictine, orange Buddhist)

MEDICAL:
- Nurse uniform types:
  * Traditional white nurse (white dress, cap, stockings)
  * Modern scrubs nurse (blue, green, pink scrubs)
  * Vintage 1950s pin-up nurse
  * Sexy nurse (short white dress, red cross)
  * Japanese nurse (white with cap)
- Doctor/medical:
  * White lab coat
  * Surgical scrubs (blue, green)
  * Surgeon (gown, mask, cap)
  * EMT/Paramedic (dark uniform, reflective)

SERVICE/HOSPITALITY:
- Maid outfit types:
  * French maid (black dress, white apron, headpiece)
  * Victorian housemaid (long black dress, white cap)
  * Hotel maid (gray/blue uniform)
  * Japanese maid cafe (frilly, colorful)
  * Sexy maid (short skirt, revealing)
- Flight attendant by era/airline:
  * Vintage 1960s stewardess
  * Modern airline (navy, red, specific airline colors)
- Waitress (diner, cocktail, Hooters-style orange)

MILITARY/LAW ENFORCEMENT:
- Police by country/type:
  * US Police (dark blue/black, badge, duty belt)
  * UK Police (black, custodian helmet)
  * Japanese Police (blue uniform)
  * SWAT (black tactical gear)
  * Sexy cop (short shorts, midriff)
- Military by branch/country:
  * US Army (green/camo ACU)
  * US Navy (white dress, blue working)
  * US Marines (dress blues, desert camo)
  * British Army (MTP camo)
  * Dress uniform vs combat/field uniform
- Security guard (black/navy uniform)

SCHOOL/ACADEMIC:
- Schoolgirl uniform types:
  * Japanese sailor fuku (white/navy, red ribbon)
  * Japanese blazer style (plaid skirt, blazer)
  * British school (gray skirt, blazer, tie)
  * Catholic school (plaid skirt, white blouse, cardigan)
  * Korean school uniform
  * Sexy schoolgirl (short plaid skirt)
- Cheerleader (team colors, pom poms)
- Gym uniform (bloomers, t-shirt)

SPORTS/FITNESS:
- Gym attire (sports bra, leggings, shorts)
- Yoga outfit (tight leggings, crop top)
- Tennis outfit (white dress/skirt)
- Swimsuit (one-piece, bikini, sports)
- Volleyball uniform (tight shorts, jersey)

FANTASY/COSPLAY:
- Bunny girl (Playboy bunny ears, leotard, cuffs, collar)
- Catsuit/latex bodysuit
- Angel costume (white, wings, halo)
- Devil costume (red, horns, tail)
- Cat girl (ears, tail, collar)
- Superhero costume (specify character)

TRADITIONAL/CULTURAL:
- Japanese: kimono (formal), yukata (casual), geisha
- Chinese: cheongsam/qipao (form-fitting), hanfu (traditional)
- Indian: saree, lehenga, salwar kameez
- Korean: hanbok (colorful, full skirt)
- German: dirndl (bodice, apron, full skirt)
- Belly dancer (coin belt, bra top, flowing skirt)

PROFESSIONAL:
- Business suit (formal black, navy, gray)
- Secretary/office (pencil skirt, blouse)
- Chef whites (double-breasted jacket, toque)
- Pilot uniform (navy with stripes, cap)
- Construction (hard hat, orange vest, jeans)

ALWAYS SPECIFY:
- Exact type/variant (Roman Catholic nun, not just "nun")
- Color scheme (black/white, navy/gold, etc.)
- Traditional vs modern vs stylized vs sexy variant
- Any modifications (crop top, short skirt, exposed areas)
"""

# =============================================================================
# Prompt Templates
# =============================================================================

ZIMAGE_SYSTEM_RULES = """
OUTPUT FORMAT: Write as flowing natural language, NOT keyword lists.

STRUCTURE YOUR DESCRIPTION:
1. Frame style and shot type (e.g., "A selfie-style close-up...", "A professional portrait...", "A candid medium shot...")
2. Subject details (face, body, clothing, pose if human; or main subject if scene)
3. Lighting quality, direction, and SOURCE (e.g., "natural window light from the left", "soft diffused daylight")
4. Background and environment
5. Technical notes (depth of field, etc.)

CRITICAL DETAILS FOR PORTRAITS:
- ETHNICITY/ANCESTRY: ALWAYS identify first - be specific: East Asian (Chinese, Japanese, Korean, Vietnamese), South Asian (Indian, Pakistani, Bengali), Southeast Asian (Thai, Filipino, Indonesian), Middle Eastern, African, African American, Caribbean, Latino/Hispanic, Caucasian/European (Northern, Southern, Eastern European), Mixed heritage. Describe facial features that indicate ethnicity.
- SHOT TYPE/FRAMING (MUST specify exactly where body is cropped):
  * EXTREME CLOSE-UP: Face only, cropped at forehead and chin
  * CLOSE-UP/HEADSHOT: Head and neck, cropped at shoulders
  * MEDIUM CLOSE-UP: Head to chest, cropped at chest/bust level
  * MEDIUM SHOT: Head to waist, cropped at waist
  * COWBOY SHOT: Head to mid-thigh, cropped at mid-thigh
  * FULL BODY: Head to feet, entire body visible including feet
  * WIDE SHOT: Full body with significant environment visible
  IMPORTANT: State exactly where the frame cuts off (e.g., "cropped at mid-thigh", "full body with feet visible")
- SELFIE DETECTION: If selfie - arm extended, phone/camera angle, front-facing camera, above eye level. State "selfie-style" and "eyes looking directly at camera"
- EYES: Describe eye color, iris detail/pattern, shape (almond, round, hooded, monolid, double-lid), expression, catchlights, gaze direction
- EYELASHES: Color (black, brown, blonde), style (natural, long, thick, curled, sparse), any mascara effect
- EYEBROWS: Shape (arched, straight, rounded, angular), thickness, color, groomed/natural, any gaps or asymmetry
- SKIN TEXTURE: Skin tone (fair, light, medium, olive, tan, brown, dark brown, deep), visible details - freckles (location, density), dimples (cheeks, chin), wrinkles/fine lines (forehead, crow's feet, smile lines), pores, moles, blemishes, skin smoothness or roughness
- BODY DEFINITION: If visible, describe muscle definition - abs/midriff cuts (six-pack, four-pack, toned), chest definition, arm muscle tone (biceps, triceps), shoulder definition, back muscles. NAVEL: shape (round, vertical, horizontal, outie/innie), depth, surrounding area
- HAIR: Describe hairstyle (length, texture, layers, parting), hair color with highlights/lowlights, how it frames the face
- MAKEUP: If visible, describe in detail - eyeshadow color, eyeliner style, mascara, blush placement, foundation coverage, contouring
- LIPS: Exact color/shade (nude, pink, red, burgundy, coral, mauve), finish (matte, glossy, satin), lip texture (smooth, chapped, dry lines, natural creases), lip shape and fullness
- LIGHTING SOURCE: Always identify WHERE the light comes from (window, overhead, artificial), direction (left/right/front/back), quality (soft/hard/diffused)
- COSTUME/UNIFORM: ALWAYS identify recognizable uniforms or costumes FIRST with SPECIFIC TYPE:
  * Religious: Roman Catholic nun habit (black/white), Dominican nun (white habit), Carmelite nun (brown), stylized/sexy nun (crop top variant)
  * Medical: traditional white nurse, modern scrubs nurse (blue/green/pink), vintage pin-up nurse, sexy nurse, white lab coat, surgical scrubs
  * Service: French maid (black/white apron), Victorian housemaid, Japanese maid cafe, sexy maid, vintage stewardess, modern flight attendant
  * Military/Law: US Police (dark blue), UK Police (black, helmet), SWAT (tactical black), sexy cop, US Army camo, US Navy (white dress/blue working), dress uniform vs combat
  * School: Japanese sailor fuku, Japanese blazer style, British school uniform, Catholic school (plaid), Korean school, sexy schoolgirl, cheerleader
  * Sports: gym attire, yoga outfit, tennis whites, one-piece swimsuit, bikini, volleyball uniform
  * Fantasy: Playboy bunny (ears, leotard), catsuit/latex, angel (white, wings), devil (red, horns), cat girl
  * Cultural: kimono/yukata, cheongsam/qipao, saree/lehenga, hanbok, dirndl, belly dancer
  * Professional: business suit (black/navy), secretary (pencil skirt), chef whites, pilot uniform
  ALWAYS specify: exact type, color scheme, traditional/modern/stylized/sexy variant, any modifications (crop top, exposed midriff).
- CLOTHING CONSTRUCTION & LAYERS (CRITICAL for accurate recreation):
  * COLOR: EXACT color of EACH garment piece - be specific (jet black, charcoal gray, cream white, ivory, navy blue, burgundy, forest green, etc.) Include color variations within same garment (ombre, two-tone, color blocking)
  * FABRIC/MATERIAL: cotton, silk, satin, velvet, leather, latex, lace, mesh, sheer, denim, wool, knit/ribbed, jersey, chiffon, sequined, metallic
  * TEXTURE/FINISH: matte, shiny/glossy, textured, smooth, ribbed, quilted, pleated, ruched
  * GARMENT TYPE: fitted/form-fitting vs loose/open/flowy, structured vs unstructured
  * LAYERING ORDER: describe from innermost to outermost WITH COLORS (e.g., "cream white ribbed turtleneck undershirt UNDER fitted jet black crop top")
  * NECKLINE: crew neck, scoop neck, v-neck, deep v, turtleneck/mock neck, off-shoulder, sweetheart, square, halter, strapless
  * HEM/LENGTH: cropped at midriff (crop top), cropped below bust, waist-length, hip-length, full-length, asymmetrical hem
  * SLEEVES: sleeveless, cap sleeve, short sleeve, 3/4 sleeve, long sleeve, bell sleeve, tight-fitted vs loose sleeves
  * FIT: body-hugging/skin-tight, fitted, semi-fitted, relaxed, loose, oversized
  * CLOSURE: pullover (no opening), front-zip, back-zip, buttons down front, wrap style, open front (cardigan/robe style)
  * CONSTRUCTION: is it ONE PIECE (dress, jumpsuit, bodysuit) or SEPARATE PIECES (top + bottom, layered tops)?
  * PATTERNS: solid, striped (width, direction), plaid/tartan, floral, polka dot, animal print, geometric, abstract
- CLOTHING STATE: buttons open/closed, collar up/down, sleeves rolled/unrolled, shirt tucked/untucked, zippers open/closed, ties loosened, exposed midriff, visible undershirt/layer
- BEVERAGES/LIQUIDS: If visible, describe liquid color accurately - red wine (deep burgundy), white wine (pale gold), beer (amber), coffee (dark brown), water (clear), cocktails (specific colors)
- ATMOSPHERIC EFFECTS: If present, describe any smoke, steam, vapor, mist, fog, dust particles, or haze - especially rising steam from cups/beverages, cigarette smoke, breath vapor, etc.

DO:
- Write complete, flowing sentences
- ALWAYS start with ethnicity/ancestry - be specific, not vague
- Use specific visual vocabulary (colors, textures, materials)
- Describe lighting direction, quality, AND source explicitly
- Include camera/lens references when relevant
- Be extremely specific about ethnicity, eyes, eyelashes, eyebrows, skin tone/texture, hair, makeup, and lips
- Identify selfie vs professional vs candid photo style
- Note any freckles, dimples, wrinkles, or unique skin features

DO NOT:
- Use comma-separated keyword lists
- Include quality tags: "8K", "masterpiece", "best quality", "ultra detailed"
- Use vague adjectives: "beautiful", "stunning", "amazing", "perfect"
- Add negative prompts (except "no text, no watermark" at end)
- Reference AI generation or model names
- Write philosophical/spiritual/narrative commentary
- Include metaphors about life, death, eternity, or cosmic themes
- Add business/corporate jargon
- Repeat the same information multiple times
- Write one extremely long run-on sentence

SENTENCE STRUCTURE:
- Write 3-6 clear, distinct sentences
- Each sentence should describe ONE aspect (subject, clothing, lighting, background)
- Keep sentences focused and factual
- STOP when you have described all visible elements - do not add commentary

END WITH: "no text, no watermark"
"""

ZIMAGE_SCENE_RULES = """
IMPORTANT: There are NO people in this image. Do NOT describe any humans, persons,
faces, bodies, or human-related elements.

CRITICAL - SHOT TYPE AND FRAMING (MUST specify first):
- EXTREME CLOSE-UP: Only face/head visible, fills most of frame, no neck/body
- CLOSE-UP: Head and partial neck visible
- MEDIUM CLOSE-UP: Head and full neck, partial body
- MEDIUM SHOT: Upper body visible
- FULL BODY: Entire animal/subject visible
- WIDE SHOT: Subject small in frame, environment dominant
- DETAIL SHOT: Specific body part (eye, paw, feather, etc.)

FOR WILDLIFE/ANIMALS:
- Species identification (specific type, e.g., "reticulated giraffe", "African elephant")
- Age (adult, juvenile, calf, etc.)
- Pose (front-facing, profile, three-quarter view)
- Eye contact (looking at camera, looking away)
- Distinctive features (patterns, markings, horns, tusks, mane)
- Expression/mood if applicable

Focus on:
- Landscape, environment, architecture
- Objects, vehicles, animals
- Lighting, atmosphere, mood
- Colors, textures, composition

DO NOT:
- Write philosophical/spiritual commentary
- Add metaphors or symbolic interpretations
- Include narrative storytelling
- Repeat information multiple times

SENTENCE STRUCTURE:
- Write 3-5 clear, factual sentences
- Each sentence describes ONE aspect
- STOP when all visible elements are described
"""


# =============================================================================
# Config Wrapper Functions (use TOML if available, else defaults)
# =============================================================================

def get_provider_tier(provider: str) -> str:
    """Get provider tier from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_provider_tier(provider)
    # Default mapping
    if provider.lower() in ["anthropic", "openai", "google", "gemini", "mistral", "deepseek", "openrouter", "xai", "grok"]:
        return "advanced"
    elif provider.lower() in ["together", "fireworks", "groq", "perplexity"]:
        return "standard"
    else:  # ollama, lmstudio, local, etc.
        return "basic"


def get_system_prompt(tier: str) -> str:
    """Get system prompt from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_system_prompt(tier)
    # Fallback defaults
    return ZIMAGE_SYSTEM_RULES


def get_user_prompt(tier: str, mode: str) -> str:
    """Get user prompt from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_user_prompt(tier, mode)
    # Fallback - return empty (will use built-in)
    return ""


def get_scene_system_prompt() -> str:
    """Get scene-only system prompt from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_scene_only_system_prompt()
    return ZIMAGE_SCENE_RULES


def get_scene_user_prompt(mode: str) -> str:
    """Get scene-only user prompt from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_scene_only_user_prompt(mode)
    return ""


def get_length_constraint(prompt_length: int) -> str:
    """Get length constraint instruction from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_length_constraint(prompt_length)
    if prompt_length <= 0:
        return ""
    return f"OUTPUT LENGTH: Write a prompt of approximately {prompt_length} words."


def clean_llm_output(text: str, provider: str = "default") -> str:
    """Clean LLM output using config patterns or basic cleanup."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.clean_output(text, provider)
    # Basic cleanup
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_agentic_prompt(tier: str, prompt_type: str) -> str:
    """Get agentic synthesis prompt from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_agentic_prompt(tier, prompt_type)
    return ""


def get_component_prompt(component: str, tier: str) -> str:
    """Get component prompt for agentic analysis from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_component_prompt(component, tier)
    return ""


def get_image_limit(provider: str) -> int:
    """Get max image size for provider from config or default."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_image_limit(provider)
    # Defaults
    limits = {
        "anthropic": 1568,
        "openai": 2048,
        "ollama": 1024,
        "lmstudio": 1024,
        "local": 1280,
    }
    return limits.get(provider.lower(), 1024)


def get_vocabulary_terms() -> Dict[str, List[str]]:
    """Load all vocabulary terms from config."""
    if _CONFIG_AVAILABLE and _config_loader:
        try:
            vocab = _config_loader.get_prompts_config().get("vocabulary", {})
            result = {}
            for key, value in vocab.items():
                if isinstance(value, dict) and "terms" in value:
                    result[key] = value["terms"]
            return result
        except Exception:
            pass
    # Fallback vocabulary
    return {
        "skin_tones": ["warm ivory", "cool beige", "golden tan", "olive", "porcelain"],
        "hair_textures": ["straight", "wavy", "curly", "coily"],
        "lighting": ["soft diffused", "dramatic side", "natural window", "golden hour", "studio"],
    }


def convert_to_tags(expanded_prompt: str, vocabulary: Dict[str, List[str]] = None) -> str:
    """
    Convert an expanded prompt to comma-separated tags, enhanced with vocabulary.

    This extracts key visual elements and converts them to booru-style tags.
    """
    if not expanded_prompt:
        return ""

    # Load vocabulary if not provided
    if vocabulary is None:
        vocabulary = get_vocabulary_terms()

    # Start with base quality tags
    tags = ["masterpiece", "best quality", "highres"]

    # Extract and convert key phrases to tags
    prompt_lower = expanded_prompt.lower()

    # Gender detection
    if any(w in prompt_lower for w in ["woman", "girl", "female", "she", "her"]):
        tags.append("1girl")
    elif any(w in prompt_lower for w in ["man", "boy", "male", "he", "his"]):
        tags.append("1boy")

    # Solo detection
    if "solo" in prompt_lower or not any(w in prompt_lower for w in ["couple", "group", "people", "two", "multiple"]):
        if "1girl" in tags or "1boy" in tags:
            tags.append("solo")

    # Hair color
    hair_colors = ["blonde", "brown", "black", "red", "white", "silver", "pink", "blue", "green", "purple", "auburn", "chestnut"]
    for color in hair_colors:
        if color in prompt_lower and "hair" in prompt_lower:
            tags.append(f"{color}_hair")
            break

    # Hair style
    hair_styles = {
        "ponytail": "ponytail", "braid": "braid", "twintails": "twintails",
        "short hair": "short_hair", "long hair": "long_hair", "medium hair": "medium_hair",
        "wavy": "wavy_hair", "curly": "curly_hair", "straight": "straight_hair",
        "bangs": "bangs", "bob": "bob_cut", "bun": "hair_bun"
    }
    for style, tag in hair_styles.items():
        if style in prompt_lower:
            tags.append(tag)

    # Eye color
    eye_colors = ["blue", "green", "brown", "amber", "hazel", "grey", "red", "purple"]
    for color in eye_colors:
        if f"{color} eye" in prompt_lower:
            tags.append(f"{color}_eyes")
            break

    # Expression
    expressions = {
        "smiling": "smile", "laughing": "laughing", "serious": "serious",
        "angry": "angry", "sad": "sad", "crying": "crying",
        "surprised": "surprised", "blushing": "blush", "confident": "confident"
    }
    for expr, tag in expressions.items():
        if expr in prompt_lower:
            tags.append(tag)
            break

    # Pose/View
    poses = {
        "facing camera": "looking_at_viewer", "looking at viewer": "looking_at_viewer",
        "from behind": "from_behind", "from side": "from_side", "profile": "profile",
        "three-quarter": "three-quarter_view", "standing": "standing", "sitting": "sitting",
        "lying": "lying", "kneeling": "kneeling"
    }
    for pose, tag in poses.items():
        if pose in prompt_lower:
            tags.append(tag)

    # Clothing (extract key garments)
    clothing = {
        "dress": "dress", "shirt": "shirt", "blouse": "blouse", "jacket": "jacket",
        "coat": "coat", "sweater": "sweater", "hoodie": "hoodie", "bikini": "bikini",
        "swimsuit": "swimsuit", "uniform": "uniform", "suit": "suit", "skirt": "skirt",
        "jeans": "jeans", "pants": "pants", "shorts": "shorts"
    }
    for item, tag in clothing.items():
        if item in prompt_lower:
            tags.append(tag)

    # Environment/Background
    backgrounds = {
        "outdoor": "outdoors", "indoor": "indoors", "studio": "studio",
        "city": "cityscape", "nature": "nature", "beach": "beach", "forest": "forest",
        "snow": "snow", "rain": "rain", "night": "night", "sunset": "sunset",
        "sunrise": "sunrise"
    }
    for bg, tag in backgrounds.items():
        if bg in prompt_lower:
            tags.append(tag)

    # Lighting
    lighting = {
        "dramatic": "dramatic_lighting", "soft": "soft_lighting", "backlighting": "backlighting",
        "rim light": "rim_lighting", "golden hour": "golden_hour", "natural light": "natural_lighting"
    }
    for light, tag in lighting.items():
        if light in prompt_lower:
            tags.append(tag)

    # Add vocabulary-matched terms
    all_vocab_terms = []
    for terms in vocabulary.values():
        all_vocab_terms.extend(terms)

    for term in all_vocab_terms:
        if term.lower() in prompt_lower and term.lower().replace(" ", "_") not in tags:
            # Convert to tag format
            tag_format = term.lower().replace(" ", "_").replace("-", "_")
            if tag_format not in tags:
                tags.append(tag_format)

    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    return ", ".join(unique_tags)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str
    model: str  # Vision model
    text_model: str = ""  # Text model (if different from vision)
    api_key: str = ""
    api_url: str = ""
    temperature: float = 0.7  # Generation temperature
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def provider_type(self) -> str:
        return "local" if self.provider.lower() in LOCAL_PROVIDERS else "api"

    def get_text_model(self) -> str:
        """Get text model, falling back to vision model if not set."""
        return self.text_model if self.text_model else self.model


# =============================================================================
# CV Analyzer (Standalone version)
# =============================================================================

class CVAnalyzerStandalone:
    """
    Standalone CV analyzer that doesn't depend on ComfyUI.
    Uses YOLO for person detection and MediaPipe for face detection.
    """

    _yolo_model = None
    _mediapipe_face = None

    PERSON_CLASS = 0

    SHOT_THRESHOLDS = {
        "extreme_close_up": 0.7,
        "close_up": 0.5,
        "medium_close_up": 0.35,
        "medium_shot": 0.2,
        "medium_full": 0.12,
        "full_shot": 0.05,
        "wide_shot": 0.0,
    }

    @classmethod
    def _get_yolo_model(cls):
        """Lazy load YOLO model."""
        if cls._yolo_model is None:
            try:
                from ultralytics import YOLO
                cls._yolo_model = YOLO("yolov8n.pt")
            except ImportError:
                return None
            except Exception as e:
                return None
        return cls._yolo_model

    @classmethod
    def _get_mediapipe_face(cls):
        """MediaPipe disabled - YOLO handles all detection."""
        # MediaPipe has protobuf compatibility issues on Windows that can't be suppressed
        # YOLO provides sufficient person detection, so MediaPipe is disabled
        return None

    @classmethod
    def detect_humans(cls, image: np.ndarray) -> Dict[str, Any]:
        """Fast human detection using YOLO and MediaPipe."""
        start_time = time.time()

        result = {
            "has_human": False,
            "human_count": 0,
            "person_boxes": [],
            "face_boxes": [],
            "detection_time_ms": 0
        }

        h, w = image.shape[:2]

        # Try YOLO first
        yolo = cls._get_yolo_model()
        if yolo is not None:
            try:
                yolo_results = yolo(image, classes=[cls.PERSON_CLASS], verbose=False)
                for r in yolo_results:
                    for box in r.boxes:
                        if float(box.conf) > 0.5:
                            xyxy = box.xyxy[0].cpu().numpy()
                            result["person_boxes"].append({
                                "bbox": xyxy.tolist(),
                                "confidence": float(box.conf),
                                "center": [(xyxy[0] + xyxy[2]) / 2 / w, (xyxy[1] + xyxy[3]) / 2 / h]
                            })
            except Exception as e:
                print(f"[CV-Analyzer] YOLO detection error: {e}")

        # Try MediaPipe face detection
        face_detector = cls._get_mediapipe_face()
        if face_detector is not None:
            try:
                if image.shape[2] == 4:
                    image_rgb = image[:, :, :3]
                else:
                    image_rgb = image

                mp_results = face_detector.process(image_rgb)
                if mp_results.detections:
                    for detection in mp_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        result["face_boxes"].append({
                            "bbox_relative": [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
                            "confidence": detection.score[0] if detection.score else 0.5,
                            "center": [bbox.xmin + bbox.width/2, bbox.ymin + bbox.height/2]
                        })
            except Exception as e:
                print(f"[CV-Analyzer] MediaPipe detection error: {e}")

        person_count = len(result["person_boxes"])
        face_count = len(result["face_boxes"])

        result["has_human"] = person_count > 0 or face_count > 0
        result["human_count"] = max(person_count, face_count)
        result["detection_time_ms"] = (time.time() - start_time) * 1000

        return result

    @classmethod
    def extract_features(cls, image: np.ndarray, human_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visual features from image."""
        import cv2

        start_time = time.time()
        h, w = image.shape[:2]

        features = {
            "colors": cls._analyze_colors(image),
            "composition": cls._analyze_composition(image, human_detection),
            "shot_type": cls._estimate_shot_type(image, human_detection),
            "lighting": cls._analyze_lighting(image),
            "aspect_ratio": cls._get_aspect_ratio(w, h),
            "feature_time_ms": 0
        }

        features["feature_time_ms"] = (time.time() - start_time) * 1000
        return features

    @classmethod
    def _analyze_colors(cls, image: np.ndarray) -> Dict[str, Any]:
        """Extract dominant colors and temperature."""
        import cv2

        small = cv2.resize(image, (100, 100))
        pixels = small.reshape(-1, 3).astype(np.float32)

        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
        except ImportError:
            colors = [
                pixels.mean(axis=0).astype(int),
                small[0, 0],
                small[0, -1],
                small[-1, 0],
                small[-1, -1]
            ]

        color_names = [cls._rgb_to_name(c) for c in colors]

        avg_color = np.mean(pixels, axis=0)
        r, g, b = avg_color
        if r > b + 20:
            temperature = "warm"
        elif b > r + 20:
            temperature = "cool"
        else:
            temperature = "neutral"

        return {
            "dominant": list(dict.fromkeys(color_names))[:5],
            "temperature": temperature
        }

    @classmethod
    def _rgb_to_name(cls, rgb) -> str:
        """Convert RGB to approximate color name."""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

        if abs(r - g) < 20 and abs(g - b) < 20:
            if max(r, g, b) > 200:
                return "white"
            elif max(r, g, b) < 50:
                return "black"
            else:
                return "gray"

        if r > g and r > b:
            if r > 200 and g > 150:
                return "orange" if g < 200 else "yellow"
            elif r > 150 and b > 100:
                return "pink" if b > 150 else "red"
            return "red" if r > 150 else "brown"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            if r > 150 and b > 150:
                return "purple"
            return "blue"

        if r > 180 and g > 180:
            return "yellow"
        if r > 150 and g > 100 and b < 100:
            return "orange"
        if r > 100 and g < 80 and b > 100:
            return "purple"
        if g > 150 and b > 150:
            return "cyan"

        return "tan" if r > g > b else "gray"

    @classmethod
    def _analyze_composition(cls, image: np.ndarray, human_detection: Dict[str, Any]) -> Dict[str, str]:
        """Analyze image composition."""
        composition_type = "unknown"

        if human_detection["has_human"]:
            centers = []
            for face in human_detection.get("face_boxes", []):
                centers.append(face["center"])
            for person in human_detection.get("person_boxes", []):
                centers.append(person["center"])

            if centers:
                avg_x = sum(c[0] for c in centers) / len(centers)
                avg_y = sum(c[1] for c in centers) / len(centers)

                if 0.33 < avg_x < 0.66 and 0.25 < avg_y < 0.75:
                    composition_type = "centered"
                elif (0.25 < avg_x < 0.4) or (0.6 < avg_x < 0.75):
                    composition_type = "rule_of_thirds"
                else:
                    composition_type = "off_center"
        else:
            composition_type = "scenic"

        return {"type": composition_type}

    @classmethod
    def _estimate_shot_type(cls, image: np.ndarray, human_detection: Dict[str, Any]) -> str:
        """Estimate shot type based on face/body size."""
        h, w = image.shape[:2]

        if not human_detection["has_human"]:
            return "scenic"

        if human_detection["face_boxes"]:
            max_face_height = 0
            for face in human_detection["face_boxes"]:
                bbox = face["bbox_relative"]
                face_h = bbox[3]
                max_face_height = max(max_face_height, face_h)

            for shot_type, threshold in cls.SHOT_THRESHOLDS.items():
                if max_face_height >= threshold:
                    return shot_type

        if human_detection["person_boxes"]:
            max_body_height = 0
            for person in human_detection["person_boxes"]:
                bbox = person["bbox"]
                body_h = (bbox[3] - bbox[1]) / h
                max_body_height = max(max_body_height, body_h)

            if max_body_height > 0.8:
                return "full_shot"
            elif max_body_height > 0.5:
                return "medium_full"
            elif max_body_height > 0.3:
                return "medium_shot"
            else:
                return "wide_shot"

        return "medium_shot"

    @classmethod
    def _analyze_lighting(cls, image: np.ndarray) -> Dict[str, str]:
        """Analyze image lighting."""
        import cv2

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)

        if mean_brightness < 60:
            brightness = "dark"
        elif mean_brightness < 120:
            brightness = "low"
        elif mean_brightness < 180:
            brightness = "normal"
        elif mean_brightness < 220:
            brightness = "bright"
        else:
            brightness = "overexposed"

        if std_contrast < 30:
            contrast = "low"
        elif std_contrast < 60:
            contrast = "medium"
        else:
            contrast = "high"

        return {"brightness": brightness, "contrast": contrast}

    @classmethod
    def _get_aspect_ratio(cls, width: int, height: int) -> str:
        """Determine aspect ratio category."""
        ratio = width / height

        if ratio > 1.6:
            return "ultrawide"
        elif ratio > 1.2:
            return "landscape"
        elif ratio > 0.85:
            return "square"
        elif ratio > 0.6:
            return "portrait"
        else:
            return "tall"

    @classmethod
    def analyze(cls, image: np.ndarray, mode: str = "standard") -> Dict[str, Any]:
        """Complete image analysis."""
        result = {
            "mode": mode,
            "has_human": False,
            "human_count": 0,
            "prompt_type": "scene_only",
            "detection_time_ms": 0,
            "feature_time_ms": 0,
            "total_time_ms": 0
        }

        start_time = time.time()

        detection = cls.detect_humans(image)
        result.update({
            "has_human": detection["has_human"],
            "human_count": detection["human_count"],
            "person_boxes": detection["person_boxes"],
            "face_boxes": detection["face_boxes"],
            "detection_time_ms": detection["detection_time_ms"],
            "prompt_type": "portrait" if detection["has_human"] else "scene_only"
        })

        if mode in ["standard", "detailed"]:
            features = cls.extract_features(image, detection)
            result.update({
                "colors": features["colors"],
                "composition": features["composition"],
                "shot_type": features["shot_type"],
                "lighting": features["lighting"],
                "aspect_ratio": features["aspect_ratio"],
                "feature_time_ms": features["feature_time_ms"]
            })

        result["total_time_ms"] = (time.time() - start_time) * 1000
        return result


# =============================================================================
# Helper Functions
# =============================================================================

def load_image(path: Union[str, Path]) -> Image.Image:
    """Load image from path."""
    return Image.open(path).convert('RGB')


def auto_resize_image(pil_image: Image.Image, target_size: int, quality: int) -> Tuple[str, Dict, Image.Image]:
    """Auto-resize image for optimal processing."""
    original_w, original_h = pil_image.size
    original_pixels = original_w * original_h

    if max(original_w, original_h) > target_size:
        if original_w > original_h:
            new_w = target_size
            new_h = int(original_h * target_size / original_w)
        else:
            new_h = target_size
            new_w = int(original_w * target_size / original_h)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w, new_h = original_w, original_h

    if pil_image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        if pil_image.mode in ('RGBA', 'LA'):
            background.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = background
        else:
            pil_image = pil_image.convert('RGB')
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)

    base64_str = base64.b64encode(buffer.read()).decode('utf-8')
    file_size = len(base64_str) * 3 // 4

    info = {
        "original_size": (original_w, original_h),
        "compressed_size": (new_w, new_h),
        "file_size_kb": file_size // 1024,
        "quality": quality,
        "compression_ratio": round(original_pixels / (new_w * new_h), 1)
    }

    return base64_str, info, pil_image


# Hallucination detection patterns - philosophical/spiritual/business nonsense
HALLUCINATION_PHRASES = [
    # Spiritual/philosophical rambling
    r'\b(cycle\s+of\s+)?life\s+(and\s+)?death\s+(and\s+)?rebirth\b',
    r'\bnirvana\s+state\b', r'\bsupreme\s+bliss\b', r'\beternal\s+(sleep|rest|peace)\b',
    r'\bsalvation\s+attained\b', r'\bredemption\s+won\b', r'\benlightenment\s+reached\b',
    r'\bspiritual\s+ascent\b', r'\bdivine\s+grace\b', r'\bsacred\s+sanctuary\b',
    r'\bcollective\s+unconscious\b', r'\buniversal\s+themes?\b', r'\btranscending\s+cultural\b',
    r'\bcosmic\s+grandeur\b', r'\bprimal\s+instinctive\b', r'\bmythological\s+tale\b',
    r'\beternity\s+awaits\b', r'\bfinal\s+rest\b', r'\bsoul\s+quietly\b',
    r'\bhumanity\s+relationship\s+universe\b', r'\bsilent\s+witness\b',
    r'\bawoken\s+moments\s+later\b', r'\bdawn\s+breaks\s+reveal\b',
    r'\blegend\s+whispered\b', r'\beverlasting\s+epic\s+saga\b',
    r'\bparchment\s+ink\s+stained\b', r'\bblood\s+sweat\s+tears\b',
    r'\bbattlefield\s+wars\s+fought\b', r'\bhonor\s+earned\s+courage\b',
    r'\bangel\s+fallen\s+heaven\b', r'\bheaven\s+gates\s+opened\b',
    r'\bthreshold\s+crossed\s+realm\b', r'\bhigher\s+planes\b',
    # Business/corporate jargon
    r'\bmarket\s+share\b', r'\bcustomer\s+base\b', r'\bproductivity\s+boosted\b',
    r'\bROI\b', r'\bstakeholders?\b', r'\bsynergy\b', r'\bleverage\b',
    r'\bscalable\b', r'\bactionable\b', r'\bbenchmarks?\b', r'\bmetrics?\b',
    r'\bKPIs?\b', r'\bpipeline\b', r'\bdeliverables?\b',
    # Abstract/narrative nonsense
    r'\bnarrative\s+potential\b', r'\bnonverbal\s+language\b', r'\bvisuals\s+alone\s+suffice\b',
    r'\bpowerful\s+message\s+about\b', r'\bconveying\s+mystery\s+ambiguity\b',
    r'\bemotional\s+resonance\b', r'\bunderlying\s+imagery\b',
    r'\bechoes?\s+timeless\b', r'\bimmortal\s+spirit\b',
    # Repetitive filler
    r'\b(overcome|overcoming)\s+(adversity|obstacles)\b',
    r'\b(strength|resolve)\s+emerged?\s+victorious\b',
    r'\bdestiny\s+realized\b', r'\bpurpose\s+fulfilled\b',
    r'\bjourney\s+continue\s+onward\b', r'\bpath\s+unknown\b',
    r'\bembrace\s+(final|ultimate)\b', r'\bawait(s|ing)\s+(discovery|embrace)\b',
]

# Compile patterns for efficiency
HALLUCINATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PHRASES]


def detect_and_fix_hallucination(prompt: str, max_words: int = 500) -> Tuple[str, bool]:
    """
    Detect and fix LLM hallucination/runaway text.

    Args:
        prompt: Raw LLM output
        max_words: Maximum allowed word count

    Returns:
        Tuple of (cleaned_prompt, was_hallucinating)
    """
    if not prompt:
        return "", False

    original_length = len(prompt.split())
    was_hallucinating = False

    # Check 1: Detect hallucination phrases
    hallucination_count = 0
    for pattern in HALLUCINATION_PATTERNS:
        matches = pattern.findall(prompt)
        hallucination_count += len(matches)

    # If 3+ hallucination phrases detected, it's rambling
    if hallucination_count >= 3:
        was_hallucinating = True
        print(f"[PromptGen] WARNING: Detected {hallucination_count} hallucination phrases - cleaning up")

        # Remove hallucination phrases
        for pattern in HALLUCINATION_PATTERNS:
            prompt = pattern.sub('', prompt)

    # Check 2: Detect repetition (same 5+ word phrase appearing twice)
    words = prompt.split()
    if len(words) > 100:
        # Look for repeated 5-word sequences
        for i in range(len(words) - 10):
            phrase = ' '.join(words[i:i+5]).lower()
            rest = ' '.join(words[i+10:]).lower()
            if phrase in rest:
                # Found repetition - truncate at first occurrence
                print(f"[PromptGen] WARNING: Detected repetitive text - truncating")
                was_hallucinating = True
                # Keep only up to slightly after first occurrence
                prompt = ' '.join(words[:i+20])
                break

    # Check 3: Hard word limit - but truncate gracefully at sentence boundaries
    words = prompt.split()
    if len(words) > max_words:
        print(f"[PromptGen] WARNING: Output too long ({len(words)} words) - truncating to ~{max_words}")
        was_hallucinating = True

        # Look for a good cutoff point - allow going up to 20% over limit to find sentence end
        search_limit = int(max_words * 1.2)
        truncated = ' '.join(words[:min(search_limit, len(words))])

        # Find sentence boundaries (. ; or , followed by space and capital or descriptive transition)
        # Priority: period > semicolon > comma
        best_cutoff = -1

        # Look for last period within acceptable range
        for i in range(len(truncated) - 1, int(len(truncated) * 0.6), -1):
            if truncated[i] == '.' and (i == len(truncated) - 1 or truncated[i+1] == ' '):
                best_cutoff = i + 1
                break

        # If no period, try semicolon
        if best_cutoff == -1:
            for i in range(len(truncated) - 1, int(len(truncated) * 0.6), -1):
                if truncated[i] == ';':
                    best_cutoff = i + 1
                    break

        # If still no good boundary, try comma (less ideal but better than mid-word)
        if best_cutoff == -1:
            for i in range(len(truncated) - 1, int(len(truncated) * 0.7), -1):
                if truncated[i] == ',':
                    best_cutoff = i + 1
                    break

        if best_cutoff > 0:
            prompt = truncated[:best_cutoff].strip()
        else:
            # Last resort: just use the word limit
            prompt = ' '.join(words[:max_words])

    # Clean up whitespace artifacts
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'\s*,\s*,+\s*', ', ', prompt)
    prompt = re.sub(r'^\s*,\s*', '', prompt)
    prompt = re.sub(r'\s*,\s*$', '', prompt)

    if was_hallucinating:
        final_length = len(prompt.split())
        print(f"[PromptGen] Cleaned: {original_length} -> {final_length} words")

    return prompt.strip(), was_hallucinating


def zimage_clean(prompt: str, provider: str = "default", max_words: int = 500) -> str:
    """Apply Z-Image cleanup to prompt using config patterns + built-in rules."""
    if not prompt:
        return ""

    # First, detect and fix hallucination/runaway text
    prompt, was_hallucinating = detect_and_fix_hallucination(prompt, max_words)

    # Then apply config-based cleanup (example patterns, markdown, provider-specific)
    prompt = clean_llm_output(prompt, provider)

    # Then apply Z-Image specific cleanup (meta-tags, abstract words)
    for pattern in ZIMAGE_META_TAGS:
        prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)

    for pattern in ZIMAGE_ABSTRACT_WORDS:
        prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)

    # Whitespace cleanup
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'\s*,\s*,\s*', ', ', prompt)
    prompt = re.sub(r'^\s*,\s*', '', prompt)
    prompt = re.sub(r'\s*,\s*$', '', prompt)

    # Ensure safety suffix
    prompt = prompt.rstrip('.,; ')
    if "no text" not in prompt.lower() or "no watermark" not in prompt.lower():
        prompt += ", no text, no watermark"

    return prompt.strip()


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Unified LLM client for various providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy initialize client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """Create appropriate client based on provider."""
        provider = self.config.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=300.0, connect=30.0)
            return anthropic.Anthropic(api_key=self.config.api_key, timeout=timeout)
        elif provider == "local":
            # Local models use LocalModelClient from sid_llm_local
            # This requires ComfyUI context - import from llm_providers
            try:
                from .llm_providers.sid_llm_local import LocalModelClient
            except ImportError:
                # Standalone mode - try direct import
                from llm_providers.sid_llm_local import LocalModelClient

            # Extract config from extra_params (passed from LLMModelConfig)
            extra = getattr(self.config, 'extra_params', {})
            # Get temperature from config (default 0.7)
            temp = getattr(self.config, 'temperature', 0.7)
            return LocalModelClient(
                model_name=self.config.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                attention_mode=extra.get("attention_mode", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                temperature=temp,  # Pass user's temperature setting
                top_p=extra.get("top_p", 0.9),
                repetition_penalty=extra.get("repetition_penalty", 1.2),
                num_beams=extra.get("num_beams", 1),
                use_torch_compile=extra.get("use_torch_compile", False),
            )
        else:
            from openai import OpenAI
            import httpx
            timeout = httpx.Timeout(timeout=120.0, connect=30.0)
            return OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.api_url if self.config.api_url else None,
                timeout=timeout
            )

    def call_vision(self, base64_image: str, system_prompt: str,
                    user_prompt: str, max_tokens: int = 1000) -> str:
        """Call vision LLM with image."""
        provider = self.config.provider.lower()

        try:
            # Get configured temperature (default 0.7)
            temp = getattr(self.config, 'temperature', 0.7)

            if provider == "anthropic":
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                            {"type": "text", "text": user_prompt}
                        ]
                    }]
                )
                return response.content[0].text.strip()
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": user_prompt}
                    ]
                })

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    messages=messages
                )
                result = response.choices[0].message.content or ""
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL | re.IGNORECASE)
                return result.strip()

        except Exception as e:
            print(f"[LLM] Vision call failed: {e}")
            return ""

    def call_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        """Call text-only LLM using text_model if available."""
        provider = self.config.provider.lower()
        text_model = self.config.get_text_model()

        try:
            if provider == "local":
                # For local models, use LocalModelClient with text model
                try:
                    from .llm_providers.sid_llm_local import LocalModelClient
                except ImportError:
                    from llm_providers.sid_llm_local import LocalModelClient

                extra = getattr(self.config, 'extra_params', {})
                temp = getattr(self.config, 'temperature', 0.7)
                text_client = LocalModelClient(
                    model_name=text_model,
                    quantization=extra.get("quantization", "4-bit"),
                    device=extra.get("device", "auto"),
                    attention_mode=extra.get("attention_mode", "auto"),
                    keep_model_loaded=extra.get("keep_model_loaded", True),
                    temperature=temp,
                )
                result = text_client.generate_text(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temp)
                return result.strip() if result else ""

            elif provider == "anthropic":
                temp = getattr(self.config, 'temperature', 0.7)
                response = self.client.messages.create(
                    model=text_model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text.strip()
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                temp = getattr(self.config, 'temperature', 0.7)
                response = self.client.chat.completions.create(
                    model=text_model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    messages=messages
                )
                result = response.choices[0].message.content or ""
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL | re.IGNORECASE)
                return result.strip()

        except Exception as e:
            print(f"[LLM] Text call failed: {e}")
            return ""


# =============================================================================
# Main Prompt Generator
# =============================================================================

@dataclass
class GenerationResult:
    """Result of prompt generation."""
    prompt: str = ""
    negative: str = ""
    caption: str = ""
    cv_analysis: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative": self.negative,
            "caption": self.caption,
            "cv_analysis": self.cv_analysis,
            "timing": self.timing,
            "metadata": self.metadata,
            "logs": self.logs
        }

    def get_metadata_str(self) -> str:
        """Format metadata and CV analysis as readable string."""
        lines = []
        lines.append("=== Image Metadata ===")

        # Original/compressed size
        if "original_size" in self.metadata:
            lines.append(f"Original Size: {self.metadata['original_size']}")
        if "compressed_size" in self.metadata:
            lines.append(f"Compressed Size: {self.metadata['compressed_size']}")
        if "file_size_kb" in self.metadata:
            lines.append(f"File Size: {self.metadata['file_size_kb']:.1f} KB")
        if "target_size" in self.metadata:
            lines.append(f"Target Size: {self.metadata['target_size']}px")
        if "mode_scale" in self.metadata:
            lines.append(f"Mode Scale: {self.metadata['mode_scale']:.0%}")
        if "word_count" in self.metadata:
            lines.append(f"Word Count: {self.metadata['word_count']}")

        # CV Analysis
        if self.cv_analysis:
            lines.append("")
            lines.append("=== CV Analysis ===")
            lines.append(f"Has Human: {self.cv_analysis.get('has_human', False)}")
            lines.append(f"Human Count: {self.cv_analysis.get('human_count', 0)}")
            if self.cv_analysis.get('shot_type'):
                lines.append(f"Shot Type: {self.cv_analysis.get('shot_type')}")
            if self.cv_analysis.get('composition'):
                comp = self.cv_analysis.get('composition', {})
                lines.append(f"Composition: {comp.get('type', 'unknown')}")
            if self.cv_analysis.get('lighting'):
                light = self.cv_analysis.get('lighting', {})
                lines.append(f"Lighting: {light.get('brightness', '?')} / {light.get('contrast', '?')}")
            if self.cv_analysis.get('colors'):
                colors = self.cv_analysis.get('colors', {})
                lines.append(f"Color Temp: {colors.get('temperature', '?')}")
                if colors.get('dominant'):
                    lines.append(f"Dominant Colors: {', '.join(colors['dominant'][:5])}")

        # Timing
        if self.timing:
            lines.append("")
            lines.append("=== Timing Summary ===")
            if "cv_detection_ms" in self.timing:
                lines.append(f"CV Detection: {self.timing['cv_detection_ms']:.0f}ms")
            if "llm_time_s" in self.timing:
                lines.append(f"LLM Generation: {self.timing['llm_time_s']:.1f}s")

            # Component breakdown if agentic mode was used
            if "components" in self.timing and self.timing["components"]:
                lines.append("")
                lines.append("─── Component Breakdown ───")
                components = self.timing["components"]
                total_comp = sum(components.values())
                for name, elapsed in components.items():
                    pct = (elapsed / total_comp * 100) if total_comp > 0 else 0
                    lines.append(f"  {name}: {elapsed:.2f}s ({pct:.0f}%)")
                lines.append(f"  ─────────────────────")
                lines.append(f"  Components Total: {total_comp:.2f}s")

            if "total_time_s" in self.timing:
                lines.append("")
                lines.append(f"═══ TOTAL TIME: {self.timing['total_time_s']:.1f}s ═══")

        return "\n".join(lines)

    def get_debug_log(self) -> str:
        """Get collected debug logs as string."""
        return "\n".join(self.logs) if self.logs else "(No logs captured)"


class PromptGenerator:
    """
    Main prompt generator class.
    Can run standalone or be used by ComfyUI wrapper.
    """

    # Mode-based image size scaling factors
    MODE_SIZE_SCALE = {
        "quick": 0.70,      # 30% smaller for faster processing
        "standard": 1.00,   # Base size
        "detailed": 1.30,   # 30% larger for more detail
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llava:latest",
        text_model: str = "",
        api_key: str = "",
        api_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        analysis_mode: str = "standard",
        enable_reasoning: bool = False,
        prompt_style: str = "expanded",
        prompt_length: int = 150,
        generate_negative: bool = False,
        generate_caption: bool = False,
        verbose: bool = True,
        extra_params: Dict[str, Any] = None
    ):
        self.config = LLMConfig(
            provider=provider,
            model=model,
            text_model=text_model,
            api_key=api_key,
            api_url=api_url,
            temperature=temperature,
            extra_params=extra_params or {}
        )
        self.llm_client = LLMClient(self.config)
        self.analysis_mode = analysis_mode.lower()
        self.enable_reasoning = enable_reasoning
        self.prompt_style = prompt_style.lower()  # "expanded" or "tags"
        self.prompt_length = prompt_length
        self.generate_negative = generate_negative
        self.generate_caption = generate_caption
        self.verbose = verbose

        # Get image sizes based on provider
        self.sizes = IMAGE_SIZES[self.config.provider_type]

        # Get model-specific target image size from extra_params (for local models)
        ep = extra_params or {}
        self.model_target_size = ep.get("target_image_size", self.sizes["llm"])

        # Log buffer for capturing debug output
        self._log_buffer: List[str] = []

    def _log(self, msg: str):
        """Print if verbose mode and capture to buffer."""
        # Always capture to buffer
        self._log_buffer.append(msg)
        # Print if verbose
        if self.verbose:
            print(msg)

    def _get_logs(self) -> List[str]:
        """Get and clear log buffer."""
        logs = self._log_buffer.copy()
        self._log_buffer.clear()
        return logs

    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        user_guidance: str = ""
    ) -> GenerationResult:
        """
        Process a single image and generate prompt.

        Args:
            image: Image path, PIL Image, or numpy array
            user_guidance: Optional user instructions

        Returns:
            GenerationResult with prompt, negative, caption
        """
        start_time = time.time()
        result = GenerationResult()

        reasoning_str = "ON" if self.enable_reasoning else "OFF"
        self._log(f"\n[PromptGen] Processing {self.config.provider} | Mode: {self.analysis_mode} | Style: {self.prompt_style} | Reasoning: {reasoning_str}")

        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = str(image)
            result.metadata["source_path"] = image_path
            pil_image = load_image(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        original_size = pil_image.size
        result.metadata["original_size"] = original_size

        # Step 1: CV Detection (YOLO + MediaPipe)
        self._log(f"[PromptGen] Step 1: Running CV detection (YOLO + MediaPipe)...")
        cv_start = time.time()

        cv_image = pil_image.copy()
        if max(cv_image.size) > self.sizes["cv"]:
            cv_image.thumbnail((self.sizes["cv"], self.sizes["cv"]), Image.LANCZOS)
        cv_np = np.array(cv_image.convert('RGB'))

        cv_analysis = CVAnalyzerStandalone.analyze(cv_np, mode=self.analysis_mode)
        result.cv_analysis = cv_analysis
        result.timing["cv_detection_ms"] = (time.time() - cv_start) * 1000

        has_human = cv_analysis["has_human"]
        human_str = 'Yes' if has_human else 'No'
        self._log(f"[PromptGen] CV done in {result.timing['cv_detection_ms']:.0f}ms | Image: {original_size[0]}x{original_size[1]} | Human: {human_str}")

        # Step 2: Prepare image for LLM with mode-based sizing
        self._log(f"[PromptGen] Step 2: Preparing image for LLM...")
        mode_scale = self.MODE_SIZE_SCALE.get(self.analysis_mode, 1.0)
        target_llm_size = int(self.model_target_size * mode_scale)

        # Ensure minimum size of 224 and cap at provider limit
        provider_limit = get_image_limit(self.config.provider)
        target_llm_size = max(224, min(target_llm_size, provider_limit))

        base64_image, resize_info, _ = auto_resize_image(
            pil_image, target_llm_size, self.sizes["quality"]
        )
        result.metadata["compressed_size"] = resize_info["compressed_size"]
        result.metadata["file_size_kb"] = resize_info["file_size_kb"]
        result.metadata["target_size"] = target_llm_size
        result.metadata["mode_scale"] = mode_scale

        # Log compressed size
        compressed_w, compressed_h = resize_info["compressed_size"]
        self._log(f"[PromptGen] Compressed: {compressed_w}x{compressed_h} ({resize_info['file_size_kb']:.0f}KB)")

        # Step 3: Build prompts
        self._log(f"[PromptGen] Step 3: Building prompts...")
        system_prompt, user_prompt = self._build_prompts(has_human, cv_analysis, user_guidance)

        # Step 4: Call Vision LLM
        self._log(f"[PromptGen] Step 4: Calling Vision LLM ({self.config.provider})...")
        llm_start = time.time()

        is_local = self.config.provider_type == "local"
        # Agentic multi-step ONLY when:
        # 1. Not a local provider (API providers only)
        # 2. Analysis mode is "detailed"
        # 3. Reasoning is enabled (supports_reasoning=True in LLM config)
        #
        # This means:
        # - Standard mode: Single API call (ignores reasoning setting)
        # - Detailed + reasoning OFF: Single API call with detailed prompts
        # - Detailed + reasoning ON: Multi-step agentic analysis
        use_agentic = not is_local and self.analysis_mode == "detailed" and self.enable_reasoning

        if use_agentic:
            self._log(f"[PromptGen] Using AGENTIC multi-step analysis (detailed + reasoning ON)")
            prompt = self._agentic_generate(base64_image, has_human, cv_analysis, user_guidance)
        else:
            prompt = self.llm_client.call_vision(
                base64_image, system_prompt, user_prompt,
                max_tokens=self.prompt_length * 3
            )

        result.timing["llm_time_s"] = time.time() - llm_start

        # Capture agentic component times if available
        if use_agentic and hasattr(self, '_last_component_times'):
            result.timing["components"] = self._last_component_times.copy()

        # Step 5: Z-Image cleanup (uses config patterns + built-in rules)
        # Use prompt_length + 50% buffer as max words to allow some flexibility
        max_words = int(self.prompt_length * 1.5)
        result.prompt = zimage_clean(prompt, self.config.provider, max_words)

        # Step 5b: Add Z-Image realism enhancers for portraits (only for expanded mode)
        if has_human and self.prompt_style == "expanded":
            realism_tags = "realistic skin texture, natural skin pores, photorealistic, ultra detailed"
            result.prompt = f"{result.prompt}, {realism_tags}"

        # Step 5c: Convert to tags if requested
        if self.prompt_style == "tags":
            self._log(f"[PromptGen] Converting to tag style with vocabulary enhancement...")
            vocabulary = get_vocabulary_terms()
            result.prompt = convert_to_tags(result.prompt, vocabulary)
            self._log(f"[PromptGen] Generated {len(result.prompt.split(','))} tags")

        # Step 5d: Apply user guidance
        if user_guidance and user_guidance.strip():
            user_guidance = user_guidance.strip()
            if self.analysis_mode == "quick":
                # Quick mode: append user guidance verbatim
                self._log(f"[PromptGen] Adding user guidance verbatim")
                result.prompt = f"{result.prompt}, {user_guidance}"
            else:
                # Standard/Detailed mode: use text LLM to enhance prompt based on guidance
                self._log(f"[PromptGen] Enhancing prompt with user guidance via LLM...")
                enhance_prompt = f"""Modify this image generation prompt based on the user's guidance.

CURRENT PROMPT:
{result.prompt}

USER GUIDANCE:
{user_guidance}

INSTRUCTIONS:
- Integrate the user's guidance into the prompt naturally
- Keep the descriptive style and technical details
- Do not add explanations, just output the enhanced prompt
- Maintain the same format and length

OUTPUT the enhanced prompt only:"""

                enhanced = self.llm_client.call_text(
                    "You enhance image generation prompts based on user guidance.",
                    enhance_prompt,
                    max_tokens=self.prompt_length * 3
                )
                if enhanced and enhanced.strip():
                    result.prompt = zimage_clean(enhanced, self.config.provider, max_words)
                    # Re-add realism tags if human
                    if has_human:
                        result.prompt = f"{result.prompt}, {realism_tags}"

        result.metadata["word_count"] = len(result.prompt.split())

        # Step 6: Generate negative (optional)
        if self.generate_negative:
            result.negative = self._get_negative(has_human)

        # Step 7: Generate caption (optional)
        if self.generate_caption:
            result.caption = self._get_caption(result.prompt)

        result.timing["total_time_s"] = time.time() - start_time

        # Final output
        self._log(f"[PromptGen] Done in {result.timing['total_time_s']:.1f}s ({result.metadata['word_count']} words)")
        self._log(f"[PromptGen] ───────────────────────────────────────────")
        self._log(f"[PromptGen] PROMPT: {result.prompt[:200]}..." if len(result.prompt) > 200 else f"[PromptGen] PROMPT: {result.prompt}")
        self._log(f"[PromptGen] ───────────────────────────────────────────\n")

        # Capture logs to result
        result.logs = self._get_logs()

        return result

    def _build_prompts(self, has_human: bool, cv_analysis: Dict, user_guidance: str) -> Tuple[str, str]:
        """Build system and user prompts from TOML config or fallback defaults."""
        tier = get_provider_tier(self.config.provider)
        length_constraint = get_length_constraint(self.prompt_length)

        # Build CV hints for context
        cv_hints = ""
        if self.analysis_mode != "quick" and "colors" in cv_analysis:
            cv_hints = f"""
CV ANALYSIS (use as guidance):
- Dominant colors: {', '.join(cv_analysis['colors']['dominant'][:5])}
- Color temperature: {cv_analysis['colors']['temperature']}
- Shot type: {cv_analysis.get('shot_type', 'unknown')}
- Composition: {cv_analysis.get('composition', {}).get('type', 'unknown')}
- Lighting: {cv_analysis.get('lighting', {}).get('brightness', 'unknown')} brightness
"""

        if has_human:
            # Try to get from TOML config
            system_base = get_system_prompt(tier)
            user_base = get_user_prompt(tier, self.analysis_mode)

            # Build system prompt
            if system_base and system_base != ZIMAGE_SYSTEM_RULES:
                # Use TOML config
                system_prompt = system_base
                if length_constraint:
                    system_prompt += f"\n\n{length_constraint}"
            else:
                # Fallback to built-in
                system_prompt = f"""You are an expert at writing prompts for AI image generation.
Describe this image of a person for AI image recreation.

{ZIMAGE_SYSTEM_RULES}

{length_constraint}"""

            # Build user prompt
            if user_base:
                # Use TOML config
                user_prompt = user_base
                if cv_hints:
                    user_prompt = f"{cv_hints}\n\n{user_prompt}"
            else:
                # Fallback to built-in
                user_prompt = f"""Describe this image in detail for AI image recreation.
{cv_hints}

CRITICAL - Describe these in DETAIL (in order):
1. ETHNICITY/ANCESTRY: MUST identify FIRST - be SPECIFIC: East Asian (Chinese/Japanese/Korean/Vietnamese), South Asian (Indian/Pakistani/Bengali), Southeast Asian (Thai/Filipino/Indonesian), Middle Eastern, African, African American, Latino/Hispanic, Caucasian/European (specify region if possible), Mixed heritage. Note facial features that indicate ethnicity.
2. SHOT TYPE/FRAMING (CRITICAL - state exactly what body parts are visible and where frame cuts off):
   - EXTREME CLOSE-UP: Face only (cropped at forehead/chin)
   - CLOSE-UP/HEADSHOT: Head and neck (cropped at shoulders)
   - MEDIUM CLOSE-UP: Head to chest (cropped at chest/bust)
   - MEDIUM SHOT: Head to waist (cropped at waist)
   - COWBOY SHOT: Head to mid-thigh (cropped at mid-thigh)
   - FULL BODY: Head to feet (entire body visible, feet visible)
   State: "cropped at [body part]" or "full body with feet visible"
3. SELFIE: Is this a selfie? Look for: arm extended, phone/camera angle, above eye level. If yes: state "selfie-style photo" and "eyes looking directly at camera".
4. EYES: exact color, iris pattern, shape (almond/round/hooded/monolid/double-lid), expression, where they're looking, catchlights
5. EYELASHES: color (black/brown/blonde), style (natural/long/thick/curled/sparse), mascara effect if visible
6. EYEBROWS: shape (arched/straight/rounded/angular), thickness, color, groomed or natural
7. SKIN: tone (fair/light/medium/olive/tan/brown/dark brown/deep), texture - freckles, dimples, wrinkles/fine lines, pores, moles
8. HAIR: exact style (length, layers, texture, parting, how it falls), color with any highlights/lowlights
9. MAKEUP: eyeshadow color/style, eyeliner, blush, foundation, contouring - if visible
10. LIPS: exact shade (nude/pink/red/burgundy/coral/mauve), finish (matte/glossy/satin), lip texture (smooth/chapped/dry lines), shape and fullness
11. LIGHTING SOURCE: WHERE does light come from (window/lamp/overhead/natural), direction, quality (soft/hard/diffused), shadows
12. COSTUME/UNIFORM: If wearing a recognizable uniform or costume, NAME IT FIRST (nun habit, nurse uniform, doctor coat, maid outfit, police uniform, schoolgirl uniform, etc.). Then note if traditional/modern/stylized/sexy variant. Don't over-describe - just identify it clearly.
13. CLOTHING CONSTRUCTION & LAYERS (CRITICAL):
    - COLOR: EXACT color of EACH piece (jet black, charcoal, cream white, ivory, navy, burgundy - be specific)
    - FABRIC: cotton, silk, satin, velvet, leather, latex, lace, mesh, sheer, denim, wool, knit/ribbed, jersey
    - TEXTURE: matte, shiny/glossy, smooth, ribbed, quilted, pleated
    - PATTERN: solid, striped, plaid, floral, polka dot, animal print
    - GARMENT TYPE: Is it fitted/form-fitting or loose/open? (e.g., "fitted crop top" NOT "open robe")
    - LAYERING: Describe what's under what WITH COLORS (e.g., "cream ribbed turtleneck UNDER fitted jet black crop top")
    - NECKLINE: crew/scoop/v-neck/turtleneck/off-shoulder/sweetheart/halter
    - HEM/LENGTH: cropped at midriff, waist-length, hip-length, full-length
    - SLEEVES: sleeveless/cap/short/3-4/long, tight or loose
    - FIT: skin-tight, fitted, semi-fitted, loose, oversized
    - CLOSURE: pullover (no opening) vs open-front (cardigan/robe style) vs buttoned vs zipped
14. CLOTHING STATE: buttons open/closed, collar position, sleeves rolled/unrolled, tucked/untucked
15. BEVERAGES/LIQUIDS: if visible, exact color - red wine (burgundy), white wine (pale gold), beer (amber), coffee (brown), water (clear)
16. ATMOSPHERIC EFFECTS: steam rising from cups/beverages, smoke, vapor, mist, fog, dust particles

Also describe:
17. Face: structure, expression, age estimate
18. BODY DEFINITION: if visible - muscle cuts (abs/midriff definition, chest, arms, shoulders), navel shape (round/vertical/horizontal, innie/outie), body fat percentage appearance
19. Body: pose, build, posture, visible body parts
20. FEET/LEGS: If visible, describe shoes/footwear, leg position. If full body, confirm feet are visible.
21. Clothing details: colors, style, fit, fabric texture (AFTER identifying costume/uniform type and construction)
22. Objects: items held or nearby with liquid colors and steam effects
23. Background: setting, colors, depth of field blur

Write as natural flowing language. START WITH: 1) ETHNICITY, 2) SHOT TYPE/FRAMING (state exactly where frame cuts off - "cropped at waist" or "full body with feet visible"), 3) selfie if applicable, then COSTUME/UNIFORM TYPE if applicable, then CLOTHING CONSTRUCTION (fitted vs open, layering, neckline, hem), then eyes, eyelashes, eyebrows, skin tone/texture, body definition, hair, makeup, lips, lighting, clothing state, beverages, and atmospheric effects."""

        else:
            # Scene-only prompts (no human)
            system_base = get_scene_system_prompt()
            user_base = get_scene_user_prompt(self.analysis_mode)

            # Build system prompt
            if system_base and system_base != ZIMAGE_SCENE_RULES:
                # Use TOML config
                system_prompt = system_base
                if length_constraint:
                    system_prompt += f"\n\n{length_constraint}"
            else:
                # Fallback to built-in
                system_prompt = f"""You are an expert at writing prompts for AI image generation.
Describe this scene for AI image recreation.

{ZIMAGE_SCENE_RULES}

{ZIMAGE_SYSTEM_RULES}

{length_constraint}"""

            # Build user prompt
            if user_base:
                # Use TOML config
                user_prompt = user_base
                if cv_hints:
                    user_prompt = f"{cv_hints}\n\n{user_prompt}"
            else:
                # Fallback to built-in
                user_prompt = f"""Describe this scene in detail.
{cv_hints}

CRITICAL - Describe in this order:
1. SHOT TYPE/FRAMING (MUST specify first):
   - EXTREME CLOSE-UP: Only face/head fills frame, no neck/body visible
   - CLOSE-UP: Head and partial neck
   - MEDIUM CLOSE-UP: Head and full neck, partial body
   - MEDIUM SHOT: Upper body visible
   - FULL BODY: Entire subject visible
   - WIDE SHOT: Subject small in frame, environment dominant

2. IF ANIMAL/WILDLIFE:
   - Species (be specific: "reticulated giraffe", "African lion", "Bengal tiger")
   - Age (adult, juvenile, calf)
   - Pose: front-facing/profile/three-quarter view
   - Eye contact: looking directly at camera, or looking away
   - Distinctive features: pattern details, markings, horns, tusks, mane, eyelashes

3. Main subject(s): what dominates the frame
4. Environment: setting, location type, depth of field blur
5. Colors: dominant palette, mood
6. Lighting: time of day (golden hour, midday, overcast), quality, direction
7. Atmospheric effects: steam, smoke, vapor, mist, fog, haze, dust particles

Remember: NO humans in this image.
START with shot type/framing, then species if animal, then details. Write as natural flowing language."""

        # Add user guidance if provided
        if user_guidance and user_guidance.strip():
            user_prompt += f"""

USER DIRECTION (apply this):
"{user_guidance.strip()}"
"""

        return system_prompt, user_prompt

    def _agentic_generate(self, base64_image: str, has_human: bool,
                          cv_analysis: Dict, user_guidance: str) -> str:
        """Agentic multi-step generation."""
        components = {}
        component_times = {}  # Track timing for each component

        def timed_vision_call(name: str, *args, **kwargs):
            """Wrapper to time individual vision calls."""
            start = time.time()
            result = self.llm_client.call_vision(*args, **kwargs)
            elapsed = time.time() - start
            component_times[name] = elapsed
            self._log(f"[Agentic] {name}: {elapsed:.2f}s")
            return result

        if has_human:
            # ALWAYS identify ethnicity FIRST
            components["ethnicity"] = timed_vision_call(
                "ethnicity",
                base64_image,
                "Identify the person's ETHNICITY/ANCESTRY. Be SPECIFIC: East Asian (Chinese/Japanese/Korean/Vietnamese), South Asian (Indian/Pakistani/Bengali), Southeast Asian (Thai/Filipino/Indonesian), Middle Eastern, African, African American, Caribbean, Latino/Hispanic, Caucasian/European (Northern/Southern/Eastern European), or Mixed heritage. Note facial features that indicate ethnicity.",
                "Identify ethnicity specifically in 1-2 sentences.", 80
            )

            # Detect shot type/framing FIRST - critical for reproduction
            components["shot_framing"] = timed_vision_call(
                "shot_framing",
                base64_image,
                "What is the SHOT TYPE/FRAMING? State exactly where the frame cuts off: EXTREME CLOSE-UP (face only, cropped at chin), CLOSE-UP/HEADSHOT (cropped at shoulders), MEDIUM CLOSE-UP (cropped at chest/bust), MEDIUM SHOT (cropped at waist), COWBOY SHOT (cropped at mid-thigh), FULL BODY (head to feet, feet visible). State: 'cropped at [body part]' or 'full body with feet visible'. Also note if feet/shoes are visible.",
                "State exact shot type and where frame cuts off in 1 sentence.", 60
            )

            # Detect selfie style
            components["selfie_check"] = timed_vision_call(
                "selfie_check",
                base64_image,
                "Is this a SELFIE? Look for: arm extended holding phone/camera, front-facing camera angle, slightly above eye level, direct eye contact. If SELFIE: say 'selfie-style photo with eyes looking directly at camera'. If NOT a selfie, say 'not a selfie' and where eyes are looking.",
                "Identify if selfie in 1 sentence.", 40
            )

            if self.analysis_mode == "detailed":
                components["face"] = timed_vision_call(
                    "face",
                    base64_image,
                    "Describe only the face in EXTREME detail: eye color, iris pattern, eye shape (almond/round/hooded/monolid/double-lid), gaze direction, catchlights. EYELASHES: color (black/brown/blonde), style (natural/long/thick/curled). EYEBROWS: shape, thickness, color. Facial structure, expression, age.",
                    "Describe the face with extreme eye, eyelash, and eyebrow detail in 3-4 sentences.", 250
                )
                components["skin_texture"] = timed_vision_call(
                    "skin_texture",
                    base64_image,
                    "Describe skin: TONE (fair/light/medium/olive/tan/brown/dark brown/deep). TEXTURE: freckles (location, density), dimples (cheeks, chin), wrinkles/fine lines (forehead, crow's feet, smile lines, nasolabial folds), visible pores, moles, blemishes, skin smoothness or roughness.",
                    "Describe skin tone and texture details in 2-3 sentences.", 150
                )
                components["makeup_lips"] = timed_vision_call(
                    "makeup_lips",
                    base64_image,
                    "Describe ONLY makeup and lips: eyeshadow color/style, eyeliner, blush, foundation, contouring. LIPS: exact shade (nude/pink/red/burgundy/coral/mauve), finish (matte/glossy/satin), lip texture (smooth/chapped/dry lines/natural creases), shape and fullness. If no makeup, say 'natural'.",
                    "Describe makeup and lip details in 2-3 sentences.", 180
                )
                components["hair"] = timed_vision_call(
                    "hair",
                    base64_image,
                    "Describe ONLY the hair: exact style (length, layers, texture, volume, parting), color with highlights/lowlights, how it frames the face and falls.",
                    "Describe the hairstyle in detail in 2-3 sentences.", 150
                )
                components["body"] = timed_vision_call(
                    "body",
                    base64_image,
                    "Describe the body: pose, build, posture. MUSCLE DEFINITION if visible: abs/midriff cuts (six-pack, four-pack, toned, flat), chest definition, arm muscles (biceps, triceps), shoulder definition. NAVEL if visible: shape (round/vertical/horizontal), innie/outie, depth.",
                    "Describe the body, muscle definition, and navel in 2-3 sentences.", 180
                )
                components["clothing"] = timed_vision_call(
                    "clothing",
                    base64_image,
                    """FIRST: Is this a recognizable UNIFORM or COSTUME? Be SPECIFIC: Roman Catholic nun habit, stylized/sexy nun (crop top variant). Traditional/modern/sexy nurse. French/Victorian/sexy maid. US/UK Police, SWAT. Japanese sailor fuku/British/Catholic schoolgirl. Playboy bunny, kimono, etc.
THEN DESCRIBE EACH GARMENT (CRITICAL):
- COLOR: EXACT color of EACH piece (jet black, charcoal gray, cream white, ivory, navy blue, burgundy - be SPECIFIC)
- FABRIC: cotton, silk, satin, velvet, leather, latex, lace, mesh, sheer, denim, wool, knit/ribbed, jersey
- TEXTURE: matte, shiny/glossy, smooth, ribbed, quilted, pleated
- PATTERN: solid, striped, plaid, floral, polka dot, animal print
- GARMENT TYPE: Is it FITTED/form-fitting or LOOSE/OPEN? (fitted crop top vs open robe - VERY DIFFERENT)
- LAYERING: What's UNDER what WITH COLORS? (e.g., "cream white ribbed turtleneck UNDER fitted jet black matte crop top")
- NECKLINE: crew/scoop/v-neck/turtleneck/off-shoulder/halter
- HEM/LENGTH: cropped at midriff, waist-length, hip-length, full-length
- SLEEVES: sleeveless/cap/short/long, tight-fitted or loose
- FIT: skin-tight, fitted, semi-fitted, loose, oversized""",
                    "Name the specific uniform type, then describe EACH garment with exact color, fabric, texture, and construction in 4-5 sentences.", 300
                )
            else:
                components["subject"] = timed_vision_call(
                    "subject",
                    base64_image,
                    """Describe the person: face (eye color, iris, eye shape, eyelash color/style, eyebrow shape), skin tone and texture (freckles, dimples, wrinkles), makeup/lip color and texture if visible, hair style and color, expression, body (pose, build, muscle definition if visible, navel shape if visible).
CLOTHING: First NAME specific uniform/costume type (Roman Catholic/Dominican nun, stylized/sexy nun with crop top, traditional/sexy nurse, French/Victorian/sexy maid, US/UK police, Japanese sailor fuku/Catholic schoolgirl, Playboy bunny, kimono, etc.) with variant (traditional/modern/stylized/sexy).
THEN FOR EACH GARMENT: EXACT COLOR (jet black, charcoal, cream white, ivory, navy - be specific). FABRIC (cotton, silk, satin, velvet, leather, knit/ribbed, jersey). TEXTURE (matte, shiny, ribbed, smooth). PATTERN (solid, striped, plaid). Is it FITTED or OPEN? LAYERING with colors (what's under what). NECKLINE. HEM (cropped at midriff, waist-length, full). SLEEVES. FIT (skin-tight, fitted, loose).""",
                    "Describe the person in 5-6 sentences with specific uniform type, exact garment colors, fabrics, and construction.", 550
                )

            components["lighting"] = timed_vision_call(
                "lighting",
                base64_image,
                "Describe ONLY the lighting: WHERE does it come from (window/lamp/sun/overhead), direction (left/right/front/back), quality (soft/hard/diffused), shadows, mood.",
                "Describe the lighting source and quality in 2-3 sentences.", 150
            )
            components["atmosphere"] = timed_vision_call(
                "atmosphere",
                base64_image,
                "Describe ONLY atmospheric effects if present: steam rising from cups/beverages, smoke, vapor, mist, fog, haze, dust particles visible in light beams, breath vapor in cold air. If none visible, say 'no atmospheric effects'.",
                "Describe any steam, smoke, or vapor effects in 1-2 sentences.", 100
            )
            components["environment"] = timed_vision_call(
                "environment",
                base64_image,
                "Describe background and objects/props held (cups, glasses, drinks). For ANY BEVERAGES: describe exact liquid color - red wine (deep burgundy), white wine (pale gold), beer (amber), coffee (dark brown), water (clear), cocktails (specific color).",
                "Describe the background and any beverages with their colors in 2-3 sentences.", 150
            )
        else:
            # Scene-only: ALWAYS identify shot type/framing FIRST
            components["shot_framing"] = timed_vision_call(
                "shot_framing",
                base64_image,
                "What is the SHOT TYPE/FRAMING? Be specific: EXTREME CLOSE-UP (only face/head fills frame, no neck/body), CLOSE-UP (head and partial neck), MEDIUM CLOSE-UP (head and full neck, partial body), MEDIUM SHOT (upper body), FULL BODY (entire subject), WIDE SHOT (subject small, environment dominant). Also state: front-facing, profile, or three-quarter view.",
                "State the exact shot type and angle in 1 sentence.", 50
            )
            components["subject"] = timed_vision_call(
                "subject",
                base64_image,
                "Describe the main subject. IF ANIMAL: species (be specific: 'reticulated giraffe', 'African elephant', 'Bengal tiger'), age (adult/juvenile/calf), eye contact (looking at camera or away), distinctive features (pattern, markings, horns, tusks, mane, eyelashes). For BEVERAGES/LIQUIDS: exact color. NO humans.",
                "Describe the main subject with species and features in 2-3 sentences.", 180
            )
            components["atmosphere"] = timed_vision_call(
                "atmosphere",
                base64_image,
                "Describe ONLY atmospheric effects: steam rising from cups/beverages, smoke, vapor, mist, fog, haze, dust particles in light beams. If none visible, say 'no atmospheric effects'. NO humans.",
                "Describe any steam, smoke, or vapor effects in 1-2 sentences.", 100
            )
            components["environment"] = timed_vision_call(
                "environment",
                base64_image,
                "Describe the environment: background (blurred/sharp, colors), lighting (golden hour, midday, overcast), depth of field. NO humans.",
                "Describe the environment and lighting in 2-3 sentences.", 150
            )

        # Synthesis step
        synthesis_prompt = f"""Combine these descriptions into one flowing prompt:

{json.dumps(components, indent=2)}

{f'User direction: {user_guidance}' if user_guidance else ''}

Write approximately {self.prompt_length} words as natural language.
End with "no text, no watermark"."""

        synthesis_start = time.time()
        result = self.llm_client.call_text(
            "You write prompts for AI image generation.",
            synthesis_prompt, self.prompt_length * 2
        )
        component_times["synthesis"] = time.time() - synthesis_start
        self._log(f"[Agentic] synthesis: {component_times['synthesis']:.2f}s")

        # Store component times for later access
        self._last_component_times = component_times

        # Print timing summary
        total_agentic = sum(component_times.values())
        self._log(f"[Agentic] ─────────────────────────────────────")
        self._log(f"[Agentic] TIMING SUMMARY ({len(component_times)} components)")
        for name, elapsed in component_times.items():
            pct = (elapsed / total_agentic * 100) if total_agentic > 0 else 0
            self._log(f"[Agentic]   {name}: {elapsed:.2f}s ({pct:.0f}%)")
        self._log(f"[Agentic] ─────────────────────────────────────")
        self._log(f"[Agentic] TOTAL: {total_agentic:.2f}s")

        return result

    def _get_negative(self, has_human: bool) -> str:
        """Get negative prompt."""
        if has_human:
            return "blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, mutated, disfigured, poorly drawn face, bad proportions, gross proportions, malformed limbs, long neck, text, watermark, signature"
        else:
            return "blurry, low quality, distorted, ugly, oversaturated, overexposed, underexposed, noisy, grainy, text, watermark, signature, human, person, people, face, body"

    def _get_caption(self, prompt: str) -> str:
        """Generate social media caption."""
        caption_prompt = f"""Based on this image description, write a short engaging social media caption (1-2 sentences):

{prompt}

Be concise and engaging."""

        return self.llm_client.call_text(
            "You write engaging social media captions.",
            caption_prompt, 100
        )

    def process_batch(
        self,
        input_path: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        user_guidance: str = "",
        extensions: List[str] = None
    ) -> List[GenerationResult]:
        """
        Process a batch of images from a directory.

        Args:
            input_path: Directory containing images
            output_file: Optional JSONL file to save results
            user_guidance: Optional guidance to apply to all images
            extensions: Image extensions to process (default: common formats)

        Returns:
            List of GenerationResult objects
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']

        input_path = Path(input_path)
        if not input_path.is_dir():
            raise ValueError(f"Input path must be a directory: {input_path}")

        # Find all images
        images = []
        for ext in extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))

        images = sorted(set(images))
        print(f"[Batch] Found {len(images)} images in {input_path}")

        results = []
        output_handle = None

        if output_file:
            output_file = Path(output_file)
            output_handle = open(output_file, 'w', encoding='utf-8')
            print(f"[Batch] Writing results to {output_file}")

        try:
            for i, img_path in enumerate(images, 1):
                print(f"\n[Batch] Processing {i}/{len(images)}: {img_path.name}")

                try:
                    result = self.process_image(img_path, user_guidance)
                    result.metadata["filename"] = img_path.name
                    results.append(result)

                    if output_handle:
                        output_handle.write(json.dumps(result.to_dict()) + '\n')
                        output_handle.flush()

                except Exception as e:
                    print(f"[Batch] ERROR processing {img_path.name}: {e}")
                    continue

                # Cleanup between images
                gc.collect()

        finally:
            if output_handle:
                output_handle.close()

        print(f"\n[Batch] Completed: {len(results)}/{len(images)} images processed")
        return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate AI image prompts from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with Ollama
  python prompt_generator_core.py --image photo.jpg --provider ollama --model llava

  # Batch process directory
  python prompt_generator_core.py --batch ./images/ --output prompts.jsonl

  # Use OpenAI
  python prompt_generator_core.py --image photo.jpg --provider openai --model gpt-4o --api-key YOUR_KEY
"""
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str, help='Single image path')
    input_group.add_argument('--batch', '-b', type=str, help='Directory for batch processing')

    # Provider settings
    parser.add_argument('--provider', '-p', type=str, default='ollama',
                        choices=['ollama', 'lmstudio', 'anthropic', 'openai', 'gemini', 'groq'],
                        help='LLM provider (default: ollama)')
    parser.add_argument('--model', '-m', type=str, default='llava:latest',
                        help='Model name (default: llava:latest)')
    parser.add_argument('--api-key', type=str, default='',
                        help='API key for cloud providers')
    parser.add_argument('--api-url', type=str, default='',
                        help='API URL (default: provider default)')

    # Generation settings
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['quick', 'standard', 'detailed'],
                        help='Analysis mode (default: standard)')
    parser.add_argument('--length', '-l', type=int, default=150,
                        help='Target prompt length in words (default: 150)')
    parser.add_argument('--guidance', '-g', type=str, default='',
                        help='User guidance/instructions')

    # Options
    parser.add_argument('--negative', action='store_true',
                        help='Generate negative prompt')
    parser.add_argument('--caption', action='store_true',
                        help='Generate social media caption')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file (JSONL for batch, JSON for single)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Set default API URL for local providers
    api_url = args.api_url
    if not api_url:
        if args.provider == 'ollama':
            api_url = 'http://localhost:11434/v1'
        elif args.provider == 'lmstudio':
            api_url = 'http://localhost:1234/v1'

    # Create generator
    generator = PromptGenerator(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_url=api_url,
        analysis_mode=args.mode,
        prompt_length=args.length,
        generate_negative=args.negative,
        generate_caption=args.caption,
        verbose=not args.quiet
    )

    # Process
    if args.image:
        # Single image
        result = generator.process_image(args.image, args.guidance)

        print("\n" + "=" * 60)
        print("PROMPT:")
        print("-" * 60)
        print(result.prompt)

        if result.negative:
            print("\nNEGATIVE:")
            print("-" * 60)
            print(result.negative)

        if result.caption:
            print("\nCAPTION:")
            print("-" * 60)
            print(result.caption)

        print("=" * 60)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nSaved to: {args.output}")

    else:
        # Batch process
        results = generator.process_batch(
            args.batch,
            output_file=args.output,
            user_guidance=args.guidance
        )

        print(f"\nProcessed {len(results)} images")
        if args.output:
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
