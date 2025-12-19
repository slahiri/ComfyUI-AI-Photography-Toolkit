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
# Config Wrapper Functions - TOML is the single source of truth
# =============================================================================
# All prompts are loaded from config/prompts/*.toml files via config_loader.
# Minimal inline fallbacks are provided only if config_loader fails to import.

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
    """Get system prompt from config. TOML is the single source of truth."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_system_prompt(tier)
    # Minimal fallback if config_loader unavailable
    return "Describe this image for AI image generation. Write flowing natural language, not keyword lists."


def get_user_prompt(tier: str, mode: str) -> str:
    """Get user prompt from config. TOML is the single source of truth."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_user_prompt(tier, mode)
    # Minimal fallback
    return "Describe this image in detail for AI recreation."


def get_scene_system_prompt() -> str:
    """Get scene-only system prompt from config. TOML is the single source of truth."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_scene_only_system_prompt()
    # Minimal fallback
    return "Describe this scene for AI image generation. There are NO people in this image."


def get_scene_user_prompt(mode: str) -> str:
    """Get scene-only user prompt from config. TOML is the single source of truth."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_scene_only_user_prompt(mode)
    # Minimal fallback
    return "Describe this scene in detail. NO humans."


def get_component_prompt(component: str, tier: str) -> str:
    """Get agentic component prompt from config. TOML is the single source of truth."""
    if _CONFIG_AVAILABLE and _config_loader:
        return _config_loader.get_component_prompt(component, tier)
    # Minimal fallback - return empty string, caller should handle
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


# =============================================================================
# Template System - Dynamic loading for hot-reload
# =============================================================================

def load_templates() -> Dict[str, Dict[str, str]]:
    """
    Load prompt templates from TOML file.

    Called dynamically each time to support hot-reload without restart.
    Add new templates to config/templates.toml and refresh browser.

    Returns:
        Dict mapping template key to {name, system, description}
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    # Find templates.toml relative to this file
    config_dir = Path(__file__).parent / "config"
    templates_file = config_dir / "templates.toml"

    if not templates_file.exists():
        print(f"[PromptGen] Templates file not found: {templates_file}")
        return {}

    try:
        with open(templates_file, "rb") as f:
            data = tomllib.load(f)

        templates = {}
        for key, value in data.items():
            if isinstance(value, dict) and "name" in value and "system" in value:
                templates[key] = {
                    "name": value.get("name", key),
                    "system": value.get("system", ""),
                    "description": value.get("description", "")
                }
        return templates
    except Exception as e:
        print(f"[PromptGen] Error loading templates: {e}")
        return {}


def get_template_names() -> List[str]:
    """
    Get list of template display names for dropdown.

    Called dynamically to support hot-reload.

    Returns:
        List of template display names
    """
    templates = load_templates()
    return [t["name"] for t in templates.values()]


def get_template_by_name(display_name: str) -> Optional[Dict[str, str]]:
    """
    Get template by its display name.

    Args:
        display_name: The name shown in dropdown

    Returns:
        Template dict with {name, system, description} or None
    """
    templates = load_templates()
    for template in templates.values():
        if template["name"] == display_name:
            return template
    return None


def cleanup_tags(raw_tags: str) -> str:
    """
    Clean up LLM-generated tags output.

    Simple cleanup function that:
    - Ensures quality prefix is present
    - Removes extra whitespace and formatting artifacts
    - Deduplicates tags while preserving order

    Args:
        raw_tags: Raw tag output from LLM
    """
    if not raw_tags:
        return ""

    # Clean up the raw output
    # Remove any leading/trailing quotes or whitespace
    cleaned = raw_tags.strip().strip('"\'')

    # Split by comma and clean each tag
    tags = []
    for tag in cleaned.split(","):
        tag = tag.strip().strip('"\'')
        if tag:
            # Convert spaces to underscores for multi-word tags
            tag = tag.replace(" ", "_")
            tags.append(tag)

    # Ensure quality prefix is at the start
    quality_tags = ["masterpiece", "best_quality", "highres"]
    final_tags = []

    # Add quality tags first if not already present
    for qt in quality_tags:
        # Check for variations (best_quality vs best quality)
        qt_variants = [qt, qt.replace("_", " ")]
        if not any(t.lower() in qt_variants for t in tags):
            final_tags.append(qt)

    # Add remaining tags, deduplicating
    seen = set(t.lower() for t in final_tags)
    for tag in tags:
        tag_lower = tag.lower()
        # Skip if already added or is a quality tag variant
        if tag_lower not in seen and tag_lower not in ["best_quality", "best quality"]:
            seen.add(tag_lower)
            final_tags.append(tag)

    return ", ".join(final_tags)


# Keep old function name as alias for compatibility
def convert_to_tags(expanded_prompt: str, vocabulary: Dict[str, List[str]] = None, has_human: bool = True) -> str:
    """
    Legacy wrapper - now just calls cleanup_tags.
    The LLM generates tags directly when Tags mode is selected.
    """
    return cleanup_tags(expanded_prompt)


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


def strip_reasoning_output(text: str) -> str:
    """
    Strip reasoning/thinking text from model output.

    Some models (especially Qwen3) output their reasoning process before the actual answer.
    This function detects and removes that reasoning to get the clean prompt.

    Examples of reasoning that gets stripped:
    - "Okay, let me start by understanding what the user wants..."
    - "First, I need to analyze the image..."
    - "Looking at this image, I can see... Now for the prompt:"
    """
    if not text:
        return text

    original_text = text

    # Pattern 1: Remove <think>...</think> blocks
    text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>\s*', '', text, flags=re.IGNORECASE)

    # Pattern 2: Look for explicit prompt markers and take content after them
    prompt_markers = [
        r'(?:^|\n)(?:Final |The |Here\'?s? (?:the )?)?[Pp]rompt[:\s]+',
        r'(?:^|\n)(?:Final |The )?[Oo]utput[:\s]+',
        r'(?:^|\n)(?:Here\'?s? )?(?:the )?(?:enhanced |modified |updated )?(?:image )?(?:generation )?prompt[:\s]+',
        r'(?:^|\n)[Dd]escription[:\s]+',
    ]

    for marker in prompt_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            after_marker = text[match.end():].strip()
            # Only use if there's substantial content after the marker
            if len(after_marker) > 50:
                text = after_marker
                break

    # Pattern 3: Detect reasoning at the start and find where actual content begins
    reasoning_starters = [
        r'^Okay,?\s*',
        r'^(?:Let me|I\'ll|I will|I need to|I should)\s+',
        r'^First,?\s*',
        r'^(?:Looking at|Analyzing|Examining)\s+',
        r'^(?:My task|The task|To do this)\s+',
        r'^(?:Now|So|Then),?\s*',
        r'^(?:Based on|According to)\s+',
        r'^(?:The user|User)\s+(?:wants|asked|requested)\s+',
    ]

    # Check if text starts with reasoning
    starts_with_reasoning = any(re.match(pattern, text, re.IGNORECASE) for pattern in reasoning_starters)

    if starts_with_reasoning:
        # Find where the actual prompt likely starts
        # Look for a sentence that starts with visual description keywords
        visual_starters = [
            r'(?:^|\.\s+)([A-Z]?(?:close-up|medium|full|portrait|headshot|shot|photo|image|picture)\b)',
            r'(?:^|\.\s+)(A\s+(?:young|beautiful|handsome|elegant|stunning)?\s*(?:woman|man|person|girl|boy|lady|gentleman)\b)',
            r'(?:^|\.\s+)((?:The\s+)?(?:subject|model|person|figure)\b)',
            r'(?:^|\.\s+)((?:Warm|Cool|Soft|Dramatic|Natural|Golden|Studio)\s+(?:light|lighting)\b)',
        ]

        for pattern in visual_starters:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Found visual content, extract from here
                start_pos = match.start()
                if text[start_pos] == '.':
                    start_pos += 1
                extracted = text[start_pos:].strip()
                if len(extracted) > 50:
                    text = extracted
                    break

    # Pattern 4: Remove trailing reasoning/meta-commentary
    trailing_patterns = [
        r'\s*(?:I hope this|Is there anything|Let me know|Feel free to).*$',
        r'\s*(?:This should|This will|This prompt will).*$',
        r'\s*\n\s*(?:Note|Explanation|Reasoning):.*$',
    ]

    for pattern in trailing_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # If we stripped too much, return original
    if len(text.strip()) < 30 and len(original_text) > 50:
        return original_text

    return text.strip()


def zimage_clean(prompt: str, provider: str = "default", max_words: int = 500) -> str:
    """Apply Z-Image cleanup to prompt using config patterns + built-in rules."""
    if not prompt:
        return ""

    # First, strip any reasoning/thinking output
    prompt = strip_reasoning_output(prompt)

    # Then, detect and fix hallucination/runaway text
    prompt, was_hallucinating = detect_and_fix_hallucination(prompt, max_words)

    # Then apply config-based cleanup (example patterns, markdown, provider-specific)
    prompt = clean_llm_output(prompt, provider)

    # Remove common LLM prefixes that some models add despite instructions
    prefixes_to_remove = [
        r'^Z-Image Prompt:\s*',
        r'^Z-Image:\s*',
        r'^Prompt:\s*',
        r'^Here is (?:the|a|your) (?:prompt|description):\s*',
        r'^Here\'s (?:the|a|your) (?:prompt|description):\s*',
        r'^The prompt is:\s*',
        r'^Output:\s*',
        r'^Generated prompt:\s*',
    ]
    for prefix in prefixes_to_remove:
        prompt = re.sub(prefix, '', prompt, flags=re.IGNORECASE)

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

            if provider == "local":
                # Local models use OpenAI-compatible interface
                # Combine system + user prompt for local models (they don't support separate system messages)
                combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": combined_prompt}
                    ]
                }]
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp
                )
                result = response.choices[0].message.content or ""
                return result.strip()

            elif provider == "anthropic":
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
            import traceback
            print(f"[LLM] Vision call failed: {e}")
            traceback.print_exc()
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
                    keep_model_loaded=extra.get("keep_model_loaded", False),
                    temperature=temp,
                    top_p=extra.get("top_p", 0.9),
                    repetition_penalty=extra.get("repetition_penalty", 1.2),
                    num_beams=extra.get("num_beams", 1),
                    use_torch_compile=extra.get("use_torch_compile", False),
                )
                try:
                    result = text_client.generate_text(user_prompt, system_prompt, max_tokens=max_tokens, temperature=temp)
                    return result.strip() if result else ""
                finally:
                    # Clean up text client after use
                    try:
                        del text_client
                        gc.collect()
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass

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
        prompt_style: str = "verbose",
        prompt_length: int = 150,
        generate_negative: bool = False,
        generate_caption: bool = False,
        nsfw_mode: bool = False,
        verbose: bool = True,
        extra_params: Dict[str, Any] = None,
        template_prompt: str = None
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
        self.prompt_style = prompt_style.lower()  # "template", "verbose", or "tags"
        self.prompt_length = prompt_length
        self.generate_negative = generate_negative
        self.generate_caption = generate_caption
        self.nsfw_mode = nsfw_mode
        self.verbose = verbose
        self.template_prompt = template_prompt  # Custom template system prompt

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

        # Template mode: Use custom template prompt directly (bypasses analysis modes)
        if self.prompt_style == "template" and self.template_prompt:
            self._log(f"[PromptGen] Using TEMPLATE mode with custom system prompt")
            # Use template's system prompt with simple user prompt
            template_user = "Describe this image following the system instructions."
            if user_guidance:
                template_user += f"\n\nAdditional guidance: {user_guidance}"
            prompt = self.llm_client.call_vision(
                base64_image, self.template_prompt, template_user,
                max_tokens=self.prompt_length * 3
            )
        # Analysis mode routing:
        # - Quick: Single vision call (fast, ~1 call) - all providers
        # - Standard: Optimized multi-aspect analysis (~3-4 calls) - all providers
        # - Detailed: Full agentic analysis (~12-14 calls) - API only, requires reasoning
        #
        # Local providers support Quick and Standard modes.
        # Detailed mode falls back to Standard for local providers.
        elif self.analysis_mode == "quick":
            # Quick mode: Single vision call
            self._log(f"[PromptGen] Using QUICK single-pass analysis")
            prompt = self.llm_client.call_vision(
                base64_image, system_prompt, user_prompt,
                max_tokens=self.prompt_length * 3
            )
        elif self.analysis_mode == "standard":
            # Standard mode: Optimized 3-4 call approach (works for all providers)
            self._log(f"[PromptGen] Using STANDARD optimized analysis (3-4 calls)")
            prompt = self._optimized_generate(base64_image, has_human, cv_analysis, user_guidance)
        elif self.analysis_mode == "detailed":
            # Detailed mode: Full agentic analysis (API only, requires reasoning)
            if is_local:
                # Local providers fall back to Standard (too many calls for local)
                self._log(f"[PromptGen] Using STANDARD optimized analysis (detailed not supported for local)")
                prompt = self._optimized_generate(base64_image, has_human, cv_analysis, user_guidance)
            elif self.enable_reasoning:
                self._log(f"[PromptGen] Using DETAILED agentic analysis (12+ calls)")
                prompt = self._agentic_generate(base64_image, has_human, cv_analysis, user_guidance)
            else:
                # Detailed without reasoning: Fall back to Standard
                self._log(f"[PromptGen] Using STANDARD optimized analysis (detailed requires reasoning)")
                prompt = self._optimized_generate(base64_image, has_human, cv_analysis, user_guidance)
        else:
            # Unknown mode: Fall back to Quick
            self._log(f"[PromptGen] Using QUICK single-pass analysis (unknown mode: {self.analysis_mode})")
            prompt = self.llm_client.call_vision(
                base64_image, system_prompt, user_prompt,
                max_tokens=self.prompt_length * 3
            )

        result.timing["llm_time_s"] = time.time() - llm_start

        # Capture component times if available (from optimized or agentic modes)
        if hasattr(self, '_last_component_times') and self._last_component_times:
            result.timing["components"] = self._last_component_times.copy()
            self._last_component_times = None  # Clear after capturing

        # Step 5: Z-Image cleanup (uses config patterns + built-in rules)
        # Use prompt_length + 50% buffer as max words to allow some flexibility
        max_words = int(self.prompt_length * 1.5)
        result.prompt = zimage_clean(prompt, self.config.provider, max_words)

        # Step 5b: Z-Image realism enhancers REMOVED
        # Based on evaluator feedback: quality meta-tags like "realistic skin texture, photorealistic,
        # ultra detailed" are unnecessary for Z-Image/Flux models - quality is built-in.
        # These tags were found to clutter prompts without improving output quality.

        # Step 5c: Clean up tags if tags mode (LLM already generates in tag format)
        if self.prompt_style == "tags":
            self._log(f"[PromptGen] Cleaning up LLM-generated tags...")
            result.prompt = cleanup_tags(result.prompt)
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
        """Build system and user prompts from TOML config (single source of truth)."""
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
            # Get prompts from TOML config (portrait.toml)
            system_prompt = get_system_prompt(tier)
            user_prompt = get_user_prompt(tier, self.analysis_mode)
        else:
            # Get prompts from TOML config (scene.toml)
            system_prompt = get_scene_system_prompt()
            user_prompt = get_scene_user_prompt(self.analysis_mode)

        # Add length constraint to system prompt
        if length_constraint:
            system_prompt += f"\n\n{length_constraint}"

        # Add CV hints to user prompt
        if cv_hints:
            user_prompt = f"{cv_hints}\n\n{user_prompt}"

        # Add user guidance if provided
        if user_guidance and user_guidance.strip():
            user_prompt += f"""

USER DIRECTION (apply this):
"{user_guidance.strip()}"
"""

        # Add style-specific instructions for Z-Image compatibility
        if self.prompt_style == "tags":
            tag_instruction = """

OUTPUT FORMAT - Z-IMAGE COMPATIBLE TAGS:
Output as comma-separated tags for Z-Image realistic image generation. NO sentences or prose.
Start with: masterpiece, best quality, highres, realistic, photorealistic
Then add descriptive tags like: 1girl, solo, long_hair, brown_eyes, smile, dress, outdoors
Use underscores for multi-word tags (e.g., long_hair, looking_at_viewer)
Keep tags concise - single concepts only (e.g., "blue_eyes" not "beautiful sparkling blue eyes")
End with: no_text, no_watermark
Example output: masterpiece, best quality, highres, realistic, photorealistic, 1girl, solo, long_hair, blonde_hair, blue_eyes, smile, white_dress, standing, outdoors, sunlight, no_text, no_watermark"""
            system_prompt += tag_instruction
            user_prompt += "\n\nRemember: Output ONLY comma-separated tags for Z-Image realistic generation, not sentences."
        else:
            # Expanded mode - natural language for Z-Image
            expanded_instruction = """

OUTPUT FORMAT - Z-IMAGE COMPATIBLE PROMPT:
Write natural flowing sentences optimized for Z-Image realistic, photorealistic image generation.
Focus on visual details that produce realistic output: lighting, textures, materials, colors.
End with: no text, no watermark"""
            system_prompt += expanded_instruction

        # Add NSFW description instructions if enabled (humans only)
        if self.nsfw_mode and has_human:
            nsfw_instruction = """

NSFW DETAILS - Describe any revealing/suggestive elements with specific terms:
- EXPOSURE: sideboob, underboob, cleavage, deep_cleavage, bare_shoulders, bare_back, backless, midriff, exposed_navel, thigh_gap, inner_thigh, hip_bones, bare_legs
- CLOTHING: see-through, sheer_fabric, transparent, low-cut, plunging_neckline, high_slit, micro_bikini, string_bikini, lingerie_visible, braless, wet_clothing, tight/form-fitting, crop_top
- POSES: arched_back, bent_forward, reclining, suggestive_pose, hand_on_hip, hand_on_thigh
- SKIN: glistening_skin, wet_skin, oiled_skin, flushed_cheeks
- EXPRESSION: bedroom_eyes, half-lidded_gaze, parted_lips, lip_bite, seductive_expression
Be specific about exactly which NSFW features are visible. Do NOT add features that are not in the image."""
            system_prompt += nsfw_instruction
            user_prompt += "\n\nInclude specific NSFW details for any revealing elements visible in the image."

        return system_prompt, user_prompt

    def _optimized_generate(self, base64_image: str, has_human: bool,
                            cv_analysis: Dict, user_guidance: str) -> str:
        """
        Optimized multi-aspect generation (3-4 calls).

        Consolidates the 12+ call agentic approach into 3-4 efficient calls:
        1. Subject Analysis - person details (ethnicity, face, skin, hair, body, clothing)
        2. Environment Analysis - framing, lighting, atmosphere, background
        3. NSFW Details (optional) - revealing elements if nsfw_mode enabled
        4. Synthesis - combine into final prompt

        This is ideal for advanced models (Claude, GPT-4o, Gemini) that can handle
        complex multi-aspect prompts in a single call.
        """
        components = {}
        component_times = {}
        tier = get_provider_tier(self.config.provider)

        def timed_vision_call(name: str, system_prompt: str, user_instruction: str, max_tokens: int):
            """Call vision API with timing."""
            start = time.time()
            result = self.llm_client.call_vision(base64_image, system_prompt, user_instruction, max_tokens)
            elapsed = time.time() - start
            component_times[name] = elapsed
            self._log(f"[Optimized] {name}: {elapsed:.2f}s")
            return result

        if has_human:
            # Call 1: Comprehensive Subject Analysis
            subject_system = """You are an expert at describing people for AI image generation prompts.
Analyze ALL aspects of this person in a single comprehensive response.
Be specific and visual - describe what you SEE, not what you interpret.
Use precise terminology for ethnicity, features, clothing, and poses."""

            subject_user = """Describe this person comprehensively, covering ALL these aspects in flowing prose:

1. IDENTITY: Ethnicity/ancestry (be specific: East Asian, South Asian, Nordic, Mediterranean, etc.), approximate age range
2. FACE: Facial structure, eye shape/color, eyebrows, eyelashes, nose, lips, expression
3. SKIN: Skin tone (use specific terms: porcelain, ivory, olive, tan, bronze, etc.), texture, any visible details
4. MAKEUP: If visible - foundation, eye makeup, lip color, blush, etc.
5. HAIR: Style, length, color, texture (straight/wavy/curly), any accessories
6. BODY: Visible body type, pose, posture, any muscle definition
7. CLOTHING: Identify the SPECIFIC uniform/outfit type first (e.g., "Roman Catholic nun habit", "French maid outfit", "Japanese sailor fuku"), then describe each garment's color, fabric, fit, and details

Write 4-6 detailed sentences covering all visible aspects. Be specific about the exact type of any uniform or costume."""

            components["subject"] = timed_vision_call(
                "subject", subject_system, subject_user, 800
            )

            # Call 2: Environment & Technical Analysis
            env_system = """You are an expert at analyzing photography and image composition.
Describe the technical and environmental aspects of this image for AI recreation."""

            env_user = """Analyze the environment and technical aspects:

1. SHOT TYPE: Exact framing (extreme close-up, close-up, medium close-up, medium shot, full shot, wide shot)
2. CAMERA: Angle (eye-level, low angle, high angle, Dutch angle), if it appears to be a selfie
3. LIGHTING: Source direction, quality (soft/hard), color temperature, any dramatic effects
4. ATMOSPHERE: Any fog, steam, smoke, bokeh, lens effects
5. BACKGROUND: Setting, depth of field, key background elements, colors

Write 2-3 sentences covering the environment and technical aspects."""

            components["environment"] = timed_vision_call(
                "environment", env_system, env_user, 400
            )

        else:
            # Scene-only (no human)
            scene_system = """You are an expert at describing scenes and objects for AI image generation.
Be specific and visual - describe what you SEE in detail."""

            scene_user = """Describe this scene comprehensively:

1. MAIN SUBJECT: What is the primary focus? Species/type if animal, object details if still life
2. COMPOSITION: Framing, angle, depth of field
3. ENVIRONMENT: Setting, background elements, atmosphere
4. LIGHTING: Source, quality, color temperature, mood
5. COLORS: Dominant color palette, any striking color contrasts

Write 3-4 detailed sentences. There are NO humans in this image."""

            components["scene"] = timed_vision_call(
                "scene", scene_system, scene_user, 600
            )

        # Call 3 (Optional): NSFW Details
        if self.nsfw_mode and has_human:
            nsfw_system = """You describe revealing or suggestive elements in images using specific terminology.
Be clinical and precise. Only describe what is actually visible."""

            nsfw_user = """Describe any revealing or suggestive elements visible:

- EXPOSURE: sideboob, underboob, cleavage, bare shoulders, bare back, midriff, thigh gap, etc.
- CLOTHING: see-through, sheer, low-cut, high slit, tight/form-fitting, etc.
- POSE: any suggestive positioning
- SKIN: wet, glistening, oiled appearance

Only describe elements that are ACTUALLY VISIBLE. Write 1-3 sentences."""

            components["nsfw"] = timed_vision_call(
                "nsfw", nsfw_system, nsfw_user, 200
            )

        # Call 4: Synthesis - integrate enhancement naturally into the prompt
        enhancement_instruction = ""
        if user_guidance:
            enhancement_instruction = f"""
USER ENHANCEMENT (MUST integrate naturally into the prompt):
"{user_guidance}"

IMPORTANT: Integrate this enhancement INTO the description naturally - modify clothing, add/remove elements,
or adjust the scene as specified. Do NOT append it at the end. The final prompt should read as if the
enhancement was always part of the original description."""

        if self.prompt_style == "tags":
            synthesis_prompt = f"""Convert these descriptions into Z-Image compatible tags for realistic image generation:

{json.dumps(components, indent=2)}
{enhancement_instruction}

OUTPUT FORMAT - Z-IMAGE COMPATIBLE TAGS:
Start with: masterpiece, best quality, highres, realistic, photorealistic
Then descriptive tags using underscores for multi-word tags (long_hair, looking_at_viewer)
{f'Include tags for: {user_guidance}' if user_guidance else ''}
NO sentences - ONLY comma-separated tags.
End with: no_text, no_watermark"""
        else:
            synthesis_prompt = f"""Combine these descriptions into a Z-Image prompt for realistic, photorealistic image generation:

{json.dumps(components, indent=2)}
{enhancement_instruction}

INSTRUCTIONS:
- Write 4-6 flowing sentences optimized for Z-Image realistic output
- Start with the subject, then clothing/appearance, then environment
- Include specific visual details: colors, textures, lighting, materials
- If enhancement is provided, weave it naturally into the description
- Output must generate realistic, photorealistic results
- End with "no text, no watermark"

Write approximately {self.prompt_length} words."""

        synthesis_start = time.time()
        result = self.llm_client.call_text(
            "You write prompts for AI image generation. Combine descriptions into flowing, detailed prompts.",
            synthesis_prompt,
            self.prompt_length * 2
        )
        component_times["synthesis"] = time.time() - synthesis_start
        self._log(f"[Optimized] synthesis: {component_times['synthesis']:.2f}s")

        # Store component times for metadata
        self._last_component_times = component_times

        # Print timing summary
        total_time = sum(component_times.values())
        self._log(f"[Optimized] ─────────────────────────────────────")
        self._log(f"[Optimized] TIMING SUMMARY ({len(component_times)} calls)")
        for name, elapsed in component_times.items():
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            self._log(f"[Optimized]   {name}: {elapsed:.2f}s ({pct:.0f}%)")
        self._log(f"[Optimized] ─────────────────────────────────────")
        self._log(f"[Optimized] TOTAL: {total_time:.2f}s")

        return result

    def _agentic_generate(self, base64_image: str, has_human: bool,
                          cv_analysis: Dict, user_guidance: str) -> str:
        """Agentic multi-step generation using component prompts from TOML."""
        components = {}
        component_times = {}  # Track timing for each component
        tier = get_provider_tier(self.config.provider)

        def timed_vision_call(name: str, component_key: str, user_instruction: str, max_tokens: int):
            """Call vision API with prompt from TOML config."""
            start = time.time()
            # Get system prompt from TOML components.toml
            system_prompt = get_component_prompt(component_key, tier)
            if not system_prompt:
                self._log(f"[Agentic] Warning: No prompt found for {component_key}.{tier}")
                return ""
            result = self.llm_client.call_vision(base64_image, system_prompt, user_instruction, max_tokens)
            elapsed = time.time() - start
            component_times[name] = elapsed
            self._log(f"[Agentic] {name}: {elapsed:.2f}s")
            return result

        if has_human:
            # Core components (always run)
            components["ethnicity"] = timed_vision_call(
                "ethnicity", "ethnicity",
                "Identify ethnicity specifically in 1-2 sentences.", 80
            )
            components["shot_framing"] = timed_vision_call(
                "shot_framing", "framing",
                "State exact shot type and where frame cuts off in 1 sentence.", 60
            )
            components["selfie_check"] = timed_vision_call(
                "selfie_check", "selfie_check",
                "Identify if selfie in 1 sentence.", 40
            )

            if self.analysis_mode == "detailed":
                # Detailed mode: separate components for each aspect
                components["face"] = timed_vision_call(
                    "face", "face",
                    "Describe the face with extreme eye, eyelash, and eyebrow detail in 3-4 sentences.", 250
                )
                components["skin_texture"] = timed_vision_call(
                    "skin_texture", "skin_texture",
                    "Describe skin tone and texture details in 2-3 sentences.", 150
                )
                components["makeup_lips"] = timed_vision_call(
                    "makeup_lips", "makeup_lips",
                    "Describe makeup and lip details in 2-3 sentences.", 180
                )
                components["hair"] = timed_vision_call(
                    "hair", "hair",
                    "Describe the hairstyle in detail in 2-3 sentences.", 150
                )
                components["body"] = timed_vision_call(
                    "body", "body_pose",
                    "Describe the body, muscle definition, and navel in 2-3 sentences.", 180
                )
                components["clothing"] = timed_vision_call(
                    "clothing", "clothing",
                    "Name the specific uniform type, then describe EACH garment with exact color, fabric, texture, and construction in 4-5 sentences.", 300
                )
            else:
                # Standard mode: combined subject description
                components["subject"] = timed_vision_call(
                    "subject", "subject_detection",
                    "Describe the person in 5-6 sentences with specific uniform type, exact garment colors, fabrics, and construction.", 550
                )

            # Common components for all human images
            components["lighting"] = timed_vision_call(
                "lighting", "lighting",
                "Describe the lighting source and quality in 2-3 sentences.", 150
            )
            components["atmosphere"] = timed_vision_call(
                "atmosphere", "atmosphere",
                "Describe any steam, smoke, or vapor effects in 1-2 sentences.", 100
            )
            components["environment"] = timed_vision_call(
                "environment", "environment",
                "Describe the background and any beverages with their colors in 2-3 sentences.", 150
            )

            # NSFW details component (only when enabled)
            if self.nsfw_mode:
                components["nsfw_details"] = timed_vision_call(
                    "nsfw_details", "nsfw_details",
                    "Describe any revealing/suggestive elements: exposed skin areas, clothing sheerness, pose suggestiveness in 2-4 sentences. Use specific terms like sideboob, underboob, cleavage, bare_shoulders, see-through, etc.", 200
                )
        else:
            # Scene-only components (no human)
            components["shot_framing"] = timed_vision_call(
                "shot_framing", "framing",
                "State the exact shot type and angle in 1 sentence.", 50
            )
            components["subject"] = timed_vision_call(
                "subject", "scene_subject",
                "Describe the main subject with species and features in 2-3 sentences.", 180
            )
            components["atmosphere"] = timed_vision_call(
                "atmosphere", "atmosphere",
                "Describe any steam, smoke, or vapor effects in 1-2 sentences.", 100
            )
            components["environment"] = timed_vision_call(
                "environment", "environment",
                "Describe the environment and lighting in 2-3 sentences.", 150
            )

        # Synthesis step - integrate enhancement naturally into the prompt
        enhancement_instruction = ""
        if user_guidance:
            enhancement_instruction = f"""
USER ENHANCEMENT (MUST integrate naturally into the prompt):
"{user_guidance}"

IMPORTANT: Integrate this enhancement INTO the description naturally - modify clothing, add/remove elements,
or adjust the scene as specified. Do NOT append it at the end. The final prompt should read as if the
enhancement was always part of the original description."""

        if self.prompt_style == "tags":
            synthesis_prompt = f"""Convert these descriptions into Z-Image compatible tags for realistic image generation:

{json.dumps(components, indent=2)}
{enhancement_instruction}

OUTPUT FORMAT - Z-IMAGE COMPATIBLE TAGS:
Start with: masterpiece, best quality, highres, realistic, photorealistic
Then descriptive tags using underscores for multi-word tags (long_hair, looking_at_viewer)
{f'Include tags for: {user_guidance}' if user_guidance else ''}
NO sentences - ONLY comma-separated tags.
End with: no_text, no_watermark"""
        else:
            # Get synthesis instructions from TOML (single source of truth)
            synthesis_instructions = get_component_prompt("synthesis", tier)
            if not synthesis_instructions:
                # Minimal fallback
                synthesis_instructions = "Combine into 4-6 flowing sentences for Z-Image realistic output. End with 'no text, no watermark'."

            synthesis_prompt = f"""Combine these descriptions into a Z-Image prompt for realistic, photorealistic image generation:

{json.dumps(components, indent=2)}
{enhancement_instruction}

{synthesis_instructions}

IMPORTANT: If enhancement is provided, weave it naturally into the description.
Maintain the same prompt length (~{self.prompt_length} words) - integrate, don't append.
Output must generate realistic, photorealistic results.

Write approximately {self.prompt_length} words."""

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
