#!/usr/bin/env python3
"""Enhance vocabulary with external datasets.

Downloads and integrates:
1. Danbooru tags from Hugging Face
2. Photography lighting patterns
3. Camera shots and angles
4. Quality/rendering terms

Usage:
    python scripts/enhance_vocabulary.py [--output vocab.json]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set, List


# =============================================================================
# Manual Vocabulary Additions (from research)
# =============================================================================

# Portrait lighting patterns (from photography guides)
LIGHTING_PATTERNS = {
    # Classic portrait lighting
    "butterfly lighting", "paramount lighting",
    "loop lighting",
    "rembrandt lighting",
    "split lighting",
    "broad lighting",
    "short lighting",
    "clamshell lighting",
    "cross lighting",
    "horror lighting",
    # Lighting modifiers
    "key light", "fill light", "rim light", "hair light", "kicker light",
    "background light", "accent light", "practical light",
    # Lighting qualities
    "hard light", "soft light", "diffused light", "specular light",
    "bounced light", "reflected light", "natural light", "available light",
    "mixed lighting", "high contrast lighting", "low contrast lighting",
    # Color/mood lighting
    "warm lighting", "cool lighting", "tungsten lighting", "daylight balanced",
    "golden hour lighting", "blue hour lighting", "magic hour",
    "chiaroscuro", "tenebrism", "film noir lighting",
    # Studio lighting setups
    "one light setup", "two light setup", "three point lighting",
    "high key lighting", "low key lighting", "flat lighting",
    "dramatic lighting", "moody lighting", "cinematic lighting",
    # Effects
    "lens flare", "light leak", "god rays", "volumetric light",
    "caustics", "dappled light", "spotlight", "backlit", "silhouette",
    "rim lit", "edge lit", "top lit", "side lit", "underlit",
}

# Camera shots and angles (from filmmaking guides)
CAMERA_SHOTS = {
    # Shot sizes
    "extreme long shot", "very long shot", "long shot", "wide shot",
    "full shot", "medium long shot", "american shot", "cowboy shot",
    "medium shot", "medium close-up", "close-up", "big close-up",
    "extreme close-up", "macro shot", "detail shot", "insert shot",
    # Angles
    "eye level", "high angle", "low angle", "bird's eye view",
    "overhead shot", "top down", "worm's eye view", "dutch angle",
    "canted angle", "tilted angle", "straight on", "three-quarter view",
    # Special shots
    "establishing shot", "master shot", "over the shoulder", "ots shot",
    "two shot", "three shot", "group shot", "reaction shot",
    "point of view", "pov shot", "subjective shot",
    "cutaway", "cut-in", "pickup shot",
    # Camera movement
    "tracking shot", "dolly shot", "crane shot", "jib shot",
    "steadicam", "handheld", "locked off", "static shot",
    "pan", "tilt", "zoom", "push in", "pull out",
    # Framing
    "headroom", "lead room", "looking room", "negative space",
    "rule of thirds", "golden ratio", "centered composition",
    "symmetrical", "asymmetrical", "balanced", "unbalanced",
    "tight framing", "loose framing", "clean single",
}

# Quality and rendering terms (from SD prompts)
QUALITY_TERMS = {
    # Rendering engines
    "octane render", "unreal engine", "cinema4d", "blender",
    "vray", "arnold render", "redshift", "keyshot",
    "houdini render", "maxwell render", "corona render",
    # Rendering techniques
    "raytracing", "path tracing", "global illumination",
    "ambient occlusion", "subsurface scattering", "caustics",
    "volumetric lighting", "volumetric fog", "god rays",
    "depth of field", "motion blur", "lens distortion",
    "chromatic aberration", "film grain", "noise",
    # Quality descriptors
    "photorealistic", "hyperrealistic", "ultra realistic",
    "highly detailed", "intricate details", "fine details",
    "sharp focus", "tack sharp", "crisp", "clean",
    "4k", "8k", "uhd", "high resolution", "hi-res",
    "studio quality", "professional", "masterpiece",
    "award winning", "trending on artstation",
    # Materials/textures
    "quixel megascans", "pbr textures", "realistic textures",
    "displacement mapping", "normal mapping", "bump mapping",
}

# Photography styles and techniques
PHOTOGRAPHY_STYLES = {
    # Portrait styles
    "headshot", "portrait", "environmental portrait",
    "candid portrait", "lifestyle portrait", "editorial portrait",
    "beauty shot", "glamour shot", "boudoir",
    # Photography types
    "street photography", "documentary", "photojournalism",
    "fine art photography", "conceptual photography",
    "fashion photography", "commercial photography",
    "product photography", "food photography", "still life",
    "landscape photography", "architectural photography",
    "wildlife photography", "macro photography", "astrophotography",
    # Techniques
    "long exposure", "double exposure", "multiple exposure",
    "high speed photography", "time lapse", "light painting",
    "infrared photography", "black and white", "monochrome",
    "selective color", "cross processing", "hdr",
    # Film/vintage
    "35mm film", "medium format", "large format",
    "polaroid", "instant film", "lomography", "holga",
    "kodachrome", "portra", "cinestill", "fujifilm",
    "expired film", "film grain", "light leaks",
}

# Environment/setting terms
ENVIRONMENT_TERMS = {
    # Weather
    "sunny", "overcast", "cloudy", "foggy", "misty", "hazy",
    "rainy", "stormy", "snowy", "windy",
    # Time of day
    "dawn", "sunrise", "morning", "midday", "afternoon",
    "sunset", "dusk", "twilight", "golden hour", "blue hour",
    "night", "midnight", "moonlit",
    # Seasons
    "spring", "summer", "autumn", "fall", "winter",
    # Urban
    "city street", "downtown", "urban", "suburban", "industrial",
    "rooftop", "alleyway", "parking garage", "subway station",
    # Nature
    "forest", "woodland", "meadow", "field", "grassland",
    "mountain", "hillside", "valley", "canyon", "cliff",
    "beach", "coastline", "ocean", "lake", "river", "waterfall",
    "desert", "tundra", "jungle", "rainforest",
}


def load_existing_vocabulary(path: Path) -> Dict[str, Set[str]]:
    """Load existing vocabulary file."""
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {cat: set(terms) for cat, terms in data.items()}


def download_danbooru_tags() -> Dict[str, Set[str]]:
    """Download Danbooru tags from Hugging Face.

    Returns organized tags by category.
    """
    print("Downloading Danbooru tags from Hugging Face...")

    try:
        from datasets import load_dataset

        # Load the dataset
        dataset = load_dataset("isek-ai/danbooru-tags-2024", split="train")

        # Organize by category
        tags_by_category = {
            "subject": set(),
            "subject_details": set(),
            "action_pose": set(),
            "environment": set(),
            "style_medium": set(),
        }

        # Category mapping based on Danbooru tag categories
        # 0: general, 1: artist, 3: copyright, 4: character, 5: meta
        for item in dataset:
            tag_name = item.get("name", "").replace("_", " ")
            category = item.get("category", 0)
            post_count = item.get("post_count", 0)

            # Skip rare tags and non-general categories
            if post_count < 1000 or category != 0:
                continue

            # Skip very long tags (likely descriptions)
            if len(tag_name) > 40:
                continue

            # Classify based on tag name patterns
            tag_lower = tag_name.lower()

            # Action/Pose patterns
            if any(p in tag_lower for p in ["sitting", "standing", "lying", "walking",
                                            "running", "holding", "looking", "pose",
                                            "arms", "legs", "hand"]):
                tags_by_category["action_pose"].add(tag_name)

            # Environment patterns
            elif any(p in tag_lower for p in ["background", "outdoors", "indoors",
                                              "sky", "water", "forest", "city",
                                              "room", "street"]):
                tags_by_category["environment"].add(tag_name)

            # Style patterns
            elif any(p in tag_lower for p in ["style", "anime", "realistic",
                                              "sketch", "painting", "render"]):
                tags_by_category["style_medium"].add(tag_name)

            # Subject details (hair, eyes, clothing, etc.)
            elif any(p in tag_lower for p in ["hair", "eyes", "dress", "shirt",
                                              "pants", "skirt", "shoes", "hat",
                                              "glasses", "jewelry", "skin"]):
                tags_by_category["subject_details"].add(tag_name)

            # Default to subject
            else:
                tags_by_category["subject"].add(tag_name)

        total = sum(len(v) for v in tags_by_category.values())
        print(f"  Downloaded {total} Danbooru tags")

        return tags_by_category

    except ImportError:
        print("  Warning: 'datasets' library not installed. Skipping Danbooru download.")
        print("  Install with: pip install datasets")
        return {}
    except Exception as e:
        print(f"  Error downloading Danbooru tags: {e}")
        return {}


def merge_vocabularies(
    existing: Dict[str, Set[str]],
    *additions: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """Merge multiple vocabulary dictionaries."""
    result = {cat: set(terms) for cat, terms in existing.items()}

    for vocab in additions:
        for category, terms in vocab.items():
            if category not in result:
                result[category] = set()
            result[category].update(terms)

    return result


def create_manual_vocabulary() -> Dict[str, Set[str]]:
    """Create vocabulary from manual additions."""
    return {
        "lighting": LIGHTING_PATTERNS,
        "technical": CAMERA_SHOTS,
        "quality_boosters": QUALITY_TERMS,
        "style_medium": PHOTOGRAPHY_STYLES,
        "environment": ENVIRONMENT_TERMS,
    }


def save_vocabulary(vocab: Dict[str, Set[str]], path: Path) -> None:
    """Save vocabulary to JSON file."""
    # Convert sets to sorted lists
    output = {
        cat: sorted(list(terms))
        for cat, terms in vocab.items()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance vocabulary with external datasets"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: classifier/vocabulary.json)"
    )
    parser.add_argument(
        "--skip-danbooru",
        action="store_true",
        help="Skip downloading Danbooru tags"
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_vocab_path = project_root / "core" / "compose" / "classifier" / "vocabulary.json"

    output_path = Path(args.output) if args.output else default_vocab_path

    print(f"Enhancing vocabulary...")
    print(f"  Output: {output_path}")

    # Load existing vocabulary
    existing = load_existing_vocabulary(output_path)
    print(f"  Existing vocabulary: {sum(len(v) for v in existing.values())} terms")

    # Create manual vocabulary additions
    manual = create_manual_vocabulary()
    print(f"  Manual additions: {sum(len(v) for v in manual.values())} terms")

    # Download Danbooru tags (optional)
    danbooru = {}
    if not args.skip_danbooru:
        danbooru = download_danbooru_tags()

    # Merge all vocabularies
    merged = merge_vocabularies(existing, manual, danbooru)

    # Report stats
    print(f"\nFinal vocabulary:")
    total = 0
    for cat, terms in sorted(merged.items()):
        print(f"  {cat}: {len(terms)} terms")
        total += len(terms)
    print(f"  TOTAL: {total} terms")

    # Save
    save_vocabulary(merged, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
