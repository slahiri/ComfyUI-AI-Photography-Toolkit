"""
Tag injector for LLM prompts.

Extracts high-confidence tags that aren't explicitly mapped to decisions
and formats them for injection into LLM prompts.
"""

from typing import Any, Dict, List, Set, Tuple
from ..types import TaggerResults


# Tags that are already handled by decision engine detectors
# These should NOT be injected as they're covered by template sections
MAPPED_TAGS: Set[str] = {
    # Subject detection
    "1girl", "1boy", "woman", "man", "female", "male", "lady", "gentleman",
    "2girls", "2boys", "multiple_girls", "multiple_boys", "couple", "group",
    "no_humans", "scenery", "landscape", "nature",
    "animal", "cat", "dog", "bird", "horse", "wildlife",
    "car", "motorcycle", "vehicle", "automobile",
    "product", "object", "item",

    # Gender indicators
    "female_focus", "male_focus",

    # NSFW indicators (handled by NSFW detector)
    "nude", "naked", "nsfw", "explicit", "suggestive", "safe",
    "nipples", "breasts", "exposed", "revealing",

    # Shot type indicators (handled by shot type detector)
    "close-up", "closeup", "close_up", "portrait", "headshot",
    "upper_body", "cowboy_shot", "full_body", "wide_shot",
    "medium_shot", "long_shot",

    # Background indicators (handled by environment trigger)
    "simple_background", "white_background", "plain_background",
    "solid_background", "gradient_background",
    "indoor", "outdoor", "studio",
}

# Tags to always exclude (meta tags, quality tags, etc.)
EXCLUDED_TAGS: Set[str] = {
    # Quality/meta tags
    "high_quality", "best_quality", "masterpiece", "high_resolution",
    "highres", "absurdres", "incredibly_absurdres", "huge_filesize",
    "4k", "8k", "hd", "hq", "detailed",

    # Technical/format tags
    "jpeg_artifacts", "blurry", "cropped", "watermark", "signature",
    "username", "artist_name", "copyright", "dated",

    # Meta descriptors
    "solo", "solo_focus", "looking_at_viewer", "facing_viewer",
    "from_front", "from_behind", "from_side", "from_above", "from_below",

    # Common but unhelpful
    "realistic", "photorealistic", "photo", "photograph",
}


def get_unmapped_tags(
    tagger_results: TaggerResults,
    threshold: float = 0.5,
    max_tags: int = 30,
) -> List[Tuple[str, float]]:
    """
    Extract high-confidence tags not covered by decision engine.

    Args:
        tagger_results: Combined tagger outputs
        threshold: Minimum confidence to include
        max_tags: Maximum number of tags to return

    Returns:
        List of (tag, confidence) tuples sorted by confidence
    """
    all_tags: Dict[str, float] = {}

    # Collect from TAGGERS (wd14, pixai, joytag)
    for tag, conf in tagger_results.wd14.items():
        if conf >= threshold:
            normalized = _normalize_tag(tag)
            if normalized and normalized not in all_tags:
                all_tags[normalized] = conf

    for tag, conf in tagger_results.pixai.items():
        if conf >= threshold:
            normalized = _normalize_tag(tag)
            if normalized and normalized not in all_tags:
                all_tags[normalized] = conf

    for tag, conf in tagger_results.joytag.items():
        if conf >= threshold:
            normalized = _normalize_tag(tag)
            if normalized and normalized not in all_tags:
                all_tags[normalized] = conf

    # Collect from ANALYZERS
    analyzer_tags = _extract_analyzer_tags(tagger_results, threshold)
    for tag, conf in analyzer_tags.items():
        if tag not in all_tags:
            all_tags[tag] = conf

    # Filter out mapped and excluded tags
    filtered_tags = {}
    for tag, conf in all_tags.items():
        tag_lower = tag.lower().replace(" ", "_")

        # Skip if in mapped set
        if tag_lower in MAPPED_TAGS:
            continue

        # Skip if in excluded set
        if tag_lower in EXCLUDED_TAGS:
            continue

        # Skip very short tags
        if len(tag) < 3:
            continue

        # Skip tags with numbers only
        if tag.replace("_", "").isdigit():
            continue

        filtered_tags[tag] = conf

    # Sort by confidence and limit
    sorted_tags = sorted(filtered_tags.items(), key=lambda x: x[1], reverse=True)
    return sorted_tags[:max_tags]


# Female anatomy tags to inject for women subjects (nudenet often misses these)
FEMALE_ANATOMY_TAGS: Dict[str, float] = {
    "breasts": 0.85,
    "cleavage": 0.80,
    "curves": 0.75,
    "feminine_body": 0.80,
    "bust": 0.75,
}


def get_female_anatomy_tags(
    subject_type: str,
    existing_tags: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """
    Get female anatomy tags to inject for women subjects.

    Since NudeNet often fails to detect female anatomy, we inject
    common anatomy tags for women subjects to ensure proper description.

    Args:
        subject_type: The detected subject type (woman, man, person, etc.)
        existing_tags: Already detected tags to avoid duplicates

    Returns:
        List of (tag, confidence) tuples to inject
    """
    # Only inject for female subjects
    if subject_type not in ["woman", "person"]:
        return []

    # Get existing tag names for deduplication
    existing_names = {tag.lower().replace("_", " ") for tag, _ in existing_tags}

    # Filter out tags that already exist
    anatomy_tags = []
    for tag, conf in FEMALE_ANATOMY_TAGS.items():
        tag_normalized = tag.lower().replace("_", " ")
        if tag_normalized not in existing_names:
            anatomy_tags.append((tag, conf))

    return anatomy_tags


def _extract_analyzer_tags(
    tagger_results: TaggerResults,
    threshold: float,
) -> Dict[str, float]:
    """
    Extract descriptive tags from analyzer outputs.

    Args:
        tagger_results: Combined tagger/analyzer outputs
        threshold: Minimum confidence to include

    Returns:
        Dict of tag -> confidence from analyzers
    """
    tags: Dict[str, float] = {}

    # Photography analyzer - lighting style tags
    if tagger_results.photography:
        photo = tagger_results.photography
        if photo.get("low_key", 0) > threshold:
            tags["low-key lighting"] = photo["low_key"]
        if photo.get("high_key", 0) > threshold:
            tags["high-key lighting"] = photo["high_key"]
        if photo.get("warm_tones", 0) > threshold:
            tags["warm color tones"] = photo["warm_tones"]
        if photo.get("cool_tones", 0) > threshold:
            tags["cool color tones"] = photo["cool_tones"]
        if photo.get("dramatic", 0) > threshold:
            tags["dramatic mood"] = photo["dramatic"]
        if photo.get("soft", 0) > threshold:
            tags["soft lighting"] = photo["soft"]

    # Lighting analyzer - specific lighting details
    if tagger_results.lighting:
        lighting = tagger_results.lighting
        if lighting.get("dominant_direction"):
            direction = lighting["dominant_direction"]
            tags[f"{direction} lighting"] = 0.8

        lighting_types = lighting.get("lighting_types", {})
        for light_type, conf in lighting_types.items():
            if conf > threshold:
                tags[f"{light_type} light"] = conf

    # Composition analyzer
    if tagger_results.composition:
        comp = tagger_results.composition
        if comp.get("rule_of_thirds_score", 0) > 0.7:
            tags["rule of thirds composition"] = comp["rule_of_thirds_score"]
        if comp.get("symmetry_score", 0) > 0.7:
            tags["symmetrical composition"] = comp["symmetry_score"]
        if comp.get("centered", False):
            tags["centered subject"] = 0.8
        if comp.get("leading_lines", False):
            tags["leading lines"] = 0.75

    # CLIP camera analyzer
    if tagger_results.clip_camera:
        camera = tagger_results.clip_camera
        if camera.get("best_perspective"):
            perspective = camera["best_perspective"]
            if perspective and perspective != "normal":
                tags[f"{perspective} perspective"] = 0.75

    # Sky/weather analyzer - atmospheric details
    if tagger_results.sky_weather:
        sky = tagger_results.sky_weather
        if sky.get("cloud_type"):
            tags[f"{sky['cloud_type']} clouds"] = sky.get("cloud_score", 0.7)
        if sky.get("atmosphere") and sky.get("atmosphere_score", 0) > threshold:
            atmos = sky["atmosphere"]
            if atmos not in ["normal", "clear"]:
                tags[f"{atmos} atmosphere"] = sky["atmosphere_score"]

    # DeepFashion analyzer - clothing details
    if tagger_results.deepfashion:
        fashion = tagger_results.deepfashion
        clothing_type = fashion.get("clothing_type", {})
        for garment, conf in clothing_type.items() if isinstance(clothing_type, dict) else []:
            if conf > threshold:
                tags[garment] = conf

        style = fashion.get("style")
        if style:
            tags[f"{style} fashion style"] = 0.75

    # Fashion color analyzer
    if tagger_results.fashion_color:
        for item in tagger_results.fashion_color[:3]:  # Top 3 fashion items
            if isinstance(item, dict):
                color = item.get("dominant_color")
                garment = item.get("garment_type")
                if color and garment:
                    tags[f"{color} {garment}"] = item.get("confidence", 0.7)

    return tags


def format_tags_for_prompt(
    tags: List[Tuple[str, float]],
    show_confidence: bool = True,
) -> str:
    """
    Format tags for injection into LLM prompt.

    Args:
        tags: List of (tag, confidence) tuples
        show_confidence: Whether to include confidence percentages

    Returns:
        Formatted string for prompt injection
    """
    if not tags:
        return ""

    if show_confidence:
        # Format: "tag1 (95%), tag2 (87%), tag3 (76%)"
        parts = [f"{_humanize_tag(tag)} ({int(conf * 100)}%)" for tag, conf in tags]
    else:
        # Format: "tag1, tag2, tag3"
        parts = [_humanize_tag(tag) for tag, _ in tags]

    return ", ".join(parts)


def categorize_tags(
    tags: List[Tuple[str, float]],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Categorize tags into semantic groups.

    Args:
        tags: List of (tag, confidence) tuples

    Returns:
        Dict mapping category to list of tags
    """
    categories = {
        "appearance": [],
        "hair": [],
        "eyes": [],
        "clothing": [],
        "expression": [],
        "pose": [],
        "environment": [],
        "lighting": [],
        "style": [],
        "other": [],
    }

    # Category keywords
    hair_keywords = ["hair", "bangs", "ponytail", "braid", "twintails", "bun"]
    eye_keywords = ["eyes", "gaze", "pupils", "eyelashes", "eyebrows"]
    clothing_keywords = ["dress", "shirt", "skirt", "pants", "jacket", "coat",
                         "bikini", "swimsuit", "uniform", "suit", "top", "shorts",
                         "hat", "glasses", "jewelry", "earrings", "necklace"]
    expression_keywords = ["smile", "smiling", "grin", "frown", "crying", "tears",
                          "happy", "sad", "angry", "surprised", "serious", "laugh"]
    pose_keywords = ["standing", "sitting", "lying", "kneeling", "walking",
                    "running", "arms", "legs", "hand", "crossed"]
    environment_keywords = ["sky", "water", "grass", "tree", "flower", "beach",
                           "mountain", "city", "street", "room", "window", "door"]
    lighting_keywords = ["sunlight", "moonlight", "shadow", "glow", "backlight",
                        "spotlight", "reflection", "lens_flare", "bokeh"]
    style_keywords = ["aesthetic", "style", "mood", "atmosphere", "tone",
                     "vintage", "modern", "artistic", "cinematic"]

    for tag, conf in tags:
        tag_lower = tag.lower()
        categorized = False

        for keyword in hair_keywords:
            if keyword in tag_lower:
                categories["hair"].append((tag, conf))
                categorized = True
                break

        if not categorized:
            for keyword in eye_keywords:
                if keyword in tag_lower:
                    categories["eyes"].append((tag, conf))
                    categorized = True
                    break

        if not categorized:
            for keyword in clothing_keywords:
                if keyword in tag_lower:
                    categories["clothing"].append((tag, conf))
                    categorized = True
                    break

        if not categorized:
            for keyword in expression_keywords:
                if keyword in tag_lower:
                    categories["expression"].append((tag, conf))
                    categorized = True
                    break

        if not categorized:
            for keyword in pose_keywords:
                if keyword in tag_lower:
                    categories["pose"].append((tag, conf))
                    categorized = True
                    break

        if not categorized:
            for keyword in environment_keywords:
                if keyword in tag_lower:
                    categories["environment"].append((tag, conf))
                    categorized = True
                    break

        if not categorized:
            for keyword in lighting_keywords:
                if keyword in tag_lower:
                    categories["lighting"].append((tag, conf))
                    categorized = True
                    break

        if not categorized:
            for keyword in style_keywords:
                if keyword in tag_lower:
                    categories["style"].append((tag, conf))
                    categorized = True
                    break

        if not categorized:
            categories["other"].append((tag, conf))

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def get_debug_info(
    tagger_results: TaggerResults,
    threshold: float = 0.5,
    max_tags: int = 30,
) -> Dict[str, Any]:
    """
    Get detailed debug info about tag extraction and filtering.

    Args:
        tagger_results: Combined tagger/analyzer outputs
        threshold: Minimum confidence threshold
        max_tags: Maximum number of tags to inject

    Returns:
        Dict with debug info for metadata
    """
    # Collect all tagger tags
    all_tagger_tags: Dict[str, float] = {}
    for tag, conf in tagger_results.wd14.items():
        if conf >= threshold:
            all_tagger_tags[tag] = conf
    for tag, conf in tagger_results.pixai.items():
        if conf >= threshold and tag not in all_tagger_tags:
            all_tagger_tags[tag] = conf
    for tag, conf in tagger_results.joytag.items():
        if conf >= threshold and tag not in all_tagger_tags:
            all_tagger_tags[tag] = conf

    # Get analyzer-derived tags
    analyzer_tags = _extract_analyzer_tags(tagger_results, threshold)

    # Get filtered (injected) tags
    injected = get_unmapped_tags(tagger_results, threshold, max_tags=max_tags)

    # Determine which tags were filtered out
    all_tags = {**all_tagger_tags}
    for tag, conf in analyzer_tags.items():
        if tag not in all_tags:
            all_tags[tag] = conf

    injected_set = {tag for tag, _ in injected}
    filtered_out = []
    for tag, conf in all_tags.items():
        normalized = tag.lower().replace(" ", "_")
        if tag not in injected_set and _normalize_tag(tag) not in injected_set:
            reason = "unknown"
            if normalized in MAPPED_TAGS:
                reason = "mapped_to_decision"
            elif normalized in EXCLUDED_TAGS:
                reason = "excluded_meta_tag"
            elif len(tag) < 3:
                reason = "too_short"
            filtered_out.append({
                "tag": tag,
                "confidence": round(conf, 3),
                "reason": reason,
            })

    return {
        "threshold": threshold,
        "max_tags": max_tags,
        "tagger_tags_count": len(all_tagger_tags),
        "analyzer_tags_count": len(analyzer_tags),
        "injected_count": len(injected),
        "filtered_count": len(filtered_out),
        "tagger_tags": [
            {"tag": t, "confidence": round(c, 3)}
            for t, c in sorted(all_tagger_tags.items(), key=lambda x: x[1], reverse=True)[:30]
        ],
        "analyzer_tags": [
            {"tag": t, "confidence": round(c, 3)}
            for t, c in sorted(analyzer_tags.items(), key=lambda x: x[1], reverse=True)
        ],
        "injected_tags": [
            {"tag": t, "confidence": round(c, 3)}
            for t, c in injected
        ],
        "filtered_out": sorted(filtered_out, key=lambda x: x["confidence"], reverse=True)[:20],
    }


def _normalize_tag(tag: str) -> str:
    """Normalize tag for comparison."""
    if not tag:
        return ""
    # Replace underscores with spaces, lowercase
    return tag.replace("_", " ").strip().lower()


def _humanize_tag(tag: str) -> str:
    """Convert tag to human-readable format."""
    # Replace underscores with spaces
    humanized = tag.replace("_", " ")
    # Title case for multi-word tags
    if " " in humanized:
        return humanized
    return humanized


def get_fashion_pose_tags(
    pose_results: Dict[str, Any],
) -> List[Tuple[str, float]]:
    """
    Extract fashion pose tags from pose detection results.

    Args:
        pose_results: Pose detection results dict

    Returns:
        List of (tag, confidence) tuples for fashion poses
    """
    tags = []

    if not pose_results or not pose_results.get("detected"):
        return tags

    # Add basic pose type
    pose_type = pose_results.get("pose_type")
    pose_conf = pose_results.get("pose_confidence", 0.5)
    if pose_type and pose_type != "unknown":
        tags.append((f"{pose_type} pose", pose_conf))

    # Add body/face angle information
    body_facing = pose_results.get("body_facing")
    face_facing = pose_results.get("face_facing")

    if body_facing == "camera":
        tags.append(("facing camera", 0.8))
    elif body_facing == "side":
        body_turned = pose_results.get("body_turned", "")
        if body_turned:
            tags.append((f"body turned {body_turned}", 0.75))
        else:
            tags.append(("side view", 0.75))
    elif body_facing == "angled":
        tags.append(("three-quarter view", 0.7))

    if face_facing == "side" or face_facing == "profile":
        face_turned = pose_results.get("face_turned", "")
        if face_turned:
            tags.append((f"looking {face_turned}", 0.8))
        else:
            tags.append(("profile view", 0.75))
    elif face_facing == "angled":
        face_turned = pose_results.get("face_turned", "")
        if face_turned:
            tags.append((f"face angled {face_turned}", 0.7))

    if pose_results.get("back_to_camera"):
        tags.append(("back to camera", 0.8))

    # Add fashion poses
    fashion_poses = pose_results.get("fashion_poses", [])
    for fp in fashion_poses[:5]:  # Top 5 fashion poses
        pose_name = fp.get("pose", "")
        conf = fp.get("confidence", 0.5)

        # Convert pose names to natural language
        pose_descriptions = {
            "contrapposto": "contrapposto S-curve pose",
            "hand_on_hip": "hand on hip",
            "power_pose": "power stance",
            "crossed_legs": "crossed legs",
            "walking": "walking pose",
            "over_shoulder": "over-the-shoulder look",
            "leaning": "leaning pose",
            "arms_crossed": "arms crossed",
            "weight_shift": "weight shifted to one leg",
            "hands_behind_back": "hands behind back",
            "looking_away": "looking away from camera",
            "back_to_camera": "back facing camera",
        }

        desc = pose_descriptions.get(pose_name, pose_name.replace("_", " "))
        if desc:
            tags.append((desc, conf))

    # Add arm position
    if pose_results.get("arms_raised"):
        tags.append(("arms raised", 0.8))
    elif pose_results.get("arms_position") == "extended":
        tags.append(("arms extended", 0.7))

    if pose_results.get("hand_near_face"):
        tags.append(("hand near face", 0.75))

    # Deduplicate and sort by confidence
    seen = set()
    unique_tags = []
    for tag, conf in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append((tag, conf))

    unique_tags.sort(key=lambda x: x[1], reverse=True)
    return unique_tags


def format_pose_for_prompt(pose_results: Dict[str, Any]) -> str:
    """
    Format pose detection results for LLM prompt injection.

    Args:
        pose_results: Pose detection results dict

    Returns:
        Formatted string describing the pose
    """
    if not pose_results or not pose_results.get("detected"):
        return ""

    parts = []

    # Body pose
    pose_type = pose_results.get("pose_type", "unknown")
    if pose_type != "unknown":
        parts.append(f"{pose_type}")

    # Body orientation
    body_facing = pose_results.get("body_facing")
    if body_facing:
        body_turned = pose_results.get("body_turned")
        if body_turned:
            parts.append(f"body facing {body_facing} turned {body_turned}")
        else:
            parts.append(f"body facing {body_facing}")

    # Face orientation
    face_facing = pose_results.get("face_facing")
    if face_facing:
        face_turned = pose_results.get("face_turned")
        if face_turned:
            parts.append(f"face in {face_facing} view looking {face_turned}")
        elif face_facing != "camera":
            parts.append(f"{face_facing} face view")

    # Fashion poses
    fashion_poses = pose_results.get("fashion_poses", [])
    if fashion_poses:
        top_pose = fashion_poses[0]
        desc = top_pose.get("description", top_pose.get("pose", "").replace("_", " "))
        if desc:
            parts.append(desc)

    if not parts:
        return ""

    return "Pose: " + ", ".join(parts)
