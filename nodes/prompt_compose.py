"""SID_PromptCompose node - Hybrid prompt composition.

Stage 1: Extract high-confidence missing tags from canonical structure
Stage 2: Assemble prompt with VLM description + bracketed technical tags
Stage 3: LLM Enhancement (optional) - Polish or deep enhance with AI

Features:
- Preserves original VLM description 100%
- Conflict resolution with configurable threshold
- Synonym deduplication
- Meta tag filtering
- Detailed coverage and conflict reports
- LLM enhancement: Local (polish) or API (deep analysis with image)
- Generation modes: 1-Shot or Multi-Pass (category by category)
"""

import json
import re
import io
import base64
from typing import Tuple, Dict, List, Set, Optional

import torch
import numpy as np
from PIL import Image

from ..core.log import log, log_start, log_end, log_error
from ..core.platform import cleanup_memory


# =============================================================================
# Synonym Groups - Prevent redundant terms
# =============================================================================

SYNONYM_GROUPS = {
    'photography': {'photograph', 'photo (medium)', 'photography', 'photorealistic', 'photo'},
    'realistic': {'realistic', 'real life', '3d', 'photorealistic'},
    'soft_lighting': {'soft light', 'natural lighting', 'soft shadow edges', 'soft lighting'},
    'hard_lighting': {'hard light', 'hard lighting', 'harsh light', 'direct light'},
    'high_key': {'high key', 'high key lighting', 'bright', 'high-key'},
    'low_key': {'low key', 'low key lighting', 'dark', 'low-key'},
    'symmetry': {'horizontal symmetry', 'partial symmetry', 'centered subject', 'symmetrical'},
    'rim_light': {'rim light', 'rim lighting', 'backlit', 'backlighting'},
    'bokeh': {'bokeh', 'shallow depth of field', 'blurred background', 'shallow dof'},
    'monochrome': {'monochrome', 'black and white', 'grayscale', 'b&w'},
}

# Meta tags to skip (not useful for image generation)
META_TAGS = {
    '1girl', '1boy', '2girls', '2boys', 'solo', 'multiple girls', 'multiple boys',
    'person', 'people', 'sensitive', 'questionable', 'explicit', 'safe',
    'highres', 'absurdres', 'incredibly absurdres', 'hi res',
}

# Category extraction limits by detail level
DETAIL_LEVELS = {
    'Low (20 tags)': {
        'lighting': 4,
        'composition': 3,
        'style': 4,
        'subject': 6,
        'scene': 3,
    },
    'Medium (50 tags)': {
        'lighting': 10,
        'composition': 8,
        'style': 12,
        'subject': 15,
        'scene': 5,
    },
    'High (100 tags)': {
        'lighting': 20,
        'composition': 15,
        'style': 25,
        'subject': 30,
        'scene': 10,
    },
    'Max (All)': None,  # No limits
}

DETAIL_LEVEL_OPTIONS = list(DETAIL_LEVELS.keys())

# =============================================================================
# LLM Enhancement Configuration
# =============================================================================

ENHANCE_MODES = ["None", "Local LLM", "API (Premium)"]
GENERATION_MODES = ["1-Shot", "Multi-Pass"]

# Local text-only LLM models (NOT VLM - these are text-only for polishing)
LOCAL_LLM_MODELS = [
    "[Local] Qwen2.5-3B",
    "[Local] Qwen2.5-7B",
    "[Local] Qwen2.5-14B",
]

# API models (can use image for deep analysis)
API_MODELS = [
    "[Anthropic] claude-sonnet-4-20250514",
    "[Anthropic] claude-3-5-sonnet-20241022",
    "[Anthropic] claude-3-5-haiku-20241022",
    "[OpenAI] gpt-4o",
    "[OpenAI] gpt-4o-mini",
    "[Gemini] gemini-2.0-flash-exp",
    "[Gemini] gemini-1.5-pro",
]

# Combined list for dropdown
LLM_MODELS = ["(Auto based on mode)"] + LOCAL_LLM_MODELS + API_MODELS

# Model ID mapping
LOCAL_LLM_MAP = {
    "[Local] Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "[Local] Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "[Local] Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
}

# =============================================================================
# LLM Prompts
# =============================================================================

# Local LLM: Polish assembled prompt into natural prose
POLISH_PROMPT = """Rewrite this image description into natural flowing prose for AI image generation.

CURRENT DESCRIPTION:
{assembled_prompt}

CRITICAL RULES:
1. You MUST include EVERY technical term from the bracketed sections
2. Do NOT summarize or condense - expand if needed
3. Weave ALL lighting terms naturally (e.g., "illuminated by soft front lighting with deep shadows")
4. Weave ALL composition terms naturally (e.g., "composed in portrait orientation with rule of thirds")
5. Weave ALL style terms naturally (e.g., "captured in a realistic photographic style with saturated colors")
6. Include EVERY subject detail mentioned
7. Output should be LONGER than input, not shorter
8. Output ONLY the rewritten description - no explanations

REWRITTEN DESCRIPTION:"""

# API: Single-pass deep analysis with all data
API_ENHANCE_PROMPT = """You are an expert at creating prompts for AI image generation. Your goal is MAXIMUM tag coverage.

## Source Data

### VLM Description:
{vlm_description}

### Detected Tags by Category (MUST ALL BE INCLUDED):
**Lighting:** {lighting_tags}
**Composition:** {composition_tags}
**Style:** {style_tags}
**Subject:** {subject_tags}
**Scene:** {scene_tags}

### Tag Coverage:
- Tags in VLM: {matched_count}
- Missing from VLM: {missing_count}

## CRITICAL Task

Create an enhanced prompt that includes EVERY tag listed above:
1. Start with VLM description as base
2. Weave in EVERY lighting tag (e.g., "front lighting with deep shadows and soft shadow edges")
3. Weave in EVERY composition tag (e.g., "portrait orientation, vertical frame, rule of thirds composition")
4. Weave in EVERY style tag (e.g., "realistic photography with vibrant saturated colors")
5. Weave in EVERY subject detail (physical features, clothing, expression)
6. Weave in EVERY scene element
7. Skip ONLY meta-tags: 1girl, solo, sensitive, explicit, safe
8. Target 300-500 words - be COMPREHENSIVE, not concise
9. Use natural flowing prose, not lists

Output ONLY the enhanced prompt:"""

# Multi-pass: Category-specific prompts - MUST include ALL tags
CATEGORY_PROMPTS = {
    'lighting': """Add lighting details to this description. You MUST include ALL of these lighting tags:

TAGS TO INCLUDE: {tags}

Current description: {current}

Write 2-3 sentences that naturally weave in EVERY lighting tag listed above. Example: "The scene is illuminated by soft front lighting with dramatic deep shadows, creating low-key lighting with butterfly lighting effects and subtle backlighting."

Lighting description:""",

    'composition': """Add composition details to this description. You MUST include ALL of these composition tags:

TAGS TO INCLUDE: {tags}

Current description: {current}

Write 2-3 sentences that naturally weave in EVERY composition tag. Example: "Framed in portrait orientation with a vertical 2:3 aspect ratio, the subject is positioned at the thirds intersection following rule of thirds composition."

Composition description:""",

    'style': """Add style details to this description. You MUST include ALL of these style tags:

TAGS TO INCLUDE: {tags}

Current description: {current}

Write 2-3 sentences that naturally weave in EVERY style tag. Example: "Captured in realistic photography style with vibrant saturated colors, black tones, and a multicolored palette creating a monochromatic yet colorful aesthetic."

Style description:""",

    'subject': """Add subject details to this description. You MUST include ALL of these subject tags:

TAGS TO INCLUDE: {tags}

Current description: {current}

Write 3-4 sentences that naturally weave in EVERY subject tag including physical features, clothing, expression, hair, eyes, skin, and pose details.

Subject description:""",

    'scene': """Add scene/environment details to this description. You MUST include ALL of these scene tags:

TAGS TO INCLUDE: {tags}

Current description: {current}

Write 2-3 sentences that naturally weave in EVERY scene tag including background, atmosphere, and setting details.

Scene description:""",
}

# =============================================================================
# Conflict Detection - Attribute categories that can conflict
# =============================================================================

# Pattern: attribute_name -> list of possible values (as regex patterns)
CONFLICT_ATTRIBUTES = {
    'hair_color': {
        'pattern': r'\b(black|brown|blonde|blond|red|ginger|white|gray|grey|silver|blue|pink|purple|green|orange|auburn|chestnut|platinum|golden|dark|light)\s+hair\b',
        'values': ['black', 'brown', 'blonde', 'blond', 'red', 'ginger', 'white', 'gray', 'grey',
                   'silver', 'blue', 'pink', 'purple', 'green', 'orange', 'auburn', 'chestnut',
                   'platinum', 'golden', 'dark', 'light'],
    },
    'eye_color': {
        'pattern': r'\b(black|brown|blue|green|hazel|gray|grey|amber|red|purple|yellow|golden)\s+eyes?\b',
        'values': ['black', 'brown', 'blue', 'green', 'hazel', 'gray', 'grey', 'amber', 'red',
                   'purple', 'yellow', 'golden'],
    },
    'hair_length': {
        'pattern': r'\b(short|medium|long|very long|shoulder.length|waist.length)\s+hair\b',
        'values': ['short', 'medium', 'long', 'very long', 'shoulder-length', 'waist-length'],
    },
}


def extract_attribute_from_tag(tag: str) -> Optional[Tuple[str, str]]:
    """
    Extract attribute type and value from a tag.

    Returns:
        (attribute_type, value) or None if not an attribute tag
    """
    tag_lower = tag.lower()

    for attr_type, config in CONFLICT_ATTRIBUTES.items():
        for value in config['values']:
            # Check if tag contains this attribute value
            if attr_type == 'hair_color' and f"{value} hair" in tag_lower:
                return (attr_type, value)
            elif attr_type == 'eye_color' and (f"{value} eyes" in tag_lower or f"{value} eye" in tag_lower):
                return (attr_type, value)
            elif attr_type == 'hair_length' and f"{value} hair" in tag_lower:
                return (attr_type, value)

    return None


def extract_attributes_from_text(text: str) -> Dict[str, str]:
    """
    Extract all attribute values from VLM text.

    Returns:
        Dict mapping attribute_type -> value found in text
    """
    text_lower = text.lower()
    found = {}

    for attr_type, config in CONFLICT_ATTRIBUTES.items():
        match = re.search(config['pattern'], text_lower)
        if match:
            # Extract the color/value part (first group or full match)
            value = match.group(1) if match.groups() else match.group(0).split()[0]
            found[attr_type] = value

    return found


def check_conflict(tag: str, vlm_attributes: Dict[str, str]) -> Optional[Dict]:
    """
    Check if a tag conflicts with VLM attributes.

    Returns:
        Conflict info dict or None if no conflict
    """
    tag_attr = extract_attribute_from_tag(tag)

    if tag_attr is None:
        return None  # Not an attribute tag, no conflict possible

    attr_type, tag_value = tag_attr

    if attr_type in vlm_attributes:
        vlm_value = vlm_attributes[attr_type]
        if tag_value != vlm_value:
            return {
                'attribute': attr_type,
                'tag_value': tag_value,
                'vlm_value': vlm_value,
                'tag': tag,
            }

    return None


# =============================================================================
# Stage 1: Extract Missing Tags with Conflict Resolution
# =============================================================================

def extract_missing_tags(
    canonical: Dict,
    tag_coverage: Dict,
    vlm_description: str,
    category_limits: Optional[Dict[str, int]],
    conflict_threshold: float = 0.9,
    verbose: bool = False
) -> Tuple[Dict[str, List[str]], List[Dict]]:
    """
    Extract tags that are missing from VLM description with conflict resolution.

    Args:
        canonical: Canonical structure from Analysis (already filtered by tag_filter)
        tag_coverage: Tag coverage analysis from Synthesis (uses matched_tags)
        vlm_description: Original VLM description for conflict detection
        category_limits: Dict of category -> max tags, or None for no limits
        conflict_threshold: Tags with confidence >= this override VLM on conflicts
        verbose: Enable detailed logging

    Returns:
        Tuple of:
        - Dict mapping category -> list of extracted tags
        - List of conflict reports
    """
    extracted = {}
    conflicts = []
    used_synonyms: Set[str] = set()

    # Get matched tags - we'll extract tags NOT in this set
    matched_tags = tag_coverage.get('matched_tags', {})
    matched_set = set(matched_tags.keys())

    # Extract attributes from VLM for conflict detection
    vlm_attributes = extract_attributes_from_text(vlm_description)

    # Categories to process
    categories = ['lighting', 'composition', 'style', 'subject', 'scene']

    for category in categories:
        if category not in canonical:
            continue

        category_data = canonical[category]
        if not isinstance(category_data, dict):
            continue

        # Get limit for this category (None = no limit)
        limit = category_limits.get(category) if category_limits else None

        # Sort by confidence descending (highest first)
        sorted_tags = sorted(
            category_data.items(),
            key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0
        )

        category_tags = []

        for tag, confidence in sorted_tags:
            # Skip if already in VLM description (matched)
            if tag in matched_set:
                continue

            # Skip meta tags
            if tag.lower() in META_TAGS:
                continue

            # Check for conflict with VLM
            conflict = check_conflict(tag, vlm_attributes)
            if conflict:
                conflict['confidence'] = confidence
                conflict['category'] = category

                if confidence >= conflict_threshold:
                    # Tag wins - include it
                    conflict['resolution'] = 'tag_wins'
                    conflicts.append(conflict)
                else:
                    # VLM wins - skip tag
                    conflict['resolution'] = 'vlm_wins'
                    conflicts.append(conflict)
                    continue  # Skip this tag

            # Check synonym collision
            synonym_key = None
            tag_lower = tag.lower()
            for key, group in SYNONYM_GROUPS.items():
                if tag_lower in group or tag in group:
                    synonym_key = key
                    break

            if synonym_key and synonym_key in used_synonyms:
                continue  # Skip - already have a synonym

            if synonym_key:
                used_synonyms.add(synonym_key)

            category_tags.append(tag)

            # Check limit (None = no limit)
            if limit is not None and len(category_tags) >= limit:
                break

        if category_tags:
            extracted[category] = category_tags

    if verbose:
        total = sum(len(tags) for tags in extracted.values())
        log("Compose", f"Extracted {total} tags from {len(extracted)} categories", force=True)
        if conflicts:
            log("Compose", f"Detected {len(conflicts)} conflicts", force=True)

    return extracted, conflicts


# =============================================================================
# Stage 2: Assemble Prompt
# =============================================================================

def assemble_prompt(
    vlm_description: str,
    extracted_tags: Dict[str, List[str]],
    verbose: bool = False
) -> str:
    """
    Assemble final prompt: VLM description + bracketed technical tags.

    Args:
        vlm_description: Original VLM description (preserved 100%)
        extracted_tags: Missing tags by category from Stage 1
        verbose: Enable detailed logging

    Returns:
        Assembled prompt string
    """
    sections = [vlm_description.strip()]

    # Build bracketed tag sections
    tag_sections = []

    # Order: Lighting, Composition, Style, Scene, Subject details
    category_order = ['lighting', 'composition', 'style', 'scene', 'subject']
    category_labels = {
        'lighting': 'Lighting',
        'composition': 'Composition',
        'style': 'Style',
        'scene': 'Scene',
        'subject': 'Details',
    }

    for category in category_order:
        if category in extracted_tags and extracted_tags[category]:
            label = category_labels.get(category, category.title())
            tags_str = ', '.join(extracted_tags[category])
            tag_sections.append(f"[{label}: {tags_str}]")

    if tag_sections:
        sections.append('\n'.join(tag_sections))

    result = '\n\n'.join(sections)

    if verbose:
        log("Compose", f"Assembled: {len(vlm_description)} chars VLM + {len(tag_sections)} sections", force=True)

    return result


# =============================================================================
# Reports
# =============================================================================

def print_coverage_report(
    tag_coverage: Dict,
    extracted_tags: Dict[str, List[str]],
    conflicts: List[Dict],
    detail_level: str,
    conflict_threshold: float
):
    """Print detailed coverage and conflict report."""

    total_tags = tag_coverage.get('total_tags', 0)
    matched_count = tag_coverage.get('matched_count', 0)
    missing_count = tag_coverage.get('missing_count', 0)

    extracted_count = sum(len(tags) for tags in extracted_tags.values())

    print("\n" + "=" * 60)
    print(f"Compose: Tag Coverage Report ({detail_level})")
    print("=" * 60)
    print(f"{'Total Canonical Tags:':<30} {total_tags}")
    print(f"{'Already in VLM:':<30} {matched_count}")
    print(f"{'Missing from VLM:':<30} {missing_count}")
    print(f"{'Extracted to Brackets:':<30} {extracted_count}")
    print("-" * 60)

    # Show extracted by category
    if extracted_tags:
        print("Extracted Tags by Category:")
        for category, tags in extracted_tags.items():
            print(f"  [{category}]: {', '.join(tags)}")

    # Conflict report
    if conflicts:
        print("-" * 60)
        print(f"Conflicts Detected: {len(conflicts)} (threshold: {conflict_threshold})")
        print("-" * 60)

        tag_wins = [c for c in conflicts if c['resolution'] == 'tag_wins']
        vlm_wins = [c for c in conflicts if c['resolution'] == 'vlm_wins']

        if tag_wins:
            print(f"  Tag Wins ({len(tag_wins)}):")
            for c in tag_wins:
                print(f"    • {c['tag']} ({c['confidence']:.2f}) overrides VLM '{c['vlm_value']} {c['attribute'].replace('_', ' ')}'")

        if vlm_wins:
            print(f"  VLM Wins ({len(vlm_wins)}):")
            for c in vlm_wins:
                print(f"    • VLM '{c['vlm_value']} {c['attribute'].replace('_', ' ')}' kept, skipped tag '{c['tag']}' ({c['confidence']:.2f})")
    else:
        print("-" * 60)
        print("No conflicts detected")

    print("=" * 60 + "\n")


# =============================================================================
# Word Composition Analysis
# =============================================================================

def normalize_word(word: str) -> str:
    """Normalize a word for comparison."""
    # Remove punctuation and lowercase
    return re.sub(r'[^\w\s]', '', word.lower()).strip()


def extract_words(text: str) -> Set[str]:
    """Extract meaningful words from text."""
    if not text:
        return set()

    # Split and normalize
    words = set()
    for word in text.split():
        normalized = normalize_word(word)
        if len(normalized) >= 3:  # Skip very short words
            words.add(normalized)
    return words


def extract_tag_words(canonical: Dict) -> Dict[str, Set[str]]:
    """Extract words from canonical tags by category."""
    category_words = {}
    all_words = set()

    for category, tags in canonical.items():
        if not isinstance(tags, dict):
            continue

        words = set()
        for tag in tags.keys():
            # Split tag into words (handle underscores and spaces)
            tag_normalized = tag.replace('_', ' ').replace('-', ' ')
            for word in tag_normalized.split():
                normalized = normalize_word(word)
                if len(normalized) >= 3:
                    words.add(normalized)
                    all_words.add(normalized)

        if words:
            category_words[category] = words

    category_words['_all'] = all_words
    return category_words


def analyze_word_composition(
    canonical: Dict,
    vlm_description: str,
    florence_caption: str,
    florence_description: str,
    final_prompt: str,
    enhance_mode: str = "None"
) -> Dict:
    """
    Analyze word composition across all prompt sources.

    Returns dict with coverage statistics for each source.
    """
    # Extract tag words by category
    tag_words = extract_tag_words(canonical)
    all_tag_words = tag_words.get('_all', set())

    # Extract words from each text source
    vlm_words = extract_words(vlm_description)
    florence_cap_words = extract_words(florence_caption)
    florence_desc_words = extract_words(florence_description)
    final_words = extract_words(final_prompt)

    # Calculate coverage for each source
    def calc_coverage(source_words: Set[str], tag_words: Set[str]) -> Dict:
        if not tag_words:
            return {"matched": 0, "total": 0, "coverage": 0.0, "matched_words": []}

        matched = source_words & tag_words
        return {
            "matched": len(matched),
            "total": len(tag_words),
            "coverage": len(matched) / len(tag_words) if tag_words else 0,
            "matched_words": sorted(list(matched))[:20],  # Top 20
        }

    # Coverage analysis
    analysis = {
        "tag_words_total": len(all_tag_words),
        "vlm": {
            "word_count": len(vlm_words),
            "tag_coverage": calc_coverage(vlm_words, all_tag_words),
        },
        "florence_caption": {
            "word_count": len(florence_cap_words),
            "tag_coverage": calc_coverage(florence_cap_words, all_tag_words),
        },
        "florence_description": {
            "word_count": len(florence_desc_words),
            "tag_coverage": calc_coverage(florence_desc_words, all_tag_words),
        },
        "final_prompt": {
            "word_count": len(final_words),
            "tag_coverage": calc_coverage(final_words, all_tag_words),
            "enhance_mode": enhance_mode,
        },
    }

    # Category-level coverage for final prompt
    category_coverage = {}
    for category, words in tag_words.items():
        if category == '_all':
            continue
        cov = calc_coverage(final_words, words)
        category_coverage[category] = {
            "coverage": cov["coverage"],
            "matched": cov["matched"],
            "total": cov["total"],
        }
    analysis["final_prompt"]["by_category"] = category_coverage

    # Words in tags but NOT in final prompt
    missing_from_final = all_tag_words - final_words
    analysis["missing_from_final"] = sorted(list(missing_from_final))[:30]

    # Words gained from enhancement (in final but not in VLM)
    if enhance_mode != "None":
        gained_words = final_words - vlm_words
        # Filter to only tag-related words
        gained_tag_words = gained_words & all_tag_words
        analysis["words_gained_from_enhancement"] = sorted(list(gained_tag_words))[:20]

    return analysis


def print_word_composition_report(analysis: Dict):
    """Print word composition analysis report."""

    print("\n" + "=" * 70)
    print("Word Composition Analysis")
    print("=" * 70)

    tag_total = analysis.get("tag_words_total", 0)
    print(f"{'Total Unique Tag Words:':<35} {tag_total}")
    print("-" * 70)

    # Source comparison table
    print(f"{'Source':<25} {'Words':<10} {'Tag Match':<12} {'Coverage':<10}")
    print("-" * 70)

    sources = [
        ("Florence Caption", "florence_caption"),
        ("Florence Description", "florence_description"),
        ("VLM Description", "vlm"),
        ("Final Prompt", "final_prompt"),
    ]

    for label, key in sources:
        data = analysis.get(key, {})
        word_count = data.get("word_count", 0)
        tag_cov = data.get("tag_coverage", {})
        matched = tag_cov.get("matched", 0)
        coverage = tag_cov.get("coverage", 0)

        enhance_note = ""
        if key == "final_prompt":
            mode = data.get("enhance_mode", "None")
            if mode != "None":
                enhance_note = f" [{mode}]"

        print(f"{label + enhance_note:<25} {word_count:<10} {matched}/{tag_total:<10} {coverage:.1%}")

    print("-" * 70)

    # Category breakdown for final prompt
    final_data = analysis.get("final_prompt", {})
    by_category = final_data.get("by_category", {})

    if by_category:
        print("\nFinal Prompt - Coverage by Category:")
        for category, cov in sorted(by_category.items()):
            bar_len = int(cov["coverage"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {category:<15} [{bar}] {cov['coverage']:.0%} ({cov['matched']}/{cov['total']})")

    # Words gained from enhancement
    gained = analysis.get("words_gained_from_enhancement", [])
    if gained:
        print(f"\nWords Gained from Enhancement ({len(gained)}):")
        print(f"  {', '.join(gained[:15])}")
        if len(gained) > 15:
            print(f"  ... and {len(gained) - 15} more")

    # Missing words
    missing = analysis.get("missing_from_final", [])
    if missing:
        print(f"\nTag Words Missing from Final ({len(missing)}):")
        print(f"  {', '.join(missing[:15])}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")

    print("=" * 70 + "\n")


# =============================================================================
# Stage 3: LLM Enhancement
# =============================================================================

class LLMEnhancer:
    """Handles LLM-based prompt enhancement (local and API)."""

    _local_model = None
    _local_tokenizer = None
    _current_model_id = None

    @classmethod
    def _pil_to_base64(cls, image: Image.Image, max_size: int = 1024) -> str:
        """Convert PIL image to base64 string."""
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.standard_b64encode(buffer.read()).decode("utf-8")

    @classmethod
    def _load_local_model(cls, model_name: str, verbose: bool = False):
        """Load local Qwen text model."""
        model_id = LOCAL_LLM_MAP.get(model_name)
        if not model_id:
            raise ValueError(f"Unknown local model: {model_name}")

        if cls._local_model is not None and cls._current_model_id == model_id:
            if verbose:
                log("Compose", f"Reusing loaded model: {model_name}")
            return

        # Unload previous model
        if cls._local_model is not None:
            cls._unload_local_model()

        start = log_start("Compose", f"Loading {model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cls._local_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            cls._local_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            cls._current_model_id = model_id
            log_end("Compose", "Model loaded", start)

        except Exception as e:
            log_error("Compose", f"Failed to load model: {e}")
            raise

    @classmethod
    def _unload_local_model(cls):
        """Unload local model and free VRAM."""
        import gc
        if cls._local_model is not None:
            start = log_start("Compose", "Releasing local LLM")
            del cls._local_model
            del cls._local_tokenizer
            cls._local_model = None
            cls._local_tokenizer = None
            cls._current_model_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            cleanup_memory(aggressive=True)
            log_end("Compose", "Model released", start)

    @classmethod
    def _run_local(cls, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Run inference on local Qwen text model."""
        if cls._local_model is None or cls._local_tokenizer is None:
            raise RuntimeError("Local model not loaded")

        messages = [
            {"role": "system", "content": "You are an expert prompt engineer for AI image generation."},
            {"role": "user", "content": prompt}
        ]

        text = cls._local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = cls._local_tokenizer([text], return_tensors="pt").to(cls._local_model.device)

        with torch.no_grad():
            outputs = cls._local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=cls._local_tokenizer.eos_token_id
            )

        # Decode only new tokens
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        result = cls._local_tokenizer.decode(generated, skip_special_tokens=True)

        return result.strip()

    @classmethod
    def _run_anthropic(cls, prompt: str, model: str, api_key: str, max_tokens: int,
                       temperature: float, image: Optional[Image.Image] = None) -> str:
        """Run Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

        if not api_key:
            raise ValueError("API key required for Anthropic")

        client = anthropic.Anthropic(api_key=api_key)

        content = []
        if image is not None:
            image_b64 = cls._pil_to_base64(image)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}
            })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are an expert prompt engineer for AI image generation.",
            messages=[{"role": "user", "content": content}]
        )

        return response.content[0].text.strip() if response.content else ""

    @classmethod
    def _run_openai(cls, prompt: str, model: str, api_key: str, max_tokens: int,
                    temperature: float, image: Optional[Image.Image] = None) -> str:
        """Run OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        if not api_key:
            raise ValueError("API key required for OpenAI")

        client = OpenAI(api_key=api_key)

        content = []
        if image is not None:
            image_b64 = cls._pil_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer for AI image generation."},
                {"role": "user", "content": content}
            ]
        )

        return response.choices[0].message.content.strip() if response.choices else ""

    @classmethod
    def _run_gemini(cls, prompt: str, model: str, api_key: str, max_tokens: int,
                    temperature: float, image: Optional[Image.Image] = None) -> str:
        """Run Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

        if not api_key:
            raise ValueError("API key required for Gemini")

        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)

        content = [prompt]
        if image is not None:
            content.append(image)

        response = model_instance.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )

        return response.text.strip() if response.text else ""

    @classmethod
    def enhance_1shot(
        cls,
        assembled_prompt: str,
        vlm_description: str,
        extracted_tags: Dict[str, List[str]],
        tag_coverage: Dict,
        enhance_mode: str,
        llm_model: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        image: Optional[Image.Image] = None,
        verbose: bool = False
    ) -> str:
        """
        1-Shot enhancement: Single pass to polish/enhance prompt.

        Local LLM: Polish assembled prompt into natural prose
        API: Deep analysis with all raw data + optional image
        """
        if enhance_mode == "Local LLM":
            # Polish approach - take assembled prompt, rewrite naturally
            prompt = POLISH_PROMPT.format(assembled_prompt=assembled_prompt)

            cls._load_local_model(llm_model, verbose)
            start = log_start("Compose", "Polishing with local LLM")
            result = cls._run_local(prompt, max_tokens, temperature)
            log_end("Compose", "Polish complete", start, f"{len(result)} chars")

            return result

        elif enhance_mode == "API (Premium)":
            # Deep analysis with all data
            prompt = API_ENHANCE_PROMPT.format(
                vlm_description=vlm_description,
                lighting_tags=", ".join(extracted_tags.get('lighting', [])) or "none detected",
                composition_tags=", ".join(extracted_tags.get('composition', [])) or "none detected",
                style_tags=", ".join(extracted_tags.get('style', [])) or "none detected",
                subject_tags=", ".join(extracted_tags.get('subject', [])) or "none detected",
                scene_tags=", ".join(extracted_tags.get('scene', [])) or "none detected",
                matched_count=tag_coverage.get('matched_count', 0),
                missing_count=tag_coverage.get('missing_count', 0)
            )

            # Determine provider from model name
            model_id = llm_model.split("] ")[1] if "] " in llm_model else llm_model

            start = log_start("Compose", f"Enhancing with {llm_model}")

            if llm_model.startswith("[Anthropic]"):
                result = cls._run_anthropic(prompt, model_id, api_key, max_tokens, temperature, image)
            elif llm_model.startswith("[OpenAI]"):
                result = cls._run_openai(prompt, model_id, api_key, max_tokens, temperature, image)
            elif llm_model.startswith("[Gemini]"):
                result = cls._run_gemini(prompt, model_id, api_key, max_tokens, temperature, image)
            else:
                raise ValueError(f"Unknown API model: {llm_model}")

            log_end("Compose", "Enhancement complete", start, f"{len(result)} chars")
            return result

        return assembled_prompt  # Fallback

    @classmethod
    def enhance_multipass(
        cls,
        vlm_description: str,
        extracted_tags: Dict[str, List[str]],
        enhance_mode: str,
        llm_model: str,
        api_key: str,
        max_tokens_per_category: int,
        temperature: float,
        image: Optional[Image.Image] = None,
        verbose: bool = False
    ) -> str:
        """
        Multi-Pass enhancement: Generate each category step by step.

        Processes categories in order: subject, scene, lighting, composition, style
        Builds up the description incrementally.
        """
        category_order = ['subject', 'scene', 'lighting', 'composition', 'style']

        # Start with VLM description as base
        current_description = vlm_description
        category_additions = {}

        start = log_start("Compose", f"Multi-pass enhancement ({len(category_order)} categories)")

        for category in category_order:
            tags = extracted_tags.get(category, [])
            if not tags:
                continue

            cat_prompt = CATEGORY_PROMPTS.get(category, "").format(
                current=current_description[:500] + "..." if len(current_description) > 500 else current_description,
                tags=", ".join(tags)
            )

            if verbose:
                log("Compose", f"Processing {category}: {len(tags)} tags")

            if enhance_mode == "Local LLM":
                cls._load_local_model(llm_model, verbose)
                addition = cls._run_local(cat_prompt, max_tokens_per_category, temperature)
            else:
                # API mode
                model_id = llm_model.split("] ")[1] if "] " in llm_model else llm_model

                if llm_model.startswith("[Anthropic]"):
                    addition = cls._run_anthropic(cat_prompt, model_id, api_key, max_tokens_per_category, temperature, image if category == 'subject' else None)
                elif llm_model.startswith("[OpenAI]"):
                    addition = cls._run_openai(cat_prompt, model_id, api_key, max_tokens_per_category, temperature, image if category == 'subject' else None)
                elif llm_model.startswith("[Gemini]"):
                    addition = cls._run_gemini(cat_prompt, model_id, api_key, max_tokens_per_category, temperature, image if category == 'subject' else None)
                else:
                    addition = ""

            if addition:
                category_additions[category] = addition.strip()
                if verbose:
                    log("Compose", f"  {category}: +{len(addition)} chars")

        log_end("Compose", "Multi-pass complete", start, f"{len(category_additions)} categories enhanced")

        # Assemble final prompt
        result_parts = [vlm_description.strip()]
        for category in category_order:
            if category in category_additions:
                result_parts.append(category_additions[category])

        return " ".join(result_parts)


# =============================================================================
# Node Class
# =============================================================================

class SID_PromptCompose:
    """
    Compose prompts from Analysis + Synthesis metadata.

    Three-stage hybrid approach:
    1. Extract missing tags with conflict resolution
    2. Assemble: VLM description + bracketed technical tags
    3. (Optional) LLM Enhancement: Polish or deep enhance with AI

    Enhancement modes:
    - None: VLM + extracted tags (Stage 1+2 only) - Free
    - Local LLM: Polish assembled prompt into natural prose - Free
    - API (Premium): Single-pass deep analysis with image - $0.01-0.03/image
    """

    CATEGORY = "SID Nodes"
    FUNCTION = "compose"
    RETURN_TYPES = ("SID_METADATA", "STRING")
    RETURN_NAMES = ("metadata", "prompt")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composition_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable prompt composition. If disabled, passes through VLM description only."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image (for pipeline continuity and API enhancement)"
                }),
                "metadata": ("SID_METADATA", {
                    "tooltip": "Metadata from SID_PromptSynthesis (includes canonical + tag_coverage)"
                }),
            },
            "optional": {
                "enhance_mode": (ENHANCE_MODES, {
                    "default": "None",
                    "tooltip": "None=Stage 1+2 only, Local LLM=Polish with Qwen, API=Deep analysis with Claude/GPT/Gemini"
                }),
                "generation_mode": (GENERATION_MODES, {
                    "default": "1-Shot",
                    "tooltip": "1-Shot=Single pass, Multi-Pass=Generate each category step by step"
                }),
                "llm_model": (LLM_MODELS, {
                    "default": "(Auto based on mode)",
                    "tooltip": "LLM model for enhancement. Auto selects based on enhance_mode."
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for Anthropic/OpenAI/Gemini (required for API mode)"
                }),
                "detail_level": (DETAIL_LEVEL_OPTIONS, {
                    "default": "Medium (50 tags)",
                    "tooltip": "How many tags to extract: Low=20, Medium=50, High=100, Max=All"
                }),
                "conflict_threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Tag confidence threshold for conflict resolution. Tags >= threshold override VLM."
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 128,
                    "tooltip": "Max tokens for LLM enhancement output (higher = more comprehensive)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "LLM sampling temperature (lower = more deterministic)"
                }),
                "release_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release local LLM after use"
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed logging"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Seed for reproducibility"
                }),
            },
        }

    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor to PIL Image."""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode="RGB")

    def _select_model(self, enhance_mode: str, llm_model: str) -> str:
        """Select appropriate model based on enhance_mode if Auto selected."""
        if llm_model != "(Auto based on mode)":
            return llm_model

        # Auto selection
        if enhance_mode == "Local LLM":
            return "[Local] Qwen2.5-3B"  # Default local model
        elif enhance_mode == "API (Premium)":
            return "[Anthropic] claude-3-5-haiku-20241022"  # Default API model (cheapest)
        return llm_model

    def compose(
        self,
        composition_enabled: bool = True,
        image=None,
        metadata=None,
        enhance_mode: str = "None",
        generation_mode: str = "1-Shot",
        llm_model: str = "(Auto based on mode)",
        api_key: str = "",
        detail_level: str = "Medium (50 tags)",
        conflict_threshold: float = 0.9,
        max_tokens: int = 512,
        temperature: float = 0.3,
        release_vram: bool = True,
        verbose: bool = False,
        seed: int = 0,
    ) -> Tuple:
        """
        Compose prompt using Stage 1 + Stage 2 + optional Stage 3 LLM enhancement.

        Returns:
            (metadata_json, composed_prompt)
        """
        import random
        import gc

        # Set seed for reproducibility
        seed_32bit = seed % (2**32)
        random.seed(seed_32bit)
        np.random.seed(seed_32bit)
        torch.manual_seed(seed_32bit)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_32bit)

        # Parse metadata
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                log_error("Compose", "Failed to parse metadata JSON")
                metadata_dict = {}
        else:
            metadata_dict = metadata if isinstance(metadata, dict) else {}

        # Get VLM description (preserved 100%)
        vlm_description = metadata_dict.get('vlm_description', '')

        # Passthrough mode when disabled
        if not composition_enabled:
            if verbose:
                log("Compose", "Composition disabled - passthrough VLM description", force=True)
            metadata_out = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
            return (metadata_out, vlm_description)

        start = log_start("Compose", "Composing prompt")

        # Get required data
        canonical = metadata_dict.get('canonical', {})
        tag_coverage = metadata_dict.get('tag_coverage', {})

        if not canonical:
            log_error("Compose", "No canonical structure in metadata")
            metadata_out = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
            return (metadata_out, vlm_description)

        # Get category limits based on detail level
        category_limits = DETAIL_LEVELS.get(detail_level)

        # Stage 1: Extract missing tags with conflict resolution
        extracted_tags, conflicts = extract_missing_tags(
            canonical=canonical,
            tag_coverage=tag_coverage,
            vlm_description=vlm_description,
            category_limits=category_limits,
            conflict_threshold=conflict_threshold,
            verbose=verbose
        )

        # Stage 2: Assemble prompt
        assembled_prompt = assemble_prompt(
            vlm_description=vlm_description,
            extracted_tags=extracted_tags,
            verbose=verbose
        )

        # Print coverage and conflict report
        print_coverage_report(
            tag_coverage=tag_coverage,
            extracted_tags=extracted_tags,
            conflicts=conflicts,
            detail_level=detail_level,
            conflict_threshold=conflict_threshold
        )

        # Stage 3: LLM Enhancement (optional)
        final_prompt = assembled_prompt
        enhancement_info = {"enabled": False}

        if enhance_mode != "None":
            selected_model = self._select_model(enhance_mode, llm_model)

            # Validate model selection matches enhance_mode
            if enhance_mode == "Local LLM" and not selected_model.startswith("[Local]"):
                log_error("Compose", f"Local LLM mode requires local model, got: {selected_model}")
                selected_model = "[Local] Qwen2.5-3B"

            if enhance_mode == "API (Premium)" and selected_model.startswith("[Local]"):
                log_error("Compose", f"API mode requires API model, got: {selected_model}")
                selected_model = "[Anthropic] claude-3-5-haiku-20241022"

            # Convert image tensor to PIL for API calls
            pil_image = None
            if image is not None and enhance_mode == "API (Premium)":
                pil_image = self._tensor_to_pil(image)

            try:
                if generation_mode == "1-Shot":
                    final_prompt = LLMEnhancer.enhance_1shot(
                        assembled_prompt=assembled_prompt,
                        vlm_description=vlm_description,
                        extracted_tags=extracted_tags,
                        tag_coverage=tag_coverage,
                        enhance_mode=enhance_mode,
                        llm_model=selected_model,
                        api_key=api_key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        image=pil_image,
                        verbose=verbose
                    )
                else:  # Multi-Pass
                    final_prompt = LLMEnhancer.enhance_multipass(
                        vlm_description=vlm_description,
                        extracted_tags=extracted_tags,
                        enhance_mode=enhance_mode,
                        llm_model=selected_model,
                        api_key=api_key,
                        max_tokens_per_category=max_tokens // 4,  # Divide tokens among categories
                        temperature=temperature,
                        image=pil_image,
                        verbose=verbose
                    )

                enhancement_info = {
                    "enabled": True,
                    "mode": enhance_mode,
                    "generation_mode": generation_mode,
                    "model": selected_model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "original_length": len(assembled_prompt),
                    "enhanced_length": len(final_prompt),
                }

                if verbose:
                    log("Compose", f"Enhanced: {len(assembled_prompt)} → {len(final_prompt)} chars")

            except Exception as e:
                log_error("Compose", f"LLM enhancement failed: {e}")
                final_prompt = assembled_prompt  # Fallback to Stage 2 output
                enhancement_info = {"enabled": False, "error": str(e)}

            # Release local model VRAM
            if enhance_mode == "Local LLM" and release_vram:
                LLMEnhancer._unload_local_model()

            # Cleanup
            if pil_image is not None:
                del pil_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Word Composition Analysis
        florence_caption = metadata_dict.get('florence_caption', '')
        florence_description = metadata_dict.get('florence_description', '')

        word_analysis = analyze_word_composition(
            canonical=canonical,
            vlm_description=vlm_description,
            florence_caption=florence_caption,
            florence_description=florence_description,
            final_prompt=final_prompt,
            enhance_mode=enhance_mode
        )

        # Print word composition report
        print_word_composition_report(word_analysis)

        # Update metadata with composition info
        metadata_dict['compose_stats'] = {
            'detail_level': detail_level,
            'conflict_threshold': conflict_threshold,
            'extracted_tags': extracted_tags,
            'total_tags_added': sum(len(tags) for tags in extracted_tags.values()),
            'categories_enhanced': list(extracted_tags.keys()),
            'conflicts': conflicts,
            'conflicts_tag_wins': len([c for c in conflicts if c['resolution'] == 'tag_wins']),
            'conflicts_vlm_wins': len([c for c in conflicts if c['resolution'] == 'vlm_wins']),
            'original_vlm_preserved': True,
            'enhancement': enhancement_info,
            'word_analysis': {
                'tag_words_total': word_analysis.get('tag_words_total', 0),
                'vlm_coverage': word_analysis.get('vlm', {}).get('tag_coverage', {}).get('coverage', 0),
                'final_coverage': word_analysis.get('final_prompt', {}).get('tag_coverage', {}).get('coverage', 0),
                'words_gained': len(word_analysis.get('words_gained_from_enhancement', [])),
                'words_missing': len(word_analysis.get('missing_from_final', [])),
            },
        }

        # Log summary
        total_added = metadata_dict['compose_stats']['total_tags_added']
        conflict_count = len(conflicts)
        final_coverage = word_analysis.get('final_prompt', {}).get('tag_coverage', {}).get('coverage', 0)
        enhance_suffix = f" → {enhance_mode}" if enhance_mode != "None" else ""
        log_end("Compose", "Done", start, f"+{total_added} tags, {conflict_count} conflicts, {final_coverage:.0%} word coverage{enhance_suffix}")

        if verbose:
            for cat, tags in extracted_tags.items():
                log("Compose", f"  [{cat}]: {', '.join(tags)}", force=True)

        metadata_out = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
        return (metadata_out, final_prompt)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SID_PromptCompose": SID_PromptCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptCompose": "SID Prompt Compose",
}
