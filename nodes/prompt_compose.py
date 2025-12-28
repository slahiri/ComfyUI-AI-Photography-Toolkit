"""SID_PromptCompose node - Hybrid prompt composition.

Stage 1: Extract high-confidence missing tags from canonical structure
Stage 2: Assemble prompt with VLM description + bracketed technical tags

Features:
- Preserves original VLM description 100%
- Conflict resolution with configurable threshold
- Synonym deduplication
- Meta tag filtering
- Detailed coverage and conflict reports
"""

import json
import re
from typing import Tuple, Dict, List, Set, Optional

from ..core.log import log, log_start, log_end, log_error


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
# Node Class
# =============================================================================

class SID_PromptCompose:
    """
    Compose prompts from Analysis + Synthesis metadata.

    Two-stage hybrid approach:
    1. Extract missing tags with conflict resolution
    2. Assemble: VLM description + bracketed technical tags

    Preserves original VLM description 100%.
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
                    "tooltip": "Input image (for pipeline continuity)"
                }),
                "metadata": ("SID_METADATA", {
                    "tooltip": "Metadata from SID_PromptSynthesis (includes canonical + tag_coverage)"
                }),
            },
            "optional": {
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

    def compose(
        self,
        composition_enabled: bool = True,
        image=None,
        metadata=None,
        detail_level: str = "Medium (50 tags)",
        conflict_threshold: float = 0.9,
        verbose: bool = False,
        seed: int = 0,
    ) -> Tuple:
        """
        Compose prompt using Stage 1 + Stage 2 with conflict resolution.

        Returns:
            (metadata_json, composed_prompt)
        """
        import random
        import numpy as np
        import torch

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
        composed_prompt = assemble_prompt(
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
        }

        # Log summary
        total_added = metadata_dict['compose_stats']['total_tags_added']
        conflict_count = len(conflicts)
        log_end("Compose", "Done", start, f"+{total_added} tags, {conflict_count} conflicts")

        if verbose:
            for cat, tags in extracted_tags.items():
                log("Compose", f"  [{cat}]: {', '.join(tags)}", force=True)

        metadata_out = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
        return (metadata_out, composed_prompt)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SID_PromptCompose": SID_PromptCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptCompose": "SID Prompt Compose",
}
