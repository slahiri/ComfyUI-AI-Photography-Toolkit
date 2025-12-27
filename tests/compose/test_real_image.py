"""Test Phase 1 with real image metadata.

Usage:
    # With sample metadata (built-in):
    python tests/compose/test_real_image.py

    # With your own metadata JSON file:
    python tests/compose/test_real_image.py path/to/metadata.json

This script demonstrates token extraction from real metadata.
Note: Full extraction is Phase 2 - this is a preview using simple logic.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load modules directly (bypass torch dependency)
import importlib.util

def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

tokenizer_base = load_module("tokenizer_base", "core/compose/tokenizer/base.py")
categories = load_module("categories", "core/compose/classifier/categories.py")

ImageToken = tokenizer_base.ImageToken
TokenType = tokenizer_base.TokenType
TokenSource = tokenizer_base.TokenSource
TokenBatch = tokenizer_base.TokenBatch

CanonicalCategory = categories.CanonicalCategory
is_meta_pattern = categories.is_meta_pattern
CATEGORY_KEYWORDS = categories.CATEGORY_KEYWORDS
SUBJECT_DETAIL_KEYWORDS = categories.SUBJECT_DETAIL_KEYWORDS
SubjectDetailType = categories.SubjectDetailType


# =============================================================================
# Sample Metadata (from real image analysis)
# =============================================================================

SAMPLE_METADATA = {
    "image_info": {
        "width": 512,
        "height": 768,
        "aspect_ratio": 0.667,
        "orientation": "portrait"
    },
    "wd14": {
        "1girl": 0.98,
        "solo": 0.95,
        "brown_hair": 0.915,
        "long_hair": 0.89,
        "brown_eyes": 0.87,
        "looking_at_viewer": 0.85,
        "white_shirt": 0.82,
        "standing": 0.78,
        "parted_lips": 0.65,
        "simple_background": 0.72,
        "grey_background": 0.68,
        "realistic": 0.91,
        "photo_(medium)": 0.88
    },
    "joytag": {
        "woman": 0.94,
        "brown hair": 0.91,
        "white top": 0.85,
        "standing": 0.80,
        "indoor": 0.75,
        "portrait": 0.82
    },
    "florence_caption": "A young woman with long brown hair stands in a studio setting.",
    "florence_description": "The image shows a young woman with long, wavy brown hair and brown eyes. She is wearing a simple white shirt and standing against a grey background. Her expression is calm with slightly parted lips. The lighting is soft and even, creating a professional portrait look.",
    "florence_analyze": {
        "gender": "female",
        "age": "young adult",
        "camera_angle": "front view",
        "distance_to_camera": "medium shot",
        "clothing": "white shirt",
        "hair_color": "brown",
        "hair_length": "long",
        "background": "grey, simple",
        "facial_expression": "calm, neutral"
    },
    "photography": {
        "portrait_score": 0.92,
        "professional_score": 0.88,
        "lighting_quality": 0.85,
        "composition_score": 0.80
    },
    "iqa": {
        "aesthetic_score": 0.78,
        "technical_score": 0.85,
        "sharpness": 0.82
    }
}


# =============================================================================
# Simple Token Extractor (Preview of Phase 2)
# =============================================================================

def extract_tokens_from_metadata(metadata: dict) -> TokenBatch:
    """Extract tokens from metadata - simple version for testing."""
    batch = TokenBatch()
    batch.image_info = metadata.get("image_info", {})

    # Source mapping for tag-based sources
    TAG_SOURCES = {
        "wd14": TokenSource.WD14,
        "joytag": TokenSource.JOYTAG,
        "pixai": TokenSource.PIXAI,
        "nudenet": TokenSource.NUDENET,
        "photography": TokenSource.PHOTOGRAPHY,
        "iqa": TokenSource.IQA,
        "composition": TokenSource.COMPOSITION,
        "saliency": TokenSource.SALIENCY,
    }

    # Extract from all tag-based sources
    for source_key, token_source in TAG_SOURCES.items():
        source_data = metadata.get(source_key, {})
        if isinstance(source_data, dict):
            for tag, conf in source_data.items():
                # Normalize tag (replace underscores with spaces)
                text = tag.replace("_", " ").replace("(", "").replace(")", "")
                batch.add(ImageToken(
                    text=text,
                    confidence=conf,
                    source=token_source,
                    token_type=TokenType.TAG,
                    metadata={"original_tag": tag}
                ))

    # Extract from fashion sources (prefix with source for context)
    for fashion_key in ["fashion_yolov8", "fashion_yolos", "fashion_segformer"]:
        fashion_data = metadata.get(fashion_key, {})
        if isinstance(fashion_data, dict):
            for tag, conf in fashion_data.items():
                batch.add(ImageToken(
                    text=tag,
                    confidence=conf,
                    source=TokenSource.UNKNOWN,  # Could add FASHION source
                    token_type=TokenType.TAG,
                    metadata={"fashion_source": fashion_key}
                ))

    # Extract from florence_analyze (key-value pairs)
    florence_analyze = metadata.get("florence_analyze", {})
    if isinstance(florence_analyze, dict):
        for key, value in florence_analyze.items():
            if value and str(value).upper() != "NA":
                batch.add(ImageToken(
                    text=str(value),
                    confidence=0.9,
                    source=TokenSource.FLORENCE_ANALYZE,
                    token_type=TokenType.KEY_VALUE,
                    metadata={"key": key}
                ))

    # Extract from florence_caption (as sentence)
    caption = metadata.get("florence_caption", "")
    if caption:
        batch.add(ImageToken(
            text=caption,
            confidence=0.85,
            source=TokenSource.FLORENCE_CAPTION,
            token_type=TokenType.SENTENCE
        ))

    # Extract from florence_description (split into sentences)
    description = metadata.get("florence_description", "")
    if description:
        # Simple sentence split
        sentences = [s.strip() for s in description.replace(".", ".|").split("|") if s.strip()]
        for sentence in sentences:
            # Filter META patterns
            if not is_meta_pattern(sentence):
                batch.add(ImageToken(
                    text=sentence,
                    confidence=0.85,
                    source=TokenSource.FLORENCE_DESCRIPTION,
                    token_type=TokenType.SENTENCE
                ))
            else:
                batch.add(ImageToken(
                    text=sentence,
                    confidence=0.85,
                    source=TokenSource.FLORENCE_DESCRIPTION,
                    token_type=TokenType.SENTENCE,
                    metadata={"filtered": True, "reason": "META pattern"}
                ))

    return batch


def classify_token_simple(token: ImageToken) -> tuple:
    """Simple classification for demo - returns (category, subcategory)."""
    text = token.text.lower()

    # Check subject details first (more specific)
    for subcat, keywords in SUBJECT_DETAIL_KEYWORDS.items():
        if keywords.matches(text):
            return (CanonicalCategory.SUBJECT_DETAILS, subcat)

    # Check main categories
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if keywords.matches(text):
            return (cat, None)

    # Check for META in sentences
    if token.token_type == TokenType.SENTENCE and is_meta_pattern(text):
        return (CanonicalCategory.FILTERED_META, None)

    return (CanonicalCategory.UNCATEGORIZED, None)


def display_tokens(batch: TokenBatch):
    """Display tokens in a readable format."""
    print("=" * 80)
    print("TOKEN EXTRACTION RESULTS")
    print("=" * 80)
    print(f"Total tokens: {len(batch)}")
    print(f"Image: {batch.image_info.get('width', '?')}x{batch.image_info.get('height', '?')}")
    print()

    # Group by source
    by_source = batch.group_by_source()

    for source in TokenSource:
        tokens = by_source.get(source, [])
        if not tokens:
            continue

        print(f"\n--- {source.value.upper()} ({len(tokens)} tokens) ---")
        for token in sorted(tokens, key=lambda t: -t.confidence):
            # Classify for display
            cat, subcat = classify_token_simple(token)
            cat_str = cat.value
            if subcat:
                cat_str = f"{cat.value}/{subcat.value}"

            # Truncate long text
            text = token.text
            if len(text) > 60:
                text = text[:57] + "..."

            # Check if filtered
            filtered = token.metadata.get("filtered", False)
            filter_mark = " [FILTERED]" if filtered else ""

            print(f"  [{token.confidence:.2f}] {text:<60} -> {cat_str}{filter_mark}")

    # Summary by category
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    cat_counts = {}
    for token in batch:
        cat, subcat = classify_token_simple(token)
        key = f"{cat.value}" + (f"/{subcat.value}" if subcat else "")
        cat_counts[key] = cat_counts.get(key, 0) + 1

    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


def main():
    # Check for custom metadata file
    if len(sys.argv) > 1:
        metadata_path = Path(sys.argv[1])
        if metadata_path.exists():
            print(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            print(f"File not found: {metadata_path}")
            print("Using sample metadata instead.")
            metadata = SAMPLE_METADATA
    else:
        print("Using built-in sample metadata.")
        print("To use your own: python test_real_image.py path/to/metadata.json")
        metadata = SAMPLE_METADATA

    print()

    # Extract tokens
    batch = extract_tokens_from_metadata(metadata)

    # Display results
    display_tokens(batch)

    # Save output for inspection
    output_path = PROJECT_ROOT / "tests" / "compose" / "token_output.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TOKEN EXTRACTION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total tokens: {len(batch)}\n\n")

        for token in batch:
            cat, subcat = classify_token_simple(token)
            cat_str = f"{cat.value}" + (f"/{subcat.value}" if subcat else "")
            f.write(f"[{token.source.value}] [{token.confidence:.2f}] {token.text}\n")
            f.write(f"  -> {cat_str}\n")
            if token.metadata.get("filtered"):
                f.write(f"  -> FILTERED: {token.metadata.get('reason')}\n")
            f.write("\n")

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
