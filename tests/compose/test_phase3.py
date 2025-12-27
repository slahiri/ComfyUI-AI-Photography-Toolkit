"""Test Phase 3: Classification with real metadata.

Tests the full tokenization → classification pipeline.
"""

import json
import sys
from pathlib import Path
from types import ModuleType
import importlib.util


def setup_package_hierarchy(compose_path: Path):
    if "core" not in sys.modules:
        core_pkg = ModuleType("core")
        core_pkg.__path__ = [str(compose_path.parent)]
        sys.modules["core"] = core_pkg
    if "core.compose" not in sys.modules:
        compose_pkg = ModuleType("core.compose")
        compose_pkg.__path__ = [str(compose_path)]
        sys.modules["core.compose"] = compose_pkg
    if "core.compose.tokenizer" not in sys.modules:
        tokenizer_pkg = ModuleType("core.compose.tokenizer")
        tokenizer_pkg.__path__ = [str(compose_path / "tokenizer")]
        sys.modules["core.compose.tokenizer"] = tokenizer_pkg
    if "core.compose.classifier" not in sys.modules:
        classifier_pkg = ModuleType("core.compose.classifier")
        classifier_pkg.__path__ = [str(compose_path / "classifier")]
        sys.modules["core.compose.classifier"] = classifier_pkg


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = ".".join(name.split(".")[:-1])
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 70)
    print("Phase 3 Test: Classification Pipeline")
    print("=" * 70)

    compose_path = Path(__file__).parent.parent.parent / "core" / "compose"
    setup_package_hierarchy(compose_path)

    # Load all modules
    base = load_module("core.compose.tokenizer.base", compose_path / "tokenizer" / "base.py")
    normalizer = load_module("core.compose.tokenizer.normalizer", compose_path / "tokenizer" / "normalizer.py")
    tagger_extractor = load_module("core.compose.tokenizer.tagger_extractor", compose_path / "tokenizer" / "tagger_extractor.py")
    analyzer_extractor = load_module("core.compose.tokenizer.analyzer_extractor", compose_path / "tokenizer" / "analyzer_extractor.py")
    categories = load_module("core.compose.classifier.categories", compose_path / "classifier" / "categories.py")
    caption_extractor = load_module("core.compose.tokenizer.caption_extractor", compose_path / "tokenizer" / "caption_extractor.py")
    classifier_base = load_module("core.compose.classifier.base", compose_path / "classifier" / "base.py")
    deterministic = load_module("core.compose.classifier.deterministic", compose_path / "classifier" / "deterministic.py")
    dictionary = load_module("core.compose.classifier.dictionary", compose_path / "classifier" / "dictionary.py")
    classifier_init = load_module("core.compose.classifier.__init__", compose_path / "classifier" / "__init__.py")

    TokenBatch = base.TokenBatch
    CanonicalCategory = categories.CanonicalCategory
    classify_token = classifier_init.classify_token
    classify_batch = classifier_init.classify_batch
    get_classification_stats = classifier_init.get_classification_stats
    resolve_conflicts = classifier_init.resolve_conflicts

    # Test with sample metadata
    sample_files = [
        ("sample_metadata.json", "Calvin Klein Image"),
        ("sample_metadata_saree.json", "Saree Image"),
        ("sample_metadata_cupcake.json", "Cupcake Image"),
        ("sample_metadata_landscape.json", "Landscape Image"),
        ("sample_metadata_car.json", "Ferrari Car Image"),
    ]

    for filename, name in sample_files:
        sample_path = Path(__file__).parent / filename
        if not sample_path.exists():
            print(f"\n{filename} not found - skipping")
            continue

        with open(sample_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print("=" * 70)

        # Phase 2: Extract tokens
        tagger_batch = tagger_extractor.extract_all_tagger_tokens(metadata, min_confidence=0.0)
        analyzer_batch = analyzer_extractor.extract_all_analyzer_tokens(metadata, min_confidence=0.0)
        caption_batch = caption_extractor.extract_all_caption_tokens(metadata, split_into_clauses=False)

        combined = TokenBatch()
        combined.image_info = metadata.get("image_info", {})
        combined.add_all(tagger_batch.tokens)
        combined.add_all(analyzer_batch.tokens)
        combined.add_all(caption_batch.tokens)

        # Normalize
        normalized = normalizer.normalize_batch(combined, min_confidence=0.3, deduplicate=True)
        print(f"\nTokens after normalization: {len(normalized.tokens)}")

        # Phase 3: Classify
        classified = classify_batch(normalized)

        # Resolve conflicts
        classified = resolve_conflicts(classified)

        # Get statistics
        stats = get_classification_stats(classified)

        print(f"\n--- Classification Statistics ---")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Processed (not uncategorized): {stats['processed_percent']}%")
        print(f"Useful content: {stats['categorized_percent']}%")
        print(f"Uncategorized: {stats['uncategorized_count']}")
        print(f"Filtered (META): {stats['filtered_count']}")

        print(f"\n--- By Category ---")
        for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

        print(f"\n--- By Classification Layer ---")
        layer_names = {1: "Florence Key", 2: "Source Routing", 3: "Dictionary", 5: "Uncategorized"}
        for layer, count in sorted(stats['by_layer'].items()):
            layer_name = layer_names.get(layer, f"Layer {layer}")
            print(f"  {layer_name}: {count}")

        if stats['by_subcategory']:
            print(f"\n--- Subject Detail Subcategories ---")
            for subcat, count in sorted(stats['by_subcategory'].items(), key=lambda x: -x[1]):
                print(f"  {subcat}: {count}")

        # Show sample classifications by category
        print(f"\n--- Sample Classifications ---")
        for cat in [CanonicalCategory.SUBJECT, CanonicalCategory.SUBJECT_DETAILS,
                    CanonicalCategory.ACTION_POSE, CanonicalCategory.ENVIRONMENT,
                    CanonicalCategory.LIGHTING]:
            classifications = classified.get_category(cat)
            if classifications:
                print(f"\n  {cat.value}:")
                for c in classifications[:5]:
                    subcat = f"/{c.subcategory.value}" if c.subcategory else ""
                    layer = c.classifier_layer
                    print(f"    [{layer}] {c.token.text[:40]}{subcat}")

        # Show uncategorized tokens
        uncategorized = classified.get_category(CanonicalCategory.UNCATEGORIZED)
        if uncategorized:
            print(f"\n--- Uncategorized ({len(uncategorized)}) ---")
            for c in uncategorized[:10]:
                print(f"    '{c.token.text[:50]}' ({c.token.source.value})")

    print("\n" + "=" * 70)
    print("Phase 3 Testing Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
