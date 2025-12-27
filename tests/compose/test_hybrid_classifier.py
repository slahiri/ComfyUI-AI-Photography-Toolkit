"""Test the hybrid classification pipeline.

Tests Layer 4 (embeddings) and Layer 4.5 (LLM) classifiers.
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
    print("Hybrid Classifier Test")
    print("=" * 70)

    compose_path = Path(__file__).parent.parent.parent / "core" / "compose"
    setup_package_hierarchy(compose_path)

    # Load modules
    base = load_module("core.compose.tokenizer.base", compose_path / "tokenizer" / "base.py")
    normalizer = load_module("core.compose.tokenizer.normalizer", compose_path / "tokenizer" / "normalizer.py")
    tagger_extractor = load_module("core.compose.tokenizer.tagger_extractor", compose_path / "tokenizer" / "tagger_extractor.py")
    analyzer_extractor = load_module("core.compose.tokenizer.analyzer_extractor", compose_path / "tokenizer" / "analyzer_extractor.py")
    categories = load_module("core.compose.classifier.categories", compose_path / "classifier" / "categories.py")
    caption_extractor = load_module("core.compose.tokenizer.caption_extractor", compose_path / "tokenizer" / "caption_extractor.py")
    classifier_base = load_module("core.compose.classifier.base", compose_path / "classifier" / "base.py")
    deterministic = load_module("core.compose.classifier.deterministic", compose_path / "classifier" / "deterministic.py")
    dictionary = load_module("core.compose.classifier.dictionary", compose_path / "classifier" / "dictionary.py")

    # Try to load embeddings
    embeddings_available = False
    try:
        embeddings = load_module("core.compose.classifier.embeddings", compose_path / "classifier" / "embeddings.py")
        embeddings_available = True
        print("[OK] Embeddings module loaded")
    except Exception as e:
        print(f"[SKIP] Embeddings module not available: {e}")

    classifier_init = load_module("core.compose.classifier.__init__", compose_path / "classifier" / "__init__.py")

    TokenBatch = base.TokenBatch
    ClassifierConfig = classifier_init.ClassifierConfig
    classify_batch = classifier_init.classify_batch
    classify_batch_hybrid = classifier_init.classify_batch_hybrid
    get_classification_stats = classifier_init.get_classification_stats
    resolve_conflicts = classifier_init.resolve_conflicts

    # Test with car metadata (has many uncategorized tokens)
    sample_path = Path(__file__).parent / "sample_metadata_car.json"
    if not sample_path.exists():
        print(f"\nCar metadata not found - skipping test")
        return

    with open(sample_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Extract and normalize tokens
    tagger_batch = tagger_extractor.extract_all_tagger_tokens(metadata, min_confidence=0.0)
    analyzer_batch = analyzer_extractor.extract_all_analyzer_tokens(metadata, min_confidence=0.0)
    caption_batch = caption_extractor.extract_all_caption_tokens(metadata, split_into_clauses=False)

    combined = TokenBatch()
    combined.image_info = metadata.get("image_info", {})
    combined.add_all(tagger_batch.tokens)
    combined.add_all(analyzer_batch.tokens)
    combined.add_all(caption_batch.tokens)

    normalized = normalizer.normalize_batch(combined, min_confidence=0.3, deduplicate=True)
    print(f"\nTokens after normalization: {len(normalized.tokens)}")

    # Test 1: Dictionary-only classification
    print("\n" + "=" * 70)
    print("Test 1: Dictionary-only (Layers 1-3)")
    print("=" * 70)

    config_dict_only = ClassifierConfig(use_embeddings=False, use_llm=False)
    classified_dict = classify_batch_hybrid(normalized, config_dict_only)
    classified_dict = resolve_conflicts(classified_dict)
    stats_dict = get_classification_stats(classified_dict)

    print(f"Processed: {stats_dict['processed_percent']}%")
    print(f"Useful: {stats_dict['categorized_percent']}%")
    print(f"Uncategorized: {stats_dict['uncategorized_count']}")
    print(f"Filtered: {stats_dict['filtered_count']}")

    # Show uncategorized tokens
    uncategorized = classified_dict.get_category(categories.CanonicalCategory.UNCATEGORIZED)
    if uncategorized:
        print(f"\nUncategorized tokens ({len(uncategorized)}):")
        for c in uncategorized[:10]:
            print(f"  - '{c.token.text}' ({c.token.source.value})")

    # Test 2: With embeddings (if available)
    if embeddings_available:
        print("\n" + "=" * 70)
        print("Test 2: Dictionary + Embeddings (Layers 1-4)")
        print("=" * 70)

        try:
            # Check if sentence-transformers is available
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model...")

            config_with_emb = ClassifierConfig(use_embeddings=True, use_llm=False)
            classified_emb = classify_batch_hybrid(normalized, config_with_emb)
            classified_emb = resolve_conflicts(classified_emb)
            stats_emb = get_classification_stats(classified_emb)

            print(f"Processed: {stats_emb['processed_percent']}%")
            print(f"Useful: {stats_emb['categorized_percent']}%")
            print(f"Uncategorized: {stats_emb['uncategorized_count']}")
            print(f"Filtered: {stats_emb['filtered_count']}")

            # Show improvement
            improvement = stats_dict['uncategorized_count'] - stats_emb['uncategorized_count']
            if improvement > 0:
                print(f"\n[OK] Embeddings classified {improvement} additional tokens!")

            # Show by layer
            print(f"\n--- By Classification Layer ---")
            layer_names = {1: "Florence Key", 2: "Source Routing", 3: "Dictionary", 4: "Embeddings", 5: "Uncategorized"}
            for layer, count in sorted(stats_emb['by_layer'].items()):
                layer_name = layer_names.get(layer, f"Layer {layer}")
                print(f"  {layer_name}: {count}")

            # Show what embeddings classified
            emb_classifications = [c for c in classified_emb.all_classifications if c.classifier_layer == 4]
            if emb_classifications:
                print(f"\n--- Embedding Classifications ({len(emb_classifications)}) ---")
                for c in emb_classifications[:10]:
                    print(f"  '{c.token.text[:40]}' -> {c.primary_category.value} ({c.confidence:.2f})")

        except ImportError:
            print("[SKIP] sentence-transformers not installed")
            print("  Install with: pip install sentence-transformers")

    print("\n" + "=" * 70)
    print("Hybrid Classifier Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
