"""Test Phase 2: Tokenization with real metadata.

Tests the tokenizer module components without needing torch.
Uses importlib to load modules with proper package context.
"""

import json
import sys
from pathlib import Path
from types import ModuleType
import importlib.util


def setup_package_hierarchy(compose_path: Path):
    """Set up the package hierarchy for relative imports."""
    # Create core package
    if "core" not in sys.modules:
        core_pkg = ModuleType("core")
        core_pkg.__path__ = [str(compose_path.parent)]
        sys.modules["core"] = core_pkg

    # Create core.compose package
    if "core.compose" not in sys.modules:
        compose_pkg = ModuleType("core.compose")
        compose_pkg.__path__ = [str(compose_path)]
        sys.modules["core.compose"] = compose_pkg

    # Create core.compose.tokenizer package
    if "core.compose.tokenizer" not in sys.modules:
        tokenizer_pkg = ModuleType("core.compose.tokenizer")
        tokenizer_pkg.__path__ = [str(compose_path / "tokenizer")]
        sys.modules["core.compose.tokenizer"] = tokenizer_pkg

    # Create core.compose.classifier package
    if "core.compose.classifier" not in sys.modules:
        classifier_pkg = ModuleType("core.compose.classifier")
        classifier_pkg.__path__ = [str(compose_path / "classifier")]
        sys.modules["core.compose.classifier"] = classifier_pkg


def load_module(name: str, path: Path):
    """Load a module with proper package context."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Set __package__ for relative imports
    module.__package__ = ".".join(name.split(".")[:-1])
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_standalone_tests():
    """Run tests that don't need the full package."""
    print("=" * 60)
    print("Phase 2 Standalone Tests")
    print("=" * 60)

    compose_path = Path(__file__).parent.parent.parent / "core" / "compose"

    # Set up package hierarchy
    setup_package_hierarchy(compose_path)

    # Test 1: base.py types
    print("\n[1] Testing base types...")
    base = load_module(
        "core.compose.tokenizer.base",
        compose_path / "tokenizer" / "base.py"
    )

    ImageToken = base.ImageToken
    TokenType = base.TokenType
    TokenSource = base.TokenSource
    TokenBatch = base.TokenBatch
    SOURCE_WEIGHTS = base.SOURCE_WEIGHTS

    token = ImageToken(
        text="test tag",
        confidence=0.9,
        source=TokenSource.WD14,
        token_type=TokenType.TAG
    )
    assert token.weighted_confidence == 0.9 * SOURCE_WEIGHTS[TokenSource.WD14]
    print("  [PASS] ImageToken creation and weighted_confidence")

    batch = TokenBatch()
    batch.add(token)
    assert len(batch.tokens) == 1
    print("  [PASS] TokenBatch add")

    # Test 2: normalizer
    print("\n[2] Testing normalizer...")
    normalizer = load_module(
        "core.compose.tokenizer.normalizer",
        compose_path / "tokenizer" / "normalizer.py"
    )

    assert normalizer.normalize_for_comparison("Test Tag") == "test tag"
    assert normalizer.normalize_for_comparison("  multiple   spaces  ") == "multiple spaces"
    print("  [PASS] normalize_for_comparison")

    # Create duplicate tokens
    t1 = ImageToken("blonde hair", 0.9, TokenSource.WD14, TokenType.TAG)
    t2 = ImageToken("blonde hair", 0.8, TokenSource.JOYTAG, TokenType.TAG)
    deduped = normalizer.deduplicate_tokens([t1, t2])
    assert len(deduped) == 1
    assert deduped[0].metadata.get("merged_from") == 2
    print("  [PASS] deduplicate_tokens")

    agg = normalizer.aggregate_confidence([(0.9, TokenSource.WD14), (0.8, TokenSource.JOYTAG)])
    assert 0.8 < agg < 1.0  # Should have agreement bonus
    print("  [PASS] aggregate_confidence")

    # Test 3: tagger_extractor
    print("\n[3] Testing tagger_extractor...")
    tagger_extractor = load_module(
        "core.compose.tokenizer.tagger_extractor",
        compose_path / "tokenizer" / "tagger_extractor.py"
    )

    assert tagger_extractor.normalize_tag("test_tag") == "test tag"
    assert tagger_extractor.normalize_tag("tag_(medium)") == "tag"
    print("  [PASS] normalize_tag")

    wd14_metadata = {"wd14": {"blonde_hair": 0.95, "blue_eyes": 0.8}}
    tokens = tagger_extractor.extract_from_wd14(wd14_metadata)
    assert len(tokens) == 2
    assert tokens[0].source == TokenSource.WD14
    print("  [PASS] extract_from_wd14")

    # Test 4: analyzer_extractor
    print("\n[4] Testing analyzer_extractor...")
    analyzer_extractor = load_module(
        "core.compose.tokenizer.analyzer_extractor",
        compose_path / "tokenizer" / "analyzer_extractor.py"
    )

    photo_metadata = {"photography": {"natural lighting": 0.9, "shallow depth": 0.7}}
    tokens = analyzer_extractor.extract_from_photography(photo_metadata)
    assert len(tokens) == 2
    assert tokens[0].source == TokenSource.PHOTOGRAPHY
    print("  [PASS] extract_from_photography")

    iqa_metadata = {"iqa": {"high quality": 0.85}}
    tokens = analyzer_extractor.extract_from_iqa(iqa_metadata)
    assert len(tokens) == 1
    print("  [PASS] extract_from_iqa")

    # Test 5: caption_extractor (needs classifier.categories first)
    print("\n[5] Testing caption_extractor...")

    # Load classifier.categories first
    categories = load_module(
        "core.compose.classifier.categories",
        compose_path / "classifier" / "categories.py"
    )

    caption_extractor = load_module(
        "core.compose.tokenizer.caption_extractor",
        compose_path / "tokenizer" / "caption_extractor.py"
    )

    sentences = caption_extractor.split_sentences("First sentence. Second sentence!")
    assert len(sentences) == 2
    print("  [PASS] split_sentences")

    clauses = caption_extractor.split_clauses("A woman wearing a dress, with long hair")
    assert len(clauses) >= 2
    print("  [PASS] split_clauses")

    caption_metadata = {"florence_caption": "A woman in a white dress."}
    tokens = caption_extractor.extract_from_florence_caption(caption_metadata)
    assert len(tokens) == 1
    assert tokens[0].source == TokenSource.FLORENCE_CAPTION
    print("  [PASS] extract_from_florence_caption")

    print("\n" + "=" * 60)
    print("All standalone tests passed!")
    print("=" * 60)

    return {
        "base": base,
        "normalizer": normalizer,
        "tagger_extractor": tagger_extractor,
        "analyzer_extractor": analyzer_extractor,
        "caption_extractor": caption_extractor,
    }


def run_integration_test(modules):
    """Run integration test with real metadata."""
    print("\n" + "=" * 60)
    print("Phase 2 Integration Test (real metadata)")
    print("=" * 60)

    sample_path = Path(__file__).parent / "sample_metadata.json"
    if not sample_path.exists():
        print("sample_metadata.json not found - skipping")
        return

    with open(sample_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    tagger_extractor = modules["tagger_extractor"]
    analyzer_extractor = modules["analyzer_extractor"]
    caption_extractor = modules["caption_extractor"]
    normalizer = modules["normalizer"]
    TokenBatch = modules["base"].TokenBatch

    print("\n--- Tagger Extraction ---")
    tagger_batch = tagger_extractor.extract_all_tagger_tokens(metadata, min_confidence=0.0)
    print(f"Tagger tokens: {len(tagger_batch.tokens)}")

    by_source = {}
    for token in tagger_batch.tokens:
        src = token.source.value
        by_source[src] = by_source.get(src, 0) + 1
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    print("\n--- Analyzer Extraction ---")
    analyzer_batch = analyzer_extractor.extract_all_analyzer_tokens(metadata, min_confidence=0.0)
    print(f"Analyzer tokens: {len(analyzer_batch.tokens)}")

    by_source = {}
    for token in analyzer_batch.tokens:
        src = token.source.value
        by_source[src] = by_source.get(src, 0) + 1
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    print("\n--- Caption Extraction ---")
    caption_batch = caption_extractor.extract_all_caption_tokens(metadata, split_into_clauses=False)
    print(f"Caption tokens: {len(caption_batch.tokens)}")

    by_source = {}
    for token in caption_batch.tokens:
        src = token.source.value
        by_source[src] = by_source.get(src, 0) + 1
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    for token in caption_batch.tokens[:3]:
        text = token.text[:60] + "..." if len(token.text) > 60 else token.text
        print(f"    {text}")

    # Combine all and normalize
    print("\n--- Full Pipeline ---")
    combined = TokenBatch()
    combined.add_all(tagger_batch.tokens)
    combined.add_all(analyzer_batch.tokens)
    combined.add_all(caption_batch.tokens)

    print(f"Before normalization: {len(combined.tokens)} tokens")
    normalized = normalizer.normalize_batch(combined, min_confidence=0.3, deduplicate=True)
    print(f"After normalization: {len(normalized.tokens)} tokens")

    # Show merged tokens
    merged = [t for t in normalized.tokens if t.metadata.get("merged_from", 0) > 1]
    if merged:
        print(f"\nMerged tokens ({len(merged)}):")
        for token in merged[:5]:
            sources = token.metadata.get("all_sources", [])
            print(f"  '{token.text}' from {sources}")

    # Show high confidence
    high_conf = sorted(
        [t for t in normalized.tokens if t.confidence > 0.8],
        key=lambda x: x.confidence,
        reverse=True
    )[:10]
    print(f"\nTop 10 high-confidence tokens:")
    for token in high_conf:
        print(f"  [{token.confidence:.2f}] {token.text[:50]}")

    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    modules = run_standalone_tests()
    run_integration_test(modules)
