"""Quick test of Phase 2 with landscape image metadata (no humans)."""

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
    print("Phase 2 Test: Landscape Image (No Humans)")
    print("=" * 70)

    sample_path = Path(__file__).parent / "sample_metadata_landscape.json"
    with open(sample_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    compose_path = Path(__file__).parent.parent.parent / "core" / "compose"
    setup_package_hierarchy(compose_path)

    base = load_module("core.compose.tokenizer.base", compose_path / "tokenizer" / "base.py")
    normalizer = load_module("core.compose.tokenizer.normalizer", compose_path / "tokenizer" / "normalizer.py")
    tagger_extractor = load_module("core.compose.tokenizer.tagger_extractor", compose_path / "tokenizer" / "tagger_extractor.py")
    analyzer_extractor = load_module("core.compose.tokenizer.analyzer_extractor", compose_path / "tokenizer" / "analyzer_extractor.py")
    load_module("core.compose.classifier.categories", compose_path / "classifier" / "categories.py")
    caption_extractor = load_module("core.compose.tokenizer.caption_extractor", compose_path / "tokenizer" / "caption_extractor.py")

    TokenBatch = base.TokenBatch

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

    print("\n  Sample captions:")
    for token in caption_batch.tokens[:5]:
        text = token.text[:70] + "..." if len(token.text) > 70 else token.text
        filtered = " [FILTERED]" if token.metadata.get("filtered") else ""
        print(f"    [{token.source.value}]{filtered} {text}")

    print("\n--- Full Pipeline ---")
    combined = TokenBatch()
    combined.add_all(tagger_batch.tokens)
    combined.add_all(analyzer_batch.tokens)
    combined.add_all(caption_batch.tokens)

    print(f"Before normalization: {len(combined.tokens)} tokens")
    normalized = normalizer.normalize_batch(combined, min_confidence=0.3, deduplicate=True)
    print(f"After normalization: {len(normalized.tokens)} tokens")

    merged = [t for t in normalized.tokens if t.metadata.get("merged_from", 0) > 1]
    if merged:
        print(f"\nMerged tokens ({len(merged)}):")
        for token in sorted(merged, key=lambda x: x.metadata["merged_from"], reverse=True)[:10]:
            sources = token.metadata.get("all_sources", [])
            print(f"  '{token.text}' [{token.confidence:.2f}] from {sources}")

    high_conf = sorted(
        [t for t in normalized.tokens if t.confidence > 0.8],
        key=lambda x: x.confidence,
        reverse=True
    )
    print(f"\nHigh-confidence tokens ({len(high_conf)}):")
    for token in high_conf[:15]:
        print(f"  [{token.confidence:.2f}] {token.text[:50]}")

    print("\n--- Key Landscape Tokens ---")
    keywords = ["mountain", "lake", "water", "snow", "fog", "tree", "forest", "sky", "cloud", "reflection", "nature", "scenery", "landscape", "serene", "peaceful"]
    found = []
    for token in normalized.tokens:
        text_lower = token.text.lower()
        for kw in keywords:
            if kw in text_lower:
                found.append((kw, token.text, token.confidence, token.source.value))
                break

    for kw, text, conf, src in sorted(set(found), key=lambda x: x[2], reverse=True)[:20]:
        text_display = text[:50] + "..." if len(text) > 50 else text
        print(f"  [{conf:.2f}] '{text_display}' ({src})")

    # Check for "no humans" detection
    print("\n--- Human Detection ---")
    human_tokens = [t for t in normalized.tokens if "human" in t.text.lower() or "1girl" in t.text.lower() or "person" in t.text.lower()]
    if human_tokens:
        for token in human_tokens:
            print(f"  [{token.confidence:.2f}] '{token.text}' ({token.source.value})")
    else:
        print("  No human-related tokens (correct for landscape)")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
