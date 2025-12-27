"""Test the full prompt composition pipeline (Phases 1-4).

Tests:
1. Tokenization (Phase 2)
2. Classification (Phase 3)
3. Assembly (Phase 4)
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
    if "core.compose.assembler" not in sys.modules:
        assembler_pkg = ModuleType("core.compose.assembler")
        assembler_pkg.__path__ = [str(compose_path / "assembler")]
        sys.modules["core.compose.assembler"] = assembler_pkg


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = ".".join(name.split(".")[:-1])
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 70)
    print("Phase 4 Test: Full Prompt Composition Pipeline")
    print("=" * 70)

    compose_path = Path(__file__).parent.parent.parent / "core" / "compose"
    setup_package_hierarchy(compose_path)

    # Load all required modules
    base = load_module("core.compose.tokenizer.base", compose_path / "tokenizer" / "base.py")
    normalizer = load_module("core.compose.tokenizer.normalizer", compose_path / "tokenizer" / "normalizer.py")
    tagger_extractor = load_module("core.compose.tokenizer.tagger_extractor", compose_path / "tokenizer" / "tagger_extractor.py")
    analyzer_extractor = load_module("core.compose.tokenizer.analyzer_extractor", compose_path / "tokenizer" / "analyzer_extractor.py")
    caption_extractor = load_module("core.compose.tokenizer.caption_extractor", compose_path / "tokenizer" / "caption_extractor.py")

    categories = load_module("core.compose.classifier.categories", compose_path / "classifier" / "categories.py")
    classifier_base = load_module("core.compose.classifier.base", compose_path / "classifier" / "base.py")
    deterministic = load_module("core.compose.classifier.deterministic", compose_path / "classifier" / "deterministic.py")
    dictionary = load_module("core.compose.classifier.dictionary", compose_path / "classifier" / "dictionary.py")
    classifier_init = load_module("core.compose.classifier.__init__", compose_path / "classifier" / "__init__.py")

    assembler_base = load_module("core.compose.assembler.base", compose_path / "assembler" / "base.py")
    assembler_rules = load_module("core.compose.assembler.rules", compose_path / "assembler" / "rules.py")
    assembler_standard = load_module("core.compose.assembler.standard", compose_path / "assembler" / "standard.py")

    TokenBatch = base.TokenBatch
    classify_batch = classifier_init.classify_batch
    resolve_conflicts = classifier_init.resolve_conflicts
    get_classification_stats = classifier_init.get_classification_stats
    assemble_prompt = assembler_standard.assemble_prompt
    AssemblerConfig = assembler_base.AssemblerConfig
    PromptStyle = assembler_base.PromptStyle

    # Test with each sample file
    test_files = [
        "sample_metadata.json",
        "sample_metadata_car.json",
    ]

    for test_file in test_files:
        sample_path = Path(__file__).parent / test_file
        if not sample_path.exists():
            print(f"\n[SKIP] {test_file} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"Testing: {test_file}")
        print("=" * 70)

        with open(sample_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Phase 2: Tokenization
        print("\n--- Phase 2: Tokenization ---")
        tagger_batch = tagger_extractor.extract_all_tagger_tokens(metadata, min_confidence=0.0)
        analyzer_batch = analyzer_extractor.extract_all_analyzer_tokens(metadata, min_confidence=0.0)
        caption_batch = caption_extractor.extract_all_caption_tokens(metadata, split_into_clauses=False)

        combined = TokenBatch()
        combined.image_info = metadata.get("image_info", {})
        combined.add_all(tagger_batch.tokens)
        combined.add_all(analyzer_batch.tokens)
        combined.add_all(caption_batch.tokens)

        normalized = normalizer.normalize_batch(combined, min_confidence=0.3, deduplicate=True)
        print(f"Tokens extracted: {len(normalized.tokens)}")

        # Phase 3: Classification
        print("\n--- Phase 3: Classification ---")
        classified = classify_batch(normalized)
        classified = resolve_conflicts(classified)
        stats = get_classification_stats(classified)

        print(f"Processed: {stats['processed_percent']}%")
        print(f"Categorized: {stats['categorized_percent']}%")
        print(f"By category:")
        for cat, count in sorted(stats['by_category'].items()):
            if count > 0 and cat not in ['uncategorized', 'filtered_meta']:
                print(f"  {cat}: {count}")

        # Phase 4: Assembly
        print("\n--- Phase 4: Assembly ---")

        # Test NATURAL style
        config_natural = AssemblerConfig(style=PromptStyle.NATURAL)
        result_natural = assemble_prompt(classified, config_natural)
        print(f"\nNATURAL style ({result_natural.word_count} words):")
        print(f"  {result_natural.prompt[:200]}..." if len(result_natural.prompt) > 200 else f"  {result_natural.prompt}")

        # Test TAGS style
        config_tags = AssemblerConfig(style=PromptStyle.TAGS)
        result_tags = assemble_prompt(classified, config_tags)
        print(f"\nTAGS style ({result_tags.token_count} tokens):")
        print(f"  {result_tags.prompt[:200]}..." if len(result_tags.prompt) > 200 else f"  {result_tags.prompt}")

        # Test HYBRID style
        config_hybrid = AssemblerConfig(style=PromptStyle.HYBRID)
        result_hybrid = assemble_prompt(classified, config_hybrid)
        print(f"\nHYBRID style:")
        print(f"  {result_hybrid.prompt[:200]}..." if len(result_hybrid.prompt) > 200 else f"  {result_hybrid.prompt}")

        # Show sections breakdown
        print(f"\n--- Sections ---")
        for section in result_natural.sections:
            if section.tokens:
                print(f"  {section.category.value}: {len(section.tokens)} tokens")

    print("\n" + "=" * 70)
    print("Phase 4 Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
