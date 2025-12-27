"""Test the full prompt composition pipeline (Phases 1-5).

Tests:
1. Tokenization (Phase 2)
2. Classification (Phase 3)
3. Assembly (Phase 4)
4. Validation (Phase 5)
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
    if "core.compose.validator" not in sys.modules:
        validator_pkg = ModuleType("core.compose.validator")
        validator_pkg.__path__ = [str(compose_path / "validator")]
        sys.modules["core.compose.validator"] = validator_pkg


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = ".".join(name.split(".")[:-1])
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 70)
    print("Phase 5 Test: Full Pipeline with Validation")
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

    validator_base = load_module("core.compose.validator.base", compose_path / "validator" / "base.py")
    validator_rules = load_module("core.compose.validator.rules", compose_path / "validator" / "rules.py")
    validator_standard = load_module("core.compose.validator.standard", compose_path / "validator" / "standard.py")

    TokenBatch = base.TokenBatch
    classify_batch = classifier_init.classify_batch
    resolve_conflicts = classifier_init.resolve_conflicts
    assemble_prompt = assembler_standard.assemble_prompt
    AssemblerConfig = assembler_base.AssemblerConfig
    PromptStyle = assembler_base.PromptStyle
    validate_prompt = validator_standard.validate_prompt
    ValidatorConfig = validator_base.ValidatorConfig

    # Test with sample file
    sample_path = Path(__file__).parent / "sample_metadata.json"
    if not sample_path.exists():
        print(f"\n[SKIP] sample_metadata.json not found")
        return

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
    print(f"Classified into {len([s for s in classified.categories if classified.get_category(s)])} categories")

    # Phase 4: Assembly
    print("\n--- Phase 4: Assembly ---")
    config = AssemblerConfig(style=PromptStyle.NATURAL)
    assembled = assemble_prompt(classified, config)
    print(f"Assembled prompt: {assembled.word_count} words, {assembled.token_count} tokens")

    # Phase 5: Validation
    print("\n--- Phase 5: Validation ---")
    validator_config = ValidatorConfig(
        max_tokens=225,
        max_words=200,
        detect_duplicates=True,
        remove_duplicates=True,
        detect_conflicts=True,
        auto_correct=True,
    )
    validation = validate_prompt(assembled, validator_config)

    print(f"\nValidation Result: {validation.get_summary()}")
    print(f"Quality Score: {validation.quality_score:.2f}")

    if validation.issues:
        print(f"\nIssues Found ({len(validation.issues)}):")
        for issue in validation.issues[:10]:
            severity = issue.severity.value.upper()
            print(f"  [{severity}] {issue.message}")
            if issue.suggestion:
                print(f"         -> {issue.suggestion}")

    if validation.corrections_made > 0:
        print(f"\nAuto-corrections: {validation.corrections_made}")
        print(f"\nOriginal length: {len(validation.original_prompt.split())} words")
        print(f"Corrected length: {len(validation.prompt.split())} words")

    # Show final prompt
    print("\n--- Final Validated Prompt ---")
    final_prompt = validation.prompt
    if len(final_prompt) > 300:
        print(f"{final_prompt[:300]}...")
    else:
        print(final_prompt)

    print(f"\n--- Metadata ---")
    print(f"Estimated CLIP tokens: {validation.metadata.get('estimated_tokens', 'N/A')}")
    print(f"Word count: {validation.metadata.get('word_count', 'N/A')}")
    print(f"Categories: {', '.join(validation.metadata.get('categories_used', []))}")

    print("\n" + "=" * 70)
    print("Phase 5 Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
