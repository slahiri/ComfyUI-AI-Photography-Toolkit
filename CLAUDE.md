# ComfyUI-AI-Photography-Toolkit

## Current State (Dec 2024)

**Branch**: `dev/4.3.0`
**Commit**: `7f1186d` - feat: add prompt composition pipeline and tokenization spec

### Active Task: Prompt Composition Rewrite

Rewriting the `SID_PromptCompose` node with a new tokenization → classification → assembly pipeline.

**Problem**: Current `StandardGenerator` produces broken output:
- Grammar errors ("her eyes are brown eyes")
- Meta-commentary not filtered ("in the middle of the image")
- Missing content, redundant phrases

**Solution**: 8-phase implementation plan

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/image_tokenization_spec.md](docs/image_tokenization_spec.md) | Full technical specification (Phases 1-7) |
| [docs/implementation_plan.md](docs/implementation_plan.md) | Implementation roadmap with task checkboxes |

## Implementation Phases

| Phase | Description | Priority | Status |
|-------|-------------|----------|--------|
| 1 | Core Infrastructure (ImageToken, TokenType, categories) | HIGH | Not Started |
| 2 | Tokenization (extractors for taggers, analyzers, captions) | HIGH | Not Started |
| 3 | Classification (5-layer cascade) | HIGH | Not Started |
| 4 | Rule-Based Assembly (templates, phrase mappings) | HIGH | Not Started |
| 5 | LLM Assembly (enhancement mode) | MEDIUM | Not Started |
| 6 | Validation (quality scoring) | LOW | Not Started |
| 7 | Integration (replace StandardGenerator) | HIGH | Not Started |
| 8 | Token Grounding Visualization | LOW | Not Started |

**MVP**: Phases 1-4, 7 (18-26 hours)
**Full**: All phases (31-46 hours)

## Target File Structure

```
core/compose/
├── tokenizer/
│   ├── __init__.py
│   ├── base.py              # ImageToken, TokenType enum
│   ├── tagger_extractor.py  # wd14, pixai, joytag, nudenet
│   ├── analyzer_extractor.py # photography, iqa, composition
│   ├── caption_extractor.py # florence captions
│   └── normalizer.py        # deduplication, confidence
├── classifier/
│   ├── __init__.py
│   ├── base.py              # TokenClassification
│   ├── categories.py        # 9 canonical categories
│   ├── deterministic.py     # Layer 1-2
│   └── dictionary.py        # Layer 3
├── assembler/
│   ├── __init__.py
│   ├── base.py
│   ├── rule_based.py        # Standard mode
│   ├── llm_based.py         # Enhance with AI mode
│   ├── templates.py
│   └── phrase_mappings.py
├── validator/               # Optional
├── grounding/               # Phase 8, optional
└── pipeline.py              # Main orchestrator
```

## 9 Canonical Categories

1. Quality Boosters
2. Subject
3. Subject Details (hair, eyes, face, body, clothing, skin)
4. Action/Pose
5. Environment/Scene
6. Lighting
7. Style/Medium
8. Composition
9. Technical Parameters

## Modes

- `Standard` → Rule-based assembly (fast, free, deterministic)
- `Enhance with AI` → LLM-based assembly (higher quality, API costs)

## Key Design Decisions

1. **ImageToken** - Unified format: text, confidence, source, token_type, metadata
2. **5-Layer Classification** - Deterministic → Source routing → Dictionary → Embeddings → Uncategorized
3. **META Filtering** - Remove "the image shows", "in the middle of" patterns
4. **Phrase Mappings** - "1girl" → "a woman", "cowboy shot" → "framed from mid-thigh up"

## Resume Work

1. Read [docs/implementation_plan.md](docs/implementation_plan.md)
2. Check task checkboxes for current progress
3. Continue from next uncompleted phase
4. Test incrementally after each phase

## Quick Reference

### Current Compose Location
- `core/compose/` - Existing implementation (to be replaced)
- `core/compose/standard.py` - Broken StandardGenerator
- `core/compose/llm.py` - LLM generator (reusable)

### Metadata Sources
- **Taggers**: wd14, pixai, joytag, nudenet
- **Analyzers**: photography, iqa, composition, saliency
- **Captions**: florence_caption, florence_description, florence_analyze, vlm_description

### Test Command
```bash
python -m pytest tests/compose/ -v
```
