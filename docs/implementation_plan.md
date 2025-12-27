# Image Tokenization & Composition - Implementation Plan

## Related Documents

- **Specification**: [image_tokenization_spec.md](./image_tokenization_spec.md) - Full technical specification
- **Project Root**: `E:\ComfyUI\custom_nodes\ComfyUI-AI-Photography-Toolkit`
- **Target Module**: `core/compose/` (replaces current StandardGenerator)

---

## Project Overview

**Goal**: Replace the current broken `StandardGenerator` with a robust tokenization → classification → assembly pipeline that produces high-quality natural language prompts from image analysis metadata.

**Two Modes**:
- `Standard` → Rule-based assembly (fast, free, deterministic)
- `Enhance with AI` → LLM-based assembly (higher quality, costs money)

---

## Implementation Phases

### Phase 1: Core Infrastructure [Priority: HIGH]

**Files to Create:**

```
core/compose/
├── tokenizer/
│   ├── __init__.py
│   ├── base.py              # ImageToken dataclass, TokenType enum
│   ├── tagger_extractor.py  # Extract from wd14, pixai, joytag, nudenet
│   ├── analyzer_extractor.py # Extract from photography, iqa, composition, saliency
│   ├── caption_extractor.py # Extract from florence captions (sentence/clause parsing)
│   └── normalizer.py        # Deduplication, confidence aggregation
├── classifier/
│   ├── __init__.py
│   ├── base.py              # TokenClassification dataclass
│   ├── deterministic.py     # Layer 1-2: structured keys, source routing
│   ├── dictionary.py        # Layer 3: keyword dictionaries
│   ├── embeddings.py        # Layer 4: semantic similarity (optional)
│   └── categories.py        # Category definitions and dictionaries
├── assembler/
│   ├── __init__.py
│   ├── base.py              # AssemblyResult dataclass
│   ├── rule_based.py        # Phase 5: Template-based assembly
│   ├── llm_based.py         # Phase 6: LLM-based assembly
│   ├── templates.py         # Template definitions
│   └── phrase_mappings.py   # Token → natural phrase dictionaries
├── validator/
│   ├── __init__.py
│   ├── content_preservation.py
│   ├── tag_alignment.py
│   ├── coherence.py
│   └── report.py
└── pipeline.py              # Main orchestrator (replaces current pipeline.py)
```

**Tasks:**

1. [x] Create `ImageToken` dataclass with fields: text, confidence, source, token_type, metadata
2. [x] Create `TokenType` enum: TAG, PHRASE, KEY_VALUE, SENTENCE
3. [x] Create `CanonicalCategory` enum with all 9 categories
4. [x] Create `TokenClassification` dataclass with primary/secondary categories

**Commit:** `c7859cf` - feat(compose): implement Phase 1 - Core Infrastructure

---

### Phase 2: Tokenization [Priority: HIGH]

**Spec Reference**: Phase 1 in image_tokenization_spec.md

**Tasks:**

1. [ ] **Tagger Extractor** (`tagger_extractor.py`)
   - Extract from wd14, pixai, joytag dictionaries
   - Handle both formats: `{tag: conf}` and `{tags: [...], attributes: {...}}`
   - Set token_type = TAG

2. [ ] **Analyzer Extractor** (`analyzer_extractor.py`)
   - Extract from photography, iqa, composition, saliency
   - Preserve source info for Layer 2 classification
   - Set token_type = TAG or KEY_VALUE

3. [ ] **Caption Extractor** (`caption_extractor.py`)
   - Sentence segmentation (split by `.!?`)
   - Clause segmentation (split by `,`, `and`, `with`, `while`)
   - Pattern matching for category-specific phrases
   - META pattern filtering (remove "the image shows", etc.)
   - Set token_type = PHRASE or SENTENCE

4. [ ] **Normalizer** (`normalizer.py`)
   - Lowercase normalization
   - Whitespace cleanup
   - Duplicate detection (exact + near-duplicate)
   - Confidence aggregation for duplicates
   - Source reliability weighting

**Test Cases:**
- [ ] Test tagger extraction with sample wd14 output
- [ ] Test caption extraction with sample florence_description
- [ ] Test META filtering removes philosophical content
- [ ] Test deduplication merges "brown hair" from multiple sources

---

### Phase 3: Classification [Priority: HIGH]

**Spec Reference**: Phase 3 in image_tokenization_spec.md

**Tasks:**

1. [ ] **Category Definitions** (`categories.py`)
   - Define all 9 canonical categories
   - Define subcategories for Subject Details (hair, eyes, face, body, clothing, skin)
   - Create keyword dictionaries with exact_matches, prefix_patterns, regex_patterns

2. [ ] **Deterministic Classifier** (`deterministic.py`)
   - Layer 1: florence_analyze key mapping (camera_angle → Composition, etc.)
   - Layer 2: Source-based routing (photography → Lighting, iqa → Quality)

3. [ ] **Dictionary Classifier** (`dictionary.py`)
   - Layer 3: Keyword lookup with exact, prefix, regex matching
   - Handle multi-category tokens (primary + secondary)

4. [ ] **Embedding Classifier** (`embeddings.py`) [OPTIONAL - defer if time-constrained]
   - Layer 4: Semantic similarity using sentence-transformers
   - Category centroid matching
   - Cache embeddings for performance

5. [ ] **Conflict Resolution**
   - Handle mutually exclusive tokens (standing vs sitting)
   - Keep highest confidence for conflicts

**Test Cases:**
- [ ] Test "brown hair" → [Subject Details/hair]
- [ ] Test "standing" → [Action/Pose]
- [ ] Test "soft light" → [Lighting]
- [ ] Test florence_analyze key "camera_angle: front" → [Composition]
- [ ] Test multi-category: "looking at viewer" → [Action/Pose] primary, [Subject Details/eyes] secondary

---

### Phase 4: Rule-Based Assembly [Priority: HIGH]

**Spec Reference**: Phase 5 in image_tokenization_spec.md

**Tasks:**

1. [ ] **Phrase Mappings** (`phrase_mappings.py`)
   - Token → natural phrase dictionaries
   - `"1girl"` → `"a woman"`
   - `"looking at viewer"` → `"looking directly at the camera"`
   - `"cowboy shot"` → `"framed from mid-thigh up"`

2. [ ] **Templates** (`templates.py`)
   - Subject templates with variations
   - Hair/eyes/clothing templates
   - Action/environment/lighting templates
   - Master template for full assembly

3. [ ] **Rule-Based Assembler** (`rule_based.py`)
   - Implement all category generators:
     - `generate_subject()`
     - `generate_subject_details()` (hair, eyes, face, body, clothing)
     - `generate_action()`
     - `generate_environment()`
     - `generate_lighting()`
     - `generate_style()`
     - `generate_composition()`
   - Adjective ordering rules (length, texture, color)
   - Pronoun resolution (She/Her based on gender detection)
   - Sentence fusion for same-subject attributes
   - Connector variation (and, with, featuring)

4. [ ] **Empty Category Handling**
   - Skip empty categories gracefully
   - Fallback phrases for critical categories

5. [ ] **Confidence-Based Filtering**
   - Only include tokens above threshold
   - Prioritize high-confidence tokens

**Test Cases:**
- [ ] Test hair tokens → "long, wavy brown hair" (correct adjective order)
- [ ] Test clothing grouping → "white Calvin Klein crop top with matching briefs"
- [ ] Test full assembly produces valid 3-4 paragraph output
- [ ] Test empty lighting category → graceful omission

---

### Phase 5: LLM-Based Assembly [Priority: MEDIUM]

**Spec Reference**: Phase 6 in image_tokenization_spec.md

**Tasks:**

1. [ ] **LLM Assembler** (`llm_based.py`)
   - Single comprehensive prompt approach (primary)
   - Section-by-section approach (optional)
   - Template + LLM hybrid approach (optional)

2. [ ] **Prompt Templates**
   - System prompt with rules and structure
   - User prompt with JSON token input
   - Negative instructions (don't invent, don't use raw tags)

3. [ ] **Provider Integration**
   - Reuse existing providers from `core/compose/llm.py`:
     - Local (Qwen)
     - Anthropic
     - OpenAI
     - Gemini

4. [ ] **Output Validation**
   - Schema validation (length, no raw tags, proper sentences)
   - Semantic validation (key elements present)
   - Retry on failure

5. [ ] **Caching**
   - Token hash → description cache
   - Section-level caching for hybrid approach

6. [ ] **Fallback Chain**
   - Retry with backoff
   - Fall back to rule-based on failure

**Test Cases:**
- [ ] Test LLM output doesn't contain raw tags like "1girl"
- [ ] Test output contains key elements from input tokens
- [ ] Test fallback to rule-based when LLM fails

---

### Phase 6: Validation Module [Priority: LOW]

**Spec Reference**: Validation section in image_tokenization_spec.md

**Tasks:**

1. [ ] **Content Preservation Checker**
   - Word-level accounting
   - Coverage percentage calculation
   - Unaccounted words report

2. [ ] **Tag Alignment Checker**
   - Compare extractions to original tags
   - Match type detection (exact, semantic, partial)
   - Category accuracy scoring

3. [ ] **Coherence Analyzer** [OPTIONAL - requires embeddings]
   - Intra-category similarity
   - Inter-category separation
   - Contamination detection

4. [ ] **Report Generator**
   - Aggregate all metrics
   - Generate actionable items
   - Overall quality score

---

### Phase 7: Integration [Priority: HIGH]

**Tasks:**

1. [ ] **Update Pipeline** (`pipeline.py`)
   - Replace current `StandardGenerator` with new tokenizer → classifier → assembler pipeline
   - Keep interface compatible with existing `ComposePipeline`

2. [ ] **Update Node** (`nodes/prompt_compose.py`)
   - No changes needed if pipeline interface unchanged
   - Verify mode switching works (Standard vs Enhance with AI)

3. [ ] **Configuration**
   - Move dictionaries to YAML config files if too large for code
   - Add verbosity control settings

4. [ ] **Cleanup**
   - Remove old `standard.py` once new pipeline verified
   - Update imports in `__init__.py`

---

### Phase 8: Token Grounding Visualization [Priority: LOW]

**Spec Reference**: Phase 7 in image_tokenization_spec.md

**Purpose**: Visual debugging and QA - show WHERE in the image each token spatially corresponds.

**Files to Create:**

```
core/compose/grounding/
├── __init__.py
├── router.py              # Select grounding method by category
├── human_parsing.py       # SCHP/Graphonomy integration
├── grounding_dino.py      # Grounding DINO + SAM
├── clip_attention.py      # CLIP-based heatmaps (fallback)
└── visualizer.py          # Overlay composition, export
```

**Tasks:**

1. [ ] **Category Router** (`router.py`)
   - Map token categories to appropriate grounding method
   - Subject Details → human_parsing
   - Environment → grounding_dino
   - Abstract (lighting, style) → clip_attention or skip

2. [ ] **Human Parsing Integration** (`human_parsing.py`)
   - SCHP or Graphonomy model loading
   - Token → segment class mapping dictionary
   - Mask extraction for hair, clothing, body parts

3. [ ] **Grounding DINO + SAM** (`grounding_dino.py`) [OPTIONAL]
   - Token → natural language prompt conversion
   - Bounding box detection
   - SAM mask refinement

4. [ ] **CLIP Attention Fallback** (`clip_attention.py`) [OPTIONAL]
   - GradCAM-based attention heatmaps
   - For abstract concepts that can't be localized

5. [ ] **Visualizer** (`visualizer.py`)
   - Single token heatmap overlay
   - Multi-token composite (color-coded by category)
   - Static image export (PNG)
   - JSON annotation export
   - Interactive HTML viewer [OPTIONAL]

6. [ ] **Integration with Validation**
   - Spatial coherence checks
   - Visual QA report generation

**Dependencies:**
- SCHP/Graphonomy (~150MB)
- face-parsing.PyTorch (~100MB) [optional]
- Grounding DINO (~700MB) [optional]
- SAM ViT-B (~400MB) [optional]

**Test Cases:**
- [ ] Human parsing correctly segments hair, clothing, body
- [ ] Token → segment mapping returns correct regions
- [ ] Visualization overlays are correctly positioned
- [ ] JSON export contains valid bounding boxes and masks

---

## File Dependency Order

Implement in this order to minimize blocking:

```
1. core/compose/tokenizer/base.py          # No dependencies
2. core/compose/classifier/categories.py   # No dependencies
3. core/compose/tokenizer/tagger_extractor.py
4. core/compose/tokenizer/analyzer_extractor.py
5. core/compose/tokenizer/caption_extractor.py
6. core/compose/tokenizer/normalizer.py
7. core/compose/classifier/deterministic.py
8. core/compose/classifier/dictionary.py
9. core/compose/assembler/phrase_mappings.py
10. core/compose/assembler/templates.py
11. core/compose/assembler/rule_based.py
12. core/compose/assembler/llm_based.py    # Can reuse existing llm.py logic
13. core/compose/pipeline.py               # Orchestrates everything
14. core/compose/validator/*               # Optional, implement last
15. core/compose/grounding/*               # Optional, Phase 8 - implement after validation
```

---

## Testing Strategy

### Unit Tests

Location: `tests/compose/`

```
tests/compose/
├── test_tokenizer.py
├── test_classifier.py
├── test_assembler.py
└── test_pipeline.py
```

### Integration Test

Create sample metadata file with real output from Analysis node, run through full pipeline, verify output quality.

### Manual Testing

1. Run in ComfyUI with real images
2. Compare output to current broken StandardGenerator
3. Verify "Enhance with AI" mode works with local Qwen model

---

## Dependencies

### Required (Already Installed)
- Python 3.10+
- PyYAML (for config files)
- re (regex, stdlib)

### Optional (For Embedding Classifier)
- sentence-transformers (`all-MiniLM-L6-v2`)

### Optional (For LLM Assembly)
- anthropic
- openai
- google-generativeai
- transformers (for local models)

---

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] Tokenization extracts from all metadata sources
- [ ] Classification assigns tokens to correct categories
- [ ] Rule-based assembly produces grammatically correct output
- [ ] No more "her eyes are brown eyes" errors
- [ ] No more "in the middle of the image" meta-commentary
- [ ] Output follows canonical schema order

### Full Implementation

- [ ] All MVP criteria
- [ ] LLM assembly works as fallback/enhancement
- [ ] Validation module can score output quality
- [ ] Caching reduces redundant computation
- [ ] Performance: <100ms for rule-based, <3s for LLM

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Caption parsing too fragile | Start with simple patterns, iterate based on failures |
| Classification accuracy low | Extensive keyword dictionaries, fall back to uncategorized |
| LLM output inconsistent | Strict validation, fallback to rule-based |
| Performance too slow | Cache aggressively, defer embedding classifier |
| Breaking existing workflows | Keep pipeline interface identical |

---

## Estimated Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Infrastructure | 2-3 hours | HIGH |
| Phase 2: Tokenization | 4-6 hours | HIGH |
| Phase 3: Classification | 4-6 hours | HIGH |
| Phase 4: Rule-Based Assembly | 6-8 hours | HIGH |
| Phase 5: LLM Assembly | 3-4 hours | MEDIUM |
| Phase 6: Validation | 4-6 hours | LOW |
| Phase 7: Integration | 2-3 hours | HIGH |
| Phase 8: Token Grounding Visualization | 6-10 hours | LOW |
| **Total (MVP: Phases 1-4,7)** | **18-26 hours** | |
| **Total (Full)** | **31-46 hours** | |

---

## Quick Start Commands

```bash
# Navigate to project
cd E:\ComfyUI\custom_nodes\ComfyUI-AI-Photography-Toolkit

# Create new module structure
mkdir -p core/compose/tokenizer
mkdir -p core/compose/classifier
mkdir -p core/compose/assembler
mkdir -p core/compose/validator

# Create __init__.py files
touch core/compose/tokenizer/__init__.py
touch core/compose/classifier/__init__.py
touch core/compose/assembler/__init__.py
touch core/compose/validator/__init__.py

# Run tests (once created)
python -m pytest tests/compose/ -v
```

---

## Context Recovery Notes

**If context is lost, read these files in order:**

1. `docs/image_tokenization_spec.md` - Full technical specification
2. `docs/implementation_plan.md` - This file, implementation roadmap
3. `CLAUDE.md` - Project overview and architecture
4. `core/compose/` - Current implementation (to be replaced)

**Current State Summary:**

The existing `StandardGenerator` in `core/compose/standard.py` produces broken output like:
- "her eyes are brown eyes" (grammar error)
- "standing in the middle of the image" (meta-commentary not filtered)
- Missing clothing details
- Redundant content

The new implementation will fix these issues with a proper tokenization → classification → assembly pipeline.

**Key Design Decisions:**

1. **Tokenization**: Extract ALL content from metadata into unified ImageToken format
2. **Classification**: 5-layer cascade (deterministic → dictionary → embeddings)
3. **Assembly (Rule-Based)**: Templates + phrase mappings + grammar rules
4. **Assembly (LLM)**: Fallback/enhancement using existing LLM providers
5. **Validation**: Optional quality scoring module
6. **Token Grounding**: Optional visual debugging showing spatial token locations

---

## Next Steps

When resuming work:

1. Read this plan and the spec document
2. Check which tasks are marked complete
3. Continue from the next uncompleted task
4. Update checkboxes as tasks complete
5. Test incrementally - don't wait until end

---

*Last Updated: December 2024*
*Version: 1.0*
