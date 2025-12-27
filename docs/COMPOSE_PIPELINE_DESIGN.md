# Compose Pipeline Design Document

## Overview

The Compose pipeline transforms image analysis metadata into prompts optimized for Z-Image generation. The core principle is **preserve information, don't destroy it**.

## Design Goals

1. **Preserve Florence prose descriptions intact** - Never tokenize/split LLM-generated descriptions
2. **Organize tags into Z-Image categories** - Map all tags to the 6-part formula
3. **NLP Mode = Structured output** - Show user all available data organized by category
4. **LLM Mode = Natural prose** - Transform organized data into flowing prose
5. **Validate by measuring preservation** - Warn if information is lost, don't auto-remove

---

## Z-Image Prompting Requirements

Based on research from fal.ai and other sources:

### The 6-Part Formula
```
Subject + Scene + Composition + Lighting + Style + Constraints
```

### Key Rules
- **Natural prose, NOT comma-separated tags** - S3-DiT processes text/image in single stream
- **80-250 words optimal** - Over 300 words degrades coherence
- **Lighting is mandatory** - Always specify lighting
- **Prepositions matter** - "on", "under", "holding" carry significant weight
- **No negative prompts** - All guidance in positive prompt (guidance_scale=0.0)

### Example Output
> "A woman with long brown hair and blue eyes wearing an elegant red dress and gold necklace, standing confidently while looking directly at the camera. The scene is set in a professional studio with a neutral background. The lighting is soft with rim lighting creating depth. Shot as a professional portrait photograph with shallow depth of field in a medium shot."

---

## Pipeline Architecture

```
INPUT (metadata)
       │
       ▼
┌─────────────────┐
│  PHASE 1:       │
│  EXTRACT        │
│  - Prose intact │
│  - Tags listed  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 2:       │
│  CATEGORIZE     │
│  - Map to       │
│    Z-Image cats │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 3:       │
│  NORMALIZE      │
│  - Dedupe       │
│  - Resolve      │
│  - Clean        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ NLP   │ │ LLM   │
│ MODE  │ │ MODE  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│STRUCT │ │PROSE  │
│OUTPUT │ │OUTPUT │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│  PHASE 5:       │
│  VALIDATE       │
│  - Preservation │
│  - Quality      │
└────────┬────────┘
         │
         ▼
      OUTPUT
```

---

## Phase Details

### Phase 1: EXTRACT

**Purpose:** Separate prose descriptions from tags, preserve prose intact.

**Input Sources:**
| Source | Type | Action |
|--------|------|--------|
| florence_description | prose | Keep intact |
| florence_caption | prose | Keep intact |
| florence_mixed_caption | prose + tags | Split, keep prose intact |
| florence_mixed_caption_plus | prose + tags | Split, keep prose intact |
| vlm_description | prose | Keep intact |
| florence_analyze | key-value | Extract as tags |
| florence_objects | detection | Extract labels as tags (with counts) |
| florence_region_captions | detection | Extract region descriptions |
| wd14_tags | tags | Collect all |
| joytag_tags | tags | Collect all |

**Output:**
```python
{
    "best_prose": "The image shows a professional portrait...",  # Single best description
    "all_tags": [
        {"text": "woman", "confidence": 0.95, "source": "wd14"},
        {"text": "brown hair", "confidence": 0.88, "source": "wd14"},
        ...
    ]
}
```

**Prose Priority Order:**
1. florence_mixed_caption_plus (if prose, not just tags)
2. florence_mixed_caption (if prose, not just tags)
3. florence_description
4. florence_caption
5. vlm_description

---

### Phase 2: CATEGORIZE

**Purpose:** Map all tags into Z-Image categories.

**Categories:**
| Category | Examples | Z-Image Role |
|----------|----------|--------------|
| SUBJECT | woman, man, person, 1girl | Subject |
| APPEARANCE | brown hair, blue eyes, tall, slim | Subject details |
| CLOTHING | red dress, shirt, jeans, jewelry | Subject details |
| POSE | standing, sitting, looking at viewer | Subject action |
| ENVIRONMENT | studio, garden, street, indoor | Scene |
| LIGHTING | soft light, golden hour, rim light | Lighting (critical) |
| STYLE | photograph, realistic, cinematic | Style |
| QUALITY | 8k, detailed, sharp, masterpiece | Style boosters |
| COMPOSITION | close-up, portrait, bokeh, 85mm | Composition/Technical |

**Classification Method:**
1. Dictionary lookup (fast, covers 90%+ of tags)
2. Pattern matching for edge cases
3. Default to APPEARANCE if uncertain

**Output:**
```python
{
    "best_prose": "...",
    "categories": {
        "SUBJECT": ["woman"],
        "APPEARANCE": ["long brown hair", "blue eyes", "fair skin"],
        "CLOTHING": ["red dress", "gold necklace"],
        "POSE": ["standing", "looking at viewer"],
        "ENVIRONMENT": ["studio", "neutral background"],
        "LIGHTING": ["soft light", "rim lighting"],
        "STYLE": ["photograph", "professional"],
        "QUALITY": ["realistic", "detailed"],
        "COMPOSITION": ["portrait", "medium shot", "shallow dof"]
    }
}
```

---

### Phase 3: NORMALIZE

**Purpose:** Clean up categorized data without losing information.

**Operations:**
1. **Deduplicate** - Remove exact duplicates across categories
2. **Resolve conflicts** - Use VLM description to resolve contradictions
3. **Remove meta tags** - Filter only obvious garbage (watermark, username, etc.)
4. **Sort by confidence** - Higher confidence tags first within each category

**What NOT to do:**
- Don't aggressively filter tags
- Don't remove tags just because they seem redundant
- Don't limit tokens per category (user controls this)

---

### Phase 4A: ASSEMBLE (NLP Mode)

**Purpose:** Output structured data showing all extracted information.

**Output Format - STRUCTURED:**
```
# DESCRIPTION
The image shows a professional portrait of a woman with long brown hair...

# SUBJECT
woman

# APPEARANCE
long brown hair, blue eyes, fair skin

# CLOTHING
red dress, gold necklace

# POSE
standing, looking at viewer

# ENVIRONMENT
studio, neutral background

# LIGHTING
soft light, rim lighting

# STYLE
photograph, professional, realistic, detailed

# COMPOSITION
portrait, medium shot, shallow dof
```

**Alternative Formats:**
- **FULL**: Same as STRUCTURED but includes all Florence outputs
- **TAGS**: Just comma-separated tags (for SDXL compatibility)

---

### Phase 4B: ENHANCE (LLM Mode)

**Purpose:** Transform organized data into natural prose using LLM.

**System Prompt:**
```
You are a Z-Image prompt engineer. Transform the provided image data into
natural prose following the 6-part formula: Subject + Scene + Composition +
Lighting + Style + Technical.

Rules:
- Output 80-250 words of flowing prose
- Use natural sentences, NOT comma-separated tags
- ALWAYS include lighting description
- Preserve ALL provided information - do not omit details
- Use prepositions naturally: "with", "wearing", "standing in", "under"
```

**User Prompt Template:**
```
Convert this image data into a Z-Image prompt:

Description: {best_prose}

Subject: {subject_tags}
Appearance: {appearance_tags}
Clothing: {clothing_tags}
Pose: {pose_tags}
Environment: {environment_tags}
Lighting: {lighting_tags}
Style: {style_tags}
Composition: {composition_tags}

Generate natural prose prompt:
```

**Expected Output:**
Natural flowing prose that incorporates all the provided information.

---

### Phase 5: VALIDATE

**Purpose:** Measure information preservation and quality.

**Preservation Check:**
```python
input_concepts = count_unique_concepts(all_tags)
output_concepts = count_concepts_in_prompt(final_prompt)
preservation_ratio = output_concepts / input_concepts

if preservation_ratio < 0.7:
    warn("Significant information loss detected")
```

**Quality Checks:**
- Word count in range (80-250 for Z-Image mode)
- Lighting mentioned (required)
- No excessive repetition (same phrase 3+ times)
- No garbage patterns (lorem ipsum, etc.)

**What NOT to do:**
- Don't auto-truncate to fit arbitrary limits
- Don't remove "redundant" information
- Don't "fix" the prompt by removing content

---

## Implementation Files

| File | Purpose |
|------|---------|
| `core/compose/extractor.py` | Phase 1: Extract prose and tags |
| `core/compose/categorizer.py` | Phase 2: Map tags to categories |
| `core/compose/normalizer.py` | Phase 3: Dedupe and clean |
| `core/compose/assembler/` | Phase 4: Build output |
| `core/compose/validator.py` | Phase 5: Check preservation |
| `core/compose/compose_pipeline.py` | Orchestrates all phases |
| `nodes/prompt_compose.py` | ComfyUI node interface |

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| mode | NLP | NLP (fast) or LLM (enhanced) |
| output_style | STRUCTURED | STRUCTURED, FULL, TAGS |
| min_confidence | 0.3 | Minimum tag confidence to include |
| llm_provider | local | local, anthropic, openai, gemini |
| llm_model | qwen25_text_1.5b | Model for LLM mode |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-27 | Initial design document |
