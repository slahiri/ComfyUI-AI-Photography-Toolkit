# Z-Image Integration Plan

> Implementation plan for adding Z-Image prompt compatibility to ComfyUI-AI-Photography-Toolkit

---

## Scope

**Provider**: Anthropic/Claude only (other providers deferred)

**Goal**: Create two new nodes for Z-Image prompt generation and enhancement

---

## Architecture: Two-Node Design

### Node 1: `SID_Z_Image_Prompt_Generator`

**Purpose**: Analyze an image and generate a Z-Image compatible narrative prompt

**Use Case**: Image → Prompt (vision-based analysis)

```
[Load Image] → [SID_Z_Image_Prompt_Generator] → [Z-Image Model]
```

### Node 2: `SID_Z_Image_Prompt_Enhancer`

**Purpose**: Transform a simple text prompt into a detailed Z-Image narrative

**Use Case**: Simple Prompt → Enhanced Prompt (text-only, no image required)

```
[Text Input] → [SID_Z_Image_Prompt_Enhancer] → [Z-Image Model]
```

---

## Node 1: SID_Z_Image_Prompt_Generator

### Description

Analyzes an input image using Claude's vision capabilities and generates a Z-Image compatible narrative prompt. Includes photography style presets for artistic control.

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Input image to analyze |
| `api_key` | STRING | "" | Anthropic API key |
| `model` | COMBO | claude-sonnet-4-5-20250929 | Claude model selection |
| `user_prompt` | STRING | "" | Additional context/instructions |
| `photography_style` | COMBO | "Detailed" | Minimal, Detailed, Technical, Ultra-Detailed |
| `color_style` | COMBO | "None" | None, B&W, Color, Sepia, etc. |
| `photographer_style` | COMBO | "None" | None + 20 famous photographers |
| `lighting_condition` | COMBO | "None" | None + 31 lighting setups |
| `include_text_quotes` | BOOLEAN | True | Wrap text elements in quotes |
| `max_length` | INT | 2000 | Max output characters (0=unlimited) |
| `temperature` | FLOAT | 0.7 | Creativity level (0.0-1.0) |
| `seed` | INT | 0 | Random seed |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `prompt` | STRING | Z-Image compatible narrative prompt |
| `debug_log` | STRING | Debug information |

### System Prompt Template

```
You are an expert visual analyst creating prompts for Z-Image-Turbo.
Analyze the provided image and generate a detailed narrative description.

## OUTPUT FORMAT
Single flowing paragraph, {min_words}-{max_words} words, natural language.
NO keyword lists. NO meta-tags. NO negative prompts.

## STRUCTURE
1. Shot type and composition
2. Subject description (visible features only)
3. Clothing/objects (colors, materials, textures)
4. Environment/background
5. Lighting (direction, quality, color temperature)
6. Style and mood

## RULES
- Describe ONLY visible elements
- Use concrete, objective language
- NO abstract adjectives (beautiful, mysterious)
- NO quality tags (8K, masterpiece, best quality)
{text_quote_rule}

{style_guidance}
```

---

## Node 2: SID_Z_Image_Prompt_Enhancer

### Description

Takes a simple text prompt and enhances it into a detailed Z-Image compatible narrative. Does NOT require an image input - uses Claude's knowledge to expand the description.

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | STRING | required | Simple prompt to enhance |
| `api_key` | STRING | "" | Anthropic API key |
| `model` | COMBO | claude-sonnet-4-5-20250929 | Claude model selection |
| `enhancement_level` | COMBO | "Detailed" | Minimal, Detailed, Comprehensive |
| `subject_type` | COMBO | "Auto" | Auto, Portrait, Landscape, Product, Action, Architecture |
| `style_direction` | STRING | "" | Optional style guidance |
| `include_lighting` | BOOLEAN | True | Add lighting descriptions |
| `include_composition` | BOOLEAN | True | Add composition details |
| `include_text_quotes` | BOOLEAN | True | Wrap text elements in quotes |
| `max_length` | INT | 2000 | Max output characters |
| `temperature` | FLOAT | 0.7 | Creativity level |
| `seed` | INT | 0 | Random seed |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `enhanced_prompt` | STRING | Enhanced Z-Image narrative |
| `debug_log` | STRING | Debug information |

### System Prompt Template

```
You are an expert prompt engineer for Z-Image-Turbo.
Transform the user's simple prompt into a detailed visual description.

## INPUT PROMPT
{user_prompt}

## YOUR TASK
Expand this into a rich, detailed narrative description suitable for
Z-Image-Turbo image generation.

## OUTPUT FORMAT
Single flowing paragraph, {min_words}-{max_words} words.
Natural language description, NOT a keyword list.

## ENHANCEMENT GUIDELINES
- Preserve the core intent of the original prompt
- Add specific visual details (colors, textures, materials)
- Include spatial relationships and composition
{lighting_instruction}
{composition_instruction}
- Keep backgrounds simple unless specified
{text_quote_rule}

## RULES
- NO meta-tags (8K, masterpiece, best quality, high quality)
- NO negative prompts or exclusions
- NO abstract adjectives without visual backing
- Every word should describe something VISIBLE

## SUBJECT-SPECIFIC GUIDANCE ({subject_type})
{subject_guidance}

{style_direction_block}
```

### Enhancement Levels

| Level | Word Count | Detail |
|-------|------------|--------|
| Minimal | 50-80 | Core elements only |
| Detailed | 100-180 | Standard enhancement |
| Comprehensive | 200-300 | Maximum detail |

### Subject Type Guidance

**Portrait:**
```
- Describe age, visible features, expression
- Include clothing details (color, material, fit)
- Specify pose and body language
- Add lighting on face/skin
```

**Landscape:**
```
- Layer foreground, middle ground, background
- Include sky and atmospheric conditions
- Describe natural lighting and time of day
- Add environmental details (weather, season)
```

**Product:**
```
- Focus on materials and surface textures
- Describe colors accurately
- Include arrangement and props
- Specify lighting for product photography
```

**Action:**
```
- Capture motion and dynamic poses
- Describe body position and movement direction
- Include motion blur if appropriate
- Add dramatic lighting
```

**Architecture:**
```
- Describe structural elements and materials
- Include perspective and viewpoint
- Add environmental context
- Specify lighting conditions
```

---

## File Structure

```
ComfyUI-AI-Photography-Toolkit/
├── __init__.py                              (updated)
├── sid_ai_prompt_generator.py               (existing - unchanged)
├── sid_zimage_prompt_generator.py           (NEW)
├── sid_zimage_prompt_enhancer.py            (NEW)
├── utils/
│   ├── __init__.py                          (NEW)
│   └── zimage_utils.py                      (NEW - shared utilities)
├── docs/
│   ├── Z-IMAGE-PROMPTING-GUIDE.md           (existing)
│   └── Z-IMAGE-INTEGRATION-PLAN.md          (this file)
└── README.md                                (updated)
```

---

## Implementation Tasks

### Phase 1: Shared Utilities

**File:** `utils/zimage_utils.py`

```python
# Output cleaning
def clean_zimage_output(text, max_length=2000):
    """Clean LLM output for Z-Image compatibility."""
    ...

# Photography style mappings
PHOTOGRAPHY_STYLES = {...}
LIGHTING_CONDITIONS = {...}
PHOTOGRAPHER_STYLES = {...}
COLOR_STYLES = {...}

# Subject type guidance
SUBJECT_GUIDANCE = {...}

# Enhancement level settings
ENHANCEMENT_LEVELS = {...}
```

**Tasks:**
1. Create `utils/__init__.py`
2. Create `utils/zimage_utils.py` with shared constants and functions
3. Implement `clean_zimage_output()` function
4. Define all style/lighting/photographer mappings

---

### Phase 2: SID_Z_Image_Prompt_Generator

**File:** `sid_zimage_prompt_generator.py`

**Tasks:**
1. Create node class `SID_ZImagePromptGenerator`
2. Define INPUT_TYPES with all parameters
3. Define RETURN_TYPES (prompt, debug_log)
4. Implement system prompt builder
5. Implement image-to-base64 conversion (reuse from existing)
6. Implement execute() method with Claude vision API call
7. Apply output cleaning

**Estimated Lines:** ~400

---

### Phase 3: SID_Z_Image_Prompt_Enhancer

**File:** `sid_zimage_prompt_enhancer.py`

**Tasks:**
1. Create node class `SID_ZImagePromptEnhancer`
2. Define INPUT_TYPES with all parameters
3. Define RETURN_TYPES (enhanced_prompt, debug_log)
4. Implement system prompt builder with subject-specific guidance
5. Implement execute() method with Claude text API call
6. Apply output cleaning

**Estimated Lines:** ~350

---

### Phase 4: Registration & Documentation

**File:** `__init__.py`

**Tasks:**
1. Import new node classes
2. Add to NODE_CLASS_MAPPINGS
3. Add to NODE_DISPLAY_NAME_MAPPINGS

**File:** `README.md`

**Tasks:**
1. Document new Z-Image nodes
2. Add usage examples
3. Add workflow diagrams

---

## Node Comparison

| Feature | Generator | Enhancer |
|---------|-----------|----------|
| Input | Image (required) | Text prompt |
| Vision API | Yes | No |
| Photography presets | Yes | No |
| Subject type | Auto-detected | User-selected |
| Style direction | Via presets | Free-form text |
| Use case | Recreate/describe image | Expand simple idea |

---

## Workflow Examples

### Workflow 1: Image Recreation

```
[Load Image]
     ↓
[SID_Z_Image_Prompt_Generator]
     ↓
prompt → [CLIP Text Encode] → [KSampler] → [Z-Image Output]
```

### Workflow 2: Prompt Enhancement

```
[Primitive String: "a cat sitting on a chair"]
     ↓
[SID_Z_Image_Prompt_Enhancer]
     ↓
enhanced_prompt → [CLIP Text Encode] → [KSampler] → [Z-Image Output]
```

### Workflow 3: Chained Enhancement

```
[Load Image]
     ↓
[SID_Z_Image_Prompt_Generator]
     ↓
prompt → [SID_Z_Image_Prompt_Enhancer] (with style_direction)
     ↓
enhanced_prompt → [Z-Image Model]
```

---

## Example Outputs

### Generator Example

**Input:** Photo of woman in coffee shop

**Output:**
```
A medium shot of an adult woman in her early thirties seated at a
weathered wooden table inside a warmly lit coffee shop. She has
shoulder-length wavy brown hair and olive skin, wearing a cream
cable-knit sweater. Her expression is relaxed with a gentle smile
as she holds a white ceramic mug with both hands. Soft natural
daylight streams through large windows to her left, creating subtle
highlights on her hair and a warm glow on her face. The background
shows blurred shapes of other patrons and exposed brick walls,
with warm Edison bulb lighting adding amber tones to the scene.
Shallow depth of field, candid portrait style.
```

### Enhancer Example

**Input:** `a cat sitting on a chair`

**Output (Detailed level):**
```
A domestic shorthair cat with orange and white tabby markings sits
upright on a vintage wooden dining chair. The cat has bright amber
eyes and alert, forward-facing ears, its gaze directed slightly to
the left of frame. Its fluffy tail curls around the edge of the
chair seat. Soft diffused window light from the right illuminates
the scene, creating gentle shadows beneath the chair. The background
shows a simple cream-colored wall with subtle texture. The
composition is centered with the cat at eye level, shallow depth
of field blurring the wooden floor planks visible at the bottom
of frame.
```

---

## Testing Plan

### Unit Tests

1. `clean_zimage_output()` - various malformed inputs
2. Template building - all parameter combinations
3. Subject guidance selection
4. Enhancement level word counts

### Integration Tests

1. Generator: Image → prompt contains no meta-tags
2. Enhancer: Simple prompt → enhanced with correct word count
3. Text quotation: Text elements properly quoted
4. Max length: Proper truncation at sentence boundaries

### Manual Testing

1. Generate prompts → feed to Z-Image model
2. Compare: Generator output vs Enhancer output for same concept
3. Test all photography styles
4. Test all subject types
5. Test enhancement levels

---

## Migration Notes

- `SID_AIPromptGenerator` remains unchanged (backward compatible)
- New nodes are additive, not replacements
- Users can choose which node fits their workflow

---

## Future Enhancements (Deferred)

1. **Multi-Provider Support**: OpenRouter, Local LLM, Direct HuggingFace
2. **CLIP Conditioning Output**: Direct conditioning output variant
3. **Session Support**: Multi-turn refinement
4. **Batch Processing**: Multiple images/prompts
5. **Style Transfer**: Apply photographer style from one image to another

---

*Updated: December 2025*
*Branch: dev/z-image*
