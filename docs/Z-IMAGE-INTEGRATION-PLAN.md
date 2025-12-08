# Z-Image Integration Plan

> Implementation plan for adding Z-Image prompt compatibility to ComfyUI-AI-Photography-Toolkit

---

## Scope

**Provider**: Anthropic/Claude only (other providers deferred)

**Goal**: Enable the SID_AIPromptGenerator node to output Z-Image compatible prompts

---

## Current State

### Existing Node: `SID_AIPromptGenerator`

**Inputs:**
- image, api_key, model, user_prompt
- photography_style, target_model, color_style
- photographer_style, lighting_condition
- seed, temperature, max_tokens

**Outputs:**
- positive (STRING) - comma-separated keywords
- negative (STRING) - comma-separated keywords
- sampler_config (STRING) - markdown guide

**Prompt Format:**
```
POSITIVE: term1, term2, term3, ...
NEGATIVE: term1, term2, term3, ...
```

---

## Target State

### Enhanced Outputs

| Output | Type | Description |
|--------|------|-------------|
| positive | STRING | Keyword format (existing) |
| negative | STRING | Keyword format (existing) |
| **narrative_prompt** | STRING | **NEW** - Z-Image compatible narrative |
| sampler_config | STRING | Markdown guide (existing) |

### New Parameters

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| **output_format** | COMBO | "keyword" | "keyword", "narrative", "both" |
| **text_quote_style** | COMBO | "none" | "none", "z-image" |
| **max_narrative_length** | INT | 2000 | 0-10000 |

---

## Implementation Tasks

### Phase 1: Parameter Addition

**File:** `sid_ai_prompt_generator.py`

1. Add `output_format` input parameter (COMBO)
2. Add `text_quote_style` input parameter (COMBO)
3. Add `max_narrative_length` input parameter (INT)
4. Add `narrative_prompt` output

**Estimated changes:** ~30 lines in INPUT_TYPES, ~5 lines in RETURN_TYPES

---

### Phase 2: Z-Image System Prompt Template

**File:** `sid_ai_prompt_generator.py`

Create new constant `ZIMAGE_SYSTEM_PROMPT_TEMPLATE`:

```python
ZIMAGE_SYSTEM_PROMPT_TEMPLATE = """
You are an expert at creating prompts for Z-Image-Turbo, a 6B parameter
text-to-image model. Your task is to analyze the provided image and generate
a detailed, narrative-style prompt.

## OUTPUT REQUIREMENTS

Generate a SINGLE PARAGRAPH of flowing natural language (80-250 words) that
describes the visual scene. This is NOT a keyword list.

## STRUCTURE

Follow this order:
1. Shot type and composition (close-up, medium shot, full-body, wide)
2. Subject description (age, features, expression, pose)
3. Clothing and accessories (specific details, colors, materials)
4. Environment/background (keep simple, avoid clutter)
5. Lighting setup (Z-Image responds exceptionally well to lighting)
6. Style and technical notes (photography style, camera feel)
7. Safety constraints if needed (append at end)

## RULES

DO:
- Write complete sentences in natural language
- Describe only VISIBLE elements (colors, shapes, textures, materials)
- Include specific lighting direction and quality
- Specify camera angle and shot type
- Use concrete, objective descriptions
{text_quote_instruction}

DO NOT:
- Use comma-separated keyword lists
- Include meta-tags: "8K", "masterpiece", "best quality", "high quality"
- Use abstract adjectives: "beautiful", "mysterious", "interesting"
- Include negative prompts or exclusions (Z-Image doesn't use them)
- Add quality boosters or model-specific tags
- Write backstory or invisible context

## PHOTOGRAPHY STYLE GUIDANCE

{photography_style_guidance}

## COLOR TREATMENT

{color_guidance}

## LIGHTING

{lighting_guidance}

## PHOTOGRAPHER REFERENCE

{photographer_guidance}

## OUTPUT FORMAT

Return ONLY the prompt text. No labels, no "POSITIVE:", no explanations.
Single paragraph, 80-250 words, natural flowing language.
"""
```

---

### Phase 3: Template Building Logic

**File:** `sid_ai_prompt_generator.py`

Modify `_build_system_prompt()` method:

```python
def _build_system_prompt(self, model_info, photography_style, color_style,
                         photographer_style, lighting_condition,
                         output_format, text_quote_style):

    if output_format == "narrative" or output_format == "both":
        # Build Z-Image narrative template
        template = self._build_zimage_template(
            photography_style, color_style,
            photographer_style, lighting_condition,
            text_quote_style
        )
    else:
        # Use existing keyword template
        template = self._build_keyword_template(...)

    return template
```

Add new method `_build_zimage_template()`:

```python
def _build_zimage_template(self, photography_style, color_style,
                            photographer_style, lighting_condition,
                            text_quote_style):

    # Text quotation instruction
    text_quote_instruction = ""
    if text_quote_style == "z-image":
        text_quote_instruction = '- Wrap any text in the image with English double quotes: "text here"'

    # Photography style mapping (narrative versions)
    photography_guidance = self._get_zimage_photography_guidance(photography_style)

    # Color treatment (positive embedding)
    color_guidance = self._get_zimage_color_guidance(color_style)

    # Lighting (detailed descriptions)
    lighting_guidance = self._get_zimage_lighting_guidance(lighting_condition)

    # Photographer aesthetic
    photographer_guidance = self._get_zimage_photographer_guidance(photographer_style)

    return ZIMAGE_SYSTEM_PROMPT_TEMPLATE.format(
        text_quote_instruction=text_quote_instruction,
        photography_style_guidance=photography_guidance,
        color_guidance=color_guidance,
        lighting_guidance=lighting_guidance,
        photographer_guidance=photographer_guidance
    )
```

---

### Phase 4: Response Parsing

**File:** `sid_ai_prompt_generator.py`

Modify `_parse_response()` to handle narrative format:

```python
def _parse_response(self, response_text, output_format):
    if output_format == "keyword":
        return self._parse_keyword_response(response_text)
    elif output_format == "narrative":
        return self._parse_narrative_response(response_text)
    else:  # "both"
        keyword_result = self._parse_keyword_response(response_text)
        narrative_result = self._parse_narrative_response(response_text)
        return {**keyword_result, **narrative_result}
```

Add `_parse_narrative_response()`:

```python
def _parse_narrative_response(self, response_text):
    cleaned = self._clean_zimage_output(response_text)
    return {"narrative_prompt": cleaned}
```

---

### Phase 5: Output Cleaning

**File:** `sid_ai_prompt_generator.py`

Add Z-Image specific cleaning function:

```python
def _clean_zimage_output(self, text, max_length=2000):
    """Clean LLM output for Z-Image compatibility."""

    # 1. Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>', '', text)

    # 2. Remove markdown code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # 3. Remove common prefixes
    prefixes = [
        "Here is the enhanced prompt:",
        "Enhanced prompt:",
        "Final prompt:",
        "POSITIVE:",
        "Prompt:",
    ]
    for prefix in prefixes:
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):].strip()

    # 4. Remove surrounding quotes
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # 5. Remove meta-tags that may slip through
    meta_tags = [
        r'\b8K\b', r'\bmasterpiece\b', r'\bbest quality\b',
        r'\bhigh quality\b', r'\bultra detailed\b', r'\bdetailed\b,',
    ]
    for tag in meta_tags:
        text = re.sub(tag, '', text, flags=re.IGNORECASE)

    # 6. Detect and truncate repetitive patterns
    # (implement pattern detection from Z-Image-Utilities)

    # 7. Enforce max length with smart truncation
    if max_length > 0 and len(text) > max_length:
        # Try to break at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.7:
            text = truncated[:last_period + 1]
        else:
            last_comma = truncated.rfind(',')
            if last_comma > max_length * 0.8:
                text = truncated[:last_comma] + '.'
            else:
                text = truncated.rstrip() + '.'

    # 8. Clean up whitespace
    text = ' '.join(text.split())

    return text.strip()
```

---

### Phase 6: Main Execute Logic

**File:** `sid_ai_prompt_generator.py`

Update main `execute()` method:

```python
def execute(self, image, api_key, model, user_prompt, photography_style,
            target_model, color_style, photographer_style, lighting_condition,
            seed, temperature, max_tokens,
            output_format="keyword", text_quote_style="none",
            max_narrative_length=2000):

    # ... existing validation ...

    # Build appropriate system prompt
    if output_format == "both":
        # Make two API calls or request both formats
        system_prompt = self._build_dual_format_prompt(...)
    elif output_format == "narrative":
        system_prompt = self._build_zimage_template(...)
    else:
        system_prompt = self._build_system_prompt(...)  # existing

    # ... API call ...

    # Parse response based on format
    result = self._parse_response(response.content[0].text, output_format)

    # Return appropriate outputs
    if output_format == "keyword":
        return NodeOutput(
            result.get("positive", ""),
            result.get("negative", ""),
            "",  # narrative_prompt empty
            sampler_config
        )
    elif output_format == "narrative":
        return NodeOutput(
            "",  # positive empty
            "",  # negative empty
            self._clean_zimage_output(result.get("narrative_prompt", ""),
                                       max_narrative_length),
            sampler_config
        )
    else:  # both
        return NodeOutput(
            result.get("positive", ""),
            result.get("negative", ""),
            self._clean_zimage_output(result.get("narrative_prompt", ""),
                                       max_narrative_length),
            sampler_config
        )
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `sid_ai_prompt_generator.py` | Add 3 inputs, 1 output, ~200 lines new code |
| `README.md` | Document new parameters and Z-Image mode |

---

## Testing Plan

### Unit Tests

1. Test `_clean_zimage_output()` with various malformed inputs
2. Test template building with all parameter combinations
3. Test response parsing for both formats

### Integration Tests

1. Generate keyword prompt → verify format
2. Generate narrative prompt → verify no meta-tags
3. Generate both formats → verify both populated
4. Test text quotation with "z-image" style
5. Test max length truncation

### Manual Testing

1. Connect narrative output to Z-Image model in ComfyUI
2. Compare image quality: keyword vs narrative for same image
3. Test with various photography styles
4. Test text-in-image scenarios with quotation

---

## Migration Notes

- Existing workflows using keyword format continue to work unchanged
- New `output_format` defaults to "keyword" for backward compatibility
- `narrative_prompt` output is empty when format is "keyword"

---

## Future Enhancements (Deferred)

1. **Multi-Provider Support**: OpenRouter, Local LLM, Direct HuggingFace
2. **CLIP Conditioning Output**: Direct conditioning like Z-Image-Utilities
3. **Session Support**: Multi-turn prompt refinement
4. **Prompt Enhancer Integration**: Connect to Z-Image's own enhancer

---

*Created: December 2025*
*Branch: dev/z-image*
