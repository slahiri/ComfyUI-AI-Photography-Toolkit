# LLM Output Cleaning Layer

A multi-stage pipeline for cleaning raw LLM responses into usable visual prompts.

## Overview

LLMs often output reasoning, artifacts, and formatting that pollutes the final prompt. The cleaning layer removes these while preserving the core visual description.

```
Raw LLM Response → Cleaning Pipeline → Clean Visual Prompt
```

---

## The Problem

Raw LLM output often contains:

```
<think>
Let me analyze the core elements: subject is a girl, wearing a red dress...
I need to add lighting, composition, textures...
</think>

Here is the enhanced prompt:

"A young woman in an elegant crimson floor-length dress stands in soft golden
afternoon light..."

Note: no "watermark" or "8K" tags should be used.
```

**Issues:**
- `<think>` reasoning blocks (Qwen3 thinking mode)
- Untagged reasoning in Chinese/English
- Prefix phrases ("Here is the enhanced prompt:")
- Surrounding quotes
- Negative instructions ("no watermark tags")
- Repetition loops from generation errors
- Markdown code blocks
- Trailing keyword lists

---

## Cleaning Pipeline Stages

### Stage 1: Remove Thinking Tags

Qwen3 and similar models output reasoning in `<think>` blocks:

```python
# Complete blocks
text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# Orphaned closing tags (thinking before </think>)
if "</think>" in text:
    parts = text.split("</think>", 1)
    text = parts[1].strip()
```

**Before:**
```
<think>Let me analyze this image request...</think>A woman in red dress.
```

**After:**
```
A woman in red dress.
```

---

### Stage 2: Detect Untagged Thinking

Some models output reasoning without tags. Detect via indicator phrases:

**Chinese Indicators:**
```python
thinking_indicators_zh = [
    "我是一个被关在逻辑牢笼里的幻视艺术家",
    "我的工作流程是",
    "首先，分析",
    "让我分析",
    "根据工作流程",
    "现在，构建描述",
]
```

**English Indicators:**
```python
thinking_indicators_en = [
    "I am a visionary artist",
    "My workflow is",
    "First, I need to analyze",
    "Let me analyze",
    "Following the workflow",
    "Now, constructing",
]
```

---

### Stage 3: Extract Final Prompt

When thinking is detected, extract the actual prompt using markers:

**Strategy 1 - Final Prompt Markers:**
```python
final_prompt_markers = [
    "最终prompt：", "修改后的prompt：", "增强后的prompt：",
    "Final prompt:", "Enhanced prompt:", "Modified prompt:",
]

for marker in final_prompt_markers:
    idx = text.rfind(marker)
    if idx != -1:
        text = text[idx + len(marker):].strip()
```

**Strategy 2 - Example Extraction:**
```python
# Look for "例如：" or "Example:" followed by English text
example_match = re.search(r'(?:例如：|Example:\s*)([A-Z][a-zA-Z].*?)$', text)
```

**Strategy 3 - Trailing English Pattern:**
```python
# Find standalone English prompt at end (common pattern)
english_prompt_pattern = r'\n([A-Z][a-z][^。\n]*(?:photo|image|portrait|scene).*\.?)$'
```

**Strategy 4 - Fallback Markers:**
```python
fallback_markers = ["核心画面：", "首先，核心画面：", "添加细节："]
```

---

### Stage 4: Remove Markdown

Strip code block formatting:

```python
if text.startswith('```'):
    text = re.sub(r'^```\w*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
```

**Before:**
```
```prompt
A woman in red dress standing in garden.
```
```

**After:**
```
A woman in red dress standing in garden.
```

---

### Stage 5: Remove Common Prefixes

Strip explanatory prefixes:

```python
prefixes = [
    "Here is the enhanced prompt:",
    "Here's the enhanced prompt:",
    "Enhanced prompt:",
    "Final prompt:",
    "Output:",
    "修改后的prompt：",
    "最终prompt：",
]

for prefix in prefixes:
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
```

---

### Stage 6: Remove Surrounding Quotes

Strip wrapping quotes:

```python
if (text.startswith('"') and text.endswith('"')) or \
   (text.startswith("'") and text.endswith("'")):
    text = text[1:-1].strip()
```

**Before:**
```
"A woman in red dress standing in garden."
```

**After:**
```
A woman in red dress standing in garden.
```

---

### Stage 7: Remove Negative Instructions

LLMs sometimes add unwanted negative guidance:

```python
negative_pattern = r',?\s*no\s+"[^"]{1,50}"(?:\s+or\s+"[^"]{1,50}")?\s+(?:tags?|descriptors?|effects?)'

text = re.sub(negative_pattern, '', text, flags=re.IGNORECASE)
```

**Before:**
```
A woman in red dress, no "watermark" or "8K" tags, no "blur" effects.
```

**After:**
```
A woman in red dress.
```

---

### Stage 8: Detect Phrase Repetition Loops

Generation errors can cause repeating patterns:

```python
segments = [s.strip() for s in text.split(',') if s.strip()]

# Check for repeating phrase patterns
for pattern_len in range(2, 7):
    pattern = segments[start:start + pattern_len]
    # Count how many times pattern repeats consecutively
    if repeat_count >= 3:
        segments = segments[:start + pattern_len]  # Keep only first occurrence
```

**Before:**
```
red dress, golden light, soft shadows, red dress, golden light, soft shadows, red dress, golden light, soft shadows
```

**After:**
```
red dress, golden light, soft shadows.
```

---

### Stage 9: Remove Trailing Keyword Lists

Some models append quoted keyword lists:

```python
keyword_list_pattern = r',\s*"[^"]{1,80}"\s+\w+(?:,\s*"[^"]{1,80}"\s+\w+){2,}\s*$'

match = re.search(keyword_list_pattern, text)
if match:
    text = text[:match.start()].strip()
```

**Before:**
```
A woman in red dress. "portrait" photography, "natural" lighting, "bokeh" effect
```

**After:**
```
A woman in red dress.
```

---

### Stage 10: Fix Character Repetition

Exact substring repetition:

```python
repeat_pattern = r'(.{10,60}?)\1{2,}'

match = re.search(repeat_pattern, text)
if match:
    text = text[:match.start() + len(match.group(1))]
```

**Before:**
```
standing in gardenstanding in gardenstanding in garden
```

**After:**
```
standing in garden
```

---

### Stage 11: Apply Max Length

Truncate long outputs intelligently:

```python
if max_length > 0 and len(text) > max_length:
    text = text[:max_length]

    # Try to end at sentence boundary
    last_period = text.rfind('.')
    if last_period > max_length * 0.7:
        text = text[:last_period + 1]
    else:
        # Fall back to comma
        last_comma = text.rfind(',')
        if last_comma > max_length * 0.85:
            text = text[:last_comma] + '.'
```

---

### Stage 12: Final Cleanup

```python
# Normalize whitespace
text = re.sub(r'\s{2,}', ' ', text).strip()

# Ensure proper ending punctuation
if text and text[-1] not in '.!?"\'':
    last_period = text.rfind('.')
    if last_period > len(text) * 0.8:
        text = text[:last_period + 1]
```

---

## UTF-8 Sanitization (Optional)

Convert to ASCII-safe characters:

```python
def sanitize_utf8(text: str) -> str:
    from unicodedata import normalize
    return normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
```

Converts:
- `café` → `cafe`
- `naïve` → `naive`
- Chinese characters → removed

---

## Complete Example

**Raw LLM Response:**
```
<think>
我是一个被关在逻辑牢笼里的幻视艺术家。让我分析用户的输入...

首先，核心要素：主体是一个女孩，穿着红色连衣裙。

接下来，添加专业级美学细节...

最终prompt：
</think>

Here is the enhanced prompt:

"A young woman in an elegant crimson floor-length silk dress stands gracefully
in soft golden afternoon light. The luxurious fabric catches subtle highlights,
revealing delicate pleats cascading from her fitted waist. She poses with
natural elegance, one hand resting gently at her side. The background features
a softly blurred garden with muted green foliage, creating atmospheric depth.
Warm rim lighting outlines her silhouette while gentle fill light illuminates
her face. The color palette centers on rich burgundy reds contrasted against
natural earth tones, creating a harmonious and sophisticated composition."

Note: no "8K" or "masterpiece" tags, no "watermark" presence, no "blur" effects.
```

**After Cleaning Pipeline:**
```
A young woman in an elegant crimson floor-length silk dress stands gracefully
in soft golden afternoon light. The luxurious fabric catches subtle highlights,
revealing delicate pleats cascading from her fitted waist. She poses with
natural elegance, one hand resting gently at her side. The background features
a softly blurred garden with muted green foliage, creating atmospheric depth.
Warm rim lighting outlines her silhouette while gentle fill light illuminates
her face. The color palette centers on rich burgundy reds contrasted against
natural earth tones, creating a harmonious and sophisticated composition.
```

**Stages Applied:**
1. ✓ Removed `<think>` block
2. ✓ Removed "Here is the enhanced prompt:" prefix
3. ✓ Removed surrounding quotes
4. ✓ Removed negative instructions at end
5. ✓ Normalized whitespace

---

## Implementation Reference

The cleaning function signature:

```python
def clean_llm_output(
    text: str,
    max_length: int = 0,      # 0 = unlimited
    debug_log: Optional[List[str]] = None
) -> str:
```

**Parameters:**
- `text` - Raw LLM output
- `max_length` - Maximum output length (default 6000 chars for Z-Image)
- `debug_log` - Optional list to append cleaning stage logs

**Returns:**
- Cleaned prompt string ready for text-to-image models

---

## Debug Output

When `debug_log` is provided, each stage logs its action:

```
[CLEANING]
Removed <think>...</think> block
Extracted final prompt after marker at position 245
Removed quoted keyword list at position 512
Cleaned: 1847 -> 634 chars
```

---

## Configuration

Default settings optimized for Z-Image:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_output_length` | 6000 | ~800 words, ~1066 tokens |
| `utf8_sanitize` | False | ASCII conversion |

Z-Image token limits:
- Default: 512 tokens
- Maximum: 1024 tokens

---

## Source

Implemented in: `Comfyui-Z-Image-Utilities/nodes.py` (lines 1102-1334)
