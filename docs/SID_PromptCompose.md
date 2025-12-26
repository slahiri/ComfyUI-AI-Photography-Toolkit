# SID_PromptCompose Node

Convert multi-model image metadata into natural language prompts optimized for specific image generation models.

## Overview

SID_PromptCompose takes analysis metadata from SID_ImageAnalysis and transforms it into structured, natural language prompts. Unlike traditional tag-based prompts (SDXL/SD style), this node generates **full descriptive sentences** following target model requirements.

## Node Specification

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | IMAGE | Yes | Input image (passed through to output) |
| `metadata` | SID_METADATA | Yes | JSON metadata from SID_ImageAnalysis |
| `provider` | DROPDOWN | Yes | LLM provider selection |
| `model` | DROPDOWN | Yes | Model selection (dynamic based on provider) |
| `output_style` | DROPDOWN | Yes | Target prompt format |
| `mode` | DROPDOWN | Yes | Generation approach |
| `api_key` | STRING | No | API key for cloud providers |
| `hf_token` | STRING | No | HuggingFace token for gated local models |
| `max_tokens` | INT | No | Maximum output tokens (default: 300) |
| `temperature` | FLOAT | No | Sampling temperature for LLM mode (default: 0.7) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `IMAGE` | IMAGE | Pass-through of input image |
| `SID_METADATA` | SID_METADATA | Pass-through of input metadata |
| `prompt` | STRING | Generated natural language prompt |

## Provider Options

| Provider | Models | VRAM/Requirements |
|----------|--------|-------------------|
| `Local` | Qwen3-VL-2B-Abliterated | ~8GB VRAM (4-bit) |
| `Local` | Qwen2.5-VL-3B | ~10GB VRAM (4-bit) |
| `Local` | Qwen2.5-VL-7B | ~12GB VRAM (4-bit) |
| `Anthropic` | claude-sonnet-4-20250514, claude-haiku | API key required |
| `OpenAI` | gpt-4o, gpt-4o-mini | API key required |
| `Gemini` | gemini-2.0-flash, gemini-1.5-pro | API key required |

## Output Styles

### z-image (Default)

Optimized for Z-Image/Flux-style models that use natural language prompts.

**Key Differences from SDXL/SD:**

| Aspect | SDXL/Stable Diffusion | Z-Image/Flux |
|--------|----------------------|--------------|
| **Syntax** | `tag1, tag2, (emphasis:1.2)` | Full descriptive sentences |
| **Negative Prompts** | Supported | NOT supported |
| **Quality Boosters** | `masterpiece, best quality, 8k` | NOT needed (auto-removed) |
| **Token Limit** | 75-150 tokens | Up to 1024 tokens |
| **Structure** | Comma-separated tags | Ordered paragraphs |

**Z-Image Template Structure:**

```
1. Subject Specification    → Who/What with detailed attributes
2. Action/State            → What they're doing or how they exist
3. Environment             → Setting, location, time of day
4. Lighting & Atmosphere   → Mood, weather, light quality
5. Visual Style            → Medium, artistic reference, camera/film
6. Composition             → Framing, angle, focus
7. Text Rendering          → Exact text in "double quotes" (if any)
```

## Generation Modes

### Fast Mode (Non-LLM)

**Approach:** Pure Python rule-based template engine

**Characteristics:**
- Speed: ~1-5ms
- Dependencies: None (pure Python)
- Deterministic: Yes (same input = same output)
- Best for: High-volume batch processing, quick previews

**Algorithm:**
1. Extract tags from all taggers (wd14, joytag, pixai, etc.)
2. Classify tags into semantic categories using regex patterns
3. Build sentences using predefined templates
4. Merge into structured prompt following Z-Image order

**Example Output:**
```
A asian woman with black hair and long hair and brown eyes. She has lips,
makeup and red lips. She is standing. Her gaze is looking at viewer.
She is wearing a dress. She is accessorized with jewelry and bracelet.
The scene is set indoors. Captured with depth of field and cowboy shot.
```

### Standard Mode (NLP)

**Approach:** Statistical clustering with semantic grouping + Florence integration

**Characteristics:**
- Speed: ~10-50ms
- Dependencies: None (pure Python NLP)
- Deterministic: Yes
- Best for: Balanced quality/speed, general use

**Algorithm:**
1. Extract and deduplicate tags across all sources
2. Cluster tags into semantic groups using keyword matching
3. Extract structured parts from Florence caption (subject, clothing, setting)
4. Merge tag clusters with Florence descriptions
5. Build natural sentences with proper flow

**Example Output:**
```
A photograph of a beautiful Indian woman with long, straight black hair,
wearing a red and white saree that is partially draped over her shoulder.
She has makeup and subtle lips. She is standing with a confident expression.
She is accessorized with jewelry and bracelet. The scene is set in a dimly
lit room. The lighting is soft, creating an intimate atmosphere. Captured
as a cowboy shot with shallow depth of field.
```

### LLM Mode

**Approach:** AI-based synthesis using selected LLM provider

**Characteristics:**
- Speed: 500ms - 3s (depends on provider)
- Dependencies: API client or local model
- Deterministic: No (temperature-based variation)
- Best for: Production quality, nuanced descriptions

**Algorithm:**
1. Extract top tags with confidence scores
2. Get Florence caption and photography analysis
3. Build structured prompt using PE (Prompt Engineering) template
4. Send to LLM with Z-Image specific instructions
5. Return generated natural language prompt

**Example Output:**
```
A confident Indian woman with long, straight black hair cascading down
her back stands in an elegantly posed position with one hand resting
on her head. She has warm brown eyes, full lips with subtle red lipstick,
and a gentle smile. She is wearing a traditional red and white saree
with intricate floral embroidery along the border, the pallu draped
gracefully over her shoulder revealing her bare shoulders. A black
bracelet adorns her left wrist. The scene is set in a dimly lit interior
room with warm wooden walls visible in the background. The lighting is
soft and warm, creating intimate shadows and a low-key atmospheric mood
with a slight golden glow. Captured as a medium cowboy shot with the
subject centrally positioned, featuring shallow depth of field with
soft bokeh in the background, emphasizing her natural beauty.
```

## Semantic Categories

Tags are classified into these categories for structured output:

| Category | Keywords/Patterns | Priority |
|----------|-------------------|----------|
| `subject` | woman, girl, man, boy, person | 1 |
| `ethnicity` | asian, indian, european, african, latin | 2 |
| `hair` | hair, haired, black hair, long hair, wavy | 3 |
| `eyes` | eyes, brown eyes, blue eyes, gaze | 4 |
| `facial` | lips, smile, expression, makeup, face | 5 |
| `pose` | standing, sitting, posing, arm, hand | 6 |
| `gaze` | looking at viewer, eye contact | 7 |
| `clothing` | dress, saree, robe, outfit, wearing | 8 |
| `accessories` | bracelet, jewelry, necklace, watch | 9 |
| `environment` | indoor, outdoor, room, background, studio | 10 |
| `lighting` | light, dim, soft, warm, dark, shadow | 11 |
| `atmosphere` | mood, intimate, dramatic, peaceful | 12 |
| `camera` | shot, portrait, bokeh, focus, depth | 13 |
| `style` | photo, realistic, cinematic, artistic | 14 |

## Tags Automatically Excluded

These tags are filtered out as Z-Image doesn't need them:

```python
EXCLUDE_TAGS = {
    # Quality boosters (Z-Image handles automatically)
    'masterpiece', 'best quality', 'highly detailed', '8k', '4k', 'uhd',
    'high resolution', 'trending on artstation', 'award winning',

    # Metadata artifacts
    'artist name', 'signature', 'watermark', 'web address',
    'patreon username', 'twitter username', 'dated',

    # Count prefixes (not needed for natural language)
    '1girl', '1boy', '2girls', 'solo', 'multiple girls',

    # Generic/unhelpful
    'sensitive', 'nsfw content'
}
```

## LLM Prompt Template

The template sent to LLM providers in LLM mode:

```
You are an expert prompt engineer for Z-Image, a text-to-image model.

CRITICAL Z-IMAGE RULES:
1. Write FULL DESCRIPTIVE SENTENCES (not comma-separated tags)
2. NO negative prompts supported - all guidance must be positive
3. NO quality boosters: "masterpiece", "best quality", "8k", "trending on artstation"
4. Describe spatial relationships clearly
5. Structure: Subject → Appearance → Pose → Clothing → Environment → Lighting → Technical

STRUCTURE:
- Subject: Who/what with physical attributes (ethnicity, age, features)
- Appearance: Facial features, expression, skin, makeup
- Pose: Body position, hand placement, gaze direction
- Clothing: Garments with colors, patterns, draping style
- Environment: Setting, background, atmospheric details
- Lighting: Type, direction, quality, mood
- Technical: Camera angle, depth of field, composition

INPUT DATA:
-----------
Image: {width}x{height}

Top Tags: {tags_with_confidence}

Florence Caption: {florence_caption}

Photography Analysis: {photography_attributes}

TASK: Generate a single cohesive Z-Image prompt (150-250 words).
Output ONLY the prompt, no explanations.
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_words` | int | 250 | Maximum words in output |
| `min_confidence` | float | 0.4 | Minimum tag confidence to include |
| `use_florence_base` | bool | True | Use Florence caption as base for NLP mode |
| `temperature` | float | 0.7 | LLM sampling temperature |
| `max_tokens` | int | 300 | Maximum LLM output tokens |

## Workflow Integration

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Image     │────▶│ SID_ImageAnalysis│────▶│ SID_PromptCompose│
│   Input     │     │                  │     │                  │
└─────────────┘     │ metadata output  │     │ prompt output    │
                    └──────────────────┘     └──────────────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │  Z-Image/Flux    │
                                             │  Generation      │
                                             └──────────────────┘
```

## Mode Comparison

| Aspect | Fast | Standard | LLM |
|--------|------|----------|-----|
| Speed | ~1-5ms | ~10-50ms | 500ms-3s |
| Cost | Free | Free | API costs |
| Natural Flow | Basic | Good | Excellent |
| Semantic Merge | Limited | Good | Excellent |
| Florence Use | No | Yes | Yes |
| Deterministic | 100% | 100% | Variable |
| Best For | Batch/Preview | General Use | Production |

## Recommendations

1. **Batch Processing (100s of images):** Use Fast mode
2. **Interactive/General Use:** Use Standard mode with Florence
3. **Production/Commercial:** Use LLM mode with Claude or GPT-4
4. **Preview then Refine:** Fast for preview → LLM for final

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| No metadata | Missing SID_ImageAnalysis | Connect metadata input |
| API key invalid | Wrong/missing API key | Check api_key input |
| Model not found | Invalid model selection | Update model dropdown |
| OOM on local | Insufficient VRAM | Use smaller model or 4-bit |
| Empty prompt | No valid tags extracted | Lower min_confidence threshold |

## Future Output Styles

The `output_style` dropdown is designed for extensibility:

- `z-image` - Current (full sentences for Flux/Z-Image)
- `sdxl` - Future (comma-separated tags with weights)
- `midjourney` - Future (MJ-style with parameters)
- `dalle` - Future (DALL-E optimized descriptions)
