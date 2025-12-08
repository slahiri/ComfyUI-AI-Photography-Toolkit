# Z-Image Prompting Best Practices Guide

> Comprehensive research compiled for integrating Z-Image prompt compatibility into ComfyUI-AI-Photography-Toolkit.

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [Core Principles](#core-principles)
3. [Technical Settings](#technical-settings)
4. [Prompt Structure](#prompt-structure)
5. [The Camera-First Approach](#the-camera-first-approach)
6. [Visual Vocabulary](#visual-vocabulary)
7. [Genre-Specific Formulas](#genre-specific-formulas)
8. [Controlling Output Without Negative Prompts](#controlling-output-without-negative-prompts)
9. [Bilingual Text Rendering](#bilingual-text-rendering)
10. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
11. [Z-Image vs Traditional SD Comparison](#z-image-vs-traditional-sd-comparison)
12. [Integration Considerations](#integration-considerations)
13. [Sources](#sources)

---

## Model Overview

**Z-Image-Turbo** is a 6 billion parameter text-to-image model from Tongyi-MAI featuring:

- **Scalable Single-Stream DiT architecture** - processes text and image tokens together
- **Sub-second inference latency** on H800 GPUs
- **16GB consumer VRAM compatible**
- **Qwen3-4B text encoder** - excellent natural language understanding
- **Bilingual support** - English and Chinese text rendering
- **Few-step distilled** - quality baked in, no CFG needed

### Model Variants

| Variant | Purpose | Steps |
|---------|---------|-------|
| Z-Image-Turbo | Fast inference, distilled | 8-9 steps |
| Z-Image-Base | Foundation model for fine-tuning | Standard |
| Z-Image-Edit | Image-to-image transformation | Varies |

---

## Core Principles

### The Golden Rule

> **"Every word should describe something visible."**

The model cannot interpret abstract concepts like "beautiful," "mysterious," or "interesting." It only understands colors, shapes, textures, materials, and compositions.

### Key Differences from Stable Diffusion

| Feature | Stable Diffusion | Z-Image-Turbo |
|---------|------------------|---------------|
| Negative Prompts | Essential | **Not Used** |
| Guidance Scale | 7-12 typical | **0.0** |
| Prompt Style | Keywords, tags | **Natural language narrative** |
| Quality Tags | "masterpiece, 8K" | **Not needed** (baked in) |
| Token Limit | 75-77 (CLIP) | **512-1024** |

---

## Technical Settings

### Optimal Parameters

```python
# Recommended settings for Z-Image-Turbo
pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,      # Results in 8 DiT forward passes
    guidance_scale=0.0,         # CFG is baked in - DO NOT change
    max_sequence_length=1024,   # Enable long prompts
    generator=torch.Generator("cuda").manual_seed(seed)
)
```

### Token Limits

| Context | Limit | Notes |
|---------|-------|-------|
| Online Demo | 512 tokens | Default limit |
| Local Usage | 1024 tokens | Set `max_sequence_length=1024` |
| Sweet Spot | 80-250 words | ~100-333 tokens |

### Resolution

- **Native**: 1024×1024
- **Drafts**: 768 or 512
- **Aspect Ratios**: Supported with padding

---

## Prompt Structure

### Universal Formula

```
[Shot & Subject] + [Age & Appearance] + [Clothing] + [Environment] + [Lighting] + [Mood] + [Style/Medium] + [Technical Notes] + [Safety Constraints]
```

### Detailed Breakdown

```
[Composition]     → Shot type, angle, framing
[Character]       → Age, ethnicity, features, expression
[Clothing]        → Outfit details, colors, condition
[Environment]     → Background, setting, props
[Lighting]        → Direction, quality, color temperature
[Style]           → Photography style, medium
[Technical]       → Camera, lens, depth of field
[Constraints]     → Safety clauses, artifact prevention
```

### Example Template

```
A close-up headshot of an adult woman in her 30s, friendly confident
expression, dark medium-length hair, wearing simple dark blazer over
light shirt, studio portrait, soft diffused front lighting, subtle
blurred gray background, realistic photography, 85mm lens, shallow
depth of field, fully clothed, modest professional outfit, no jewelry
except small necklace, no logos, no text, no watermark, safe for work.
```

---

## The Camera-First Approach

Z-Image behaves best when treated like a **camera instruction manual** rather than a creative writing exercise.

### Do This

```
Close-up portrait, 45-degree angle, woman looking slightly left,
soft key light from upper right, rim light on hair, dark backdrop
```

### Not This

```
A mysteriously beautiful woman with an enigmatic gaze that speaks
volumes about her inner world, captured in a moment of quiet contemplation
```

### Shot Types

| Type | Use Case |
|------|----------|
| Close-up | Face, details |
| Medium shot | Torso up |
| Full-body | Complete figure |
| Wide shot | Subject + environment |

### Camera Angles

| Angle | Effect |
|-------|--------|
| Front view | Direct, confrontational |
| 45-degree | Dynamic, flattering |
| Profile | Dramatic, artistic |
| Looking up | Powerful, imposing |
| Looking down | Vulnerable, intimate |

---

## Visual Vocabulary

### Skin & Complexion

**Tone**: warm ivory, cool beige, golden tan, deep brown, porcelain, olive

**Texture**: smooth, weathered, freckled, dewy, matte, sun-damaged

**Details**: beauty marks, stubble, acne scars, laugh lines

### Eyes

**Shape**: almond, round, hooded, deep-set, upturned, downturned

**Color**: steel grey, warm brown, pale blue, amber, hazel, green

**State**: bright and alert, half-lidded, bloodshot, glassy, sharp focus

### Hair

**Texture**: straight and silky, wavy, tight curls, coily, frizzy, fine

**State**: clean and fresh, windswept, disheveled, damp, slicked back

**Style**: bob, pixie cut, long layers, braided, ponytail, loose

### Lighting Keywords (Z-Image responds exceptionally well to these)

| Type | Description |
|------|-------------|
| Soft diffused daylight | Natural, flattering |
| Cinematic warm key light | Film-like, dramatic |
| Noir high-contrast | Dramatic shadows |
| Studio portrait lighting | Professional, controlled |
| Rim lighting | Edge definition, separation |
| Rembrandt lighting | Classic triangle on cheek |
| Butterfly lighting | Beauty, glamour |
| Split lighting | Half-face illuminated |
| Loop lighting | Soft shadows, natural |

---

## Genre-Specific Formulas

### Portrait Photography

```
[Shot type] of [age] [ethnicity] [gender], [facial features],
[eye color] eyes, [hair details], [expression], wearing [clothing],
[pose], [background], [lighting setup], [camera] with [lens],
[depth of field]
```

**Example:**
```
Medium close-up portrait of adult East Asian woman in her late 20s,
oval face with high cheekbones, warm brown almond-shaped eyes,
shoulder-length straight black hair with subtle highlights,
gentle smile with closed lips, wearing cream cashmere turtleneck,
shoulders slightly angled, soft gray studio backdrop,
Rembrandt lighting with fill from left, Canon EOS R5 with 85mm f/1.4,
shallow depth of field with soft bokeh
```

### Full Body / Fashion

```
[Shot type] of [build] [age] [person], [posture], [complete outfit],
[footwear], [accessories], [specific pose], [environment],
[lighting], [camera angle]
```

### Action Shots

```
[Action description], [body position], [leading limbs],
[fabric/hair movement], [expression matching effort],
[motion blur if any], [dramatic lighting], [camera angle]
```

### Product / Still Life

```
[Product], [materials], [colors], [textures], [arrangement],
[props], [surface], [lighting direction], [angle],
professional product photography
```

### Landscape

```
[Foreground elements], [middle ground], [horizon], [sky conditions],
[atmospheric effects], [color palette], [mood], [time of day],
wide-angle photography
```

---

## Controlling Output Without Negative Prompts

Since Z-Image-Turbo does NOT support negative prompts, all constraints must be embedded positively.

### Safety & Content Control

```
# Always include for human subjects:
adult, fully clothed, modest outfit, safe for work, non-sexual

# Artifact prevention:
no text, no watermark, no logos, no extra limbs, no artifacts

# Anatomical correctness:
natural hands and fingers, correct human anatomy, five fingers per hand
```

### The Repetition Technique

Reinforce important constraints multiple times:

```
...wearing modest business suit, fully clothed, professional attire,
covered shoulders, safe for work, non-sexual, appropriate workplace attire...
```

### End-of-Prompt Safety Clause

Always append safety constraints at the end:

```
...[main prompt content]..., safe for work, non-sexual, no nudity,
no revealing clothing, no text, no watermark, correct anatomy
```

### Removing Default Associations

Some words carry "baggage" (default visual associations):

| Instead of | Use |
|------------|-----|
| "CEO" | "office worker, adult woman, professional attire" |
| "model" | "adult woman, casual clothing, natural pose" |
| "businessman" | "software developer, adult man, wearing hoodie" |

---

## Bilingual Text Rendering

Z-Image-Turbo excels at rendering both English and Chinese text in images.

### Best Practices

1. **Keep text in single languages** - don't mix within one text element
2. **Describe placement precisely**: `large white title at top center`
3. **Guard text integrity**: `no additional text except title, no random watermarks`
4. **Quote exact text**: `sign displaying "OPEN 24 HOURS"`

### Text Quotation Syntax

For any text that should appear in the image, use English double quotes:

```
A coffee shop storefront with a chalkboard sign displaying "Today's Special: Vanilla Latte $4.50"
```

```
Movie poster with bold red title "THE LAST SUNSET" at the top
```

---

## Anti-Patterns to Avoid

### Don't Use These

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| "beautiful" | Not visible | Describe specific features |
| "mysterious" | Abstract | Describe expression, lighting |
| "nice" | Subjective | State exact colors/materials |
| "8K, masterpiece" | Meta-tags | Quality is baked in |
| "best quality" | Not needed | Already optimized |
| Long backstories | Invisible | Focus on visuals only |
| Celebrity names | Unreliable | Describe actual features |
| Contradictions | Confusing | Maintain consistency |

### Keyword Lists vs Narratives

**Don't (keyword style):**
```
woman, portrait, beautiful, elegant, professional, studio, lighting,
high quality, 8K, masterpiece, detailed, sharp
```

**Do (narrative style):**
```
A professional studio portrait of an adult woman in her 30s with
warm brown eyes and shoulder-length auburn hair. She wears a navy
blazer and white blouse, looking directly at camera with a confident
expression. Soft key light from upper left creates subtle shadows,
gray seamless backdrop, shallow depth of field.
```

---

## Z-Image vs Traditional SD Comparison

### Prompt Translation Examples

**For "Portrait of a woman":**

| Model | Prompt Style |
|-------|--------------|
| SD 1.5 | `portrait of woman, professional photo, studio lighting, 8K, masterpiece, best quality, detailed face, sharp focus` |
| SDXL | `portrait photograph of a woman, professional studio lighting, high quality, detailed, sharp` |
| **Z-Image** | `A professional studio portrait of an adult woman with warm olive skin and dark wavy hair. She has bright hazel eyes and a gentle smile. Soft diffused lighting from front-left, neutral gray backdrop. Medium format photography, shallow depth of field, realistic skin texture.` |

### Token Efficiency

| Model | Optimal Length | Max Length |
|-------|----------------|------------|
| SD 1.5 | 50-75 tokens | 77 tokens |
| SDXL | 27-75 tokens | 77 tokens |
| Z-Image | 100-300 tokens | 1024 tokens |

---

## Integration Considerations

### For AI-Photography-Toolkit Enhancement

When generating prompts for Z-Image compatibility:

1. **Output Format**: Natural language narrative, not comma-separated keywords
2. **No Meta-Tags**: Remove "8K", "masterpiece", "best quality", etc.
3. **No Negative Section**: Embed all constraints positively
4. **Text Quotation**: Use `"quoted text"` for text-in-image
5. **Camera-First**: Structure like camera instructions
6. **Lighting Priority**: Z-Image responds exceptionally well to lighting descriptions
7. **Length**: 80-250 words is optimal
8. **Safety Clauses**: Append at end when needed

### System Prompt Modifications

The toolkit's Claude system prompt should instruct:

```
Generate prompts as flowing natural language descriptions, NOT keyword lists.

DO:
- Write complete sentences describing the visual scene
- Include specific lighting setups and directions
- Describe camera angle, shot type, and composition
- Use concrete, visible details (colors, textures, materials)
- Quote any text that should appear in the image: "text here"

DO NOT:
- Use comma-separated keyword lists
- Include meta-tags (8K, masterpiece, best quality, high quality)
- Use abstract adjectives (beautiful, mysterious, interesting)
- Include negative prompts or exclusions
- Add quality boosters or model-specific tags
```

### Output Cleaning

When outputting Z-Image compatible prompts:

1. Remove `<think>...</think>` tags
2. Strip markdown code blocks
3. Remove common prefixes ("Here is the enhanced prompt:")
4. Detect and truncate repetitive patterns
5. Enforce max character limit with smart truncation
6. Ensure proper sentence ending

---

## Sources

### Official Documentation
- [Tongyi-MAI/Z-Image-Turbo - HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Z-Image GitHub Repository](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image-Turbo Prompting Guide Discussion](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8)

### Community Guides
- [Z-Image-Turbo Prompting Guide.md - GitHub Gist](https://gist.github.com/illuminatianon/c42f8e57f1e3ebf037dd58043da9de32)
- [Z-Image Prompt Engineering Masterclass](https://z-image.vip/blog/z-image-prompt-engineering-masterclass)
- [Ultimate Z-Image Prompting Guide - Atlabs AI](https://www.atlabs.ai/blog/ultimate-z-image-prompting-guide)

### Tutorials
- [Z-Image Turbo: Fast Uncensored Image Generation in ComfyUI - Next Diffusion](https://www.nextdiffusion.ai/tutorials/z-image-turbo-fast-uncensored-image-generation-comfyui)
- [Z Image Turbo Tutorial - Stable Diffusion Tutorials](https://www.stablediffusiontutorials.com/2025/11/z-image-turbo.html)
- [How to Use Z-Image Turbo Guide 2025](https://z-image.cc/blog/how-to-use-z-image-turbo-guide-2025)

### ComfyUI Integration
- [Z-Image ComfyUI Workflow Example - ComfyUI Docs](https://docs.comfy.org/tutorials/image/z-image/z-image-turbo)
- [Comfyui-Z-Image-Utilities](https://github.com/Koko-boya/Comfyui-Z-Image-Utilities)

---

*Last Updated: December 2025*
*Compiled for ComfyUI-AI-Photography-Toolkit Z-Image Integration*
