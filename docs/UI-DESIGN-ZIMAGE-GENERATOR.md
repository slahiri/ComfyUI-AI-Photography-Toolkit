# UI Design: SID_Z_Image_Prompt_Generator

> Visual design specification for the Z-Image Prompt Generator node

---

## Node Overview

**Node ID:** `SID_ZImagePromptGenerator`
**Display Name:** `SID Z-Image Prompt Generator`
**Category:** `SID Photography Toolkit/Z-Image`

**Purpose:** Analyze an input image and generate a Z-Image compatible narrative prompt

---

## UI Layout (Top to Bottom)

```
┌─────────────────────────────────────────────────────────────────┐
│  SID Z-Image Prompt Generator                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  IMAGE INPUT                                    [●]      │   │
│  │  ○ image                                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ════════════════════ API SETTINGS ═══════════════════════     │
│                                                                 │
│  API Key        [••••••••••••••••••••••••••••••••]              │
│                                                                 │
│  Model          [claude-sonnet-4-5-20250929      ▼]            │
│                                                                 │
│  ════════════════════ STYLE PRESETS ══════════════════════     │
│                                                                 │
│  Detail Level   [Detailed                        ▼]            │
│                  ├─ Minimal (50-80 words)                      │
│                  ├─ Detailed (100-180 words) ✓                 │
│                  ├─ Technical (150-220 words)                  │
│                  └─ Comprehensive (200-300 words)              │
│                                                                 │
│  Color Style    [None                            ▼]            │
│                  ├─ None (analyze from image) ✓                │
│                  ├─ Black and White                            │
│                  ├─ Color                                      │
│                  ├─ Monochrome                                 │
│                  ├─ Sepia                                      │
│                  ├─ High Contrast B&W                          │
│                  ├─ Low Key                                    │
│                  └─ High Key                                   │
│                                                                 │
│  Photographer   [None                            ▼]            │
│                  ├─ None ✓                                     │
│                  ├─ Helmut Newton                              │
│                  ├─ Peter Lindbergh                            │
│                  ├─ Annie Leibovitz                            │
│                  ├─ Richard Avedon                             │
│                  └─ ... (20 options)                           │
│                                                                 │
│  Lighting       [None                            ▼]            │
│                  ├─ None (analyze from image) ✓                │
│                  ├─ ─── Studio ───                             │
│                  ├─ Natural Window Light                       │
│                  ├─ Studio Strobes                             │
│                  ├─ Softbox Lighting                           │
│                  ├─ ─── Techniques ───                         │
│                  ├─ Rembrandt Lighting                         │
│                  ├─ Butterfly Lighting                         │
│                  ├─ ─── Outdoor ───                            │
│                  ├─ Golden Hour                                │
│                  ├─ Blue Hour                                  │
│                  └─ ... (31 options)                           │
│                                                                 │
│  ════════════════════ OUTPUT OPTIONS ═════════════════════     │
│                                                                 │
│  Text Quoting   [☑] Quote text elements ("text")               │
│                                                                 │
│  Max Length     [═══════════════════●════] 2000                │
│                  0                      4000                   │
│                                                                 │
│  ════════════════════ GENERATION ═════════════════════════     │
│                                                                 │
│  User Prompt    ┌─────────────────────────────────────────┐    │
│  (optional)     │ Add any specific instructions...        │    │
│                 │                                         │    │
│                 └─────────────────────────────────────────┘    │
│                                                                 │
│  Temperature    [═════════●═══════════════] 0.7                │
│                  0.0                       1.0                 │
│                                                                 │
│  Seed           [0                              ] [🎲]          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  OUTPUTS                                                        │
│                                                                 │
│  prompt ○────────────────────────────────────────────────────   │
│  debug_log ○─────────────────────────────────────────────────   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input Parameters Specification

### Required Inputs

| # | Parameter | Type | Widget | Default | Description |
|---|-----------|------|--------|---------|-------------|
| 1 | `image` | IMAGE | Input slot | - | Input image to analyze |
| 2 | `api_key` | STRING | Password field | "" | Anthropic API key |

### Style Presets (Combo Dropdowns)

| # | Parameter | Type | Default | Options | Tooltip |
|---|-----------|------|---------|---------|---------|
| 3 | `model` | COMBO | claude-sonnet-4-5 | See models list | Claude model for analysis |
| 4 | `detail_level` | COMBO | "Detailed" | Minimal, Detailed, Technical, Comprehensive | Output detail and word count |
| 5 | `color_style` | COMBO | "None" | None + 8 styles | Color treatment to apply |
| 6 | `photographer_style` | COMBO | "None" | None + 20 photographers | Emulate famous photographer |
| 7 | `lighting_style` | COMBO | "None" | None + 31 lighting setups | Specific lighting to describe |

### Output Options

| # | Parameter | Type | Widget | Default | Description |
|---|-----------|------|--------|---------|-------------|
| 8 | `include_text_quotes` | BOOLEAN | Checkbox | True | Wrap text elements in "quotes" |
| 9 | `max_length` | INT | Slider | 2000 | Maximum output characters (0-4000) |

### Generation Settings

| # | Parameter | Type | Widget | Default | Description |
|---|-----------|------|--------|---------|-------------|
| 10 | `user_prompt` | STRING | Multiline text | "" | Additional instructions (optional) |
| 11 | `temperature` | FLOAT | Slider | 0.7 | Creativity level (0.0-1.0) |
| 12 | `seed` | INT | Number + randomize | 0 | Random seed for reproducibility |

---

## Output Parameters

| Output | Type | Description |
|--------|------|-------------|
| `prompt` | STRING | Z-Image compatible narrative prompt |
| `debug_log` | STRING | Debug information and API response details |

---

## Model Options

```python
CLAUDE_MODELS = [
    "claude-sonnet-4-5-20250929",    # Latest Sonnet (default)
    "claude-haiku-4-5-20251001",     # Fast, economical
    "claude-opus-4-1-20250805",      # Most capable
    "claude-3-5-haiku-20241022",     # Previous gen fast
    "claude-3-haiku-20240307",       # Previous gen economical
]
```

---

## Detail Level Specifications

| Level | Word Count | Use Case | Description |
|-------|------------|----------|-------------|
| **Minimal** | 50-80 | Quick tests, simple scenes | Core elements only: subject, setting, lighting, style |
| **Detailed** | 100-180 | Standard use (default) | Balanced: composition, subject details, lighting, atmosphere |
| **Technical** | 150-220 | Photography recreation | Includes camera feel, lens characteristics, technical aspects |
| **Comprehensive** | 200-300 | Maximum detail | Full description: every visible element, materials, textures, spatial relationships |

---

## Color Style Options

| Option | Description | When to Use |
|--------|-------------|-------------|
| None | Analyze from image | Let AI determine color treatment |
| Black and White | Monochrome, grayscale | Classic B&W photography |
| Color | Full color, vibrant | Standard color photography |
| Monochrome | Single color tone | Artistic monotone |
| Sepia | Warm brown vintage | Nostalgic, historical feel |
| High Contrast B&W | Dramatic monochrome | Bold, graphic B&W |
| Low Key | Dark, moody tones | Dramatic, shadowy |
| High Key | Bright, airy | Light, ethereal |

---

## Photographer Style Options (20)

Organized by aesthetic:

**Classic Masters:**
- Helmut Newton, Richard Avedon, Irving Penn, Herb Ritts

**Natural/Intimate:**
- Peter Lindbergh, Annie Leibovitz, Patrick Demarchelier, Paolo Roversi

**Fashion Editorial:**
- Mario Testino, Steven Meisel, Mert and Marcus, Bruce Weber

**Creative/Artistic:**
- Tim Walker, David LaChapelle, Steven Klein, Ellen von Unwerth

**Contemporary/Raw:**
- Juergen Teller, Terry Richardson, Rankin, Albert Watson

---

## Lighting Style Options (31)

**Studio Lighting (8):**
- Natural Window Light, Studio Strobes, Softbox Lighting, Beauty Dish
- Ring Light, Umbrella Lighting, Grid Spot, Reflector Fill

**Studio Techniques (10):**
- Rembrandt, Split, Butterfly, Loop, Broad, Short Lighting
- High Key Studio, Low Key Studio, Clamshell, Edge/Rim Lighting

**Outdoor/Natural (9):**
- Golden Hour, Blue Hour, Harsh Midday Sun, Overcast Diffused
- Open Shade, Backlit, Dusk/Twilight, Sunrise, Sunset

**Special/Creative (4):**
- Chiaroscuro, Dramatic Side Light, Silhouette, Candlelight
- Neon/Colorful, Practical Lights, Mixed Lighting, Night Photography

---

## Tooltips

```python
TOOLTIPS = {
    "image": "Input image to analyze and generate a Z-Image prompt from",
    "api_key": "Anthropic API key (get from https://console.anthropic.com/)",
    "model": "Claude model for image analysis. Sonnet recommended for balance of speed/quality",
    "detail_level": "Controls output length and detail. Minimal=50-80 words, Comprehensive=200-300 words",
    "color_style": "Override color treatment. 'None' analyzes from image",
    "photographer_style": "Emulate a famous photographer's aesthetic. 'None' for neutral style",
    "lighting_style": "Specify lighting description. 'None' analyzes from image",
    "include_text_quotes": "Wrap any text elements in double quotes for Z-Image text rendering",
    "max_length": "Maximum output characters. Z-Image works best with 1500-3000 chars",
    "user_prompt": "Additional instructions to guide the prompt generation",
    "temperature": "Creativity level. Lower=focused, Higher=creative. 0.7 recommended",
    "seed": "Random seed for reproducible results. Use different seeds for variation",
}
```

---

## Visual Grouping (UI Sections)

### Section 1: Input
- Image input slot (required)

### Section 2: API Settings
- API Key (password field)
- Model selector

### Section 3: Style Presets
- Detail Level
- Color Style
- Photographer Style
- Lighting Style

### Section 4: Output Options
- Text Quoting checkbox
- Max Length slider

### Section 5: Generation
- User Prompt (multiline, optional)
- Temperature slider
- Seed with randomize button

### Section 6: Outputs
- prompt (STRING)
- debug_log (STRING)

---

## Widget Specifications

### Sliders

```python
# Temperature
"temperature": ("FLOAT", {
    "default": 0.7,
    "min": 0.0,
    "max": 1.0,
    "step": 0.05,
    "round": 0.01,
    "display": "slider",
})

# Max Length
"max_length": ("INT", {
    "default": 2000,
    "min": 0,
    "max": 4000,
    "step": 100,
    "display": "slider",
})
```

### Seed with Control

```python
"seed": ("INT", {
    "default": 0,
    "min": 0,
    "max": 2147483647,
    "control_after_generate": True,  # Adds randomize button
})
```

### Multiline Text

```python
"user_prompt": ("STRING", {
    "default": "",
    "multiline": True,
    "placeholder": "Add specific instructions (optional)...",
})
```

---

## Node Colors (Suggested)

Following ComfyUI conventions:
- **Header:** Purple/Violet (`#6B5B95`) - indicates AI/LLM node
- **Body:** Dark gray (`#353535`)
- **Inputs:** Green connection dots
- **Outputs:** Blue/Cyan connection dots

---

## Comparison with Existing Nodes

| Feature | SID_AIPromptGenerator | Z-Image Utilities | SID_ZImagePromptGenerator |
|---------|----------------------|-------------------|---------------------------|
| Image Input | ✓ | Optional | ✓ (required) |
| Keyword Output | ✓ | ✗ | ✗ |
| Narrative Output | ✗ | ✓ | ✓ |
| Photography Presets | ✓ | ✗ | ✓ |
| Multi-Provider | ✗ | ✓ | ✗ (Claude only) |
| Detail Levels | 4 styles | ✗ | 4 levels |
| Text Quoting | ✗ | Built-in | ✓ (toggle) |
| Session Support | ✗ | ✓ | ✗ (future) |

---

## Example Usage Flow

```
1. User loads image
2. Connects to SID_ZImagePromptGenerator
3. Selects:
   - Detail Level: "Detailed"
   - Photographer Style: "Peter Lindbergh"
   - Lighting: "None" (auto-detect)
4. Node analyzes image
5. Outputs narrative prompt like:

   "A medium shot portrait of an adult woman in her early thirties
    with shoulder-length dark wavy hair and olive skin. She wears
    a simple white cotton t-shirt, looking directly at camera with
    a relaxed, natural expression. Soft diffused window light from
    the left creates gentle shadows on her face, emphasizing natural
    skin texture. The background is a blurred neutral gray studio
    backdrop. The image has a raw, intimate quality reminiscent of
    Peter Lindbergh's authentic portraiture style - minimal retouching,
    natural beauty, film noir influenced lighting. Shallow depth of
    field, centered composition, candid documentary feel."

6. User connects prompt output to Z-Image model
```

---

## Implementation Notes

1. **API Key Security:** Use password field type, never log the key
2. **Image Processing:** Reuse `_image_to_base64()` from existing node
3. **Error Handling:** Return error message in both outputs on failure
4. **Debug Log:** Include all settings, API response time, token counts
5. **Validation:** Check API key before making request

---

*Created: December 2025*
*For: ComfyUI-AI-Photography-Toolkit Z-Image Integration*
