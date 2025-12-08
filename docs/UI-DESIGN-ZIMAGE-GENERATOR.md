# UI Design: SID_Z_Image_Prompt_Generator (Agentic)

> Multi-stage agentic image analysis for Z-Image prompt generation

---

## Design Philosophy

**Agentic Multi-Stage Analysis:**
1. **Stage 1:** Classify scene type, shot framing, and photography genre
2. **Stage 2:** Extract image dimensions and metadata
3. **Stage 3:** Based on scene type, identify relevant attributes
4. **Stage 4:** Generate structured descriptions per attribute category
5. **Stage 5:** Compile final prompt with token tracking
6. **Stage 6:** Generate Z-Image recommendations

**No camera/technology details** - pure visual description only.
Technology details added by Enhancer node.

---

## Comprehensive Scene Classification

### A. Shot Framing (Distance to Subject)

| Code | Name | Frame Coverage | Primary Use |
|------|------|----------------|-------------|
| **ECU** | Extreme Close-Up | Single feature (eye, lips, detail) | Emotion, texture, detail |
| **BCU** | Big Close-Up | Face only (forehead to chin) | Intense emotion, beauty |
| **CU** | Close-Up | Head and neck | Expression, portrait |
| **MCU** | Medium Close-Up | Head and shoulders | Conversation, headshot |
| **MS** | Medium Shot | Waist up | Upper body, gestures |
| **MCS** | Medium Cowboy Shot | Mid-thigh up | Western style, holster visible |
| **MLS** | Medium Long Shot | Knees up (3/4 shot) | Body language, outfit |
| **FS** | Full Shot | Head to toe | Complete figure, fashion |
| **LS** | Long Shot | Full body + surroundings | Context, environment |
| **ELS** | Extreme Long Shot | Wide vista, tiny subject | Landscape, establishing |
| **VLS** | Very Long Shot | Subject small in frame | Scale, isolation |

### B. Photography Genre Classification

#### People Photography

| Genre | Code | Description | Key Attributes |
|-------|------|-------------|----------------|
| **Portrait** | PRT | Face-focused, expressive | Face, eyes, expression, lighting |
| **Headshot** | HST | Professional/actor portrait | Face, shoulders, clean background |
| **Fashion** | FSH | Style/clothing focused | Outfit, pose, styling, editorial |
| **Beauty** | BTY | Makeup/skincare focused | Skin, makeup, lighting, closeup |
| **Glamour** | GLM | Sensual, elegant | Pose, lighting, mood, styling |
| **Boudoir** | BDR | Intimate, private setting | Mood, fabric, soft light |
| **Fitness** | FIT | Athletic, body-focused | Muscle definition, action, sweat |
| **Maternity** | MTR | Pregnancy portraits | Belly, soft light, emotion |
| **Newborn** | NBN | Baby photography | Soft, delicate, props |
| **Family** | FAM | Group/family portraits | Arrangement, interaction |
| **Couple** | CPL | Two people together | Connection, pose, emotion |
| **Senior** | SNR | Graduation/milestone | Achievement, personality |
| **Corporate** | CRP | Business professional | Clean, confident, branded |

#### Event Photography

| Genre | Code | Description | Key Attributes |
|-------|------|-------------|----------------|
| **Wedding** | WED | Ceremony/celebration | Emotion, dress, venue |
| **Concert** | CON | Live music/performance | Stage lighting, action |
| **Sports** | SPT | Athletic events | Motion, peak action, emotion |
| **Event** | EVT | General gatherings | Candid, venue, crowd |
| **Street** | STR | Urban candid | Spontaneous, context, story |
| **Documentary** | DOC | Real-life storytelling | Authentic, narrative |
| **Photojournalism** | PJR | News events | Action, story, impact |

#### Nature & Environment

| Genre | Code | Description | Key Attributes |
|-------|------|-------------|----------------|
| **Landscape** | LND | Wide natural vistas | Horizon, sky, depth, light |
| **Seascape** | SEA | Ocean/water scenes | Water, waves, horizon |
| **Cityscape** | CTY | Urban skylines | Buildings, lights, atmosphere |
| **Architecture** | ARC | Buildings/structures | Lines, perspective, detail |
| **Interior** | INT | Indoor spaces | Room, furniture, light |
| **Nature** | NAT | Natural world | Plants, elements, organic |
| **Wildlife** | WLD | Animals in habitat | Animal, behavior, environment |
| **Bird** | BRD | Avian subjects | Bird, plumage, action |
| **Pet** | PET | Domestic animals | Animal, expression, personality |
| **Macro** | MAC | Extreme close-up | Texture, detail, magnification |
| **Flower** | FLR | Botanical subjects | Petals, color, arrangement |
| **Underwater** | UNW | Below water surface | Blue tones, marine life |
| **Aerial/Drone** | AER | Bird's eye view | Patterns, scale, perspective |
| **Astro** | AST | Night sky/stars | Stars, milky way, long exposure |
| **Storm** | STM | Weather phenomena | Clouds, lightning, drama |

#### Commercial & Product

| Genre | Code | Description | Key Attributes |
|-------|------|-------------|----------------|
| **Product** | PRD | Commercial items | Object, lighting, clean |
| **Food** | FOD | Culinary subjects | Appetizing, styling, fresh |
| **Beverage** | BEV | Drinks photography | Liquid, condensation, glass |
| **Jewelry** | JWL | Precious items | Sparkle, detail, luxury |
| **Automotive** | AUT | Vehicles | Curves, reflections, power |
| **Real Estate** | RLE | Property listing | Rooms, space, features |
| **Advertising** | ADV | Commercial campaigns | Message, brand, appeal |

#### Artistic & Creative

| Genre | Code | Description | Key Attributes |
|-------|------|-------------|----------------|
| **Fine Art** | ART | Artistic expression | Concept, mood, unique |
| **Abstract** | ABS | Non-representational | Shapes, colors, patterns |
| **Surreal** | SUR | Dreamlike imagery | Unusual, manipulated |
| **Conceptual** | CNC | Idea-driven | Symbolism, meaning |
| **Still Life** | STL | Arranged objects | Composition, light, objects |
| **Black & White** | BNW | Monochrome | Contrast, tones, shadow |
| **Double Exposure** | DBL | Layered images | Blend, ghosting |
| **Long Exposure** | LNG | Extended shutter | Motion blur, light trails |
| **Silhouette** | SIL | Backlit outline | Shape, contrast |
| **Minimalist** | MIN | Simple, clean | Negative space, single element |

#### Lifestyle & Travel

| Genre | Code | Description | Key Attributes |
|-------|------|-------------|----------------|
| **Lifestyle** | LIF | Curated living | Aspirational, authentic feel |
| **Travel** | TRV | Destinations | Culture, landmarks, adventure |
| **Candid** | CND | Unposed moments | Natural, spontaneous |
| **Self-Portrait/Selfie** | SLF | Self-captured | Personal, casual |

---

## Attribute Categories by Scene Type

### Portrait/People (ECU, CU, MCU)

```yaml
subject:
  gender: male, female, non-binary
  age_range: child, teen, young adult, adult, middle-aged, elderly
  ethnicity: descriptive skin tone only

face:
  shape: oval, round, heart, square, oblong, diamond
  angle: frontal, 3/4 left, 3/4 right, profile left, profile right
  tilt: level, tilted left, tilted right, chin up, chin down

eyes:
  color: specific shade (hazel, deep brown, steel blue, etc.)
  shape: almond, round, hooded, upturned, downturned, monolid
  state: open, half-lidded, closed, squinting
  gaze: direct at camera, looking left/right/up/down, distant
  expression: soft, intense, playful, serious, tired
  makeup: none, natural, bold, smokey, colorful (describe colors)
  lashes: natural, mascara, false lashes
  brows: natural, groomed, bold, thin

skin:
  tone: porcelain, ivory, fair, light, medium, olive, tan, brown, deep brown, dark
  undertone: warm, cool, neutral
  texture: smooth, freckled, textured, weathered
  condition: natural, dewy, matte, glowing, oily
  features: beauty marks, scars, dimples, wrinkles

nose:
  shape: straight, curved, button, prominent

lips:
  shape: full, thin, heart, wide, cupid's bow
  color: natural pink, nude, red, berry, coral (if makeup)
  state: closed, slightly parted, smiling, pursed

teeth:
  visibility: not visible, slightly visible, showing

expression:
  overall: happy, sad, serious, thoughtful, surprised, neutral
  intensity: subtle, moderate, dramatic
  authenticity: genuine, posed

hair:
  color: black, dark brown, chestnut, auburn, red, blonde, platinum, gray, white
  highlights: none, subtle, balayage, streaks
  length: bald, buzz, short, medium, shoulder-length, long, very long
  texture: straight, wavy, curly, coily, kinky
  style: loose, ponytail, bun, braids, updo, messy, slicked
  state: clean, wet, windswept, disheveled
  parting: center, side left, side right, none
  bangs: none, full, side-swept, curtain

facial_hair: (if applicable)
  type: none, stubble, beard, mustache, goatee
  length: short, medium, long
  grooming: neat, rugged, styled
```

### Upper Body (MCU, MS)

```yaml
body_upper:
  shoulders: narrow, average, broad, sloped
  bust:
    size: small, medium, large, very large
    shape: natural, rounded, perky
    cleavage: none, subtle, moderate, prominent, deep
  torso:
    type: slim, toned, athletic, soft, muscular
    waist: narrow, defined, average, wide
    midriff: covered, partially visible, fully exposed
  back:
    visible: yes/no
    exposure: none, upper back, full back, backless

clothing_upper:
  type: shirt, blouse, t-shirt, sweater, jacket, coat, dress, tank top, crop top,
        bralette, bikini top, lingerie, corset, bodysuit, swimwear
  neckline:
    style: crew, v-neck, deep-v, plunging, scoop, sweetheart, off-shoulder,
           halter, strapless, turtleneck, collared, keyhole, cowl
    depth: high, moderate, low, very low
  sleeves: sleeveless, spaghetti strap, cap, short, 3/4, long, rolled up
  coverage:
    chest: full, moderate, minimal, sheer
    shoulders: covered, one exposed, both exposed
    midriff: covered, peek, cropped, exposed
    back: covered, low-back, backless, cutout
  color: specific colors and patterns
  pattern: solid, striped, floral, plaid, abstract, printed, sheer
  material: cotton, silk, satin, lace, mesh, sheer, leather, latex, knit, denim
  fit: loose, relaxed, fitted, tight, form-fitting, oversized
  condition: pristine, casual, rumpled, wet, see-through when wet
  cutouts: none, side, front, back, multiple
  straps: thick, thin, spaghetti, halter, criss-cross, none

accessories:
  necklace: none, choker, pendant, chain, statement, body chain
  earrings: none, studs, hoops, dangles, statement
  glasses: none, prescription, sunglasses (describe style)
  watch: none, casual, luxury, smart
  hat: none, baseball cap, beanie, fedora, sun hat
  scarf: none, describe style and drape
  body_jewelry: none, belly piercing, body chain, anklet

pose_upper:
  shoulders: squared, angled, one raised, relaxed, tense
  arms: at sides, crossed, raised, one on hip, behind head, covering chest
  hands: visible/hidden, gesturing, touching face, in hair, on body
  head_position: straight, tilted, turned, thrown back
  body_angle: frontal, 3/4, profile, back view
  chest_position: straight, turned, arched, leaning forward
```

### Full Body (MLS, FS, LS)

```yaml
body:
  build: slim, petite, athletic, toned, average, curvy, hourglass, plus-size, muscular
  height: appears short, average, tall
  posture: upright, relaxed, slouched, dynamic, arched
  proportions:
    bust: small, medium, large, very large
    waist: narrow, defined, average
    hips: narrow, average, wide, curvy
    legs: slim, toned, athletic, curvy, long

body_exposure:
  chest: fully covered, cleavage visible, sideboob, underboob
  stomach: covered, midriff visible, navel visible, fully exposed
  back: covered, lower back visible, full back exposed
  hips: covered, hip bones visible, side hip exposed
  legs:
    thigh: covered, partially visible, fully visible, inner thigh visible
    coverage_level: full, knee, mid-thigh, high-thigh, minimal
  buttocks: fully covered, outlined, partially visible, cheeks visible

clothing_full:
  outfit_style: casual, formal, business, athletic, beachwear, evening,
                lingerie, swimwear, bodycon, revealing
  top: [reference clothing_upper]
  bottom:
    type: pants, jeans, skirt, shorts, mini skirt, micro skirt,
          leggings, bikini bottom, thong, boyshorts, g-string
    fit: skinny, straight, wide-leg, A-line, bodycon, high-cut
    length: full, ankle, calf, knee, mid-thigh, mini, micro
    rise: high-waist, mid-rise, low-rise, very low
    slit: none, side slit, front slit, high slit, double slit
    color: specific
  dress_specifics: (if applicable)
    length: floor, midi, knee, above-knee, mini, micro
    neckline: [reference clothing_upper neckline]
    back: covered, low-back, backless, cutout
    slit: none, side, front, high slit to hip
  swimwear: (if applicable)
    top_style: triangle, bandeau, halter, underwire, string, micro
    bottom_style: brief, bikini, high-cut, brazilian, thong, g-string, micro
    coverage: full, moderate, minimal, micro
    tie_details: side-tie, back-tie, front-tie
  lingerie: (if applicable)
    type: bra, bralette, teddy, bodysuit, babydoll, corset
    style: push-up, balconette, demi, full coverage, sheer
    matching: yes/no, set description
  footwear:
    type: barefoot, sneakers, heels, stilettos, platform, boots, sandals, flats
    color: specific
    heel_height: flat, low, medium, high, very high
    style: strappy, platform, pointed, open-toe, thigh-high

pose_full:
  stance: standing, sitting, lying, kneeling, crouching, bending, walking, running
  weight: centered, shifted left, shifted right, contrapposto
  legs: together, apart, crossed, one bent, spread, wrapped
  feet: together, apart, one forward, pointed, flexed
  hips: straight, cocked to side, tilted, thrust forward/back
  back: straight, arched, curved, twisted
  dynamic: static, in motion, jumping, leaning, stretching, lounging
  sensuality: none, subtle, moderate, pronounced

interaction:
  with_props: holding, sitting on, leaning against, straddling, embracing
  with_environment: integrated, isolated, interacting
  with_clothing: natural, adjusting, pulling, lifting, sliding off
  with_body: hands on hips, touching hair, touching face, arms wrapped
```

### Environment/Background

```yaml
setting:
  type: studio, indoor, outdoor, mixed
  location: bedroom, living room, office, cafe, street, beach, forest, mountain, urban, rural

background:
  clarity: sharp, soft, very blurred (bokeh)
  complexity: simple, moderate, complex
  color: neutral, colorful, dark, light
  elements: wall, window, nature, architecture, crowd, none

lighting:
  type: natural, artificial, mixed
  source: sun, window, studio lights, practical lights, neon
  direction: front, side, back, top, bottom, rim
  quality: hard, soft, diffused, dramatic
  color_temperature: warm, neutral, cool, mixed
  time_of_day: morning, midday, golden hour, blue hour, night
  shadows: minimal, soft, dramatic, hard

atmosphere:
  mood: bright, moody, dreamy, dramatic, intimate, energetic
  weather: clear, cloudy, foggy, rainy, snowy (if outdoor)
  depth: shallow, moderate, deep
```

### Objects/Products

```yaml
object:
  type: specific item category
  shape: geometric, organic, complex
  size: small, medium, large

material:
  type: metal, glass, plastic, fabric, wood, stone, ceramic
  finish: matte, glossy, textured, brushed
  transparency: opaque, translucent, transparent

surface:
  texture: smooth, rough, patterned
  reflectivity: none, subtle, mirror-like

color:
  primary: main color
  secondary: accent colors
  pattern: solid, gradient, patterned

arrangement:
  composition: centered, rule of thirds, asymmetric
  props: supporting elements
  surface: table, fabric, transparent, floating
```

---

## Agentic Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: CLASSIFICATION                      │
├─────────────────────────────────────────────────────────────────┤
│  LLM Call #1: Analyze and classify                              │
│  Outputs:                                                       │
│    - shot_framing: ECU/CU/MCU/MS/MLS/FS/LS/ELS                  │
│    - genre: PRT/FSH/LND/PRD/etc                                 │
│    - secondary_tags: selfie, candid, editorial, etc             │
│    - subject_count: 0, 1, 2, group                              │
│    - confidence: 0.0-1.0                                        │
│  Tokens: ~50-100                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: METADATA                            │
├─────────────────────────────────────────────────────────────────┤
│  Direct extraction (no LLM):                                    │
│    - dimensions: width × height                                 │
│    - aspect_ratio: calculated                                   │
│    - orientation: portrait/landscape/square                     │
│    - color_space: RGB/RGBA                                      │
│  Tokens: 0                                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 3: ATTRIBUTE MAPPING                   │
├─────────────────────────────────────────────────────────────────┤
│  Rule-based selection based on classification:                  │
│                                                                 │
│  ECU/BCU/CU + PRT → face, eyes, skin, lips, expression, hair   │
│  MCU/MS + FSH → face, hair, clothing_upper, accessories, pose  │
│  FS/LS + FSH → body, clothing_full, pose_full, environment     │
│  LND/SEA/CTY → scene, composition, atmosphere, lighting         │
│  PRD/FOD → object, material, surface, arrangement               │
│  Tokens: 0 (deterministic)                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 4: DETAILED ANALYSIS                   │
├─────────────────────────────────────────────────────────────────┤
│  LLM Call #2: Structured attribute extraction                   │
│  Input: Image + attribute schema for this scene type            │
│  Output: JSON with specific values for each attribute           │
│  Tokens: ~200-400                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 5: PROMPT COMPOSITION                  │
├─────────────────────────────────────────────────────────────────┤
│  Template-based narrative generation                            │
│  Order: Shot → Subject → Key Details → Environment → Lighting   │
│  Tokens: output ~100-300                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 6: Z-IMAGE OPTIMIZATION                │
├─────────────────────────────────────────────────────────────────┤
│  Generate recommendations:                                      │
│    - optimal_resolution for Z-Image                             │
│    - resize_needed: true/false                                  │
│    - resize_method: lanczos/bicubic                             │
│    - aspect_ratio_suggestion                                    │
│    - quality_estimate: low/medium/high                          │
│    - suggested_steps: 8-9                                       │
│    - guidance_scale: 0.0                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  SID Z-Image Prompt Generator                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ○ image ─────────────────────────────────────────────── [●]    │
│                                                                 │
│  ═══════════════════ API SETTINGS ════════════════════════     │
│                                                                 │
│  API Key        [••••••••••••••••••••••••••••••••]              │
│  Model          [claude-sonnet-4-5-20250929      ▼]            │
│                                                                 │
│  ═══════════════════ ANALYSIS OPTIONS ════════════════════     │
│                                                                 │
│  Detail Level   [Standard                        ▼]            │
│                  ├─ Quick (1 LLM call, ~100 tokens)             │
│                  ├─ Standard (2 calls, ~300 tokens) ✓           │
│                  └─ Deep (3 calls, ~500 tokens)                 │
│                                                                 │
│  Focus Override [Auto-detect                     ▼]            │
│                  ├─ Auto-detect ✓                               │
│                  ├─ Portrait/People                             │
│                  ├─ Full Body/Fashion                           │
│                  ├─ Landscape/Environment                       │
│                  ├─ Product/Object                              │
│                  ├─ Food/Beverage                               │
│                  └─ Architecture/Interior                       │
│                                                                 │
│  Content Detail [standard                        ▼]            │
│                  ├─ minimal (basic description)                 │
│                  ├─ standard (normal detail) ✓                  │
│                  ├─ detailed (body proportions, clothing cuts)  │
│                  └─ explicit (full NSFW detail for Z-Image)     │
│                                                                 │
│  ═══════════════════ PROMPT DIRECTION ════════════════════     │
│                                                                 │
│  User Prompt    ┌─────────────────────────────────────────┐    │
│  (optional)     │ Guide the output: "focus on the dress"  │    │
│                 │ or provide full prompt to enhance...    │    │
│                 └─────────────────────────────────────────┘    │
│                                                                 │
│  Prompt Mode    [enhance                        ▼]             │
│                  ├─ analyze (image only, ignore prompt)         │
│                  ├─ enhance (prompt guides analysis) ✓          │
│                  ├─ priority (prompt first, image fills gaps)   │
│                  └─ override (use prompt, minimal image ref)    │
│                                                                 │
│  Focus Areas    [☑] Subject details                             │
│                 [☑] Environment/background                      │
│                 [☑] Lighting description                        │
│                 [☑] Colors and materials                        │
│                 [☐] Mood/atmosphere                             │
│                 [☑] Quote text elements ("text")                │
│                                                                 │
│  ═══════════════════ OUTPUT SETTINGS ═════════════════════     │
│                                                                 │
│  Max Tokens     [═══════════════●══════] 300                   │
│                  50                    500                     │
│                                                                 │
│  ═══════════════════ GENERATION ══════════════════════════     │
│                                                                 │
│  Temperature    [═════════●═══════════════] 0.7                │
│                                                                 │
│  Seed           [0                              ] [🎲]          │
│  Seed Mode      [fixed                          ▼]             │
│                  ├─ fixed (deterministic output) ✓              │
│                  ├─ randomize (new seed each run)               │
│                  ├─ increment (+1 each run)                     │
│                  └─ decrement (-1 each run)                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  OUTPUTS                                                        │
│                                                                 │
│  prompt ○─────────────────────── Z-Image ready narrative        │
│  structured_data ○───────────── JSON with all attributes        │
│  metadata ○──────────────────── Image info + recommendations    │
│  debug_log ○─────────────────── Processing details              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input Parameters

| # | Parameter | Type | Default | Description |
|---|-----------|------|---------|-------------|
| 1 | `image` | IMAGE | required | Input image to analyze |
| 2 | `api_key` | STRING | "" | Anthropic API key |
| 3 | `model` | COMBO | claude-sonnet-4-5 | Claude model |
| 4 | `detail_level` | COMBO | "Standard" | Quick/Standard/Deep |
| 5 | `focus_override` | COMBO | "Auto-detect" | Force genre focus |
| 6 | `content_detail` | COMBO | "standard" | Body/clothing detail level |
| 7 | `user_prompt` | STRING | "" | Guide or override the analysis |
| 8 | `prompt_mode` | COMBO | "enhance" | How to use user_prompt |
| 9 | `focus_subject` | BOOL | True | Include subject details |
| 10 | `focus_environment` | BOOL | True | Include background/environment |
| 11 | `focus_lighting` | BOOL | True | Include lighting description |
| 12 | `focus_colors` | BOOL | True | Include colors and materials |
| 13 | `focus_mood` | BOOL | False | Include mood/atmosphere |
| 14 | `include_text_quotes` | BOOL | True | Quote visible text |
| 15 | `max_tokens` | INT | 300 | Target output tokens |
| 16 | `temperature` | FLOAT | 0.7 | Creativity (0-1) |
| 17 | `seed` | INT | 0 | Random seed |
| 18 | `seed_mode` | COMBO | "fixed" | Seed behavior control |

---

## Content Detail Levels

Controls how much body and clothing detail is captured.

| Level | Body Detail | Clothing Detail | Use Case |
|-------|-------------|-----------------|----------|
| **minimal** | Basic build only | Type and color | SFW, quick |
| **standard** | Build, posture | Type, fit, color, material | General use |
| **detailed** | Proportions, exposure areas | Cuts, coverage, reveals | Fashion, glamour |
| **explicit** | Full NSFW attributes | Full exposure mapping | Adult content (Z-Image) |

### Minimal Level

```yaml
body: "slim build"
clothing: "black dress"
```

### Standard Level

```yaml
body: "slim athletic build, upright posture"
clothing: "fitted black mini dress, off-shoulder neckline, satin material"
```

### Detailed Level

```yaml
body:
  build: "slim athletic, hourglass proportions"
  bust: "medium"
  waist: "narrow, defined"
  exposure: "midriff visible, cleavage subtle"
clothing:
  type: "bodycon mini dress"
  neckline: "sweetheart, low"
  back: "low-back"
  slit: "high slit on left"
  material: "black satin"
  coverage: "minimal shoulders, exposed back"
```

### Explicit Level (NSFW)

```yaml
body:
  build: "slim athletic, hourglass figure"
  proportions:
    bust: "large, rounded"
    waist: "narrow"
    hips: "curvy"
  exposure:
    chest: "deep cleavage, sideboob visible"
    stomach: "navel exposed, toned midriff"
    back: "fully exposed"
    buttocks: "partially visible through slit"
clothing:
  type: "micro bikini"
  top:
    style: "string triangle"
    coverage: "minimal"
  bottom:
    style: "brazilian cut, high-cut legs"
    coverage: "minimal"
    tie: "side-tie"
pose:
  sensuality: "pronounced"
  back: "arched"
  hips: "thrust to side"
```

---

## Seed Mode Behavior

| Mode | Behavior | Use Case |
|------|----------|----------|
| **fixed** | Same input + same seed = identical output | Reproducibility, caching |
| **randomize** | New random seed each execution | Variation, exploration |
| **increment** | Seed increases by 1 each run | Batch variations |
| **decrement** | Seed decreases by 1 each run | Reverse exploration |

### Fixed Mode (Deterministic)

When `seed_mode = "fixed"`:
- Creates a hash from: `image_hash + seed + all_parameters`
- If hash matches previous run, returns cached output
- Guarantees identical prompt for identical inputs
- Useful for reproducible workflows

```python
# Deterministic caching logic
def get_cache_key(image, seed, **params):
    image_hash = hash_image_tensor(image)
    param_hash = hash(frozenset(params.items()))
    return f"{image_hash}_{seed}_{param_hash}"

cache_key = get_cache_key(image, seed,
    model=model,
    detail_level=detail_level,
    focus_override=focus_override,
    include_lighting=include_lighting,
    include_text_quotes=include_text_quotes,
    max_tokens=max_tokens,
    user_context=user_context,
    temperature=temperature
)

if cache_key in OUTPUT_CACHE:
    return OUTPUT_CACHE[cache_key]  # Identical output
else:
    result = run_llm_analysis(...)
    OUTPUT_CACHE[cache_key] = result
    return result
```

### Randomize Mode

When `seed_mode = "randomize"`:
- Generates new random seed (0 to 2^31-1) before each execution
- Always makes fresh LLM API call (no caching)
- Each execution produces different prompt variations
- Seed widget updates to show the used seed

### Increment/Decrement Mode

When `seed_mode = "increment"` or `"decrement"`:
- Modifies seed by ±1 after successful execution
- Updates the seed widget value in UI
- Useful for generating controlled sequential variations
- Wraps around at boundaries (0 ↔ 2^31-1)

---

## Prompt Mode Behavior

Controls how `user_prompt` interacts with image analysis.

| Mode | Priority | Behavior |
|------|----------|----------|
| **analyze** | Image 100% | Ignores user_prompt entirely, pure image analysis |
| **enhance** | Image 80%, Prompt 20% | Prompt guides/influences the analysis |
| **priority** | Prompt 70%, Image 30% | Prompt is primary, image fills gaps |
| **override** | Prompt 90%, Image 10% | User prompt dominates, minimal image reference |

### Analyze Mode (Default when no prompt)

```
Input: Image only
Output: Pure visual description from image analysis
Use: Standard image-to-prompt generation
```

### Enhance Mode (Default when prompt provided)

```
Input: Image + "focus on the vintage clothing"
Output: Full image analysis with extra detail on clothing
Use: Guide attention to specific elements
```

The user prompt acts as a directive:
- "focus on the eyes" → More detail in eyes section
- "describe in moody tone" → Atmosphere emphasis
- "emphasize the lighting" → Extended lighting description

### Priority Mode

```
Input: Image + "woman in red dress at sunset beach"
Output: Uses prompt as primary structure, image validates/enhances
Use: When you have a concept and want image to fill details
```

Processing:
1. Parse user prompt for key elements
2. Match/validate against image
3. Fill gaps with image analysis
4. Prompt elements take precedence if conflict

### Override Mode

```
Input: Image + "A cyberpunk cityscape with neon lights..."
Output: Primarily the user prompt, image adds minor details
Use: When you want to transform/reimagine the image
```

Processing:
1. User prompt is the foundation (90%)
2. Image provides only: colors, composition hints, basic structure
3. Minimal image analysis, maximum prompt preservation

### Priority Examples

**Same image, different modes:**

| Mode | User Prompt | Output Focus |
|------|-------------|--------------|
| analyze | (ignored) | "Woman in pink swimwear at tropical beach..." |
| enhance | "tropical paradise" | "Woman at idyllic tropical paradise with palm trees..." |
| priority | "fashion model photoshoot" | "Professional fashion model in designer swimwear..." |
| override | "mermaid emerging from sea" | "Mermaid with flowing hair emerging from turquoise sea..." |

---

## Focus Area Toggles

Control which sections appear in the output prompt.

| Toggle | When ON | When OFF |
|--------|---------|----------|
| `focus_subject` | Detailed subject description | Brief mention only |
| `focus_environment` | Full background/setting | Minimal or "blurred background" |
| `focus_lighting` | Light source, direction, quality | Skip lighting details |
| `focus_colors` | Material colors and textures | Skip color specifics |
| `focus_mood` | Atmosphere, emotion, feeling | No mood descriptors |
| `include_text_quotes` | Text in `"quotes"` | Plain text mention |

### Token Budget Allocation

When all toggles ON (300 tokens):
```
Subject:      ~100 tokens (33%)
Environment:   ~60 tokens (20%)
Lighting:      ~50 tokens (17%)
Colors:        ~50 tokens (17%)
Mood:          ~40 tokens (13%)
```

When only Subject + Lighting ON:
```
Subject:      ~180 tokens (60%)
Lighting:     ~120 tokens (40%)
```

---

## Output Parameters

| Output | Type | Description |
|--------|------|-------------|
| `prompt` | STRING | Z-Image narrative prompt |
| `structured_data` | STRING | JSON with categorized attributes |
| `metadata` | STRING | JSON with image info + Z-Image recommendations |
| `debug_log` | STRING | Stage-by-stage processing log |

---

## Output Examples

### prompt (STRING)
```
A close-up portrait of a young adult woman with warm olive skin
and long wavy chestnut brown hair flowing over her right shoulder.
Her dark brown almond-shaped eyes feature bold teal-green eyeshadow
blended across the lids, with defined natural brows. Natural pink
lips slightly parted. She wears a white zip-front jacket. Soft
diffused natural daylight from the front-left creates subtle
warmth on her face. Shallow depth of field with blurred car
interior in background. Intimate selfie composition.
```

### structured_data (JSON)
```json
{
  "classification": {
    "shot_framing": "CU",
    "shot_label": "Close-Up",
    "genre": "PRT",
    "genre_label": "Portrait",
    "secondary": ["selfie", "beauty"],
    "subject_count": 1,
    "confidence": 0.94
  },
  "attributes": {
    "subject": {
      "gender": "female",
      "age_range": "young adult",
      "ethnicity": "South Asian"
    },
    "face": {
      "shape": "oval",
      "angle": "frontal"
    },
    "eyes": {
      "color": "dark brown",
      "shape": "almond",
      "gaze": "direct at camera",
      "makeup": "bold teal-green eyeshadow",
      "brows": "natural, defined"
    },
    "skin": {
      "tone": "warm olive",
      "texture": "smooth",
      "condition": "natural glow"
    },
    "lips": {
      "color": "natural pink",
      "state": "slightly parted"
    },
    "hair": {
      "color": "chestnut brown",
      "length": "long",
      "texture": "wavy",
      "style": "loose, flowing right"
    },
    "clothing_visible": {
      "type": "zip jacket",
      "color": "white",
      "neckline": "v-neck with zipper"
    },
    "lighting": {
      "type": "natural",
      "direction": "front-left",
      "quality": "soft diffused"
    },
    "background": {
      "type": "car interior",
      "clarity": "very blurred"
    }
  },
  "prompt_stats": {
    "word_count": 89,
    "estimated_tokens": 118
  }
}
```

### metadata (JSON)
```json
{
  "image_info": {
    "width": 1080,
    "height": 1350,
    "aspect_ratio": "4:5",
    "aspect_decimal": 0.8,
    "orientation": "portrait",
    "megapixels": 1.46
  },
  "z_image_recommendations": {
    "optimal_resolution": [1024, 1280],
    "current_compatible": false,
    "resize_needed": true,
    "resize_direction": "downscale",
    "resize_method": "lanczos",
    "aspect_ratio_ok": true,
    "nearest_native": "1024x1280 (4:5)",
    "quality_estimate": "high",
    "suggested_settings": {
      "steps": 9,
      "guidance_scale": 0.0,
      "sampler": "euler"
    }
  },
  "content_flags": {
    "has_text": false,
    "has_multiple_subjects": false,
    "complexity": "medium"
  }
}
```

### debug_log (STRING)
```
═══════════════════════════════════════════════════════════════
SID Z-Image Prompt Generator - Debug Log
═══════════════════════════════════════════════════════════════
Timestamp: 2025-12-08 14:32:15
Mode: Standard (2 LLM calls)

[STAGE 1] Classification (LLM Call #1)
  Shot Framing: CU (Close-Up) - 94% confidence
  Genre: PRT (Portrait)
  Secondary: selfie, beauty
  Subject Count: 1
  Tokens used: 67
  Duration: 0.9s

[STAGE 2] Metadata Extraction
  Dimensions: 1080 × 1350 px
  Aspect Ratio: 4:5 (0.80)
  Orientation: Portrait

[STAGE 3] Attribute Mapping
  Scene: CU + PRT
  Selected categories: subject, face, eyes, skin, lips, hair,
                       clothing_visible, lighting, background

[STAGE 4] Detailed Analysis (LLM Call #2)
  Attributes extracted: 9 categories, 28 properties
  Tokens used: 289
  Duration: 2.3s

[STAGE 5] Prompt Composition
  Template: portrait_closeup
  Word count: 89
  Estimated tokens: 118

[STAGE 6] Z-Image Recommendations
  Resize: 1080×1350 → 1024×1280 (lanczos downscale)
  Quality estimate: HIGH

═══════════════════════════════════════════════════════════════
Total LLM tokens: 356
Total duration: 3.4s
═══════════════════════════════════════════════════════════════
```

---

## Sources

- [Adobe: 28 Types of Photography](https://www.adobe.com/creativecloud/photography/type.html)
- [Pixpa: 33 Types of Photography](https://www.pixpa.com/blog/types-of-photography)
- [Shutterstock: 30 Types of Photography](https://www.shutterstock.com/blog/types-of-photography)
- [StudioBinder: Camera Shot Types](https://www.studiobinder.com/blog/ultimate-guide-to-camera-shots/)
- [MasterClass: 15 Types of Photography](https://www.masterclass.com/articles/15-different-types-of-photography-explained)
- [Photography Life: 26 Types of Photography](https://photographylife.com/types-of-photography)

---

*Updated: December 2025*
*Branch: dev/z-image*
