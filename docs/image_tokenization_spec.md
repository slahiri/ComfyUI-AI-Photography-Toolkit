# Image Tokenization & Classification Specification

## Overview

This document defines the approach for extracting, normalizing, classifying, and validating image metadata tokens for prompt composition.

**Version**: 1.1
**Status**: Design Specification
**Date**: December 2024
**Last Updated**: Added Phase 7 (Token Grounding Visualization)

---

## Related Documents

| Document | Description |
|----------|-------------|
| [Implementation Plan](./implementation_plan.md) | Step-by-step implementation roadmap with tasks and priorities |
| [CLAUDE.md](../CLAUDE.md) | Project overview and architecture |

---

## Context Recovery

**If resuming after context loss:**
1. Read this spec for technical details
2. Read [implementation_plan.md](./implementation_plan.md) for current progress and next steps
3. Check task checkboxes in implementation plan to see what's done

---

## Table of Contents

1. [Phase 1: Tokenization](#phase-1-tokenization)
2. [Phase 2: Normalization](#phase-2-normalization)
3. [Phase 3: Classification](#phase-3-classification)
4. [Phase 4: Assembly](#phase-4-assembly)
5. [Phase 5: Assembly (Rule-Based)](#phase-5-assembly-rule-based-text-generation)
6. [Phase 6: LLM-Based Assembly](#phase-6-llm-based-assembly-alternative)
7. [Phase 7: Token Grounding Visualization](#phase-7-token-grounding-visualization-optional)
8. [Validation & Quality Analysis](#validation--quality-analysis)

---

## Phase 1: Tokenization

### 1.1 Token Structure

```
ImageToken {
    text: string              # The actual content
    confidence: float         # 0.0-1.0 confidence score
    source: string            # wd14, pixai, joytag, florence_caption, etc.
    token_type: enum          # TAG | PHRASE | KEY_VALUE | SENTENCE
    metadata: object          # additional context
}
```

### 1.2 Extraction Strategy by Source Type

#### A. TAGGER OUTPUTS (wd14, pixai, joytag, nudenet)

**Input:**
```json
"wd14": {
    "1girl": 0.999,
    "solo": 0.978,
    "realistic": 0.975
}
```

**Extraction:**
- Direct key-value extraction
- Key → text, Value → confidence
- Token type: TAG

**Output:**
```
ImageToken(text="1girl", confidence=0.999, source="wd14", token_type=TAG)
ImageToken(text="solo", confidence=0.978, source="wd14", token_type=TAG)
```

---

#### B. ANALYZER OUTPUTS (photography, iqa, composition, saliency)

**Input:**
```json
"photography": {
    "light gray tones": 0.9,
    "high key": 0.8,
    "portrait orientation": 0.95
}
```

**Extraction:**
- Same as taggers but mark source_type as ANALYZER
- These often have implicit category hints (photography → Lighting/Style)

---

#### C. VLM CAPTIONS (florence_caption, florence_description, florence_mixed_caption)

**Input:**
```
"Photo of a young woman with long, wavy brown hair, wearing a white Calvin Klein
crop top and matching panties. She has a fair complexion and is standing confidently
with her hands on her hips. The background is plain and light grey, and the lighting
is soft and even."
```

**Extraction Strategy:**

**Step 1: Sentence Segmentation**
- Split by `.`, `!`, `?`
- Handle edge cases (abbreviations, decimals)

**Step 2: Clause Extraction**
- Split sentences by `,`, `and`, `with`, `while`
- Preserve meaning units

**Step 3: Phrase Pattern Extraction**
Use regex patterns for common structures:

| Pattern | Example | Extraction |
|---------|---------|------------|
| `a/an [adj]* [noun]` | "a young woman" | subject phrase |
| `with [noun phrase]` | "with long, wavy brown hair" | attribute |
| `wearing [noun phrase]` | "wearing a white crop top" | clothing |
| `[verb]ing [adverb]*` | "standing confidently" | action |
| `The [noun] is [adj]` | "The background is plain" | environment |
| `[noun] is [adj] and [adj]` | "lighting is soft and even" | lighting |

**Step 4: Output Multiple Tokens per Caption**
```
ImageToken(text="young woman", token_type=PHRASE, source="florence_caption")
ImageToken(text="long wavy brown hair", token_type=PHRASE, source="florence_caption")
ImageToken(text="wearing white Calvin Klein crop top", token_type=PHRASE, ...)
ImageToken(text="standing confidently", token_type=PHRASE, ...)
ImageToken(text="background plain light grey", token_type=PHRASE, ...)
ImageToken(text="lighting soft and even", token_type=PHRASE, ...)
```

---

#### D. VLM STRUCTURED OUTPUTS (florence_analyze)

**Input:**
```
"camera_angle: front view, art_style: photo realistic, location: indoor,
background: plain light grey background, clothing: white crop top,
action: standing, facial_expression: neutral, hair_color: dark brown hair"
```

**Extraction:**
- Parse key-value pairs by `:` and `,`/`;`
- Preserve the key as metadata (helps classification)
- Token type: KEY_VALUE

**Output:**
```
ImageToken(text="front view", key="camera_angle", token_type=KEY_VALUE, ...)
ImageToken(text="photo realistic", key="art_style", token_type=KEY_VALUE, ...)
ImageToken(text="indoor", key="location", token_type=KEY_VALUE, ...)
```

---

#### E. MIXED TAG SECTIONS (from florence_mixed_caption_plus)

**Input:**
```
"1girl, solo, long hair, breasts, looking at viewer, large breasts,
brown hair (long hair), black hair, navel, cleavage, brown eyes, standing"
```

**Extraction:**
- Split by `,`
- Strip whitespace
- Handle parenthetical qualifiers: `brown hair (long hair)` → keep as single token or split
- Token type: TAG
- Confidence: 1.0 (VLM generated, no score)

---

### 1.3 Phrase Decomposition Algorithm for Natural Text

```
INPUT: "a woman with a fair complexion, dark brown hair, and a confident expression,
        standing with her arms at her sides, looking directly at the camera"

STEP 1: Identify Main Subject
        → "a woman" [SUBJECT]

STEP 2: Extract "with" clauses
        → "fair complexion" [SUBJECT_DETAIL]
        → "dark brown hair" [SUBJECT_DETAIL]
        → "confident expression" [SUBJECT_DETAIL]
        → "arms at her sides" [ACTION_POSE]

STEP 3: Extract "-ing" verb phrases
        → "standing with her arms at her sides" [ACTION_POSE]
        → "looking directly at the camera" [ACTION_POSE]

STEP 4: Chunk remaining phrases
        → Individual attribute phrases
```

**Linguistic Chunking Approach:**

Use a lightweight NLP chunker (spaCy, Stanza) to identify:
- **NP (Noun Phrases)**: subjects, objects, attributes
- **VP (Verb Phrases)**: actions
- **ADJP (Adjective Phrases)**: qualities
- **PP (Prepositional Phrases)**: locations, relationships

---

## Phase 2: Normalization

### 2.1 Text Normalization Pipeline

```
Raw Token
    │
    ▼
┌─────────────────────────────┐
│  Lowercase                  │
│  "Brown Hair" → "brown hair"│
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Whitespace Normalization   │
│  "long  hair" → "long hair" │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Special Char Handling      │
│  "1girl" → "1girl"          │
│  "calvin klein" → keep as-is│
└─────────────────────────────┘
    │
    ▼
Normalized Token
```

### 2.2 Deduplication Strategy

**Exact Duplicates:**
- Same normalized text → merge, keep highest confidence

**Semantic Duplicates:**
- "1girl" vs "1 girl" vs "one girl" → merge
- "brown hair" vs "hair brown" → merge
- Use embedding similarity > 0.95 threshold

**Hierarchical Duplicates:**
- "long wavy brown hair" contains "brown hair" and "long hair"
- Keep the more specific one OR keep both with relationship flag

**Confidence Aggregation for Duplicates:**
```
merged_confidence = 1 - ∏(1 - conf_i)  # probability union
OR
merged_confidence = max(conf_i)         # simpler
OR
merged_confidence = weighted_average(conf_i, source_weight_i)
```

### 2.3 Source Reliability Weighting

| Source | Weight | Rationale |
|--------|--------|-----------|
| wd14 | 1.0 | Well-calibrated tagger |
| joytag | 0.95 | Good general tagger |
| pixai | 0.9 | Slightly less calibrated |
| florence_caption | 0.85 | Good but verbose |
| florence_analyze | 0.9 | Structured, reliable |
| nudenet | 0.8 | Specific domain |

---

## Phase 3: Classification

### 3.1 Classification Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Deterministic Rules (Structured Keys)                  │
│                                                                  │
│  If token has key from florence_analyze:                         │
│    camera_angle, distance_to_camera → [Composition]              │
│    art_style → [Style/Medium]                                    │
│    location, background → [Environment/Scene]                    │
│    action, facing_direction → [Action/Pose]                      │
│    clothing, pants, shoes, accessory → [Subject Details]         │
│    hair_*, eye_*, facial_*, body, race → [Subject Details]       │
│    gender → [Subject]                                            │
│    text → [Technical Parameters]                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                        Not matched
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Source-Based Routing                                   │
│                                                                  │
│  If source == "photography":                                     │
│    terms with "light", "tone", "contrast" → [Lighting]           │
│    terms with "sharp", "focus", "quality" → [Quality Boosters]   │
│    terms with "orientation", "frame" → [Composition]             │
│                                                                  │
│  If source == "iqa":                                             │
│    All → [Quality Boosters]                                      │
│                                                                  │
│  If source == "composition" or "saliency":                       │
│    All → [Composition]                                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                        Not matched
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: Keyword Dictionary Lookup                              │
│                                                                  │
│  For each category, check:                                       │
│    1. Exact match in category.exact_matches                      │
│    2. Prefix/suffix match in category.patterns                   │
│    3. Regex match in category.regex_patterns                     │
│                                                                  │
│  If match found → assign to category                             │
└─────────────────────────────────────────────────────────────────┘
                               │
                        Not matched
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: Embedding Similarity (Optional)                        │
│                                                                  │
│  Compute embedding of token                                      │
│  Compare to category centroid embeddings                         │
│  Assign to highest similarity category if > threshold            │
└─────────────────────────────────────────────────────────────────┘
                               │
                        Not matched
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5: Uncategorized Bucket                                   │
│                                                                  │
│  Preserve for manual review or assembly-phase handling           │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Keyword Dictionary Structure

```
CategoryDictionary {
    category: string
    exact_matches: Set[string]           # O(1) lookup
    prefix_patterns: List[string]        # "hair_*", "eye_*"
    regex_patterns: List[regex]          # complex patterns
    semantic_anchors: List[string]       # for embedding fallback
    negative_keywords: Set[string]       # explicit exclusions
}
```

**Category Dictionaries:**

#### [Quality Boosters]
```
exact_matches: {
    "realistic", "photorealistic", "photo realistic", "high quality",
    "masterpiece", "best quality", "ultra detailed", "8k", "4k", "hdr",
    "sharp", "in focus", "detailed", "professional", "excellent quality",
    "good aesthetic quality", "excellent technical quality", "sharp and clean",
    "excellent clarity", "photograph", "photography", "photo medium"
}
semantic_anchors: [
    "high quality professional image",
    "sharp detailed realistic photo",
    "masterpiece best quality"
]
```

#### [Subject]
```
exact_matches: {
    "1girl", "2girls", "3girls", "1boy", "2boys", "solo", "multiple girls",
    "multiple boys", "1woman", "1man", "couple", "group", "crowd",
    "female", "male", "person", "people", "model"
}
prefix_patterns: ["*girl*", "*boy*", "*woman*", "*man*"]
semantic_anchors: [
    "person woman man subject",
    "single solo individual",
    "group multiple people"
]
```

#### [Subject Details]
```
subcategories: {
    "hair": {
        exact: {"long hair", "short hair", "brown hair", "blonde hair", ...}
        prefix: ["hair_*", "*_hair"]
    },
    "eyes": {
        exact: {"brown eyes", "blue eyes", "looking at viewer", ...}
        prefix: ["eye_*", "*_eyes"]
    },
    "face": {
        exact: {"lips", "nose", "parted lips", "smile", ...}
        prefix: ["facial_*"]
    },
    "body": {
        exact: {"breasts", "navel", "slim", "curvy", "toned", ...}
        prefix: ["body_*"]
    },
    "clothing": {
        exact: {"underwear", "panties", "bra", "dress", "shirt", ...}
        prefix: ["clothing_*", "wearing_*"]
    },
    "skin": {
        exact: {"fair complexion", "tan", "pale skin", "dark skin", ...}
        prefix: ["skin_*", "race_*"]
    }
}
```

#### [Action/Pose]
```
exact_matches: {
    "standing", "sitting", "walking", "running", "lying", "kneeling",
    "arms at sides", "hands on hips", "cowboy shot", "looking at viewer",
    "looking away", "from behind", "from side", "facing viewer"
}
verb_patterns: ["*ing", "*_shot"]  # standing, cowboy_shot
semantic_anchors: [
    "person standing sitting posing",
    "action movement pose position",
    "looking facing direction"
]
```

#### [Environment/Scene]
```
exact_matches: {
    "indoor", "outdoor", "background", "simple background", "grey background",
    "white background", "gradient background", "plain background", "studio",
    "bedroom", "street", "forest", "beach", "city"
}
prefix_patterns: ["background_*", "location_*"]
semantic_anchors: [
    "background environment setting location",
    "indoor outdoor scene place"
]
```

#### [Lighting]
```
exact_matches: {
    "soft light", "hard light", "natural light", "studio light", "rim light",
    "backlight", "bright", "dark", "shadows", "high key", "low key",
    "soft and even", "dramatic lighting", "neutral tones", "muted colors",
    "light gray tones", "desaturated", "low contrast", "high contrast"
}
semantic_anchors: [
    "lighting illumination bright dark",
    "soft hard natural studio light",
    "shadows contrast tones"
]
```

#### [Style/Medium]
```
exact_matches: {
    "photo", "photograph", "digital art", "illustration", "3d render",
    "anime", "realistic", "oil painting", "watercolor", "sketch",
    "minimalist", "modern", "vintage", "retro", "artistic"
}
prefix_patterns: ["art_style_*", "style_*"]
semantic_anchors: [
    "art style medium technique",
    "photo realistic illustration painting"
]
```

#### [Composition]
```
exact_matches: {
    "front view", "side view", "portrait", "landscape", "close up",
    "full body", "upper body", "cowboy shot", "dutch angle", "birds eye",
    "centered", "rule of thirds", "symmetry", "portrait orientation",
    "vertical frame", "horizontal frame", "2:3 aspect ratio"
}
prefix_patterns: ["camera_angle_*", "distance_*", "composition_*"]
semantic_anchors: [
    "camera angle view shot framing",
    "composition layout arrangement",
    "full body close up portrait"
]
```

#### [Technical Parameters]
```
exact_matches: {
    "width", "height", "resolution", "aspect ratio", "jpeg artifacts",
    "low resolution", "high resolution", "text", "watermark"
}
prefix_patterns: ["image_info_*", "text_*"]
regex_patterns: [r"\d+x\d+", r"\d+:\d+"]  # dimensions, ratios
```

---

### 3.3 Embedding-Based Classification

**Approach: Category Centroid Matching**

#### Step 1: Build Category Embeddings (One-time, Offline)

For each category, create a centroid embedding:

```
For category in categories:
    anchor_texts = category.semantic_anchors + sample(category.exact_matches, 50)
    embeddings = [embed(text) for text in anchor_texts]
    category_centroid = mean(embeddings)
    category_embeddings[category] = normalize(category_centroid)
```

#### Step 2: Runtime Classification

```
token_embedding = normalize(embed(token.text))

similarities = {}
for category, centroid in category_embeddings:
    similarities[category] = cosine_similarity(token_embedding, centroid)

best_category = argmax(similarities)
confidence = similarities[best_category]

if confidence > THRESHOLD:  # 0.5 recommended
    return (best_category, confidence)
else:
    return (UNKNOWN, confidence)
```

#### Recommended Embedding Model

| Model | Speed | Quality | Memory |
|-------|-------|---------|--------|
| `all-MiniLM-L6-v2` | Very Fast | Good | 80MB |
| `all-mpnet-base-v2` | Medium | Better | 420MB |
| `paraphrase-MiniLM-L6-v2` | Very Fast | Good for short | 80MB |

**Recommendation:** `all-MiniLM-L6-v2` for best speed/quality tradeoff

---

### 3.4 Multi-Category Token Handling

Some tokens genuinely belong to multiple categories:

**Example:** `"looking at viewer"`
- [Action/Pose] - it's an action
- [Subject Details] → eyes - eye direction

**Strategy: Multi-Label with Primary Flag**

```
TokenClassification {
    token: ImageToken
    primary_category: string
    secondary_categories: List[string]
    confidence_per_category: Dict[string, float]
}
```

---

### 3.5 Caption → Canonical Category Extraction

#### The Challenge

Caption text interleaves multiple categories in single sentences:

```
INPUT:
"she has long, wavy brown hair and is wearing a white crop top with
the Calvin Klein logo, standing confidently against a plain grey background"

CONTAINS:
├── [Subject Details/hair]: "long, wavy brown hair"
├── [Subject Details/clothing]: "white crop top with Calvin Klein logo"
├── [Action/Pose]: "standing confidently"
└── [Environment]: "plain grey background"
```

#### Strategy: Hierarchical Clause Extraction

```
┌─────────────────────────────────────────────────────────────────┐
│                     FULL CAPTION                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LEVEL 1: SENTENCE SEGMENTATION                      │
│                                                                  │
│  Split by: . ! ?                                                │
│  Handle: abbreviations (Dr., Mr.), decimals (0.5)               │
│  Result: List of complete sentences                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LEVEL 2: CLAUSE SEGMENTATION                        │
│                                                                  │
│  Split by: , | and | with | while | as | who | which            │
│  Preserve: quoted text, parenthetical content                   │
│                                                                  │
│  "she has long hair, wearing a dress, standing confidently"     │
│       ↓                                                          │
│  ["she has long hair", "wearing a dress", "standing confidently"]│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LEVEL 3: PHRASE PATTERN MATCHING                    │
│                                                                  │
│  Each clause matched against category-specific patterns         │
│  Multiple matches per clause allowed                            │
│  Unmatched portions preserved for fallback                      │
└─────────────────────────────────────────────────────────────────┘
```

---

#### Category-Specific Extraction Patterns

##### [Subject] Patterns
```
Trigger Structures:
├── "a/an [adjective]* [ethnicity]? [subject_noun]"
│   Examples: "a young woman", "an attractive asian model"
│
├── "the [subject_noun]"
│   Examples: "the woman", "the model"
│
└── Standalone: "woman", "man", "model", "person", "girl", "boy"

Subject Nouns: woman, man, girl, boy, lady, gentleman, person,
              model, figure, individual, subject

Ethnicity Terms: asian, indian, european, african, latin,
                japanese, korean, chinese, caucasian
```

##### [Subject Details] Patterns

**Hair:**
```
├── "[length]? [texture]? [color] hair"
│   Examples: "long wavy brown hair", "short blonde hair"
│
├── "hair [is/styled/worn] [description]"
│   Examples: "hair is styled in loose waves"
│
├── "[subject] has [hair description]"
│   Examples: "she has dark curly hair"
│
Length: long, short, medium, shoulder-length
Texture: wavy, curly, straight, frizzy
Color: brown, black, blonde, red, white, gray, dark, light, auburn
```

**Eyes:**
```
├── "[color] eyes"
│   Examples: "brown eyes", "deep blue eyes"
│
├── "eyes are [description]"
│   Examples: "eyes are looking directly at camera"
│
Color: brown, blue, green, hazel, dark, light, amber, gray
```

**Clothing:**
```
├── "wearing [clothing description]"
│   Examples: "wearing a white crop top"
│
├── "dressed in [clothing description]"
│   Examples: "dressed in formal attire"
│
├── "in a [clothing item]"
│   Examples: "in a red dress"
│
├── "[clothing item] with [detail]"
│   Examples: "top with Calvin Klein logo"
│
├── "paired with [clothing]"
│   Examples: "paired with matching briefs"

Clothing Terms: dress, top, shirt, blouse, pants, shorts, skirt,
               bra, panties, underwear, bikini, lingerie, suit,
               jacket, coat, saree, lehenga, kimono, etc.
```

**Body:**
```
├── "[body descriptor] physique/figure/build"
│   Examples: "slim toned physique", "athletic build"
│
├── "[subject] has a [body description]"
│   Examples: "she has a curvy figure"

Body Terms: slim, curvy, athletic, toned, petite, tall,
           muscular, fit, slender
```

**Skin/Complexion:**
```
├── "[skin descriptor] skin/complexion"
│   Examples: "fair complexion", "olive skin"

Skin Terms: fair, pale, tan, olive, dark, brown, warm, smooth,
           light, medium, deep
```

##### [Action/Pose] Patterns
```
├── "[verb-ing] [adverb]? [prepositional phrase]?"
│   Examples: "standing confidently", "sitting on a chair"
│
├── "looking [direction/target]"
│   Examples: "looking at the camera", "looking away"
│
├── "[body part] [position]"
│   Examples: "arms at her sides", "hands on hips"
│
├── "[expression type] expression"
│   Examples: "neutral expression", "confident expression"

Action Verbs: standing, sitting, walking, running, lying, kneeling,
             posing, leaning, reclining, crouching

Gaze: looking at camera, looking directly, looking away,
     looking down, looking up, looking to the side

Expression: neutral, confident, serious, playful, sensual,
           relaxed, pensive, happy, smiling
```

##### [Environment/Scene] Patterns
```
├── "against [a/the]? [background description]"
│   Examples: "against a plain grey background"
│
├── "in [a/the]? [location]"
│   Examples: "in a studio", "in an outdoor setting"
│
├── "[location type] background/setting/scene"
│   Examples: "studio background", "outdoor scene"
│
├── "background is [description]"
│   Examples: "background is plain and light"

Location Terms: indoor, outdoor, studio, bedroom, street,
               beach, forest, city, garden, office

Background Terms: plain, simple, gradient, white, grey, black,
                 blurred, bokeh, natural
```

##### [Lighting] Patterns
```
├── "[quality] light/lighting"
│   Examples: "soft lighting", "natural light"
│
├── "lighting is [description]"
│   Examples: "lighting is soft and even"
│
├── "[tone] tones"
│   Examples: "warm tones", "cool tones"

Quality: soft, hard, natural, artificial, dramatic, dim, bright,
        diffused, harsh, even, flat

Tone: warm, cool, neutral, muted, vibrant, desaturated

Style: high key, low key, rim light, backlit, side lit,
      front lit, Rembrandt
```

##### [Style/Medium] Patterns
```
├── "[style] photograph/photo/shot/shoot"
│   Examples: "photo-realistic shoot", "fashion photograph"
│
├── "[medium] style"
│   Examples: "minimalist style", "editorial style"

Medium: photograph, photo, portrait, editorial, fashion,
       artistic, cinematic, fine art, commercial

Style: minimalist, modern, vintage, retro, classic,
      professional, high-end, casual
```

##### [Composition] Patterns
```
├── "[angle] camera angle/view/shot"
│   Examples: "front camera angle", "low angle shot"
│
├── "[framing] shot"
│   Examples: "cowboy shot", "full body shot", "close-up"

Angle: front, side, back, low, high, dutch, bird's eye,
      worm's eye, three-quarter

Framing: close-up, medium shot, full body, cowboy shot,
        portrait, wide shot, headshot, bust shot

Rules: centered, rule of thirds, symmetry, diagonal,
      golden ratio
```

---

#### META Patterns (Filter Out)

These patterns indicate meta-commentary that should be **removed, not categorized**:

```
FILTER PATTERNS:
├── "the image/photo/picture [shows/captures/depicts/features/has]"
├── "in the [middle/center/foreground/background] of the [image/frame]"
├── "the overall [mood/tone/atmosphere/aesthetic/look]"
├── "with a focus on"
├── "showcasing [her/his/the]"
├── "which [adds/creates/gives/lends]"
├── "conveying a sense of"
├── "creating [a/an] [sense/feeling/atmosphere] of"
├── "the viewer [can/is/may]"
├── "it is worth noting"
```

**Action**: Remove entire clause if META pattern matches anywhere in clause.

---

#### Multi-Match Handling

A single clause can match multiple categories:

```
INPUT CLAUSE:
"she has long brown hair and brown eyes"

EXTRACTION:
├── Match 1: "[subject] has [hair description]"
│   → [Subject Details/hair]: "long brown hair"
│
└── Match 2: "[color] eyes"
    → [Subject Details/eyes]: "brown eyes"

RESULT: Both extractions kept, clause fully consumed
```

---

#### Unmatched Content Handling

```
CLAUSE: "with the Calvin Klein logo on the front"

Pattern Match: None directly

FALLBACK STRATEGY:
1. Check if clause modifies previous extraction
   → Previous: "white crop top" [Subject Details/clothing]
   → This clause is a modifier → append to clothing
   → Result: "white crop top with the Calvin Klein logo"

2. If no modifier relationship, use keyword classification
   → "logo" suggests clothing/branding
   → Assign to [Subject Details/clothing]

3. If still unmatched, preserve in [Uncategorized]
   → Will be processed in Assembly phase
```

---

#### Full Example Walkthrough

**INPUT:**
```
"A photo-realistic shoot from a front camera angle about a model in a
white Calvin Klein crop top and matching briefs, photographed against
a plain light grey background. a woman standing in the middle of the
image, with a neutral expression, looking directly at the camera. she
has long, wavy brown hair and is wearing a white crop top with the
Calvin Klein logo on the front, paired with matching white briefs."
```

**SENTENCE 1:**
```
"A photo-realistic shoot from a front camera angle about a model in a
white Calvin Klein crop top and matching briefs, photographed against
a plain light grey background."

Clauses:
├── "A photo-realistic shoot from a front camera angle about a model"
│   → [Style/Medium]: "photo-realistic shoot"
│   → [Composition]: "front camera angle"
│   → [Subject]: "model"
│
├── "in a white Calvin Klein crop top"
│   → [Subject Details/clothing]: "white Calvin Klein crop top"
│
├── "and matching briefs"
│   → [Subject Details/clothing]: "matching briefs"
│
└── "photographed against a plain light grey background"
    → [Environment]: "plain light grey background"
```

**SENTENCE 2:**
```
"a woman standing in the middle of the image, with a neutral expression,
looking directly at the camera."

Clauses:
├── "a woman standing in the middle of the image"
│   → [Subject]: "a woman"
│   → [Action/Pose]: "standing"
│   → [META - FILTER]: "in the middle of the image" ← REMOVED
│
├── "with a neutral expression"
│   → [Action/Pose]: "neutral expression"
│
└── "looking directly at the camera"
    → [Action/Pose]: "looking directly at the camera"
```

**SENTENCE 3:**
```
"she has long, wavy brown hair and is wearing a white crop top with
the Calvin Klein logo on the front, paired with matching white briefs."

Clauses:
├── "she has long, wavy brown hair"
│   → [Subject Details/hair]: "long, wavy brown hair"
│
├── "and is wearing a white crop top with the Calvin Klein logo on the front"
│   → [Subject Details/clothing]: "white crop top with Calvin Klein logo"
│
└── "paired with matching white briefs"
    → [Subject Details/clothing]: "matching white briefs"
```

**AGGREGATED OUTPUT:**
```json
{
  "Subject": ["model", "a woman"],

  "Subject Details": {
    "hair": ["long, wavy brown hair"],
    "clothing": [
      "white Calvin Klein crop top",
      "matching briefs",
      "white crop top with Calvin Klein logo",
      "matching white briefs"
    ]
  },

  "Action/Pose": [
    "standing",
    "neutral expression",
    "looking directly at the camera"
  ],

  "Environment": ["plain light grey background"],

  "Style/Medium": ["photo-realistic shoot"],

  "Composition": ["front camera angle"],

  "Filtered (META)": ["in the middle of the image"]
}
```

---

## Phase 4: Assembly

### 4.1 Grouping & Ranking

```
For each category:
    tokens = filter(all_tokens, category == this_category)

    # Remove duplicates within category
    tokens = deduplicate(tokens)

    # Rank by confidence
    tokens = sort(tokens, key=confidence, descending=True)

    result[category] = tokens
```

### 4.2 Output Structure

```
ClassifiedImage {
    source_file: string
    extraction_timestamp: datetime
    total_tokens: int

    categories: {
        "Quality Boosters": [...],
        "Subject": [...],
        "Subject Details": {
            "hair": [...],
            "eyes": [...],
            "body": [...],
            "clothing": [...],
            ...
        },
        "Action/Pose": [...],
        "Environment/Scene": [...],
        "Lighting": [...],
        "Style/Medium": [...],
        "Composition": [...],
        "Technical Parameters": [...]
    },

    uncategorized: [...],
    filtered_meta: [...]
}
```

---

## Performance Optimization

### Caching Strategy

```
Token → Classification Cache (LRU, 10k entries)
    │
    ├── Key: normalized_token_text
    ├── Value: (category, confidence, timestamp)
    │
    └── Benefits:
        - Same tag across images → instant lookup
        - Batch processing efficiency
        - Reduces embedding computation
```

### Batch Processing

```
For multiple images:
    → Collect unique tokens across batch
    → Classify unique tokens (batch embedding)
    → Distribute classifications back to images
```

### Estimated Performance

| Component | Time per Token | Batch Size Impact |
|-----------|---------------|-------------------|
| Deterministic Rules | <0.01ms | None |
| Dictionary Lookup | <0.1ms | None |
| Fuzzy Match | 1-5ms | Minor |
| Embedding + Similarity | 5-20ms | 10x speedup with batching |
| Full Pipeline (cached) | <1ms | N/A |
| Full Pipeline (cold) | 10-50ms | 5-10x speedup |

---

## Accuracy Improvement Strategies

### 1. Active Learning Loop

```
Unclassified/Low-Confidence Tokens
    → Human Review Interface
    → Manual Classification
    → Add to Training Set
    → Retrain Classifier Weekly
    → Update Dictionaries
```

### 2. Confidence Calibration

- Track prediction accuracy per confidence bucket
- Adjust thresholds based on real-world performance
- Different thresholds per category (some are easier)

### 3. Category-Specific Models

For high-importance categories, train specialized classifiers:
- Subject Details → clothing classifier
- Lighting → tone/lighting classifier
- These can be more accurate than generic approach

### 4. Ensemble Voting

```
classification_1 = dictionary_lookup(token)
classification_2 = embedding_similarity(token)
classification_3 = trained_classifier(token)

final = weighted_vote([
    (classification_1, 0.3),
    (classification_2, 0.35),
    (classification_3, 0.35)
])
```

---

## Implementation Priority

| Phase | Priority | Accuracy | Effort |
|-------|----------|----------|--------|
| 1. Tokenization Pipeline | High | N/A | Medium |
| 2. Deterministic Rules (Layer 1-2) | High | 95%+ | Low |
| 3. Keyword Dictionaries (Layer 3) | High | 85%+ | Medium |
| 4. Embedding Similarity (Layer 4) | Medium | 75%+ | Medium |
| 5. Trained Classifier (Layer 5) | Low | 80%+ | High |
| 6. Caching & Optimization | Medium | N/A | Medium |
| 7. Active Learning Loop | Low | +5-10% | High |

---

# Validation & Quality Analysis

## Overview

This section defines the approach for validating whether the tokenization and classification pipeline is working correctly.

**Goals:**
1. **Completeness**: Did we lose any content?
2. **Correctness**: Is each phrase in the right category?
3. **Cleanliness**: Did we properly filter meta-commentary?

---

## Validation Strategy 1: Content Preservation Check

### Concept

Every meaningful word/phrase from the input caption should appear in exactly one output category (or be explicitly filtered as META).

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT ACCOUNTING                            │
│                                                                  │
│  INPUT CAPTION                                                   │
│  ─────────────────                                               │
│  Word count: 47                                                  │
│  Unique content words: 32 (excluding: a, the, is, and, with)    │
│                                                                  │
│  OUTPUT CATEGORIES                                               │
│  ─────────────────────                                           │
│  Words in [Subject]: 3                                           │
│  Words in [Subject Details]: 12                                  │
│  Words in [Action/Pose]: 6                                       │
│  Words in [Environment]: 4                                       │
│  Words in [Style]: 2                                             │
│  Words in [Composition]: 3                                       │
│  Words in [META/Filtered]: 5                                     │
│                                                                  │
│  TOTAL ACCOUNTED: 35                                             │
│  UNACCOUNTED: 0                                                  │
│                                                                  │
│  COVERAGE: 100% ✓                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Coverage** | (words in categories + filtered) / total content words | > 95% |
| **Loss Rate** | unaccounted words / total content words | < 5% |
| **Filter Rate** | filtered words / total content words | 5-15% typical |

### Output: Unaccounted Words Report

```
UNACCOUNTED WORDS:
├── "Calvin" - appeared in input, not in any output
├── "Klein" - appeared in input, not in any output
└── "logo" - appeared in input, not in any output

DIAGNOSIS: Clothing pattern missed brand names
ACTION: Expand clothing pattern to capture brand + logo terms
```

---

## Validation Strategy 2: Cross-Reference with Tags

### Concept

Tags from wd14/joytag/pixai are independently generated labels. If extraction is correct, extracted phrases should align with these tags.

```
┌─────────────────────────────────────────────────────────────────┐
│                 TAG ↔ EXTRACTION ALIGNMENT                       │
│                                                                  │
│  TAG SOURCE: wd14                                                │
│  ──────────────                                                  │
│  "brown hair" (0.915) ←→ [Subject Details/hair]: "brown hair" ✓ │
│  "long hair" (0.9)    ←→ [Subject Details/hair]: "long...hair" ✓│
│  "standing" (0.581)   ←→ [Action/Pose]: "standing" ✓            │
│  "brown eyes" (0.87)  ←→ [Subject Details/eyes]: ??? ✗          │
│  "sports bra" (0.868) ←→ [Subject Details/clothing]: "crop top" ~│
│                                                                  │
│  ALIGNMENT SCORE: 4/5 = 80%                                      │
│  SEMANTIC MATCH: 4.5/5 = 90% (sports bra ≈ crop top)            │
└─────────────────────────────────────────────────────────────────┘
```

### Alignment Types

| Type | Definition | Example |
|------|------------|---------|
| **Exact Match** | Tag text appears in extraction | "brown hair" ↔ "brown hair" |
| **Semantic Match** | Different words, same meaning | "sports bra" ↔ "crop top" |
| **Partial Match** | Tag is subset of extraction | "hair" ↔ "long wavy brown hair" |
| **Category Match** | Tag and extraction in same category | Both in [Subject Details/hair] |
| **Missing** | Tag has no corresponding extraction | "brown eyes" not extracted |
| **Conflict** | Tag and extraction in different categories | Tag says pose, extraction says clothing |

### Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Tag Coverage** | tags with matching extraction / total tags | > 80% |
| **Category Accuracy** | correct category matches / total matches | > 90% |
| **Conflict Rate** | category conflicts / total matches | < 5% |

---

## Validation Strategy 3: Category Coherence Analysis

### Concept

Content within a category should be semantically related. Content across categories should be semantically distinct.

```
┌─────────────────────────────────────────────────────────────────┐
│              INTRA-CATEGORY COHERENCE                            │
│                                                                  │
│  [Subject Details/hair]:                                         │
│  ├── "long wavy brown hair"                                      │
│  ├── "dark hair"                                                 │
│  └── "hair styled in loose waves"                                │
│                                                                  │
│  Pairwise Similarity (embedding cosine):                         │
│  ├── (1,2): 0.82                                                 │
│  ├── (1,3): 0.78                                                 │
│  └── (2,3): 0.71                                                 │
│                                                                  │
│  MEAN INTRA-SIMILARITY: 0.77 ✓ (good coherence)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              INTER-CATEGORY SEPARATION                           │
│                                                                  │
│  [Subject Details/hair] vs [Environment]:                        │
│  ├── "long wavy brown hair" ↔ "grey background": 0.12           │
│  ├── "dark hair" ↔ "studio setting": 0.08                       │
│                                                                  │
│  MEAN INTER-SIMILARITY: 0.10 ✓ (good separation)                │
└─────────────────────────────────────────────────────────────────┘
```

### Coherence Score

```
Category Coherence = (Mean Intra-Similarity) / (Mean Inter-Similarity)

Good: > 3.0 (within-category 3x more similar than across)
Acceptable: 2.0 - 3.0
Poor: < 2.0 (categories not well separated)
```

### Contamination Detection

```
CONTAMINATION CHECK:
─────────────────────
[Action/Pose] contains:
├── "standing confidently" ✓ (belongs here)
├── "neutral expression" ✓ (belongs here)
└── "white crop top" ✗ (should be in clothing!)

CONTAMINATION: 1 item misplaced
CONTAMINATION RATE: 1/3 = 33% ← Problem!
```

---

## Validation Strategy 4: Round-Trip Reconstruction

### Concept

If we extract correctly and reassemble, the meaning should be preserved.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROUND-TRIP TEST                               │
│                                                                  │
│  ORIGINAL CAPTION:                                               │
│  "A young woman with long brown hair, wearing a white dress,    │
│   standing in a garden with soft natural lighting."              │
│                                                                  │
│  EXTRACTED → REASSEMBLED:                                        │
│  "A young woman. She has long brown hair. She is wearing a      │
│   white dress. She is standing. The scene is set in a garden.   │
│   The lighting features soft natural lighting."                  │
│                                                                  │
│  SEMANTIC SIMILARITY (embedding):                                │
│  Original ↔ Reconstructed: 0.89                                  │
│                                                                  │
│  VERDICT: High preservation ✓                                    │
└─────────────────────────────────────────────────────────────────┘
```

### What Round-Trip Catches

| Issue | Symptom |
|-------|---------|
| Lost content | Similarity drops significantly |
| Wrong categorization | Reconstructed sentence sounds wrong |
| Over-filtering | Important details missing in reconstruction |
| Duplication | Same phrase appears multiple times |

---

## Validation Strategy 5: META Filter Accuracy

### Concept

Verify that filtered content is truly meta-commentary, not descriptive content.

```
┌─────────────────────────────────────────────────────────────────┐
│                META FILTER VALIDATION                            │
│                                                                  │
│  FILTERED PHRASES:                                               │
│  ├── "in the middle of the image" ← Correct (positional meta)   │
│  ├── "the image captures" ← Correct (meta preamble)             │
│  ├── "showcasing her curves" ← Correct (meta commentary)        │
│  └── "beautiful woman" ← INCORRECT! (descriptive content)       │
│                                                                  │
│  FALSE POSITIVE: "beautiful woman" was filtered but is content  │
│  ACTION: Refine filter to not catch adjective+noun phrases      │
└─────────────────────────────────────────────────────────────────┘
```

### Filter Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **True Positive Rate** | Correctly filtered meta / total meta | > 95% |
| **False Positive Rate** | Incorrectly filtered content / total content | < 2% |
| **False Negative Rate** | Missed meta / total meta | < 10% |

### Manual Review Sample

For validation, manually label 50-100 filtered phrases:
- ✓ Correctly filtered (true meta)
- ✗ Incorrectly filtered (was real content)
- ? Ambiguous

---

## Validation Strategy 6: Ground Truth Benchmark

### Concept

Create a manually annotated test set of 20-50 captions with correct categorizations.

```
┌─────────────────────────────────────────────────────────────────┐
│                 GROUND TRUTH COMPARISON                          │
│                                                                  │
│  TEST CAPTION #7:                                                │
│  "A woman in a red dress standing by the ocean at sunset"       │
│                                                                  │
│  GROUND TRUTH (manual):                 SYSTEM OUTPUT:           │
│  ────────────────────                   ─────────────────        │
│  [Subject]: "woman"                     [Subject]: "woman" ✓     │
│  [Clothing]: "red dress"                [Clothing]: "red dress" ✓│
│  [Action]: "standing"                   [Action]: "standing" ✓   │
│  [Environment]: "by the ocean"          [Environment]: "ocean" ~  │
│  [Lighting]: "at sunset"                [Lighting]: "sunset" ✓   │
│                                                                  │
│  PRECISION: 5/5 = 100%                                           │
│  RECALL: 4.5/5 = 90% (partial on environment)                   │
│  F1 SCORE: 0.95                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmark Metrics

| Metric | Per Category | Overall |
|--------|--------------|---------|
| Precision | TP / (TP + FP) | Macro average |
| Recall | TP / (TP + FN) | Macro average |
| F1 Score | 2 * P * R / (P + R) | Macro average |

---

## Combined Validation Report

```
╔═══════════════════════════════════════════════════════════════════╗
║            EXTRACTION QUALITY REPORT                               ║
║            Caption: florence_description                           ║
║            Date: 2024-12-27                                        ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. CONTENT PRESERVATION                                           ║
║     Coverage: 96.2%  ✓                                             ║
║     Lost words: ["Calvin", "Klein"] (brand names)                  ║
║                                                                    ║
║  2. TAG ALIGNMENT                                                  ║
║     Matched: 18/22 tags (81.8%)  ✓                                 ║
║     Category accuracy: 17/18 (94.4%)  ✓                            ║
║     Missing: ["brown eyes", "cowboy shot", "3d", "thigh gap"]     ║
║                                                                    ║
║  3. CATEGORY COHERENCE                                             ║
║     [Subject Details/hair]: 0.81 intra  ✓                          ║
║     [Subject Details/clothing]: 0.73 intra  ✓                      ║
║     [Action/Pose]: 0.68 intra  ~                                   ║
║     [Environment]: 0.85 intra  ✓                                   ║
║     Overall separation ratio: 4.2  ✓                               ║
║                                                                    ║
║  4. META FILTERING                                                 ║
║     Filtered: 3 phrases                                            ║
║     False positives: 0  ✓                                          ║
║     Missed meta: 1 ("with a focus on her curves")                  ║
║                                                                    ║
║  5. ROUND-TRIP SIMILARITY                                          ║
║     Original ↔ Reconstructed: 0.87  ✓                              ║
║                                                                    ║
╠═══════════════════════════════════════════════════════════════════╣
║  OVERALL SCORE: 91/100  GRADE: A-                                  ║
║                                                                    ║
║  ISSUES TO ADDRESS:                                                ║
║  1. Brand name extraction (Calvin Klein)                           ║
║  2. Missing eye color extraction                                   ║
║  3. One meta phrase not filtered                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Validation Module Structure

```
ValidationAnalyzer
├── ContentPreservationChecker
│   ├── tokenize_input()
│   ├── collect_output_tokens()
│   ├── compute_coverage()
│   └── report_unaccounted()
│
├── TagAlignmentChecker
│   ├── load_tags_from_metadata()
│   ├── match_tags_to_extractions()
│   ├── classify_match_type()
│   └── compute_alignment_score()
│
├── CoherenceAnalyzer
│   ├── embed_category_contents()
│   ├── compute_intra_similarity()
│   ├── compute_inter_similarity()
│   └── detect_contamination()
│
├── MetaFilterValidator
│   ├── collect_filtered_phrases()
│   ├── check_against_patterns()
│   └── flag_potential_false_positives()
│
├── RoundTripTester
│   ├── reconstruct_from_categories()
│   ├── compute_semantic_similarity()
│   └── identify_lost_content()
│
└── ReportGenerator
    ├── aggregate_metrics()
    ├── compute_overall_score()
    └── generate_actionable_items()
```

### Running Validation

```
INPUT:
├── Original metadata (with all captions and tags)
├── Extraction output (categorized tokens)

PROCESS:
├── Run all validation strategies
├── Aggregate results
├── Generate report

OUTPUT:
├── Quality score (0-100)
├── Per-category breakdown
├── Specific issues to fix
├── Unaccounted content list
```

---

## Appendix: Canonical Categories Reference

| Category | Subcategories | Description |
|----------|---------------|-------------|
| Quality Boosters | - | Technical quality terms |
| Subject | - | Who/what is the main subject |
| Subject Details | hair, eyes, face, body, clothing, skin | Appearance details |
| Action/Pose | - | What they're doing, expression, gaze |
| Environment/Scene | - | Setting, location, background |
| Lighting | - | Light quality, direction, mood, tones |
| Style/Medium | - | Artistic style, medium |
| Composition | - | Framing, camera angle, shot type |
| Technical Parameters | - | Resolution, aspect ratio, artifacts |

---

# Phase 5: Assembly (Rule-Based Text Generation)

## Overview

This phase converts classified tokens into natural, readable text without using LLMs. The approach uses:
- Token → Natural Phrase Mappings
- Category-specific templates
- Grammar rules (adjective ordering, pronoun resolution)
- Sentence fusion techniques

## Assembly Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  Classified Tokens (per category)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: Token → Phrase Mapping                      │
│                                                                  │
│  "1girl" → "a woman"                                            │
│  "looking at viewer" → "looking directly at the camera"         │
│  "cowboy shot" → "framed from mid-thigh up"                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: Attribute Extraction & Ordering             │
│                                                                  │
│  Hair: [long, wavy, brown] → ordered by [length, texture, color]│
│  Result: "long, wavy brown hair"                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: Template Application                        │
│                                                                  │
│  Template: "with {length}, {texture} {color} hair"              │
│  Result: "with long, wavy brown hair"                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: Sentence Fusion                             │
│                                                                  │
│  Input: ["She has long hair", "brown hair", "wavy hair"]        │
│  Output: "She has long, wavy brown hair"                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: Paragraph Assembly                          │
│                                                                  │
│  Combine sections with appropriate connectors                   │
│  Apply pronoun resolution                                       │
│  Handle empty categories                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Final Readable Text                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Category-Specific Generation Strategies

### 1. [Subject] Generation

**Input Tokens:**
```
["1girl", "solo", "young woman", "mid-twenties"]
```

**Semantic Mapping:**

| Token Pattern | Role | Example |
|---------------|------|---------|
| `1girl`, `1boy`, `1woman` | count + gender | "a woman" |
| `solo` | composition marker | (implies singular) |
| `young`, `mid-twenties` | age modifier | "young", "in her mid-twenties" |
| `2girls`, `multiple` | plurality | "two women" |

**Token → Natural Phrase Mapping:**
```
{
    "1girl": "a woman",
    "1boy": "a man",
    "2girls": "two women",
    "solo": null,  # absorbed into singularity
    "young woman": "a young woman",
    "mid-twenties": "in her mid-twenties",
    "model": "a model",
    "asian": "Asian"
}
```

**Template:**
```
SUBJECT_TEMPLATE = "{article} {age_modifier} {ethnicity} {gender_noun} {age_phrase}"

Where:
- article = "a" | "an" | "" (for plurals)
- age_modifier = "young" | ""
- ethnicity = "Asian" | "" (optional, only if present)
- gender_noun = "woman" | "man" | "women" | "model"
- age_phrase = "in her mid-twenties" | ""
```

**Generation Logic:**
```
def generate_subject(tokens):
    # Priority: most specific description wins
    if "young woman" in tokens:
        base = "a young woman"
    elif "1girl" in tokens or "1woman" in tokens:
        base = "a woman"
    elif "model" in tokens:
        base = "a model"
    else:
        base = "a person"

    # Add age phrase if different from base
    if "mid-twenties" in tokens and "young" not in base:
        base += " in her mid-twenties"

    # Add ethnicity if present
    if "asian" in tokens:
        base = base.replace("a ", "an Asian ")

    return base
```

**Output:**
```
"A young woman in her mid-twenties"
```

---

### 2. [Subject Details] Generation

This is the most complex category with multiple subcategories.

**Input Tokens (organized):**
```
{
    "hair": ["long hair", "wavy hair", "brown hair"],
    "eyes": ["brown eyes", "looking at viewer"],
    "face": ["lips", "parted lips", "neutral expression", "fair complexion"],
    "body": ["slim", "curvy", "navel", "breasts", "medium breasts"],
    "clothing": ["white crop top", "white panties", "Calvin Klein", "underwear"]
}
```

#### A. Hair Description

**Ordering Rules:**
```
HAIR_ORDER = [length, texture, color, style]

length: short, medium, long
texture: straight, wavy, curly
color: brown, blonde, black, red, etc.
style: ponytail, bun, braids, etc.
```

**Template:**
```
"with {length} {texture} {color} hair"
OR
"with {color} hair styled in {style}"
```

**Token Consolidation:**
```
Input: ["long hair", "wavy hair", "brown hair"]

Step 1: Extract attributes
    length = "long"
    texture = "wavy"
    color = "brown"

Step 2: Apply template
    "with long, wavy brown hair"
```

**Grammar Rules:**
- If multiple adjectives: use comma between different types
- Color always comes last before noun
- "long, wavy brown hair" ✓
- "brown long wavy hair" ✗

---

#### B. Eyes Description

**Template:**
```
"{color} eyes {gaze_phrase}"
```

**Gaze Phrase Mapping:**
```
{
    "looking at viewer": "looking directly at the camera",
    "looking away": "looking away",
    "closed eyes": "with eyes closed",
    "looking down": "gazing downward"
}
```

**Output:**
```
"brown eyes looking directly at the camera"
```

---

#### C. Face/Expression Description

**Template:**
```
"with a {expression} expression and {complexion} complexion"
OR
"with {lip_description}"
```

**Generation:**
```
Input: ["neutral expression", "fair complexion", "parted lips"]

Output: "with a neutral expression, fair complexion, and slightly parted lips"
```

---

#### D. Body Description

**Filtering:** Some tokens are implicit/redundant for natural description

**Include vs Exclude:**
```
Include (natural): slim, curvy, toned, athletic, petite
Exclude (tag-like): navel, breasts, thighs, stomach (unless specifically relevant)
```

**Template:**
```
"with a {body_type} figure"
OR
"with a {body_type} build"
```

**Output:**
```
"with a slim, curvy figure"
```

---

#### E. Clothing Description

**Ordering Rules:**
```
CLOTHING_ORDER = [upper_body, lower_body, footwear, accessories]
```

**Grouping Logic:**
```
Input: ["white crop top", "white panties", "Calvin Klein logo"]

Step 1: Identify garments
    upper = "white crop top"
    lower = "white panties"
    brand = "Calvin Klein"

Step 2: Combine logically
    If brand present and matches items:
        "a white Calvin Klein crop top and matching panties"
    Else:
        "a white crop top paired with white panties"
```

**Connector Words:**
```
{
    "matching_set": "and matching",
    "contrast": "paired with",
    "layered": "over",
    "accessory": "with"
}
```

**Output:**
```
"wearing a white Calvin Klein crop top with matching white briefs"
```

---

#### F. Combined Subject Details Generation

**Master Template:**
```
SUBJECT_DETAILS_TEMPLATE = """
{hair_description}, {eye_description}.
She has {face_description} and {body_description}.
She is {clothing_description}.
"""
```

**Full Output:**
```
"with long, wavy brown hair and brown eyes looking directly at the camera.
She has a neutral expression with fair complexion and slightly parted lips,
and a slim, curvy figure. She is wearing a white Calvin Klein crop top
with matching white briefs."
```

---

### 3. [Action/Pose] Generation

**Input Tokens:**
```
["standing", "arms at sides", "facing viewer", "cowboy shot"]
```

**Token → Phrase Mapping:**
```
{
    "standing": "standing",
    "sitting": "sitting",
    "arms at sides": "with her arms at her sides",
    "hands on hips": "with her hands on her hips",
    "facing viewer": "facing the camera",
    "cowboy shot": null,  # composition, not action
    "looking at viewer": "looking at the viewer"
}
```

**Template:**
```
ACTION_TEMPLATE = "{primary_pose} {arm_position}, {facing_direction}"
```

**Ordering:**
```
1. Primary pose (standing, sitting, lying)
2. Body position (arms, legs)
3. Facing direction
4. Gaze direction (if different from facing)
```

**Output:**
```
"standing confidently with her arms at her sides, facing the camera"
```

---

### 4. [Environment/Scene] Generation

**Input Tokens:**
```
["indoor", "plain light grey background", "studio", "simple background"]
```

**Deduplication & Priority:**
```
Priority: specific > general
"plain light grey background" > "simple background"
"studio" + "indoor" → "indoor studio setting"
```

**Templates:**
```
BACKGROUND_TEMPLATES = [
    "against a {background_description}",
    "in {location_type} with a {background_description}",
    "set against a {background_description}"
]
```

**Output:**
```
"against a plain, light grey background in an indoor studio setting"
```

---

### 5. [Lighting] Generation

**Input Tokens:**
```
["soft light", "even lighting", "high key", "neutral tones", "low contrast"]
```

**Semantic Grouping:**
```
light_quality: soft, hard, natural, artificial
light_style: high key, low key, dramatic
tones: neutral, warm, cool, muted
contrast: low, high
```

**Template:**
```
LIGHTING_TEMPLATE = "The lighting is {quality} and {style}, with {tones} and {contrast} contrast."

OR (simpler)

"lit with {quality} {style} lighting"
```

**Natural Combinations:**
```
{
    ("soft light", "even lighting"): "soft, even lighting",
    ("high key", "bright"): "bright, high-key lighting",
    ("natural light", "warm tones"): "warm natural lighting"
}
```

**Output:**
```
"The image features soft, even lighting with neutral tones and low contrast."
```

---

### 6. [Style/Medium] Generation

**Input Tokens:**
```
["photo realistic", "photograph", "minimalist aesthetic", "modern", "professional"]
```

**Template:**
```
STYLE_TEMPLATE = "A {style} {medium} with a {aesthetic} aesthetic."

OR

"This is a {style} {medium}, {aesthetic_description}."
```

**Token Categories:**
```
medium: photograph, digital art, illustration, 3D render
style: realistic, stylized, artistic
aesthetic: minimalist, modern, vintage, professional
```

**Output:**
```
"A photo-realistic photograph with a minimalist, modern aesthetic."
```

---

### 7. [Composition] Generation

**Input Tokens:**
```
["front view", "full body", "centered", "cowboy shot", "portrait orientation", "rule of thirds"]
```

**Template:**
```
COMPOSITION_TEMPLATE = "Shot from a {camera_angle}, showing {framing}, {composition_style}."
```

**Mappings:**
```
camera_angle: front view, side view, three-quarter view, from above, from below
framing: full body, upper body, close-up, medium shot, cowboy shot
composition_style: centered, rule of thirds, symmetrical
orientation: portrait, landscape
```

**Natural Phrasing:**
```
{
    "cowboy shot": "framed from mid-thigh up",
    "full body": "showing the full figure",
    "front view": "from a frontal angle",
    "centered": "with the subject centered",
    "rule of thirds": "following the rule of thirds"
}
```

**Output:**
```
"Captured from a frontal angle, framed from mid-thigh up, with the subject centered in the frame."
```

---

### 8. [Quality Boosters] Generation

**Strategy:** These are typically not verbalized in natural descriptions, but can be used as prefix qualifiers.

**Input Tokens:**
```
["realistic", "sharp", "detailed", "professional quality", "high resolution"]
```

**Template (if needed):**
```
QUALITY_TEMPLATE = "A {quality_list} image..."
```

**Output:**
```
"A sharp, highly detailed, professional-quality image..."
```

**Alternative:** Integrate into Style/Medium
```
"A sharp, realistic photograph with professional quality..."
```

---

### 9. [Technical Parameters] Generation

**Strategy:** Usually omitted from natural descriptions, but can be appended as metadata.

**Input Tokens:**
```
["288x512", "portrait orientation", "2:3 aspect ratio"]
```

**Template (if needed):**
```
TECHNICAL_TEMPLATE = "Image dimensions: {width}x{height} ({orientation}, {aspect_ratio})"
```

**Output (usually separate or omitted):**
```
"(288×512 pixels, portrait orientation, 2:3 aspect ratio)"
```

---

## Complete Assembly: Full Description Generation

### Master Template Structure

```
FULL_DESCRIPTION_TEMPLATE = """
{quality_prefix}{style_medium}

{subject_intro} {subject_details_hair}, {subject_details_eyes}. {subject_details_face}. {subject_details_body}. {clothing_description}.

{action_pose}.

{environment_scene}. {lighting_description}.

{composition_description}.
"""
```

### Assembly Algorithm

```
def generate_full_description(classified_tokens):
    sections = {}

    # Generate each section independently
    sections['quality'] = generate_quality(classified_tokens['Quality Boosters'])
    sections['style'] = generate_style(classified_tokens['Style/Medium'])
    sections['subject'] = generate_subject(classified_tokens['Subject'])
    sections['details'] = generate_subject_details(classified_tokens['Subject Details'])
    sections['action'] = generate_action(classified_tokens['Action/Pose'])
    sections['environment'] = generate_environment(classified_tokens['Environment/Scene'])
    sections['lighting'] = generate_lighting(classified_tokens['Lighting'])
    sections['composition'] = generate_composition(classified_tokens['Composition'])

    # Assemble with appropriate connectors
    paragraphs = []

    # Paragraph 1: Style intro
    p1 = f"{sections['quality']} {sections['style']}"
    paragraphs.append(p1.strip())

    # Paragraph 2: Subject description
    p2 = f"{sections['subject']} {sections['details']}"
    paragraphs.append(p2.strip())

    # Paragraph 3: Action and environment
    p3 = f"{sections['action']}. {sections['environment']}"
    paragraphs.append(p3.strip())

    # Paragraph 4: Technical (lighting, composition)
    p4 = f"{sections['lighting']} {sections['composition']}"
    paragraphs.append(p4.strip())

    # Join paragraphs
    return "\n\n".join(paragraphs)
```

---

## Example: Full Pipeline Execution

### Input (Classified Tokens)

```json
{
    "Quality Boosters": ["realistic", "sharp", "professional quality"],
    "Subject": ["1girl", "solo", "young woman", "mid-twenties"],
    "Subject Details": {
        "hair": ["long hair", "wavy hair", "brown hair"],
        "eyes": ["brown eyes", "looking at viewer"],
        "face": ["neutral expression", "fair complexion", "parted lips"],
        "body": ["slim", "curvy"],
        "clothing": ["white crop top", "white panties", "Calvin Klein"]
    },
    "Action/Pose": ["standing", "arms at sides", "facing viewer"],
    "Environment/Scene": ["indoor", "plain light grey background", "studio"],
    "Lighting": ["soft light", "even lighting", "neutral tones", "low contrast"],
    "Style/Medium": ["photograph", "photo realistic", "minimalist", "modern"],
    "Composition": ["front view", "cowboy shot", "centered", "portrait orientation"]
}
```

### Output (Generated Description)

```
A sharp, professional-quality photograph with a minimalist, modern aesthetic.

A young woman in her mid-twenties with long, wavy brown hair and brown eyes
looking directly at the camera. She has a neutral expression with fair complexion
and slightly parted lips, and a slim, curvy figure. She is wearing a white
Calvin Klein crop top with matching white briefs.

She stands confidently with her arms at her sides, facing the camera. The scene
is set against a plain, light grey background in an indoor studio setting.

The image features soft, even lighting with neutral tones and low contrast.
Captured from a frontal angle, framed from mid-thigh up, with the subject
centered in the frame.
```

---

## Advanced Techniques for Better Output

### 1. Sentence Variation

**Problem:** Same template = repetitive output

**Solution:** Multiple template variants per category

```
SUBJECT_TEMPLATES = [
    "{article} {age} {gender}",
    "{article} {gender} {age_phrase}",
    "{article} {ethnicity} {gender} who appears to be {age_phrase}",
]

# Random or round-robin selection
template = random.choice(SUBJECT_TEMPLATES)
```

### 2. Connector Variation

```
CONNECTORS = {
    "addition": ["and", "with", "featuring", "along with"],
    "contrast": ["but", "while", "yet"],
    "elaboration": ["specifically", "particularly", "notably"]
}

# Vary connector usage
connector = random.choice(CONNECTORS["addition"])
```

### 3. Sentence Fusion

**Before:**
```
"She has long hair. Her hair is brown. Her hair is wavy."
```

**After (fused):**
```
"She has long, wavy brown hair."
```

**Fusion Rules:**
- Same subject + different attributes → combine attributes
- Use comma for adjective lists
- Apply adjective ordering rules

### 4. Pronoun Resolution

**Track subject for pronoun usage:**
```
First mention: "A young woman..."
Subsequent: "She has..." / "Her hair..." / "She is wearing..."
```

**Gender-neutral option:**
```
First mention: "A person..."
Subsequent: "They have..." / "Their hair..."
```

### 5. Confidence-Based Inclusion

```
def should_include(token, threshold=0.6):
    return token.confidence >= threshold

# Only include high-confidence details
details = [t for t in tokens if should_include(t)]
```

### 6. Avoiding Redundancy

**Rule: Don't repeat information**
```
# If "young woman" is used in Subject, don't repeat "young" in details
if "young" in subject_text:
    remove "young" from subject_details modifiers
```

---

## Implementation Components

### Required Libraries (No ML)

| Library | Purpose |
|---------|---------|
| `inflect` | Pluralization, articles (a/an) |
| `pattern` | Grammar, conjugation |
| `nltk` (rules only) | Tokenization, POS tagging (non-ML) |

### Core Data Structures

```
# Token to natural phrase dictionary
PHRASE_MAPPINGS = {
    category: {
        token: natural_phrase
    }
}

# Grammar rules
ADJECTIVE_ORDER = ["opinion", "size", "age", "shape", "color", "origin", "material"]

# Templates per category
TEMPLATES = {
    category: [template1, template2, ...]
}

# Connectors
CONNECTORS = {
    type: [variants]
}
```

---

## Edge Cases & Solutions

### 1. Empty Categories

```
if not environment_tokens:
    # Option A: Skip entirely
    environment_text = ""

    # Option B: Generic fallback
    environment_text = "against a neutral background"

    # Option C: Infer from other categories
    if "photograph" in style_tokens:
        environment_text = "in a studio setting"
```

### 2. Conflicting Information

```
Input: ["standing", "sitting", "lying down"]  # Can't be all three

MUTUALLY_EXCLUSIVE = {
    "pose": ["standing", "sitting", "lying", "kneeling"],
    "gaze": ["looking at viewer", "looking away", "eyes closed"]
}

def resolve_conflicts(tokens, category):
    if category in MUTUALLY_EXCLUSIVE:
        # Keep highest confidence
        return max(tokens, key=lambda t: t.confidence)
```

### 3. Clothing Relationship Detection

```
def detect_clothing_relationship(items):
    colors = extract_colors(items)
    if len(set(colors)) == 1:
        return "matching"
    elif is_formal_set(items):
        return "paired_with"
    else:
        return "combined_with"
```

### 4. Context Passing for Pronouns

```
def generate_action(tokens, context):
    pronoun = context.get('subject_pronoun', 'They')
    return f"{pronoun} {action_phrase}"
```

---

## Quality Improvements Without LLM

| Technique | Improvement | Effort |
|-----------|-------------|--------|
| More templates | +Variety | Low |
| Better phrase mappings | +Naturalness | Medium |
| Adjective ordering rules | +Grammar | Low |
| Sentence fusion | +Fluency | Medium |
| Pronoun tracking | +Coherence | Low |
| Confidence thresholds | +Accuracy | Low |
| Synonym variation | +Variety | Medium |
| Paragraph structure | +Readability | Low |

---

## Limitations vs LLM Approach

| Aspect | Rule-Based | LLM |
|--------|------------|-----|
| Consistency | High (deterministic) | Variable |
| Speed | Very fast | Slower |
| Cost | Zero inference cost | API/GPU cost |
| Creativity | Limited to templates | High |
| Edge cases | Needs manual handling | Handles gracefully |
| Maintenance | Template updates | Prompt tuning |
| Grammar errors | Possible if rules incomplete | Rare |

---

## Verbosity Control

```
VERBOSITY = {
    "minimal": {
        "subject_details": ["hair", "clothing"],
        "max_adjectives": 2
    },
    "standard": {
        "subject_details": ["hair", "eyes", "clothing", "body"],
        "max_adjectives": 3
    },
    "detailed": {
        "subject_details": "all",
        "max_adjectives": "unlimited"
    }
}
```

---

# Phase 6: LLM-Based Assembly (Alternative)

## Overview

This phase provides an alternative to rule-based assembly using Large Language Models for natural language generation. Use this when higher quality, more natural output is needed and latency/cost are acceptable.

## When to Use LLM vs Rule-Based

| Use Case | Rule-Based | LLM |
|----------|------------|-----|
| High volume, low variance | ✓ Best | Overkill |
| Need creative variety | Limited | ✓ Best |
| Complex token relationships | Brittle | ✓ Best |
| Edge cases / unusual combinations | Fails | ✓ Handles gracefully |
| Strict format control | ✓ Best | Needs constraints |
| Cost-sensitive | ✓ Best | Expensive at scale |
| Real-time / low latency | ✓ Best | Slower |

---

## LLM Assembly Approaches

### Approach 1: Single Comprehensive Prompt

**Input Preparation:**

Transform classified tokens into clean structured format:

```json
{
  "Quality Boosters": ["realistic", "sharp", "professional quality"],
  "Subject": ["1girl", "solo", "young woman", "mid-twenties"],
  "Subject Details": {
    "hair": ["long hair", "wavy hair", "brown hair"],
    "eyes": ["brown eyes", "looking at viewer"],
    "face": ["neutral expression", "fair complexion", "parted lips"],
    "body": ["slim", "curvy"],
    "clothing": ["white crop top", "white panties", "Calvin Klein"]
  },
  "Action/Pose": ["standing", "arms at sides", "facing viewer"],
  "Environment/Scene": ["indoor", "plain light grey background", "studio"],
  "Lighting": ["soft light", "even lighting", "neutral tones", "low contrast"],
  "Style/Medium": ["photograph", "photo realistic", "minimalist", "modern"],
  "Composition": ["front view", "cowboy shot", "centered", "portrait orientation"]
}
```

**System Prompt:**

```
You are an expert image description writer. Your task is to convert structured
image metadata into natural, fluent, professional image descriptions.

RULES:
1. Write in third person, present tense
2. Be concise but comprehensive - aim for 3-4 paragraphs
3. Flow naturally - don't just list attributes
4. Prioritize important visual elements
5. Use appropriate art/photography terminology
6. Don't invent details not present in the input
7. Ignore duplicate or redundant tokens
8. Convert tag-style tokens (like "1girl") into natural language ("a woman")

STRUCTURE:
- Paragraph 1: Overall style, medium, and quality
- Paragraph 2: Subject description (appearance, clothing)
- Paragraph 3: Pose, action, and environment
- Paragraph 4: Lighting and composition (if notable)

OUTPUT FORMAT:
Return only the description text, no headers or labels.
```

**User Prompt:**

```
Convert the following classified image tokens into a natural, readable description:

{json_structured_tokens}
```

**Expected Output:**

```
A sharp, professional-quality photograph with a minimalist, modern aesthetic,
captured in a photo-realistic style.

The image features a young woman, appearing to be in her mid-twenties, with long,
wavy brown hair and brown eyes that gaze directly at the camera. She has a fair
complexion and a neutral expression, her lips slightly parted. Her figure is slim
yet curvy, and she wears a white Calvin Klein crop top paired with matching white briefs.

She stands confidently with her arms relaxed at her sides, facing the viewer
directly. The setting is an indoor studio with a plain, light grey background
that provides clean contrast to her attire.

The lighting is soft and even, creating neutral tones with low contrast that
complement the minimalist aesthetic. The shot is composed from a frontal angle,
framed from mid-thigh up in a classic cowboy shot, with the subject centered
in the portrait-oriented frame.
```

---

### Approach 2: Section-by-Section Generation

Generate each section independently with focused prompts:

**Subject Details Prompt:**

```
Convert these subject detail tokens into 2-3 natural sentences describing
a person's appearance:

Hair: ["long hair", "wavy hair", "brown hair"]
Eyes: ["brown eyes", "looking at viewer"]
Face: ["neutral expression", "fair complexion", "parted lips"]
Body: ["slim", "curvy"]
Clothing: ["white crop top", "white panties", "Calvin Klein"]

Use natural flow, don't list attributes mechanically. Use "she/her" pronouns.
```

**Advantages:**

| Benefit | Explanation |
|---------|-------------|
| Parallel execution | 5 calls simultaneously = faster total time |
| Granular control | Different prompts per section |
| Easier debugging | Isolate which section has issues |
| Caching | Cache common section outputs |
| Fallback | If one fails, others still work |

---

### Approach 3: Few-Shot Prompting

Provide examples for consistent output style:

```
You convert structured image tokens into natural descriptions.

EXAMPLE 1:
Input:
{
  "Subject": ["1boy", "solo"],
  "Subject Details": {"hair": ["short black hair"], "clothing": ["blue suit", "red tie"]},
  "Action/Pose": ["sitting", "arms crossed"],
  "Environment/Scene": ["office", "window background"]
}

Output:
A man with short black hair sits with his arms crossed, dressed in a sharp blue
suit with a red tie. The setting is a professional office environment, with
windows visible in the background.

EXAMPLE 2:
Input:
{
  "Subject": ["2girls"],
  "Subject Details": {"hair": ["blonde hair", "black hair"], "clothing": ["summer dresses"]},
  "Action/Pose": ["walking", "holding hands"],
  "Environment/Scene": ["beach", "sunset"]
}

Output:
Two women walk hand in hand along the beach - one with flowing blonde hair,
the other with dark black hair. Both wear light summer dresses that catch the
warm glow of the sunset behind them.

NOW CONVERT:
Input:
{your_actual_tokens}

Output:
```

---

### Approach 4: Structured Output with Schema

Force consistent structure using JSON mode:

**Prompt:**

```
Convert the image tokens into a structured description. Return JSON only.

Input tokens:
{json_tokens}

Return this exact JSON structure:
{
  "opening_line": "One sentence about style/medium/quality",
  "subject_description": "1-2 sentences about the main subject",
  "appearance_details": "2-3 sentences about physical appearance and clothing",
  "action_and_scene": "1-2 sentences about pose and environment",
  "technical_notes": "1 sentence about lighting/composition (or null if unremarkable)",
  "full_description": "All above combined into flowing paragraphs"
}
```

**Output:**

```json
{
  "opening_line": "A sharp, photo-realistic image with a minimalist modern aesthetic.",
  "subject_description": "The photograph features a young woman in her mid-twenties.",
  "appearance_details": "She has long, wavy brown hair and brown eyes that meet the camera directly. Her expression is neutral with slightly parted lips, and she has a fair complexion. Her slim, curvy figure is dressed in a white Calvin Klein crop top with matching white briefs.",
  "action_and_scene": "She stands confidently with arms relaxed at her sides, facing the viewer against a plain light grey studio background.",
  "technical_notes": "Soft, even lighting with neutral tones complements the centered, cowboy-shot composition.",
  "full_description": "A sharp, photo-realistic image with a minimalist modern aesthetic. The photograph features a young woman in her mid-twenties with long, wavy brown hair and brown eyes that meet the camera directly. Her expression is neutral with slightly parted lips, and she has a fair complexion. Her slim, curvy figure is dressed in a white Calvin Klein crop top with matching white briefs. She stands confidently with arms relaxed at her sides, facing the viewer against a plain light grey studio background. Soft, even lighting with neutral tones complements the centered, cowboy-shot composition."
}
```

**Benefits:**
- Guaranteed structure for downstream processing
- Can use individual fields for different purposes
- Easy validation
- Enables A/B testing of different sections

---

### Approach 5: Template + LLM Hybrid (Recommended)

Use templates for structure, LLM for natural phrasing within slots:

**Template:**

```
"{style_sentence}

{subject_intro} with {hair_description} and {eye_description}. {face_body_description}. {clothing_description}.

{action_scene_description}.

{lighting_composition_description}."
```

**LLM Calls (Small, Focused):**

```
Call 1: "Convert these hair tokens to a natural phrase: ['long hair', 'wavy hair', 'brown hair']"
→ "long, wavy brown hair"

Call 2: "Convert these clothing tokens to a natural sentence: ['white crop top', 'white panties', 'Calvin Klein']"
→ "She wears a white Calvin Klein crop top with matching white briefs"

... etc
```

**Assembly:**

```python
description = TEMPLATE.format(
    style_sentence=llm_style_output,
    subject_intro=llm_subject_output,
    hair_description=llm_hair_output,
    eye_description=llm_eye_output,
    # ... etc
)
```

**Benefits:**
- Controlled overall structure
- Natural language within each slot
- Cheaper (smaller prompts)
- Parallelizable
- Cacheable at slot level

---

## Model Selection

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| Claude 3.5 Haiku | Fast | Good | Low | High volume |
| Claude 3.5 Sonnet | Medium | Excellent | Medium | Quality focus |
| GPT-4o-mini | Fast | Good | Low | High volume |
| GPT-4o | Medium | Excellent | High | Quality focus |
| Llama 3 8B (local) | Fast | Decent | Free | Cost-sensitive |
| Mistral 7B (local) | Fast | Decent | Free | Cost-sensitive |
| Qwen 2.5 7B (local) | Fast | Good | Free | Cost-sensitive |

### Recommendation by Volume

| Daily Volume | Recommended Approach |
|--------------|---------------------|
| < 1,000 | Single comprehensive prompt, Sonnet/GPT-4o |
| 1,000 - 10,000 | Section-by-section, Haiku/GPT-4o-mini |
| 10,000 - 100,000 | Hybrid template + LLM slots, Haiku |
| > 100,000 | Rule-based primary, LLM for edge cases only |

---

## Prompt Engineering Best Practices

### 1. Token Normalization Before LLM

```
BEFORE sending to LLM:
- Remove duplicate tokens
- Sort by confidence (highest first)
- Limit to top N per category (prevent overload)
- Normalize formats (1girl → "woman", etc.)
```

### 2. Negative Instructions

```
DO NOT:
- Add information not present in the tokens
- Use phrases like "the image shows" repeatedly
- List attributes with bullet points
- Include technical tags verbatim (like "1girl" or "cowboy shot")
- Speculate about context, story, or narrative
- Add emotional interpretations unless expression tokens indicate it
```

### 3. Style Calibration

```
TONE: Professional, neutral, descriptive
VOICE: Third person, present tense
LENGTH: 100-200 words (adjust as needed)
VOCABULARY: Art/photography terminology appropriate
AVOID: Flowery language, excessive adjectives, repetition
```

### 4. Temperature Settings

| Use Case | Temperature |
|----------|-------------|
| Consistent output | 0.0 - 0.3 |
| Slight variation | 0.4 - 0.6 |
| Creative variety | 0.7 - 1.0 |

**Recommendation:** 0.1 - 0.3 for reproducible prompt generation.

### 5. Confidence Integration

```
Include confidence in prompt for prioritization:

High confidence (>0.9): ["1girl", "brown hair", "standing"]
Medium confidence (0.7-0.9): ["wavy hair", "slim"]
Lower confidence (0.5-0.7): ["curvy", "parted lips"]

Prioritize high-confidence tokens. Include medium-confidence if relevant.
Mention lower-confidence details only if they add value.
```

---

## Output Validation

### Schema Validation

```python
def validate_output(text):
    checks = {
        "min_length": len(text) > 50,
        "max_length": len(text) < 1000,
        "no_raw_tags": not re.search(r'\b1girl\b|\bcowboy shot\b', text),
        "has_sentences": text.count('.') >= 2,
        "no_bullets": '•' not in text and '- ' not in text,
        "no_headers": not re.search(r'^#+\s', text, re.MULTILINE),
    }
    return all(checks.values()), checks
```

### Semantic Validation

```python
def semantic_check(input_tokens, output_text):
    # Check key elements are mentioned
    required_elements = []

    if "brown hair" in input_tokens["Subject Details"]["hair"]:
        required_elements.append(("brown" in output_text and "hair" in output_text))

    if "standing" in input_tokens["Action/Pose"]:
        required_elements.append("stand" in output_text.lower())

    return all(required_elements)
```

---

## Caching Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 1: Full token hash → Complete description                 │
│           (Exact match cache)                                    │
│                                                                  │
│  Level 2: Section token hash → Section text                      │
│           (Partial reuse)                                        │
│                                                                  │
│  Level 3: Common phrase cache                                    │
│           "long wavy brown hair" → "long, wavy brown hair"       │
└─────────────────────────────────────────────────────────────────┘
```

### Cache Key Generation

```python
def generate_cache_key(tokens):
    # Normalize and sort for consistent hashing
    normalized = {
        category: sorted(set(token.lower().strip() for token in items))
        for category, items in tokens.items()
    }
    return hashlib.md5(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
```

---

## Error Handling

### Fallback Chain

```
FALLBACK CHAIN:
1. Retry with backoff (up to 3 attempts)
2. Try alternative model (Haiku → local Llama)
3. Fall back to rule-based assembly
4. Return structured tokens as-is (worst case)
```

### Error Types

```
ERROR TYPES:
├── Rate Limit
│   → Exponential backoff, queue system
│
├── Timeout (>30s)
│   → Reduce token count, simpler prompt
│
├── Content Filter Triggered
│   → Log, fallback to rule-based
│
├── Invalid JSON (structured output)
│   → Retry with stricter prompt, parse fallback
│
└── Semantic Validation Failed
    → Retry once, then rule-based fallback
```

### Content Safety Handling

```
MITIGATION:
- Pre-filter tokens before sending to LLM
- Use less restrictive models for certain content
- Have rule-based fallback for filtered requests
```

---

## Cost Estimation

### Per-Image Cost (Approximate)

| Approach | Input Tokens | Output Tokens | Cost (Claude Haiku) |
|----------|--------------|---------------|---------------------|
| Single prompt | ~500 | ~200 | $0.0002 |
| Section-by-section (5 calls) | ~300 total | ~200 total | $0.0002 |
| Hybrid (10 slot calls) | ~200 total | ~100 total | $0.00012 |

### At Scale

| Daily Volume | Monthly Cost (Haiku) | Monthly Cost (Sonnet) |
|--------------|---------------------|----------------------|
| 1,000 | ~$6 | ~$45 |
| 10,000 | ~$60 | ~$450 |
| 100,000 | ~$600 | ~$4,500 |
| 1,000,000 | ~$6,000 | ~$45,000 |

### Typical ComfyUI User (50-200 images/day)

| Approach | Daily Cost | Monthly Cost |
|----------|------------|--------------|
| Haiku (single prompt) | $0.01 - $0.04 | $0.30 - $1.20 |
| GPT-4o-mini | $0.01 - $0.05 | $0.30 - $1.50 |
| Local Qwen 7B | $0 (GPU time only) | $0 |

---

## Hybrid Decision: LLM vs Rule-Based

```
IF all tokens have confidence > 0.8:
    Use rule-based (predictable inputs)
ELSE IF any category has unusual/low-confidence tokens:
    Use LLM (handles edge cases better)
```

### Mode Selection in Application

```
MODE = "Standard"      → Rule-based assembly (Phase 5)
MODE = "Enhance with AI" → LLM assembly (Phase 6)
```

---

## Batch Processing for High Volume

For Section-by-Section approach with 5 calls per image:

```
DON'T: 5 sequential API calls per image

DO: Batch all section calls across N images
    - Collect 100 images
    - Send all "Subject" sections in one batch
    - Send all "Clothing" sections in one batch
    - Recombine results
```

This reduces API overhead significantly.

---

## Prompt Versioning

```python
PROMPT_VERSIONS = {
    "v1.0": original_prompt,
    "v1.1": improved_prompt,
    "v2.0": major_revision
}

# Track which version generated each output
metadata["prompt_version"] = "v1.1"
```

Essential for debugging and A/B testing.

---

## Comparison Summary: Rule-Based vs LLM Assembly

| Aspect | Rule-Based (Phase 5) | LLM (Phase 6) |
|--------|---------------------|---------------|
| **Output Quality** | Good, predictable | Excellent, natural |
| **Variety** | Limited by templates | High variety |
| **Edge Cases** | May fail gracefully | Handles well |
| **Speed** | Very fast (<10ms) | Slower (500ms-3s) |
| **Cost** | Free | API costs |
| **Consistency** | Deterministic | Variable |
| **Maintenance** | Template updates | Prompt tuning |
| **Dependencies** | None (pure Python) | API/GPU required |

---

# Phase 7: Token Grounding Visualization (Optional)

## Overview

This phase provides visual debugging and QA capabilities by showing WHERE in the image each token/tag spatially corresponds. This is useful for:
- Validating classification accuracy
- Debugging token extraction
- Understanding model attention
- User-facing visualization features

**Priority**: LOW (implement after core pipeline is complete)

---

## Grounding Approaches

### Approach 1: CLIP-Based Attention Heatmaps

**Concept:** CLIP encodes images and text into shared embedding space. Extract attention maps to see where the model "looks" for each token.

```
Image → CLIP Vision Encoder → Attention Maps (per patch)
Token → CLIP Text Encoder → Text Embedding

For each token:
    Compute similarity between text embedding and each image patch
    → Heatmap of patch-level similarities
```

**Methods:**
- Direct patch similarity
- GradCAM on CLIP similarity score

**Tools:** `CLIP-ViT`, `pytorch-grad-cam`

| Pros | Cons |
|------|------|
| Fast (~20ms GPU) | Low spatial precision |
| Works for abstract concepts | Gradient-based approximation |
| Minimal overhead | Noisy for small regions |

**Best For:** Abstract tokens (lighting, style, quality) that can't be precisely localized.

---

### Approach 2: Grounding DINO + SAM

**Concept:** Two-stage approach - Grounding DINO finds bounding boxes for text queries, SAM refines to precise masks.

```
┌─────────────────────────────────────────────────────────────────┐
│  Token: "white crop top"                                         │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GROUNDING DINO                                          │    │
│  │  Input: Image + "white crop top"                         │    │
│  │  Output: Bounding box [(x1,y1), (x2,y2)]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SAM (Segment Anything)                                  │    │
│  │  Input: Image + Box prompt                               │    │
│  │  Output: Precise segmentation mask                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Token → Prompt Conversion:**

| Category | Token | Grounding Prompt |
|----------|-------|------------------|
| Clothing | "white crop top" | "white crop top" |
| Hair | "brown hair" | "hair" |
| Environment | "grey background" | "background" |
| Action/Pose | "arms at sides" | "arms" |

| Pros | Cons |
|------|------|
| Precise segmentation | Slower (two models) |
| State-of-the-art accuracy | ~1.1GB model overhead |
| Handles complex scenes | Abstract concepts can't be grounded |

**Best For:** Objects, clothing, body parts, props.

---

### Approach 3: Semantic Segmentation + Token Mapping

**Concept:** Use human parsing model to get pixel-level labels, map tokens to segment classes.

```
┌─────────────────────────────────────────────────────────────────┐
│  HUMAN PARSING MODEL (SCHP / Graphonomy)                        │
│                                                                  │
│  Input: Image                                                    │
│  Output: Per-pixel class labels                                  │
│                                                                  │
│  Classes: hair, face, upper_clothes, lower_clothes, arms,       │
│           legs, shoes, background, skin, etc.                    │
└─────────────────────────────────────────────────────────────────┘
```

**Token → Segment Class Mapping:**

```
TOKEN_TO_SEGMENT = {
    # Hair tokens
    "long hair": "hair",
    "brown hair": "hair",
    "wavy hair": "hair",

    # Face tokens
    "brown eyes": "eyes",
    "lips": "lips",
    "parted lips": "lips",

    # Body tokens
    "navel": "torso",
    "breasts": "torso",
    "arms at sides": "arms",

    # Clothing tokens
    "white crop top": "upper_clothes",
    "white panties": "lower_clothes",
    "bra": "upper_clothes",

    # Background
    "grey background": "background",
}
```

| Pros | Cons |
|------|------|
| Very precise for known classes | Limited to predefined classes |
| Fast inference (~50ms) | Requires class mapping dictionary |
| Consistent results | Can't handle novel concepts |

**Best For:** Subject Details subcategories (hair, clothing, body, face).

---

### Approach 4: Florence-2 Grounding

**Concept:** Florence-2 has built-in grounding - can output bounding boxes for concepts it describes.

```
Florence-2 Tasks:
├── <CAPTION_TO_PHRASE_GROUNDING>
│   Input: "brown hair, white top, standing"
│   Output: Bounding boxes for each phrase
│
├── <REFERRING_EXPRESSION_SEGMENTATION>
│   Input: "the woman's hair"
│   Output: Segmentation mask
│
└── <OPEN_VOCABULARY_DETECTION>
    Input: "crop top"
    Output: Detection boxes
```

| Pros | Cons |
|------|------|
| Already loaded for Analysis | Grounding quality varies |
| Multi-task capable | Not all variants support grounding |
| Zero additional models | Box-level (not mask) |

**Best For:** Leveraging existing Florence-2 model with minimal overhead.

---

## Recommended Strategy: Category-Based Router

```python
def select_grounding_method(token, category):
    # Subject Details - use parsing models
    if category == "Subject Details":
        subcategory = token.metadata.get("subcategory")
        if subcategory in ["hair", "face", "body", "clothing"]:
            return "human_parsing"
        else:
            return "grounding_dino"

    # Environment - use scene parsing or grounding
    elif category == "Environment/Scene":
        if "background" in token.text:
            return "scene_parsing"
        else:
            return "grounding_dino"

    # Action/Pose - use pose estimation
    elif category == "Action/Pose":
        return "pose_estimation"

    # Abstract concepts (can't ground spatially)
    elif category in ["Lighting", "Style/Medium", "Quality Boosters"]:
        return "clip_attention"  # or skip

    # Default fallback
    else:
        return "grounding_dino"
```

---

## Visualization Strategies

### 1. Single Token Heatmap

Overlay mask/heatmap for one token at a time.

```
┌─────────────────────────────────────────────────────────────────┐
│  Image with "brown hair" highlighted                             │
│                                                                  │
│        ████████████                                              │
│       ██ HAIR MASK ██                                            │
│        ████████████                                              │
│                                                                  │
│         [Person]                                                 │
│                                                                  │
│  Token: brown hair (0.915)                                       │
│  Category: Subject Details > Hair                                │
│  Source: wd14                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Multi-Token Composite

Color-coded overlays for multiple tokens, grouped by category.

```
Color Scheme:
├── Hair tokens: Yellow
├── Clothing tokens: Blue
├── Face tokens: Pink
├── Body tokens: Orange
├── Environment: Green
└── Action/Pose: Purple
```

### 3. Interactive HTML Viewer

```html
<div id="viewer">
    <img src="base_image.png" id="base">
    <div class="overlay" data-token="brown_hair">
        <img src="mask_hair.png">
    </div>
</div>

<div id="token-list">
    <label><input type="checkbox" data-token="brown_hair"> brown hair (0.91)</label>
    <label><input type="checkbox" data-token="crop_top"> white crop top (0.87)</label>
</div>

<div id="category-filter">
    [All] [Subject] [Details] [Environment] [Action]
</div>
```

---

## Handling Non-Groundable Tokens

Some tokens can't be spatially grounded:

| Category | Examples | Strategy |
|----------|----------|----------|
| Quality Boosters | "realistic", "sharp", "4k" | Global indicator (badge/icon) |
| Style/Medium | "photograph", "minimalist" | Global indicator |
| Lighting | "soft light", "high key" | Whole-image tint or skip |
| Composition | "centered", "rule of thirds" | Grid overlay |
| Technical | "portrait orientation" | Frame indicator |

---

## Output Formats

### 1. Static Image Export

```
Output files:
├── image_original.png
├── image_heatmap_hair.png
├── image_heatmap_clothing.png
├── image_heatmap_combined.png
└── image_with_legend.png
```

### 2. JSON Annotation Format

```json
{
    "image": "image_001.jpg",
    "dimensions": [512, 288],
    "token_regions": [
        {
            "token": "brown hair",
            "category": "Subject Details > Hair",
            "confidence": 0.915,
            "source": "wd14",
            "region": {
                "type": "mask",
                "rle": "...",
                "bbox": [120, 50, 280, 180],
                "area": 12500,
                "centroid": [200, 115]
            },
            "grounding_method": "human_parsing",
            "grounding_confidence": 0.92
        }
    ],
    "ungrounded_tokens": [
        {"token": "realistic", "reason": "abstract_concept"},
        {"token": "soft light", "reason": "global_attribute"}
    ]
}
```

---

## Recommended Implementation Stack

| Component | Tool | Purpose | Size |
|-----------|------|---------|------|
| Human Parsing | `SCHP` or `Graphonomy` | Body, clothes, hair | ~150MB |
| Face Parsing | `face-parsing.PyTorch` | Eyes, lips, nose | ~100MB |
| General Grounding | `Grounding DINO` | Objects, props | ~700MB |
| Mask Refinement | `SAM` (ViT-B) | Bounding box → mask | ~400MB |
| Fallback | `CLIP + GradCAM` | Abstract concepts | ~350MB |
| Visualization | `matplotlib` + `OpenCV` | Overlay composition | - |

### Performance Estimates

| Model | GPU | CPU |
|-------|-----|-----|
| SCHP (human parsing) | 50ms | 500ms |
| Face parsing | 30ms | 300ms |
| Grounding DINO | 100ms | 2s |
| SAM (ViT-B) | 50ms | 1s |
| CLIP ViT-B/32 | 20ms | 200ms |

**Full pipeline per image:** ~300-500ms (GPU), ~3-5s (CPU)

---

## Integration with Validation

Token grounding can enhance validation (Phase 6) by:

1. **Spatial Coherence Check**: Tokens in same category should have overlapping or adjacent regions
2. **Coverage Verification**: Subject Details tokens should cover expected body areas
3. **Conflict Detection**: Mutually exclusive tokens (standing vs sitting) should have non-overlapping regions
4. **Visual QA Report**: Generate annotated images showing classification accuracy

---

## Implementation Priority

This phase is **optional** and should only be implemented after the core pipeline (Phases 1-6) is complete and validated.

| Task | Priority | Effort |
|------|----------|--------|
| Human parsing integration | Medium | 4-6 hours |
| Grounding DINO + SAM | Low | 6-8 hours |
| Interactive HTML viewer | Low | 4-6 hours |
| JSON export format | Medium | 2-3 hours |
| Integration with validation | Low | 3-4 hours |
