# Z-Image Turbo: Complete Prompting Guide
## Detailed Structure for All Image Categories

---

## Part 1: Core Principles

### The Fundamental Rule
**Every word should describe something visible.**

The model cannot see "beautiful," "mysterious," or "interesting." It sees colors, shapes, textures, and compositions.

| Vague (Bad) | Visual (Good) |
|-------------|---------------|
| beautiful woman | 25-year-old woman with high cheekbones and full lips |
| mysterious expression | half-lidded eyes, slight smirk, gaze averted |
| nice clothes | tailored navy wool peacoat with brass buttons |
| good lighting | warm side light from left at 45 degrees |
| interesting background | rain-wet Tokyo street with neon reflections |

### Z-Image Turbo Technical Requirements

| Parameter | Value | Notes |
|-----------|-------|-------|
| `guidance_scale` | **0.0** | CFG is baked into Turbo - keep at 0 |
| `num_inference_steps` | **8** | Model is distilled for 8 steps |
| Negative prompts | **Not supported** | Put all constraints in positive prompt |
| `max_sequence_length` | **1024** | Use it! More detail = more control |
| Text encoder | **Qwen3-4B** | Understands natural language well |

---

## Part 2: Universal Prompt Formula

### The Master Template

```
[Subject] + [Details] + [Action/Pose] + [Setting] + [Lighting] + [Style] + [Technical] + [Constraints]
```

### Layered Structure (from multiple sources)

**Layer 1: Core Subject**
- Who/what is the primary focus
- Basic identity and role

**Layer 2: Specific Details**
- Physical attributes (age, features, build)
- Materials, textures, conditions
- Distinguishing characteristics

**Layer 3: Action/State**
- What the subject is doing
- Pose and body language
- Movement or stillness

**Layer 4: Environment**
- Location and setting
- Foreground/middle ground/background
- Time of day, weather, season

**Layer 5: Lighting**
- Direction (from where)
- Quality (hard/soft)
- Color temperature (warm/cool)
- Named setups (Rembrandt, butterfly, etc.)

**Layer 6: Style/Medium**
- Artistic reference (photography, painting, illustration)
- Camera/lens specifications
- Film stock or digital look
- Artist/movement references

**Layer 7: Technical Quality**
- Resolution indicators
- Focus and depth of field
- Quality modifiers

**Layer 8: Constraints (Negative-as-Positive)**
- "no watermark, no logos, no text"
- "clean background, no distractions"
- "correct anatomy, natural hands"

---

## Part 3: Category-Specific Formulas

### 3.1 Portrait Photography

#### Formula
```
Portrait of a [age] [ethnicity] [gender] with [face shape] face 
and [skin description]. [Eye description]. [Hair description]. 
[Expression description]. Wearing [clothing details]. [Pose]. 
[Background]. [Lighting setup]. Shot on [camera], [lens] at 
[aperture], [quality keywords].
```

#### Component Vocabulary

**Skin & Complexion**
- Tone: warm ivory, cool beige, golden tan, deep brown, olive, ruddy, sallow, sun-kissed
- Texture: smooth, porcelain, weathered, freckled, matte, dewy, natural pores visible
- Details: fine lines around eyes, crow's feet, smile lines, acne scars, beauty mark, stubble shadow

**Eyes**
- Shape: almond, round, hooded, deep-set, wide-set, upturned, monolid
- Color: steel grey, warm brown, pale blue, amber, hazel with gold flecks
- State: bloodshot, glassy, sharp, tired, bright and alert, narrowed

**Hair**
- Texture: straight and silky, wavy with volume, tight curls, coily, frizzy
- State: clean, windswept, disheveled, meticulously styled, damp
- Details: grey streaks at temples, roots showing, natural highlights

**Expressions**
- Composed with slight knowing smile
- Thoughtful gaze directed camera left
- Half-lidded eyes with slight smirk
- Direct, confident stare

#### Example Prompt
```
Portrait of a 45-year-old Scandinavian man with a square jaw and 
weathered fair skin showing fine lines around his eyes. Steel blue 
eyes with a direct, thoughtful gaze. Short salt-and-pepper hair, 
neatly cropped on the sides. Composed expression with a slight 
knowing smile. Wearing a charcoal wool turtleneck. Head and 
shoulders, turned slightly to camera left. Clean grey gradient 
background. Soft studio lighting from upper left with subtle fill 
from right. Shot on Hasselblad X2D, 80mm at f/2.8, editorial 
portrait photography.
```

---

### 3.2 Full Body / Fashion Photography

#### Formula
```
Full body shot of [subject description]. [Build and posture]. 
[Outfit description from top to bottom]. [Shoes/footwear]. 
[Accessories]. [Pose with specific limb positions]. [Setting]. 
[Lighting]. [Camera angle]. Shot on [camera/lens]. [Style keywords].
```

#### Component Vocabulary

**Build & Posture**
- Tall and athletic, confident posture
- Petite frame, weight on left leg
- Shoulders back, relaxed stance
- Lean build, model proportions

**Outfit Description Order**
1. Outerwear (jacket, coat)
2. Top layer
3. Bottom (pants, skirt)
4. Footwear
5. Accessories

**Clothing Condition**
- New: crisp, pressed, bright saturated colors, sharp creases
- Worn: faded, soft, stretched, pilled, comfortable
- Damaged: torn, patched, stained, frayed

**Pose Specifics**
- Left hand in jacket pocket, right holding phone
- Weight on left leg, hip cocked
- Arms crossed, head tilted
- Walking mid-stride, one foot forward

#### Example Prompt
```
Full body shot of a tall, athletic young woman in her early 20s. 
Confident posture, weight on left leg, shoulders back. Wearing 
an oversized vintage denim jacket with enamel pins on lapel, 
white cropped tank top underneath, high-waisted black wide-leg 
trousers. White platform sneakers. Silver layered necklaces, 
small hoop earrings. Left hand in jacket pocket, right hand 
holding phone at side, looking at camera with head tilted. 
Urban street corner with graffiti wall behind. Late afternoon 
golden hour light from right. Eye-level medium shot. Shot on 
Sony A7IV, 35mm at f/2. Streetwear editorial, contemporary 
fashion photography.
```

---

### 3.3 Action / Dynamic Shots

#### Formula
```
[Subject] in mid-[action]. [Body position details]. [Which limbs 
lead the motion]. [Fabric/hair movement]. [Expression matching 
effort]. [Motion effects]. [Background]. [Camera angle and distance]. 
[Lighting]. [Style].
```

#### Component Vocabulary

**Movement Descriptions**
- Explosive push-off from right foot
- Left knee driving upward
- Arms pumping opposite to legs
- Torso leaning forward at 30 degrees

**Motion Effects**
- Slight motion blur on trailing arm
- Hair streaming behind
- Fabric billowing
- Speed trails (for stylized)

**Camera Angles for Action**
- Low angle from knee height (power)
- High angle (vulnerability)
- Dutch angle (tension)
- Tracking shot (following motion)

#### Example Prompt
```
Female athlete in mid-sprint, explosive push-off from right foot. 
Left knee driving upward, arms pumping opposite to legs. Torso 
leaning forward at 30 degrees, shoulders rotated into the run. 
Ponytail streaming behind, loose strands across face. Teeth 
gritted, eyes focused on distant finish line. Slight motion 
blur on trailing arm. Red athletic tank top, black compression 
shorts, white racing spikes. Track and field stadium, blurred 
crowd in background. Low angle shot from knee height, capturing 
power. Dramatic side lighting highlighting muscle definition. 
Sports photography, Olympic quality, peak action moment.
```

---

### 3.4 Product Photography

#### Formula
```
[Product] on [surface]. [Product details - materials, colors, 
textures]. [Arrangement if multiple items]. [Props]. [Background]. 
[Lighting setup]. [Camera angle]. [Style]. [Constraints].
```

#### Component Vocabulary

**Surface Types**
- Brushed concrete, white marble with gold veins
- Matte black slate, weathered wood
- Reflective acrylic, frosted glass
- Natural linen, textured paper

**Product Material Descriptions**
- Rose gold case, brushed steel
- Deep blue sunburst dial
- Brown alligator leather strap
- Polished chrome, matte black

**Lighting for Products**
- Soft directional from upper right
- Dual soft boxes at 45 degrees
- Dramatic single key with subtle fill
- Rim light for separation

**Camera Angles**
- 45-degree overhead (most common)
- Straight overhead (flat lay)
- Eye-level hero shot
- Low angle for authority

#### Example Prompt
```
Luxury mechanical watch on brushed concrete surface. Rose gold 
case, 42mm diameter, exhibition caseback showing movement. Deep 
blue sunburst dial with applied hour markers. Brown alligator 
leather strap with deployment clasp. Watch positioned at 10:10. 
Small succulent plant in grey ceramic pot to the left. Subtle 
water droplets on concrete suggesting morning dew. Dark grey 
fabric backdrop. Soft directional lighting from upper right 
creating gentle shadows, small highlight on crystal. 45-degree 
overhead angle. High-end product photography, luxury watch 
advertisement aesthetic, no text, no watermark, no logos.
```

---

### 3.5 Landscape / Environment Photography

#### Formula
```
[Time of day] view of [location type]. [Foreground elements]. 
[Middle ground]. [Background/horizon]. [Sky description]. 
[Atmospheric effects]. [Color palette]. [Mood]. [Camera settings].
```

#### Component Vocabulary

**Time of Day**
- Golden hour (warm, long shadows)
- Blue hour (cool, soft)
- High noon (harsh, minimal shadows)
- Twilight (mixed warm/cool)
- Night (artificial light, stars)

**Atmospheric Effects**
- Thin mist hovering over water
- Morning fog in valleys
- Dust particles in light beams
- Rain with visible droplets
- Snow flurries

**Color Palettes**
- Cool blue shadows contrasting warm highlights
- Muted earth tones
- Vibrant saturated colors
- Monochromatic with accent color

**Foreground/Middle/Background Structure**
- Foreground: immediate elements (rocks, flowers, path)
- Middle ground: main subject (lake, field, building)
- Background: distant elements (mountains, horizon, sky)

#### Example Prompt
```
Golden hour view of a Norwegian fjord. Rocky shoreline with tide 
pools in immediate foreground, reflecting the orange sky. Calm 
deep blue water in middle ground with a single wooden fishing 
boat anchored. Steep mountain cliffs rising on both sides, 
snow-capped peaks catching the last warm light. Dramatic sky 
with scattered altocumulus clouds painted in pink and gold 
gradients. Thin mist hovering over the water surface. Cool 
blue shadows contrasting warm highlights. Serene, majestic 
atmosphere. Wide angle landscape, shot on Phase One IQ4, 24mm, 
f/11, panoramic aspect ratio.
```

---

### 3.6 Food Photography

#### Formula
```
[Food item/dish] [arrangement/plating]. [Surface/plate description]. 
[Garnishes and props]. [Steam/freshness indicators]. [Background 
elements]. [Lighting setup]. [Camera angle]. [Style reference].
```

#### Component Vocabulary

**Freshness Indicators**
- Steam rising from hot dish
- Condensation on cold glass
- Glistening sauce
- Melting cheese pull
- Fresh herb garnish

**Surfaces**
- Rustic wooden table with visible grain
- White marble with grey veins
- Dark slate plate
- Ceramic bowl with imperfections

**Camera Angles for Food**
- Overhead/flat lay (45° - 90°)
- 45-degree (most common, shows dimension)
- Eye-level (burgers, stacked items)
- Extreme close-up/macro (texture details)

**Lighting for Food**
- Soft natural window light
- Backlit for steam visibility
- Side light for texture
- Diffused overhead for flat lay

#### Example Prompt
```
Freshly baked lasagna with crispy golden top layer, cheese 
bubbling at edges. Served in rustic terracotta baking dish. 
Fresh basil leaves scattered on top, steam rising visibly. 
Rustic wooden table surface with visible grain. Glass of 
red wine and crusty bread loaf in soft focus background. 
Soft natural window light from left creating gentle shadows. 
45-degree angle shot. Shot using Hasselblad camera, ISO 100. 
Professional color grading, soft shadows, clean sharp focus. 
High-end retouching, food magazine photography style.
```

---

### 3.7 Anime / Illustration

#### Formula
```
[Style reference] [character description]. [Hair details]. 
[Eye details with anime conventions]. [Outfit with specific style]. 
[Pose and expression]. [Background/setting]. [Lighting effects]. 
[Quality modifiers]. [Text elements if any].
```

#### Component Vocabulary

**Style References**
- Studio Ghibli style, soft watercolor textures
- Modern digital anime, clean cel shading
- 90s retro anime, visible film grain
- Manga panel, black and white with screentones

**Anime-Specific Features**
- Large expressive eyes with detailed highlights
- Stylized hair with distinct color and flow
- Exaggerated expressions
- Dynamic action lines

**Quality Modifiers for Anime**
- High quality anime art
- Detailed anime illustration
- Cel-shaded, vibrant colors
- Soft anime aesthetic

#### Example Prompt
```
In a Studio Ghibli-esque anime style, a ethereal guardian spirit 
with flowing platinum blonde tresses tied in loose waves, her 
wide blue-green eyes shimmering with sakura petal reflections 
and faint freckles like stardust. Plump lips curved in a gentle, 
knowing smile, wearing a form-fitting black leather corset top 
embroidered with glowing cherry blossom runes. She stands poised 
on a floating torii gate amid a twilight cherry orchard, wind-swept 
petals swirling around her outstretched hand summoning faint 
ethereal lights. Soft cel-shading with vibrant pinks, lavenders, 
and golds, highly detailed hair strands and fabric folds, 8K 
anime rendering with subtle bloom effects.
```

---

### 3.8 Concept Art / Fantasy

#### Formula
```
[Style reference] [scene/subject description]. [Key visual elements]. 
[Atmospheric conditions]. [Color scheme]. [Lighting type]. 
[Artistic technique]. [Quality modifiers].
```

#### Example Prompt
```
Ancient library hidden inside a giant tree, spiraling wooden 
stairs wrapped around the trunk, glowing floating books as 
light sources, warm golden ambient light, dust particles 
visible in the air, small cat sleeping on an open book on a 
reading desk, intricate carved runes on the bark, fantasy 
concept art, highly detailed, 8k illustration look.
```

---

### 3.9 E-Commerce / Commercial

#### Formula
```
[Product type] [positioning]. [Product details]. [Surface/background]. 
[Lifestyle context if any]. [Lighting]. [Output specifications]. 
[Brand-safe constraints].
```

#### Example Prompt
```
Professional studio product shot of stacked folded T-shirts in 
assorted bright colors on a flat white surface. Clean minimal 
shadows, bright even lighting from soft boxes. Sharp focus on 
fabric texture, visible weave detail. Pure white seamless 
background. High-key commercial photography style. No text, 
no logos, no watermarks, clean product cutout ready.
```

---

### 3.10 Architecture / Interior

#### Formula
```
[View type] of [space description]. [Architectural elements]. 
[Materials and textures]. [Furniture/objects if interior]. 
[Natural/artificial lighting]. [Time of day]. [Atmosphere]. 
[Camera specs]. [Style].
```

#### Example Prompt
```
Wide angle interior shot of a minimalist Japanese living room. 
Tatami mat flooring, shoji screen doors filtering soft afternoon 
light. Low wooden coffee table with single ceramic vase holding 
bamboo stems. Floor cushions in muted earth tones. Clean white 
walls with natural wood accents. Subtle shadows from screen 
doors creating geometric patterns on floor. Shot on Sony A7R IV, 
16mm wide angle, f/8. Architectural photography, Japandi 
aesthetic, serene atmosphere.
```

---

### 3.11 Macro Photography

#### Formula
```
Extreme close-up of [subject]. [Surface texture details]. 
[Material properties]. [Scale reference if helpful]. [Lighting 
for texture]. [Focus plane]. [Background treatment]. [Technical].
```

#### Example Prompt
```
Extreme close-up macro shot of a single perfect droplet of 
morning dew resting on a thick, moss-covered stone. The 
droplet reflects surrounding forest light. Intricate moss 
texture visible with individual strands. Extreme shallow 
depth of field, subject in razor sharp focus, background 
dissolving into soft green blur. Natural softbox lighting 
from overcast sky, showcasing the texture of the moss and 
the reflection within the water drop. Shot on Sony A7R IV, 
90mm macro lens at f/2.8, hyper-detailed, 8K resolution.
```

---

### 3.12 Text-in-Image / Typography

Z-Image excels at bilingual text rendering (English + Chinese).

#### Formula
```
[Format type] with [text placement description]. [Exact text 
in quotes]. [Typography style]. [Background/setting]. [Color 
scheme]. [Composition]. [Style]. [Constraints].
```

#### Example Prompt
```
A minimalist movie poster, dark blue background with subtle 
city skyline silhouette at the bottom. Big English title text 
"QUIET STREETS" centered near the top in bold sans-serif 
white letters. Small Chinese subtitle text "静谧之城" below 
it in elegant script. Clean modern typography, simple layout. 
High contrast, cinematic atmosphere. No extra text, no logos, 
no watermark, safe for work.
```

---

## Part 4: Lighting Reference Guide

### Direction Vocabulary
- **From above (overhead)**: Dramatic shadows under features
- **From below (uplight)**: Unnatural, horror effect
- **From left/right at 45°**: Classic key light position
- **From behind (rim/backlight)**: Separation, halo effect
- **Frontal (flat)**: Minimal shadows, even illumination

### Quality Vocabulary
- **Hard/harsh**: Sharp shadows, high contrast
- **Soft/diffused**: Gentle shadows, wrapped light
- **Dappled**: Through leaves, blinds, patterns
- **Even**: Studio, overcast, flat

### Color Temperature
- **Warm**: Golden, amber, orange, candlelight
- **Cool**: Blue, white, moonlight
- **Mixed**: Warm key + cool fill (cinematic)
- **Neon**: Pink, cyan, magenta (cyberpunk)

### Named Lighting Setups
- **Rembrandt**: Triangle shadow under eye, dramatic
- **Butterfly**: Shadow under nose, glamorous
- **Split**: Half face lit, half in shadow
- **Loop**: Small shadow from nose, natural
- **Rim**: Outline/edge lighting, separation

---

## Part 5: Prompt Density Levels

### Minimal (~20-50 words)
For stylized work with creative freedom:
```
Young woman, short pink hair, leather jacket, confident smirk, 
neon city background
```

### Standard (~80-150 words)
Balanced control for most use cases:
```
Portrait of a young woman in her early 20s. Bubblegum pink 
choppy pixie cut. Wearing a black leather biker jacket over 
white tee. Confident smirk, one eyebrow raised. Neon-lit Tokyo 
alley at night, rain-wet pavement. Cinematic lighting with 
cyan and magenta neon reflections. Shot on Sony A7IV, 50mm f/1.4.
```

### Maximum (~200-350 words)
Full control for precise results - see the masterclass examples above.

---

## Part 6: Constraint Encoding (Replacing Negative Prompts)

Since Z-Image ignores negative prompts, encode constraints positively:

### Quality Constraints
- `sharp focus on the subject`
- `clean detailed image`
- `no motion blur, no grainy noise`
- `correct human anatomy, natural hands and fingers`

### Content Constraints
- `no text, no watermark, no logos, no branding`
- `simple, uncluttered background`
- `no extra people in the background`
- `no distracting elements`

### Safety Constraints
- `fully clothed, modest outfit`
- `safe for work, non-sexual`
- `no revealing clothing, no suggestive poses`

### Placement
Add constraints at the END of your prompt, after establishing the main scene.

---

## Part 7: Iteration Strategy

1. **Start Sparse**: Core subject + style only
2. **Identify Issues**: What's wrong with the result?
3. **Add Specifics**: Target the problem areas
4. **Lock What Works**: Keep successful elements
5. **Refine**: Small adjustments, not complete rewrites

### Example Iteration

**V1**: "A woman in a coffee shop"
*Problem: Generic, no personality*

**V2**: "A 28-year-old woman with red hair sitting in a coffee shop, reading"
*Problem: Still generic, lighting flat*

**V3**: "A 28-year-old woman with auburn wavy hair sitting by the window in a cozy coffee shop, reading a worn paperback novel. Warm morning light streaming through the window. Cream knit sweater. Thoughtful expression."
*Problem: Good but composition unclear*

**V4**: "Medium shot of a 28-year-old woman with auburn wavy hair sitting by a rain-streaked window in a cozy coffee shop. Reading a worn paperback novel, one hand holding coffee mug. Cream cable-knit sweater. Thoughtful expression, slight smile at something she read. Warm morning light through window creating soft shadows. Shallow depth of field, background softly blurred. Intimate lifestyle photography."
*Success: Specific, atmospheric, well-composed*

---

## Part 8: Quick Reference Cards

### Portrait Checklist
- [ ] Age/gender indicators
- [ ] 2-3 specific facial features
- [ ] Eye color and expression
- [ ] Hair color + style + state
- [ ] Clothing with condition
- [ ] Background treatment
- [ ] Lighting direction + quality
- [ ] Camera/lens specs

### Full Body Checklist
- [ ] Build and posture
- [ ] Complete outfit (top to bottom)
- [ ] Footwear
- [ ] Accessories
- [ ] Specific pose with limb positions
- [ ] Environment/setting
- [ ] Camera angle

### Product Checklist
- [ ] Product identity
- [ ] Materials and textures
- [ ] Surface/background
- [ ] Props (minimal)
- [ ] Lighting setup
- [ ] Camera angle
- [ ] Brand constraints (no logos, etc.)

### Landscape Checklist
- [ ] Time of day
- [ ] Foreground elements
- [ ] Middle ground (main subject)
- [ ] Background/horizon
- [ ] Sky description
- [ ] Atmospheric effects
- [ ] Color palette
- [ ] Camera settings

---

## Part 9: Aspect Ratio Guidelines

| Ratio | Best For |
|-------|----------|
| Square (1:1) | Avatars, logos, social media |
| Landscape (4:3) | Cinematic scenes, environments |
| Landscape (16:9) | Banners, YouTube thumbnails |
| Portrait (4:3) | Character shots, headshots |
| Portrait (16:9) | Mobile wallpapers, full-body |

---

## Part 10: Sources & Attribution

This guide synthesizes information from:

**Official Sources:**
- Tongyi-MAI GitHub repository
- Hugging Face model card and discussions
- Official Prompt Enhancer (pe.py) template

**Community Guides:**
- z-image.vip Prompt Engineering Masterclass
- fal.ai Z-Image Turbo Prompt Guide
- GitHub Gist by illuminatianon
- Medium articles on Z-Image prompting
- zimage.net blog tutorials
- Atlabs AI prompting guide

**Note:** The category-specific formulas are synthesized best practices from multiple community sources, not official Tongyi-MAI documentation. The core technical requirements (CFG=0, 8 steps, no negative prompts) are from official sources.

---

## Summary

Z-Image Turbo requires a different prompting mindset:

1. **Write naturally** - Full sentences, not tag soup
2. **Be visually specific** - Describe what's visible, not abstract concepts
3. **Layer your prompts** - Subject → Details → Setting → Lighting → Style → Technical
4. **Encode constraints positively** - Say what you want, not what you don't
5. **Use the formula** for your image type as a starting template
6. **Iterate strategically** - Start sparse, add specifics, refine

The model rewards detailed, structured, camera-direction-style prompts. Take advantage of the 1024 token limit for complex scenes.


| Model                      | Type           | Details                                                                     | Link                                                               |
  |----------------------------|----------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------|
  | FashionCLIP                | Zero-shot      | Fine-tuned CLIP for fashion, good for matching text to clothing             | https://huggingface.co/patrickjohncyh/fashion-clip                 |
  | YOLOS-Fashionpedia         | Detection      | 27 apparel categories + accessories (shirt, pants, dress, bag, shoes, etc.) | https://huggingface.co/valentinafeve/yolos-fashionpedia            |
  | YOLOv8n Clothing           | Detection      | Fast clothing detection                                                     | https://huggingface.co/kesimeg/yolov8n-clothing-detection          |
  | SegFormer Clothes          | Segmentation   | Pixel-level clothing segmentation                                           | https://huggingface.co/mattmdjaga/segformer_b2_clothes             |
  | Wargon Clothing Classifier | Classification | ViT for clothing categories                                                 | https://huggingface.co/wargoninnovation/wargon-clothing-classifier |

  Pose Models

  | Model           | Type      | Details                               | Link                                                      |
  |-----------------|-----------|---------------------------------------|-----------------------------------------------------------|
  | Sapiens Pose 1B | Keypoints | 308 keypoints (body+face+hands+feet)  | https://huggingface.co/facebook/sapiens-pose-1b           |
  | MediaPipe Pose  | Keypoints | Lightweight, real-time pose           | https://huggingface.co/qualcomm/MediaPipe-Pose-Estimation |
  | MotionBERT      | 3D Pose   | 17 body keypoints, action recognition | https://huggingface.co/walterzhu/MotionBERT               |



  LLM Models Supported

  | Provider   | Default Model             | Description                                  |
  |------------|---------------------------|----------------------------------------------|
  | OpenRouter | qwen/qwen3-235b-a22b:free | Cloud API (Qwen3 235B free tier)             |
  | Local      | Any model                 | Ollama, LM Studio, vLLM, text-gen-webui      |
  | Direct     | HuggingFace repo          | Direct loading with 4-bit/8-bit quantization |

  Key Details

  - Default: qwen/qwen3-235b-a22b:free (Qwen3 235B via OpenRouter)
  - Vision Support: Auto-detects VL models (Qwen-VL, etc.) for image input
  - Prompt Templates: Chinese and English templates that transform short prompts into detailed visual descriptions
  - Output Cleaning: Removes <think> tags, repetition loops, and metadata

  The Prompt Enhancement Approach

  The node uses a sophisticated system prompt (lines 378-416) that instructs the LLM to:
  1. Analyze core elements (subject, quantity, action, state)
  2. Apply "generative reasoning" for conceptual prompts
  3. Add professional aesthetics (composition, lighting, materials, colors)
  4. Handle text elements with explicit quotation marks
  5. Output objective, concrete descriptions (no "8K", "masterpiece" meta-tags)

  This is different from tagging - it's prompt rewriting to make short prompts more detailed for image generation.
