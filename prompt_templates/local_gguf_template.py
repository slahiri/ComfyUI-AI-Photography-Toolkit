"""
Local GGUF Prompt Template - High Resolution

For local GGUF models (LLaVA, Qwen, MiniCPM-V, Moondream).

Strategy:
- Quick/Standard modes: Use SAME detailed prompts as Claude (no quality compromise)
- Deep mode: Use MORE iterations with FOCUSED prompts to achieve same quality

Deep mode breaks analysis into focused steps:
1. Subject analysis (appearance, features, expression)
2. Clothing/accessories analysis (precise colors, materials, fit)
3. Environment analysis (background, props, spatial)
4. Lighting & color analysis (direction, quality, palette)
5. Style & composition (photography technique, framing)
6. Consolidate all attributes
7. Generate final prompt

Each focused step extracts ONE aspect with HIGH PRECISION.
"""

import json
from typing import Dict, Any, Optional, List, Tuple

from .claude_template import ClaudePromptTemplate


class LocalGGUFPromptTemplate(ClaudePromptTemplate):
    """
    High-resolution prompt template for local GGUF vision models.

    INHERITS from Claude template for Quick/Standard modes (same quality prompts).
    OVERRIDES Deep mode with multi-iteration focused analysis for HIGH ACCURACY.
    """

    @property
    def name(self) -> str:
        return "Local GGUF (High Resolution Multi-iteration)"

    @property
    def uses_multi_iteration_deep_mode(self) -> bool:
        """Local models use multi-iteration approach for deep analysis."""
        return True

    def get_deep_mode_stages(self) -> List[str]:
        """Return the stages for deep mode multi-iteration analysis."""
        return [
            "subject",      # Analyze subject/person physical features, ethnicity, skin, cosmetics
            "pose",         # Analyze detailed body positioning
            "clothing",     # Analyze clothing/accessories in detail
            "environment",  # Analyze background/setting
            "lighting",     # Analyze lighting and colors
            "style",        # Analyze photography style and composition
        ]

    def get_focused_analysis_max_tokens(self) -> int:
        """Max tokens for each focused analysis stage response."""
        return 800  # Sufficient for detailed JSON extraction per stage

    def build_focused_analysis_prompt(
        self,
        stage: str,
        classification: Dict[str, Any],
        content_detail: str,
        previous_results: Dict[str, Any] = None,
    ) -> Tuple[str, str]:
        """
        Build focused prompt for a specific analysis stage in Deep mode.
        Each stage analyzes ONE aspect with HIGH PRECISION for accurate reproduction.
        """
        shot_label = classification.get('shot_label', 'photo')
        genre_label = classification.get('genre_label', 'image')

        if stage == "subject":
            if content_detail == "explicit":
                detail_hint = """EXTRACT WITH MAXIMUM PRECISION:

ETHNICITY & HERITAGE:
- Apparent ethnicity/heritage (e.g., East Asian, South Asian, African, European, Latin, Middle Eastern, Mixed)
- Distinctive ethnic features that should be preserved in reproduction

SKIN - BE EXTREMELY PRECISE:
- Base tone: very fair, fair, light, light-medium, medium, medium-tan, tan, deep tan, brown, deep brown, dark brown, ebony
- Undertone: warm (golden/yellow/peachy), cool (pink/red/blue), neutral, olive
- Surface quality: matte, natural, dewy, oily, luminous
- Any visible details: freckles, beauty marks, texture

FACE SHAPE:
- Exact shape (oval, round, heart, square, oblong, diamond)
- Jawline definition, cheekbone prominence

HEAD & GAZE DIRECTION:
- Head facing: straight at camera, turned left/right (degree), tilted
- Eyes looking: directly at camera, looking left/right/up/down, distant gaze, looking at something specific

EYES - DETAILED:
- Color: precise with variations (e.g., "hazel with green outer ring and amber center")
- Shape: almond, round, hooded, monolid, upturned, downturned, deep-set, prominent
- Size: small, medium, large relative to face
- Gaze direction: where exactly are they looking
- Expression: what emotion the eyes convey
- Eyelids: open wide, relaxed, half-closed, sleepy
- Visible white (sclera): how much visible on each side

EYEBROWS - DETAILED:
- Shape: arched, straight, curved, S-shaped, rounded
- Thickness: thin, medium, thick, bushy
- Grooming: natural, shaped, filled in, feathered
- Position: raised, neutral, furrowed
- Color: match hair or different, any ombre

NOSE - DETAILED:
- Shape: straight, curved, upturned, Roman/aquiline, button, wide, narrow
- Bridge: high, low, flat, prominent
- Tip: rounded, pointed, bulbous, upturned
- Nostrils: visible/hidden, width
- Size relative to face

LIPS - DETAILED:
- Shape: full, thin, heart-shaped, bow-shaped, wide, narrow
- Upper lip: defined cupid's bow, flat, full
- Lower lip: fuller than upper, equal, thinner
- Natural color: pink, mauve, coral, brown tones
- Position: closed, slightly parted, smiling, pouting
- Texture: smooth, slightly chapped

HAIR:
- Exact color with compound descriptors
- Texture: straight, wavy, curly, coily, kinky
- Style, length, volume, shine level

COSMETICS/MAKEUP (if present):
- Foundation: coverage level, finish (matte/dewy/satin)
- Eye makeup: eyeshadow colors and placement, eyeliner style/thickness/wing, mascara intensity, false lashes
- Lip color: exact shade and finish (matte/glossy/satin/sheer)
- Blush/contour: placement and intensity
- Brows: filled/natural, shape enhancement, color
- Highlighter: placement (cheekbones, nose, cupid's bow)
- Any special effects or artistic makeup

BODY: Proportions, build type, skin exposure areas"""
            elif content_detail == "detailed":
                detail_hint = """EXTRACT COMPREHENSIVELY:

ETHNICITY & SKIN:
- Apparent ethnicity/heritage
- Skin tone (specific shade + undertone)
- Surface quality

FACE & FEATURES:
- Shape, key features reflecting heritage
- Expression

HAIR:
- Color (precise), style, texture, length

EYES:
- Color, shape, gaze direction

COSMETICS (if visible):
- Foundation finish
- Eye makeup style and colors
- Lip color and finish
- Any other visible makeup

BODY: Build, proportions, posture"""
            else:
                detail_hint = """EXTRACT KEY FEATURES:
- Apparent ethnicity/heritage
- Skin tone and undertone
- Face shape, features, expression
- Hair color, style, texture
- Eye color and shape
- Basic makeup if visible
- Build type"""

            system_prompt = f"""You are analyzing the SUBJECT in this {shot_label} {genre_label} for accurate reproduction.

{detail_hint}

SKIN TONE PRECISION - Use specific descriptors:
- NOT "fair skin" → "porcelain skin with cool pink undertones"
- NOT "tan skin" → "warm golden-tan skin with olive undertones"
- NOT "dark skin" → "rich deep brown skin with warm undertones"
- NOT "brown skin" → "medium caramel skin with golden undertones"

COLOR PRECISION RULES:
- NOT "brown hair" → "warm chestnut brown with subtle auburn highlights"
- NOT "blue eyes" → "deep sapphire blue with lighter rings around iris"
- NOT "red lipstick" → "classic cherry red lipstick with satin finish"

Output ONLY valid JSON:
{{
  "subject": {{
    "ethnicity": {{
      "apparent_heritage": "<specific ethnicity/heritage>",
      "distinctive_features": ["<heritage-specific features to preserve>"]
    }},
    "skin": {{
      "tone": "<specific shade: porcelain/ivory/fair/light/medium/tan/caramel/brown/deep brown/ebony>",
      "undertone": "<warm/cool/neutral/olive>",
      "surface": "<matte/natural/dewy/luminous>",
      "details": "<freckles, marks, texture>"
    }},
    "face_shape": "<oval, round, heart, square, diamond, oblong>",
    "head_direction": {{
      "facing": "<at camera, turned left/right, profile>",
      "tilt": "<straight, tilted left/right>"
    }},
    "eyes": {{
      "color": "<precise color with variations>",
      "shape": "<almond, round, hooded, monolid, etc>",
      "size": "<small, medium, large>",
      "gaze_direction": "<at camera, looking left/right/up/down>",
      "expression": "<emotion conveyed>",
      "eyelids": "<open wide, relaxed, half-closed>"
    }},
    "eyebrows": {{
      "shape": "<arched, straight, curved>",
      "thickness": "<thin, medium, thick>",
      "grooming": "<natural, shaped, filled>",
      "color": "<color description>"
    }},
    "nose": {{
      "shape": "<straight, upturned, Roman, button>",
      "bridge": "<high, low, flat>",
      "tip": "<rounded, pointed>",
      "size": "<relative to face>"
    }},
    "lips": {{
      "shape": "<full, thin, heart-shaped>",
      "upper_lip": "<cupid's bow, flat, full>",
      "lower_lip": "<description>",
      "natural_color": "<pink, mauve, coral, brown>",
      "position": "<closed, parted, smiling>"
    }},
    "hair": {{
      "color": "<compound color descriptor>",
      "texture": "<straight, wavy, curly, coily>",
      "style": "<description>",
      "length": "<short, medium, long>",
      "shine": "<matte, natural, glossy>"
    }},
    "cosmetics": {{
      "foundation": "<coverage and finish>",
      "eye_makeup": "<shadow colors, liner style, mascara, lashes>",
      "lip_color": "<exact shade and finish>",
      "blush_contour": "<placement and intensity>",
      "brows": "<filled/natural, color>",
      "highlighter": "<placement if visible>",
      "other": "<any other makeup>"
    }},
    "expression": "<overall facial expression>",
    "build": "<body type if visible>",
    "notable_features": ["<any distinctive features>"]
  }}
}}"""
            user_message = "Analyze the subject with precision. Include ethnicity, exact skin tone, facial features (eyes, eyebrows, nose, lips), gaze direction, and all cosmetics."

        elif stage == "pose":
            system_prompt = f"""You are analyzing the POSE and BODY POSITIONING in this {shot_label} for accurate reproduction.

ANALYZE EVERY BODY PART POSITION:

HEAD:
- Tilt: left/right angle, forward/back angle
- Turn: facing direction (camera, left, right, profile)
- Chin: raised, lowered, level

NECK:
- Visibility: fully visible, partially hidden, hidden
- Position: straight, elongated, turned

SHOULDERS:
- Level or angled (which side higher)
- Position: squared to camera, turned, one forward
- Tension: relaxed, raised, pulled back

ARMS (describe EACH arm):
- Position: raised, lowered, bent at elbow, extended, crossed
- Angle: close to body, away from body
- What it's doing: resting, reaching, holding something

HANDS (describe EACH hand):
- Location: on hip, in hair, on face, touching neck, by side, behind back, etc.
- Position: open, closed, relaxed, gripping
- Fingers: spread, together, pointing, curled

PALMS:
- Orientation: facing up, down, forward, inward, toward camera
- Visibility: visible, hidden, partially visible

TORSO:
- Posture: straight, curved, arched back
- Lean: forward, back, left, right
- Twist: any rotation at waist

HIPS:
- Angle: level, tilted (which side up)
- Position: squared, turned, one pushed out
- Weight bearing: which hip carries weight

LEGS (describe EACH leg if visible):
- Position: straight, bent at knee, crossed
- Stance: together, apart, one forward
- Angle: parallel, angled outward/inward

FEET (if visible):
- Position: flat, pointed, flexed
- Direction: forward, angled, one turned out
- Stance width: narrow, normal, wide

WEIGHT DISTRIBUTION:
- Center of gravity: centered, shifted left/right, forward/back
- Supporting leg/side

Output ONLY valid JSON:
{{
  "pose": {{
    "head": {{
      "tilt": "<left/right angle, forward/back>",
      "turn": "<direction facing>",
      "chin": "<raised/lowered/level>"
    }},
    "neck": {{
      "visibility": "<visible/hidden/partial>",
      "position": "<straight/elongated/turned>"
    }},
    "shoulders": {{
      "level": "<level/left higher/right higher>",
      "position": "<squared/turned/one forward>",
      "tension": "<relaxed/tense/raised>"
    }},
    "left_arm": {{
      "position": "<raised/lowered/bent/extended>",
      "doing": "<what it's doing>"
    }},
    "right_arm": {{
      "position": "<raised/lowered/bent/extended>",
      "doing": "<what it's doing>"
    }},
    "left_hand": {{
      "location": "<where it is>",
      "position": "<open/closed/relaxed>",
      "palm": "<up/down/forward/hidden>"
    }},
    "right_hand": {{
      "location": "<where it is>",
      "position": "<open/closed/relaxed>",
      "palm": "<up/down/forward/hidden>"
    }},
    "torso": {{
      "posture": "<straight/curved/arched>",
      "lean": "<direction if any>",
      "twist": "<any rotation>"
    }},
    "hips": {{
      "angle": "<level/tilted>",
      "weight_side": "<which side bears weight>"
    }},
    "left_leg": "<position if visible>",
    "right_leg": "<position if visible>",
    "feet": "<position and stance if visible>",
    "overall_gesture": "<confident/relaxed/dynamic/posed/casual/etc>"
  }}
}}"""
            user_message = "Analyze the exact pose and position of every body part. Be precise about each arm, hand, and leg."

        elif stage == "clothing":
            if content_detail == "explicit":
                detail_hint = """EXTRACT EVERY CLOTHING DETAIL WITH MAXIMUM PRECISION:

GARMENT ANALYSIS:
- Each garment: exact type, color (precise compound descriptor), material, fit
- Material properties: texture, sheen, weight, transparency, stretch
- Fit behavior: clinging, loose, tension points, how fabric drapes

COVERAGE & EXPOSURE ANALYSIS (be precise):
- Neckline: exact type (V-neck depth, scoop, sweetheart, etc.)
- Cleavage: visibility level (none, subtle, moderate, prominent, deep)
- Side exposure: any sideboob visibility, armhole gaps
- Back: coverage level, any cutouts or open back
- Midriff: covered or exposed, crop top length
- Leg visibility: skirt/dress length, slit positions and height
- Thigh exposure: how much visible, any gaps between garments
- Gaps/openings: any spaces in clothing, unbuttoned areas, slits

FABRIC BEHAVIOR:
- How material conforms to body curves
- Visible tension or pull points
- Drape and flow characteristics
- Any sheerness or see-through areas"""
            elif content_detail == "detailed":
                detail_hint = """EXTRACT CLOTHING DETAILS COMPREHENSIVELY:

GARMENT ANALYSIS:
- Each garment: type, color (precise), material, fit
- Material: texture, sheen, behavior

COVERAGE ANALYSIS:
- Neckline type and depth
- Cleavage visibility if present
- Leg/thigh exposure levels
- Back coverage
- Any gaps or openings in clothing

FABRIC BEHAVIOR:
- How it fits the body
- Drape and flow"""
            else:
                detail_hint = """EXTRACT CLOTHING DETAILS:
- Each visible garment: type, color, material
- Fit: how it sits on the body
- Fabric: texture, behavior
- Coverage: general coverage level
- Accessories: type, color, material"""

            system_prompt = f"""You are analyzing CLOTHING and ACCESSORIES in this {shot_label} for accurate reproduction.

{detail_hint}

COLOR PRECISION:
- NOT "black dress" → "deep matte black fitted dress"
- NOT "gold jewelry" → "warm yellow gold pendant with delicate chain"
- NOT "blue jeans" → "medium-wash indigo denim jeans with slight fade"

MATERIAL PRECISION:
- Include: texture, sheen, weight, transparency
- Examples: "lightweight sheer chiffon", "structured cotton twill", "soft jersey knit", "stretchy ribbed cotton"

Output ONLY valid JSON:
{{
  "clothing": {{
    "garments": [
      {{
        "type": "<garment type>",
        "color": "<compound color descriptor>",
        "material": "<fabric type, texture, sheen, weight>",
        "fit": "<how it fits - tight, loose, draped, structured>",
        "details": "<patterns, embellishments, notable features>"
      }}
    ],
    "coverage": {{
      "neckline": "<type and depth>",
      "cleavage": "<none|subtle|moderate|prominent|deep>",
      "side_exposure": "<any sideboob, armhole gaps>",
      "back": "<coverage description>",
      "midriff": "<covered|partially exposed|fully exposed>",
      "legs": "<coverage level, visible length>",
      "thighs": "<visibility level, any gaps>",
      "gaps_openings": "<any slits, unbuttoned areas, spaces>"
    }},
    "fabric_behavior": {{
      "body_conforming": "<how it follows curves>",
      "tension_points": "<where fabric pulls or stretches>",
      "transparency": "<opaque|slightly sheer|sheer|see-through>"
    }},
    "accessories": [
      {{
        "type": "<accessory type>",
        "material": "<material description>",
        "color": "<precise color>",
        "details": "<any notable features>"
      }}
    ],
    "overall_style": "<fashion style descriptor>"
  }}
}}"""
            user_message = "Analyze all clothing with precise materials, coverage levels, and any gaps or exposure areas."

        elif stage == "environment":
            system_prompt = f"""You are analyzing the ENVIRONMENT/BACKGROUND in this {shot_label} for accurate reproduction.

EXTRACT WITH SPATIAL PRECISION:
- SETTING: Indoor/outdoor, specific location type
- BACKGROUND: What's directly behind subject
- MIDGROUND: Elements between subject and background
- FOREGROUND: Any elements in front of subject
- SURFACES: Floor, walls, textures, colors
- PROPS: Any objects, furniture, decorative elements
- DEPTH: How much background is in focus, bokeh quality

SPATIAL RELATIONSHIPS:
- Subject position in frame (centered, rule of thirds, etc)
- Distance relationships between elements
- Depth layers (what's sharp vs blurred)

Output ONLY valid JSON:
{{
  "environment": {{
    "setting_type": "<indoor/outdoor and specific location>",
    "background": {{
      "description": "<what's behind subject>",
      "colors": ["<dominant colors>"],
      "focus": "<sharp, soft, heavily blurred bokeh>"
    }},
    "surfaces": {{
      "floor": "<if visible: material, color>",
      "walls": "<if visible: material, color>",
      "other": "<any other surfaces>"
    }},
    "props": ["<list of visible objects with descriptions>"],
    "spatial": {{
      "subject_position": "<where subject is in frame>",
      "depth_layers": "<description of depth relationships>",
      "composition": "<rule of thirds, centered, etc>"
    }}
  }}
}}"""
            user_message = "Analyze the environment and spatial composition with precision."

        elif stage == "lighting":
            system_prompt = f"""You are analyzing LIGHTING and COLORS in this {genre_label} for accurate reproduction.

LIGHTING ANALYSIS - BE PRECISE:
- KEY LIGHT: Direction (use clock positions: "from 10 o'clock"), intensity, quality
- FILL LIGHT: Presence, ratio to key light
- RIM/HAIR LIGHT: Position, effect on edges
- AMBIENT: Overall environmental light

SHADOW ANALYSIS:
- Direction shadows fall
- Softness (hard edge vs soft gradient)
- Density (deep black vs lifted)

COLOR ANALYSIS:
- Color temperature: warm (golden hour), neutral, cool (blue hour)
- Dominant palette: list the 3-5 main colors
- Color harmony: complementary, analogous, monochromatic
- Saturation levels: muted, natural, vibrant

Output ONLY valid JSON:
{{
  "lighting": {{
    "key_light": {{
      "direction": "<clock position or description>",
      "quality": "<hard, soft, diffused>",
      "intensity": "<low, medium, high>",
      "color_temp": "<warm, neutral, cool with Kelvin estimate>"
    }},
    "fill_light": {{
      "present": <true|false>,
      "ratio": "<e.g., 2:1, 4:1>",
      "source": "<natural, reflector, etc>"
    }},
    "rim_light": {{
      "present": <true|false>,
      "position": "<if present, where>",
      "effect": "<hair glow, edge definition, etc>"
    }},
    "shadows": {{
      "direction": "<where shadows fall>",
      "softness": "<hard edge, soft, very soft>",
      "density": "<deep, medium, lifted>"
    }}
  }},
  "colors": {{
    "temperature": "<overall color temperature>",
    "dominant_palette": ["<color1>", "<color2>", "<color3>"],
    "saturation": "<muted, natural, vibrant>",
    "mood": "<what the colors convey>"
  }}
}}"""
            user_message = "Analyze lighting setup and color palette with technical precision."

        elif stage == "style":
            system_prompt = f"""You are analyzing the PHOTOGRAPHY STYLE of this {genre_label} for accurate reproduction.

TECHNICAL ANALYSIS:
- LENS: Estimated focal length effect (wide distortion, normal, telephoto compression)
- DEPTH OF FIELD: Shallow (subject sharp, background bokeh), medium, deep (everything sharp)
- APERTURE EFFECT: Bokeh quality if visible

COMPOSITION ANALYSIS:
- Framing: How subject fills frame
- Rule of thirds / centered / dynamic
- Leading lines if present
- Negative space usage

STYLE INDICATORS:
- Photography genre style markers
- Post-processing look (natural, edited, film look)
- Overall aesthetic (minimalist, dramatic, soft, editorial)

Output ONLY valid JSON:
{{
  "style": {{
    "technical": {{
      "focal_length": "<wide, normal, telephoto>",
      "depth_of_field": "<shallow, medium, deep>",
      "bokeh_quality": "<if visible: creamy, busy, circular, etc>",
      "sharpness": "<soft, natural, sharp>"
    }},
    "composition": {{
      "framing": "<how subject is framed>",
      "rule": "<rule of thirds, centered, etc>",
      "balance": "<symmetric, asymmetric>",
      "negative_space": "<minimal, moderate, significant>"
    }},
    "aesthetic": {{
      "style": "<photography style name>",
      "processing": "<natural, edited, film, etc>",
      "mood": "<overall visual mood>",
      "genre_markers": ["<style indicators>"]
    }}
  }}
}}"""
            user_message = "Analyze photography style and composition technique."

        else:
            system_prompt = f"Analyze this {genre_label}. Output JSON."
            user_message = "Describe what you see."

        return system_prompt, user_message

    def build_consolidation_prompt(
        self,
        classification: Dict[str, Any],
        stage_results: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Consolidate results from multiple focused analysis stages.
        This is a text-only call (no image needed).
        """
        results_text = json.dumps(stage_results, indent=2)

        system_prompt = """You are consolidating detailed image analysis results for Z-Image prompt generation.

Your task: Merge all analysis stages into a comprehensive, unified attribute structure.

CONSOLIDATION RULES:
1. PRESERVE ALL DETAILS - Don't summarize or lose precision
2. RESOLVE CONFLICTS - If stages disagree, note both observations
3. MAINTAIN PRECISION - Keep compound color descriptors intact
4. ORGANIZE LOGICALLY - Group related attributes together
5. REMOVE DUPLICATES - But keep the more detailed version

Output clean, comprehensive JSON."""

        user_message = f"""Consolidate these analysis results into one unified JSON structure:

{results_text}

Requirements:
1. Include ALL extracted details from every stage
2. Maintain precise color and material descriptors
3. Keep lighting and spatial information
4. Preserve style and composition details
5. Output as clean JSON ready for prompt generation"""

        return system_prompt, user_message

    def get_temperature_modifier(self, stage: str) -> float:
        """Temperature modifiers optimized for accuracy in local models."""
        modifiers = {
            "classification": 0.3,      # Low for accurate classification
            "analysis": 0.4,            # Low for precise extraction
            "focused_analysis": 0.35,   # Very low for consistent detailed JSON
            "consolidation": 0.2,       # Very low for accurate merging
            "composition": 0.5,         # Moderate for natural flow
            "refinement": 0.3,          # Low for focused refinement
        }
        return modifiers.get(stage, 0.4)
