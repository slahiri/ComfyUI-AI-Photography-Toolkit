"""
SID_ZImagePromptGenerator_Advanced_V2 Node

Component-based image analysis for Z-Image prompt generation.
Analyzes images by breaking them into individual components (face, hair, eyes, etc.)
and generating focused descriptions for each, then assembling into a cohesive prompt.

Key improvements over V1:
- Component-based analysis (focused LLM calls per component)
- Explicit framing/composition in prompts
- Single detail_mode setting (Minimal/Standard/Detailed/Explicit/Raw)
- Better ethnicity and hair style detection
- Structured prompt assembly with framing first
"""

import base64
import io
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from comfy_api.latest import io as comfy_io
import comfy.utils

from .llm_providers.llm_model_type import LLMModelConfig
from .prompt_templates import get_prompt_template_for_provider

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


# Component definitions for analysis
COMPONENTS = {
    "framing": {
        "name": "Framing & Composition",
        "prompt": """Analyze ONLY the framing and composition of this image:

1. COLOR MODE (CRITICAL):
   - Is this a COLOR photo or BLACK AND WHITE / MONOCHROME?
   - If B&W: is it high contrast, soft, sepia-toned?

2. SHOT TYPE (choose one):
   - ECU (Extreme Close-Up): Only eyes/lips visible
   - BCU (Big Close-Up): Face fills frame
   - CU (Close-Up): Head and neck
   - MCU (Medium Close-Up): Head and shoulders, upper chest
   - MS (Medium Shot): Waist up
   - MFS (Medium Full Shot): Knees up
   - FS (Full Shot): Full body, head to toe
   - LS (Long Shot): Full body with environment

3. SUBJECT POSE (CRITICAL):
   - STANDING: Subject is upright on feet
   - SEATED: Subject is sitting (on chair, floor, stool, etc.)
   - LYING: Subject is lying down
   - KNEELING: Subject is on knees
   - LEANING: Subject is leaning against something

4. SUBJECT POSITION: Where in frame? (centered, left third, right third)

5. FRAME FILL: What percentage of frame height does subject fill? (30%, 50%, 70%, 90%)

6. CAMERA ANGLE: Eye level, slightly above, slightly below, dramatic angle?

7. DEPTH OF FIELD: Shallow (blurred background), medium, deep (all in focus)?

Output JSON:
{
    "color_mode": "<color/black_and_white/sepia>",
    "color_style": "<if B&W: high contrast, soft, film noir, etc.>",
    "shot_type": "<code>",
    "shot_description": "<e.g., 'head and shoulders visible, upper chest in frame'>",
    "subject_pose": "<standing/seated/lying/kneeling/leaning>",
    "subject_position": "<position>",
    "frame_fill_percent": <number>,
    "camera_angle": "<angle>",
    "depth_of_field": "<shallow/medium/deep>",
    "background_blur": "<description of blur level>",
    "prompt_description": "<complete framing description for prompt, e.g. 'MCU head and shoulders portrait, subject fills 70% of frame, shallow depth of field with soft bokeh background'>"
}"""
    },

    "ethnicity": {
        "name": "Ethnicity & Demographics",
        "prompt": """Analyze ONLY the subject's ethnicity and demographics:

CRITICAL: Look carefully at facial structure, eye shape, and features before deciding.

1. APPARENT ETHNICITY (be specific - check EACH category):

   ASIAN FEATURES TO CHECK:
   - East Asian (Chinese, Japanese, Korean): epicanthic fold, flatter nose bridge, straight black hair
   - Southeast Asian (Thai, Vietnamese, Filipino): similar to East Asian but often darker skin
   - South Asian (Indian, Pakistani): larger eyes, higher nose bridge, can have lighter or darker skin

   NON-ASIAN FEATURES:
   - European: deeper-set eyes, higher nose bridge, varied hair colors
   - Middle Eastern: can look similar to South Asian or Mediterranean
   - African: varied features by region
   - Latin American/Hispanic: mixed features
   - Mixed heritage: specify which mix

2. KEY DISTINGUISHING FEATURES:
   - Eye shape: Does the eye have an epicanthic fold? Monolid? Double lid?
   - Nose bridge: Flat, medium, or high?
   - Face shape: Round, oval, heart, square?
   - Cheekbone position: High, medium, flat?

3. SKIN TONE (be precise):
   - Base: porcelain, ivory, fair, light, light-medium, medium, medium-tan, tan, caramel, brown, dark brown, deep, ebony
   - Undertone: warm (golden/yellow/peachy), cool (pink/blue), neutral, olive

4. AGE RANGE: child, teen, young adult (20s), adult (30s), middle-aged (40s-50s), senior

5. GENDER PRESENTATION: female, male, androgynous

Output JSON:
{
    "ethnicity": "<specific ethnicity>",
    "ethnicity_features": "<distinctive features that led to this conclusion>",
    "eye_shape_details": "<epicanthic fold present? monolid? etc.>",
    "skin_tone_base": "<base tone>",
    "skin_tone_undertone": "<undertone>",
    "skin_description": "<full description for prompt>",
    "age_range": "<range>",
    "gender": "<gender>",
    "prompt_description": "<complete ethnicity and demographics description for prompt, e.g. 'South Asian woman in her 20s with warm medium skin tone and olive undertones'>"
}"""
    },

    "hair": {
        "name": "Hair Analysis",
        "prompt": """Analyze ONLY the hair in this image:

CRITICAL: Start your description with the OVERALL ARRANGEMENT (how hair is styled on top of head) BEFORE describing any flowing parts.

1. HAIR ARRANGEMENT (CRITICAL - choose one):
   - UPDO/BUN: Hair pulled up and secured (bun, chignon, top knot) - NO hair hanging loose
   - PONYTAIL: Hair gathered and hanging from one point
   - BRAIDED: Any braid style (French, Dutch, fishtail, box braids)
   - HALF-UP: Top/crown pulled back and secured, bottom portion hangs down
   - LOOSE/DOWN: ALL hair flowing freely, nothing secured or pulled back
   - PINNED/STYLED: Elaborately styled with pins/accessories
   - SHORT: Too short to be styled up

2. CROWN/TOP DESCRIPTION (CRITICAL):
   - Is hair at the crown pulled back, slicked, volumized, or loose?
   - Is there a visible parting? Center, side, none?
   - Are any pins, clips, or elastic visible?

3. HAIR COLOR (be specific):
   - Base color: black, dark brown, medium brown, light brown, dark blonde, etc.
   - Highlights/lowlights: describe if present

4. HAIR TEXTURE: straight, wavy, curly, coily, kinky

5. HAIR LENGTH: pixie, short, chin-length, shoulder-length, mid-back, waist-length

6. HAIR CONDITION: glossy, matte, frizzy, sleek, volumized

Output JSON:
{
    "arrangement": "<updo/ponytail/braided/half-up/loose/pinned/short>",
    "crown_description": "<how hair is styled at top/crown of head>",
    "arrangement_details": "<specific style description>",
    "base_color": "<color>",
    "highlights": "<if any>",
    "texture": "<texture>",
    "length": "<length>",
    "condition": "<condition>",
    "prompt_description": "<START with arrangement (e.g. 'half-up styled hair with...'), THEN describe rest>"
}"""
    },

    "face": {
        "name": "Face Shape & Position",
        "prompt": """Analyze ONLY the face shape and positioning:

1. FACE SHAPE: oval, round, heart, square, oblong, diamond, rectangle

2. FACE ANGLE:
   - Frontal (looking straight at camera)
   - 3/4 left (turned slightly left, showing right side more)
   - 3/4 right (turned slightly right, showing left side more)
   - Profile left (side view, facing left)
   - Profile right (side view, facing right)

3. HEAD TILT: level, tilted left, tilted right, chin up, chin down

4. FACE PROPORTIONS: Any notable features? High cheekbones, defined jaw, etc.

Output JSON:
{
    "shape": "<shape>",
    "angle": "<angle>",
    "angle_degrees": "<approximate degrees from frontal>",
    "tilt": "<tilt>",
    "notable_features": "<features>",
    "prompt_description": "<description for prompt>"
}"""
    },

    "eyes": {
        "name": "Eyes Analysis",
        "prompt": """Analyze ONLY the eyes in this image:

1. EYE COLOR (be very specific):
   - Base: dark brown, medium brown, light brown, hazel, green, blue-green, blue, gray, black
   - Variations: lighter ring around pupil? Darker limbal ring? Flecks of other colors?

2. EYE SHAPE: almond, round, hooded, monolid, upturned, downturned, deep-set, prominent

3. GAZE DIRECTION:
   - Direct at camera
   - Looking left/right/up/down
   - Looking at something specific

4. EYE EXPRESSION: soft, intense, playful, serious, sultry, tired, alert

5. EYE MAKEUP (if any):
   - Eyeshadow: colors, placement
   - Eyeliner: style (winged, tight-line, smudged), color
   - Mascara: natural, dramatic, false lashes?
   - Brows: shape, thickness, groomed/natural

Output JSON:
{
    "color": "<specific color>",
    "color_details": "<variations, flecks, rings>",
    "shape": "<shape>",
    "gaze": "<direction>",
    "expression": "<expression>",
    "makeup_eyeshadow": "<if any>",
    "makeup_eyeliner": "<if any>",
    "makeup_lashes": "<description>",
    "eyebrows": "<description>",
    "prompt_description": "<full description for prompt>"
}"""
    },

    "nose_lips": {
        "name": "Nose & Lips",
        "prompt": """Analyze ONLY the nose and lips:

NOSE:
1. Shape: straight, Roman/aquiline, button, upturned, hooked, flat, wide, narrow
2. Bridge: high, medium, low
3. Tip: rounded, pointed, upturned
4. Size relative to face: small, medium, large

LIPS:
1. Shape: full, thin, heart-shaped, wide, bow-shaped
2. Upper vs lower lip: balanced, fuller upper, fuller lower
3. Cupid's bow: defined, subtle, flat
4. Natural color: pink, nude, red, berry, coral
5. State: closed, slightly parted, smiling, pursed
6. Lip makeup: lipstick color and finish if worn (matte, glossy, satin)

Output JSON:
{
    "nose_shape": "<shape>",
    "nose_details": "<other details>",
    "lip_shape": "<shape>",
    "lip_fullness": "<description>",
    "lip_color": "<natural or makeup color>",
    "lip_state": "<state>",
    "lip_finish": "<if makeup: matte/glossy/satin>",
    "prompt_description": "<combined description for prompt>"
}"""
    },

    "body_pose": {
        "name": "Body & Pose",
        "prompt": """Analyze ONLY the visible body and pose:

1. VISIBLE BODY PARTS: List what's visible (head, neck, shoulders, arms, hands, torso, legs, feet)

2. POSTURE: straight/upright, relaxed, slouched, leaning

3. SHOULDER POSITION: level, one raised, squared to camera, angled

4. ARM POSITIONS (if visible):
   - Left arm: position and gesture
   - Right arm: position and gesture

5. HAND POSITIONS (if visible):
   - Left hand: where placed, gesture
   - Right hand: where placed, gesture

6. BODY ANGLE: Facing camera directly, angled left, angled right, torso twist

7. WEIGHT DISTRIBUTION: centered, on left leg, on right leg

Output JSON:
{
    "visible_parts": ["<list>"],
    "posture": "<posture>",
    "shoulders": "<description>",
    "left_arm": "<if visible>",
    "right_arm": "<if visible>",
    "left_hand": "<if visible>",
    "right_hand": "<if visible>",
    "body_angle": "<angle>",
    "overall_pose": "<description>",
    "prompt_description": "<pose description for prompt>"
}"""
    },

    "clothing": {
        "name": "Clothing Analysis",
        "prompt": """Analyze ONLY the clothing ACTUALLY VISIBLE in this image.

CRITICAL RULES:
- ONLY describe what you can ACTUALLY SEE in the image
- Do NOT invent, assume, or imagine parts of garments that are not visible
- If you can only see the top of a dress, just describe the visible neckline/bodice
- Do NOT describe cutouts, midriff, sleeves, or features that are NOT in frame
- If clothing is cut off by the frame, say "visible portion only" or "cropped at [location]"

For EACH visible garment, describe:

1. GARMENT TYPE: dress, top, blouse, etc.
   - If unsure if dress or top (cropped at frame), say "dress or top"

2. NECKLINE (what you can see):
   - sweetheart, V-neck, scoop, strapless, off-shoulder, halter, etc.

3. COLOR: Be specific

4. MATERIAL/FABRIC: silk, satin, cotton, crepe, etc.

5. FIT: fitted, loose, structured, etc.

6. VISIBLE FEATURES ONLY:
   - What straps/sleeves can you actually see?
   - Is midriff actually visible or is image cropped above it?

Output JSON:
{
    "garments": [
        {
            "type": "<type or 'dress or top - cropped'>",
            "neckline": "<visible neckline style>",
            "color": "<specific color>",
            "material": "<material>",
            "fit": "<fit>",
            "visible_features": "<only what you can see>",
            "cropped_at": "<where image cuts off garment, or 'fully visible'>"
        }
    ],
    "overall_style": "<formal/casual/elegant/etc>",
    "prompt_description": "<describe ONLY visible portions - do not invent details>"
}"""
    },

    "accessories": {
        "name": "Accessories",
        "prompt": """Analyze ONLY the accessories/jewelry in this image:

For EACH visible accessory:

1. TYPE: necklace, earrings, bracelet, ring, watch, hair accessory, glasses, hat, etc.

2. STYLE:
   - Necklace: choker, pendant, chain, statement, layered
   - Earrings: studs, hoops, dangles, drops, chandeliers
   - Other: describe style

3. MATERIAL: gold, silver, platinum, crystals, diamonds, pearls, beads, fabric

4. SIZE: delicate, medium, statement/large

5. DETAILS: specific design elements, stones, patterns

Output JSON:
{
    "accessories": [
        {
            "type": "<type>",
            "style": "<style>",
            "material": "<material>",
            "color": "<color>",
            "size": "<size>",
            "details": "<specific details>",
            "position": "<where on body>"
        }
    ],
    "prompt_description": "<accessories description for prompt>"
}"""
    },

    "lighting": {
        "name": "Lighting & Environment",
        "prompt": """Analyze ONLY the lighting and background:

LIGHTING:
1. Type: natural (sun, window), artificial (studio), mixed
2. Direction: front, side left, side right, back, top, bottom (use clock positions)
3. Quality: soft/diffused, hard/harsh, dramatic
4. Color temperature: warm (golden), neutral, cool (blue)
5. Key light position: Where is main light coming from?
6. Fill light: Is there fill? Shadow density?
7. Rim/hair light: Any backlighting on edges?

SHADOWS:
1. Density: minimal, moderate, dramatic
2. Direction: where do shadows fall?
3. Softness: soft edges, hard edges

BACKGROUND:
1. Type: solid color, gradient, environment, bokeh
2. Color: specific color description
3. Blur level: sharp, slightly soft, soft bokeh, very blurred
4. Elements: any visible elements in background?

Output JSON:
{
    "light_type": "<type>",
    "light_direction": "<direction with clock position>",
    "light_quality": "<quality>",
    "color_temperature": "<temperature>",
    "shadow_density": "<density>",
    "shadow_direction": "<direction>",
    "background_type": "<type>",
    "background_color": "<color>",
    "background_blur": "<blur level>",
    "background_elements": "<any elements>",
    "prompt_description": "<lighting and background for prompt>"
}"""
    },

    "intimate_apparel": {
        "name": "Intimate Apparel Details",
        "prompt": """Analyze ALL intimate apparel/lingerie in this image in detail:

For EACH visible intimate garment, describe:

1. BRA/BRALETTE/BUSTIER (if visible):
   - Style: push-up, balconette, bralette, sports bra, bustier, corset, bandeau, triangle
   - Cup coverage: full coverage, demi, plunge, sheer
   - Straps: thick, thin, spaghetti, strapless, halter, racerback, convertible
   - Material: lace, satin, silk, cotton, mesh, velvet, leather
   - Color and pattern: solid, printed, embroidered
   - Details: underwire, boning, padding, hooks, clasps, ribbons, bows
   - Trim: lace trim, scalloped edges, piping

2. PANTIES/BOTTOMS (if visible):
   - Style: thong, G-string, bikini, hipster, boyshort, high-waist, cheeky
   - Coverage: minimal, moderate, full
   - Material: lace, satin, silk, cotton, mesh
   - Color and pattern
   - Waistband: thin elastic, wide band, string ties
   - Details: bows, ribbons, cutouts

3. GARTER BELT/SUSPENDERS (if visible):
   - Style: classic, modern, high-waist
   - Number of straps: 4, 6, 8
   - Material: lace, satin, elastic
   - Clip style: metal, plastic
   - Color

4. STOCKINGS/HOSIERY (if visible):
   - Type: thigh-high, pantyhose, knee-high
   - Pattern: solid, fishnet (small/large), seamed, patterned
   - Denier: sheer, semi-opaque, opaque
   - Top band: lace, silicone, plain
   - Color

5. BODYSUIT/TEDDY (if visible):
   - Style: plunge, high-neck, halter
   - Coverage: sheer, opaque, cut-outs
   - Closure: snap, hook, open

Output JSON:
{
    "bra": {
        "style": "<style>",
        "coverage": "<coverage>",
        "straps": "<strap type>",
        "material": "<material>",
        "color": "<color>",
        "details": "<trim, embellishments>",
        "description": "<full description>"
    },
    "bottoms": {
        "style": "<style>",
        "coverage": "<coverage>",
        "material": "<material>",
        "color": "<color>",
        "details": "<details>",
        "description": "<full description>"
    },
    "garter": {
        "present": true/false,
        "style": "<style>",
        "straps": "<number>",
        "color": "<color>",
        "description": "<full description>"
    },
    "stockings": {
        "present": true/false,
        "type": "<type>",
        "pattern": "<pattern>",
        "color": "<color>",
        "description": "<full description>"
    },
    "bodysuit": {
        "present": true/false,
        "description": "<if present>"
    },
    "prompt_description": "<complete intimate apparel description for prompt>"
}"""
    },

    "tattoos": {
        "name": "Tattoos & Body Art",
        "prompt": """Analyze ALL visible tattoos and body art in this image:

For EACH visible tattoo, describe:

1. LOCATION on body:
   - Specific placement (e.g., "left upper arm", "right ribcage", "lower back", "chest", "thigh", "ankle", "wrist", "shoulder blade", "neck", "behind ear")

2. SIZE:
   - Small (coin-sized), Medium (palm-sized), Large (hand-sized+), Sleeve, Full back piece

3. STYLE:
   - Traditional/Old School
   - Neo-traditional
   - Realism/Photorealistic
   - Watercolor
   - Blackwork/Tribal
   - Fine line/Minimalist
   - Japanese/Irezumi
   - Geometric
   - Script/Lettering
   - Dotwork

4. SUBJECT MATTER:
   - What does the tattoo depict? (flowers, animals, portraits, symbols, text, abstract)

5. COLORS:
   - Black and gray only
   - Full color (list main colors)
   - Red and black
   - Single color accent

6. VISIBILITY:
   - Fully visible, partially visible, glimpse only

If NO tattoos visible, state "no visible tattoos".

Output JSON:
{
    "has_tattoos": true/false,
    "tattoos": [
        {
            "location": "<body location>",
            "size": "<size>",
            "style": "<tattoo style>",
            "subject": "<what it depicts>",
            "colors": "<colors>",
            "visibility": "<how visible>",
            "description": "<full description>"
        }
    ],
    "prompt_description": "<all tattoos description for prompt, or 'no visible tattoos'>"
}"""
    },

    # Subject detection - runs FIRST to determine what components to use
    "subject_detection": {
        "name": "Subject Detection",
        "prompt": """Analyze this image and identify the PRIMARY SUBJECT TYPE.

Choose ONE primary subject type:

1. WOMAN - Adult female human as primary subject
2. MAN - Adult male human as primary subject
3. COUPLE - Two people (romantic/intimate context)
4. GROUP - Multiple people (3+)
5. CHILD - Child or teenager as primary subject
6. ANIMAL - Animal as primary subject (pet, wildlife)
7. VEHICLE - Car, motorcycle, plane, boat, etc.
8. PRODUCT - Commercial product photography
9. FOOD - Food/beverage photography
10. ARCHITECTURE - Building, interior, structure
11. LANDSCAPE - Nature, scenery, outdoor environment
12. ABSTRACT - Abstract art, patterns, textures
13. OTHER - None of the above

Also identify:
- Is there ANY human visible (even partially)?
- What is the main focus of the image?

Output JSON:
{
    "subject_type": "<one of the types above>",
    "has_human": true/false,
    "human_gender": "<female/male/multiple/none>",
    "human_visible_parts": "<what parts visible: face, body, hands only, silhouette, etc.>",
    "main_focus": "<brief description of main subject>",
    "secondary_elements": "<other notable elements>"
}"""
    },

    # Object/Vehicle description
    "object_description": {
        "name": "Object/Vehicle Description",
        "prompt": """Describe this object/vehicle in detail:

1. TYPE: What is it exactly? (make, model if identifiable)
2. COLOR: Primary and secondary colors
3. CONDITION: New, worn, vintage, damaged?
4. STYLE: Modern, classic, futuristic, industrial?
5. NOTABLE FEATURES: Distinctive elements, modifications, details
6. POSITION/ANGLE: How is it positioned in frame?
7. MOTION: Static, moving, motion blur?
8. ENVIRONMENT: Where is it? What surrounds it?

Output JSON:
{
    "type": "<specific type>",
    "make_model": "<if identifiable>",
    "colors": {"primary": "<color>", "secondary": "<color>"},
    "condition": "<condition>",
    "style": "<style>",
    "notable_features": ["<feature1>", "<feature2>"],
    "position": "<position in frame>",
    "motion": "<static/moving/motion blur>",
    "environment": "<surroundings>",
    "prompt_description": "<full description for prompt>"
}"""
    },

    # Scenery/Landscape description
    "scenery_description": {
        "name": "Scenery/Landscape Description",
        "prompt": """Describe this landscape/scenery in detail:

1. TYPE: Beach, mountain, forest, urban, desert, etc.
2. TIME OF DAY: Dawn, morning, noon, afternoon, sunset, dusk, night
3. WEATHER: Clear, cloudy, rainy, foggy, stormy, snowy
4. SEASON: Spring, summer, autumn, winter
5. KEY ELEMENTS: Water, trees, buildings, sky features
6. COLORS: Dominant color palette
7. MOOD: Serene, dramatic, mysterious, vibrant
8. DEPTH: Foreground, midground, background elements

Output JSON:
{
    "type": "<landscape type>",
    "time_of_day": "<time>",
    "weather": "<conditions>",
    "season": "<season>",
    "key_elements": ["<element1>", "<element2>"],
    "color_palette": ["<color1>", "<color2>"],
    "mood": "<mood>",
    "foreground": "<description>",
    "midground": "<description>",
    "background": "<description>",
    "prompt_description": "<full description for prompt>"
}"""
    }
}


# Subject type to component mapping
SUBJECT_COMPONENTS = {
    "WOMAN": {
        "Standard": ["framing", "ethnicity", "hair", "face", "eyes", "body_pose", "clothing", "lighting"],
        "Detailed": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
        "Extreme": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
    },
    "MAN": {
        "Standard": ["framing", "ethnicity", "hair", "face", "eyes", "body_pose", "clothing", "lighting"],
        "Detailed": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "tattoos", "accessories", "lighting"],
        "Extreme": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "tattoos", "accessories", "lighting"],
    },
    "COUPLE": {
        "Standard": ["framing", "ethnicity", "hair", "face", "body_pose", "clothing", "lighting"],
        "Detailed": ["framing", "ethnicity", "hair", "face", "eyes", "body_pose", "clothing", "intimate_apparel", "accessories", "lighting"],
        "Extreme": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
    },
    "GROUP": {
        "Standard": ["framing", "lighting"],
        "Detailed": ["framing", "clothing", "lighting"],
        "Extreme": ["framing", "clothing", "accessories", "lighting"],
    },
    "VEHICLE": {
        "Standard": ["framing", "object_description", "lighting"],
        "Detailed": ["framing", "object_description", "lighting"],
        "Extreme": ["framing", "object_description", "lighting"],
    },
    "LANDSCAPE": {
        "Standard": ["framing", "scenery_description", "lighting"],
        "Detailed": ["framing", "scenery_description", "lighting"],
        "Extreme": ["framing", "scenery_description", "lighting"],
    },
    "ANIMAL": {
        "Standard": ["framing", "object_description", "lighting"],
        "Detailed": ["framing", "object_description", "lighting"],
        "Extreme": ["framing", "object_description", "lighting"],
    },
    "PRODUCT": {
        "Standard": ["framing", "object_description", "lighting"],
        "Detailed": ["framing", "object_description", "lighting"],
        "Extreme": ["framing", "object_description", "lighting"],
    },
    "ARCHITECTURE": {
        "Standard": ["framing", "scenery_description", "lighting"],
        "Detailed": ["framing", "scenery_description", "lighting"],
        "Extreme": ["framing", "scenery_description", "lighting"],
    },
    "OTHER": {
        "Standard": ["framing", "object_description", "lighting"],
        "Detailed": ["framing", "object_description", "lighting"],
        "Extreme": ["framing", "object_description", "lighting"],
    },
}


# Detail mode configurations - 3 levels only
DETAIL_MODES = {
    "Standard": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "body_pose", "clothing", "lighting"],
        "description": "Balanced analysis - main features",
        "include_measurements": False,
    },
    "Detailed": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
        "description": "Comprehensive analysis - all features including intimate apparel and tattoos",
        "include_measurements": True,
    },
    "Extreme": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
        "description": "Maximum detail - all components, explicit descriptions, raw output",
        "include_measurements": True,
        "include_exposure": True,
        "raw_mode": True,
    },
}


class SID_ZImagePromptGenerator_Advanced_V2(comfy_io.ComfyNode):
    """
    Advanced V2 Z-Image prompt generator with component-based analysis.

    Breaks down images into individual components (hair, eyes, face, etc.)
    and analyzes each separately for maximum accuracy.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema."""

        return comfy_io.Schema(
            node_id="SID_ZImagePromptGenerator_Advanced_V2",
            display_name="SID Z-Image Prompt Generator V2",
            category="SID Photography Toolkit/Z-Image",
            description="Advanced V2: Component-based analysis for accurate prompts. Analyzes hair, face, eyes, clothing separately.",
            is_output_node=True,
            inputs=[
                comfy_io.Image.Input(
                    "image",
                    tooltip="Input image to analyze"
                ),

                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="Connect LLM provider node (e.g., SID_Anthropic_LLM)"
                ),

                # Prompt generation mode
                comfy_io.Combo.Input(
                    "prompt_mode",
                    options=["Single Shot", "Agentic (Reasoning Models)"],
                    default="Single Shot",
                    tooltip="Single Shot=fast single call, Agentic=deep analysis (requires reasoning-capable model like Claude 3.5/o1)"
                ),

                # Single detail mode - 3 levels
                comfy_io.Combo.Input(
                    "detail_mode",
                    options=["Standard", "Detailed", "Extreme"],
                    default="Standard",
                    tooltip="Standard=balanced, Detailed=comprehensive, Extreme=maximum detail with raw output"
                ),

                # Optional user guidance
                comfy_io.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    tooltip="Optional: Guide the analysis or add specific requirements"
                ),

                # Focus toggles
                comfy_io.Boolean.Input(
                    "include_lighting",
                    default=True,
                    tooltip="Include lighting and background description"
                ),

                comfy_io.Boolean.Input(
                    "include_pose",
                    default=True,
                    tooltip="Include body pose details"
                ),

                # Generation settings
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Random seed for reproducibility (use controls after input to randomize/increment/decrement)"
                ),

                comfy_io.Boolean.Input(
                    "cache_results",
                    default=True,
                    tooltip="Cache results to avoid repeated API calls"
                ),
            ],
            outputs=[
                comfy_io.String.Output(
                    "prompt",
                    display_name="prompt",
                    tooltip="Generated Z-Image prompt"
                ),
                comfy_io.Int.Output(
                    "width",
                    display_name="width",
                    tooltip="Image width in pixels"
                ),
                comfy_io.Int.Output(
                    "height",
                    display_name="height",
                    tooltip="Image height in pixels"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        llm_model: LLMModelConfig,
        prompt_mode: str,
        detail_mode: str,
        user_prompt: str,
        include_lighting: bool,
        include_pose: bool,
        seed: int,
        cache_results: bool,
    ) -> comfy_io.NodeOutput:
        """Execute the prompt generation pipeline."""

        # Get image dimensions early for error returns
        img_tensor = image[0]
        img_height, img_width = img_tensor.shape[0], img_tensor.shape[1]

        # Check if model supports reasoning for agentic mode
        supports_reasoning = llm_model.supports_reasoning

        # Guardrail: Log and validate max_tokens against model limits
        max_tokens = llm_model.max_tokens
        model_max = (llm_model.extra_params or {}).get("model_max_output_tokens", "unknown")
        print(f"[V2] Model: {llm_model.model} (provider: {llm_model.provider})")
        print(f"[V2] Max tokens: {max_tokens} (model limit: {model_max})")
        print(f"[V2] Prompt Mode: {prompt_mode}")

        # Warn if max_tokens is very low for prompt generation
        if max_tokens < 300:
            print(f"[V2] Warning: max_tokens={max_tokens} is very low, may result in truncated prompts")

        try:
            if prompt_mode == "Agentic (Reasoning Models)":
                # Check if model supports reasoning
                if not supports_reasoning:
                    print(f"[V2] WARNING: Agentic mode selected but model doesn't support reasoning!")
                    print(f"[V2] Falling back to Single Shot mode. Enable reasoning in LLM node or use Claude 3.5/o1.")
                    # Fall back to single shot
                    return cls._execute_single_shot_pipeline(
                        image, llm_model, detail_mode, user_prompt,
                        include_lighting, include_pose, seed, cache_results
                    )
                # Agentic approach for reasoning models
                print(f"[V2] Using AGENTIC mode (reasoning enabled)")
                return cls._execute_agentic_pipeline(
                    image, llm_model, detail_mode, user_prompt,
                    include_lighting, include_pose, seed, cache_results
                )
            else:
                # Single Shot - fast single call approach
                print(f"[V2] Using SINGLE SHOT mode")
                return cls._execute_single_shot_pipeline(
                    image, llm_model, detail_mode, user_prompt,
                    include_lighting, include_pose, seed, cache_results
                )
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[V2] {error_msg}")
            import traceback
            traceback.print_exc()
            return comfy_io.NodeOutput(error_msg, img_width, img_height, ui={"text": (error_msg,)})

    @classmethod
    def _execute_pipeline(
        cls,
        image,
        llm_model: LLMModelConfig,
        detail_mode: str,
        user_prompt: str,
        include_lighting: bool,
        include_pose: bool,
        seed: int,
        cache_results: bool,
    ) -> comfy_io.NodeOutput:
        """
        [DEPRECATED] Multi-step iterative pipeline.

        This method is deprecated in favor of _execute_single_shot_pipeline.
        Kept for backward compatibility but no longer used by default.
        """

        start_time = time.time()
        log_lines = []

        def log(msg: str):
            log_lines.append(msg)
            print(f"[V2] {msg}")

        log("=" * 60)
        log("SID Z-Image Prompt Generator V2 - Component Analysis")
        log("=" * 60)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Detail Mode: {detail_mode}")
        if user_prompt and user_prompt.strip():
            log(f"User Request: '{user_prompt.strip()[:50]}...' (will be incorporated into analysis)")
        log("")

        # Get image dimensions
        img_tensor = image[0]
        height, width = img_tensor.shape[0], img_tensor.shape[1]
        log(f"Image: {width}x{height}")

        # Convert image to base64
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Check for max_image_size in extra_params (GGUF optimization)
        max_image_size = llm_model.extra_params.get("max_image_size") if llm_model.extra_params else None
        if max_image_size and max(width, height) > max_image_size:
            # Resize while preserving aspect ratio
            if width > height:
                new_width = max_image_size
                new_height = int(height * (max_image_size / width))
            else:
                new_height = max_image_size
                new_width = int(width * (max_image_size / height))
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            log(f"Image resized: {width}x{height} -> {new_width}x{new_height} (max_size={max_image_size})")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        base64_image = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        # Get LLM client
        client = cls._get_client(llm_model)
        model = llm_model.model
        provider = llm_model.provider

        log(f"Provider: {provider}")
        log(f"Model: {model}")
        log("")

        # Initialize component results dictionary
        component_results = {}

        # STEP 1: Subject Detection - determine what type of subject is in the image
        log("Step 1: Subject Detection...")
        subject_result = cls._analyze_component(
            client, model, base64_image,
            "subject_detection", COMPONENTS["subject_detection"]["prompt"],
            {}
        )

        subject_type = subject_result.get("subject_type", "OTHER").upper()
        has_human = subject_result.get("has_human", False)
        human_gender = subject_result.get("human_gender", "none")

        log(f"  Subject Type: {subject_type}")
        log(f"  Has Human: {has_human}")
        if has_human:
            log(f"  Human Gender: {human_gender}")
        log(f"  Main Focus: {subject_result.get('main_focus', 'unknown')}")
        log("")

        # Store subject detection result
        component_results["subject_detection"] = subject_result

        # STEP 2: Get components based on subject type
        # Map some subject types to standard categories
        subject_mapping = {
            "WOMAN": "WOMAN",
            "MAN": "MAN",
            "COUPLE": "COUPLE",
            "GROUP": "GROUP",
            "CHILD": "OTHER",  # Use generic for safety
            "ANIMAL": "ANIMAL",
            "VEHICLE": "VEHICLE",
            "PRODUCT": "PRODUCT",
            "FOOD": "PRODUCT",
            "ARCHITECTURE": "ARCHITECTURE",
            "LANDSCAPE": "LANDSCAPE",
            "ABSTRACT": "OTHER",
            "OTHER": "OTHER",
        }

        mapped_subject = subject_mapping.get(subject_type, "OTHER")

        # Get components for this subject type and detail mode
        if mapped_subject in SUBJECT_COMPONENTS:
            components_to_analyze = SUBJECT_COMPONENTS[mapped_subject].get(
                detail_mode, SUBJECT_COMPONENTS[mapped_subject]["Standard"]
            ).copy()
        else:
            components_to_analyze = SUBJECT_COMPONENTS["OTHER"]["Standard"].copy()

        # Get mode config for other settings
        mode_config = DETAIL_MODES.get(detail_mode, DETAIL_MODES["Standard"])

        # Adjust based on toggles
        if not include_lighting and "lighting" in components_to_analyze:
            components_to_analyze.remove("lighting")
        if not include_pose and "body_pose" in components_to_analyze:
            components_to_analyze.remove("body_pose")

        log(f"Step 2: Component Analysis for {mapped_subject}")
        log(f"  Components: {len(components_to_analyze)}")
        log(f"  {', '.join(components_to_analyze)}")
        log("")

        # Progress bar (+2 for subject detection and assembly)
        total_steps = len(components_to_analyze) + 2
        pbar = comfy.utils.ProgressBar(total_steps)
        pbar.update(1)  # Subject detection done

        # Analyze each component
        for i, component_key in enumerate(components_to_analyze):
            component = COMPONENTS[component_key]
            log(f"[{i+1}/{len(components_to_analyze)}] Analyzing: {component['name']}")

            comp_start = time.time()

            result = cls._analyze_component(
                client, model, base64_image,
                component_key, component["prompt"],
                mode_config, user_prompt
            )

            component_results[component_key] = result

            # Log key findings
            if result:
                if component_key == "framing":
                    log(f"    Shot: {result.get('shot_type', 'unknown')}, Fill: {result.get('frame_fill_percent', '?')}%")
                elif component_key == "ethnicity":
                    log(f"    Ethnicity: {result.get('ethnicity', 'unknown')}, Skin: {result.get('skin_tone_base', 'unknown')}")
                elif component_key == "hair":
                    log(f"    Style: {result.get('arrangement', 'unknown')}, Color: {result.get('base_color', 'unknown')}")
                elif component_key == "eyes":
                    log(f"    Color: {result.get('color', 'unknown')}, Shape: {result.get('shape', 'unknown')}")

            log(f"    Duration: {time.time() - comp_start:.1f}s")
            pbar.update(1)

        log("")
        log("Assembling prompt...")

        # Assemble the final prompt
        prompt = cls._assemble_prompt(component_results, user_prompt, mode_config)

        # Calculate stats for logging
        total_time = time.time() - start_time
        word_count = len(prompt.split())
        token_estimate = len(prompt) // 4

        log(f"Final prompt: {word_count} words, ~{token_estimate} tokens")
        log(f"Total time: {total_time:.1f}s")

        pbar.update(1)

        return comfy_io.NodeOutput(prompt, width, height, ui={"text": (prompt,)})

    # =========================================================================
    # SINGLE SHOT PIPELINE - Fast single-call approach (replaces iterative)
    # =========================================================================

    @classmethod
    def _execute_single_shot_pipeline(
        cls,
        image,
        llm_model: LLMModelConfig,
        detail_mode: str,
        user_prompt: str,
        include_lighting: bool,
        include_pose: bool,
        seed: int,
        cache_results: bool,
    ) -> comfy_io.NodeOutput:
        """
        Single-shot pipeline - generates prompt in one comprehensive call.

        Uses our component knowledge condensed into a single, well-structured prompt.
        Much faster than iterative, works well with all vision models.
        """

        start_time = time.time()

        def log(msg: str):
            print(f"[V2-SINGLE] {msg}")

        log("=" * 60)
        log("SID Z-Image Prompt Generator V2 - SINGLE SHOT MODE")
        log("=" * 60)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Detail Mode: {detail_mode}")

        # Get image dimensions
        img_tensor = image[0]
        height, width = img_tensor.shape[0], img_tensor.shape[1]
        log(f"Image: {width}x{height}")

        # Convert image to base64
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Check for max_image_size in extra_params
        max_image_size = llm_model.extra_params.get("max_image_size") if llm_model.extra_params else None
        if max_image_size and max(width, height) > max_image_size:
            if width > height:
                new_width = max_image_size
                new_height = int(height * (max_image_size / width))
            else:
                new_height = max_image_size
                new_width = int(width * (max_image_size / height))
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            log(f"Image resized: {width}x{height} -> {new_width}x{new_height}")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        base64_image = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        # Get LLM client
        client = cls._get_client(llm_model)
        model = llm_model.model

        log(f"Provider: {llm_model.provider}")
        log(f"Model: {model}")

        # Progress bar
        pbar = comfy.utils.ProgressBar(2)

        # Build comprehensive single-shot prompt based on detail mode
        system_prompt = cls._build_single_shot_system_prompt(detail_mode, user_prompt)
        analysis_prompt = cls._build_single_shot_analysis_prompt(
            detail_mode, user_prompt, include_lighting, include_pose
        )

        log("Generating prompt (single call)...")
        pbar.update(1)

        # Make single LLM call
        try:
            if hasattr(client, 'messages'):
                # Anthropic
                response = client.messages.create(
                    model=model,
                    max_tokens=llm_model.max_tokens,
                    temperature=llm_model.temperature,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image,
                                }
                            },
                            {
                                "type": "text",
                                "text": analysis_prompt
                            }
                        ]
                    }]
                )
                prompt = response.content[0].text
            else:
                # OpenAI-style
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=llm_model.max_tokens,
                    temperature=llm_model.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": analysis_prompt
                                }
                            ]
                        }
                    ]
                )
                prompt = response.choices[0].message.content

        except Exception as e:
            log(f"Error: {e}")
            raise

        # Clean up the prompt (remove any markdown or extra formatting)
        prompt = cls._clean_single_shot_output(prompt)

        # Add user focus if provided
        if user_prompt and user_prompt.strip():
            prompt = f"[USER FOCUS: {user_prompt.strip()}] {prompt}"

        pbar.update(1)

        # Calculate stats
        total_time = time.time() - start_time
        word_count = len(prompt.split())
        token_estimate = len(prompt) // 4

        log(f"Final prompt: {word_count} words, ~{token_estimate} tokens")
        log(f"Total time: {total_time:.1f}s")
        log("=" * 60)

        return comfy_io.NodeOutput(prompt, width, height, ui={"text": (prompt,)})

    @classmethod
    def _build_single_shot_system_prompt(cls, detail_mode: str, user_prompt: str) -> str:
        """Build the system prompt for single-shot generation."""

        base_system = """You are an expert visual analyst specializing in generating precise, flowing prompts for text-to-image AI models.

Your task is to analyze the provided image and generate a cohesive, descriptive prompt that captures all visual elements.

CRITICAL RULES:
1. Describe ONLY what you can see - do not invent or assume details
2. Start with framing/composition, then subject, then details, then lighting/background
3. Use natural, flowing language - not a list of tags
4. Be specific about colors, textures, and positions
5. Output ONLY the prompt text - no explanations, headers, or formatting"""

        if user_prompt and user_prompt.strip():
            base_system += f"""

IMPORTANT USER REQUEST: Pay special attention to: "{user_prompt.strip()}"
Ensure your prompt captures and emphasizes any details related to this request."""

        return base_system

    @classmethod
    def _build_single_shot_analysis_prompt(
        cls,
        detail_mode: str,
        user_prompt: str,
        include_lighting: bool,
        include_pose: bool
    ) -> str:
        """Build the analysis prompt based on detail mode."""

        if detail_mode == "Standard":
            return """Analyze this image and generate a detailed prompt covering:

1. FRAMING: Shot type (close-up, medium, full), camera angle, depth of field
2. SUBJECT: Ethnicity, gender, age, skin tone, facial features
3. HAIR: Arrangement (updo, loose, braided), color, texture, length
4. FACE: Shape, expression, gaze direction, makeup if visible
5. CLOTHING: Garment types, colors, materials, style (ONLY describe what's visible)
""" + ("""6. POSE: Body position, posture, hand placement
""" if include_pose else "") + ("""7. LIGHTING: Direction, quality, shadows, background
""" if include_lighting else "") + """
Generate a flowing, natural paragraph that combines these elements into a cohesive prompt.
Output ONLY the prompt text."""

        elif detail_mode == "Detailed":
            return """Analyze this image comprehensively and generate a detailed prompt covering:

1. FRAMING & COMPOSITION:
   - Color mode (color or black & white)
   - Shot type (ECU, CU, MCU, MS, FS, LS)
   - Subject pose (standing, seated, lying, kneeling)
   - Camera angle and depth of field
   - How much of frame subject fills

2. SUBJECT DEMOGRAPHICS:
   - Ethnicity (be specific: East Asian, South Asian, European, etc.)
   - Skin tone with undertones (warm, cool, olive)
   - Age range and gender

3. HAIR (CRITICAL - describe arrangement first):
   - Is it up (bun, ponytail, braided) or down (loose)?
   - Crown/top styling
   - Color, texture, length

4. FACIAL FEATURES:
   - Face shape and angle
   - Eye color, shape, expression
   - Makeup details if present
   - Lips and nose

5. BODY & POSE:
   - Visible body parts
   - Posture and body angle
   - Hand positions if visible

6. CLOTHING & INTIMATE APPAREL:
   - All visible garments with specific details
   - Materials, colors, style
   - Any lingerie/intimate apparel visible

7. ACCESSORIES & TATTOOS:
   - Jewelry (type, material, placement)
   - Visible tattoos (location, style, subject)

8. LIGHTING & ENVIRONMENT:
   - Light direction and quality
   - Shadow characteristics
   - Background type and color

Generate a rich, flowing description that combines ALL these elements.
Output ONLY the prompt text - no headers or formatting."""

        else:  # Extreme
            return """Generate an EXTREMELY detailed, comprehensive prompt for this image.

Analyze EVERY visible element with maximum precision:

FRAMING: Exact shot type, subject position in frame, frame fill percentage, camera angle, depth of field, background blur level. Note if black & white.

SUBJECT: Precise ethnicity (not just "Asian" - specify East/South/Southeast Asian), exact skin tone with undertones, age range, gender presentation.

HAIR: ARRANGEMENT FIRST (updo/ponytail/braided/half-up/loose), then crown styling, color with any highlights, texture, length, condition.

FACE: Shape, angle from frontal, head tilt, all notable proportions.

EYES: Exact color with variations/rings, shape, gaze direction, expression, all makeup details (shadow colors, liner style, lash type).

NOSE & LIPS: Shapes, proportions, lip color/state, any makeup.

BODY: All visible parts, exact posture, shoulder position, arm positions, hand positions if visible, body angle, weight distribution.

CLOTHING: Every visible garment in detail - type, neckline, color, material, fit, visible features. Note if cropped by frame.

INTIMATE APPAREL: If visible - bra style/coverage/straps/material, bottoms style, garters, stockings (pattern, denier, color).

TATTOOS: Every visible tattoo - exact location, size, style, subject matter, colors.

ACCESSORIES: All jewelry with type, style, material, size, placement.

LIGHTING: Direction (use clock positions), quality, color temperature, shadow density/direction, rim lighting.

BACKGROUND: Type, color, blur level, any visible elements.

Generate an extremely detailed, flowing narrative prompt.
Output ONLY the raw prompt text - no formatting, headers, or explanations."""

    @classmethod
    def _clean_single_shot_output(cls, text: str) -> str:
        """Clean up the LLM output to get just the prompt."""
        import re

        # Remove common prefixes
        prefixes_to_remove = [
            r"^Here'?s? (?:the |a )?(?:detailed |comprehensive )?prompt[:\s]*",
            r"^(?:The )?prompt[:\s]*",
            r"^Output[:\s]*",
            r"^Description[:\s]*",
        ]
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove markdown formatting
        text = re.sub(r'```[a-z]*\n?', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # Headers

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', ' ', text)  # Multiple newlines to space
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()

        return text

    @classmethod
    def _get_client(cls, llm_model: LLMModelConfig):
        """Get the appropriate LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            # Set longer timeout for vision requests (can be slow)
            timeout = httpx.Timeout(timeout=600.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)
        elif provider == "openai" or provider == "openai_compatible":
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None
            )
        elif provider == "grok":
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key,
                base_url="https://api.x.ai/v1"
            )
        elif provider == "gguf":
            # Local GGUF model - create LocalGGUFClient from extra_params
            from .llm_providers.sid_gguf_llm import LocalGGUFClient
            extra = llm_model.extra_params or {}
            return LocalGGUFClient(
                model_path=extra.get("model_path", ""),
                mmproj_path=extra.get("mmproj_path"),
                chat_format=extra.get("chat_format", "llava-1-5"),
                n_ctx=extra.get("n_ctx", 4096),
                n_gpu_layers=extra.get("n_gpu_layers", -1),
                verbose=False,
            )
        elif provider == "qwenvl":
            # QwenVL model using HuggingFace transformers
            from .llm_providers.sid_qwenvl_llm import QwenVLClient
            extra = llm_model.extra_params or {}
            return QwenVLClient(
                model_name=llm_model.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                attention_mode=extra.get("attention_mode", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                top_p=extra.get("top_p", 0.9),
                repetition_penalty=extra.get("repetition_penalty", 1.2),
                num_beams=extra.get("num_beams", 1),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def _analyze_component(
        cls,
        client,
        model: str,
        base64_image: str,
        component_key: str,
        component_prompt: str,
        mode_config: dict,
        user_prompt: str = "",
    ) -> dict:
        """Analyze a single component of the image."""

        # Build system prompt with user requirements if provided
        system_prompt = """You are an expert visual analyst. Analyze ONLY the specific aspect requested.
Be precise and specific in your descriptions. Output valid JSON only."""

        # If user has special requirements, incorporate them into analysis
        if user_prompt and user_prompt.strip():
            system_prompt += f"""

IMPORTANT USER REQUEST: The user has specifically asked for the following. Pay special attention to this aspect and ensure your analysis addresses it thoroughly:
"{user_prompt.strip()}"

When analyzing, prioritize capturing any details related to the user's request."""

        try:
            # Determine if Anthropic or OpenAI-style API
            if hasattr(client, 'messages'):
                # Anthropic
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image,
                                }
                            },
                            {
                                "type": "text",
                                "text": component_prompt
                            }
                        ]
                    }]
                )
                response_text = response.content[0].text
            else:
                # OpenAI-style
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": component_prompt
                                }
                            ]
                        }
                    ]
                )
                response_text = response.choices[0].message.content

            # Parse JSON response
            return cls._parse_json_response(response_text)

        except Exception as e:
            print(f"[V2] Error analyzing {component_key}: {e}")
            return {"error": str(e)}

    @classmethod
    def _parse_json_response(cls, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        import re

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1)

        # Try to find raw JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return empty dict if parsing fails
        return {}

    @classmethod
    def _assemble_prompt(
        cls,
        components: dict,
        user_prompt: str,
        mode_config: dict,
    ) -> str:
        """Assemble the final prompt from component analyses."""

        sections = []

        # 0. COLOR MODE (if B&W, add at very start)
        color_prefix = ""
        if "framing" in components and components["framing"]:
            f = components["framing"]
            color_mode = f.get("color_mode", "color")
            if color_mode and color_mode.lower() in ["black_and_white", "monochrome", "b&w", "bw"]:
                color_style = f.get("color_style", "")
                color_prefix = f"black and white {color_style} photograph, ".strip()

        # 1. FRAMING (always first)
        if "framing" in components and components["framing"]:
            f = components["framing"]
            desc = f.get("prompt_description", "")
            if desc:
                # Add color prefix if B&W
                if color_prefix:
                    desc = color_prefix + desc
                sections.append(desc)
            else:
                # Fallback to manual construction
                shot_desc = f.get("shot_description", "portrait")
                frame_fill = f.get("frame_fill_percent", 70)
                dof = f.get("depth_of_field", "shallow")
                bg_blur = f.get("background_blur", "soft bokeh")
                subject_pose = f.get("subject_pose", "")

                # Build framing text with pose
                framing_text = f"{f.get('shot_type', 'MCU')} {shot_desc}"
                if subject_pose and subject_pose.lower() != "standing":
                    framing_text += f", {subject_pose} pose"
                framing_text += f", subject fills {frame_fill}% of frame, {dof} depth of field with {bg_blur} background"

                # Add color prefix if B&W
                if color_prefix:
                    framing_text = color_prefix + framing_text
                sections.append(framing_text)

        # 2. ETHNICITY & SUBJECT
        if "ethnicity" in components and components["ethnicity"]:
            e = components["ethnicity"]
            desc = e.get("prompt_description", "")
            if desc:
                sections.append(desc)
            else:
                # Fallback to manual construction
                ethnicity = e.get("ethnicity", "")
                gender = e.get("gender", "person")
                skin = e.get("skin_description", e.get("skin_tone_base", ""))

                subject_text = f"{ethnicity} {gender}"
                if skin:
                    subject_text += f" with {skin}"
                sections.append(subject_text)

        # 3. FACE
        if "face" in components and components["face"]:
            f = components["face"]
            desc = f.get("prompt_description", "")
            if desc:
                sections.append(desc)
            else:
                shape = f.get("shape", "")
                angle = f.get("angle", "")
                if shape or angle:
                    sections.append(f"{shape} face, {angle}".strip(", "))

        # 4. HAIR (critical - emphasize arrangement first)
        if "hair" in components and components["hair"]:
            h = components["hair"]
            desc = h.get("prompt_description", "")
            if desc:
                sections.append(desc)
            else:
                arrangement = h.get("arrangement", "")
                crown = h.get("crown_description", "")
                color = h.get("base_color", "")
                texture = h.get("texture", "")
                # Build hair description emphasizing arrangement
                if arrangement in ["updo", "half-up", "ponytail", "braided"]:
                    hair_text = f"{arrangement} {color} {texture} hair"
                    if crown:
                        hair_text += f" with {crown}"
                else:
                    hair_text = f"{color} {texture} hair"
                sections.append(hair_text)

        # 5. EYES
        if "eyes" in components and components["eyes"]:
            e = components["eyes"]
            desc = e.get("prompt_description", "")
            if desc:
                sections.append(desc)
            else:
                color = e.get("color", "")
                shape = e.get("shape", "")
                gaze = e.get("gaze", "")
                sections.append(f"{color} {shape} eyes, {gaze}".strip(", "))

        # 6. NOSE & LIPS
        if "nose_lips" in components and components["nose_lips"]:
            nl = components["nose_lips"]
            desc = nl.get("prompt_description", "")
            if desc:
                sections.append(desc)

        # 7. BODY POSE
        if "body_pose" in components and components["body_pose"]:
            bp = components["body_pose"]
            desc = bp.get("prompt_description", "")
            if desc:
                sections.append(desc)

        # 8. CLOTHING
        if "clothing" in components and components["clothing"]:
            c = components["clothing"]
            desc = c.get("prompt_description", "")
            if desc:
                sections.append(desc)

        # 8b. INTIMATE APPAREL (detailed lingerie description)
        if "intimate_apparel" in components and components["intimate_apparel"]:
            ia = components["intimate_apparel"]
            desc = ia.get("prompt_description", "")
            if desc:
                sections.append(desc)

        # 8c. TATTOOS
        if "tattoos" in components and components["tattoos"]:
            t = components["tattoos"]
            has_tattoos = t.get("has_tattoos", False)
            if has_tattoos:
                desc = t.get("prompt_description", "")
                if desc and desc.lower() != "no visible tattoos":
                    sections.append(desc)

        # 9. ACCESSORIES
        if "accessories" in components and components["accessories"]:
            a = components["accessories"]
            desc = a.get("prompt_description", "")
            if desc:
                sections.append(desc)

        # 9b. OBJECT/VEHICLE DESCRIPTION (for non-human subjects)
        if "object_description" in components and components["object_description"]:
            od = components["object_description"]
            desc = od.get("prompt_description", "")
            if desc:
                sections.append(desc)

        # 9c. SCENERY/LANDSCAPE DESCRIPTION (for landscape/architecture)
        if "scenery_description" in components and components["scenery_description"]:
            sd = components["scenery_description"]
            desc = sd.get("prompt_description", "")
            if desc:
                sections.append(desc)

        # 10. LIGHTING & BACKGROUND (always last)
        if "lighting" in components and components["lighting"]:
            l = components["lighting"]
            desc = l.get("prompt_description", "")
            if desc:
                sections.append(desc)
            else:
                light_dir = l.get("light_direction", "")
                light_qual = l.get("light_quality", "")
                bg_color = l.get("background_color", "")
                if light_dir or light_qual:
                    sections.append(f"{light_qual} {light_dir} lighting".strip())
                if bg_color:
                    sections.append(f"{bg_color} background")

        # Combine sections
        if mode_config.get("raw_mode"):
            # Raw mode: just concatenate everything
            prompt = ". ".join(s for s in sections if s)
        else:
            # Normal mode: flow naturally
            prompt = ", ".join(s for s in sections if s)
            # Clean up
            prompt = prompt.replace(", ,", ",").replace("  ", " ")

        # Add user prompt if provided - structure it prominently at the beginning
        # This ensures the user's special requirements are emphasized in the final prompt
        if user_prompt and user_prompt.strip():
            user_requirement = user_prompt.strip()
            # Structure the user requirement as a primary emphasis
            # The LLM analysis should have already incorporated these details,
            # but we also add them at the front to ensure they're prioritized in generation
            prompt = f"[USER FOCUS: {user_requirement}] {prompt}"

        return prompt.strip()

    @classmethod
    def _generate_recommendations(
        cls,
        width: int,
        height: int,
        word_count: int,
        token_estimate: int,
        subject_type: str,
        detail_mode: str,
    ) -> str:
        """Generate recommendations for optimal Z-Image output."""

        recommendations = []
        warnings = []
        tips = []

        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 1
        total_pixels = width * height
        megapixels = total_pixels / 1_000_000

        # Z-Image optimal dimensions (common SDXL resolutions)
        optimal_resolutions = {
            "1:1": (1024, 1024),
            "4:3": (1152, 896),
            "3:4": (896, 1152),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
        }

        # Determine closest aspect ratio
        if 0.9 <= aspect_ratio <= 1.1:
            closest_ratio = "1:1"
        elif 1.2 <= aspect_ratio <= 1.4:
            closest_ratio = "4:3"
        elif 0.7 <= aspect_ratio <= 0.85:
            closest_ratio = "3:4"
        elif aspect_ratio >= 1.7:
            closest_ratio = "16:9"
        elif aspect_ratio <= 0.6:
            closest_ratio = "9:16"
        elif 1.4 <= aspect_ratio <= 1.6:
            closest_ratio = "3:2"
        elif 0.6 <= aspect_ratio <= 0.75:
            closest_ratio = "2:3"
        else:
            closest_ratio = "custom"

        optimal_w, optimal_h = optimal_resolutions.get(closest_ratio, (1024, 1024))

        # Image dimension recommendations
        if width < 512 or height < 512:
            warnings.append("⚠️  LOW RESOLUTION: Image is below 512px. Consider upscaling for better results.")
        elif width > 2048 or height > 2048:
            recommendations.append(f"📐 RESIZE: Consider resizing to {optimal_w}x{optimal_h} for optimal Z-Image processing.")
        else:
            tips.append(f"✓ Resolution ({width}x{height}) is acceptable.")

        # Aspect ratio recommendation
        if closest_ratio == "custom":
            recommendations.append(f"📐 ASPECT RATIO: Non-standard ratio ({aspect_ratio:.2f}). Consider cropping to standard ratio.")
        else:
            tips.append(f"✓ Aspect ratio ~{closest_ratio} detected.")

        # Megapixel check
        if megapixels > 4:
            recommendations.append(f"📉 HIGH MEGAPIXELS: {megapixels:.1f}MP detected. Resize to ~1MP for faster processing.")

        # Prompt length recommendations
        if word_count < 50:
            recommendations.append("📝 SHORT PROMPT: Consider using 'Detailed' mode for more descriptive prompts.")
        elif word_count > 300:
            warnings.append("⚠️  LONG PROMPT: Prompt exceeds 300 words. Some details may be ignored by Z-Image.")
        else:
            tips.append(f"✓ Prompt length ({word_count} words) is optimal.")

        # Token recommendations
        if token_estimate > 500:
            warnings.append(f"⚠️  HIGH TOKENS: ~{token_estimate} tokens. Consider trimming for efficiency.")

        # Subject-specific tips
        if subject_type == "WOMAN":
            tips.append("👤 Portrait detected - ensure face is well-lit and in focus for best results.")
        elif subject_type == "VEHICLE":
            tips.append("🚗 Vehicle detected - Z-Image works best with clean backgrounds.")
        elif subject_type == "LANDSCAPE":
            tips.append("🏞️ Landscape detected - consider 16:9 aspect ratio for cinematic output.")

        # Detail mode tips
        if detail_mode == "Standard" and subject_type in ["WOMAN", "MAN"]:
            tips.append("💡 TIP: Use 'Detailed' mode for intimate apparel and tattoo detection.")

        # Build output
        output_lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║  RECOMMENDATIONS FOR OPTIMAL Z-IMAGE OUTPUT              ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Current:  {width}x{height} ({closest_ratio}, {megapixels:.1f}MP){' '*(28-len(str(width))-len(str(height))-len(closest_ratio)-len(f'{megapixels:.1f}'))}║",
            f"║  Optimal:  {optimal_w}x{optimal_h}{' '*(44-len(str(optimal_w))-len(str(optimal_h)))}║",
            "╠══════════════════════════════════════════════════════════╣",
        ]

        if warnings:
            output_lines.append("║  WARNINGS                                                ║")
            for w in warnings:
                # Truncate long warnings
                w_short = w[:56] if len(w) > 56 else w
                output_lines.append(f"║  {w_short:<56} ║")
            output_lines.append("╠══════════════════════════════════════════════════════════╣")

        if recommendations:
            output_lines.append("║  RECOMMENDATIONS                                         ║")
            for r in recommendations:
                r_short = r[:56] if len(r) > 56 else r
                output_lines.append(f"║  {r_short:<56} ║")
            output_lines.append("╠══════════════════════════════════════════════════════════╣")

        if tips:
            output_lines.append("║  TIPS                                                    ║")
            for t in tips:
                t_short = t[:56] if len(t) > 56 else t
                output_lines.append(f"║  {t_short:<56} ║")

        output_lines.append("╚══════════════════════════════════════════════════════════╝")

        return "\n".join(output_lines)

    # =========================================================================
    # AGENTIC PIPELINE - For reasoning-capable models (ISOLATED from iterative)
    # =========================================================================

    @classmethod
    def _execute_agentic_pipeline(
        cls,
        image,
        llm_model: LLMModelConfig,
        detail_mode: str,
        user_prompt: str,
        include_lighting: bool,
        include_pose: bool,
        seed: int,
        cache_results: bool,
    ) -> comfy_io.NodeOutput:
        """
        Agentic pipeline for reasoning-capable models.

        Uses extended thinking / reasoning to analyze all components in a single
        comprehensive call, reducing API costs while leveraging model capabilities.

        This method is COMPLETELY ISOLATED from _execute_pipeline.
        """

        start_time = time.time()
        log_lines = []

        def log(msg: str):
            log_lines.append(msg)
            print(f"[V2-AGENTIC] {msg}")

        log("=" * 60)
        log("SID Z-Image Prompt Generator V2 - AGENTIC MODE")
        log("=" * 60)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Detail Mode: {detail_mode}")
        log(f"Model: {llm_model.model} (reasoning enabled)")
        if user_prompt and user_prompt.strip():
            log(f"User Request: '{user_prompt.strip()[:50]}...' (HIGH PRIORITY in analysis)")
        log("")

        # Get image dimensions
        img_tensor = image[0]
        height, width = img_tensor.shape[0], img_tensor.shape[1]
        log(f"Image: {width}x{height}")

        # Convert image to base64
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Check for max_image_size in extra_params (GGUF optimization)
        max_image_size = llm_model.extra_params.get("max_image_size") if llm_model.extra_params else None
        if max_image_size and max(width, height) > max_image_size:
            # Resize while preserving aspect ratio
            if width > height:
                new_width = max_image_size
                new_height = int(height * (max_image_size / width))
            else:
                new_height = max_image_size
                new_width = int(width * (max_image_size / height))
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            log(f"Image resized: {width}x{height} -> {new_width}x{new_height} (max_size={max_image_size})")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        base64_image = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        # Get LLM client
        client = cls._get_client(llm_model)
        model = llm_model.model
        provider = llm_model.provider

        # Progress bar (2 steps: subject detection + comprehensive analysis)
        pbar = comfy.utils.ProgressBar(3)

        # STEP 1: Subject Detection (same as iterative, to determine components)
        log("Step 1: Subject Detection...")
        subject_result = cls._analyze_component(
            client, model, base64_image,
            "subject_detection", COMPONENTS["subject_detection"]["prompt"],
            {}
        )

        subject_type = subject_result.get("subject_type", "OTHER").upper()
        has_human = subject_result.get("has_human", False)

        log(f"  Subject Type: {subject_type}")
        log(f"  Main Focus: {subject_result.get('main_focus', 'unknown')}")
        pbar.update(1)

        # STEP 2: Determine components to analyze
        subject_mapping = {
            "WOMAN": "WOMAN", "MAN": "MAN", "COUPLE": "COUPLE", "GROUP": "GROUP",
            "CHILD": "OTHER", "ANIMAL": "ANIMAL", "VEHICLE": "VEHICLE",
            "PRODUCT": "PRODUCT", "FOOD": "PRODUCT", "ARCHITECTURE": "ARCHITECTURE",
            "LANDSCAPE": "LANDSCAPE", "ABSTRACT": "OTHER", "OTHER": "OTHER",
        }
        mapped_subject = subject_mapping.get(subject_type, "OTHER")

        # Get components for this subject type and detail mode
        if mapped_subject in SUBJECT_COMPONENTS:
            components_to_analyze = SUBJECT_COMPONENTS[mapped_subject].get(
                detail_mode, SUBJECT_COMPONENTS[mapped_subject]["Standard"]
            ).copy()
        else:
            components_to_analyze = SUBJECT_COMPONENTS["OTHER"]["Standard"].copy()

        # Get mode config
        mode_config = DETAIL_MODES.get(detail_mode, DETAIL_MODES["Standard"])

        # Adjust based on toggles
        if not include_lighting and "lighting" in components_to_analyze:
            components_to_analyze.remove("lighting")
        if not include_pose and "body_pose" in components_to_analyze:
            components_to_analyze.remove("body_pose")

        log(f"Step 2: Building comprehensive analysis prompt...")
        log(f"  Components: {len(components_to_analyze)}")
        log(f"  {', '.join(components_to_analyze)}")

        # STEP 3: Build comprehensive agentic prompt from COMPONENTS
        agentic_prompt = cls._build_agentic_prompt(
            subject_type, mapped_subject, detail_mode, components_to_analyze, user_prompt
        )
        pbar.update(1)

        # STEP 4: Single comprehensive analysis call with reasoning
        log("Step 3: Comprehensive analysis (single reasoning call)...")
        log("  This may take longer but uses fewer API calls...")

        analysis_start = time.time()
        component_results = cls._call_reasoning_llm(
            client, model, base64_image, agentic_prompt, llm_model
        )
        analysis_time = time.time() - analysis_start

        log(f"  Analysis completed in {analysis_time:.1f}s")

        # Add subject detection to results
        component_results["subject_detection"] = subject_result

        # Log key findings
        if "framing" in component_results:
            f = component_results["framing"]
            log(f"  Framing: {f.get('shot_type', 'unknown')}, {f.get('subject_pose', 'unknown')}")
        if "ethnicity" in component_results:
            e = component_results["ethnicity"]
            log(f"  Subject: {e.get('ethnicity', 'unknown')} {e.get('gender', 'person')}")
        if "hair" in component_results:
            h = component_results["hair"]
            log(f"  Hair: {h.get('arrangement', 'unknown')} {h.get('base_color', '')}")

        pbar.update(1)

        # STEP 5: Assemble prompt (reuse existing method)
        log("")
        log("Assembling prompt...")
        prompt = cls._assemble_prompt(component_results, user_prompt, mode_config)

        # Calculate stats
        total_time = time.time() - start_time
        word_count = len(prompt.split())
        token_estimate = len(prompt) // 4

        log(f"Final prompt: {word_count} words, ~{token_estimate} tokens")
        log(f"Total time: {total_time:.1f}s (vs ~{len(components_to_analyze) * 3}s iterative)")
        log(f"API calls: 2 (vs {len(components_to_analyze) + 1} iterative)")
        log("=" * 60)

        return comfy_io.NodeOutput(prompt, width, height, ui={"text": (prompt,)})

    @classmethod
    def _build_agentic_prompt(
        cls,
        subject_type: str,
        mapped_subject: str,
        detail_mode: str,
        components_to_analyze: List[str],
        user_prompt: str = "",
    ) -> str:
        """
        Build comprehensive analysis prompt dynamically from COMPONENTS dictionary.
        This reuses the existing component definitions - no duplication.
        """

        # Build component specifications from existing COMPONENTS dict
        component_specs = []
        for comp_key in components_to_analyze:
            if comp_key in COMPONENTS:
                comp = COMPONENTS[comp_key]
                component_specs.append(f"""
### {comp['name'].upper()} ({comp_key})

{comp['prompt']}
""")

        components_section = "\n".join(component_specs)

        # Build user request section if provided
        user_request_section = ""
        if user_prompt and user_prompt.strip():
            user_request_section = f"""

## CRITICAL: USER'S SPECIAL REQUEST

The user has specifically asked for the following. This is a HIGH PRIORITY requirement:

"{user_prompt.strip()}"

**You MUST**:
1. Pay special attention to this aspect throughout your analysis
2. Ensure your component analyses capture details related to this request
3. Make sure the final prompt_description fields emphasize any elements matching this request
4. Place relevant details for this request PROMINENTLY in your descriptions

"""

        # Build the comprehensive agentic prompt
        agentic_prompt = f"""You are an expert visual analyst for Z-Image prompt generation.
Use your reasoning capabilities to methodically analyze this image.
{user_request_section}
## ANALYSIS CONTEXT

- **Subject Type**: {subject_type}
- **Detail Mode**: {detail_mode}
- **Components to Analyze**: {', '.join(components_to_analyze)}

## INSTRUCTIONS

1. Examine the overall image first to understand composition and framing
2. For EACH component listed below, provide detailed analysis
3. Think through each component carefully before providing your analysis
4. Return ALL analyses in a single comprehensive JSON object
5. {"PRIORITIZE capturing any details related to the user's special request above" if user_prompt else "Be thorough in your analysis"}

## CRITICAL: OUTPUT FORMAT

Return a single JSON object containing all component analyses. Each key should be the component name (e.g., "framing", "ethnicity", "hair") with its analysis as the value.

Example structure:
```json
{{
    "framing": {{ ... framing analysis ... }},
    "ethnicity": {{ ... ethnicity analysis ... }},
    "hair": {{ ... hair analysis ... }},
    ... etc for each component ...
}}
```

## COMPONENT SPECIFICATIONS

Analyze EACH of the following components. Follow the JSON structure specified for each:

{components_section}

## FINAL REMINDER

- Analyze ALL {len(components_to_analyze)} components listed above
- Return a SINGLE JSON object containing all analyses
- Be specific and detailed in your descriptions
- Do NOT invent details you cannot see in the image
"""

        return agentic_prompt

    @classmethod
    def _call_reasoning_llm(
        cls,
        client,
        model: str,
        base64_image: str,
        agentic_prompt: str,
        llm_model: LLMModelConfig,
    ) -> dict:
        """
        Call LLM with reasoning/extended thinking enabled.
        Handles both Anthropic (extended thinking) and OpenAI (o1 series).
        """

        provider = llm_model.provider.lower()

        try:
            if provider == "anthropic" and hasattr(client, 'messages'):
                # Anthropic with extended thinking
                response = client.messages.create(
                    model=model,
                    max_tokens=16000,  # Higher for comprehensive analysis
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 10000  # Allow extensive thinking
                    },
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image,
                                }
                            },
                            {
                                "type": "text",
                                "text": agentic_prompt
                            }
                        ]
                    }]
                )

                # Extract text from response (may have thinking blocks)
                response_text = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        response_text = block.text
                        break

            else:
                # OpenAI o1 series or other reasoning models
                response = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=16000,  # o1 uses max_completion_tokens
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": agentic_prompt
                                }
                            ]
                        }
                    ]
                )
                response_text = response.choices[0].message.content

            # Parse the comprehensive JSON response
            return cls._parse_json_response(response_text)

        except Exception as e:
            print(f"[V2-AGENTIC] Error in reasoning call: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
