"""
Standalone Prompt Components

Extracted from sid_zimage_prompt_generator_advanced_v2.py for testing without ComfyUI.
"""

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
    "shot_description": "<describe what body parts are visible in THIS image>",
    "subject_pose": "<standing/seated/lying/kneeling/leaning>",
    "subject_position": "<position>",
    "frame_fill_percent": <number>,
    "camera_angle": "<angle>",
    "depth_of_field": "<shallow/medium/deep>",
    "background_blur": "<description of blur level>",
    "prompt_description": "<YOUR framing description - DO NOT copy the example, describe THIS image>"
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
    "prompt_description": "<YOUR ethnicity/demographics description for THIS image - DO NOT copy examples>"
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
    "prompt_description": "<YOUR hair description - START with arrangement, then color/texture/length>"
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


# Detail mode configurations
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
