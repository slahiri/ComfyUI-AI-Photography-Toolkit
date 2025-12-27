"""Z-Image Canonical Scene Model.

Based on comprehensive research from:
- https://www.atlabs.ai/blog/ultimate-z-image-prompting-guide
- https://zimage.net/blog/z-image-prompting-masterclass
- https://www.z-image.win/blog/z-image-prompt-formula-6-part-template
- https://z-image.vip/blog/z-image-prompt-engineering-masterclass

Z-Image uses a 6-part prompt structure:
Subject + Scene + Composition + Lighting + Style + Constraints

Key insight: Z-Image prefers NATURAL LANGUAGE over "tag soup".
Use full sentences describing relationships, not comma-separated keywords.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


class SceneType(Enum):
    """Primary scene type categories for Z-Image prompts."""
    # People-focused
    PORTRAIT = "portrait"           # Head/shoulders, facial focus
    FASHION = "fashion"             # Full body, outfit emphasis
    ACTION = "action"               # Dynamic motion shots
    GROUP = "group"                 # Multiple people

    # Object-focused
    PRODUCT = "product"             # Commercial product photography
    FOOD = "food"                   # Food photography
    STILL_LIFE = "still_life"       # Artistic object arrangements

    # Environment-focused
    LANDSCAPE = "landscape"         # Wide environmental shots
    ARCHITECTURE = "architecture"   # Buildings, interiors
    STREET = "street"               # Urban photography

    # Stylized
    CINEMATIC = "cinematic"         # Film-style compositions
    EDITORIAL = "editorial"         # Magazine/fashion editorial
    ANIME = "anime"                 # Anime/illustration styles
    VINTAGE = "vintage"             # Retro/film aesthetics
    CYBERPUNK = "cyberpunk"         # Sci-fi/futuristic

    # Special
    TEXT_OVERLAY = "text_overlay"   # Images with text elements
    STORYBOARD = "storyboard"       # Multi-panel layouts


class PromptCategory(Enum):
    """The 6 canonical prompt categories."""
    SUBJECT = "subject"         # Who/what is the focus
    SCENE = "scene"             # Environment and context
    COMPOSITION = "composition" # Framing, angle, layout
    LIGHTING = "lighting"       # Light direction, quality, mood
    STYLE = "style"             # Artistic direction, medium
    CONSTRAINTS = "constraints" # Rules, safety, artifacts


# =============================================================================
# VOCABULARY DICTIONARIES
# =============================================================================

SUBJECT_VOCABULARY = {
    # Person descriptors
    "age": [
        "young child", "teenager", "young adult in their 20s",
        "adult in their 30s", "middle-aged", "elderly", "senior"
    ],
    "ethnicity": [
        "Asian", "African", "European", "Latin American", "Middle Eastern",
        "South Asian", "Pacific Islander", "mixed heritage"
    ],
    "build": [
        "slim", "athletic", "muscular", "curvy", "average build",
        "petite", "tall and slender", "broad-shouldered"
    ],
    "expression": [
        "confident smile", "neutral expression", "contemplative gaze",
        "joyful laugh", "serious expression", "gentle smile",
        "determined look", "playful smirk", "pensive expression"
    ],
    "hair_texture": [
        "straight silky hair", "wavy hair", "tight curls", "coily hair",
        "windswept hair", "sleek hair", "tousled hair", "braided hair"
    ],
    "hair_color": [
        "jet black", "dark brown", "chestnut", "auburn", "copper red",
        "blonde", "platinum blonde", "silver grey", "ombre"
    ],
    "skin_tone": [
        "warm ivory", "cool beige", "golden tan", "olive", "deep brown",
        "porcelain", "sun-kissed", "fair with freckles"
    ],
    "skin_texture": [
        "smooth skin", "realistic skin texture", "visible pores",
        "weathered skin", "dewy skin", "matte skin", "natural imperfections"
    ],
}

SCENE_VOCABULARY = {
    "location_indoor": [
        "modern studio", "cozy living room", "professional office",
        "industrial loft", "minimalist space", "vintage cafe",
        "art gallery", "elegant restaurant", "home kitchen"
    ],
    "location_outdoor": [
        "city street", "urban alley", "beach at sunset", "mountain vista",
        "forest clearing", "rooftop terrace", "garden path", "park bench",
        "busy marketplace", "quiet countryside"
    ],
    "time_of_day": [
        "golden hour", "blue hour", "midday sun", "overcast afternoon",
        "night scene", "dawn", "dusk", "midnight"
    ],
    "weather": [
        "clear sky", "rainy", "foggy", "misty", "snowy", "overcast",
        "stormy", "sunny with clouds", "hazy"
    ],
    "atmosphere": [
        "calm and peaceful", "energetic and vibrant", "mysterious",
        "romantic", "melancholic", "dramatic", "serene", "tense"
    ],
    "background": [
        "blurred background", "bokeh lights", "clean white background",
        "gradient background", "textured wall", "natural scenery",
        "studio backdrop", "environmental context"
    ],
}

COMPOSITION_VOCABULARY = {
    "shot_type": [
        "extreme close-up", "close-up", "medium close-up", "medium shot",
        "medium full shot", "full shot", "wide shot", "extreme wide shot"
    ],
    "camera_angle": [
        "eye level", "low angle", "high angle", "bird's eye view",
        "worm's eye view", "dutch angle", "straight-on", "3/4 angle"
    ],
    "framing": [
        "centered composition", "rule of thirds", "golden ratio",
        "symmetrical framing", "asymmetrical balance", "leading lines",
        "frame within frame", "negative space"
    ],
    "depth": [
        "shallow depth of field", "deep focus", "selective focus",
        "layered depth", "foreground interest", "background separation"
    ],
    "lens": [
        "35mm lens", "50mm lens", "85mm portrait lens", "135mm telephoto",
        "24mm wide angle", "macro lens", "fisheye effect"
    ],
    "aspect_ratio": [
        "1:1 square", "4:3 standard", "16:9 widescreen", "9:16 vertical",
        "2.39:1 cinematic", "3:2 classic", "4:5 portrait"
    ],
}

LIGHTING_VOCABULARY = {
    "direction": [
        "front lighting", "side lighting", "backlighting", "rim lighting",
        "overhead lighting", "under lighting", "45-degree key light",
        "three-point lighting"
    ],
    "quality": [
        "soft diffused light", "hard direct light", "dappled light",
        "even lighting", "dramatic shadows", "high contrast",
        "low contrast", "harsh shadows"
    ],
    "color_temperature": [
        "warm golden light", "cool blue light", "neutral daylight",
        "candlelight warmth", "moonlight cool", "neon colors",
        "mixed temperature", "tungsten orange"
    ],
    "named_setups": [
        "Rembrandt lighting", "butterfly lighting", "split lighting",
        "loop lighting", "broad lighting", "short lighting",
        "clamshell lighting", "paramount lighting"
    ],
    "source": [
        "natural window light", "studio softbox", "ring light",
        "practical lights", "golden hour sun", "overcast sky",
        "neon signs", "street lamps"
    ],
    "effects": [
        "volumetric lighting", "god rays", "lens flare", "specular highlights",
        "rim highlights", "catchlights", "atmospheric haze", "light leaks"
    ],
}

STYLE_VOCABULARY = {
    "medium": [
        "photorealistic", "digital painting", "oil painting", "watercolor",
        "pencil sketch", "ink illustration", "3D render", "vector art",
        "mixed media", "collage"
    ],
    "aesthetic": [
        "cinematic", "editorial", "documentary", "fine art", "commercial",
        "candid", "glamour", "minimalist", "maximalist", "vintage"
    ],
    "film_stock": [
        "Kodak Portra 400", "Fujifilm Pro 400H", "Kodak Ektar 100",
        "Ilford HP5", "Cinestill 800T", "Kodachrome", "Tri-X 400"
    ],
    "camera_reference": [
        "shot on Sony A7IV", "Leica M6", "Hasselblad medium format",
        "Canon 5D Mark IV", "Nikon Z8", "RED cinema camera",
        "iPhone portrait mode", "disposable camera"
    ],
    "color_grading": [
        "warm and inviting", "cool and moody", "desaturated", "vibrant",
        "high contrast", "low contrast", "teal and orange", "monochrome",
        "pastel tones", "earthy tones"
    ],
    "texture_quality": [
        "highly detailed", "8K resolution", "sharp details", "film grain",
        "soft focus", "crisp and clean", "subtle noise", "smooth render"
    ],
    "art_reference": [
        "Studio Ghibli style", "Blade Runner 2049 aesthetic", "Wes Anderson colors",
        "Annie Leibovitz portrait", "National Geographic documentary",
        "Vogue editorial", "Tim Walker fantasy"
    ],
}

CONSTRAINT_VOCABULARY = {
    "quality": [
        "highly detailed", "8K resolution", "photorealistic quality",
        "sharp focus", "professional quality", "masterpiece"
    ],
    "safety": [
        "fully clothed", "modest attire", "safe for work",
        "appropriate content", "family friendly"
    ],
    "artifacts": [
        "no watermark", "no logo", "no text unless specified",
        "no extra limbs", "accurate anatomy", "5 fingers each hand",
        "no distortion", "clean edges"
    ],
    "exclusions": [
        "no background clutter", "no distracting elements",
        "only the subject", "clean composition", "minimal props"
    ],
}


# =============================================================================
# SCENE TYPE TEMPLATES
# =============================================================================

@dataclass
class SceneTemplate:
    """Template for a specific scene type."""
    scene_type: SceneType
    description: str
    required_categories: List[PromptCategory]
    primary_vocabulary: Dict[str, List[str]]
    template_pattern: str
    example_prompt: str
    tips: List[str] = field(default_factory=list)


SCENE_TEMPLATES: Dict[SceneType, SceneTemplate] = {

    SceneType.PORTRAIT: SceneTemplate(
        scene_type=SceneType.PORTRAIT,
        description="Head and shoulders portrait with facial focus",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.COMPOSITION,
            PromptCategory.LIGHTING,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "shot_type": ["close-up", "medium close-up", "headshot"],
            "lighting": ["soft diffused light", "Rembrandt lighting", "butterfly lighting"],
            "background": ["blurred background", "studio backdrop", "bokeh"],
            "lens": ["85mm portrait lens", "135mm telephoto"],
        },
        template_pattern=(
            "A {shot_type} portrait of {subject_description}. "
            "{expression_detail}. {hair_and_features}. "
            "{clothing_upper}. {background_description}. "
            "{lighting_setup}. {style_keywords}. {technical_specs}."
        ),
        example_prompt=(
            "A close-up portrait of a woman in her 30s with warm ivory skin "
            "and gentle smile. She has wavy chestnut hair falling past her shoulders "
            "and bright hazel eyes with visible catchlights. Wearing a simple dark blazer "
            "over a cream blouse. Soft diffused daylight from a large window to the left "
            "creates gentle Rembrandt lighting with subtle shadows. Blurred neutral grey "
            "background with soft bokeh. Shot on 85mm lens at f/1.8, shallow depth of field. "
            "Professional editorial portrait style, realistic skin texture, 8K resolution."
        ),
        tips=[
            "Include specific age range for realistic output",
            "Describe eye color and catchlight presence",
            "Specify lighting direction (left, right, front)",
            "Use lens focal length for appropriate compression",
        ],
    ),

    SceneType.FASHION: SceneTemplate(
        scene_type=SceneType.FASHION,
        description="Full body shot emphasizing clothing and style",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.COMPOSITION,
            PromptCategory.LIGHTING,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "shot_type": ["full body shot", "3/4 shot", "full length"],
            "pose": ["confident stance", "walking pose", "dynamic pose"],
            "outfit_detail": ["fabric texture", "color coordination", "layered outfit"],
            "style": ["editorial fashion", "streetwear", "haute couture"],
        },
        template_pattern=(
            "A {shot_type} of {subject_description} {pose_description}. "
            "{outfit_head_to_toe}. {footwear_and_accessories}. "
            "{location_and_mood}. {lighting_setup}. "
            "{style_direction}. {technical_specs}."
        ),
        example_prompt=(
            "A full body editorial fashion shot of a tall young woman with athletic build "
            "standing confidently with weight on one hip. She wears a tailored camel wool coat "
            "over a black turtleneck, high-waisted cream trousers with sharp creases, "
            "and pointed black leather ankle boots. Minimal gold jewelry. "
            "Shot on an empty city rooftop at golden hour, warm light from behind creating "
            "rim lighting on her silhouette. Urban skyline softly blurred in background. "
            "High fashion editorial style, Vogue aesthetic, cinematic color grading. "
            "Shot on medium format Hasselblad, shallow depth of field."
        ),
        tips=[
            "Describe outfit from head to toe systematically",
            "Include fabric textures and material properties",
            "Specify pose with body weight distribution",
            "Mention footwear explicitly",
        ],
    ),

    SceneType.PRODUCT: SceneTemplate(
        scene_type=SceneType.PRODUCT,
        description="Commercial product photography on clean surface",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.COMPOSITION,
            PromptCategory.LIGHTING,
            PromptCategory.CONSTRAINTS,
        ],
        primary_vocabulary={
            "surface": ["white marble surface", "wooden table", "gradient backdrop"],
            "lighting": ["soft box lighting", "rim highlights", "product lighting"],
            "material": ["glass", "metal", "leather", "matte", "glossy"],
            "angle": ["45-degree angle", "straight-on", "slight overhead"],
        },
        template_pattern=(
            "A professional product photo of {product_description}. "
            "{material_and_finish}. Placed on {surface_description}. "
            "{arrangement_and_props}. {lighting_setup}. "
            "{style_keywords}. {constraints}."
        ),
        example_prompt=(
            "A professional product photo of a modern wireless headset in matte black "
            "with brushed aluminum accents. The headset is placed on a white marble surface "
            "with soft shadow beneath. Clean white background with subtle gradient. "
            "Soft box lighting from top left creating gentle highlights on the metal parts "
            "and showing material texture. Subtle reflection on the surface. "
            "Commercial product photography style, extremely sharp details, 4K resolution. "
            "No hands, no people, no text, no watermark."
        ),
        tips=[
            "Describe material properties precisely (matte, glossy, brushed)",
            "Specify lighting to show texture without harsh glare",
            "Include shadow and reflection details",
            "Always add constraints: no hands, no people, no text",
        ],
    ),

    SceneType.LANDSCAPE: SceneTemplate(
        scene_type=SceneType.LANDSCAPE,
        description="Wide environmental/nature photography",
        required_categories=[
            PromptCategory.SCENE,
            PromptCategory.COMPOSITION,
            PromptCategory.LIGHTING,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "time": ["golden hour", "blue hour", "midday", "sunrise", "sunset"],
            "weather": ["clear sky", "dramatic clouds", "misty", "stormy"],
            "depth_layers": ["foreground", "middle ground", "background", "horizon"],
            "mood": ["serene", "dramatic", "ethereal", "majestic"],
        },
        template_pattern=(
            "A {time_of_day} view of {location_description}. "
            "{foreground_elements}. {middle_ground}. {background_and_sky}. "
            "{atmospheric_conditions}. {color_palette_mood}. "
            "{camera_settings}. {style_keywords}."
        ),
        example_prompt=(
            "A golden hour view of a mountain valley with a winding river. "
            "In the foreground, wildflowers and tall grass catch the warm light. "
            "The middle ground shows the silver river curving through green meadows "
            "with scattered pine trees. Snow-capped peaks rise in the background "
            "against a sky of soft orange and pink gradients with wispy clouds. "
            "Warm golden light casting long shadows, slight atmospheric haze "
            "adding depth. Serene and majestic mood. Shot on 24mm wide angle lens, "
            "deep focus from front to back. National Geographic style landscape photography, "
            "vibrant but natural colors, 8K resolution."
        ),
        tips=[
            "Build depth with foreground, middle, and background layers",
            "Specify sky conditions and cloud types",
            "Include atmospheric effects (haze, mist, fog)",
            "Use wide angle lens references (16-35mm)",
        ],
    ),

    SceneType.CINEMATIC: SceneTemplate(
        scene_type=SceneType.CINEMATIC,
        description="Film-style dramatic compositions",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.COMPOSITION,
            PromptCategory.LIGHTING,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "aspect_ratio": ["2.39:1 anamorphic", "cinemascope", "widescreen"],
            "lighting": ["dramatic lighting", "chiaroscuro", "noir lighting", "volumetric"],
            "color_grade": ["teal and orange", "desaturated", "high contrast", "moody"],
            "reference": ["Blade Runner", "Roger Deakins", "film noir", "neo-noir"],
        },
        template_pattern=(
            "A cinematic {shot_type} of {subject_description}. "
            "{scene_and_mood}. {dramatic_lighting}. "
            "{color_grading}. {film_reference}. "
            "{technical_specs}. {aspect_ratio}."
        ),
        example_prompt=(
            "A cinematic medium shot of a detective in a long trench coat standing alone "
            "on a rain-slicked city street at night. Neon signs reflect off the wet pavement "
            "in pools of red and blue. Steam rises from a nearby grate. Dramatic side lighting "
            "from a streetlamp creates deep shadows on half his face, film noir style. "
            "The background shows blurred city lights and silhouettes of passing cars. "
            "Moody atmosphere with high contrast. Teal and orange color grading. "
            "Shot like a Roger Deakins frame, anamorphic lens flare visible. "
            "2.39:1 cinemascope aspect ratio, 35mm film grain, cinematic depth of field."
        ),
        tips=[
            "Reference specific cinematographers or films",
            "Specify aspect ratio for cinematic framing",
            "Include atmospheric elements (rain, fog, steam)",
            "Use dramatic lighting with strong shadows",
        ],
    ),

    SceneType.ANIME: SceneTemplate(
        scene_type=SceneType.ANIME,
        description="Anime and illustration styles",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "style": ["cel-shaded", "Studio Ghibli", "90s anime", "modern anime"],
            "features": ["large expressive eyes", "stylized hair", "clean linework"],
            "atmosphere": ["ethereal", "whimsical", "dramatic", "serene"],
            "effects": ["soft glow", "sparkles", "wind effects", "cherry blossoms"],
        },
        template_pattern=(
            "A {style_reference} illustration of {character_description}. "
            "{scene_and_setting}. {artistic_effects}. "
            "{color_palette}. {rendering_style}."
        ),
        example_prompt=(
            "A Studio Ghibli style illustration of a young girl with flowing auburn hair "
            "sitting on a grassy hillside overlooking a distant seaside town. "
            "She wears a simple blue dress with a white apron, holding a small bouquet "
            "of wildflowers. Soft wind blows her hair gently. Fluffy cumulus clouds "
            "dot a bright blue sky. Warm afternoon sunlight creates a peaceful atmosphere. "
            "Soft cel-shading with vibrant but gentle colors. Hand-painted background style "
            "with visible brushwork. Whimsical and nostalgic mood. Clean linework, "
            "large expressive eyes catching the light."
        ),
        tips=[
            "Reference specific anime studios or artists",
            "Describe hair and eye style for anime aesthetic",
            "Include atmospheric effects (wind, light rays)",
            "Specify cel-shading or painting technique",
        ],
    ),

    SceneType.EDITORIAL: SceneTemplate(
        scene_type=SceneType.EDITORIAL,
        description="Magazine and fashion editorial style",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.COMPOSITION,
            PromptCategory.LIGHTING,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "style": ["Vogue editorial", "Harper's Bazaar", "high fashion", "avant-garde"],
            "composition": ["dynamic pose", "editorial angle", "dramatic framing"],
            "lighting": ["studio lighting", "dramatic shadows", "beauty lighting"],
            "mood": ["sophisticated", "bold", "elegant", "provocative"],
        },
        template_pattern=(
            "A {publication_style} editorial photo of {subject_description}. "
            "{pose_and_attitude}. {outfit_and_styling}. "
            "{set_design}. {lighting_setup}. "
            "{color_and_mood}. {technical_quality}."
        ),
        example_prompt=(
            "A Vogue editorial fashion photo of a model with striking features "
            "and slicked-back dark hair. Bold red lipstick and dramatic winged eyeliner. "
            "She poses with one hand raised near her face in an artistic gesture, "
            "expression confident and intense. Wearing an avant-garde structured blazer "
            "in electric blue with exaggerated shoulders. Minimalist white studio background "
            "with hard shadow creating graphic contrast. High-key beauty lighting "
            "from directly above with reflector fill. Bold, sophisticated mood. "
            "High fashion editorial photography, extremely sharp details, "
            "8K resolution, magazine cover quality."
        ),
        tips=[
            "Reference specific fashion magazines",
            "Include makeup and styling details",
            "Describe pose attitude and hand placement",
            "Use high-key or dramatic studio lighting",
        ],
    ),

    SceneType.VINTAGE: SceneTemplate(
        scene_type=SceneType.VINTAGE,
        description="Retro and film aesthetic photography",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "era": ["1950s", "1970s", "1980s", "90s aesthetic"],
            "film": ["Kodak Portra", "Polaroid", "disposable camera", "VHS"],
            "effects": ["film grain", "light leaks", "faded colors", "vignette"],
            "imperfections": ["slight blur", "red eye", "date stamp", "scratches"],
        },
        template_pattern=(
            "A {era} style photograph of {subject_description}. "
            "{setting_period_appropriate}. {film_stock_reference}. "
            "{intentional_imperfections}. {color_treatment}. {mood}."
        ),
        example_prompt=(
            "A 1970s style photograph of a young couple at a summer beach, "
            "shot on Kodak Ektachrome film. They wear period-appropriate swimwear "
            "and sunglasses, laughing together on a colorful beach towel. "
            "Slightly faded colors with warm yellow-orange cast. Visible film grain "
            "and subtle light leaks at the edges. Soft focus characteristic of vintage lenses. "
            "Date stamp '78 in orange text at bottom right corner. "
            "Nostalgic, carefree summer mood. Natural sunlight with slight overexposure "
            "on highlights. Authentic vintage snapshot aesthetic."
        ),
        tips=[
            "Specify era for period-appropriate styling",
            "Reference specific film stocks",
            "Include intentional imperfections (grain, leaks)",
            "Add date stamps or vintage text elements",
        ],
    ),

    SceneType.CYBERPUNK: SceneTemplate(
        scene_type=SceneType.CYBERPUNK,
        description="Futuristic sci-fi and cyberpunk aesthetics",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.LIGHTING,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "elements": ["neon signs", "holographic", "chrome", "LED", "tech implants"],
            "lighting": ["neon glow", "volumetric light", "rim lighting", "high contrast"],
            "environment": ["rain-slicked streets", "towering megastructures", "dense urban"],
            "color": ["cyan and magenta", "neon pink", "electric blue", "toxic green"],
        },
        template_pattern=(
            "A cyberpunk scene of {subject_description}. "
            "{futuristic_environment}. {neon_lighting}. "
            "{tech_elements}. {atmospheric_effects}. "
            "{color_palette}. {style_reference}."
        ),
        example_prompt=(
            "A cyberpunk portrait of a young woman with platinum blonde hair glowing "
            "under holographic rain. She has subtle LED implants along her cheekbones "
            "and chrome cybernetic ear modifications. Wearing a high-collar black jacket "
            "with fiber optic wiring along the seams. Standing in a narrow alley "
            "with towering neon signs in Japanese and English, steam rising from vents. "
            "Intense rim lighting in magenta from a sign behind her, key light in cyan "
            "from a holographic advertisement to the left. Rain-slicked surfaces "
            "reflecting the neon colors. High contrast, vibrant neon color palette. "
            "Blade Runner 2049 aesthetic, 8K cinematic quality."
        ),
        tips=[
            "Include specific neon colors (cyan, magenta, pink)",
            "Add rain and reflective wet surfaces",
            "Reference Blade Runner or similar films",
            "Describe tech elements and implants",
        ],
    ),

    SceneType.FOOD: SceneTemplate(
        scene_type=SceneType.FOOD,
        description="Appetizing food photography",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.COMPOSITION,
            PromptCategory.LIGHTING,
            PromptCategory.STYLE,
        ],
        primary_vocabulary={
            "presentation": ["plated", "styled", "garnished", "drizzled"],
            "freshness": ["glistening", "steaming", "fresh", "crisp"],
            "angle": ["overhead flat lay", "45-degree", "eye level", "macro detail"],
            "props": ["rustic table", "linen napkin", "ceramic plate", "utensils"],
        },
        template_pattern=(
            "A {style} food photograph of {dish_description}. "
            "{plating_and_garnish}. {props_and_surface}. "
            "{freshness_indicators}. {lighting_setup}. "
            "{color_and_mood}. {technical_quality}."
        ),
        example_prompt=(
            "A professional food photograph of a beautifully plated pasta dish. "
            "Fresh tagliatelle with rich bolognese sauce, topped with shaved parmesan "
            "and fresh basil leaves. The sauce glistens under the light, steam rising gently. "
            "Served in a rustic white ceramic bowl on a weathered wooden table "
            "with a linen napkin and vintage fork beside it. Soft natural window light "
            "from the left creating gentle shadows. A few scattered ingredients "
            "(tomatoes, herbs) in soft focus background. Warm, appetizing color palette. "
            "High-end restaurant editorial style, shallow depth of field, "
            "extremely sharp focus on the food, 8K detail."
        ),
        tips=[
            "Include freshness cues (steam, glistening, drips)",
            "Describe garnishes and toppings",
            "Use warm lighting for appetizing appeal",
            "Add contextual props (utensils, napkins)",
        ],
    ),

    SceneType.TEXT_OVERLAY: SceneTemplate(
        scene_type=SceneType.TEXT_OVERLAY,
        description="Images with integrated text elements",
        required_categories=[
            PromptCategory.SUBJECT,
            PromptCategory.SCENE,
            PromptCategory.STYLE,
            PromptCategory.CONSTRAINTS,
        ],
        primary_vocabulary={
            "text_style": ["neon sign", "graffiti", "elegant typography", "bold sans-serif"],
            "placement": ["centered", "top third", "integrated into scene"],
            "legibility": ["high contrast", "clean kerning", "readable", "clear typography"],
        },
        template_pattern=(
            "A {scene_description} with {text_element} displaying \"{exact_text}\". "
            "{text_style_and_placement}. {integration_with_scene}. "
            "{lighting_for_text}. {style_keywords}. "
            "{text_constraints}."
        ),
        example_prompt=(
            "A moody urban night scene with a vintage neon sign displaying \"OPEN 24 HRS\" "
            "mounted above a late-night diner. The neon tubes glow in warm red and white, "
            "slightly flickering. Rain-wet street below reflects the neon light. "
            "The diner windows emit warm yellow light from inside. "
            "Cinematic night photography with film grain. "
            "Typography: ONLY these exact words, high readability, clean neon tube lettering, "
            "no additional text, no watermarks."
        ),
        tips=[
            "Wrap exact text in quotes",
            "Specify text style (neon, graffiti, elegant)",
            "Add constraint: ONLY these exact words",
            "Include readability and kerning notes",
        ],
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_vocabulary_for_scene(scene_type: SceneType) -> Dict[str, List[str]]:
    """Get relevant vocabulary for a specific scene type."""
    template = SCENE_TEMPLATES.get(scene_type)
    if not template:
        return {}

    vocab = {}

    # Add primary vocabulary from template
    vocab.update(template.primary_vocabulary)

    # Add relevant general vocabulary based on required categories
    for category in template.required_categories:
        if category == PromptCategory.SUBJECT:
            vocab.update(SUBJECT_VOCABULARY)
        elif category == PromptCategory.SCENE:
            vocab.update(SCENE_VOCABULARY)
        elif category == PromptCategory.COMPOSITION:
            vocab.update(COMPOSITION_VOCABULARY)
        elif category == PromptCategory.LIGHTING:
            vocab.update(LIGHTING_VOCABULARY)
        elif category == PromptCategory.STYLE:
            vocab.update(STYLE_VOCABULARY)
        elif category == PromptCategory.CONSTRAINTS:
            vocab.update(CONSTRAINT_VOCABULARY)

    return vocab


def detect_scene_type(tags: Dict[str, float]) -> SceneType:
    """Detect likely scene type from tagger output."""
    # Score each scene type based on tag presence
    scores = {st: 0.0 for st in SceneType}

    tag_lower = {k.lower(): v for k, v in tags.items()}

    # Portrait indicators
    portrait_tags = {"portrait", "headshot", "close-up", "face", "1girl", "1boy", "solo"}
    for tag in portrait_tags:
        if tag in tag_lower:
            scores[SceneType.PORTRAIT] += tag_lower[tag]

    # Fashion indicators
    fashion_tags = {"fashion", "dress", "outfit", "clothing", "full body", "model", "pose"}
    for tag in fashion_tags:
        if tag in tag_lower:
            scores[SceneType.FASHION] += tag_lower[tag]

    # Product indicators
    product_tags = {"product", "object", "item", "still life", "white background"}
    for tag in product_tags:
        if tag in tag_lower:
            scores[SceneType.PRODUCT] += tag_lower[tag]

    # Landscape indicators
    landscape_tags = {"landscape", "scenery", "mountain", "sky", "nature", "outdoor", "horizon"}
    for tag in landscape_tags:
        if tag in tag_lower:
            scores[SceneType.LANDSCAPE] += tag_lower[tag]

    # Cinematic indicators
    cinematic_tags = {"cinematic", "dramatic", "film", "noir", "moody", "atmospheric"}
    for tag in cinematic_tags:
        if tag in tag_lower:
            scores[SceneType.CINEMATIC] += tag_lower[tag]

    # Anime indicators
    anime_tags = {"anime", "manga", "illustration", "2d", "cel-shaded", "cartoon"}
    for tag in anime_tags:
        if tag in tag_lower:
            scores[SceneType.ANIME] += tag_lower[tag]

    # Food indicators
    food_tags = {"food", "dish", "meal", "cuisine", "plate", "restaurant"}
    for tag in food_tags:
        if tag in tag_lower:
            scores[SceneType.FOOD] += tag_lower[tag]

    # Return highest scoring scene type
    return max(scores, key=scores.get)


def get_all_scene_types() -> List[SceneType]:
    """Get all available scene types."""
    return list(SceneType)


def get_template(scene_type: SceneType) -> Optional[SceneTemplate]:
    """Get template for a specific scene type."""
    return SCENE_TEMPLATES.get(scene_type)
