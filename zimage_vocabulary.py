"""
Z-Image Vocabulary Registry

Comprehensive vocabulary system optimized for Z-Image/Z-Image-Turbo model.
Based on official documentation, community research, and best practices.

Key Principles:
- Z-Image uses natural language, NOT tag soup
- Optimal prompt length: 80-250 words
- No negative prompts (ignored by model)
- Strong response to camera/lighting specifications
- Structure: [Composition] + [Subject] + [Environment] + [Lighting] + [Mood] + [Technical]

References:
- https://github.com/Tongyi-MAI/Z-Image
- https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo
- https://gist.github.com/illuminatianon/c42f8e57f1e3ebf037dd58043da9de32
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum


class VocabCategory(Enum):
    """Vocabulary categories aligned with Z-Image prompt structure."""
    COMPOSITION = "composition"
    SHOT_TYPE = "shot_type"
    FACE_ANGLE = "face_angle"
    LIGHTING = "lighting"
    CAMERA = "camera"
    LENS = "lens"
    FILM_STOCK = "film_stock"
    COLOR_PALETTE = "color_palette"
    MOOD = "mood"
    MATERIAL = "material"
    ENVIRONMENT = "environment"
    CLOTHING = "clothing"
    QUALITY = "quality"
    SAFETY = "safety"


@dataclass
class VocabTerm:
    """A vocabulary term with metadata."""
    term: str
    category: VocabCategory
    effectiveness: float  # 0.0-1.0 score for Z-Image response
    use_case: str  # When to use this term
    alternatives: List[str] = None  # Synonyms/variations

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


# =============================================================================
# COMPOSITION & FRAMING
# =============================================================================

COMPOSITION_VOCAB: Dict[str, VocabTerm] = {
    "centered_subject": VocabTerm(
        term="subject centered in frame",
        category=VocabCategory.COMPOSITION,
        effectiveness=0.90,
        use_case="portraits, product shots",
        alternatives=["centered composition", "subject in center"]
    ),
    "rule_of_thirds": VocabTerm(
        term="composed using rule of thirds",
        category=VocabCategory.COMPOSITION,
        effectiveness=0.85,
        use_case="dynamic compositions",
        alternatives=["off-center positioning", "asymmetric composition"]
    ),
    "negative_space": VocabTerm(
        term="with generous negative space",
        category=VocabCategory.COMPOSITION,
        effectiveness=0.80,
        use_case="minimalist, editorial",
        alternatives=["breathing room around subject", "clean uncluttered frame"]
    ),
    "tight_crop": VocabTerm(
        term="tightly cropped",
        category=VocabCategory.COMPOSITION,
        effectiveness=0.85,
        use_case="detail shots, intimacy",
        alternatives=["close framing", "minimal background"]
    ),
    "full_frame": VocabTerm(
        term="full frame composition showing complete subject",
        category=VocabCategory.COMPOSITION,
        effectiveness=0.85,
        use_case="full body, environmental",
        alternatives=["wide framing", "complete subject visible"]
    ),
}

# =============================================================================
# SHOT TYPES
# =============================================================================

SHOT_TYPE_VOCAB: Dict[str, VocabTerm] = {
    "extreme_closeup": VocabTerm(
        term="extreme close-up",
        category=VocabCategory.SHOT_TYPE,
        effectiveness=0.95,
        use_case="details, eyes, texture",
        alternatives=["macro shot", "detail shot"]
    ),
    "closeup": VocabTerm(
        term="close-up shot",
        category=VocabCategory.SHOT_TYPE,
        effectiveness=0.95,
        use_case="face, head and shoulders",
        alternatives=["head shot", "tight portrait"]
    ),
    "medium_closeup": VocabTerm(
        term="medium close-up",
        category=VocabCategory.SHOT_TYPE,
        effectiveness=0.90,
        use_case="chest up",
        alternatives=["bust shot"]
    ),
    "medium_shot": VocabTerm(
        term="medium shot",
        category=VocabCategory.SHOT_TYPE,
        effectiveness=0.90,
        use_case="waist up",
        alternatives=["mid shot", "waist-up shot"]
    ),
    "medium_full": VocabTerm(
        term="medium full shot",
        category=VocabCategory.SHOT_TYPE,
        effectiveness=0.85,
        use_case="knees up",
        alternatives=["three-quarter shot", "american shot"]
    ),
    "full_shot": VocabTerm(
        term="full body shot",
        category=VocabCategory.SHOT_TYPE,
        effectiveness=0.90,
        use_case="complete figure",
        alternatives=["full shot", "full-length shot"]
    ),
    "wide_shot": VocabTerm(
        term="wide shot",
        category=VocabCategory.SHOT_TYPE,
        effectiveness=0.85,
        use_case="subject in environment",
        alternatives=["long shot", "establishing shot"]
    ),
}

# =============================================================================
# FACE ANGLES (Critical for Z-Image)
# =============================================================================

FACE_ANGLE_VOCAB: Dict[str, VocabTerm] = {
    "front_view": VocabTerm(
        term="front view, facing camera directly",
        category=VocabCategory.FACE_ANGLE,
        effectiveness=0.95,
        use_case="direct engagement, ID photos",
        alternatives=["straight on", "looking at camera"]
    ),
    "three_quarter": VocabTerm(
        term="three-quarter view, face angled 45 degrees",
        category=VocabCategory.FACE_ANGLE,
        effectiveness=0.95,
        use_case="classic portrait angle",
        alternatives=["45 degree angle", "turned slightly"]
    ),
    "profile": VocabTerm(
        term="profile view, side of face",
        category=VocabCategory.FACE_ANGLE,
        effectiveness=0.90,
        use_case="silhouettes, dramatic",
        alternatives=["side view", "90 degree angle"]
    ),
    "looking_up": VocabTerm(
        term="looking slightly upward",
        category=VocabCategory.FACE_ANGLE,
        effectiveness=0.85,
        use_case="aspirational, heroic",
        alternatives=["chin raised", "upward gaze"]
    ),
    "looking_down": VocabTerm(
        term="looking slightly downward",
        category=VocabCategory.FACE_ANGLE,
        effectiveness=0.85,
        use_case="introspective, shy",
        alternatives=["chin lowered", "downward gaze"]
    ),
    "over_shoulder": VocabTerm(
        term="looking over shoulder",
        category=VocabCategory.FACE_ANGLE,
        effectiveness=0.85,
        use_case="casual, candid feel",
        alternatives=["glancing back", "shoulder glance"]
    ),
}

# =============================================================================
# LIGHTING (Z-Image responds strongly to lighting)
# =============================================================================

LIGHTING_VOCAB: Dict[str, VocabTerm] = {
    # Natural Lighting
    "soft_diffused": VocabTerm(
        term="soft diffused daylight",
        category=VocabCategory.LIGHTING,
        effectiveness=0.95,
        use_case="portraits, beauty",
        alternatives=["soft natural light", "diffused window light"]
    ),
    "golden_hour": VocabTerm(
        term="warm golden hour sunlight",
        category=VocabCategory.LIGHTING,
        effectiveness=0.95,
        use_case="romantic, warm portraits",
        alternatives=["sunset lighting", "magic hour light"]
    ),
    "blue_hour": VocabTerm(
        term="cool blue hour ambient light",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="moody, cinematic",
        alternatives=["twilight lighting", "dusk ambient"]
    ),
    "overcast": VocabTerm(
        term="soft overcast daylight",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="even, flattering light",
        alternatives=["cloudy day light", "diffused sky light"]
    ),

    # Studio Lighting
    "studio_softbox": VocabTerm(
        term="studio softbox lighting",
        category=VocabCategory.LIGHTING,
        effectiveness=0.95,
        use_case="professional portraits",
        alternatives=["soft studio light", "diffused studio lighting"]
    ),
    "rembrandt": VocabTerm(
        term="Rembrandt lighting with triangle shadow under eye",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="dramatic portraits",
        alternatives=["classic portrait lighting", "chiaroscuro"]
    ),
    "butterfly": VocabTerm(
        term="butterfly lighting from above",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="beauty, glamour",
        alternatives=["paramount lighting", "top-down beauty light"]
    ),
    "split": VocabTerm(
        term="split lighting, half face in shadow",
        category=VocabCategory.LIGHTING,
        effectiveness=0.85,
        use_case="dramatic, mysterious",
        alternatives=["side lighting", "half-lit face"]
    ),
    "rim": VocabTerm(
        term="rim lighting creating edge separation",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="subject separation, drama",
        alternatives=["backlight edge", "hair light", "edge lighting"]
    ),

    # Cinematic Lighting
    "cinematic_warm": VocabTerm(
        term="cinematic warm key light",
        category=VocabCategory.LIGHTING,
        effectiveness=0.95,
        use_case="film look, narrative",
        alternatives=["movie lighting", "warm cinematic"]
    ),
    "noir": VocabTerm(
        term="high-contrast noir lighting with deep shadows",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="dramatic, mystery",
        alternatives=["film noir style", "dramatic shadows"]
    ),
    "neon": VocabTerm(
        term="colorful neon lighting",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="cyberpunk, urban night",
        alternatives=["neon glow", "colored city lights"]
    ),

    # Volumetric/Atmospheric
    "volumetric": VocabTerm(
        term="volumetric light rays",
        category=VocabCategory.LIGHTING,
        effectiveness=0.90,
        use_case="atmospheric, ethereal",
        alternatives=["god rays", "light beams through atmosphere"]
    ),
    "hazy": VocabTerm(
        term="soft hazy atmospheric light",
        category=VocabCategory.LIGHTING,
        effectiveness=0.85,
        use_case="dreamy, nostalgic",
        alternatives=["misty light", "foggy atmosphere"]
    ),
}

# =============================================================================
# CAMERA & EQUIPMENT
# =============================================================================

CAMERA_VOCAB: Dict[str, VocabTerm] = {
    "dslr_pro": VocabTerm(
        term="shot on professional DSLR camera",
        category=VocabCategory.CAMERA,
        effectiveness=0.85,
        use_case="general high quality",
        alternatives=["professional camera", "high-end DSLR"]
    ),
    "canon_5d": VocabTerm(
        term="shot on Canon 5D Mark IV",
        category=VocabCategory.CAMERA,
        effectiveness=0.90,
        use_case="professional portraits",
        alternatives=["Canon EOS 5D", "full-frame Canon"]
    ),
    "sony_a7": VocabTerm(
        term="shot on Sony A7 IV",
        category=VocabCategory.CAMERA,
        effectiveness=0.90,
        use_case="mirrorless quality",
        alternatives=["Sony Alpha", "Sony mirrorless"]
    ),
    "hasselblad": VocabTerm(
        term="shot on Hasselblad medium format",
        category=VocabCategory.CAMERA,
        effectiveness=0.90,
        use_case="fashion, editorial",
        alternatives=["medium format camera", "Hasselblad H6D"]
    ),
    "leica": VocabTerm(
        term="shot on Leica M",
        category=VocabCategory.CAMERA,
        effectiveness=0.90,
        use_case="street, documentary",
        alternatives=["Leica rangefinder", "Leica M10"]
    ),
    "smartphone": VocabTerm(
        term="smartphone photo",
        category=VocabCategory.CAMERA,
        effectiveness=0.85,
        use_case="casual, authentic",
        alternatives=["iPhone photo", "mobile phone shot"]
    ),
    "polaroid": VocabTerm(
        term="Polaroid instant photo",
        category=VocabCategory.CAMERA,
        effectiveness=0.85,
        use_case="nostalgic, vintage",
        alternatives=["instant camera shot", "Instax style"]
    ),
}

# =============================================================================
# LENS SPECIFICATIONS
# =============================================================================

LENS_VOCAB: Dict[str, VocabTerm] = {
    "24mm": VocabTerm(
        term="24mm wide angle lens",
        category=VocabCategory.LENS,
        effectiveness=0.90,
        use_case="environmental, architecture",
        alternatives=["wide angle", "24mm focal length"]
    ),
    "35mm": VocabTerm(
        term="35mm lens",
        category=VocabCategory.LENS,
        effectiveness=0.90,
        use_case="street, documentary",
        alternatives=["35mm focal length", "standard wide"]
    ),
    "50mm": VocabTerm(
        term="50mm lens",
        category=VocabCategory.LENS,
        effectiveness=0.95,
        use_case="natural perspective, portraits",
        alternatives=["nifty fifty", "50mm standard lens"]
    ),
    "85mm": VocabTerm(
        term="85mm portrait lens",
        category=VocabCategory.LENS,
        effectiveness=0.95,
        use_case="portraits, headshots",
        alternatives=["85mm telephoto", "portrait focal length"]
    ),
    "105mm": VocabTerm(
        term="105mm macro lens",
        category=VocabCategory.LENS,
        effectiveness=0.90,
        use_case="details, macro",
        alternatives=["macro lens", "close-up lens"]
    ),
    "135mm": VocabTerm(
        term="135mm telephoto lens",
        category=VocabCategory.LENS,
        effectiveness=0.90,
        use_case="compressed perspective, portraits",
        alternatives=["telephoto portrait", "135mm focal length"]
    ),
    "200mm": VocabTerm(
        term="200mm telephoto lens",
        category=VocabCategory.LENS,
        effectiveness=0.85,
        use_case="sports, wildlife, compression",
        alternatives=["long telephoto", "200mm focal length"]
    ),
}

# =============================================================================
# APERTURE & DEPTH OF FIELD
# =============================================================================

APERTURE_VOCAB: Dict[str, VocabTerm] = {
    "f1_4": VocabTerm(
        term="f/1.4 wide open aperture, extremely shallow depth of field",
        category=VocabCategory.LENS,
        effectiveness=0.95,
        use_case="dreamy bokeh, isolation",
        alternatives=["wide open", "maximum aperture"]
    ),
    "f1_8": VocabTerm(
        term="f/1.8 aperture with shallow depth of field",
        category=VocabCategory.LENS,
        effectiveness=0.95,
        use_case="portraits with bokeh",
        alternatives=["shallow focus", "blurred background"]
    ),
    "f2_8": VocabTerm(
        term="f/2.8 aperture with soft background blur",
        category=VocabCategory.LENS,
        effectiveness=0.90,
        use_case="balanced bokeh",
        alternatives=["moderate depth of field", "subtle background blur"]
    ),
    "f5_6": VocabTerm(
        term="f/5.6 aperture with moderate depth of field",
        category=VocabCategory.LENS,
        effectiveness=0.85,
        use_case="group shots, some background detail",
        alternatives=["mid-range aperture"]
    ),
    "f8": VocabTerm(
        term="f/8 aperture with sharp depth of field",
        category=VocabCategory.LENS,
        effectiveness=0.85,
        use_case="landscape, architecture",
        alternatives=["stopped down", "deep focus"]
    ),
    "f11_sharp": VocabTerm(
        term="f/11 aperture, everything in sharp focus",
        category=VocabCategory.LENS,
        effectiveness=0.85,
        use_case="maximum sharpness",
        alternatives=["deep depth of field", "sharp throughout"]
    ),
}

# =============================================================================
# FILM STOCK & COLOR PROCESSING
# =============================================================================

FILM_STOCK_VOCAB: Dict[str, VocabTerm] = {
    "portra_400": VocabTerm(
        term="Kodak Portra 400 film aesthetic",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.95,
        use_case="portraits, warm skin tones",
        alternatives=["Portra colors", "warm film look"]
    ),
    "portra_800": VocabTerm(
        term="Kodak Portra 800 film grain",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.90,
        use_case="low light, visible grain",
        alternatives=["high ISO film", "grainy portrait film"]
    ),
    "ektar": VocabTerm(
        term="Kodak Ektar 100 vibrant colors",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.90,
        use_case="saturated, punchy colors",
        alternatives=["vivid film colors", "saturated film"]
    ),
    "tri_x": VocabTerm(
        term="Kodak Tri-X 400 black and white film grain",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.90,
        use_case="classic B&W",
        alternatives=["classic black and white", "contrasty B&W"]
    ),
    "hp5": VocabTerm(
        term="Ilford HP5 black and white aesthetic",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.85,
        use_case="documentary B&W",
        alternatives=["Ilford film look", "British B&W film"]
    ),
    "fuji_400h": VocabTerm(
        term="Fuji Pro 400H pastel tones",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.90,
        use_case="soft pastels, wedding",
        alternatives=["soft film colors", "pastel film"]
    ),
    "cinestill_800t": VocabTerm(
        term="Cinestill 800T tungsten halation",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.90,
        use_case="night photography, red halation",
        alternatives=["cinematic tungsten", "night film with halation"]
    ),
    "velvia": VocabTerm(
        term="Fuji Velvia slide film saturated colors",
        category=VocabCategory.FILM_STOCK,
        effectiveness=0.85,
        use_case="landscape, vivid saturation",
        alternatives=["slide film", "ultra saturated colors"]
    ),
}

# =============================================================================
# COLOR PALETTE
# =============================================================================

COLOR_PALETTE_VOCAB: Dict[str, VocabTerm] = {
    "warm": VocabTerm(
        term="warm color palette with golden and amber tones",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.90,
        use_case="inviting, cozy",
        alternatives=["warm tones", "golden color grade"]
    ),
    "cool": VocabTerm(
        term="cool color palette with blue and teal tones",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.90,
        use_case="modern, professional",
        alternatives=["cool tones", "blue color grade"]
    ),
    "muted": VocabTerm(
        term="muted desaturated colors",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.85,
        use_case="editorial, moody",
        alternatives=["low saturation", "faded colors"]
    ),
    "vibrant": VocabTerm(
        term="vibrant saturated colors",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.85,
        use_case="energetic, commercial",
        alternatives=["punchy colors", "high saturation"]
    ),
    "monochrome": VocabTerm(
        term="monochromatic color scheme",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.85,
        use_case="artistic, cohesive",
        alternatives=["single color palette", "tonal harmony"]
    ),
    "pastel": VocabTerm(
        term="soft pastel color palette",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.85,
        use_case="gentle, feminine",
        alternatives=["soft colors", "light muted tones"]
    ),
    "neon": VocabTerm(
        term="neon color palette with electric hues",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.90,
        use_case="cyberpunk, nightlife",
        alternatives=["electric colors", "fluorescent palette"]
    ),
    "earth_tones": VocabTerm(
        term="natural earth tone colors",
        category=VocabCategory.COLOR_PALETTE,
        effectiveness=0.85,
        use_case="organic, natural",
        alternatives=["brown and green palette", "organic colors"]
    ),
}

# =============================================================================
# MOOD & ATMOSPHERE
# =============================================================================

MOOD_VOCAB: Dict[str, VocabTerm] = {
    "serene": VocabTerm(
        term="serene peaceful atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.85,
        use_case="calm, meditative",
        alternatives=["tranquil", "peaceful mood"]
    ),
    "dramatic": VocabTerm(
        term="dramatic intense atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.90,
        use_case="impactful, emotional",
        alternatives=["intense mood", "powerful atmosphere"]
    ),
    "intimate": VocabTerm(
        term="intimate personal atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.85,
        use_case="close, personal",
        alternatives=["personal mood", "close atmosphere"]
    ),
    "ethereal": VocabTerm(
        term="ethereal dreamy atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.90,
        use_case="fantasy, soft",
        alternatives=["dreamy", "otherworldly mood"]
    ),
    "gritty": VocabTerm(
        term="gritty raw atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.85,
        use_case="documentary, street",
        alternatives=["raw mood", "unpolished feel"]
    ),
    "nostalgic": VocabTerm(
        term="nostalgic wistful atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.85,
        use_case="memories, vintage",
        alternatives=["wistful mood", "reminiscent atmosphere"]
    ),
    "mysterious": VocabTerm(
        term="mysterious enigmatic atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.85,
        use_case="intrigue, noir",
        alternatives=["enigmatic mood", "secretive atmosphere"]
    ),
    "joyful": VocabTerm(
        term="joyful vibrant atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.85,
        use_case="happy, celebratory",
        alternatives=["happy mood", "celebratory atmosphere"]
    ),
    "melancholic": VocabTerm(
        term="melancholic contemplative atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.85,
        use_case="sad, thoughtful",
        alternatives=["somber mood", "pensive atmosphere"]
    ),
    "cinematic": VocabTerm(
        term="cinematic filmic atmosphere",
        category=VocabCategory.MOOD,
        effectiveness=0.95,
        use_case="movie-like quality",
        alternatives=["film-like mood", "movie atmosphere"]
    ),
}

# =============================================================================
# MATERIALS & TEXTURES
# =============================================================================

MATERIAL_VOCAB: Dict[str, VocabTerm] = {
    "dewy_skin": VocabTerm(
        term="dewy luminous skin",
        category=VocabCategory.MATERIAL,
        effectiveness=0.90,
        use_case="beauty, fresh look",
        alternatives=["glowing skin", "fresh dewy complexion"]
    ),
    "matte_skin": VocabTerm(
        term="natural matte skin texture",
        category=VocabCategory.MATERIAL,
        effectiveness=0.85,
        use_case="natural beauty",
        alternatives=["natural skin", "non-glossy skin"]
    ),
    "leather": VocabTerm(
        term="rich leather texture",
        category=VocabCategory.MATERIAL,
        effectiveness=0.85,
        use_case="fashion, accessories",
        alternatives=["leather material", "textured leather"]
    ),
    "silk": VocabTerm(
        term="smooth silk fabric with subtle sheen",
        category=VocabCategory.MATERIAL,
        effectiveness=0.85,
        use_case="fashion, luxury",
        alternatives=["silky fabric", "shiny smooth fabric"]
    ),
    "wool": VocabTerm(
        term="textured wool fabric",
        category=VocabCategory.MATERIAL,
        effectiveness=0.80,
        use_case="cozy, winter",
        alternatives=["knit texture", "woolen material"]
    ),
    "denim": VocabTerm(
        term="worn denim texture",
        category=VocabCategory.MATERIAL,
        effectiveness=0.85,
        use_case="casual, everyday",
        alternatives=["jeans material", "denim fabric"]
    ),
    "glass": VocabTerm(
        term="transparent glass with reflections",
        category=VocabCategory.MATERIAL,
        effectiveness=0.85,
        use_case="product, architectural",
        alternatives=["glass surface", "reflective glass"]
    ),
    "metal": VocabTerm(
        term="polished metal surface",
        category=VocabCategory.MATERIAL,
        effectiveness=0.85,
        use_case="industrial, jewelry",
        alternatives=["metallic finish", "shiny metal"]
    ),
}

# =============================================================================
# ENVIRONMENT & BACKGROUND
# =============================================================================

ENVIRONMENT_VOCAB: Dict[str, VocabTerm] = {
    "studio_white": VocabTerm(
        term="clean white studio background",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.95,
        use_case="product, headshots",
        alternatives=["white backdrop", "plain white background"]
    ),
    "studio_gray": VocabTerm(
        term="neutral gray studio background",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.90,
        use_case="professional portraits",
        alternatives=["gray backdrop", "neutral background"]
    ),
    "studio_black": VocabTerm(
        term="dark black studio background",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.90,
        use_case="dramatic, low key",
        alternatives=["black backdrop", "dark background"]
    ),
    "blurred_urban": VocabTerm(
        term="soft blurred urban background",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.90,
        use_case="street portraits",
        alternatives=["out of focus city", "bokeh city lights"]
    ),
    "nature_outdoor": VocabTerm(
        term="natural outdoor environment with trees and foliage",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.85,
        use_case="outdoor portraits",
        alternatives=["park setting", "nature background"]
    ),
    "office_modern": VocabTerm(
        term="modern office interior",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.85,
        use_case="corporate, professional",
        alternatives=["workspace background", "office setting"]
    ),
    "home_cozy": VocabTerm(
        term="cozy home interior",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.85,
        use_case="lifestyle, casual",
        alternatives=["living room setting", "domestic interior"]
    ),
    "cafe": VocabTerm(
        term="warm cafe interior",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.85,
        use_case="casual, lifestyle",
        alternatives=["coffee shop", "cafe setting"]
    ),
    "beach": VocabTerm(
        term="sandy beach with ocean in background",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.85,
        use_case="vacation, summer",
        alternatives=["seaside", "ocean background"]
    ),
    "street": VocabTerm(
        term="urban street scene",
        category=VocabCategory.ENVIRONMENT,
        effectiveness=0.85,
        use_case="street photography",
        alternatives=["city street", "urban setting"]
    ),
}

# =============================================================================
# ANTI-PATTERNS (Terms Z-Image ignores or handles poorly)
# =============================================================================

ANTI_PATTERNS: Set[str] = {
    # Quality boosters (Z-Image doesn't need these)
    "8K", "4K", "masterpiece", "best quality", "highly detailed",
    "ultra detailed", "super detailed", "extremely detailed",
    "high resolution", "HD", "UHD", "award winning",

    # Negative prompt style (Z-Image ignores negative prompts)
    "no bad", "without", "no ugly", "not",

    # Redundant technical terms
    "photorealistic", "hyperrealistic", "photo realistic",
    "raw photo", "DSLR quality",

    # Vague modifiers
    "beautiful", "gorgeous", "stunning", "amazing", "perfect",
    "very", "extremely", "incredibly", "absolutely",

    # Anime/SD specific
    "anime style", "manga", "illustration", "digital art",
    "(masterpiece:1.2)", "score_9", "score_8_up",
}

# =============================================================================
# SYNONYM MAPPINGS (Generic → Z-Image optimized)
# =============================================================================

SYNONYM_MAP: Dict[str, str] = {
    # Lighting synonyms
    "beautiful lighting": "soft diffused daylight",
    "good lighting": "balanced natural lighting",
    "nice light": "soft flattering light",
    "dramatic light": "high-contrast dramatic lighting",

    # Background synonyms
    "blurry background": "shallow depth of field with soft bokeh",
    "nice background": "clean uncluttered background",
    "simple background": "minimal plain background",

    # Quality synonyms (replace with nothing - not needed)
    "high quality": "",
    "best quality": "",
    "professional photo": "",
    "amazing photo": "",

    # Generic descriptors → specific
    "pretty woman": "attractive adult woman",
    "handsome man": "well-groomed adult man",
    "old person": "elderly adult",
    "young person": "young adult",

    # Pose synonyms
    "looking at camera": "direct eye contact with camera, front-facing",
    "side view": "profile view, face turned 90 degrees",
    "natural pose": "relaxed natural posture",
}

# =============================================================================
# SAFETY PHRASES (Required for SFW generation)
# =============================================================================

SAFETY_PHRASES: Dict[str, str] = {
    "sfw_basic": "safe for work, fully clothed, no nudity",
    "sfw_full": "safe for work, non-sexual, fully clothed, no nudity, no revealing clothing, no suggestive poses",
    "modest_clothing": "modest outfit, appropriate clothing, conservative attire",
    "anatomy_fix": "correct human anatomy, natural hands and fingers, no extra limbs",
    "artifact_fix": "no text, no watermark, no logos, no UI elements, sharp focus, clean image",
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES: Dict[str, str] = {
    "portrait_professional": (
        "{shot_type} of {subject}, {face_angle}. "
        "{clothing_description}. {lighting}. "
        "Shot with {lens}, {aperture}. "
        "{environment}. {mood}. "
        "Professional photography, {safety}."
    ),
    "portrait_casual": (
        "{shot_type} of {subject}, {expression}, {face_angle}. "
        "Wearing {clothing}. {lighting}. "
        "{lens} lens, {aperture}. "
        "{environment}. {mood} atmosphere. {safety}."
    ),
    "product": (
        "Product photography of {subject}. "
        "{lighting}, {environment}. "
        "Sharp focus on product details, clean composition. "
        "{color_palette}. Professional commercial photography."
    ),
    "street": (
        "Street photography, {shot_type} of {subject}. "
        "{environment}. {lighting}. "
        "Candid natural moment, {mood}. "
        "Shot on {camera} with {lens}. {film_stock} aesthetic."
    ),
    "fashion": (
        "Fashion photography, {shot_type} of {subject}. "
        "Wearing {clothing}. {face_angle}. "
        "{lighting}. {environment}. "
        "{mood} editorial atmosphere. "
        "Shot on {camera} with {lens}, {aperture}."
    ),
}


# =============================================================================
# VOCABULARY REGISTRY CLASS
# =============================================================================

class ZImageVocabulary:
    """
    Central registry for Z-Image optimized vocabulary.
    Provides methods to look up, translate, and validate prompts.
    """

    def __init__(self):
        # Combine all vocabularies
        self.all_vocab: Dict[str, VocabTerm] = {}
        self.all_vocab.update(COMPOSITION_VOCAB)
        self.all_vocab.update(SHOT_TYPE_VOCAB)
        self.all_vocab.update(FACE_ANGLE_VOCAB)
        self.all_vocab.update(LIGHTING_VOCAB)
        self.all_vocab.update(CAMERA_VOCAB)
        self.all_vocab.update(LENS_VOCAB)
        self.all_vocab.update(APERTURE_VOCAB)
        self.all_vocab.update(FILM_STOCK_VOCAB)
        self.all_vocab.update(COLOR_PALETTE_VOCAB)
        self.all_vocab.update(MOOD_VOCAB)
        self.all_vocab.update(MATERIAL_VOCAB)
        self.all_vocab.update(ENVIRONMENT_VOCAB)

        # Category-specific lookups
        self._by_category: Dict[VocabCategory, Dict[str, VocabTerm]] = {}
        for key, term in self.all_vocab.items():
            if term.category not in self._by_category:
                self._by_category[term.category] = {}
            self._by_category[term.category][key] = term

    def get_term(self, key: str) -> Optional[VocabTerm]:
        """Get a vocabulary term by key."""
        return self.all_vocab.get(key)

    def get_category(self, category: VocabCategory) -> Dict[str, VocabTerm]:
        """Get all terms in a category."""
        return self._by_category.get(category, {})

    def get_best_terms(self, category: VocabCategory, count: int = 5) -> List[VocabTerm]:
        """Get the highest effectiveness terms in a category."""
        terms = list(self._by_category.get(category, {}).values())
        terms.sort(key=lambda t: t.effectiveness, reverse=True)
        return terms[:count]

    def translate_synonym(self, text: str) -> str:
        """Replace generic terms with Z-Image optimized versions."""
        result = text
        for generic, optimized in SYNONYM_MAP.items():
            if generic.lower() in result.lower():
                # Case-insensitive replacement
                import re
                result = re.sub(re.escape(generic), optimized, result, flags=re.IGNORECASE)
        return result

    def remove_anti_patterns(self, text: str) -> str:
        """Remove terms that Z-Image ignores or handles poorly."""
        result = text
        for pattern in ANTI_PATTERNS:
            import re
            # Remove pattern with optional surrounding commas/spaces
            result = re.sub(
                r',?\s*' + re.escape(pattern) + r'\s*,?',
                ', ',
                result,
                flags=re.IGNORECASE
            )
        # Clean up double commas and spaces
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r'\s+', ' ', result)
        return result.strip(' ,')

    def get_template(self, template_name: str) -> Optional[str]:
        """Get a prompt template by name."""
        return PROMPT_TEMPLATES.get(template_name)

    def get_safety_phrase(self, level: str = "sfw_basic") -> str:
        """Get a safety phrase for SFW content."""
        return SAFETY_PHRASES.get(level, SAFETY_PHRASES["sfw_basic"])

    def is_anti_pattern(self, term: str) -> bool:
        """Check if a term is an anti-pattern."""
        return term.lower() in {p.lower() for p in ANTI_PATTERNS}

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Suggest vocabulary improvements for a prompt."""
        suggestions = []
        prompt_lower = prompt.lower()

        # Check for missing essential elements
        has_lighting = any(
            term.term.lower() in prompt_lower or any(alt.lower() in prompt_lower for alt in term.alternatives)
            for term in self._by_category.get(VocabCategory.LIGHTING, {}).values()
        )
        if not has_lighting:
            suggestions.append("Add lighting description (e.g., 'soft diffused daylight', 'cinematic warm key light')")

        has_composition = any(
            term.term.lower() in prompt_lower or any(alt.lower() in prompt_lower for alt in term.alternatives)
            for term in self._by_category.get(VocabCategory.SHOT_TYPE, {}).values()
        )
        if not has_composition:
            suggestions.append("Add shot type (e.g., 'close-up', 'medium shot', 'full body shot')")

        # Check for anti-patterns
        for pattern in ANTI_PATTERNS:
            if pattern.lower() in prompt_lower:
                suggestions.append(f"Remove '{pattern}' - Z-Image doesn't need quality boosters")

        # Check for synonyms that could be improved
        for generic, optimized in SYNONYM_MAP.items():
            if generic.lower() in prompt_lower and optimized:
                suggestions.append(f"Replace '{generic}' with '{optimized}'")

        return suggestions


# Global instance
vocab = ZImageVocabulary()


def get_vocabulary() -> ZImageVocabulary:
    """Get the global vocabulary instance."""
    return vocab
