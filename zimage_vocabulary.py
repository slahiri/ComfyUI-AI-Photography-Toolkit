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
    CAMERA_ANGLE = "camera_angle"  # Low/high/dutch angles
    LIGHTING = "lighting"
    LIGHTING_PORTRAIT = "lighting_portrait"  # Portrait-specific lighting
    LIGHTING_EFFECT = "lighting_effect"  # Atmospheric/CGI effects
    CAMERA = "camera"
    LENS = "lens"
    LENS_EFFECT = "lens_effect"  # Bokeh, flares, aberrations
    FILM_STOCK = "film_stock"
    COLOR_PALETTE = "color_palette"
    MOOD = "mood"
    MATERIAL = "material"
    MATERIAL_EFFECT = "material_effect"  # Creative material shaping
    ENVIRONMENT = "environment"
    ART_STYLE = "art_style"  # Historical/animation styles
    THEME = "theme"  # Fantasy/sci-fi/aesthetic themes
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
# CAMERA ANGLES (From PromptForge)
# =============================================================================

CAMERA_ANGLE_VOCAB: Dict[str, VocabTerm] = {
    "low_angle": VocabTerm(
        term="low-angle shot, camera positioned below eye level looking up",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.90,
        use_case="power, dominance, monumentality",
        alternatives=["looking up at subject", "hero angle"]
    ),
    "high_angle": VocabTerm(
        term="high-angle shot, camera positioned above looking down",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.90,
        use_case="vulnerability, overview",
        alternatives=["looking down at subject", "overhead perspective"]
    ),
    "eye_level": VocabTerm(
        term="eye-level shot, neutral natural perspective",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.95,
        use_case="neutral, realistic",
        alternatives=["straight-on angle", "natural perspective"]
    ),
    "extreme_low_angle": VocabTerm(
        term="extreme low-angle shot, camera near ground looking sharply upward",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.85,
        use_case="dramatic scale, towering effect",
        alternatives=["ground-level up", "ant's eye view"]
    ),
    "worm_eye_view": VocabTerm(
        term="worm's-eye view, camera at ground level looking up",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.85,
        use_case="exaggerated scale, dominance",
        alternatives=["ground perspective", "extremely low angle"]
    ),
    "bird_eye_view": VocabTerm(
        term="bird's-eye view, aerial perspective looking down",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.90,
        use_case="overview, spatial context",
        alternatives=["aerial shot", "overhead view"]
    ),
    "dutch_angle": VocabTerm(
        term="Dutch angle, camera tilted on roll axis creating diagonal composition",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.85,
        use_case="tension, disorientation, dynamic",
        alternatives=["canted angle", "tilted frame"]
    ),
    "overhead_shot": VocabTerm(
        term="overhead shot, camera directly above pointing straight down",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.90,
        use_case="flat lay, patterns, spatial design",
        alternatives=["top-down shot", "flat lay perspective"]
    ),
    "oblique_angle": VocabTerm(
        term="oblique angle, camera at diagonal to subject",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.85,
        use_case="depth, dimension, dynamic",
        alternatives=["diagonal perspective", "angled view"]
    ),
    "pov_shot": VocabTerm(
        term="point-of-view shot, camera replicating character's perspective",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.85,
        use_case="immersive, subjective experience",
        alternatives=["first-person perspective", "subjective camera"]
    ),
    "over_shoulder_shot": VocabTerm(
        term="over-the-shoulder shot, camera behind one subject looking at another",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.90,
        use_case="conversation, spatial relationship",
        alternatives=["OTS shot", "shoulder framing"]
    ),
    "drone_shot": VocabTerm(
        term="drone shot, elevated aerial view with smooth floating quality",
        category=VocabCategory.CAMERA_ANGLE,
        effectiveness=0.90,
        use_case="sweeping vistas, contemporary aesthetic",
        alternatives=["aerial drone perspective", "flying camera view"]
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
# PORTRAIT LIGHTING (From PromptForge - Studio/Portrait specific)
# =============================================================================

LIGHTING_PORTRAIT_VOCAB: Dict[str, VocabTerm] = {
    "loop_lighting": VocabTerm(
        term="loop lighting with small shadow beside nose",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.90,
        use_case="flattering portraits, versatile",
        alternatives=["loop shadow lighting", "slightly angled key light"]
    ),
    "clamshell_lighting": VocabTerm(
        term="clamshell lighting with fill from below",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.90,
        use_case="beauty photography, glamour",
        alternatives=["beauty dish clamshell", "above and below lighting"]
    ),
    "cross_lighting": VocabTerm(
        term="cross lighting with two opposing light sources",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.85,
        use_case="dramatic, sculptural portraits",
        alternatives=["opposing lights", "double side lighting"]
    ),
    "broad_lighting": VocabTerm(
        term="broad lighting illuminating the side of face facing camera",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.85,
        use_case="wider face appearance, high key",
        alternatives=["front-side lit", "near-side illumination"]
    ),
    "short_lighting": VocabTerm(
        term="short lighting illuminating the side of face away from camera",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.85,
        use_case="slimming effect, dramatic shadows",
        alternatives=["narrow lighting", "far-side illumination"]
    ),
    "paramount_lighting": VocabTerm(
        term="paramount lighting with butterfly shadow under nose",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.90,
        use_case="classic Hollywood glamour",
        alternatives=["butterfly lighting", "high key beauty light"]
    ),
    "kicker_light": VocabTerm(
        term="kicker light creating accent on subject edge",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.85,
        use_case="subject separation, depth",
        alternatives=["accent light", "edge highlight"]
    ),
    "hair_light": VocabTerm(
        term="hair light separating subject from background",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.85,
        use_case="studio portraits, depth",
        alternatives=["top light", "background separation light"]
    ),
    "beauty_dish": VocabTerm(
        term="beauty dish lighting with crisp yet soft quality",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.90,
        use_case="beauty, fashion portraits",
        alternatives=["beauty dish modifier", "parabolic reflector light"]
    ),
    "ring_light": VocabTerm(
        term="ring light creating circular catchlights in eyes",
        category=VocabCategory.LIGHTING_PORTRAIT,
        effectiveness=0.85,
        use_case="beauty, vlogging, even illumination",
        alternatives=["diva light", "ring flash aesthetic"]
    ),
}

# =============================================================================
# LIGHTING EFFECTS (From PromptForge - Atmospheric/CGI)
# =============================================================================

LIGHTING_EFFECT_VOCAB: Dict[str, VocabTerm] = {
    "god_rays": VocabTerm(
        term="god rays with visible light beams through atmosphere",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.90,
        use_case="ethereal, spiritual, dramatic",
        alternatives=["crepuscular rays", "sunbeams through clouds"]
    ),
    "caustics": VocabTerm(
        term="caustic light patterns from water or glass refraction",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.85,
        use_case="underwater scenes, pool lighting",
        alternatives=["water caustics", "refracted light patterns"]
    ),
    "volumetric_fog": VocabTerm(
        term="volumetric fog with visible light scattering",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.90,
        use_case="atmospheric, mysterious",
        alternatives=["atmospheric haze", "misty light scattering"]
    ),
    "subsurface_scattering": VocabTerm(
        term="subsurface scattering showing light through translucent material",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.85,
        use_case="skin glow, leaves, wax, ears backlit",
        alternatives=["SSS effect", "translucent glow"]
    ),
    "dappled_light": VocabTerm(
        term="dappled light filtering through leaves creating shadow patterns",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.90,
        use_case="natural outdoor, forests",
        alternatives=["leaf shadow patterns", "filtered sunlight"]
    ),
    "specular_highlights": VocabTerm(
        term="specular highlights on wet or glossy surfaces",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.85,
        use_case="rain, wet streets, glossy materials",
        alternatives=["shiny reflections", "gloss highlights"]
    ),
    "practical_lights": VocabTerm(
        term="practical lights visible in scene as light sources",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.85,
        use_case="realistic scenes, film lighting",
        alternatives=["in-frame light sources", "visible lamps"]
    ),
    "lens_flare_lighting": VocabTerm(
        term="lens flare from bright light source in frame",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.85,
        use_case="dreamy, cinematic, sun-kissed",
        alternatives=["sun flare", "optical flare effect"]
    ),
    "global_illumination": VocabTerm(
        term="global illumination with realistic light bounce",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.85,
        use_case="photorealistic CGI, natural lighting",
        alternatives=["indirect lighting", "radiosity"]
    ),
    "ambient_occlusion": VocabTerm(
        term="ambient occlusion darkening crevices and corners",
        category=VocabCategory.LIGHTING_EFFECT,
        effectiveness=0.80,
        use_case="depth, realism, 3D renders",
        alternatives=["contact shadows", "cavity darkening"]
    ),
}

# =============================================================================
# LENS EFFECTS (From PromptForge - Bokeh, Flares, Aberrations)
# =============================================================================

LENS_EFFECT_VOCAB: Dict[str, VocabTerm] = {
    "bokeh_balls": VocabTerm(
        term="beautiful bokeh balls in background from out of focus lights",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.95,
        use_case="portraits, night scenes",
        alternatives=["light orbs", "defocused light circles"]
    ),
    "swirly_bokeh": VocabTerm(
        term="swirly bokeh with spiral background blur",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.85,
        use_case="vintage lens aesthetic, artistic",
        alternatives=["helios bokeh", "rotating background blur"]
    ),
    "bubble_bokeh": VocabTerm(
        term="bubble bokeh with soap-bubble-like highlights",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.85,
        use_case="dreamy, vintage lens look",
        alternatives=["spherical aberration bokeh", "outlined bokeh circles"]
    ),
    "anamorphic_flare": VocabTerm(
        term="anamorphic lens flare with horizontal blue streaks",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.90,
        use_case="cinematic widescreen, sci-fi",
        alternatives=["horizontal flare streaks", "cinema lens flare"]
    ),
    "chromatic_aberration": VocabTerm(
        term="chromatic aberration with color fringing at edges",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.80,
        use_case="retro, lo-fi aesthetic",
        alternatives=["color fringing", "RGB split edge"]
    ),
    "barrel_distortion": VocabTerm(
        term="barrel distortion from wide angle lens",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.80,
        use_case="fisheye, exaggerated perspective",
        alternatives=["fisheye distortion", "curved edges"]
    ),
    "vignette": VocabTerm(
        term="natural vignette with darkened corners",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.85,
        use_case="focus attention, vintage feel",
        alternatives=["corner falloff", "edge darkening"]
    ),
    "tilt_shift": VocabTerm(
        term="tilt-shift effect with selective focus plane",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.85,
        use_case="miniature effect, architectural",
        alternatives=["selective plane focus", "miniature fake"]
    ),
    "soft_focus": VocabTerm(
        term="soft focus with gentle halation glow",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.85,
        use_case="romantic, dreamy portraits",
        alternatives=["diffused focus", "glowing soft look"]
    ),
    "double_exposure": VocabTerm(
        term="double exposure blending two images",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.85,
        use_case="artistic, conceptual",
        alternatives=["multiple exposure", "overlaid images"]
    ),
    "motion_blur": VocabTerm(
        term="motion blur showing movement",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.85,
        use_case="action, speed, panning shots",
        alternatives=["movement blur", "panning blur"]
    ),
    "lens_dust": VocabTerm(
        term="visible lens dust particles catching light",
        category=VocabCategory.LENS_EFFECT,
        effectiveness=0.75,
        use_case="vintage, nostalgic, lo-fi",
        alternatives=["dust particles", "sensor dust aesthetic"]
    ),
}

# =============================================================================
# ART STYLES (From PromptForge - Historical & Animation)
# =============================================================================

ART_STYLE_VOCAB: Dict[str, VocabTerm] = {
    # Historical Art Movements
    "renaissance": VocabTerm(
        term="Renaissance art style with classical proportions and chiaroscuro",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.90,
        use_case="classical portraits, religious themes",
        alternatives=["High Renaissance", "Da Vinci style"]
    ),
    "baroque": VocabTerm(
        term="Baroque style with dramatic lighting and rich detail",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.90,
        use_case="dramatic, ornate compositions",
        alternatives=["Caravaggio style", "dramatic baroque"]
    ),
    "impressionism": VocabTerm(
        term="Impressionist painting style with visible brushstrokes and light study",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.90,
        use_case="soft, painterly, outdoor scenes",
        alternatives=["Monet style", "impressionist brush technique"]
    ),
    "art_nouveau": VocabTerm(
        term="Art Nouveau style with flowing organic curves and decorative elements",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="decorative, nature-inspired",
        alternatives=["Mucha style", "organic flowing lines"]
    ),
    "art_deco": VocabTerm(
        term="Art Deco style with geometric patterns and bold symmetry",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="1920s glamour, geometric",
        alternatives=["Gatsby era", "geometric deco"]
    ),
    "surrealism": VocabTerm(
        term="Surrealist style with dreamlike impossible imagery",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="dream sequences, conceptual",
        alternatives=["Dali style", "surreal dreamscape"]
    ),
    "pop_art": VocabTerm(
        term="Pop Art style with bold colors and halftone dots",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="retro, commercial, bold",
        alternatives=["Warhol style", "comic book aesthetic"]
    ),
    "ukiyo_e": VocabTerm(
        term="Ukiyo-e Japanese woodblock print style with flat colors and bold outlines",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="Japanese aesthetic, waves, nature",
        alternatives=["Hokusai style", "Japanese woodblock"]
    ),
    "watercolor": VocabTerm(
        term="watercolor painting with soft washes and bleeding edges",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="soft, delicate, organic",
        alternatives=["watercolour", "wet wash painting"]
    ),
    "oil_painting": VocabTerm(
        term="oil painting with rich impasto texture and blended colors",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.90,
        use_case="classical, rich textures",
        alternatives=["impasto technique", "classical oil portrait"]
    ),

    # Animation/Digital Styles
    "anime": VocabTerm(
        term="anime art style with large expressive eyes and stylized features",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="Japanese animation aesthetic",
        alternatives=["Japanese anime", "manga style"]
    ),
    "pixar": VocabTerm(
        term="Pixar 3D animation style with appealing character design",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="family-friendly, animated",
        alternatives=["Disney Pixar", "CGI animation style"]
    ),
    "studio_ghibli": VocabTerm(
        term="Studio Ghibli style with painterly backgrounds and whimsical detail",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.90,
        use_case="whimsical, nature, fantasy",
        alternatives=["Miyazaki style", "Ghibli aesthetic"]
    ),
    "comic_book": VocabTerm(
        term="comic book illustration with bold ink lines and dynamic poses",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="superheroes, action",
        alternatives=["graphic novel style", "comic illustration"]
    ),
    "concept_art": VocabTerm(
        term="concept art style with painterly detail and dramatic composition",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.90,
        use_case="entertainment design, world-building",
        alternatives=["game concept art", "film pre-production art"]
    ),
    "hyperrealism": VocabTerm(
        term="hyperrealistic painting indistinguishable from photography",
        category=VocabCategory.ART_STYLE,
        effectiveness=0.85,
        use_case="extreme detail, photo-like",
        alternatives=["photorealistic painting", "hyperreal"]
    ),
}

# =============================================================================
# THEMES/AESTHETICS (From PromptForge - Fantasy/Sci-Fi/Aesthetic)
# =============================================================================

THEME_VOCAB: Dict[str, VocabTerm] = {
    # Sci-Fi/Futuristic
    "cyberpunk": VocabTerm(
        term="cyberpunk aesthetic with neon lights and high-tech dystopia",
        category=VocabCategory.THEME,
        effectiveness=0.95,
        use_case="futuristic urban, tech noir",
        alternatives=["cyber noir", "neon dystopia"]
    ),
    "synthwave": VocabTerm(
        term="synthwave aesthetic with retro-futuristic neon and chrome",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="80s retro-future, vaporwave adjacent",
        alternatives=["outrun aesthetic", "retrowave"]
    ),
    "solarpunk": VocabTerm(
        term="solarpunk aesthetic with green technology and optimistic future",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="eco-futurism, sustainable tech",
        alternatives=["eco-futurism", "green utopia"]
    ),
    "dieselpunk": VocabTerm(
        term="dieselpunk aesthetic with 1920s-40s retro-futurism",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="alt-history, industrial retro",
        alternatives=["decopunk", "WWI-era futurism"]
    ),
    "biopunk": VocabTerm(
        term="biopunk aesthetic with organic technology and genetic modification",
        category=VocabCategory.THEME,
        effectiveness=0.80,
        use_case="biotech horror, organic sci-fi",
        alternatives=["organic tech", "bio-mechanical"]
    ),

    # Fantasy/Historical
    "steampunk": VocabTerm(
        term="steampunk aesthetic with Victorian brass, gears, and steam machinery",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="Victorian sci-fi, mechanical",
        alternatives=["Victorian futurism", "clockwork aesthetic"]
    ),
    "dark_fantasy": VocabTerm(
        term="dark fantasy aesthetic with gothic elements and ominous atmosphere",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="horror fantasy, grim settings",
        alternatives=["grimdark", "gothic fantasy"]
    ),
    "high_fantasy": VocabTerm(
        term="high fantasy aesthetic with magical realms and epic scope",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="Tolkien-esque, epic fantasy",
        alternatives=["epic fantasy", "Lord of the Rings aesthetic"]
    ),
    "fairy_tale": VocabTerm(
        term="fairy tale aesthetic with enchanted forests and magical creatures",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="storybook, whimsical magic",
        alternatives=["storybook", "enchanted"]
    ),
    "mythology": VocabTerm(
        term="mythological aesthetic with gods, heroes, and ancient legends",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="Greek/Norse myths, legends",
        alternatives=["legendary", "mythic"]
    ),

    # Modern Aesthetics
    "cottagecore": VocabTerm(
        term="cottagecore aesthetic with pastoral countryside and cozy domestic life",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="rural idyll, nostalgic pastoral",
        alternatives=["pastoral aesthetic", "countryside romance"]
    ),
    "dark_academia": VocabTerm(
        term="dark academia aesthetic with old libraries, tweed, and scholarly atmosphere",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="intellectual, vintage scholarly",
        alternatives=["gothic academia", "scholarly vintage"]
    ),
    "light_academia": VocabTerm(
        term="light academia aesthetic with warm libraries, cream tones, and classical learning",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="bright scholarly, Mediterranean",
        alternatives=["classical academia", "bright scholarly"]
    ),
    "vaporwave": VocabTerm(
        term="vaporwave aesthetic with glitchy 90s nostalgia and pastel neon",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="retro internet, 90s nostalgia",
        alternatives=["aesthetic wave", "90s digital nostalgia"]
    ),
    "liminal_space": VocabTerm(
        term="liminal space aesthetic with empty transitional spaces and uncanny atmosphere",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="surreal, unsettling emptiness",
        alternatives=["backrooms", "empty transitional space"]
    ),
    "brutalist": VocabTerm(
        term="brutalist aesthetic with raw concrete and monumental architecture",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="architectural, modernist",
        alternatives=["concrete brutalism", "raw architecture"]
    ),
    "minimalist": VocabTerm(
        term="minimalist aesthetic with clean lines and negative space",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="modern, clean, simple",
        alternatives=["minimal design", "clean aesthetic"]
    ),
    "maximalist": VocabTerm(
        term="maximalist aesthetic with rich patterns, textures, and bold colors",
        category=VocabCategory.THEME,
        effectiveness=0.85,
        use_case="opulent, layered, decorative",
        alternatives=["more is more", "ornate maximalism"]
    ),
    "gothic": VocabTerm(
        term="gothic aesthetic with dark romanticism and Victorian mourning",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="dark romantic, Victorian dark",
        alternatives=["gothic romance", "dark Victorian"]
    ),
    "noir": VocabTerm(
        term="film noir aesthetic with high contrast shadows and cynical atmosphere",
        category=VocabCategory.THEME,
        effectiveness=0.90,
        use_case="detective, mystery, 1940s",
        alternatives=["noir thriller", "hard-boiled aesthetic"]
    ),
}

# =============================================================================
# MATERIAL EFFECTS (From PromptForge - Creative Materials)
# =============================================================================

MATERIAL_EFFECT_VOCAB: Dict[str, VocabTerm] = {
    "fire_shaped": VocabTerm(
        term="subject made of fire with flickering flames and ember trails",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="elemental, magical effects",
        alternatives=["flame creature", "fire elemental form"]
    ),
    "ice_crystal": VocabTerm(
        term="subject made of crystalline ice with frost and refraction",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="winter magic, frozen effects",
        alternatives=["frozen crystal form", "ice sculpture"]
    ),
    "water_formed": VocabTerm(
        term="subject made of flowing water with liquid transparency",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="elemental, fluid forms",
        alternatives=["water elemental", "liquid form"]
    ),
    "smoke_ethereal": VocabTerm(
        term="subject made of swirling smoke with wispy tendrils",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="ghostly, ethereal effects",
        alternatives=["smoke form", "vapor creature"]
    ),
    "holographic": VocabTerm(
        term="holographic projection with scan lines and blue glow",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.90,
        use_case="sci-fi, futuristic tech",
        alternatives=["hologram", "digital projection"]
    ),
    "neon_glow": VocabTerm(
        term="neon light tubes forming subject with electric glow",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="urban, cyberpunk, signs",
        alternatives=["neon sign form", "light tube art"]
    ),
    "wireframe": VocabTerm(
        term="wireframe mesh with visible polygons and vertices",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.80,
        use_case="3D modeling, digital art",
        alternatives=["polygon mesh", "3D wireframe"]
    ),
    "paper_craft": VocabTerm(
        term="paper craft construction with folded layers and visible edges",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="origami, paper art",
        alternatives=["paper sculpture", "origami style"]
    ),
    "stained_glass": VocabTerm(
        term="stained glass with colored panes and lead borders",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.90,
        use_case="religious, decorative, art",
        alternatives=["leaded glass art", "church window style"]
    ),
    "mosaic": VocabTerm(
        term="mosaic tiles forming subject with visible grout lines",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="classical, decorative",
        alternatives=["tile mosaic", "tessellated art"]
    ),
    "chrome_liquid": VocabTerm(
        term="liquid chrome with mirror-like reflections and flowing surface",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.90,
        use_case="sci-fi, surreal, T-1000 style",
        alternatives=["mercury", "liquid metal"]
    ),
    "gold_leaf": VocabTerm(
        term="gold leaf covering with rich metallic sheen and fine cracks",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="luxury, religious art, opulent",
        alternatives=["gilded", "gold plated"]
    ),
    "crystalline": VocabTerm(
        term="crystalline structure with faceted surfaces and light refraction",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.90,
        use_case="gemstone, magical, precious",
        alternatives=["crystal formation", "gem-like structure"]
    ),
    "porcelain": VocabTerm(
        term="porcelain with smooth glazed surface and delicate details",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="doll-like, delicate, artistic",
        alternatives=["ceramic", "china figurine"]
    ),
    "marble_carved": VocabTerm(
        term="carved marble with veining and classical sculpture quality",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.90,
        use_case="classical sculpture, Greek/Roman",
        alternatives=["marble statue", "classical sculpture"]
    ),
    "particle_dissolve": VocabTerm(
        term="subject dissolving into particles floating away",
        category=VocabCategory.MATERIAL_EFFECT,
        effectiveness=0.85,
        use_case="disintegration effect, magical",
        alternatives=["ash dissolve", "particle dispersion"]
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
        self.all_vocab.update(CAMERA_ANGLE_VOCAB)
        self.all_vocab.update(LIGHTING_VOCAB)
        self.all_vocab.update(LIGHTING_PORTRAIT_VOCAB)
        self.all_vocab.update(LIGHTING_EFFECT_VOCAB)
        self.all_vocab.update(CAMERA_VOCAB)
        self.all_vocab.update(LENS_VOCAB)
        self.all_vocab.update(LENS_EFFECT_VOCAB)
        self.all_vocab.update(APERTURE_VOCAB)
        self.all_vocab.update(FILM_STOCK_VOCAB)
        self.all_vocab.update(COLOR_PALETTE_VOCAB)
        self.all_vocab.update(MOOD_VOCAB)
        self.all_vocab.update(MATERIAL_VOCAB)
        self.all_vocab.update(MATERIAL_EFFECT_VOCAB)
        self.all_vocab.update(ENVIRONMENT_VOCAB)
        self.all_vocab.update(ART_STYLE_VOCAB)
        self.all_vocab.update(THEME_VOCAB)

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
