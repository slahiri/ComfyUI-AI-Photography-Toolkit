"""Canonical Structurer - Scene Detection and Category Classification.

This module:
1. Detects scene types from tagger outputs (1 primary + optional secondary)
2. Classifies all tags into 6 canonical prompt categories
3. Generates structured JSON with natural language phrases
4. Produces high-resolution text for each category

Based on Z-Image canonical prompt structure:
Subject + Scene + Composition + Lighting + Style + Constraints

NLP Features:
- Semantic similarity matching using sentence-transformers
- Lemmatization using spaCy for better word normalization
- Synonym mapping and plural handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import re
import numpy as np


# =============================================================================
# SCENE TYPES
# =============================================================================

class SceneType(Enum):
    """Primary scene type categories."""
    PORTRAIT = "portrait"
    FASHION = "fashion"
    ACTION = "action"
    GROUP = "group"
    PRODUCT = "product"
    FOOD = "food"
    STILL_LIFE = "still_life"
    LANDSCAPE = "landscape"
    ARCHITECTURE = "architecture"
    STREET = "street"
    CINEMATIC = "cinematic"
    EDITORIAL = "editorial"
    ANIME = "anime"
    VINTAGE = "vintage"
    CYBERPUNK = "cyberpunk"
    TEXT_OVERLAY = "text_overlay"


class CanonicalCategory(Enum):
    """The 6 canonical prompt categories."""
    SUBJECT = "subject"
    SCENE = "scene"
    COMPOSITION = "composition"
    LIGHTING = "lighting"
    STYLE = "style"
    CONSTRAINTS = "constraints"


# =============================================================================
# SCENE-BASED CATEGORY RELEVANCE
# =============================================================================
# Maps each scene type to which canonical categories are relevant
# Categories not listed are excluded from output for that scene type

SCENE_CATEGORY_RELEVANCE: Dict[SceneType, Set[CanonicalCategory]] = {
    # PORTRAIT - focus on subject, lighting, composition, style
    # Added SCENE for background info (grey background, studio, etc.)
    SceneType.PORTRAIT: {
        CanonicalCategory.SUBJECT,      # Person details, expression, hair
        CanonicalCategory.SCENE,        # Background (studio, grey bg, etc.)
        CanonicalCategory.COMPOSITION,  # Shot type, angle, framing
        CanonicalCategory.LIGHTING,     # Portrait lighting is crucial
        CanonicalCategory.STYLE,        # Aesthetic, mood
        CanonicalCategory.CONSTRAINTS,  # Quality
    },
    # FASHION - full body, outfit emphasis
    SceneType.FASHION: {
        CanonicalCategory.SUBJECT,      # Person + clothing details
        CanonicalCategory.SCENE,        # Background/studio
        CanonicalCategory.COMPOSITION,  # Full body framing
        CanonicalCategory.LIGHTING,     # Fashion lighting
        CanonicalCategory.STYLE,        # Fashion aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # ACTION - dynamic motion shots
    SceneType.ACTION: {
        CanonicalCategory.SUBJECT,      # Who/what is in action
        CanonicalCategory.SCENE,        # Environment
        CanonicalCategory.COMPOSITION,  # Dynamic angles
        CanonicalCategory.STYLE,        # Action aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # GROUP - multiple people
    SceneType.GROUP: {
        CanonicalCategory.SUBJECT,      # People details
        CanonicalCategory.SCENE,        # Setting
        CanonicalCategory.COMPOSITION,  # Group arrangement
        CanonicalCategory.LIGHTING,
        CanonicalCategory.STYLE,
        CanonicalCategory.CONSTRAINTS,
    },
    # PRODUCT - commercial product photography
    SceneType.PRODUCT: {
        CanonicalCategory.SUBJECT,      # Product details
        CanonicalCategory.SCENE,        # Clean background
        CanonicalCategory.COMPOSITION,  # Product framing
        CanonicalCategory.LIGHTING,     # Product lighting crucial
        CanonicalCategory.STYLE,        # Commercial style
        CanonicalCategory.CONSTRAINTS,
    },
    # FOOD - food photography
    SceneType.FOOD: {
        CanonicalCategory.SUBJECT,      # Food items
        CanonicalCategory.SCENE,        # Table setting, props
        CanonicalCategory.COMPOSITION,  # Food arrangement
        CanonicalCategory.LIGHTING,     # Food lighting
        CanonicalCategory.STYLE,        # Food photography style
        CanonicalCategory.CONSTRAINTS,
    },
    # STILL_LIFE - artistic object arrangements
    SceneType.STILL_LIFE: {
        CanonicalCategory.SUBJECT,      # Objects
        CanonicalCategory.SCENE,        # Background
        CanonicalCategory.COMPOSITION,  # Arrangement
        CanonicalCategory.LIGHTING,     # Artistic lighting
        CanonicalCategory.STYLE,        # Artistic style
        CanonicalCategory.CONSTRAINTS,
    },
    # LANDSCAPE - wide environmental shots (NO subject/fashion)
    SceneType.LANDSCAPE: {
        CanonicalCategory.SCENE,        # Environment is the focus
        CanonicalCategory.COMPOSITION,  # Landscape framing
        CanonicalCategory.LIGHTING,     # Natural lighting
        CanonicalCategory.STYLE,        # Landscape aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # ARCHITECTURE - buildings, interiors
    SceneType.ARCHITECTURE: {
        CanonicalCategory.SUBJECT,      # Building/structure details
        CanonicalCategory.SCENE,        # Location context
        CanonicalCategory.COMPOSITION,  # Architectural framing
        CanonicalCategory.LIGHTING,     # Architectural lighting
        CanonicalCategory.STYLE,        # Architectural style
        CanonicalCategory.CONSTRAINTS,
    },
    # STREET - urban photography
    SceneType.STREET: {
        CanonicalCategory.SUBJECT,      # People/objects in scene
        CanonicalCategory.SCENE,        # Urban environment
        CanonicalCategory.COMPOSITION,  # Street framing
        CanonicalCategory.LIGHTING,     # Natural/urban lighting
        CanonicalCategory.STYLE,        # Documentary style
        CanonicalCategory.CONSTRAINTS,
    },
    # CINEMATIC - film-style compositions
    SceneType.CINEMATIC: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,
        CanonicalCategory.COMPOSITION,  # Cinematic framing crucial
        CanonicalCategory.LIGHTING,     # Dramatic lighting
        CanonicalCategory.STYLE,        # Film aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # EDITORIAL - magazine/fashion editorial
    SceneType.EDITORIAL: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.LIGHTING,
        CanonicalCategory.STYLE,
        CanonicalCategory.CONSTRAINTS,
    },
    # ANIME - anime/illustration styles (different subject handling)
    SceneType.ANIME: {
        CanonicalCategory.SUBJECT,      # Character details
        CanonicalCategory.SCENE,        # Background
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.STYLE,        # Anime style crucial
        CanonicalCategory.CONSTRAINTS,
    },
    # VINTAGE - retro/film aesthetics
    SceneType.VINTAGE: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.LIGHTING,
        CanonicalCategory.STYLE,        # Vintage style crucial
        CanonicalCategory.CONSTRAINTS,
    },
    # CYBERPUNK - sci-fi/futuristic
    SceneType.CYBERPUNK: {
        CanonicalCategory.SUBJECT,
        CanonicalCategory.SCENE,        # Futuristic environment
        CanonicalCategory.COMPOSITION,
        CanonicalCategory.LIGHTING,     # Neon lighting
        CanonicalCategory.STYLE,        # Cyberpunk aesthetic
        CanonicalCategory.CONSTRAINTS,
    },
    # TEXT_OVERLAY - images with text elements
    SceneType.TEXT_OVERLAY: {
        CanonicalCategory.SUBJECT,      # Text content
        CanonicalCategory.SCENE,        # Background
        CanonicalCategory.COMPOSITION,  # Text placement
        CanonicalCategory.STYLE,
        CanonicalCategory.CONSTRAINTS,
    },
}

# Default: all categories relevant
DEFAULT_RELEVANT_CATEGORIES = set(CanonicalCategory)


# =============================================================================
# SCENE DETECTION KEYWORDS
# =============================================================================

SCENE_KEYWORDS: Dict[SceneType, Dict[str, float]] = {
    SceneType.PORTRAIT: {
        # High confidence
        "portrait": 1.0, "headshot": 1.0, "close-up": 0.8, "face": 0.7,
        "1girl": 0.6, "1boy": 0.6, "solo": 0.5, "looking at viewer": 0.6,
        "upper body": 0.7, "head": 0.5, "selfie": 0.8,
        # Medium confidence
        "eyes": 0.4, "lips": 0.4, "smile": 0.3, "expression": 0.4,
    },
    SceneType.FASHION: {
        "fashion": 1.0, "full body": 0.8, "outfit": 0.9, "model": 0.7,
        "dress": 0.6, "clothing": 0.5, "pose": 0.5, "standing": 0.4,
        "elegant": 0.5, "stylish": 0.6, "haute couture": 1.0,
        "runway": 0.9, "designer": 0.7, "vogue": 0.8,
    },
    SceneType.ACTION: {
        "action": 1.0, "dynamic": 0.8, "motion": 0.9, "running": 0.8,
        "jumping": 0.8, "fighting": 0.7, "sports": 0.7, "movement": 0.7,
        "athletic": 0.6, "exercise": 0.6, "weightlifting": 0.7,
        "dancing": 0.7, "martial arts": 0.8,
    },
    SceneType.GROUP: {
        "group": 1.0, "multiple": 0.8, "crowd": 0.9, "people": 0.6,
        "2girls": 0.7, "2boys": 0.7, "3+": 0.8, "team": 0.7,
        "audience": 0.7, "6+boys": 0.9, "6+girls": 0.9,
        "multiple boys": 0.8, "multiple girls": 0.8,
    },
    SceneType.PRODUCT: {
        "product": 1.0, "commercial": 0.8, "advertisement": 0.8,
        "white background": 0.7, "studio": 0.5, "isolated": 0.6,
        "object": 0.4, "item": 0.4, "packshot": 1.0,
    },
    SceneType.FOOD: {
        "food": 1.0, "dish": 0.9, "meal": 0.9, "cuisine": 0.8,
        "plate": 0.6, "restaurant": 0.7, "cooking": 0.7, "delicious": 0.6,
        "appetizing": 0.7, "gourmet": 0.8, "recipe": 0.7,
    },
    SceneType.STILL_LIFE: {
        "still life": 1.0, "arrangement": 0.8, "objects": 0.5,
        "table": 0.4, "flowers": 0.5, "vase": 0.6, "fruit": 0.5,
        "composition": 0.3, "artistic": 0.4,
    },
    SceneType.LANDSCAPE: {
        "landscape": 1.0, "scenery": 0.9, "nature": 0.7, "outdoor": 0.6,
        "mountain": 0.8, "sky": 0.5, "horizon": 0.7, "vista": 0.9,
        "panorama": 0.9, "sunset": 0.6, "sunrise": 0.6, "beach": 0.6,
        "forest": 0.7, "ocean": 0.7, "field": 0.6,
    },
    SceneType.ARCHITECTURE: {
        "architecture": 1.0, "building": 0.8, "interior": 0.8,
        "exterior": 0.7, "room": 0.6, "house": 0.5, "structure": 0.6,
        "cathedral": 0.9, "skyscraper": 0.8, "modern architecture": 1.0,
    },
    SceneType.STREET: {
        "street": 0.9, "urban": 0.8, "city": 0.7, "downtown": 0.8,
        "sidewalk": 0.7, "alley": 0.7, "crosswalk": 0.7,
        "street photography": 1.0, "candid": 0.6, "documentary": 0.6,
    },
    SceneType.CINEMATIC: {
        "cinematic": 1.0, "film": 0.7, "movie": 0.7, "dramatic": 0.7,
        "noir": 0.9, "moody": 0.6, "atmospheric": 0.6,
        "anamorphic": 0.9, "widescreen": 0.7, "theatrical": 0.8,
    },
    SceneType.EDITORIAL: {
        "editorial": 1.0, "magazine": 0.9, "vogue": 0.9, "fashion editorial": 1.0,
        "high fashion": 0.9, "glamour": 0.7, "beauty shot": 0.8,
        "cover": 0.6, "spread": 0.7,
    },
    SceneType.ANIME: {
        "anime": 1.0, "manga": 0.9, "illustration": 0.7, "2d": 0.6,
        "cel-shaded": 0.9, "cartoon": 0.6, "animated": 0.7,
        "japanese animation": 1.0, "studio ghibli": 1.0, "pixiv": 0.8,
    },
    SceneType.VINTAGE: {
        "vintage": 1.0, "retro": 0.9, "old": 0.5, "classic": 0.6,
        "film grain": 0.8, "polaroid": 0.9, "analog": 0.8,
        "nostalgic": 0.7, "1970s": 0.9, "1980s": 0.9, "1950s": 0.9,
    },
    SceneType.CYBERPUNK: {
        "cyberpunk": 1.0, "futuristic": 0.8, "sci-fi": 0.7, "neon": 0.7,
        "dystopian": 0.9, "blade runner": 1.0, "holographic": 0.8,
        "tech": 0.5, "cyber": 0.9, "synthwave": 0.8,
    },
    SceneType.TEXT_OVERLAY: {
        "text": 0.7, "typography": 1.0, "sign": 0.6, "logo": 0.6,
        "lettering": 0.9, "words": 0.6, "neon sign": 0.8,
        "graffiti": 0.7, "poster": 0.7, "title": 0.6,
    },
}


# =============================================================================
# CANONICAL CATEGORY KEYWORDS
# =============================================================================

CATEGORY_KEYWORDS: Dict[CanonicalCategory, Dict[str, Set[str]]] = {
    CanonicalCategory.SUBJECT: {
        "person_type": {
            "1girl", "1boy", "woman", "man", "girl", "boy", "child", "elderly",
            "model", "person", "figure", "character", "subject", "solo",
            "couple", "group", "crowd", "people", "multiple",
            # Extended human detection terms (WD14 tags)
            "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple girls",
            "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple boys",
            "1other", "androgynous", "adult", "human",
        },
        "body_parts": {
            "face", "eyes", "lips", "hair", "nose", "ears", "hands", "arms",
            "legs", "body", "torso", "head", "shoulders", "neck", "chest",
            "back", "feet", "fingers", "nails", "skin",
            # Extended body terms
            "thighs", "breasts", "cleavage", "navel", "midriff",
            # Additional body parts
            "stomach", "belly", "collarbone", "abs", "waist", "hips",
            "ankles", "wrists", "elbows", "knees", "chin", "jaw", "forehead",
            "cheeks", "eyebrows", "eyelashes", "mouth", "tongue", "teeth",
            "armpits", "bare shoulders", "bare arms", "bare legs", "bare feet",
            "female face", "male face", "human face",
            # NudeNet exposure terms (body visibility, not content flags)
            "exposed belly", "exposed armpits", "exposed feet",
            "covered female breast", "covered male chest", "covered buttocks",
            "covered belly", "covered feet", "covered armpits",
            "covered female genitalia", "covered male genitalia",
        },
        "attributes": {
            "young", "old", "tall", "short", "thin", "muscular", "athletic",
            "beautiful", "handsome", "elegant", "stylish", "casual",
            # Body type descriptors
            "toned", "curvy", "slim", "fit", "petite", "voluptuous",
            "slender", "thick", "plump", "lean", "buff", "ripped",
            "wide hips", "narrow waist", "thigh gap", "hourglass figure",
            # Size descriptors
            "medium breasts", "large breasts", "small breasts",
            "thick lips", "thin lips", "full lips", "pink lips",
            # Body visibility descriptors
            "sideboob", "underboob", "cleavage visible", "back exposed",
        },
        "appearance": {
            # Makeup
            "makeup", "lipstick", "eyeshadow", "eyeliner", "mascara",
            "blush", "foundation", "contour", "highlighter", "lip gloss",
            "natural makeup", "heavy makeup", "no makeup", "minimal makeup",
            "smoky eyes", "winged eyeliner", "red lipstick", "nude lipstick",
            # Skin/complexion
            "tan", "pale", "fair skin", "dark skin", "freckles", "moles",
            "clear skin", "glowing skin", "dewy skin", "matte skin",
            "very dark skin", "light skin", "olive skin", "bronze skin",
            # Ethnicity/appearance descriptors
            "asian", "caucasian", "african", "latina", "hispanic",
            "dark-skinned female", "dark-skinned male", "light-skinned",
            "tanned", "sun-kissed", "bronzed",
            # Other appearance
            "piercings", "tattoos", "scars", "beauty mark",
            "mole", "mole on arm", "mole on face", "mole on body",
            "scar", "scar on leg", "scar on arm", "scar on face",
            # Condition/state
            "wet", "sweaty", "oily", "dry",
        },
        "hair": {
            "blonde", "brunette", "redhead", "black hair", "brown hair",
            "long hair", "short hair", "curly", "straight", "wavy",
            "ponytail", "braid", "bun", "bangs",
            # Extended hair (WD14 tags)
            "blonde hair", "white hair", "silver hair", "gray hair", "pink hair",
            "blue hair", "green hair", "purple hair", "red hair", "orange hair",
            "multicolored hair", "streaked hair", "gradient hair", "two-tone hair",
            "hair over one eye", "hair between eyes", "messy hair", "wet hair",
            "hair bun", "twintails", "pigtails", "side ponytail", "high ponytail",
            "low ponytail", "french braid", "twin braids", "bob cut", "pixie cut",
            "hime cut", "ahoge", "hair ornament", "hair ribbon", "hairpin",
        },
        "expression": {
            "smile", "smiling", "laughing", "serious", "neutral", "happy",
            "sad", "angry", "surprised", "contemplative", "pensive",
            # Extended expressions (WD14 tags)
            "closed eyes", "open mouth", "looking at viewer", "wink",
            "grin", "frown", "crying", "blushing", "embarrassed",
            "expressionless", "blank expression", "stoic", "deadpan",
            "pouting", "smirk", "scowl", "grimace", "confused",
        },
        "clothing": {
            # Basic clothing
            "dress", "shirt", "pants", "skirt", "jacket", "coat", "suit",
            "jeans", "shorts", "blouse", "sweater", "hoodie", "t-shirt",
            "shoes", "boots", "sneakers", "heels", "accessories",
            "jewelry", "hat", "glasses", "watch", "bag",
            "sports bra", "sportswear", "athletic wear", "athleisure", "activewear",
            # Underwear/Intimate apparel
            "bra", "underwear", "panties", "lingerie", "briefs", "boxers",
            "thong", "bikini", "swimsuit", "swimwear", "bathing suit",
            "resort wear", "beachwear", "coverup", "beach coverup", "sarong",
            "white bra", "white panties", "white sports bra", "training bra",
            "underwear only", "bralette", "corset", "bustier", "teddy",
            "negligee", "babydoll", "chemise", "slip", "camisole set",
            # Extended clothing types
            "gown", "jumpsuit", "romper", "bodysuit", "cardigan", "vest",
            "blazer", "trench coat", "parka", "windbreaker", "denim jacket",
            "leather jacket", "bomber jacket", "crop top", "tank top",
            "polo shirt", "button-up", "oxford shirt", "flannel",
            "maxi dress", "midi dress", "mini dress", "cocktail dress",
            "evening gown", "wedding dress", "sundress", "wrap dress",
            "pencil skirt", "a-line skirt", "pleated skirt", "midi skirt",
            "maxi skirt", "mini skirt", "high-waisted", "low-rise",
            "wide-leg pants", "skinny jeans", "straight-leg", "bootcut",
            "cargo pants", "joggers", "sweatpants", "leggings", "culottes",
            # Ethnic/Traditional wear
            "sari", "saree", "lehenga", "salwar kameez", "kurta", "sherwani",
            "dupatta", "churidar", "anarkali", "gharara", "sharara",
            "kimono", "yukata", "hakama", "obi", "furisode",
            "hanbok", "jeogori", "chima",
            "cheongsam", "qipao", "changshan", "hanfu",
            "ao dai", "ao ba ba",
            "kebaya", "sarong", "batik",
            "dashiki", "kente", "agbada", "boubou", "kaftan",
            "dirndl", "lederhosen", "kilt",
            "abaya", "hijab", "niqab", "burqa", "thobe", "dishdasha", "bisht",
            "poncho", "huipil", "rebozo",
            # Robes and ceremonial
            "robe", "bathrobe", "toga", "roman toga", "red robe",
            "caftan", "muumuu", "housecoat", "dressing gown",
            # Length descriptors
            "floor length", "ankle length", "knee length", "tea length",
            "full length", "maxi length", "midi length", "mini length",
            "hip length", "thigh length", "cropped", "crop length",
            "waist length", "longline", "tunic length", "micro length",
            # Fabric types
            "silk", "cotton", "linen", "wool", "cashmere", "velvet",
            "satin", "chiffon", "lace", "tulle", "organza", "taffeta",
            "denim", "leather", "suede", "corduroy", "tweed", "flannel",
            "latex", "pvc", "vinyl", "rubber", "neoprene", "spandex", "lycra",
            "sequin", "beaded", "embroidered", "printed", "pleated", "draped",
            "ruched", "ruffled", "gathered", "smocked", "shirred",
            # Patterns
            "floral", "striped", "plaid", "checkered", "polka dot",
            "paisley", "geometric", "abstract", "animal print", "leopard print",
            "zebra print", "camouflage", "tie-dye", "ombre", "color block",
            "ikat", "batik print", "tribal", "ethnic print", "brocade",
            # Necklines/Details
            "v-neck", "crew neck", "scoop neck", "halter", "off-shoulder",
            "strapless", "one-shoulder", "turtleneck", "cowl neck",
            "sweetheart neckline", "square neckline", "boat neck",
            "ruffled", "lace-trimmed", "embellished", "tailored",
            "v-neckline", "deep v-neck", "plunging neckline", "round neckline",
            "bateau neckline", "high neck", "mock neck", "keyhole neckline",
            "illusion neckline", "off-shoulder neckline", "bandeau neckline",
            "peter pan collar", "mandarin collar", "stand collar", "notched collar",
            # Sleeve types
            "sleeveless", "cap sleeves", "short sleeves", "elbow sleeves",
            "three-quarter sleeves", "long sleeves", "full sleeves",
            "puff sleeves", "balloon sleeves", "bishop sleeves", "bell sleeves",
            "flutter sleeves", "angel sleeves", "batwing sleeves", "dolman sleeves",
            "raglan sleeves", "kimono sleeves", "juliet sleeves",
            "cold shoulder", "slit sleeves", "ruffle sleeves",
            # Extended tops
            "graphic tee", "plain tee", "v-neck shirt", "button-up shirt",
            "collared shirt", "polo shirt", "oxford shirt", "chambray shirt",
            "dress shirt", "knit sweater", "cable knit", "pullover hoodie",
            "zip-up hoodie", "sweatshirt", "camisole", "halter top",
            "peplum top", "wrap top", "tube top", "bandeau top", "henley shirt",
            # Extended dresses
            "a-line dress", "bodycon dress", "sheath dress", "shift dress",
            "empire waist dress", "fit and flare dress", "mermaid dress",
            "trumpet dress", "ballgown", "princess dress", "slip dress",
            "peasant dress", "tiered dress", "smock dress", "pinafore dress",
            "jumper dress", "tunic dress", "ball gown", "prom dress",
            "formal gown", "bridal gown", "bridesmaid dress", "flapper dress",
            "vintage dress", "retro dress", "boho dress", "lace dress",
            "sequin dress", "beaded dress", "floral dress", "backless dress",
            # Extended bottoms
            "ripped jeans", "high-waisted jeans", "low-rise jeans",
            "mom jeans", "boyfriend jeans", "dress pants", "chinos",
            "khakis", "trousers", "slacks", "denim shorts", "cargo shorts",
            "athletic shorts", "bermuda shorts", "yoga pants", "track pants",
            "palazzo pants", "capris", "cigarette pants", "harem pants",
            "drop crotch pants", "tapered pants",
            "straight leg jeans", "wide leg jeans", "bootcut jeans",
            # Extended outerwear
            "suit jacket", "sport coat", "faux leather jacket", "moto jacket",
            "jean jacket", "trucker jacket", "varsity jacket", "flight jacket",
            "overcoat", "peacoat", "wool coat", "puffer jacket", "down jacket",
            "quilted jacket", "rain jacket", "anorak", "cape", "shawl",
            "kimono jacket", "cardigan coat",
            # Extended footwear
            "tennis shoes", "kitten heels", "platform heels",
            "chelsea boots", "cowboy boots", "rain boots", "winter boots",
            "flip flops", "slides", "penny loafers", "moccasins",
            "driving shoes", "ballet flats", "pointed toe flats", "mary janes",
            "brogues", "derby shoes", "monk straps",
            # Extended accessories
            "baseball cap", "beanie", "fedora", "beret", "bucket hat", "sun hat",
            "aviator sunglasses", "cat eye sunglasses", "round sunglasses",
            "eyeglasses", "silk scarf", "wool scarf", "infinity scarf",
            "necktie", "leather belt", "chain belt", "statement belt",
            "tote bag", "shoulder bag", "satchel", "messenger bag", "briefcase",
            "duffle bag", "pendant", "chain necklace", "hoop earrings",
            "stud earrings", "drop earrings", "bangle", "cuff bracelet",
            "luxury watch",
            # Extended materials
            "faux leather", "vegan leather", "jersey", "knit fabric",
            "crochet", "mesh", "sheer fabric", "sequins",
            "distressed", "ripped", "frayed", "stonewashed",
            "torn pants", "torn jeans", "torn clothes", "torn shirt",
            "no socks", "barefoot", "no shoes", "untied footwear",
            "metallic fabric", "holographic", "iridescent", "patent leather",
            "faux fur", "fleece", "quilted", "padded",
            # Extended patterns
            "floral pattern", "striped pattern", "plaid pattern", "checkered pattern",
            "polka dots", "snake print", "geometric pattern", "abstract pattern",
            "paisley pattern", "solid", "solid color", "two-tone", "multicolor",
            "houndstooth", "herringbone", "gingham",
            # Clothing details/text
            "clothes writing", "logo", "brand logo", "text on clothing",
            "english text", "graphic print", "slogan", "printed text",
            # Clothing brands (common tags)
            "nike", "nike (company)", "adidas", "puma", "converse",
            "vans", "reebok", "new balance", "gucci", "prada",
            "louis vuitton", "chanel", "versace", "dior", "balenciaga",
            "supreme", "off-white", "champion", "levis", "levi's",
            # Outfit colors
            "black outfit", "white outfit", "red outfit", "blue outfit",
            "green outfit", "pink outfit", "yellow outfit", "orange outfit",
            "purple outfit", "navy blue", "burgundy", "maroon", "olive green",
            "teal", "monochrome outfit", "color blocked",
            # Uniforms
            "uniform", "military uniform", "police uniform", "nurse uniform",
            "doctor uniform", "school uniform", "sailor uniform", "maid uniform",
            "chef uniform", "flight attendant uniform", "pilot uniform",
            "firefighter uniform", "sports uniform", "team jersey",
            "athletic uniform", "business suit", "corporate attire",
            "office wear", "service uniform", "hospitality uniform", "security uniform",
            # Cosplay/Costume
            "cosplay", "costume", "cosplay outfit", "character costume",
            "anime cosplay", "video game cosplay", "superhero costume",
            "fantasy costume", "medieval costume", "renaissance costume",
            "halloween costume", "theatrical costume", "stage costume",
            "historical costume", "period costume", "sci-fi costume",
            # Religious/Ceremonial clothing
            "nun", "habit", "traditional nun", "coif", "wimple", "cornette",
            "cassock", "vestment", "surplice", "chasuble", "stole", "alb",
            "priest", "monk", "friar", "clergy", "religious habit",
            "hijab", "niqab", "burqa", "abaya", "chador", "jilbab",
            "kippa", "yarmulke", "tallit", "tzitzit",
            "buddhist robe", "kasaya", "saffron robe",
            "ceremonial robe", "ritual clothing", "liturgical vestment",
            # Extended traditional wear
            "traditional clothing", "cultural attire", "ethnic wear", "traditional dress",
            "silk sari", "cotton sari", "banarasi sari", "kanjeevaram sari",
            "designer sari", "lehenga choli", "bridal lehenga", "designer lehenga",
            "ghagra choli", "salwar suit", "punjabi suit", "anarkali suit",
            "palazzo suit", "kurti", "kurta pajama", "achkan", "bandhgala",
            "nehru jacket", "chunni", "stole", "pashmina", "chaniya choli",
            "pattu pavadai", "half sari", "langa voni", "dhoti", "lungi", "mundu",
            "choli", "crop blouse", "backless blouse", "patiala",
            "indo-western", "fusion wear", "contemporary ethnic",
            "agal", "keffiyeh", "turban", "pagri",
            # Fit/Silhouette
            "fitted clothing", "loose fit", "oversized fit", "relaxed fit",
            "tailored clothing", "structured silhouette", "flowy silhouette",
            "mid-rise", "form-fitting", "body-hugging", "baggy", "boxy",
            # Extended dresses
            "knee-length dress", "floor-length dress", "shirt dress",
            "strapless dress", "one-shoulder dress", "off-shoulder dress",
            # Extended tops
            "off-shoulder top", "flannel shirt",
            # Extended necklines
            "halter neckline", "strapless neckline",
            # Footwear
            "sandals", "loafers", "oxford shoes", "pumps", "stilettos",
            "wedges", "flats", "mules", "espadrilles", "ankle boots",
            "knee-high boots", "thigh-high boots", "combat boots",
            "platform shoes", "athletic shoes", "running shoes",
            # Accessories
            "necklace", "bracelet", "earrings", "ring", "brooch",
            "scarf", "tie", "bow tie", "belt", "suspenders",
            "handbag", "clutch", "backpack", "tote", "crossbody bag",
            "sunglasses", "reading glasses", "headband", "hair clip",
            "crown", "tiara", "veil", "fascinator",
        },
        "pose": {
            # Basic stances
            "standing", "sitting", "lying", "walking", "running", "jumping",
            "dancing", "posing", "leaning", "crouching", "kneeling",
            "arms up", "arm up", "arms crossed", "hands on hips",
            "hand up", "hands up", "hand in own hair", "hand in hair",
            # Extended stances (from pose vocabulary)
            "standing straight", "standing relaxed", "standing tall",
            "seated", "sitting cross-legged", "sitting on floor",
            "on stool", "on chair", "on bed", "on couch", "on sofa",
            "squatting", "bending", "leaning forward", "leaning back",
            "lying down", "reclining", "prone position",
            "on back", "on side", "on stomach", "on all fours",
            "spread legs", "legs spread", "knees apart",
            "wading", "swimming", "floating", "diving",
            # Arm positions
            "arms raised", "one arm raised", "arms overhead",
            "arms at sides", "arms relaxed", "arms behind back",
            "hand on hip", "akimbo pose", "arms folded", "hugging self",
            "hands in pockets", "hand in pocket",
            "hands clasped", "hands together", "prayer hands",
            "pointing", "gesturing", "waving",
            "touching face", "hand on chin", "hand on cheek",
            "arms outstretched", "arms extended", "reaching",
            # Leg positions
            "legs together", "legs apart", "wide stance",
            "one leg forward", "weight on one leg", "contrapposto",
            "legs crossed", "crossed ankles",
            "walking pose", "mid-stride", "stepping forward",
            "kicking", "mid-air",
            # Head/face orientations
            "facing camera", "facing viewer", "facing away", "profile view",
            "three-quarter view",
            "looking at camera", "looking away", "looking down", "looking up",
            "looking to the side", "looking left", "looking right", "side glance",
            "head straight", "head tilted", "head tilted left", "head tilted right",
            "chin up", "chin down", "head turned",
            "looking over shoulder", "glancing sideways",
            # Position relative to environment
            "against wall", "leaning on wall", "leaning against wall",
            "against door", "in doorway", "in window", "by window",
            "on floor", "on ground", "on grass", "on sand",
            # Body language/mood
            "confident pose", "relaxed pose", "casual pose", "formal stance",
            "power pose", "open body language", "closed body language",
            "dynamic pose", "static pose", "action pose",
            "elegant pose", "graceful pose", "dramatic pose",
            "playful pose", "serious expression", "contemplative pose",
            # Photography pose types
            "model pose", "fashion pose", "editorial pose",
            "portrait pose", "headshot",
            "candid pose", "natural pose", "posed shot",
            "glamour pose", "beauty shot", "lifestyle pose",
            "selfie", "mirror selfie", "group selfie",
        },
        # Wildlife and animals
        "animal": {
            # Big cats
            "lion", "tiger", "leopard", "cheetah", "jaguar", "panther",
            "cougar", "puma", "lynx", "bobcat", "caracal", "serval",
            # Bears
            "bear", "grizzly bear", "polar bear", "black bear", "panda",
            "koala",
            # Canines
            "wolf", "fox", "coyote", "jackal", "dingo", "hyena",
            "dog", "puppy", "husky", "german shepherd", "golden retriever",
            "labrador", "poodle", "bulldog", "beagle", "corgi", "shiba inu",
            # Felines
            "cat", "kitten", "persian cat", "siamese cat", "maine coon",
            "tabby cat", "calico cat", "black cat",
            # African wildlife
            "elephant", "giraffe", "zebra", "rhinoceros", "hippopotamus",
            "wildebeest", "gazelle", "antelope", "impala", "oryx", "kudu",
            "buffalo", "warthog", "meerkat", "lemur",
            # North American wildlife
            "deer", "elk", "moose", "caribou", "bison", "buffalo",
            "raccoon", "skunk", "badger", "beaver", "otter", "porcupine",
            "squirrel", "chipmunk", "groundhog", "prairie dog",
            # Asian wildlife
            "red panda", "snow leopard", "tiger", "asian elephant",
            "orangutan", "gibbon", "macaque", "langur",
            # Primates
            "monkey", "gorilla", "chimpanzee", "orangutan", "baboon",
            "mandrill", "capuchin", "spider monkey", "howler monkey",
            # Marine mammals
            "dolphin", "whale", "orca", "seal", "sea lion", "walrus",
            "manatee", "dugong", "narwhal", "beluga whale", "humpback whale",
            # Birds
            "bird", "eagle", "hawk", "falcon", "owl", "parrot", "peacock",
            "flamingo", "swan", "duck", "goose", "heron", "crane", "stork",
            "pelican", "seagull", "albatross", "penguin", "puffin",
            "hummingbird", "cardinal", "blue jay", "robin", "sparrow",
            "crow", "raven", "magpie", "woodpecker", "toucan", "macaw",
            "cockatoo", "budgie", "canary", "finch", "kingfisher",
            "chicken", "rooster", "turkey", "pheasant", "quail",
            # Birds of prey
            "bald eagle", "golden eagle", "red-tailed hawk", "osprey",
            "peregrine falcon", "kestrel", "vulture", "condor",
            "snowy owl", "barn owl", "great horned owl",
            # Reptiles
            "snake", "python", "cobra", "viper", "boa", "rattlesnake",
            "lizard", "gecko", "iguana", "chameleon", "monitor lizard",
            "komodo dragon", "crocodile", "alligator", "caiman",
            "turtle", "tortoise", "sea turtle",
            # Amphibians
            "frog", "toad", "salamander", "newt", "axolotl",
            # Fish
            "fish", "shark", "ray", "manta ray", "stingray",
            "clownfish", "angelfish", "betta fish", "goldfish", "koi",
            "salmon", "trout", "bass", "tuna", "swordfish", "marlin",
            "seahorse", "jellyfish", "octopus", "squid", "cuttlefish",
            # Insects/Arachnids
            "butterfly", "moth", "dragonfly", "damselfly", "bee",
            "bumblebee", "wasp", "hornet", "beetle", "ladybug",
            "grasshopper", "cricket", "mantis", "cicada", "firefly",
            "ant", "spider", "tarantula", "scorpion", "centipede",
            # Farm animals
            "horse", "pony", "donkey", "mule", "zebra",
            "cow", "bull", "calf", "ox", "yak",
            "sheep", "lamb", "goat", "kid", "ram",
            "pig", "piglet", "boar", "hog",
            "llama", "alpaca", "camel", "dromedary",
            # Small pets
            "rabbit", "bunny", "hamster", "guinea pig", "gerbil",
            "chinchilla", "ferret", "hedgehog", "sugar glider",
            # Mythical/Fictional
            "dragon", "phoenix", "unicorn", "griffin", "pegasus",
            # General terms
            "animal", "wildlife", "creature", "beast", "mammal",
            "predator", "prey", "herbivore", "carnivore",
            "animal focus", "no humans",
        },
        # Content ratings (describes what the image contains)
        "content_rating": {
            "safe content", "suggestive content", "nsfw content",
            "explicit content", "questionable content", "mature content",
            "adult content", "sensitive",
        },
        # Clothing state (describes how subject's clothing appears)
        "clothing_state": {
            "fully clothed", "partially undressed", "clothes lift",
            "shirt lift", "skirt lift", "dress lift", "pants pull",
            "unbuttoned", "partially unbuttoned", "unzipped", "open shirt",
            "loose clothing", "wet clothing", "see-through clothing",
            "tight clothing", "revealing outfit", "skimpy outfit", "barely covered",
            "open clothes", "unbuttoned shirt",
        },
    },
    CanonicalCategory.SCENE: {
        "location_indoor": {
            "studio", "room", "interior", "indoor", "indoors", "inside",
            "bedroom", "living room", "kitchen", "bathroom", "office",
            "gym", "restaurant", "cafe", "bar", "club", "museum",
            "gallery", "theater", "stadium", "arena",
            # Extended indoor locations
            "lobby", "hallway", "corridor", "staircase", "elevator",
            "elevator door", "door", "doorway", "entrance", "exit",
            "library", "classroom", "laboratory", "hospital", "clinic",
            "salon", "spa", "hotel", "apartment", "penthouse", "loft",
            "warehouse", "factory", "garage", "workshop", "attic",
            "basement", "cellar", "wine cellar", "storage room",
            "dressing room", "locker room", "fitting room", "changing room",
            "beauty salon", "hair salon", "nail salon", "makeup room",
            "conference room", "boardroom", "reception", "waiting room",
            "ballroom", "banquet hall", "auditorium", "concert hall",
            "cinema", "movie theater", "arcade", "bowling alley",
            "nightclub", "discotheque", "disco", "club", "bar", "pub", "lounge",
            "church", "cathedral", "mosque", "temple", "synagogue",
            "train station", "subway", "airport", "bus station",
            "shopping mall", "boutique", "store", "supermarket",
        },
        "location_outdoor": {
            "outdoor", "outdoors", "outside", "street", "park", "garden",
            "beach", "mountain", "forest", "field", "meadow", "desert",
            "ocean", "lake", "river", "city", "urban", "rural",
            "water", "sea", "shore", "nature", "wilderness", "wild",
            "countryside", "rooftop", "balcony", "terrace",
            # Extended outdoor locations
            "savanna", "prairie", "steppe", "tundra", "arctic",
            "tropical", "rainforest", "jungle", "woodland", "grove",
            "canyon", "valley", "gorge", "ravine", "cliff",
            "waterfall", "stream", "creek", "pond", "marsh", "swamp",
            "wetland", "delta", "estuary", "lagoon", "bay", "cove",
            "harbor", "port", "marina", "pier", "dock", "boardwalk",
            "promenade", "boulevard", "avenue", "alley", "courtyard",
            "plaza", "square", "marketplace", "bazaar", "souk",
            "vineyard", "orchard", "farm", "ranch", "pasture",
            "village", "town", "suburb", "downtown", "skyline",
            "highway", "freeway", "bridge", "tunnel", "overpass",
            "parking lot", "construction site", "industrial area",
            "ruins", "ancient ruins", "archaeological site",
            "cemetery", "graveyard", "memorial", "monument",
            "playground", "sports field", "golf course", "tennis court",
            "swimming pool", "ski slope", "ski resort", "campsite",
            "trail", "hiking path", "mountain path", "scenic overlook",
        },
        "time": {
            "day", "night", "morning", "evening", "afternoon", "dusk",
            "dawn", "sunset", "sunrise", "golden hour", "blue hour",
            "midnight", "noon", "twilight",
            # Seasons
            "summer", "winter", "spring", "fall", "autumn",
        },
        "weather": {
            # Basic weather
            "sunny", "cloudy", "rainy", "snowy", "foggy", "misty",
            "stormy", "windy", "overcast", "clear", "hazy",
            # Extended weather
            "partly cloudy", "mostly cloudy", "scattered clouds",
            "thunderstorm", "lightning", "thunder", "rain", "drizzle",
            "downpour", "shower", "monsoon", "tropical storm", "hurricane",
            "snow", "blizzard", "snowfall", "flurries", "ice storm",
            "sleet", "hail", "frost", "freezing", "cold front",
            "fog", "dense fog", "mist", "smog", "dust", "sandstorm",
            "humid", "dry", "arid", "drought", "heat wave",
            "breeze", "gust", "gale", "tornado", "cyclone",
        },
        "sky": {
            # Sky conditions
            "sky", "blue sky", "clear sky", "open sky", "vast sky",
            "dramatic sky", "moody sky", "stormy sky", "overcast sky",
            # Cloud types
            "clouds", "cloud", "cloudscape", "cloud formation",
            "cirrus", "cirrus clouds", "wispy clouds",
            "cumulus", "cumulus clouds", "fluffy clouds", "puffy clouds",
            "stratus", "stratus clouds", "layered clouds",
            "cumulonimbus", "storm clouds", "thunderhead",
            "altocumulus", "altostratus", "nimbostratus",
            "lenticular clouds", "mammatus clouds", "contrails",
            # Celestial
            "sun", "sunlight", "sunbeam", "sun rays", "solar",
            "moon", "moonlight", "full moon", "crescent moon", "half moon",
            "lunar", "moonrise", "moonset",
            "stars", "starry", "starry sky", "starfield", "star trail",
            "constellation", "milky way", "galaxy", "nebula", "cosmos",
            "aurora", "aurora borealis", "northern lights",
            "aurora australis", "southern lights",
            "meteor", "meteor shower", "shooting star", "comet",
            "planet", "venus", "mars", "jupiter", "saturn",
            # Sky phenomena
            "rainbow", "double rainbow", "sun halo", "moon halo",
            "sundogs", "light pillars", "crepuscular rays", "god rays",
            "alpenglow", "afterglow", "zodiacal light",
            "eclipse", "solar eclipse", "lunar eclipse",
        },
        "atmosphere": {
            "peaceful", "calm", "serene", "energetic", "vibrant",
            "mysterious", "romantic", "dramatic", "tense", "relaxed",
            "cozy", "warm", "cold", "eerie", "magical",
            # Extended atmosphere
            "tranquil", "idyllic", "pastoral", "bucolic",
            "majestic", "grand", "sublime", "awe-inspiring",
            "melancholic", "nostalgic", "wistful", "bittersweet",
            "ominous", "foreboding", "haunting", "spooky", "creepy",
            "ethereal", "dreamlike", "surreal", "fantastical",
            "intimate", "secluded", "private", "isolated", "remote",
            "bustling", "lively", "crowded", "busy", "hectic",
            "desolate", "barren", "abandoned", "empty", "lonely",
        },
        "background": {
            "background", "backdrop", "scenery", "environment", "setting",
            "blurred background", "blurry background", "out of focus background",
            # Studio/solid backgrounds
            "simple background", "plain background", "solid background",
            "grey background", "gray background", "white background",
            "black background", "blue background", "green background",
            "red background", "pink background", "yellow background",
            "gradient background", "textured background", "studio background",
            "seamless background", "paper background", "muslin background",
            "bokeh background", "neutral background", "dark background",
            "light background", "colorful background",
            # Walls and surfaces
            "brick wall", "brick background", "exposed brick", "red brick",
            "wall", "concrete wall", "stone wall", "wooden wall",
            "graffiti wall", "painted wall", "textured wall", "white wall",
            "industrial wall", "metal wall", "tile wall", "tiled wall",
        },
        "props_furniture": {
            # Seating
            "stool", "chair", "armchair", "sofa", "couch", "bench",
            "barstool", "office chair", "throne", "ottoman",
            # Tables/surfaces
            "table", "desk", "counter", "podium", "pedestal",
            # Beds
            "bed", "mattress", "futon", "daybed",
            # Studio props
            "backdrop stand", "light stand", "reflector", "umbrella",
            "props", "studio equipment",
        },
        # Famous landmarks and locations
        "landmark": {
            # Europe
            "eiffel tower", "louvre", "arc de triomphe", "notre dame",
            "big ben", "tower of london", "buckingham palace", "stonehenge",
            "colosseum", "leaning tower of pisa", "vatican", "sistine chapel",
            "sagrada familia", "alhambra", "parthenon", "acropolis",
            "neuschwanstein castle", "brandenburg gate", "berlin wall",
            "amsterdam canals", "windmills", "red square", "kremlin",
            "charles bridge", "prague castle", "schonbrunn palace",
            # Americas
            "statue of liberty", "empire state building", "times square",
            "brooklyn bridge", "central park", "golden gate bridge",
            "hollywood sign", "las vegas strip", "grand canyon",
            "yellowstone", "niagara falls", "mount rushmore",
            "white house", "capitol building", "lincoln memorial",
            "space needle", "cn tower", "christ the redeemer",
            "machu picchu", "chichen itza", "teotihuacan",
            # Asia
            "great wall of china", "forbidden city", "terracotta army",
            "tiananmen square", "shanghai skyline", "the bund",
            "mount fuji", "tokyo tower", "tokyo skyline", "shibuya crossing",
            "sensoji temple", "fushimi inari", "golden pavilion",
            "taj mahal", "red fort", "gateway of india", "varanasi ghats",
            "angkor wat", "petronas towers", "marina bay sands",
            "burj khalifa", "palm jumeirah", "petra", "wailing wall",
            "hagia sophia", "blue mosque", "cappadocia",
            # Africa/Oceania
            "pyramids of giza", "sphinx", "karnak temple",
            "victoria falls", "table mountain", "kilimanjaro",
            "serengeti", "masai mara", "sahara desert",
            "sydney opera house", "harbour bridge", "uluru", "ayers rock",
            "great barrier reef", "milford sound", "hobbiton",
            # Natural landmarks
            "grand canyon", "yosemite", "yellowstone", "zion",
            "swiss alps", "matterhorn", "mont blanc", "dolomites",
            "himalayas", "everest", "k2", "annapurna",
            "amazon rainforest", "galapagos", "iceland geysers",
            "northern lights", "dead sea", "cliffs of moher",
            "ha long bay", "zhangjiajie", "guilin", "santorini",
        },
    },
    CanonicalCategory.COMPOSITION: {
        "shot_type": {
            "close-up", "extreme close-up", "medium shot", "medium close-up",
            "full shot", "full body", "full body shot", "wide shot", "long shot",
            "extreme long shot", "establishing shot", "detail shot",
            "headshot", "portrait", "half body", "upper body",
            # Extended shot types (from photography vocabulary)
            "cowboy shot", "american shot", "over-the-shoulder",
            "point of view shot", "two-shot", "group shot",
            "insert shot", "cutaway", "reaction shot",
        },
        "angle": {
            "eye level", "low angle", "high angle", "bird's eye",
            "worm's eye", "dutch angle", "tilted", "straight on",
            "front view", "side view", "back view", "3/4 view",
            "profile", "from above", "from below",
            # Extended angles
            "overhead shot", "canted angle", "oblique angle",
            "from behind", "from side",
            "bird's eye view", "worm's eye view", "aerial view",
            "tilted frame", "tilted angle",
        },
        "framing": {
            "centered", "rule of thirds", "golden ratio", "symmetrical",
            "asymmetrical", "balanced", "off-center", "frame within frame",
            "leading lines", "diagonal", "horizontal", "vertical",
            # Extended framing (from composition vocabulary)
            "center composition", "triangular composition", "triangle composition",
            "pattern composition", "texture composition", "s-curve composition",
            "golden spiral", "diagonal composition", "negative space",
            "fill the frame", "minimalist composition", "layered depth",
            "horizontal symmetry", "vertical symmetry", "partial symmetry",
            "natural framing", "elements at thirds",
            # Framing terms from photography vocabulary
            "centered composition", "symmetrical composition", "asymmetrical composition",
            "framing", "dynamic composition", "static composition",
            # Subject positioning
            "subject at thirds intersection", "subject on vertical third",
            "subject on horizontal third", "centered subject", "central composition",
            "subject at golden ratio", "subject fills frame",
            "off-center subject", "asymmetric placement",
            "subject in upper left", "subject in upper right",
            "subject in lower left", "subject in lower right",
            "subject on left side", "subject on right side",
            "subject in upper area", "subject in lower area",
            "small subject in frame", "environmental context",
            "multiple subjects",
            # Line elements
            "horizontal lines", "vertical lines", "diagonal lines", "curved lines",
            "strong linear elements", "symmetry axis",
            # Additional composition elements
            "minimalist elements", "balanced composition", "dynamic tension",
            # Horizon/emphasis
            "high horizon", "low horizon", "ground emphasis", "sky emphasis",
            # Weight distribution
            "top-weighted composition", "bottom-weighted composition",
            "left-weighted composition", "right-weighted composition",
            "balanced weight", "visual weight",
        },
        "focus": {
            "shallow depth of field", "deep focus", "selective focus",
            "bokeh", "sharp", "soft focus", "tack sharp", "in focus",
            "out of focus", "blurred",
            # Extended focus terms
            "sharp focus", "slightly soft", "background blur", "foreground blur",
            "rack focus", "split focus", "deep depth of field",
        },
        "lens": {
            "wide angle", "telephoto", "macro", "fisheye", "tilt-shift",
            "35mm", "50mm", "85mm", "135mm", "24mm", "70-200mm",
            "prime lens", "zoom lens", "anamorphic",
        },
        "aspect": {
            "portrait orientation", "landscape orientation", "square",
            "16:9", "4:3", "2.39:1", "cinemascope", "widescreen",
            "vertical", "horizontal",
            # Extended aspect/format terms
            "square format", "horizontal frame", "vertical frame",
            "panoramic", "ultra-wide", "cinematic aspect ratio",
            # Common aspect ratios
            "2:3 aspect ratio", "3:2 aspect ratio", "4:5 aspect ratio",
            "1:1 aspect ratio", "9:16 aspect ratio", "16:9 aspect ratio",
            "1:1", "3:2", "4:5", "9:16", "21:9",
        },
    },
    CanonicalCategory.LIGHTING: {
        "direction": {
            "front lighting", "side lighting", "back lighting", "rim lighting",
            "top lighting", "under lighting", "ambient", "directional",
            "backlighting",
            "light from left", "light from right", "light from above",
            "light from below", "light from front", "light from back",
        },
        "quality": {
            "soft light", "hard light", "diffused", "harsh", "gentle",
            "soft lighting", "hard lighting", "diffused light",
            "natural lighting", "artificial lighting", "mixed lighting",
            "direct sunlight", "ambient light", "overcast lighting",
            "harsh shadows", "soft shadows",
            "even lighting", "dramatic lighting", "flat lighting",
            "high contrast", "low contrast", "soft shadows", "hard shadows",
            # Shadow descriptions
            "no visible shadows", "visible shadows", "soft shadow edges",
            "hard shadow edges", "minimal shadows", "deep shadows",
            "dramatic shadows", "light shadows", "heavy shadows",
        },
        "type": {
            "natural light", "artificial light", "studio lighting",
            "window light", "sunlight", "moonlight", "candlelight",
            "neon", "fluorescent", "led", "flash", "strobe",
        },
        "color": {
            "warm light", "cool light", "golden", "blue", "orange",
            "white balance", "color temperature", "tinted",
        },
        "style": {
            "rembrandt", "butterfly", "split", "loop", "broad", "short",
            "paramount", "clamshell", "beauty lighting",
            "chiaroscuro", "low key", "high key",
            # Extended lighting style names
            "rembrandt lighting", "butterfly lighting", "loop lighting",
            "split lighting", "high key lighting", "low key lighting",
            "silhouette", "magic hour", "flash photography",
        },
        "effects": {
            "volumetric", "god rays", "lens flare", "light leak",
            "specular", "highlights", "catchlights", "rim highlights",
            "glow", "halo", "backlit",
            # Specular/highlight effects
            "specular highlights", "catchlights visible", "rim light",
            "bright", "bright lighting", "properly exposed",
        },
    },
    CanonicalCategory.STYLE: {
        "medium": {
            "photograph", "photography", "photo", "digital", "film",
            "painting", "illustration", "drawing", "sketch", "render",
            "3d", "cgi", "artwork", "art",
            # Camera types
            "DSLR", "mirrorless", "point and shoot", "medium format",
            "full frame", "crop sensor", "smartphone", "instant camera",
            "film camera", "instant camera", "polaroid", "vintage camera",
            "webcam", "drone camera", "GoPro", "action camera",
            "professional studio camera", "disposable camera",
        },
        "camera_technical": {
            # Focal length
            "14mm", "24mm", "35mm", "50mm", "85mm", "135mm", "200mm", "400mm",
            "wide angle", "ultra wide angle", "standard lens", "portrait lens",
            "telephoto", "super telephoto", "macro lens", "fisheye",
            "wide angle lens", "telephoto lens", "zoom lens", "prime lens",
            "wide angle distortion", "telephoto compression",
            "wide angle perspective", "ultra wide angle perspective",
            "14mm ultra wide", "24mm wide angle", "35mm street",
            "50mm standard", "85mm portrait", "135mm telephoto", "200mm telephoto",
            # Aperture/f-stop
            "f/1.2", "f/1.4", "f/1.8", "f/2", "f/2.8", "f/4", "f/5.6", "f/8", "f/11", "f/16",
            "fast lens", "very fast lens", "moderate aperture", "standard aperture",
            "small aperture", "narrow aperture", "very narrow aperture",
            "wide open aperture", "stopped down",
            # ISO
            "ISO 100", "ISO 200", "ISO 400", "ISO 800", "ISO 1600", "ISO 3200", "ISO 6400",
            "base ISO", "low ISO", "moderate ISO", "high ISO", "very high ISO",
            # Shutter speed
            "long exposure", "slow shutter", "handheld shutter", "fast shutter",
            "action stopping", "high speed", "motion blur",
            "1/60s", "1/125s", "1/250s", "1/500s", "1/1000s", "1/2000s",
            # Exposure modes
            "manual", "program auto", "aperture priority", "shutter priority",
            "portrait mode", "landscape mode", "action program",
            # Metering
            "spot metering", "center-weighted metering", "average metering",
            "pattern metering", "partial metering", "multi-spot metering",
            # Flash
            "flash fired", "no flash", "fill flash", "bounce flash",
            # Lens characteristics
            "sharp corner to corner", "soft vignetting", "chromatic aberration",
            "starburst", "creamy bokeh", "smooth bokeh", "busy bokeh",
            "background blur", "foreground blur", "tilt-shift",
            "anamorphic", "soft focus lens",
            # Camera brands
            "Canon camera", "Nikon camera", "Sony camera", "Fujifilm camera",
            "Panasonic camera", "Olympus camera", "Leica camera", "Hasselblad camera",
        },
        "aesthetic": {
            "realistic", "photorealistic", "hyperrealistic", "stylized",
            "artistic", "cinematic", "editorial", "commercial",
            "fine art", "documentary", "candid", "glamour",
            "minimalist", "maximalist", "abstract",
            # Commercial/product terms
            "product placement", "product", "advertisement", "advertorial",
            "brand", "sponsored", "promotional",
            # Cultural/music aesthetics
            "k-pop", "j-pop", "idol", "korean style", "japanese style",
            # Retro/vintage aesthetics
            "pinup", "pinup style", "pin-up", "pin up", "retro pinup",
            "parody", "fine art parody", "art parody",
            # Reality style
            "real life", "real world", "realistic style",
            # Original/creative
            "original", "original character", "original content",
        },
        "genre": {
            "portrait photography", "fashion photography", "street photography",
            "landscape photography", "product photography", "food photography",
            "architectural photography", "sports photography",
            "wildlife photography", "macro photography",
            # Extended photography genres
            "beauty photography", "headshot photography", "boudoir photography",
            "environmental portrait", "lifestyle photography", "event photography",
            "concert photography", "travel photography", "nature photography",
            "underwater photography", "aerial photography", "drone photography",
            "night photography", "astrophotography", "long exposure photography",
            "black and white photography", "film photography", "instant photography",
            # Additional photography styles
            "editorial photography", "documentary photography", "candid photography",
            "studio photography", "outdoor photography", "location photography",
            "commercial photography", "fine art photography", "glamour photography",
            "posed portrait", "natural portrait",
        },
        "fashion_style": {
            # Fashion outfit styles (from FashionCLIP vocabulary)
            "casual outfit", "formal outfit", "business attire", "sportswear",
            "streetwear", "evening wear", "loungewear", "swimwear", "activewear",
            "workwear", "smart casual", "business casual", "semi-formal",
            "avant-garde fashion", "bohemian style", "boho chic", "preppy style",
            "punk style", "grunge style", "gothic fashion", "hippie style",
            "retro fashion", "vintage fashion", "geek chic", "normcore",
            "androgynous fashion", "gender-neutral clothing", "unisex outfit",
            "haute couture", "high fashion", "ready-to-wear", "designer fashion",
            "modest fashion", "conservative dress", "eclectic style",
            "maximalist fashion", "minimalist fashion", "chic style", "elegant style",
            "edgy style", "rocker style", "biker style", "classic style",
            "romantic style", "feminine style", "masculine style",
            "coastal style", "nautical style", "resort wear",
            "western style", "country style", "korean fashion", "japanese fashion",
            # Occasions
            "wedding attire", "bridal wear", "formal event", "black tie event",
            "cocktail party", "garden party", "beach wedding",
            "prom outfit", "graduation outfit", "job interview outfit",
            "date night outfit", "party outfit", "festival outfit",
            "vacation outfit", "travel outfit", "gym outfit", "workout wear",
        },
        "film_reference": {
            "kodak", "fujifilm", "ilford", "portra", "ektar", "tri-x",
            "cinestill", "polaroid", "vintage film", "film grain",
        },
        "art_reference": {
            "renaissance", "baroque", "impressionist", "modern",
            "contemporary", "pop art", "surreal", "fantasy",
            "anime", "manga", "studio ghibli", "pixar",
        },
        "color_grade": {
            "vibrant", "muted", "desaturated", "saturated",
            "warm tones", "cool tones", "neutral tones",
            "teal and orange", "monochrome", "black and white",
            "sepia", "cross-processed",
            # Extended color grading
            "color graded", "cinematic color", "film look",
            "bright", "dark", "moody", "airy", "dreamy",
            "high contrast", "low contrast", "balanced contrast",
            "pastel colors", "bright colors", "neon colors",
            "earth tones", "jewel tones",
            # Additional color terms
            "vibrant colors", "muted colors", "sepia tone",
            # Color descriptors from fashion/photography
            "cream primary color", "beige accent", "gray accent",
            "light gray tones", "multicolored", "monochromatic",
            "white", "black", "gray", "cream", "beige", "tan",
            "ivory", "off-white", "charcoal",
        },
        "texture": {
            # Material textures
            "matte finish", "glossy", "shiny", "smooth", "rough",
            "textured", "grainy", "soft", "silky", "velvet",
            "metallic", "iridescent", "pearlescent",
        },
    },
    CanonicalCategory.CONSTRAINTS: {
        "quality": {
            "high quality", "8k", "4k", "hd", "ultra hd", "detailed",
            "highly detailed", "sharp", "crisp", "clean", "professional",
            "masterpiece", "best quality",
            # Aesthetic quality descriptors
            "exceptional aesthetic quality", "excellent aesthetic quality",
            "good aesthetic quality", "average aesthetic quality",
            "below average aesthetic", "visually stunning",
            "pleasing composition", "professional quality", "amateur quality",
            "artistic merit", "visual appeal", "eye-catching",
            # Technical quality descriptors
            "excellent technical quality", "good technical quality",
            "acceptable technical quality", "low technical quality",
            "poor technical quality", "professional grade",
            "broadcast quality", "web quality", "print quality",
        },
        "technical": {
            "raw", "uncompressed", "high resolution", "noise-free",
            "sharp focus", "perfect focus", "accurate colors",
            # Extended technical terms
            "low resolution", "crisp details", "digital noise", "clean image",
            "long exposure", "motion blur", "frozen motion",
            "HDR", "high dynamic range", "bracketed exposure",
        },
        "clarity": {
            # Image clarity descriptors
            "sharp and clean", "excellent clarity", "good clarity",
            "minimal noise", "slight blur or noise", "noticeable blur",
            "visible noise", "significant blur", "heavy noise",
            "motion blur present", "camera shake", "out of focus areas",
            "tack sharp details",
        },
        "artifacts": {
            "no watermark", "no logo", "no text", "no blur",
            "no noise", "no grain", "no distortion",
            # Watermark/signature artifacts present
            "watermark", "signature", "artist name", "artist signature",
            "patreon username", "gumroad username", "deviantart logo",
            "web address", "url", "logo", "username", "social media handle",
            "copyright", "date stamp", "timestamp", "instagram handle",
        },
        "safety": {
            "safe for work", "sfw", "appropriate", "clean",
            "family friendly", "non-offensive",
        },
        "nsfw_exposed": {
            # NudeNet exposed classes
            "exposed female breast", "exposed male chest", "exposed buttocks",
            "exposed female genitalia", "exposed male genitalia", "exposed anus",
            "exposed belly", "exposed feet", "exposed armpits",
            "topless", "bottomless", "nude", "naked", "fully nude",
            "partial nudity", "implied nudity", "artistic nudity",
        },
        "nsfw_covered": {
            # NudeNet covered classes
            "covered female breast", "covered buttocks", "covered belly",
            "covered feet", "covered armpits", "covered female genitalia",
            "clothed", "partially clothed", "underwear visible",
            "bra visible", "panties visible", "lingerie",
        },
        "nsfw_body_parts": {
            # Additional body part terms for NSFW context (visibility terms like sideboob are in SUBJECT/attributes)
            "breasts", "cleavage", "underboob", "nipples",
            "buttocks", "ass", "thighs", "legs", "midriff", "navel",
            "back", "shoulders", "armpits", "feet", "toes",
            "crotch", "groin", "pubic area",
        },
        "nsfw_pose": {
            # Suggestive poses
            "seductive pose", "provocative pose", "alluring pose",
            "lying on bed", "lying on back", "lying on side",
            "legs spread", "legs apart", "bent over", "on all fours",
            "hands on hips", "arched back", "looking back",
            "over shoulder", "bedroom eyes", "sultry expression",
            "biting lip", "tongue out", "blowing kiss",
        },
        "nsfw_context": {
            # Context/location for suggestive content
            "bedroom", "bathroom", "shower", "bathtub", "pool",
            "beach", "locker room", "dressing room", "mirror selfie",
            "boudoir", "intimate setting", "private moment",
        },
        "nsfw_accessories": {
            # Accessories often associated with suggestive content
            "collar", "choker", "garter", "stockings", "fishnets",
            "high heels", "platform shoes", "thigh highs",
            "blindfold", "handcuffs", "rope", "chains",
        },
    },
}


# =============================================================================
# SYNONYM MAPPINGS FOR TAG NORMALIZATION
# =============================================================================

# Maps alternative terms to canonical terms in CATEGORY_KEYWORDS
# Format: "alternative_term": "canonical_term"
TAG_SYNONYMS: Dict[str, str] = {
    # Clothing synonyms
    "trousers": "pants",
    "slacks": "pants",
    "tee": "t-shirt",
    "tee shirt": "t-shirt",
    "tshirt": "t-shirt",
    "polo": "polo shirt",
    "button up": "button-up shirt",
    "button down": "button-up shirt",
    "pullover": "sweater",
    "jumper": "sweater",
    "trainers": "sneakers",
    "runners": "running shoes",
    "kicks": "sneakers",
    "tennis": "tennis shoes",
    "flips": "flip flops",
    "thongs": "flip flops",
    "specs": "glasses",
    "shades": "sunglasses",
    "sunnies": "sunglasses",
    "purse": "handbag",
    "pocketbook": "handbag",
    "rucksack": "backpack",
    "knapsack": "backpack",
    "cap": "hat",
    "beanie cap": "beanie",
    "ballcap": "baseball cap",
    "waistcoat": "vest",
    "gilet": "vest",
    "puffer": "puffer jacket",
    "mac": "rain jacket",
    "mackintosh": "rain jacket",
    "overalls": "jumpsuit",
    "dungarees": "jumpsuit",
    "onesie": "jumpsuit",
    "bathrobe": "loungewear",
    "dressing gown": "loungewear",
    "nightgown": "loungewear",
    "pjs": "loungewear",
    "pajamas": "loungewear",
    "pyjamas": "loungewear",
    "undershirt": "tank top",
    "singlet": "tank top",
    "wife beater": "tank top",
    "muscle shirt": "tank top",
    "cardi": "cardigan",
    "cardie": "cardigan",
    "trackies": "sweatpants",
    "track pants": "sweatpants",
    "yoga leggings": "leggings",
    "jeggings": "leggings",
    "mini": "mini skirt",
    "midi": "midi skirt",
    "maxi": "maxi skirt",
    "frock": "dress",
    "gown": "evening gown",
    "cocktail": "cocktail dress",
    "lbd": "mini dress",
    "little black dress": "mini dress",

    # Hair synonyms
    "blonde": "blonde hair",
    "brunette": "brown hair",
    "redhead": "red hair",
    "ginger": "red hair",
    "auburn": "red hair",
    "raven": "black hair",
    "platinum": "white hair",
    "silver": "silver hair",
    "grey": "gray hair",
    "locks": "long hair",
    "tresses": "long hair",
    "buzz cut": "short hair",
    "crew cut": "short hair",
    "updo": "hair bun",
    "topknot": "hair bun",
    "plaits": "braids",
    "pigtail": "pigtails",
    "twin tail": "twintails",
    "fringe": "bangs",

    # Expression synonyms
    "grinning": "grin",
    "frowning": "frown",
    "weeping": "crying",
    "smirk": "smile",
    "beaming": "smile",
    "scowl": "angry",
    "glare": "angry",
    "poker face": "neutral",
    "deadpan": "neutral",
    "pout": "sad",
    "pouting": "sad",

    # Pose synonyms
    "sat": "sitting",
    "seated": "sitting",
    "stood": "standing",
    "lay": "lying",
    "laying": "lying",
    "reclined": "reclining",
    "crouched": "crouching",
    "knelt": "kneeling",
    "squatted": "squatting",
    "bent": "bending",
    "leant": "leaning",
    "posed": "posing",
    "walked": "walking",
    "ran": "running",
    "jumped": "jumping",
    "danced": "dancing",
    "akimbo": "hands on hips",
    "arms akimbo": "hands on hips",

    # Animal synonyms
    "kitty": "cat",
    "kitten": "cat",
    "feline": "cat",
    "doggy": "dog",
    "puppy": "dog",
    "pup": "dog",
    "canine": "dog",
    "hound": "dog",
    "pooch": "dog",
    "birdie": "bird",
    "bunny": "rabbit",
    "hare": "rabbit",
    "horsie": "horse",
    "pony": "horse",
    "piggy": "pig",
    "piglet": "pig",
    "cub": "bear",
    "foal": "horse",
    "calf": "cow",
    "lamb": "sheep",
    "chick": "chicken",
    "duckling": "duck",
    "gosling": "goose",
    "fawn": "deer",
    "joey": "kangaroo",

    # Photography synonyms
    "telephoto lens": "telephoto",
    "wide lens": "wide angle",
    "macro lens": "macro",
    "fisheye lens": "fisheye",
    "portrait lens": "85mm",
    "nifty fifty": "50mm",
    "dof": "depth of field",
    "shallow dof": "shallow depth of field",
    "deep dof": "deep depth of field",
    "oof": "out of focus",
    "boke": "bokeh",

    # Lighting synonyms
    "backlight": "backlit",
    "back light": "backlit",
    "front light": "front lighting",
    "side light": "side lighting",
    "rim light": "rim lighting",
    "top light": "top lighting",
    "under light": "under lighting",
    "natural light": "natural lighting",
    "artificial light": "artificial lighting",
    "hard light": "hard lighting",
    "soft light": "soft lighting",
    "diffused": "diffused light",
    "harsh light": "hard lighting",
    "gentle light": "soft lighting",
    "golden light": "golden hour",
    "blue light": "blue hour",
    "magic light": "magic hour",
    "strobe": "flash photography",
    "flash": "flash photography",
    "speedlight": "flash photography",
    "studio light": "studio lighting",
    "ambient": "ambient light",
    "available light": "ambient light",

    # Composition synonyms
    "thirds": "rule of thirds",
    "centered": "center composition",
    "centred": "center composition",
    "symmetry": "symmetrical",
    "asymmetry": "asymmetrical",
    "close up": "close-up",
    "closeup": "close-up",
    "medium": "medium shot",
    "wide": "wide shot",
    "long": "long shot",
    "establishing": "establishing shot",
    "full length": "full shot",
    "head shot": "headshot",
    "head-shot": "headshot",
    "3/4": "3/4 view",
    "three quarter": "3/4 view",
    "three-quarter": "3/4 view",
    "birds eye": "bird's eye",
    "birdseye": "bird's eye",
    "worms eye": "worm's eye",
    "wormseye": "worm's eye",
    "overhead": "overhead shot",
    "topdown": "overhead shot",
    "top down": "overhead shot",
    "top-down": "overhead shot",
    "pov": "point of view shot",
    "ots": "over-the-shoulder",

    # Scene/Location synonyms
    "indoors": "indoor",
    "inside": "indoor",
    "interior": "indoor",
    "outdoors": "outdoor",
    "outside": "outdoor",
    "exterior": "outdoor",
    "daytime": "day",
    "nighttime": "night",
    "nightfall": "night",
    "daybreak": "dawn",
    "sunup": "sunrise",
    "sundown": "sunset",
    "dusk": "twilight",
    "cloudy": "overcast",
    "rainy": "rain",
    "snowy": "snow",
    "foggy": "fog",
    "misty": "mist",
    "stormy": "storm",
    "windy": "wind",
    "sunny": "clear",
    "woods": "forest",
    "woodland": "forest",
    "sea": "ocean",
    "seaside": "beach",
    "shore": "beach",
    "coastline": "coast",
    "riverside": "river",
    "lakeside": "lake",
    "mountainside": "mountain",
    "hilltop": "hill",
    "valley floor": "valley",
    "plains": "prairie",
    "grasslands": "grassland",
    "wetlands": "wetland",
    "swampland": "swamp",
    "marshland": "marsh",
    "wilderness": "nature",
    "wild": "nature",
    "countryside": "rural",
    "downtown": "urban",
    "cityscape": "city",
    "metropolis": "city",

    # Quality synonyms
    "hires": "high resolution",
    "hi-res": "high resolution",
    "hi res": "high resolution",
    "lores": "low resolution",
    "lo-res": "low resolution",
    "lo res": "low resolution",
    "4k": "4k",
    "8k": "8k",
    "uhd": "ultra hd",
    "ultra high definition": "ultra hd",
    "hd": "hd",
    "high definition": "hd",
    "blurry": "blurred",
    "fuzzy": "blurred",
    "noisy": "noise",
    "grainy": "grain",
    "crispy": "crisp",
    "tack sharp": "tack sharp",
    "razor sharp": "tack sharp",
    "pin sharp": "tack sharp",

    # Style synonyms
    "b&w": "black and white",
    "bw": "black and white",
    "b/w": "black and white",
    "monochromatic": "monochrome",
    "mono": "monochrome",
    "colour": "color",
    "colourful": "colorful",
    "vibrant": "vibrant colors",
    "muted": "muted colors",
    "desaturated": "muted colors",
    "saturated": "vibrant colors",
    "warm": "warm tones",
    "cool": "cool tones",
    "cold": "cool tones",
    "cinematic": "cinematic color",
    "filmic": "film look",
    "vintage": "vintage fashion",
    "retro": "retro fashion",
    "pinup (style)": "pinup",
    "pinup \\(style\\)": "pinup",
    "pin-up (style)": "pinup",
    "pin-up style": "pinup",
    "photo (medium)": "photograph",
    "photoshop (medium)": "digital",
    "digital (medium)": "digital",

    # NSFW synonyms
    "sfw": "safe for work",
    "nsfw": "nsfw content",
    "explicit": "explicit content",
    "suggestive": "suggestive content",
    "adult": "adult content",
    "mature": "mature content",
    "nude": "nude",
    "naked": "nude",
    "topless": "topless",
    "bottomless": "bottomless",
    "exposed": "exposed",
    "revealing": "revealing outfit",
    "skimpy": "skimpy outfit",
}

# Plural to singular mappings (common English plurals)
PLURAL_MAPPINGS: Dict[str, str] = {
    "dresses": "dress",
    "shirts": "shirt",
    "pants": "pants",  # pants is already singular in fashion context
    "jeans": "jeans",  # jeans is already singular
    "shorts": "shorts",  # shorts is already singular
    "skirts": "skirt",
    "jackets": "jacket",
    "coats": "coat",
    "suits": "suit",
    "shoes": "shoes",  # shoes is already singular in fashion context
    "boots": "boots",  # boots is already singular
    "sneakers": "sneakers",  # sneakers is already singular
    "heels": "heels",  # heels is already singular
    "glasses": "glasses",  # glasses is already singular
    "hats": "hat",
    "bags": "bag",
    "scarves": "scarf",
    "ties": "tie",
    "belts": "belt",
    "watches": "watch",
    "earrings": "earrings",  # earrings is already singular
    "necklaces": "necklace",
    "bracelets": "bracelet",
    "rings": "ring",
    "gloves": "gloves",  # gloves is already singular
    "socks": "socks",  # socks is already singular
    "stockings": "stockings",  # stockings is already singular
    "leggings": "leggings",  # leggings is already singular
    "animals": "animal",
    "birds": "bird",
    "cats": "cat",
    "dogs": "dog",
    "horses": "horse",
    "lions": "lion",
    "tigers": "tiger",
    "elephants": "elephant",
    "giraffes": "giraffe",
    "bears": "bear",
    "wolves": "wolf",
    "foxes": "fox",
    "rabbits": "rabbit",
    "fish": "fish",  # fish is same singular/plural
    "sharks": "shark",
    "whales": "whale",
    "dolphins": "dolphin",
    "butterflies": "butterfly",
    "trees": "tree",
    "flowers": "flower",
    "mountains": "mountain",
    "beaches": "beach",
    "forests": "forest",
    "rivers": "river",
    "lakes": "lake",
    "clouds": "cloud",
    "stars": "stars",  # stars often used as collective
    "shadows": "shadow",
    "highlights": "highlights",  # highlights is already singular in photo context
    "reflections": "reflection",
    "portraits": "portrait",
    "landscapes": "landscape",
    "photos": "photo",
    "photographs": "photograph",
    "pictures": "picture",
    "images": "image",

    # Escaped tag variants (from booru-style tags)
    "nike \\(company\\)": "nike (company)",
    "genderswap \\(mtf\\)": "genderswap",
    "genderswap \\(ftm\\)": "genderswap",
}


def normalize_tag(tag: str) -> str:
    """
    Normalize a tag using synonyms and plural mappings.

    Args:
        tag: Raw tag text

    Returns:
        Normalized tag text
    """
    tag_lower = tag.lower().strip()

    # Check synonym mapping first
    if tag_lower in TAG_SYNONYMS:
        return TAG_SYNONYMS[tag_lower]

    # Check plural mapping
    if tag_lower in PLURAL_MAPPINGS:
        return PLURAL_MAPPINGS[tag_lower]

    # Try removing common suffixes for plurals not in mapping
    if tag_lower.endswith('ies') and len(tag_lower) > 4:
        # e.g., "puppies" -> "puppy"
        singular = tag_lower[:-3] + 'y'
        if singular in TAG_SYNONYMS:
            return TAG_SYNONYMS[singular]
        return singular
    elif tag_lower.endswith('es') and len(tag_lower) > 3:
        # e.g., "dresses" -> "dress"
        singular = tag_lower[:-2]
        if singular in TAG_SYNONYMS:
            return TAG_SYNONYMS[singular]
        # Also try just removing 's' for words like "glasses"
        singular_s = tag_lower[:-1]
        if singular_s in TAG_SYNONYMS:
            return TAG_SYNONYMS[singular_s]
    elif tag_lower.endswith('s') and len(tag_lower) > 2:
        # e.g., "cats" -> "cat"
        singular = tag_lower[:-1]
        if singular in TAG_SYNONYMS:
            return TAG_SYNONYMS[singular]

    return tag_lower


# =============================================================================
# NLP-BASED TAG MATCHING
# =============================================================================

class NLPTagMatcher:
    """
    NLP-based semantic tag matching using sentence-transformers and spaCy.

    Features:
    - Semantic similarity matching (finds conceptually similar terms)
    - Lemmatization (handles verb forms, irregular plurals)
    - Lazy loading (models loaded only when needed)
    - Caching (embeddings computed once and reused)

    Usage:
        matcher = NLPTagMatcher()
        result = matcher.find_best_match("azure gown", threshold=0.7)
        # Returns: ("blue dress", 0.85, "SUBJECT", "clothing")
    """

    # Lightweight model - good balance of speed and quality
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    # Class-level flag to only warn once about missing dependencies
    _warned_no_transformers = False
    _warned_no_spacy = False

    def __init__(
        self,
        model_name: str = None,
        similarity_threshold: float = 0.40,  # Lowered for better semantic fallback
        use_lemmatization: bool = True,
    ):
        """
        Initialize NLP tag matcher.

        Args:
            model_name: Sentence-transformer model name (default: all-MiniLM-L6-v2)
            similarity_threshold: Minimum cosine similarity for a match (0-1)
            use_lemmatization: Whether to use spaCy lemmatization
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.similarity_threshold = similarity_threshold
        self.use_lemmatization = use_lemmatization

        # Lazy-loaded components
        self._model = None
        self._nlp = None
        self._keyword_embeddings = None
        self._keyword_list = None
        self._keyword_to_category = None
        self._is_initialized = False

    def _load_model(self) -> bool:
        """Load sentence-transformer model. Returns True if successful."""
        if self._model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            print(f"[NLPTagMatcher] Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"[NLPTagMatcher] Model loaded successfully")
            return True
        except ImportError:
            if not NLPTagMatcher._warned_no_transformers:
                NLPTagMatcher._warned_no_transformers = True
                print("[NLPTagMatcher] sentence-transformers not installed. "
                      "Install with: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"[NLPTagMatcher] Failed to load model: {e}")
            return False

    def _load_spacy(self) -> bool:
        """Load spaCy model. Returns True if successful."""
        if self._nlp is not None:
            return True

        if not self.use_lemmatization:
            return False

        try:
            import spacy
            # Try to load English model
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[NLPTagMatcher] Downloading spaCy English model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"],
                             check=True, capture_output=True)
                self._nlp = spacy.load("en_core_web_sm")
            print("[NLPTagMatcher] spaCy loaded successfully")
            return True
        except ImportError:
            if not NLPTagMatcher._warned_no_spacy:
                NLPTagMatcher._warned_no_spacy = True
                print("[NLPTagMatcher] spaCy not installed. "
                      "Install with: pip install spacy")
            return False
        except Exception as e:
            print(f"[NLPTagMatcher] Failed to load spaCy: {e}")
            return False

    def _build_keyword_index(self) -> None:
        """Build embeddings for all keywords in CATEGORY_KEYWORDS."""
        if self._keyword_embeddings is not None:
            return

        print("[NLPTagMatcher] Building keyword embeddings...")

        # Collect all keywords with their category info
        self._keyword_list = []
        self._keyword_to_category = {}

        for category, subcategories in CATEGORY_KEYWORDS.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    key = keyword.lower()
                    if key not in self._keyword_to_category:
                        self._keyword_list.append(key)
                        self._keyword_to_category[key] = (category, subcategory)

        # Compute embeddings for all keywords
        self._keyword_embeddings = self._model.encode(
            self._keyword_list,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # For faster cosine similarity
        )

        print(f"[NLPTagMatcher] Indexed {len(self._keyword_list)} keywords")

    def initialize(self) -> bool:
        """
        Initialize the NLP matcher (load models and build index).

        Returns True if initialization successful.
        """
        if self._is_initialized:
            return True

        # Load sentence-transformer model
        if not self._load_model():
            return False

        # Load spaCy (optional)
        if self.use_lemmatization:
            self._load_spacy()

        # Build keyword embeddings
        self._build_keyword_index()

        self._is_initialized = True
        return True

    def lemmatize(self, text: str) -> str:
        """
        Lemmatize text using spaCy.

        Args:
            text: Input text

        Returns:
            Lemmatized text (or original if spaCy not available)
        """
        if self._nlp is None:
            return text

        doc = self._nlp(text.lower())
        lemmas = [token.lemma_ for token in doc if not token.is_punct]
        return " ".join(lemmas)

    def find_best_match(
        self,
        tag: str,
        threshold: float = None,
    ) -> Optional[Tuple[str, float, 'CanonicalCategory', str]]:
        """
        Find the best matching keyword for a tag using semantic similarity.

        Args:
            tag: Input tag to match
            threshold: Override default similarity threshold

        Returns:
            Tuple of (matched_keyword, similarity, category, subcategory)
            or None if no match above threshold
        """
        if not self.initialize():
            return None

        threshold = threshold or self.similarity_threshold

        # Normalize the tag
        tag_lower = tag.lower().strip()

        # Try lemmatization first
        if self._nlp is not None:
            tag_lemmatized = self.lemmatize(tag_lower)
        else:
            tag_lemmatized = tag_lower

        # Compute embedding for the tag
        tag_embedding = self._model.encode(
            [tag_lemmatized],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Compute cosine similarities (dot product since embeddings are normalized)
        similarities = np.dot(self._keyword_embeddings, tag_embedding)

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        if best_similarity >= threshold:
            matched_keyword = self._keyword_list[best_idx]
            category, subcategory = self._keyword_to_category[matched_keyword]
            return (matched_keyword, float(best_similarity), category, subcategory)

        # Log near-misses for debugging (similarity between 0.30 and threshold)
        if best_similarity >= 0.30:
            matched_keyword = self._keyword_list[best_idx]
            print(f"[NLP] Near-miss: '{tag}' → '{matched_keyword}' ({best_similarity:.3f} < {threshold})")

        return None

    def find_top_matches(
        self,
        tag: str,
        top_k: int = 5,
        threshold: float = None,
    ) -> List[Tuple[str, float, 'CanonicalCategory', str]]:
        """
        Find top-k matching keywords for a tag.

        Args:
            tag: Input tag to match
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold

        Returns:
            List of (matched_keyword, similarity, category, subcategory) tuples
        """
        if not self.initialize():
            return []

        threshold = threshold or self.similarity_threshold

        # Normalize and lemmatize
        tag_lower = tag.lower().strip()
        if self._nlp is not None:
            tag_lower = self.lemmatize(tag_lower)

        # Compute embedding
        tag_embedding = self._model.encode(
            [tag_lower],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Compute similarities
        similarities = np.dot(self._keyword_embeddings, tag_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                keyword = self._keyword_list[idx]
                category, subcategory = self._keyword_to_category[keyword]
                results.append((keyword, float(sim), category, subcategory))

        return results

    def unload(self) -> None:
        """Unload models to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._nlp is not None:
            del self._nlp
            self._nlp = None
        if self._keyword_embeddings is not None:
            del self._keyword_embeddings
            self._keyword_embeddings = None
        self._is_initialized = False
        print("[NLPTagMatcher] Models unloaded")


# Global NLP matcher instance (lazy-loaded)
_nlp_matcher: Optional[NLPTagMatcher] = None


def get_nlp_matcher() -> NLPTagMatcher:
    """Get or create the global NLP matcher instance."""
    global _nlp_matcher
    if _nlp_matcher is None:
        _nlp_matcher = NLPTagMatcher()
    return _nlp_matcher


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SceneDetectionResult:
    """Result of scene type detection."""
    primary: SceneType
    primary_confidence: float
    secondary: Optional[SceneType] = None
    secondary_confidence: float = 0.0
    all_scores: Dict[SceneType, float] = field(default_factory=dict)


@dataclass
class CategoryItem:
    """A single item classified into a category."""
    text: str
    confidence: float
    source: str  # Which tagger it came from
    subcategory: str  # e.g., "hair", "lighting_direction"


@dataclass
class CanonicalStructure:
    """The full canonical structure with all categories."""
    scene_type: SceneDetectionResult
    subject: List[CategoryItem] = field(default_factory=list)
    scene: List[CategoryItem] = field(default_factory=list)
    composition: List[CategoryItem] = field(default_factory=list)
    lighting: List[CategoryItem] = field(default_factory=list)
    style: List[CategoryItem] = field(default_factory=list)
    constraints: List[CategoryItem] = field(default_factory=list)

    # Uncategorized tags (above threshold, not matched to any category)
    uncategorized: Dict[str, float] = field(default_factory=dict)

    # Below threshold tags (filtered out by tag_filter)
    below_threshold: Dict[str, float] = field(default_factory=dict)

    # Generated text for each category
    subject_text: str = ""
    scene_text: str = ""
    composition_text: str = ""
    lighting_text: str = ""
    style_text: str = ""
    constraints_text: str = ""

    # Full combined prompt
    full_prompt: str = ""


# =============================================================================
# SCENE TYPE DETECTOR
# =============================================================================

class SceneTypeDetector:
    """Detects scene type from tagger outputs."""

    def __init__(self):
        self.keywords = SCENE_KEYWORDS

    def detect(
        self,
        tags: Dict[str, float],
        florence_caption: str = "",
        florence_description: str = "",
    ) -> SceneDetectionResult:
        """
        Detect scene type from tags and Florence captions.

        Args:
            tags: Dict of tag -> confidence from all taggers
            florence_caption: Florence detailed caption
            florence_description: Florence more detailed description

        Returns:
            SceneDetectionResult with primary and optional secondary scene type
        """
        scores: Dict[SceneType, float] = {st: 0.0 for st in SceneType}

        # Normalize tags to lowercase
        tags_lower = {k.lower(): v for k, v in tags.items()}

        # Check for wildlife/nature override indicators
        has_no_humans = tags_lower.get("no humans", 0) > 0.5
        has_animal = tags_lower.get("animal", 0) > 0.5 or tags_lower.get("animal focus", 0) > 0.5

        # Score based on tag presence
        for scene_type, keywords in self.keywords.items():
            for keyword, weight in keywords.items():
                keyword_lower = keyword.lower()
                # Exact match
                if keyword_lower in tags_lower:
                    scores[scene_type] += tags_lower[keyword_lower] * weight
                # Partial match (keyword in tag)
                else:
                    for tag, conf in tags_lower.items():
                        if keyword_lower in tag:
                            scores[scene_type] += conf * weight * 0.5

        # Boost from Florence captions
        combined_caption = f"{florence_caption} {florence_description}".lower()
        for scene_type, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword.lower() in combined_caption:
                    scores[scene_type] += 0.3

        # Wildlife/Nature override: if "no humans" + "animal", force LANDSCAPE over PORTRAIT
        if has_no_humans and has_animal:
            # Boost landscape significantly
            scores[SceneType.LANDSCAPE] += 3.0
            # Reduce human-focused scene types
            scores[SceneType.PORTRAIT] *= 0.1
            scores[SceneType.FASHION] *= 0.1
            scores[SceneType.GROUP] *= 0.1

        # Person-in-scene override: if strong person detection, boost PORTRAIT/FASHION
        # This ensures "woman on beach" is PORTRAIT, not LANDSCAPE
        has_person = (
            tags_lower.get("1girl", 0) > 0.9 or
            tags_lower.get("1boy", 0) > 0.9 or
            tags_lower.get("solo", 0) > 0.9
        )
        if has_person and not has_no_humans:
            # Significant boost to person-focused scene types
            scores[SceneType.PORTRAIT] += 2.0
            scores[SceneType.FASHION] += 1.5
            # Reduce pure environment scene types
            scores[SceneType.LANDSCAPE] *= 0.5

        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_scores[0][0]
        primary_conf = sorted_scores[0][1]

        # Secondary only if significantly present
        secondary = None
        secondary_conf = 0.0
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.3:
            secondary = sorted_scores[1][0]
            secondary_conf = sorted_scores[1][1]

        return SceneDetectionResult(
            primary=primary,
            primary_confidence=primary_conf,
            secondary=secondary,
            secondary_confidence=secondary_conf,
            all_scores=scores,
        )


# =============================================================================
# CANONICAL CATEGORY CLASSIFIER
# =============================================================================

class CanonicalClassifier:
    """Classifies tags into canonical categories."""

    def __init__(self):
        self.category_keywords = CATEGORY_KEYWORDS
        self._build_reverse_index()

    def _build_reverse_index(self):
        """Build reverse index from keyword to (category, subcategory)."""
        self.reverse_index: Dict[str, Tuple[CanonicalCategory, str]] = {}
        for category, subcategories in self.category_keywords.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    key = keyword.lower()
                    # Prefer more specific matches
                    if key not in self.reverse_index:
                        self.reverse_index[key] = (category, subcategory)

    def classify_tag(
        self,
        tag: str,
        confidence: float,
        source: str,
    ) -> Optional[CategoryItem]:
        """
        Classify a single tag into a canonical category.

        Uses synonym mapping and plural normalization for better matching.

        Returns (CategoryItem, category) if classified, (None, None) if unclassified.
        """
        tag_lower = tag.lower().strip()

        # Step 1: Exact match on original tag
        if tag_lower in self.reverse_index:
            category, subcategory = self.reverse_index[tag_lower]
            return CategoryItem(
                text=tag,
                confidence=confidence,
                source=source,
                subcategory=subcategory,
            ), category

        # Step 2: Normalize using synonyms and plurals, then try exact match
        normalized = normalize_tag(tag_lower)
        if normalized != tag_lower and normalized in self.reverse_index:
            category, subcategory = self.reverse_index[normalized]
            return CategoryItem(
                text=tag,
                confidence=confidence,
                source=source,
                subcategory=subcategory,
            ), category

        # Step 3: Partial match - find best matching keyword
        best_match = None
        best_category = None
        best_subcategory = None

        # Try both original and normalized forms
        search_terms = {tag_lower, normalized}

        for search_term in search_terms:
            for keyword, (category, subcategory) in self.reverse_index.items():
                if keyword in search_term or search_term in keyword:
                    if best_match is None or len(keyword) > len(best_match):
                        best_match = keyword
                        best_category = category
                        best_subcategory = subcategory

        if best_match:
            return CategoryItem(
                text=tag,
                confidence=confidence,
                source=source,
                subcategory=best_subcategory,
            ), best_category

        # Step 3b: Component matching - split compound tags and try each word
        # This catches "brick wall" → "wall", "pink sports bra" → "sports bra" or "bra"
        components = tag_lower.replace("_", " ").replace("-", " ").split()
        if len(components) > 1:
            # Try progressively smaller n-grams: "sports bra", then individual words
            for n in range(len(components) - 1, 0, -1):
                for i in range(len(components) - n + 1):
                    component = " ".join(components[i:i + n])
                    if component in self.reverse_index:
                        category, subcategory = self.reverse_index[component]
                        return CategoryItem(
                            text=tag,
                            confidence=confidence * 0.95,  # Slight penalty for partial
                            source=source,
                            subcategory=subcategory,
                        ), category

        # Step 4: NLP semantic fallback (if sentence-transformers installed)
        try:
            nlp_matcher = get_nlp_matcher()
            nlp_result = nlp_matcher.find_best_match(tag)
            if nlp_result:
                matched_keyword, similarity, category_str, subcategory = nlp_result
                # Convert category string to enum
                try:
                    category = CanonicalCategory[category_str]
                except KeyError:
                    category = CanonicalCategory.STYLE  # Fallback
                return CategoryItem(
                    text=tag,
                    confidence=confidence * similarity,  # Adjust confidence by similarity
                    source=source,
                    subcategory=subcategory,
                ), category
        except Exception as e:
            # NLP not available or failed - log once for diagnosis
            if not hasattr(CanonicalClassifier, '_nlp_error_logged'):
                print(f"[CanonicalClassifier] NLP fallback unavailable: {e}")
                CanonicalClassifier._nlp_error_logged = True

        return None, None

    def classify_all(
        self,
        metadata: Dict[str, Dict[str, float]],
        threshold: float = 0.0,
    ) -> Tuple[Dict[CanonicalCategory, List[CategoryItem]], Dict[str, float], Dict[str, float]]:
        """
        Classify all tags from metadata into canonical categories.

        Args:
            metadata: Dict of source -> {tag: confidence}
            threshold: Minimum confidence to include in categories (0-1)

        Returns:
            Tuple of:
                - Dict of category -> list of CategoryItems (above threshold, categorized)
                - Dict of uncategorized tags -> confidence (above threshold, not categorized)
                - Dict of below_threshold tags -> confidence (below threshold)
        """
        # First pass: aggregate all tags with max confidence and track sources
        aggregated: Dict[str, Tuple[float, str]] = {}  # tag -> (max_conf, source)

        for source, tags in metadata.items():
            if not isinstance(tags, dict):
                continue
            # Skip non-tag metadata
            if source.startswith("florence_") or source == "image_info":
                continue
            # Skip bbox_analysis and canonical (not tag sources)
            if source in ("bbox_analysis", "canonical", "session_id", "tag_filter"):
                continue

            for tag, confidence in tags.items():
                if not isinstance(confidence, (int, float)):
                    continue

                # Keep highest confidence across all sources
                if tag not in aggregated or confidence > aggregated[tag][0]:
                    aggregated[tag] = (confidence, source)

        # Second pass: classify based on max confidence
        result = {cat: [] for cat in CanonicalCategory}
        uncategorized: Dict[str, float] = {}
        below_threshold: Dict[str, float] = {}

        for tag, (confidence, source) in aggregated.items():
            # Check threshold
            if confidence < threshold:
                below_threshold[tag] = confidence
                continue

            # Above threshold - try to classify
            item, category = self.classify_tag(tag, confidence, source)
            if item and category:
                result[category].append(item)
            else:
                # Unclassified
                uncategorized[tag] = confidence

        # Sort each category by confidence
        for category in result:
            result[category].sort(key=lambda x: x.confidence, reverse=True)

        # Sort uncategorized by confidence
        uncategorized = dict(sorted(uncategorized.items(), key=lambda x: x[1], reverse=True))

        # Sort below_threshold by confidence
        below_threshold = dict(sorted(below_threshold.items(), key=lambda x: x[1], reverse=True))

        return result, uncategorized, below_threshold


# =============================================================================
# TEXT GENERATOR
# =============================================================================

class CanonicalTextGenerator:
    """Generates natural language text for each category."""

    def generate_subject_text(self, items: List[CategoryItem]) -> str:
        """Generate subject description text."""
        if not items:
            return ""

        # Group by subcategory
        groups = self._group_by_subcategory(items)

        parts = []

        # Animals first (if present, likely wildlife photo)
        if "animal" in groups:
            animals = [item.text for item in groups["animal"][:3]]
            if animals:
                parts.append(", ".join(animals))

        # Person type
        if "person_type" in groups:
            person = groups["person_type"][0].text
            parts.append(person)

        # Attributes
        if "attributes" in groups:
            attrs = [item.text for item in groups["attributes"][:3]]
            if attrs:
                parts.append(", ".join(attrs))

        # Hair
        if "hair" in groups:
            hair = [item.text for item in groups["hair"][:2]]
            if hair:
                parts.append(f"with {' '.join(hair)}")

        # Expression
        if "expression" in groups:
            expr = groups["expression"][0].text
            parts.append(f"with {expr}")

        # Clothing
        if "clothing" in groups:
            clothes = [item.text for item in groups["clothing"][:5]]
            if clothes:
                parts.append(f"wearing {', '.join(clothes)}")

        # Pose
        if "pose" in groups:
            pose = groups["pose"][0].text
            parts.append(pose)

        return ". ".join(parts) if parts else ""

    def generate_scene_text(self, items: List[CategoryItem]) -> str:
        """Generate scene/environment description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Landmark first (if present, it's a notable location)
        if "landmark" in groups:
            landmark = groups["landmark"][0].text
            parts.append(f"at {landmark}")

        # Location
        location = None
        if "location_indoor" in groups:
            location = groups["location_indoor"][0].text
        elif "location_outdoor" in groups:
            location = groups["location_outdoor"][0].text

        if location:
            parts.append(f"in {location}")

        # Time
        if "time" in groups:
            time = groups["time"][0].text
            parts.append(f"at {time}")

        # Sky conditions
        if "sky" in groups:
            sky = [item.text for item in groups["sky"][:2]]
            if sky:
                parts.append(", ".join(sky))

        # Weather
        if "weather" in groups:
            weather = groups["weather"][0].text
            parts.append(f"{weather} conditions")

        # Atmosphere
        if "atmosphere" in groups:
            atmos = [item.text for item in groups["atmosphere"][:2]]
            if atmos:
                parts.append(f"{', '.join(atmos)} atmosphere")

        # Background
        if "background" in groups:
            bg = groups["background"][0].text
            parts.append(bg)

        return ". ".join(parts) if parts else ""

    def generate_composition_text(self, items: List[CategoryItem]) -> str:
        """Generate composition description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Shot type
        if "shot_type" in groups:
            shot = groups["shot_type"][0].text
            parts.append(shot)

        # Angle
        if "angle" in groups:
            angle = groups["angle"][0].text
            parts.append(f"from {angle}")

        # Framing
        if "framing" in groups:
            framing = [item.text for item in groups["framing"][:2]]
            if framing:
                parts.append(", ".join(framing))

        # Lens
        if "lens" in groups:
            lens = groups["lens"][0].text
            parts.append(f"shot with {lens}")

        # Focus
        if "focus" in groups:
            focus = groups["focus"][0].text
            parts.append(focus)

        # Aspect
        if "aspect" in groups:
            aspect = groups["aspect"][0].text
            parts.append(aspect)

        return ". ".join(parts) if parts else ""

    def generate_lighting_text(self, items: List[CategoryItem]) -> str:
        """Generate lighting description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Direction
        if "direction" in groups:
            direction = [item.text for item in groups["direction"][:2]]
            parts.append(", ".join(direction))

        # Quality
        if "quality" in groups:
            quality = [item.text for item in groups["quality"][:2]]
            parts.append(", ".join(quality))

        # Type
        if "type" in groups:
            light_type = groups["type"][0].text
            parts.append(light_type)

        # Style
        if "style" in groups:
            style = groups["style"][0].text
            parts.append(f"{style} lighting")

        # Color
        if "color" in groups:
            color = groups["color"][0].text
            parts.append(color)

        # Effects
        if "effects" in groups:
            effects = [item.text for item in groups["effects"][:3]]
            if effects:
                parts.append(f"with {', '.join(effects)}")

        return ". ".join(parts) if parts else ""

    def generate_style_text(self, items: List[CategoryItem]) -> str:
        """Generate style description text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Medium
        if "medium" in groups:
            medium = groups["medium"][0].text
            parts.append(medium)

        # Aesthetic
        if "aesthetic" in groups:
            aesthetic = [item.text for item in groups["aesthetic"][:2]]
            parts.append(", ".join(aesthetic))

        # Genre
        if "genre" in groups:
            genre = groups["genre"][0].text
            parts.append(genre)

        # Color grade
        if "color_grade" in groups:
            grade = [item.text for item in groups["color_grade"][:2]]
            parts.append(", ".join(grade))

        # Art reference
        if "art_reference" in groups:
            ref = groups["art_reference"][0].text
            parts.append(f"inspired by {ref}")

        # Film reference
        if "film_reference" in groups:
            film = groups["film_reference"][0].text
            parts.append(f"shot on {film}")

        return ". ".join(parts) if parts else ""

    def generate_constraints_text(self, items: List[CategoryItem]) -> str:
        """Generate constraints text."""
        if not items:
            return ""

        groups = self._group_by_subcategory(items)
        parts = []

        # Quality
        if "quality" in groups:
            quality = [item.text for item in groups["quality"][:3]]
            parts.append(", ".join(quality))

        # Technical
        if "technical" in groups:
            tech = [item.text for item in groups["technical"][:2]]
            parts.append(", ".join(tech))

        return ". ".join(parts) if parts else ""

    def _group_by_subcategory(
        self,
        items: List[CategoryItem]
    ) -> Dict[str, List[CategoryItem]]:
        """Group items by subcategory."""
        groups: Dict[str, List[CategoryItem]] = {}
        for item in items:
            if item.subcategory not in groups:
                groups[item.subcategory] = []
            groups[item.subcategory].append(item)
        return groups


# =============================================================================
# MAIN STRUCTURER
# =============================================================================

class CanonicalStructurer:
    """
    Main class that orchestrates scene detection, classification, and text generation.
    """

    def __init__(self):
        self.detector = SceneTypeDetector()
        self.classifier = CanonicalClassifier()
        self.generator = CanonicalTextGenerator()

    def structure(self, metadata: Dict, threshold: float = 0.0) -> CanonicalStructure:
        """
        Process metadata into canonical structure.

        Args:
            metadata: Full metadata dict from tagging pipeline
            threshold: Minimum confidence for categorization (from tag_filter)

        Returns:
            CanonicalStructure with all categories populated
        """
        # Collect all tags with confidences (for scene detection - uses all tags)
        all_tags = {}
        for source, data in metadata.items():
            if isinstance(data, dict) and not source.startswith("florence_"):
                if source not in ("image_info", "bbox_analysis", "canonical", "session_id", "tag_filter"):
                    for tag, conf in data.items():
                        if isinstance(conf, (int, float)):
                            # Keep highest confidence if duplicate
                            if tag not in all_tags or conf > all_tags[tag]:
                                all_tags[tag] = conf

        # Get Florence captions
        florence_caption = metadata.get("florence_caption", "")
        florence_description = metadata.get("florence_description", "")

        # Detect scene type (uses all tags, not filtered)
        scene_result = self.detector.detect(
            all_tags,
            florence_caption,
            florence_description,
        )

        # Classify into categories (applies threshold)
        classified, uncategorized, below_threshold = self.classifier.classify_all(metadata, threshold)

        # Create structure
        structure = CanonicalStructure(
            scene_type=scene_result,
            subject=classified[CanonicalCategory.SUBJECT],
            scene=classified[CanonicalCategory.SCENE],
            composition=classified[CanonicalCategory.COMPOSITION],
            lighting=classified[CanonicalCategory.LIGHTING],
            style=classified[CanonicalCategory.STYLE],
            constraints=classified[CanonicalCategory.CONSTRAINTS],
            uncategorized=uncategorized,
            below_threshold=below_threshold,
        )

        # Generate text for each category
        structure.subject_text = self.generator.generate_subject_text(structure.subject)
        structure.scene_text = self.generator.generate_scene_text(structure.scene)
        structure.composition_text = self.generator.generate_composition_text(structure.composition)
        structure.lighting_text = self.generator.generate_lighting_text(structure.lighting)
        structure.style_text = self.generator.generate_style_text(structure.style)
        structure.constraints_text = self.generator.generate_constraints_text(structure.constraints)

        # Generate full prompt
        structure.full_prompt = self._combine_prompt(structure)

        return structure

    def _combine_prompt(self, structure: CanonicalStructure) -> str:
        """Combine all category texts into full prompt."""
        parts = []

        if structure.subject_text:
            parts.append(structure.subject_text)
        if structure.scene_text:
            parts.append(structure.scene_text)
        if structure.composition_text:
            parts.append(structure.composition_text)
        if structure.lighting_text:
            parts.append(structure.lighting_text)
        if structure.style_text:
            parts.append(structure.style_text)
        if structure.constraints_text:
            parts.append(structure.constraints_text)

        return ". ".join(parts)

    def _items_to_dict(self, items: List[CategoryItem]) -> Dict[str, float]:
        """Convert items to simple {tag: confidence} dict, keeping highest confidence for duplicates."""
        result = {}
        for item in items:
            tag = item.text
            conf = round(item.confidence, 3)
            if tag not in result or conf > result[tag]:
                result[tag] = conf
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def to_json(self, structure: CanonicalStructure) -> Dict:
        """
        Convert structure to JSON-serializable dict.

        Applies scene-based category filtering - categories not relevant
        to the detected scene type are excluded from output.

        When secondary scene has high confidence (>0.7), its categories
        are also included to handle mixed scenes like "person on beach".
        """
        # Get relevant categories for this scene type
        scene_type = structure.scene_type.primary
        relevant = set(SCENE_CATEGORY_RELEVANCE.get(scene_type, DEFAULT_RELEVANT_CATEGORIES))

        # Include secondary scene's categories if confidence is moderate
        # Lowered from 0.7 to 0.5 to include SCENE for portrait+fashion combos
        if (structure.scene_type.secondary and
            structure.scene_type.secondary_confidence > 0.5):
            secondary_relevant = SCENE_CATEGORY_RELEVANCE.get(
                structure.scene_type.secondary, set()
            )
            relevant = relevant.union(secondary_relevant)

        result = {
            "scene_detection": {
                "primary": scene_type.value,
                "primary_confidence": round(structure.scene_type.primary_confidence, 3),
                "secondary": structure.scene_type.secondary.value if structure.scene_type.secondary else None,
                "secondary_confidence": round(structure.scene_type.secondary_confidence, 3),
            },
        }

        # Only include categories relevant to this scene type
        if CanonicalCategory.SUBJECT in relevant:
            result["subject"] = self._items_to_dict(structure.subject)

        if CanonicalCategory.SCENE in relevant:
            result["scene"] = self._items_to_dict(structure.scene)

        if CanonicalCategory.COMPOSITION in relevant:
            result["composition"] = self._items_to_dict(structure.composition)

        if CanonicalCategory.LIGHTING in relevant:
            result["lighting"] = self._items_to_dict(structure.lighting)

        if CanonicalCategory.STYLE in relevant:
            result["style"] = self._items_to_dict(structure.style)

        if CanonicalCategory.CONSTRAINTS in relevant:
            result["constraints"] = self._items_to_dict(structure.constraints)

        # Uncategorized and below_threshold are always included
        result["uncategorized"] = {
            tag: round(conf, 3) for tag, conf in structure.uncategorized.items()
        }
        result["below_threshold"] = {
            tag: round(conf, 3) for tag, conf in structure.below_threshold.items()
        }

        return result
