"""Canonical category definitions for token classification.

This module defines the 9 canonical categories used to organize
image tokens for prompt assembly.
"""

from enum import Enum, auto
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


class CanonicalCategory(Enum):
    """The 9 canonical categories for image description.

    Tokens are classified into these categories to enable
    structured, grammatically correct prompt assembly.
    """
    QUALITY_BOOSTERS = "quality_boosters"
    SUBJECT = "subject"
    SUBJECT_DETAILS = "subject_details"
    ACTION_POSE = "action_pose"
    ENVIRONMENT = "environment"
    LIGHTING = "lighting"
    STYLE_MEDIUM = "style_medium"
    COMPOSITION = "composition"
    TECHNICAL = "technical"

    # Special categories
    UNCATEGORIZED = "uncategorized"
    FILTERED_META = "filtered_meta"


class SubjectDetailType(Enum):
    """Subcategories within Subject Details."""
    HAIR = "hair"
    EYES = "eyes"
    FACE = "face"
    BODY = "body"
    CLOTHING = "clothing"
    SKIN = "skin"
    ACCESSORIES = "accessories"


# Category display names for output
CATEGORY_DISPLAY_NAMES: Dict[CanonicalCategory, str] = {
    CanonicalCategory.QUALITY_BOOSTERS: "Quality Boosters",
    CanonicalCategory.SUBJECT: "Subject",
    CanonicalCategory.SUBJECT_DETAILS: "Subject Details",
    CanonicalCategory.ACTION_POSE: "Action/Pose",
    CanonicalCategory.ENVIRONMENT: "Environment/Scene",
    CanonicalCategory.LIGHTING: "Lighting",
    CanonicalCategory.STYLE_MEDIUM: "Style/Medium",
    CanonicalCategory.COMPOSITION: "Composition",
    CanonicalCategory.TECHNICAL: "Technical Parameters",
    CanonicalCategory.UNCATEGORIZED: "Uncategorized",
    CanonicalCategory.FILTERED_META: "Filtered (META)",
}


# Canonical output order for prompt assembly
CATEGORY_ORDER: List[CanonicalCategory] = [
    CanonicalCategory.QUALITY_BOOSTERS,
    CanonicalCategory.STYLE_MEDIUM,
    CanonicalCategory.SUBJECT,
    CanonicalCategory.SUBJECT_DETAILS,
    CanonicalCategory.ACTION_POSE,
    CanonicalCategory.ENVIRONMENT,
    CanonicalCategory.LIGHTING,
    CanonicalCategory.COMPOSITION,
    CanonicalCategory.TECHNICAL,
]


# Subject details subcategory order
SUBJECT_DETAIL_ORDER: List[SubjectDetailType] = [
    SubjectDetailType.HAIR,
    SubjectDetailType.EYES,
    SubjectDetailType.FACE,
    SubjectDetailType.SKIN,
    SubjectDetailType.BODY,
    SubjectDetailType.CLOTHING,
    SubjectDetailType.ACCESSORIES,
]


@dataclass
class CategoryKeywords:
    """Keyword matching rules for a category.

    Used by the dictionary classifier (Layer 3) to match tokens
    to categories based on exact matches, patterns, and regex.
    """
    exact_matches: Set[str] = field(default_factory=set)
    prefix_patterns: List[str] = field(default_factory=list)
    suffix_patterns: List[str] = field(default_factory=list)
    contains_patterns: List[str] = field(default_factory=list)
    negative_keywords: Set[str] = field(default_factory=set)

    def matches(self, text: str) -> bool:
        """Check if text matches this category's keywords."""
        text_lower = text.lower().strip()

        # Check negative keywords first
        if text_lower in self.negative_keywords:
            return False

        # Exact match
        if text_lower in self.exact_matches:
            return True

        # Prefix patterns
        for pattern in self.prefix_patterns:
            if text_lower.startswith(pattern.rstrip("*")):
                return True

        # Suffix patterns
        for pattern in self.suffix_patterns:
            if text_lower.endswith(pattern.lstrip("*")):
                return True

        # Contains patterns
        for pattern in self.contains_patterns:
            if pattern.strip("*") in text_lower:
                return True

        return False


# =============================================================================
# Category Keyword Dictionaries
# =============================================================================

QUALITY_KEYWORDS = CategoryKeywords(
    exact_matches={
        "realistic", "photorealistic", "photo realistic", "high quality",
        "masterpiece", "best quality", "ultra detailed", "8k", "4k", "hdr",
        "sharp", "in focus", "detailed", "professional", "excellent quality",
        "good aesthetic quality", "excellent technical quality", "sharp and clean",
        "excellent clarity", "photograph", "photography", "photo medium",
        "high resolution", "ultra hd", "crisp", "clear",
    },
)

SUBJECT_KEYWORDS = CategoryKeywords(
    exact_matches={
        "1girl", "2girls", "3girls", "1boy", "2boys", "solo", "multiple girls",
        "multiple boys", "1woman", "1man", "couple", "group", "crowd",
        "female", "male", "person", "people", "model", "woman", "man",
        "girl", "boy", "lady", "gentleman", "figure", "individual",
        # Non-human subjects
        "no humans", "no human", "animal", "cat", "dog", "bird", "horse",
        "object", "food", "vehicle", "car", "boat", "airplane",
        # Vehicles
        "motor vehicle", "sports car", "racecar", "race car", "race vehicle",
        "vehicle focus", "motorcycle", "bike", "bicycle", "truck", "bus",
        "train", "ship", "helicopter", "jet", "aircraft",
        # Car brands
        "ferrari", "lamborghini", "porsche", "bmw", "mercedes", "audi",
        "nissan", "toyota", "honda", "ford", "chevrolet", "tesla",
        "maserati", "bugatti", "mclaren", "aston martin", "bentley",
        # Car parts
        "spoiler", "spoiler (automobile)", "wheel", "wheels", "tire", "tires",
        "headlight", "taillight", "bumper", "hood", "trunk", "engine",
        "steering wheel", "dashboard", "seat", "seats",
    },
    contains_patterns=["girl", "boy", "woman", "man", "vehicle", "car"],
)

SUBJECT_DETAIL_KEYWORDS: Dict[SubjectDetailType, CategoryKeywords] = {
    SubjectDetailType.HAIR: CategoryKeywords(
        exact_matches={
            "long hair", "short hair", "medium hair", "brown hair", "black hair",
            "blonde hair", "red hair", "white hair", "grey hair", "gray hair",
            "wavy hair", "curly hair", "straight hair", "ponytail", "twintails",
            "braid", "braids", "bun", "hair bun", "bangs", "side bangs",
            "hair over one eye", "hair between eyes", "ahoge", "bob cut",
            "pixie cut", "shoulder length hair", "very long hair",
            # Ethnic/cultural hairstyles
            "afro", "afro hairstyle", "african hairstyle", "dreadlocks", "locs",
            "cornrows", "box braids", "twist out", "natural hair", "coily hair",
            "kinky hair", "textured hair", "fade", "undercut", "mohawk",
            # Facial hair
            "beard", "full beard", "thick beard", "mustache", "thick mustache",
            "goatee", "stubble", "facial hair", "clean shaven", "beardlock",
            # Body hair
            "chest hair", "arm hair", "body hair", "hairy", "very hairy",
            "thick arm hair", "navel hair",
        },
        suffix_patterns=["*_hair", "* hair", "* hairstyle"],
        prefix_patterns=["hair_*", "hair *", "afro *", "african *"],
        contains_patterns=["hair", "beard", "hairstyle"],
    ),
    SubjectDetailType.EYES: CategoryKeywords(
        exact_matches={
            "brown eyes", "blue eyes", "green eyes", "red eyes", "purple eyes",
            "yellow eyes", "heterochromia", "closed eyes", "half-closed eyes",
            "looking at viewer", "looking away", "looking down", "looking up",
            "looking to the side", "eye contact", "empty eyes", "glowing eyes",
        },
        suffix_patterns=["*_eyes", "* eyes"],
        prefix_patterns=["eye_*", "eye *"],
    ),
    SubjectDetailType.FACE: CategoryKeywords(
        exact_matches={
            "lips", "parted lips", "closed mouth", "open mouth", "smile",
            "smiling", "serious", "neutral expression", "blush", "makeup",
            "lipstick", "eyeshadow", "freckles", "mole", "nose", "ears",
            "expression", "facial expression", "grin", "frown", "pout",
            "teeth", "tongue", "jaw", "chin", "cheeks", "forehead",
            "red lips", "pink lips", "lip gloss", "eyeliner", "mascara",
            "beauty mark", "dimples", "happy", "sad", "angry", "surprised",
            "confident", "alluring", "sensual", "sultry", "playful",
            # NudeNet face outputs
            "female face", "male face", "face", "head",
            # Mole positions on face
            "mole on face", "mole on cheek", "mole under eye",
            "mole near mouth", "mole on chin",
        },
        contains_patterns=["expression", "lips", "mouth", "makeup", "face"],
    ),
    SubjectDetailType.BODY: CategoryKeywords(
        exact_matches={
            "slim", "curvy", "athletic", "muscular", "petite", "tall", "short",
            "thin", "fit", "toned", "slender", "breasts", "large breasts",
            "medium breasts", "small breasts", "navel", "midriff", "abs",
            "thighs", "legs", "arms", "hands", "feet", "back", "shoulders",
            "stomach", "collarbone", "cleavage", "sideboob", "underboob",
            "armpits", "armpit", "hips", "wide hips", "narrow waist",
            "waist", "chest", "torso", "bare shoulders", "bare arms",
            "bare legs", "bare back", "bare feet", "fingernails", "toenails",
            "neck", "butt", "ass", "thigh gap", "belly", "tummy",
            # NudeNet outputs
            "exposed belly", "exposed armpits", "exposed arms", "exposed legs",
            "covered female breast", "covered male breast", "exposed breast",
            "covered female genitalia", "covered male genitalia",
            "exposed female breast", "exposed male breast",
            "covered buttocks", "exposed buttocks", "covered feet", "exposed feet",
            # Moles/beauty marks on body
            "mole on breast", "mole on arm", "mole on back", "mole on stomach",
            "mole on thigh", "mole on shoulder", "beauty mark on body",
            # Exposure/nudity descriptors
            "nude", "naked", "topless", "bottomless", "partially nude",
            "one breast out", "breast exposed", "nipple", "nipples",
            "exposed nipple", "exposed nipples", "areola", "areolae",
        },
        contains_patterns=["breasts", "thigh", "physique", "figure", "build", "hips",
                          "exposed", "covered", "nude", "naked"],
    ),
    SubjectDetailType.CLOTHING: CategoryKeywords(
        exact_matches={
            "dress", "shirt", "blouse", "top", "crop top", "t-shirt", "sweater",
            "jacket", "coat", "pants", "jeans", "shorts", "skirt", "miniskirt",
            "underwear", "panties", "bra", "bikini", "lingerie", "swimsuit",
            "uniform", "suit", "tie", "boots", "shoes", "heels", "sneakers",
            "socks", "stockings", "thighhighs", "hat", "glasses", "jewelry",
            "necklace", "earrings", "bracelet", "watch", "ring",
            # Bikini variants
            "bikini top", "bikini bottom", "string bikini", "micro bikini",
            "side-tie bikini", "side-tie bikini bottom", "highleg bikini",
            "gold bikini", "strapless", "halterneck", "one-piece swimsuit",
            "monokini", "tankini", "highleg", "highleg swimsuit",
            # More underwear
            "sports bra", "bralette", "panty", "thong", "briefs", "boxers",
            "underwear only", "no bra", "no panties",
            # Ethnic/traditional
            "sari", "saree", "lehenga", "salwar", "kameez", "kurta", "sherwani",
            "dupatta", "kimono", "hanbok", "cheongsam", "ao dai", "abaya",
            "hijab", "thobe", "kaftan", "indian clothes", "traditional",
            # Other
            "apron", "robe", "toga", "cape", "cloak", "vest", "cardigan",
            "hoodie", "blouse", "tunic", "gown", "cocktail dress",
            "red dress", "white dress", "black dress", "multicolored dress",
            "backless dress", "strapless dress", "two-tone dress",
            "multicolored clothes", "red sari", "white panties", "red robe",
            # Necklines and visibility
            "plunging neckline", "v-neckline", "deep neckline", "low neckline",
            "upper clothing visible", "lower clothing visible",
            "clothing visible", "partially clothed", "fully clothed",
            # Shoulder styles
            "one shoulder", "off shoulder", "off-shoulder", "bare shoulder",
            "cold shoulder", "asymmetric neckline", "asymmetrical top",
            "spaghetti straps", "strapless top", "tube top",
        },
        prefix_patterns=["wearing_*", "wearing *", "dressed_*"],
        contains_patterns=["dress", "shirt", "pants", "skirt", "top", "bra", "bikini", "swimsuit", "neckline", "clothing", "shoulder"],
    ),
    SubjectDetailType.SKIN: CategoryKeywords(
        exact_matches={
            "fair skin", "pale skin", "tan", "tanned", "dark skin", "brown skin",
            "olive skin", "light skin", "fair complexion", "complexion",
            "smooth skin", "freckled skin", "porcelain skin", "bronze skin",
            "golden skin", "caramel skin", "ebony skin", "ivory skin",
            # Ethnicity - General
            "asian", "caucasian", "african", "hispanic", "latina", "latino",
            "indian", "middle eastern", "european", "american",
            "dark-skinned female", "dark-skinned male", "light-skinned",
            # Ethnicity - Specific Asian
            "east asian", "south asian", "southeast asian", "chinese", "japanese",
            "korean", "vietnamese", "thai", "filipino", "filipina", "malaysian",
            "indonesian", "taiwanese", "singaporean",
            # Ethnicity - South Asian/Indian subcontinent
            "south indian", "north indian", "bengali", "punjabi", "gujarati",
            "marathi", "tamil", "telugu", "pakistani", "bangladeshi", "nepali",
            "sri lankan", "desi",
            # Ethnicity - Middle Eastern
            "arab", "arabian", "persian", "turkish", "iranian", "israeli",
            "lebanese", "syrian", "egyptian", "iraqi", "jordanian", "emirati",
            # Ethnicity - African
            "african american", "black", "west african", "east african",
            "north african", "nigerian", "ethiopian", "kenyan", "ghanaian",
            # Ethnicity - European
            "eastern european", "western european", "scandinavian", "nordic",
            "mediterranean", "slavic", "british", "french", "german", "italian",
            "spanish", "greek", "russian", "polish", "irish",
            # Ethnicity - Latin American
            "brazilian", "mexican", "colombian", "argentinian", "peruvian",
            "venezuelan", "cuban", "puerto rican", "dominican", "chilean",
            # Ethnicity - Mixed/Other
            "mixed race", "multiracial", "biracial", "eurasian", "afro-asian",
            "mestizo", "mestiza", "mulatto", "mulatta", "pacific islander",
            "polynesian", "hawaiian", "maori", "aboriginal", "indigenous",
        },
        contains_patterns=["skin", "complexion", "skinned"],
    ),
    SubjectDetailType.ACCESSORIES: CategoryKeywords(
        exact_matches={
            "glasses", "sunglasses", "earrings", "necklace", "bracelet", "watch",
            "ring", "piercing", "tattoo", "headband", "hairpin", "hair ornament",
            "choker", "scarf", "belt", "bag", "purse", "umbrella",
            "wristwatch", "wrist watch", "pendant", "anklet", "brooch",
            "hair clip", "hair tie", "ponytail holder", "ribbon",
            "crown", "tiara", "veil", "gloves", "mittens",
        },
        contains_patterns=["jewelry", "accessory", "ornament"],
    ),
}

ACTION_POSE_KEYWORDS = CategoryKeywords(
    exact_matches={
        "standing", "sitting", "lying", "lying down", "kneeling", "crouching",
        "walking", "running", "jumping", "dancing", "posing", "leaning",
        "arms at sides", "arms up", "arms crossed", "hands on hips",
        "hand on hip", "hands behind back", "v sign", "peace sign",
        "pointing", "waving", "reaching", "stretching",
        "facing viewer", "facing away", "from behind", "from side",
        "cowboy shot", "profile", "three-quarter view",
        # More arm/hand poses
        "arm up", "hand up", "arm behind head", "hand on head",
        "hand on own head", "hand in hair", "hand in own hair",
        "hands up", "arm at side", "crossed arms",
        "holding", "carrying", "eating", "drinking",
        # More body poses
        "head tilt", "head turn", "looking back", "looking over shoulder",
        "bending", "squatting", "on knees", "on back", "on side",
        "twisted torso", "arched back",
        # Leg poses
        "legs together", "legs apart", "legs crossed", "crossed legs",
        "one leg raised", "leg up", "legs spread", "knees together",
        "knees apart", "feet together", "feet apart",
        # Carrying/holding variations
        "carrying bag", "holding bag", "holding phone", "holding camera",
        "holding food", "holding drink", "holding cup", "holding glass",
    },
    suffix_patterns=["*_shot", "*ing"],
    prefix_patterns=["looking at *", "holding *", "carrying *"],
    contains_patterns=["hand on", "arm up", "hands", "legs", "holding", "carrying"],
)

ENVIRONMENT_KEYWORDS = CategoryKeywords(
    exact_matches={
        "indoor", "indoors", "outdoor", "outdoors", "background",
        "simple background", "white background", "grey background",
        "gray background", "black background", "gradient background",
        "plain background", "blurred background", "bokeh",
        "studio", "bedroom", "bathroom", "kitchen", "living room", "office",
        "street", "city", "urban", "nature", "forest", "beach", "ocean",
        "mountain", "sky", "sunset", "sunrise", "night", "day",
        "window", "door", "wall", "floor", "ceiling", "bed", "couch",
        # Nature/landscape
        "scenery", "landscape", "water", "lake", "river", "stream", "pond",
        "sea", "waterfall", "reflection", "reflective water", "snow", "ice",
        "cloud", "clouds", "cloudy sky", "fog", "mist", "rain", "storm",
        "tree", "trees", "plant", "plants", "flower", "flowers", "grass",
        "jungle", "tropical", "garden", "park", "field", "meadow",
        "hill", "hills", "valley", "desert", "canyon", "cliff",
        "mountainous horizon", "blue sky", "grey sky", "white sky",
        "pine tree", "palm tree", "fern", "foliage", "riverbank", "shore",
        "village", "town", "rural", "bridge", "road", "path", "trail",
        "building", "buildings", "house", "cabin", "balcony", "terrace",
        "railing", "fence", "lamp", "lamppost", "power lines",
        # Urban/automotive scenes
        "racetrack", "race track", "track", "garage", "parking lot",
        "parking", "showroom", "dealership", "highway", "freeway",
        "intersection", "traffic", "sign", "traffic light", "crosswalk",
        # Seasons and weather
        "winter", "summer", "spring", "autumn", "fall", "snowy", "rainy",
        "sunny", "cloudy", "foggy", "misty", "stormy", "windy",
        # Agricultural/rural
        "rice paddy", "rice field", "wheat field", "corn field", "farmland",
        "farm", "barn", "stable", "countryside", "pasture", "orchard",
        # Props and objects
        "food", "cake", "cupcake", "fruit", "strawberry", "cherry",
        "ice cream", "dessert", "pastry", "drink", "wine", "champagne",
        "coffee", "tea", "cup", "plate", "table", "chair", "sofa",
        "mirror", "lamp", "candle", "book", "phone", "camera",
        "umbrella", "balloon", "gift", "toy", "flower bouquet",
        "whipped cream", "frosting", "icing", "sprinkles", "chocolate",
        "candy", "cookie", "cookies", "biscuit", "muffin", "donut",
        # More landscape/nature
        "mountains", "mountain peak", "mountain range", "peak", "summit",
        "lakefront", "lakeshore", "seaside", "coastline", "riverside",
        "reflection in water", "water reflection", "mirrored water",
        "alpine", "glacier", "tundra", "highlands", "lowlands",
        # Descriptive nature terms
        "serene", "tranquil", "peaceful", "calm", "still",
        "majestic", "imposing", "towering", "vast", "expansive",
        "lush", "green", "verdant", "pristine", "untouched",
        "ethereal", "misty", "hazy", "soft light", "golden light",
    },
    suffix_patterns=["* background", "*_background"],
    contains_patterns=["background", "indoor", "outdoor", "room", "sky", "mountain", "lake", "river",
                      "serene", "tranquil", "peaceful", "majestic", "lush"],
)

LIGHTING_KEYWORDS = CategoryKeywords(
    exact_matches={
        "soft light", "soft lighting", "hard light", "hard lighting",
        "natural light", "natural lighting", "artificial light",
        "studio light", "studio lighting", "rim light", "backlight",
        "backlighting", "side light", "front light", "dramatic lighting",
        "bright", "dark", "dim", "shadow", "shadows", "high key", "low key",
        "warm light", "cool light", "golden hour", "blue hour",
        "soft and even", "even lighting", "flat lighting",
        "warm tones", "cool tones", "neutral tones", "muted colors",
        "vibrant colors", "desaturated", "low contrast", "high contrast",
        "light gray tones", "monochrome lighting",
        # Color themes
        "blue theme", "red theme", "green theme", "orange theme",
        "purple theme", "pink theme", "golden theme", "sepia theme",
        "monochrome", "black and white", "grayscale", "color grading",
        "color palette", "warm palette", "cool palette",
    },
    contains_patterns=["light", "lighting", "tones", "contrast", "shadows", "theme"],
)

STYLE_MEDIUM_KEYWORDS = CategoryKeywords(
    exact_matches={
        "photo", "photograph", "photography", "digital art", "illustration",
        "3d render", "3d", "cgi", "anime", "realistic", "semi-realistic",
        "oil painting", "watercolor", "sketch", "drawing", "artwork",
        "minimalist", "modern", "vintage", "retro", "classic", "artistic",
        "professional", "editorial", "fashion", "portrait", "glamour",
        "fine art", "commercial", "cinematic",
        # Additional styles
        "real life", "pinup", "cosplay", "boudoir", "glamour shot",
        "headshot photography", "fashion photography", "studio photography",
        "street photography", "candid", "posed", "lifestyle",
        # Artwork type
        "original", "original art", "fan art", "fanart", "concept art",
        "official", "official media", "key visual",
    },
    contains_patterns=["style", "art", "painting", "render"],
)

COMPOSITION_KEYWORDS = CategoryKeywords(
    exact_matches={
        "front view", "side view", "back view", "three-quarter view",
        "close up", "close-up", "closeup", "medium shot", "full body",
        "full shot", "upper body", "lower body", "cowboy shot", "bust shot",
        "headshot", "portrait", "landscape", "wide shot", "extreme close-up",
        "dutch angle", "low angle", "high angle", "birds eye", "worms eye",
        "centered", "rule of thirds", "symmetry", "asymmetry",
        "portrait orientation", "landscape orientation", "square",
        "2:3 aspect ratio", "3:4 aspect ratio", "16:9 aspect ratio",
        # Distance values from florence_analyze
        "close", "middle", "far", "very close", "very far",
        "medium distance", "near", "distant",
    },
    prefix_patterns=["camera_angle_*", "distance_*", "composition_*"],
    contains_patterns=["shot", "view", "angle", "framing"],
)

TECHNICAL_KEYWORDS = CategoryKeywords(
    exact_matches={
        "jpeg artifacts", "noise", "grain", "blur", "motion blur",
        "depth of field", "shallow dof", "bokeh", "vignette",
        "chromatic aberration", "lens flare", "watermark", "signature",
        "text", "logo", "border", "frame",
        # Content rating
        "sensitive", "nsfw", "nsfw content", "safe content", "explicit",
        "questionable", "general", "rating safe", "rating explicit",
        # Other technical
        "low resolution", "high resolution", "blurry", "sharp focus",
        "cropped", "letterboxed", "pillarboxed",
    },
    contains_patterns=["resolution", "aspect ratio", "artifacts", "watermark"],
)

# Map categories to their keywords
CATEGORY_KEYWORDS: Dict[CanonicalCategory, CategoryKeywords] = {
    CanonicalCategory.QUALITY_BOOSTERS: QUALITY_KEYWORDS,
    CanonicalCategory.SUBJECT: SUBJECT_KEYWORDS,
    CanonicalCategory.ACTION_POSE: ACTION_POSE_KEYWORDS,
    CanonicalCategory.ENVIRONMENT: ENVIRONMENT_KEYWORDS,
    CanonicalCategory.LIGHTING: LIGHTING_KEYWORDS,
    CanonicalCategory.STYLE_MEDIUM: STYLE_MEDIUM_KEYWORDS,
    CanonicalCategory.COMPOSITION: COMPOSITION_KEYWORDS,
    CanonicalCategory.TECHNICAL: TECHNICAL_KEYWORDS,
}


# =============================================================================
# META Patterns (to filter out)
# =============================================================================

META_FILTER_PATTERNS: List[str] = [
    "the image shows",
    "the image depicts",
    "the image captures",
    "the image features",
    "the image has",
    "the photo shows",
    "the photograph shows",
    "in the middle of the image",
    "in the center of the image",
    "in the foreground",
    "in the background of the image",
    "the overall mood",
    "the overall tone",
    "the overall atmosphere",
    "the overall aesthetic",
    "the overall look",
    "with a focus on",
    "focusing on",
    "showcasing",
    "which adds",
    "which creates",
    "which gives",
    "which lends",
    "conveying a sense of",
    "creating a sense of",
    "creating an atmosphere of",
    "the viewer can",
    "the viewer is",
    "the viewer may",
    "it is worth noting",
    "this image",
    "this photograph",
    "this photo",
    # Long captions that are meta-commentary
    "a stunning landscape photograph",
    "a stunning portrait",
    "a breathtaking view",
    "taken from a distance",
    # Usernames and watermarks (should be filtered)
    "patreon username",
    "gumroad username",
    "twitter username",
    "instagram username",
    "deviantart username",
    "pixiv username",
    "onlyfans username",
    "username",
    "artist name",
    "artist signature",
    # Software/editing
    "photoshop",
    "lightroom",
    "photoshopped",
    "digitally enhanced",
    "photoshop (medium)",
    # Generic meta
    "source:",
    "official art",
    # Noise/nonsense tokens from taggers
    "what", "blood", "death", "parody",
    # Parsing artifacts and placeholder values
    "na>",
    "na)",
    "na:",
    "na|",
    "na and body:",
    "na'",
    "na\\",
    "na |",
    "| na",
    ": na",
    "n/a",
    "accessory: na",
    "na/na",
    # Long descriptive captions that are meta
    "the saree is a traditional",
    "the dress is a",
    "the outfit is a",
    "the clothing is a",
    "the overall effect is",
    "her eyes are closed",
    "his eyes are closed",
    "appears to be enjoying",
    "appears to be in her",
    "appears to be in his",
    "high-quality, edited",
    "edited photo",
    # Caption sentence starters to filter (full sentences)
    "he is holding",
    "she is holding",
    "he is wearing",
    "she is wearing",
    "he stands",
    "she stands",
    "his eyes are",
    "her eyes are",
    "his dark hair is styled",
    "her dark hair is styled",
    "the lighting is dramatic",
    "the background is",
    # Common false positive terms
    "general",  # WD14 meta tag
    "original",  # Style tag that's too generic
    "fantasy",  # Genre tag
    "fine art parody",  # Specific style that's usually wrong
    # Suggestive content label (not useful for prompts)
    "suggestive content",
]


def is_meta_pattern(text: str) -> bool:
    """Check if text contains META patterns that should be filtered."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in META_FILTER_PATTERNS)


# =============================================================================
# Florence Analyze Key Mappings (Layer 1)
# =============================================================================

FLORENCE_KEY_TO_CATEGORY: Dict[str, CanonicalCategory] = {
    # Composition
    "camera_angle": CanonicalCategory.COMPOSITION,
    "distance_to_camera": CanonicalCategory.COMPOSITION,
    "framing": CanonicalCategory.COMPOSITION,

    # Style
    "art_style": CanonicalCategory.STYLE_MEDIUM,
    "style": CanonicalCategory.STYLE_MEDIUM,
    "medium": CanonicalCategory.STYLE_MEDIUM,

    # Environment
    "location": CanonicalCategory.ENVIRONMENT,
    "background": CanonicalCategory.ENVIRONMENT,
    "setting": CanonicalCategory.ENVIRONMENT,
    "scene": CanonicalCategory.ENVIRONMENT,

    # Action/Pose
    "action": CanonicalCategory.ACTION_POSE,
    "pose": CanonicalCategory.ACTION_POSE,
    "facing_direction": CanonicalCategory.ACTION_POSE,
    "facial_expression": CanonicalCategory.ACTION_POSE,

    # Subject Details
    "clothing": CanonicalCategory.SUBJECT_DETAILS,
    "pants": CanonicalCategory.SUBJECT_DETAILS,
    "shoes": CanonicalCategory.SUBJECT_DETAILS,
    "accessory": CanonicalCategory.SUBJECT_DETAILS,
    "hair_color": CanonicalCategory.SUBJECT_DETAILS,
    "hair_style": CanonicalCategory.SUBJECT_DETAILS,
    "hair_length": CanonicalCategory.SUBJECT_DETAILS,
    "eye_color": CanonicalCategory.SUBJECT_DETAILS,
    "body": CanonicalCategory.SUBJECT_DETAILS,
    "race": CanonicalCategory.SUBJECT_DETAILS,
    "ethnicity": CanonicalCategory.SUBJECT_DETAILS,

    # Subject
    "gender": CanonicalCategory.SUBJECT,
    "age": CanonicalCategory.SUBJECT,
    "subject": CanonicalCategory.SUBJECT,

    # Technical
    "text": CanonicalCategory.TECHNICAL,
    "watermark": CanonicalCategory.TECHNICAL,
}


def get_florence_key_category(key: str) -> Optional[CanonicalCategory]:
    """Get category for a florence_analyze key."""
    return FLORENCE_KEY_TO_CATEGORY.get(key.lower().strip())
