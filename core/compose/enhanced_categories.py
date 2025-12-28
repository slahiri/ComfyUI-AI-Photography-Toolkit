"""
Enhanced Category System for Image Metadata Classification
==========================================================

Migrated from canonical_structurer.py with improved structure:
- 10 granular categories (vs 6)
- Detailed subcategories for better phrase building
- Pattern-based fallback classification
- Preserves all existing vocabulary

Categories:
1. SUBJECT - Who (count, type, gender)
2. APPEARANCE - Physical traits (hair, eyes, face, body, skin, makeup)
3. CLOTHING - What wearing (garment, style, material, pattern, jewelry)
4. POSE_ACTION - Position/action (pose, gaze, expression, facing)
5. ENVIRONMENT - Where (location, background, props, weather)
6. LIGHTING - Light setup (direction, quality, color, style)
7. STYLE_QUALITY - Aesthetic (medium, style, quality, tone)
8. COMPOSITION - Framing (shot_type, angle, technique, format)
9. SCENE - Scene type (portrait, fashion, landscape)
10. CONSTRAINTS - Flags (content, safety, artifacts)
"""

from typing import Dict, Set, Tuple, Optional, List
from enum import Enum
import re


class Category(Enum):
    SUBJECT = "subject"
    APPEARANCE = "appearance"
    CLOTHING = "clothing"
    POSE_ACTION = "pose_action"
    ENVIRONMENT = "environment"
    LIGHTING = "lighting"
    STYLE_QUALITY = "style_quality"
    COMPOSITION = "composition"
    SCENE = "scene"
    CONSTRAINTS = "constraints"
    UNCATEGORIZED = "uncategorized"


# Category order for prompt assembly (Z-Image optimized)
CATEGORY_ORDER = [
    Category.SUBJECT,
    Category.APPEARANCE,
    Category.CLOTHING,
    Category.POSE_ACTION,
    Category.ENVIRONMENT,
    Category.LIGHTING,
    Category.STYLE_QUALITY,
    Category.COMPOSITION,
    Category.SCENE,
    Category.CONSTRAINTS,
]


# =============================================================================
# CATEGORY KEYWORDS - Migrated and reorganized from canonical_structurer.py
# =============================================================================

CATEGORY_KEYWORDS: Dict[Category, Dict[str, Set[str]]] = {

    # =========================================================================
    # SUBJECT - Who (count, type, gender only)
    # =========================================================================
    Category.SUBJECT: {
        "count": {
            "1girl", "1boy", "2girls", "3girls", "4girls", "5girls", "6+girls",
            "2boys", "3boys", "4boys", "5boys", "6+boys",
            "multiple girls", "multiple boys", "solo", "couple", "group", "crowd",
            "1woman", "1man", "multiple", "people",
        },
        "type": {
            "woman", "man", "girl", "boy", "child", "elderly",
            "model", "person", "figure", "character", "subject",
            "adult", "human", "1other", "androgynous",
            "human face", "female face", "male face",
        },
        "gender": {
            "female", "male",
        },
        "animal": {
            # Big cats
            "lion", "tiger", "leopard", "cheetah", "jaguar", "panther",
            "cougar", "puma", "lynx", "bobcat", "caracal", "serval",
            # Bears
            "bear", "grizzly bear", "polar bear", "black bear", "panda", "koala",
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
            "deer", "elk", "moose", "caribou", "bison",
            "raccoon", "skunk", "badger", "beaver", "otter", "porcupine",
            "squirrel", "chipmunk", "groundhog", "prairie dog",
            # Marine mammals
            "dolphin", "whale", "orca", "seal", "sea lion", "walrus",
            "manatee", "dugong", "narwhal", "beluga whale", "humpback whale",
            # Birds
            "bird", "eagle", "hawk", "falcon", "owl", "parrot", "peacock",
            "flamingo", "swan", "duck", "goose", "heron", "crane", "stork",
            "penguin", "puffin", "hummingbird", "cardinal", "crow", "raven",
            # Reptiles
            "snake", "python", "cobra", "lizard", "gecko", "iguana",
            "crocodile", "alligator", "turtle", "tortoise",
            # Fish
            "fish", "shark", "dolphin", "whale", "octopus", "jellyfish",
            # Insects
            "butterfly", "moth", "dragonfly", "bee", "beetle", "spider",
            # Farm animals
            "horse", "pony", "donkey", "cow", "bull", "sheep", "goat",
            "pig", "chicken", "rooster", "llama", "alpaca", "camel",
            # Pets
            "rabbit", "hamster", "guinea pig", "ferret", "hedgehog",
            # Mythical
            "dragon", "phoenix", "unicorn", "griffin", "pegasus",
            # General
            "animal", "wildlife", "creature", "animal focus", "no humans",
        },
    },

    # =========================================================================
    # APPEARANCE - Physical traits (hair, eyes, face, body, skin, makeup)
    # =========================================================================
    Category.APPEARANCE: {
        "hair_color": {
            "blonde hair", "brown hair", "black hair", "red hair", "white hair",
            "silver hair", "gray hair", "pink hair", "blue hair", "green hair",
            "purple hair", "orange hair", "multicolored hair", "streaked hair",
            "gradient hair", "two-tone hair", "dark hair", "light hair",
            "blonde", "brunette", "redhead", "dark brown hair",
        },
        "hair_style": {
            "long hair", "short hair", "medium hair", "curly hair", "straight hair",
            "wavy hair", "ponytail", "braid", "bun", "bangs", "twintails",
            "pigtails", "side ponytail", "high ponytail", "low ponytail",
            "french braid", "twin braids", "bob cut", "pixie cut", "hime cut",
            "ahoge", "hair ornament", "hair ribbon", "hairpin", "messy hair",
            "wet hair", "hair over one eye", "hair between eyes", "hair bun",
            "long wavy hair",
        },
        "eyes": {
            "blue eyes", "brown eyes", "green eyes", "black eyes", "red eyes",
            "dark eyes", "heterochromia", "eye color",
        },
        "face": {
            "lips", "parted lips", "closed mouth", "open mouth", "smile",
            "mole", "freckles", "nose", "thick lips", "thin lips", "full lips",
            "pink lips", "red lips", "eyelashes", "eyebrows",
        },
        "body": {
            "slim", "slender", "tall", "petite", "muscular", "toned", "curvy",
            "fit", "athletic", "voluptuous", "thick", "plump", "lean", "buff",
            "ripped", "wide hips", "narrow waist", "thigh gap", "hourglass figure",
            "medium breasts", "large breasts", "small breasts", "breasts",
            "cleavage", "sideboob", "underboob",
            "thighs", "back", "collarbone", "abs", "waist", "hips",
            "bare shoulders", "bare arms", "bare legs", "bare feet",
            "arched back", "curves", "physique",
        },
        "skin": {
            "dark skin", "fair skin", "pale skin", "tan", "tan skin", "light skin",
            "dark-skinned female", "dark-skinned male", "light-skinned",
            "smooth skin", "glowing skin", "clear skin", "dewy skin", "matte skin",
            "very dark skin", "olive skin", "bronze skin", "tanned", "sun-kissed",
            "freckles", "moles",
        },
        "makeup": {
            "makeup", "lipstick", "eyeshadow", "eyeliner", "mascara",
            "blush", "foundation", "contour", "highlighter", "lip gloss",
            "natural makeup", "heavy makeup", "no makeup", "minimal makeup",
            "smoky eyes", "winged eyeliner", "red lipstick", "nude lipstick",
            "gold eyeshadow", "gold lips", "nail polish", "smoky eye",
        },
        "ethnicity": {
            "asian", "caucasian", "african", "latina", "hispanic", "indian",
        },
        "body_parts": {
            "face", "eyes", "lips", "hair", "nose", "ears", "hands", "arms",
            "legs", "body", "torso", "head", "shoulders", "neck", "chest",
            "back", "feet", "fingers", "nails", "skin", "mouth", "tongue",
            "teeth", "chin", "jaw", "forehead", "cheeks", "stomach", "belly",
            "ankles", "wrists", "elbows", "knees", "armpits", "navel", "midriff",
            # Exposure details
            "sideboob", "underboob", "cleavage", "deep cleavage",
            "slight sideboob visible", "moderate sideboob visible",
            "significant sideboob visible", "extreme sideboob visible",
            "slight cleavage visible", "moderate cleavage visible",
            "deep cleavage visible", "extreme cleavage visible",
            "slight underboob visible", "underboob visible",
            "midriff exposed", "navel visible", "back exposed",
            "shoulders bare", "legs exposed", "thighs visible",
        },
        "body_type": {
            "slim", "slender", "petite", "curvy", "voluptuous", "athletic",
            "muscular", "fit", "toned", "average build", "full-figured",
            "plus-size", "hourglass figure", "pear shaped", "apple shaped",
            "slim body type", "athletic body type", "curvy body type",
            "voluptuous body type", "petite body type", "tall and slender",
            "muscular build", "toned physique", "soft body",
            "thick thighs", "wide hips", "narrow waist",
        },
        "breast_size": {
            "flat chest", "small breasts", "medium breasts", "large breasts",
            "very large breasts", "huge breasts", "petite bust", "full bust",
            "busty", "busty figure", "well endowed",
        },
        "body_features": {
            "piercings", "tattoos", "scars", "beauty mark", "tattoo", "arm tattoo",
            "mole", "mole on arm", "mole on face", "mole on body",
            "scar", "scar on leg", "scar on arm", "scar on face",
        },
        "condition": {
            "wet", "sweaty", "oily", "dry",
        },
    },

    # =========================================================================
    # CLOTHING - What wearing (garments, style, material, pattern, jewelry)
    # =========================================================================
    Category.CLOTHING: {
        "garment_top": {
            "shirt", "blouse", "sweater", "hoodie", "t-shirt", "tank top",
            "crop top", "camisole", "tube top", "halter top", "bandeau top",
            "polo shirt", "button-up shirt", "dress shirt", "oxford shirt",
            "flannel shirt", "collared shirt", "chambray shirt", "henley shirt",
            "graphic tee", "plain tee", "v-neck shirt", "knit sweater",
            "cable knit", "pullover hoodie", "zip-up hoodie", "sweatshirt",
            "cardigan", "vest", "blazer", "sports bra", "bra", "bralette",
            "white shirt", "peplum top", "wrap top", "off-shoulder top",
        },
        "garment_bottom": {
            "pants", "jeans", "shorts", "skirt", "leggings",
            "pencil skirt", "a-line skirt", "pleated skirt", "midi skirt",
            "maxi skirt", "mini skirt", "miniskirt",
            "wide-leg pants", "skinny jeans", "straight-leg", "bootcut",
            "cargo pants", "joggers", "sweatpants", "culottes",
            "high-waisted", "low-rise", "mid-rise",
            "ripped jeans", "high-waisted jeans", "low-rise jeans",
            "mom jeans", "boyfriend jeans", "dress pants", "chinos",
            "khakis", "trousers", "slacks", "denim shorts", "cargo shorts",
            "athletic shorts", "bermuda shorts", "yoga pants", "track pants",
            "palazzo pants", "capris", "cigarette pants", "harem pants",
            "straight leg jeans", "wide leg jeans", "bootcut jeans",
            "torn pants", "torn jeans",
        },
        "garment_dress": {
            "dress", "gown", "jumpsuit", "romper", "bodysuit",
            "maxi dress", "midi dress", "mini dress", "cocktail dress",
            "evening gown", "wedding dress", "sundress", "wrap dress",
            "a-line dress", "bodycon dress", "sheath dress", "shift dress",
            "empire waist dress", "fit and flare dress", "mermaid dress",
            "trumpet dress", "ballgown", "princess dress", "slip dress",
            "peasant dress", "tiered dress", "smock dress", "pinafore dress",
            "jumper dress", "tunic dress", "ball gown", "prom dress",
            "formal gown", "bridal gown", "bridesmaid dress", "flapper dress",
            "vintage dress", "retro dress", "boho dress", "lace dress",
            "sequin dress", "beaded dress", "floral dress", "backless dress",
            "white dress", "black dress", "red dress", "strapless dress",
            "one-shoulder dress", "off-shoulder dress", "knee-length dress",
            "floor-length dress", "shirt dress",
        },
        "garment_outerwear": {
            "jacket", "coat", "blazer", "trench coat", "parka", "windbreaker",
            "denim jacket", "leather jacket", "bomber jacket", "suit jacket",
            "sport coat", "faux leather jacket", "moto jacket", "jean jacket",
            "trucker jacket", "varsity jacket", "flight jacket", "overcoat",
            "peacoat", "wool coat", "puffer jacket", "down jacket",
            "quilted jacket", "rain jacket", "anorak", "cape", "shawl",
            "kimono jacket", "cardigan coat",
        },
        "garment_underwear": {
            "underwear", "panties", "lingerie", "briefs", "boxers", "thong",
            "bikini", "swimsuit", "swimwear", "bathing suit",
            "white bra", "white panties", "white sports bra", "training bra",
            "underwear only", "corset", "bustier", "teddy",
            "negligee", "babydoll", "chemise", "slip", "camisole set",
            "resort wear", "beachwear", "coverup", "beach coverup", "sarong",
        },
        "garment_uniform": {
            "uniform", "military uniform", "police uniform", "nurse uniform",
            "doctor uniform", "school uniform", "sailor uniform", "maid uniform",
            "chef uniform", "flight attendant uniform", "pilot uniform",
            "firefighter uniform", "sports uniform", "team jersey",
            "athletic uniform", "business suit", "corporate attire",
            "office wear", "service uniform", "suit",
        },
        "garment_ethnic": {
            "sari", "saree", "lehenga", "salwar kameez", "kurta", "sherwani",
            "dupatta", "churidar", "anarkali", "gharara", "sharara",
            "kimono", "yukata", "hakama", "obi", "furisode",
            "hanbok", "jeogori", "chima",
            "cheongsam", "qipao", "changshan", "hanfu",
            "ao dai", "ao ba ba", "kebaya", "batik",
            "dashiki", "kente", "agbada", "boubou", "kaftan",
            "dirndl", "lederhosen", "kilt",
            "abaya", "hijab", "niqab", "burqa", "thobe", "dishdasha", "bisht",
            "poncho", "huipil", "rebozo",
            "silk sari", "cotton sari", "banarasi sari", "kanjeevaram sari",
            "designer sari", "lehenga choli", "bridal lehenga", "designer lehenga",
            "ghagra choli", "salwar suit", "punjabi suit", "anarkali suit",
            "palazzo suit", "kurti", "kurta pajama", "achkan", "bandhgala",
            "nehru jacket", "chunni", "stole", "pashmina", "chaniya choli",
            "pattu pavadai", "half sari", "langa voni", "dhoti", "lungi", "mundu",
            "choli", "crop blouse", "backless blouse", "patiala",
            "indo-western", "fusion wear", "contemporary ethnic",
            "agal", "keffiyeh", "turban", "pagri",
            "traditional clothing", "cultural attire", "ethnic wear", "traditional dress",
        },
        "garment_religious": {
            "nun", "habit", "traditional nun", "coif", "wimple", "cornette",
            "cassock", "vestment", "surplice", "chasuble", "stole", "alb",
            "priest", "monk", "friar", "clergy", "religious habit",
            "kippa", "yarmulke", "tallit", "tzitzit",
            "buddhist robe", "kasaya", "saffron robe",
            "ceremonial robe", "ritual clothing", "liturgical vestment",
            "robe", "bathrobe", "toga", "roman toga", "red robe",
            "caftan", "muumuu", "housecoat", "dressing gown",
        },
        "garment_costume": {
            "cosplay", "costume", "cosplay outfit", "character costume",
            "anime cosplay", "video game cosplay", "superhero costume",
            "fantasy costume", "medieval costume", "renaissance costume",
            "halloween costume", "theatrical costume", "stage costume",
            "historical costume", "period costume", "sci-fi costume",
        },
        "footwear": {
            "shoes", "boots", "sneakers", "heels", "sandals", "loafers",
            "oxford shoes", "pumps", "stilettos", "wedges", "flats", "mules",
            "espadrilles", "ankle boots", "knee-high boots", "thigh-high boots",
            "combat boots", "platform shoes", "athletic shoes", "running shoes",
            "tennis shoes", "kitten heels", "platform heels",
            "chelsea boots", "cowboy boots", "rain boots", "winter boots",
            "flip flops", "slides", "penny loafers", "moccasins",
            "driving shoes", "ballet flats", "pointed toe flats", "mary janes",
            "brogues", "derby shoes", "monk straps",
            "no socks", "barefoot", "no shoes",
        },
        "accessory": {
            "accessories", "hat", "glasses", "watch", "bag",
            "baseball cap", "beanie", "fedora", "beret", "bucket hat", "sun hat",
            "aviator sunglasses", "cat eye sunglasses", "round sunglasses",
            "sunglasses", "eyeglasses", "reading glasses",
            "silk scarf", "wool scarf", "infinity scarf", "scarf",
            "necktie", "tie", "bow tie",
            "leather belt", "chain belt", "statement belt", "belt", "suspenders",
            "tote bag", "shoulder bag", "satchel", "messenger bag", "briefcase",
            "duffle bag", "handbag", "clutch", "backpack", "tote", "crossbody bag",
            "headband", "hair clip", "crown", "tiara", "veil", "fascinator",
        },
        "jewelry": {
            "jewelry", "necklace", "bracelet", "earrings", "ring", "brooch",
            "choker", "pendant", "chain necklace", "hoop earrings",
            "stud earrings", "drop earrings", "bangle", "cuff bracelet",
            "luxury watch", "diamond necklace", "pearl necklace", "pearl earrings",
            "gold bracelet", "gold bracelets", "multiple bracelets", "arm cuff",
            "gold bands",
        },
        "material": {
            "silk", "cotton", "linen", "wool", "cashmere", "velvet",
            "satin", "chiffon", "lace", "tulle", "organza", "taffeta",
            "denim", "leather", "suede", "corduroy", "tweed", "flannel",
            "latex", "pvc", "vinyl", "rubber", "neoprene", "spandex", "lycra",
            "sequin", "beaded", "embroidered", "printed", "pleated", "draped",
            "ruched", "ruffled", "gathered", "smocked", "shirred",
            "faux leather", "vegan leather", "jersey", "knit fabric",
            "crochet", "mesh", "sheer fabric", "sequins",
            "metallic fabric", "holographic", "iridescent", "patent leather",
            "faux fur", "fleece", "quilted", "padded",
        },
        "pattern": {
            "floral", "striped", "plaid", "checkered", "polka dot",
            "paisley", "geometric", "abstract", "animal print", "leopard print",
            "zebra print", "camouflage", "tie-dye", "ombre", "color block",
            "ikat", "batik print", "tribal", "ethnic print", "brocade",
            "floral pattern", "striped pattern", "plaid pattern", "checkered pattern",
            "polka dots", "snake print", "geometric pattern", "abstract pattern",
            "paisley pattern", "solid", "solid color", "two-tone", "multicolor",
            "houndstooth", "herringbone", "gingham",
        },
        "fit": {
            "fitted clothing", "loose fit", "oversized fit", "relaxed fit",
            "tailored clothing", "structured silhouette", "flowy silhouette",
            "form-fitting", "body-hugging", "baggy", "boxy",
            "tight", "tight clothes", "taut clothes", "skin tight",
        },
        "neckline": {
            "v-neck", "crew neck", "scoop neck", "halter", "off-shoulder",
            "strapless", "one-shoulder", "turtleneck", "cowl neck",
            "sweetheart neckline", "square neckline", "boat neck",
            "v-neckline", "deep v-neck", "plunging neckline", "round neckline",
            "bateau neckline", "high neck", "mock neck", "keyhole neckline",
            "illusion neckline", "off-shoulder neckline", "bandeau neckline",
            "peter pan collar", "mandarin collar", "stand collar", "notched collar",
            "halter neckline", "strapless neckline",
        },
        "sleeve": {
            "sleeveless", "cap sleeves", "short sleeves", "elbow sleeves",
            "three-quarter sleeves", "long sleeves", "full sleeves",
            "puff sleeves", "balloon sleeves", "bishop sleeves", "bell sleeves",
            "flutter sleeves", "angel sleeves", "batwing sleeves", "dolman sleeves",
            "raglan sleeves", "kimono sleeves", "juliet sleeves",
            "cold shoulder", "slit sleeves", "ruffle sleeves",
        },
        "length": {
            "floor length", "ankle length", "knee length", "tea length",
            "full length", "maxi length", "midi length", "mini length",
            "hip length", "thigh length", "cropped", "crop length",
            "waist length", "longline", "tunic length", "micro length",
        },
        "state": {
            "fully clothed", "partially undressed", "clothes lift",
            "shirt lift", "skirt lift", "dress lift", "pants pull",
            "unbuttoned", "partially unbuttoned", "unzipped", "open shirt",
            "loose clothing", "wet clothing", "see-through clothing",
            "tight clothing", "revealing outfit", "skimpy outfit", "barely covered",
            "open clothes", "unbuttoned shirt",
            "distressed", "ripped", "frayed", "stonewashed",
            "torn pants", "torn jeans", "torn clothes", "torn shirt",
            "shiny clothes",
        },
        "color": {
            "black outfit", "white outfit", "red outfit", "blue outfit",
            "green outfit", "pink outfit", "yellow outfit", "orange outfit",
            "purple outfit", "navy blue", "burgundy", "maroon", "olive green",
            "teal", "monochrome outfit", "color blocked",
        },
        "brand": {
            "nike", "nike (company)", "adidas", "puma", "converse",
            "vans", "reebok", "new balance", "gucci", "prada",
            "louis vuitton", "chanel", "versace", "dior", "balenciaga",
            "supreme", "off-white", "champion", "levis", "levi's",
        },
        "detail": {
            "clothes writing", "logo", "brand logo", "text on clothing",
            "english text", "graphic print", "slogan", "printed text",
            "ruffled", "lace-trimmed", "embellished", "tailored",
        },
    },

    # =========================================================================
    # POSE_ACTION - Position and action (pose, gaze, expression, facing)
    # =========================================================================
    Category.POSE_ACTION: {
        "stance": {
            "standing", "sitting", "lying", "walking", "running", "jumping",
            "dancing", "posing", "leaning", "crouching", "kneeling",
            "standing straight", "standing relaxed", "standing tall",
            "seated", "sitting cross-legged", "sitting on floor",
            "on stool", "on chair", "on bed", "on couch", "on sofa",
            "squatting", "bending", "leaning forward", "leaning back",
            "lying down", "reclining", "prone position",
            "on back", "on side", "on stomach", "on all fours",
            "wading", "swimming", "floating", "diving",
        },
        "arm_position": {
            "arms up", "arm up", "arms crossed", "hands on hips",
            "hand up", "hands up", "hand in own hair", "hand in hair",
            "arms raised", "one arm raised", "arms overhead",
            "arms at sides", "arms relaxed", "arms behind back",
            "hand on hip", "akimbo pose", "arms folded", "hugging self",
            "hands in pockets", "hand in pocket",
            "hands clasped", "hands together", "prayer hands",
            "pointing", "gesturing", "waving",
            "touching face", "hand on chin", "hand on cheek",
            "arms outstretched", "arms extended", "reaching",
            "arm support",
            # Detailed arm poses from CLIP
            "both arms raised", "arms raised above head", "arms stretched overhead",
            "hands behind head", "arms behind head",
            "hands in hair", "hand running through hair",
            "hands touching face", "hands covering face", "hands covering eyes",
            "fingers spread over eyes", "fingers over eyes", "hand over mouth",
            "hands clasped together", "pointing gesture", "peace sign", "waving hand",
            "arms spread wide", "reaching forward", "arms near face", "hand near face",
        },
        "leg_position": {
            "legs together", "legs apart", "wide stance",
            "one leg forward", "weight on one leg", "contrapposto",
            "legs crossed", "crossed ankles",
            "walking pose", "mid-stride", "stepping forward",
            "kicking", "mid-air",
            "spread legs", "legs spread", "knees apart",
        },
        "head_position": {
            "head straight", "head tilted", "head tilted left", "head tilted right",
            "chin up", "chin down", "head turned",
        },
        "gaze": {
            "looking at viewer", "looking at camera", "looking away",
            "looking down", "looking up", "looking to the side",
            "looking left", "looking right", "side glance",
            "looking over shoulder", "glancing sideways", "looking back",
            "eye contact",
        },
        "facing": {
            "facing camera", "facing viewer", "facing away",
            "profile view", "three-quarter view",
            "front view", "side view", "back view",
        },
        "expression": {
            "smile", "smiling", "laughing", "serious", "neutral", "happy",
            "sad", "angry", "surprised", "contemplative", "pensive",
            "closed eyes", "open mouth", "wink",
            "grin", "frown", "crying", "blushing", "embarrassed",
            "expressionless", "blank expression", "stoic", "deadpan",
            "pouting", "smirk", "scowl", "grimace", "confused",
            "confident", "confident expression", "sultry expression",
            "bedroom eyes", "biting lip", "tongue out", "blowing kiss",
        },
        "body_language": {
            "confident pose", "relaxed pose", "casual pose", "formal stance",
            "power pose", "open body language", "closed body language",
            "dynamic pose", "static pose", "action pose",
            "elegant pose", "graceful pose", "dramatic pose",
            "playful pose", "serious expression", "contemplative pose",
        },
        "pose_type": {
            "model pose", "fashion pose", "editorial pose",
            "portrait pose", "headshot",
            "candid pose", "natural pose", "posed shot",
            "glamour pose", "beauty shot", "lifestyle pose",
            "selfie", "mirror selfie", "group selfie",
            "seductive pose", "provocative pose", "alluring pose",
        },
        "position_relative": {
            "against wall", "leaning on wall", "leaning against wall",
            "against door", "in doorway", "in window", "by window",
            "on floor", "on ground", "on grass", "on sand",
        },
        "framing": {
            "upper body", "full body", "cowboy shot", "from behind",
            "from side", "back view", "profile", "three-quarter view",
            "half body",
        },
    },

    # =========================================================================
    # ENVIRONMENT - Where (location, background, props, weather, time)
    # =========================================================================
    Category.ENVIRONMENT: {
        "location_indoor": {
            "studio", "room", "interior", "indoor", "indoors", "inside",
            "bedroom", "living room", "kitchen", "bathroom", "office",
            "gym", "restaurant", "cafe", "bar", "club", "museum",
            "gallery", "theater", "stadium", "arena",
            "lobby", "hallway", "corridor", "staircase", "elevator",
            "library", "classroom", "laboratory", "hospital", "clinic",
            "salon", "spa", "hotel", "apartment", "penthouse", "loft",
            "warehouse", "factory", "garage", "workshop", "attic",
            "basement", "cellar", "wine cellar", "storage room",
            "dressing room", "locker room", "fitting room", "changing room",
            "beauty salon", "hair salon", "nail salon", "makeup room",
            "conference room", "boardroom", "reception", "waiting room",
            "ballroom", "banquet hall", "auditorium", "concert hall",
            "cinema", "movie theater", "arcade", "bowling alley",
            "nightclub", "discotheque", "disco", "pub", "lounge",
            "church", "cathedral", "mosque", "temple", "synagogue",
            "train station", "subway", "airport", "bus station",
            "shopping mall", "boutique", "store", "supermarket",
        },
        "location_outdoor": {
            "outdoor", "outdoors", "outside", "street", "park", "garden",
            "beach", "mountain", "forest", "field", "meadow", "desert",
            "ocean", "lake", "river", "city", "urban", "rural",
            "water", "sea", "shore", "nature", "wilderness", "wild",
            "pool", "poolside", "swimming pool", "by the pool", "near pool",
            "countryside", "rooftop", "balcony", "terrace",
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
        "background": {
            "background", "backdrop", "scenery", "environment", "setting",
            "blurred background", "blurry background", "out of focus background",
            "simple background", "plain background", "solid background",
            "grey background", "gray background", "white background",
            "black background", "blue background", "green background",
            "red background", "pink background", "yellow background",
            "gradient background", "textured background", "studio background",
            "seamless background", "paper background", "muslin background",
            "bokeh background", "neutral background", "dark background",
            "light background", "colorful background",
            "brick wall", "brick background", "exposed brick", "red brick",
            "wall", "concrete wall", "stone wall", "wooden wall",
            "graffiti wall", "painted wall", "textured wall", "white wall",
            "industrial wall", "metal wall", "tile wall", "tiled wall",
            "beige background", "brown background", "beige primary color",
        },
        "props": {
            "stool", "chair", "armchair", "sofa", "couch", "bench",
            "barstool", "office chair", "throne", "ottoman",
            "table", "desk", "counter", "podium", "pedestal",
            "bed", "mattress", "futon", "daybed",
            "backdrop stand", "light stand", "reflector", "umbrella",
            "props", "studio equipment",
            "door", "doorway", "entrance", "exit", "window",
            "elevator door",
        },
        "time": {
            "day", "night", "morning", "evening", "afternoon", "dusk",
            "dawn", "sunset", "sunrise", "golden hour", "blue hour",
            "midnight", "noon", "twilight",
            "summer", "winter", "spring", "fall", "autumn",
        },
        "weather": {
            "sunny", "cloudy", "rainy", "snowy", "foggy", "misty",
            "stormy", "windy", "overcast", "clear", "hazy",
            "partly cloudy", "mostly cloudy", "scattered clouds",
            "thunderstorm", "lightning", "thunder", "rain", "drizzle",
            "snow", "blizzard", "fog", "dense fog", "mist",
            "humid", "dry", "arid", "breeze", "gust",
        },
        "sky": {
            "sky", "blue sky", "clear sky", "open sky", "vast sky",
            "dramatic sky", "moody sky", "stormy sky", "overcast sky",
            "clouds", "cloud", "cloudscape", "cloud formation",
            "sun", "sunlight", "sunbeam", "sun rays",
            "moon", "moonlight", "full moon", "crescent moon",
            "stars", "starry", "starry sky", "milky way",
            "aurora", "aurora borealis", "northern lights",
            "rainbow", "sunset", "sunrise",
        },
        "vegetation": {
            "tree", "trees", "palm tree", "palm trees", "pine tree", "oak tree",
            "tropical tree", "coconut tree", "willow tree", "cherry tree",
            "plant", "plants", "flower", "flowers", "grass", "lawn",
            "bush", "bushes", "shrub", "shrubs", "hedge", "hedges",
            "fern", "ferns", "moss", "ivy", "vine", "vines",
            "garden", "botanical", "greenhouse", "conservatory",
            "tropical plants", "foliage", "greenery", "vegetation",
        },
        "atmosphere": {
            "peaceful", "calm", "serene", "energetic", "vibrant",
            "mysterious", "romantic", "dramatic", "tense", "relaxed",
            "cozy", "warm", "cold", "eerie", "magical",
            "tranquil", "idyllic", "pastoral", "bucolic",
            "majestic", "grand", "sublime", "awe-inspiring",
            "melancholic", "nostalgic", "wistful", "bittersweet",
            "ominous", "foreboding", "haunting", "spooky", "creepy",
            "ethereal", "dreamlike", "surreal", "fantastical",
            "intimate", "secluded", "private", "isolated", "remote",
            "bustling", "lively", "crowded", "busy", "hectic",
            "desolate", "barren", "abandoned", "empty", "lonely",
        },
        "landmark": {
            "eiffel tower", "louvre", "colosseum", "taj mahal",
            "great wall of china", "mount fuji", "pyramids of giza",
            "statue of liberty", "big ben", "sydney opera house",
            "golden gate bridge", "times square", "central park",
            "grand canyon", "niagara falls", "santorini",
        },
    },

    # =========================================================================
    # LIGHTING - Light setup
    # =========================================================================
    Category.LIGHTING: {
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
            "even lighting", "dramatic lighting", "flat lighting",
            "high contrast", "low contrast",
            "bright", "bright lighting", "dim",
        },
        "shadow": {
            "harsh shadows", "soft shadows", "hard shadows",
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
            "golden hour lighting", "warm lighting", "cool lighting",
        },
        "style": {
            "rembrandt", "butterfly", "split", "loop", "broad", "short",
            "paramount", "clamshell", "beauty lighting",
            "chiaroscuro", "low key", "high key",
            "rembrandt lighting", "butterfly lighting", "loop lighting",
            "split lighting", "high key lighting", "low key lighting",
            "silhouette", "magic hour", "flash photography",
        },
        "effect": {
            "volumetric", "god rays", "lens flare", "light leak",
            "specular", "highlights", "catchlights", "rim highlights",
            "glow", "halo", "backlit",
            "specular highlights", "catchlights visible", "rim light",
            "properly exposed",
        },
    },

    # =========================================================================
    # STYLE_QUALITY - Aesthetic and quality
    # =========================================================================
    Category.STYLE_QUALITY: {
        "medium": {
            "photograph", "photography", "photo", "digital", "film",
            "painting", "illustration", "drawing", "sketch", "render",
            "3d", "cgi", "artwork", "art",
            "DSLR", "mirrorless", "point and shoot", "medium format",
            "full frame", "crop sensor", "smartphone", "instant camera",
            "film camera", "polaroid", "vintage camera",
            "webcam", "drone camera", "GoPro", "action camera",
            "professional studio camera", "disposable camera",
        },
        "style": {
            "realistic", "photorealistic", "hyperrealistic", "stylized",
            "artistic", "cinematic", "editorial", "commercial",
            "fine art", "documentary", "candid", "glamour",
            "minimalist", "maximalist", "abstract",
            "k-pop", "j-pop", "idol", "korean style", "japanese style",
            "pinup", "pinup style", "pin-up", "pin up", "retro pinup",
            "real life", "real world", "realistic style",
            "original", "original character", "original content",
        },
        "quality": {
            "high quality", "8k", "4k", "hd", "ultra hd", "detailed",
            "highly detailed", "sharp", "crisp", "clean", "professional",
            "masterpiece", "best quality",
            "exceptional aesthetic quality", "excellent aesthetic quality",
            "good aesthetic quality", "average aesthetic quality",
            "excellent technical quality", "good technical quality",
            "professional quality", "amateur quality",
            "low resolution", "high resolution",
            "good focus", "sharp focus", "good clarity",
            "minimal noise", "tack sharp details",
        },
        "tone": {
            "vibrant", "muted", "desaturated", "saturated",
            "warm tones", "cool tones", "neutral tones", "tan tones",
            "teal and orange", "monochrome", "black and white",
            "sepia", "cross-processed",
            "color graded", "cinematic color", "film look",
            "bright", "dark", "moody", "airy", "dreamy",
            "pastel colors", "bright colors", "neon colors",
            "earth tones", "jewel tones", "vibrant colors", "muted colors",
            "monochromatic", "multicolored",
            # Color tones from photography tagger
            "saddle brown tones", "brown tones", "golden tones", "blue tones",
            "red tones", "green tones", "pink tones", "purple tones",
            "orange tones", "yellow tones", "gray tones", "grey tones",
        },
        "camera_technical": {
            "14mm", "24mm", "35mm", "50mm", "85mm", "135mm", "200mm", "400mm",
            "wide angle", "ultra wide angle", "standard lens", "portrait lens",
            "telephoto", "super telephoto", "macro lens", "fisheye",
            "f/1.2", "f/1.4", "f/1.8", "f/2", "f/2.8", "f/4", "f/5.6", "f/8",
            "ISO 100", "ISO 400", "ISO 800", "ISO 1600", "ISO 3200",
            "long exposure", "slow shutter", "fast shutter",
            "motion blur", "sharp corner to corner", "creamy bokeh",
            "85mm portrait lens", "portrait lens compression",
        },
        "genre": {
            "portrait photography", "fashion photography", "street photography",
            "landscape photography", "product photography", "food photography",
            "architectural photography", "sports photography",
            "wildlife photography", "macro photography",
            "beauty photography", "headshot photography", "boudoir photography",
            "environmental portrait", "lifestyle photography",
            "black and white photography", "film photography",
        },
        "texture": {
            "matte finish", "glossy", "shiny", "smooth", "rough",
            "textured", "grainy", "soft", "silky", "velvet",
            "metallic", "iridescent", "pearlescent",
        },
        "film_reference": {
            "kodak", "fujifilm", "ilford", "portra", "ektar", "tri-x",
            "cinestill", "polaroid", "vintage film", "film grain",
        },
    },

    # =========================================================================
    # COMPOSITION - Framing and camera
    # =========================================================================
    Category.COMPOSITION: {
        "shot_type": {
            "close-up", "extreme close-up", "medium shot", "medium close-up",
            "full shot", "full body", "full body shot", "wide shot", "long shot",
            "extreme long shot", "establishing shot", "detail shot",
            "headshot", "portrait", "half body", "upper body",
            "cowboy shot", "american shot", "over-the-shoulder",
            "point of view shot", "two-shot", "group shot",
        },
        "angle": {
            "eye level", "low angle", "high angle", "bird's eye",
            "worm's eye", "dutch angle", "tilted", "straight on",
            "front view", "side view", "back view", "3/4 view",
            "profile", "from above", "from below",
            "overhead shot", "canted angle", "oblique angle",
            "bird's eye view", "worm's eye view", "aerial view",
            "eye level shot", "low angle shot", "high angle shot",
        },
        "technique": {
            "centered", "rule of thirds", "golden ratio", "symmetrical",
            "asymmetrical", "balanced", "off-center", "frame within frame",
            "leading lines", "diagonal", "horizontal", "vertical",
            "center composition", "triangular composition",
            "golden spiral", "diagonal composition", "negative space",
            "fill the frame", "minimalist composition", "layered depth",
            "horizontal symmetry", "vertical symmetry", "partial symmetry",
            "natural framing", "elements at thirds",
            "centered composition", "symmetrical composition",
            "dynamic composition", "static composition",
            "subject at thirds intersection", "subject on vertical third",
            "subject at golden ratio", "centered subject", "central composition",
            "rule of thirds composition",
            "horizontal lines", "vertical lines", "diagonal lines", "curved lines",
            "strong linear elements", "symmetry axis",
            "high horizon", "low horizon", "ground emphasis", "sky emphasis",
            "multiple subjects",
        },
        "focus": {
            "shallow depth of field", "deep focus", "selective focus",
            "bokeh", "sharp", "soft focus", "tack sharp", "in focus",
            "out of focus", "blurred", "blurry",
            "sharp focus", "slightly soft", "background blur", "foreground blur",
            "deep depth of field", "depth of field",
        },
        "format": {
            "portrait orientation", "landscape orientation", "square",
            "16:9", "4:3", "2.39:1", "cinemascope", "widescreen",
            "vertical", "horizontal",
            "square format", "horizontal frame", "vertical frame",
            "panoramic", "ultra-wide", "cinematic aspect ratio",
            "2:3 aspect ratio", "3:2 aspect ratio", "4:5 aspect ratio",
            "1:1 aspect ratio", "9:16 aspect ratio", "16:9 aspect ratio",
        },
        "framing": {
            "photographic framing", "centered framing",
        },
    },

    # =========================================================================
    # SCENE - Scene type
    # =========================================================================
    Category.SCENE: {
        "type": {
            "portrait", "fashion", "glamour", "editorial",
            "street photography", "landscape", "product", "food",
            "architectural", "sports", "wildlife", "macro",
            "beauty", "boudoir", "lifestyle", "event",
            "travel", "nature", "night", "astrophotography",
        },
        "fashion_style": {
            "casual outfit", "formal outfit", "business attire", "sportswear",
            "streetwear", "evening wear", "loungewear", "swimwear", "activewear",
            "workwear", "smart casual", "business casual", "semi-formal",
            "avant-garde fashion", "bohemian style", "boho chic", "preppy style",
            "punk style", "grunge style", "gothic fashion", "hippie style",
            "retro fashion", "vintage fashion", "minimalist fashion",
            "haute couture", "high fashion", "ready-to-wear", "designer fashion",
            "modest fashion", "chic style", "elegant style",
            "edgy style", "rocker style", "biker style", "classic style",
            "romantic style", "feminine style", "masculine style",
            "korean fashion", "japanese fashion",
        },
    },

    # =========================================================================
    # CONSTRAINTS - Content flags and quality
    # =========================================================================
    Category.CONSTRAINTS: {
        "content_rating": {
            "safe content", "suggestive content", "nsfw content",
            "explicit content", "questionable content", "mature content",
            "adult content", "sensitive",
        },
        "safety": {
            "safe for work", "sfw", "appropriate", "clean",
            "family friendly", "non-offensive",
        },
        "artifacts": {
            "no watermark", "no logo", "no text", "no blur",
            "no noise", "no grain", "no distortion",
            "watermark", "signature", "artist name", "artist signature",
            "patreon username", "gumroad username", "deviantart logo",
            "web address", "url", "logo", "username", "social media handle",
            "copyright", "date stamp", "timestamp", "instagram handle",
        },
        "nsfw_exposed": {
            "exposed female breast", "exposed male chest", "exposed buttocks",
            "exposed female genitalia", "exposed male genitalia", "exposed anus",
            "topless", "bottomless", "nude", "naked", "fully nude",
            "partial nudity", "implied nudity", "artistic nudity",
        },
        "nsfw_covered": {
            "covered female breast", "covered buttocks", "covered belly",
            "covered feet", "covered armpits", "covered female genitalia",
            "clothed", "partially clothed", "underwear visible",
            "bra visible", "panties visible", "lingerie visible",
        },
        "nsfw_accessory": {
            "collar", "garter", "stockings", "fishnets",
            "thigh highs", "blindfold", "handcuffs", "rope", "chains",
        },
        "booru_meta": {
            # Booru/anime meme tags that should be filtered for real photos
            "rei no pool", "that pool", "infamous pool", "the pool",
            "bad id", "bad pixiv id", "bad twitter id",
            "commentary", "commentary request", "translated",
            "highres", "absurdres", "incredibly absurdres",
            "tagme", "check translation", "paid reward",
            "commission", "skeb commission", "fanbox reward",
            "patreon reward", "gumroad", "fantia",
        },
    },
}


# =============================================================================
# FILTERED TERMS - Tags that should be removed entirely
# =============================================================================

FILTERED_TERMS: set = {
    # Booru meta tags
    "rei no pool", "that pool", "infamous pool",
    "bad id", "bad pixiv id", "bad twitter id",
    "commentary", "commentary request", "translated",
    "tagme", "check translation", "paid reward",
    "commission", "skeb commission", "fanbox reward",
    "patreon reward", "gumroad", "fantia",
    # False positives for real photos
    "3d",  # Usually wrong when applied to real photos
    "mixed media",  # Usually wrong for photos
    # Generic/meaningless
    "general", "original", "fantasy",
}


# =============================================================================
# PATTERN-BASED CLASSIFICATION
# =============================================================================

CATEGORY_PATTERNS: List[Tuple[str, Category, Optional[str]]] = [
    # Hair patterns
    (r"^(.+)\s+hair$", Category.APPEARANCE, "hair_color"),
    (r"^(long|short|medium|wavy|curly|straight)\s+hair", Category.APPEARANCE, "hair_style"),

    # Eye patterns
    (r"^(.+)\s+eyes?$", Category.APPEARANCE, "eyes"),

    # Skin patterns
    (r"^(.+)\s+skin$", Category.APPEARANCE, "skin"),
    (r"^(dark|light|fair|pale|tan)\s+skin", Category.APPEARANCE, "skin"),

    # Body patterns
    (r"^(small|medium|large)\s+breasts?$", Category.APPEARANCE, "body"),
    (r"^(covered|exposed)\s+", Category.CONSTRAINTS, "nsfw_exposed"),

    # Clothing patterns
    (r"^(.+)\s+dress$", Category.CLOTHING, "garment_dress"),
    (r"^(.+)\s+(shirt|blouse|top)$", Category.CLOTHING, "garment_top"),
    (r"^(.+)\s+(pants|jeans|skirt|shorts)$", Category.CLOTHING, "garment_bottom"),
    (r"^wearing\s+(.+)$", Category.CLOTHING, "garment_top"),
    (r"^(.+)\s+(necklace|earrings?|bracelet|ring)$", Category.CLOTHING, "jewelry"),

    # Pose patterns
    (r"^looking\s+(at|to|away|up|down|back)", Category.POSE_ACTION, "gaze"),
    (r"^facing\s+(viewer|away|camera)", Category.POSE_ACTION, "facing"),
    (r"^hand\s+on\s+", Category.POSE_ACTION, "arm_position"),
    (r"^from\s+(behind|side|above|below)", Category.POSE_ACTION, "framing"),

    # Environment patterns
    (r"^(.+)\s+(background|wall)$", Category.ENVIRONMENT, "background"),
    (r"^(plain|simple|blurred)\s+background", Category.ENVIRONMENT, "background"),

    # Lighting patterns
    (r"^(.+)\s+light(ing)?$", Category.LIGHTING, "quality"),
    (r"^light\s+from\s+", Category.LIGHTING, "direction"),
    (r"^(.+)\s+shadows?$", Category.LIGHTING, "shadow"),
    (r"^(high|low)\s+(key|contrast)", Category.LIGHTING, "style"),

    # Quality patterns
    (r"^(excellent|good|average|poor)\s+(.+)\s+quality$", Category.STYLE_QUALITY, "quality"),
    (r"^(high|low)\s+resolution$", Category.STYLE_QUALITY, "quality"),

    # Composition patterns
    (r"^(.+)\s+(shot|angle|view)$", Category.COMPOSITION, "angle"),
    (r"^(.+)\s+(composition|framing)$", Category.COMPOSITION, "technique"),
    (r"^(\d+:\d+)\s+aspect\s+ratio$", Category.COMPOSITION, "format"),
    (r"^(portrait|landscape|vertical|horizontal)\s+(orientation|frame)$", Category.COMPOSITION, "format"),

    # Style patterns
    (r"^(.+)\s+tones?$", Category.STYLE_QUALITY, "tone"),
]


# =============================================================================
# SYNONYM MAPPINGS
# =============================================================================

TAG_SYNONYMS: Dict[str, str] = {
    # Clothing
    "trousers": "pants",
    "slacks": "pants",
    "tee": "t-shirt",
    "polo": "polo shirt",
    "trainers": "sneakers",
    "specs": "glasses",
    "shades": "sunglasses",
    "purse": "handbag",
    "rucksack": "backpack",
    "cap": "hat",
    "waistcoat": "vest",

    # Hair
    "blonde": "blonde hair",
    "brunette": "brown hair",
    "redhead": "red hair",
    "ginger": "red hair",
    "auburn": "red hair",
    "raven": "black hair",
    "platinum": "white hair",
    "fringe": "bangs",

    # Expression
    "grinning": "grin",
    "frowning": "frown",
    "weeping": "crying",
}


# =============================================================================
# CLASSIFIER
# =============================================================================

class EnhancedCategoryClassifier:
    """
    Enhanced category classifier with dictionary lookup, pattern matching,
    and optional NLP/LLM fallback.
    """

    def __init__(self):
        self._build_reverse_index()

    def _build_reverse_index(self) -> None:
        """Build reverse index from keyword to (category, subcategory)."""
        self.reverse_index: Dict[str, Tuple[Category, str]] = {}

        for category, subcategories in CATEGORY_KEYWORDS.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    key = keyword.lower().strip()
                    if key not in self.reverse_index:
                        self.reverse_index[key] = (category, subcategory)

    def _normalize(self, tag: str) -> str:
        """Normalize tag for lookup."""
        tag = tag.lower().strip()
        tag = tag.replace("_", " ")
        tag = tag.replace("\\(", "(").replace("\\)", ")")

        # Apply synonyms
        if tag in TAG_SYNONYMS:
            tag = TAG_SYNONYMS[tag]

        return tag

    def classify_tag(self, tag: str) -> Tuple[Category, Optional[str], str]:
        """
        Classify a single tag.

        Returns: (category, subcategory, method)
        """
        normalized = self._normalize(tag)

        # 0. Check if tag should be filtered
        if normalized in FILTERED_TERMS:
            return Category.CONSTRAINTS, "booru_meta", "filtered"

        # 1. Direct lookup
        if normalized in self.reverse_index:
            cat, subcat = self.reverse_index[normalized]
            return cat, subcat, "lookup"

        # 2. Substring matching (keyword in tag or tag in keyword)
        best_match = None
        best_len = 0
        for keyword, (category, subcategory) in self.reverse_index.items():
            if keyword in normalized or normalized in keyword:
                if len(keyword) > best_len:
                    best_match = (category, subcategory)
                    best_len = len(keyword)

        if best_match:
            return best_match[0], best_match[1], "substring"

        # 3. Component matching (split and try each part)
        components = normalized.replace("-", " ").split()
        if len(components) > 1:
            for n in range(len(components) - 1, 0, -1):
                for i in range(len(components) - n + 1):
                    component = " ".join(components[i:i + n])
                    if component in self.reverse_index:
                        cat, subcat = self.reverse_index[component]
                        return cat, subcat, "component"

        # 4. Pattern matching
        for pattern, category, subcategory in CATEGORY_PATTERNS:
            if re.match(pattern, normalized, re.IGNORECASE):
                return category, subcategory, "pattern"

        # 5. Unknown
        return Category.UNCATEGORIZED, None, "unknown"

    def classify_all(
        self,
        tags: Dict[str, float],
        min_confidence: float = 0.0,
    ) -> Dict[str, Dict]:
        """
        Classify all tags and organize by category.

        Returns dict with structure:
        {
            "subject": {
                "tags": {"1girl": {"confidence": 0.99, "subcategory": "count", "method": "lookup"}},
                "subcategories": {"count": {"1girl": 0.99}}
            },
            ...
        }
        """
        result = {cat.value: {"tags": {}, "subcategories": {}} for cat in Category}

        for tag, confidence in tags.items():
            if confidence < min_confidence:
                continue

            category, subcategory, method = self.classify_tag(tag)

            # Add to category
            result[category.value]["tags"][tag] = {
                "confidence": confidence,
                "subcategory": subcategory,
                "method": method,
            }

            # Organize by subcategory
            if subcategory:
                if subcategory not in result[category.value]["subcategories"]:
                    result[category.value]["subcategories"][subcategory] = {}
                result[category.value]["subcategories"][subcategory][tag] = confidence

        return result

    def get_uncategorized(self, tags: Dict[str, float]) -> Dict[str, float]:
        """Get only uncategorized tags."""
        uncategorized = {}
        for tag, confidence in tags.items():
            category, _, _ = self.classify_tag(tag)
            if category == Category.UNCATEGORIZED:
                uncategorized[tag] = confidence
        return uncategorized

    def get_summary(self, classified: Dict) -> Dict:
        """Get summary of classification results."""
        summary = {}
        total = 0
        uncategorized_count = 0

        for category, data in classified.items():
            count = len(data.get("tags", {}))
            total += count
            if category == "uncategorized":
                uncategorized_count = count
            if count > 0:
                top_tags = sorted(
                    data["tags"].items(),
                    key=lambda x: x[1]["confidence"],
                    reverse=True
                )[:5]
                summary[category] = {
                    "count": count,
                    "top_tags": [(t, d["confidence"]) for t, d in top_tags],
                }

        summary["_total"] = total
        summary["_uncategorized_ratio"] = uncategorized_count / total if total > 0 else 0

        return summary


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_classifier_instance: Optional[EnhancedCategoryClassifier] = None


def get_classifier() -> EnhancedCategoryClassifier:
    """Get or create the global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EnhancedCategoryClassifier()
    return _classifier_instance


def classify_tags(tags: Dict[str, float], min_confidence: float = 0.0) -> Dict:
    """Classify tags using the global classifier."""
    return get_classifier().classify_all(tags, min_confidence)


def get_uncategorized_tags(tags: Dict[str, float]) -> Dict[str, float]:
    """Get uncategorized tags using the global classifier."""
    return get_classifier().get_uncategorized(tags)
