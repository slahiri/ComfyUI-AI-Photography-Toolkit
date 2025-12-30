"""
Detail-specific prompts for enhanced section descriptions.

Provides specific prompts based on:
1. Animal types (from wildlife analyzer)
2. Clothing types (from fashion analyzers)
3. Detected tags (from taggers)

This creates more targeted prompts that guide the LLM to describe
specific detected elements in detail.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from ..types import TaggerResults, Decisions


# =============================================================================
# ANIMAL-TYPE SPECIFIC PROMPTS
# =============================================================================

ANIMAL_PROMPTS: Dict[str, Dict[str, str]] = {
    # Domestic animals
    "dog": {
        "species": "Identify the dog: breed if recognizable, size category (small/medium/large), age impression (puppy/adult/senior).",
        "appearance": "Describe the dog's appearance: coat type (short/long/curly/wire), color and markings, ear shape (floppy/erect), tail, and distinctive breed features.",
        "expression": "Describe the dog's expression: eyes, mouth (panting/closed), ear position, and overall demeanor (alert/relaxed/playful/sleepy).",
    },
    "cat": {
        "species": "Identify the cat: breed if recognizable (Persian, Siamese, tabby, etc.), size, age impression.",
        "appearance": "Describe the cat's appearance: fur length and texture, color pattern (solid/tabby/calico/pointed), eye color, ear shape, and whiskers.",
        "expression": "Describe the cat's expression: eye shape and dilation, ear position, whisker orientation, and demeanor (curious/relaxed/alert/sleepy).",
    },
    "horse": {
        "species": "Identify the horse: breed if recognizable, size (pony/horse/draft), coloring type (bay/chestnut/black/gray/pinto).",
        "appearance": "Describe the horse's appearance: coat color and sheen, mane and tail (color, length, grooming), markings (blaze/star/socks), muscular build.",
        "expression": "Describe the horse's demeanor: ear position, eye expression, head carriage, and overall stance (alert/calm/spirited).",
    },
    "bird": {
        "species": "Identify the bird: species or type if recognizable, size relative to common birds.",
        "appearance": "Describe the bird's appearance: plumage colors and patterns, beak shape and color, eye color, wing position, tail feathers.",
        "expression": "Describe the bird's posture and demeanor: head position, eye alertness, feather state (fluffed/sleek), and activity.",
    },

    # Wild cats
    "lion": {
        "species": "Identify: male (with mane) or female lion, age impression, physical condition.",
        "appearance": "Describe the lion's appearance: coat color and condition, mane details (male), muscular build, facial features, paws.",
        "expression": "Describe the lion's expression: eyes, mouth, ear position, and demeanor (regal/alert/relaxed/aggressive).",
    },
    "tiger": {
        "species": "Identify the tiger: subspecies if apparent (Bengal/Siberian), size impression.",
        "appearance": "Describe the tiger's appearance: orange coat with black stripes pattern, white markings, muscular build, facial ruff.",
        "expression": "Describe the tiger's expression and demeanor: intensity of gaze, ear position, mouth, overall presence.",
    },
    "leopard": {
        "species": "Identify: leopard or snow leopard, size and build.",
        "appearance": "Describe the leopard's appearance: coat color, rosette pattern, spots, muscular agile build, long tail.",
        "expression": "Describe the leopard's expression: eyes, ear position, and stealthy demeanor.",
    },
    "cheetah": {
        "species": "Identify the cheetah: distinctive features, build.",
        "appearance": "Describe the cheetah's appearance: tan coat with black spots, tear marks on face, slender athletic build, long legs.",
        "expression": "Describe the cheetah's expression and alert posture.",
    },

    # Other wildlife
    "bear": {
        "species": "Identify the bear: species (brown/black/polar/panda), size impression.",
        "appearance": "Describe the bear's appearance: fur color and texture, build (massive/stocky), facial features, claws if visible.",
        "expression": "Describe the bear's demeanor: posture, facial expression, activity.",
    },
    "wolf": {
        "species": "Identify: wolf or wolf-like canid, pack context if visible.",
        "appearance": "Describe the wolf's appearance: coat color (gray/white/black/mixed), thick fur, build, golden eyes, facial features.",
        "expression": "Describe the wolf's expression: intensity, ear position, and wild demeanor.",
    },
    "fox": {
        "species": "Identify the fox: species (red/arctic/fennec), distinctive features.",
        "appearance": "Describe the fox's appearance: coat color, bushy tail, pointed ears, slender build, facial features.",
        "expression": "Describe the fox's expression: clever eyes, ear position, alert demeanor.",
    },
    "deer": {
        "species": "Identify the deer: species if recognizable, sex (antlers indicate male), age.",
        "appearance": "Describe the deer's appearance: coat color, antlers if present (points, velvet), build, white tail markings.",
        "expression": "Describe the deer's demeanor: alertness, ear position, graceful stance.",
    },
    "elephant": {
        "species": "Identify: African or Asian elephant, approximate age (calf/adult/elder).",
        "appearance": "Describe the elephant's appearance: skin texture and color, ear size and shape, tusks if present, trunk details.",
        "expression": "Describe the elephant's demeanor: trunk position, ear spread, eye expression.",
    },
    "monkey": {
        "species": "Identify the primate: species type if recognizable, size.",
        "appearance": "Describe the primate's appearance: fur color and pattern, facial features, hands, tail if present.",
        "expression": "Describe the primate's expression: facial expression, eye contact, posture.",
    },
    "rabbit": {
        "species": "Identify: wild rabbit, hare, or domestic breed.",
        "appearance": "Describe the rabbit's appearance: fur color and texture, ear length and position, size, eye color.",
        "expression": "Describe the rabbit's demeanor: alertness, ear position, nose twitching.",
    },

    # Marine/aquatic
    "fish": {
        "species": "Identify the fish: type if recognizable, size relative to surroundings.",
        "appearance": "Describe the fish's appearance: body shape, scale colors and patterns, fin shapes, eye.",
        "expression": "Describe the fish's movement and orientation in water.",
    },
    "dolphin": {
        "species": "Identify: dolphin species if apparent, pod context.",
        "appearance": "Describe the dolphin's appearance: sleek gray body, dorsal fin, beak shape, eye.",
        "expression": "Describe the dolphin's demeanor and apparent playfulness or activity.",
    },
    "whale": {
        "species": "Identify the whale: species if recognizable (humpback/orca/blue/etc.).",
        "appearance": "Describe the whale's appearance: body size and shape, coloring, distinctive features (barnacles, markings).",
        "expression": "Describe the whale's activity: breaching, spouting, diving.",
    },

    # Reptiles
    "snake": {
        "species": "Identify the snake: type if recognizable, size.",
        "appearance": "Describe the snake's appearance: scale pattern and colors, head shape, body thickness, coiled or extended.",
        "expression": "Describe the snake's posture: tongue, head position, alertness.",
    },
    "lizard": {
        "species": "Identify the lizard: type if recognizable (gecko/iguana/chameleon/etc.).",
        "appearance": "Describe the lizard's appearance: skin texture and color, body shape, tail, eyes.",
        "expression": "Describe the lizard's posture and alertness.",
    },
    "turtle": {
        "species": "Identify: turtle or tortoise, type if recognizable.",
        "appearance": "Describe the turtle's appearance: shell pattern and color, head and limb features, size.",
        "expression": "Describe the turtle's activity and demeanor.",
    },
}

# Generic fallbacks for animal categories
ANIMAL_CATEGORY_PROMPTS: Dict[str, Dict[str, str]] = {
    "mammal": {
        "species": "Identify the mammal: species, size, and distinctive features.",
        "appearance": "Describe the mammal's appearance: fur/hair, coloring, build, and notable physical features.",
        "expression": "Describe the mammal's expression and demeanor.",
    },
    "bird": {
        "species": "Identify the bird: species or type, size.",
        "appearance": "Describe the bird's appearance: plumage, beak, wings, tail.",
        "expression": "Describe the bird's posture and activity.",
    },
    "reptile": {
        "species": "Identify the reptile: type and species if recognizable.",
        "appearance": "Describe the reptile's appearance: scales, coloring, body shape.",
        "expression": "Describe the reptile's posture and activity.",
    },
    "fish": {
        "species": "Identify the fish: type and species if recognizable.",
        "appearance": "Describe the fish's appearance: body shape, fins, coloring.",
        "expression": "Describe the fish's movement in water.",
    },
    "insect": {
        "species": "Identify the insect: type and species if recognizable.",
        "appearance": "Describe the insect's appearance: body segments, wings, coloring, legs.",
        "expression": "Describe the insect's activity and position.",
    },
}


# =============================================================================
# CLOTHING-TYPE SPECIFIC PROMPTS
# =============================================================================

CLOTHING_PROMPTS: Dict[str, str] = {
    # Dresses
    "dress": "Describe the dress in detail: style (A-line/sheath/maxi/mini/cocktail), neckline (V-neck/sweetheart/halter/off-shoulder), sleeve length, fabric type and texture, color and pattern, fit, and any embellishments or details.",
    "gown": "Describe the gown: formal style, silhouette (ballgown/mermaid/empire), train if present, fabric (silk/satin/tulle/lace), color, and elegant details.",
    "sundress": "Describe the sundress: casual summer style, strap type, length, print or pattern, fabric weight, and relaxed fit.",

    # Tops
    "blouse": "Describe the blouse: style, collar type, sleeve style, fabric (silk/cotton/chiffon), color, buttons or closures, and feminine details.",
    "shirt": "Describe the shirt: style (button-down/polo/t-shirt), collar, sleeve length, fabric, color or pattern, fit (fitted/relaxed).",
    "sweater": "Describe the sweater: style (pullover/cardigan/turtleneck), knit pattern, material (wool/cashmere/cotton), color, and fit.",
    "tank top": "Describe the tank top: strap style, neckline, fit, fabric, color.",
    "crop top": "Describe the crop top: style, neckline, sleeve type, fabric, color, and midriff exposure level.",

    # Bottoms
    "pants": "Describe the pants: style (trousers/chinos/cargo/wide-leg), fit (slim/relaxed/tailored), fabric, color, and waist details.",
    "jeans": "Describe the jeans: cut (skinny/straight/bootcut/mom/boyfriend), wash (dark/light/distressed), rise, and fit.",
    "skirt": "Describe the skirt: style (pencil/A-line/pleated/maxi/mini), length, fabric, color or pattern, and waist details.",
    "shorts": "Describe the shorts: style, length, fabric, color, and fit.",

    # Formal wear
    "suit": "Describe the suit: style (single/double-breasted), lapel type, fit (slim/classic), fabric, color, accompanying shirt and tie if visible.",
    "blazer": "Describe the blazer: style, button configuration, lapels, fabric, color, and fit.",
    "tuxedo": "Describe the tuxedo: style, lapels (peak/shawl), fabric sheen, bow tie, and formal accessories.",

    # Outerwear
    "jacket": "Describe the jacket: type (leather/denim/bomber/blazer), closure style, collar, pockets, fabric, color, and fit.",
    "coat": "Describe the coat: style (trench/peacoat/overcoat/parka), length, collar, fabric, color, and buttons or closures.",
    "hoodie": "Describe the hoodie: style (pullover/zip-up), hood details, pocket type, fabric weight, color or graphics.",

    # Swimwear
    "bikini": "Describe the bikini: top style (triangle/bandeau/halter), bottom style (brief/high-waist/string), color or pattern, and coverage.",
    "swimsuit": "Describe the swimsuit: style (one-piece/tankini), neckline, back design, color or pattern, and cut.",

    # Underwear/Intimate
    "lingerie": "Describe the lingerie: style and pieces, fabric (lace/silk/satin), color, and delicate details.",
    "bra": "Describe the bra: style (push-up/bralette/sports), fabric, color, straps, and decorative elements.",

    # Athletic
    "sportswear": "Describe the athletic wear: type (leggings/sports bra/tank/shorts), fabric (moisture-wicking/compression), color, brand details if visible.",
    "uniform": "Describe the uniform: type (military/school/sports/work), colors, insignia or logos, condition, and formal elements.",

    # Accessories
    "hat": "Describe the hat: style (fedora/beanie/cap/sun hat), material, color, and how it's worn.",
    "glasses": "Describe the eyewear: style (glasses/sunglasses), frame shape and color, lens type.",
    "jewelry": "Describe the jewelry: pieces (necklace/earrings/bracelet/rings), metal type, stones or gems, style (delicate/statement).",
    "shoes": "Describe the footwear: type (heels/sneakers/boots/sandals), style, color, heel height if applicable.",
    "bag": "Describe the bag: type (handbag/clutch/backpack/tote), size, material, color, brand if recognizable.",
}


# =============================================================================
# TAG-DRIVEN DETAIL ENHANCEMENT
# =============================================================================

# Maps detected tags to enhanced prompts for specific sections
TAG_ENHANCEMENTS: Dict[str, Dict[str, str]] = {
    # Hair style enhancements
    "ponytail": {
        "hair": "Describe the ponytail: placement (high/low/side), tightness, hair texture flowing from it, any wisps or flyaways, and hair accessories.",
    },
    "braid": {
        "hair": "Describe the braid: style (French/Dutch/fishtail/boxer braids), thickness, neatness, length, and any incorporated accessories.",
    },
    "bun": {
        "hair": "Describe the bun: style (messy/sleek/top knot/low bun), size, neatness, and any pins or accessories.",
    },
    "twintails": {
        "hair": "Describe the twintails: placement, length, style (straight/curly/wavy), and any ribbons or accessories.",
    },
    "bangs": {
        "hair": "Describe the bangs: style (blunt/side-swept/curtain/wispy), length, and how they frame the face.",
    },
    "short hair": {
        "hair": "Describe the short hair: style (pixie/bob/undercut/buzz), texture, styling, and how it frames the face.",
    },
    "long hair": {
        "hair": "Describe the long hair: length, texture (straight/wavy/curly), volume, ends condition, and how it falls.",
    },
    "curly hair": {
        "hair": "Describe the curly hair: curl pattern (loose waves/ringlets/coils), volume, definition, and styling.",
    },

    # Hair color enhancements
    "blonde hair": {
        "hair": "Describe the blonde hair: shade (platinum/golden/honey/strawberry/dirty blonde), highlights or lowlights, roots if visible.",
    },
    "red hair": {
        "hair": "Describe the red hair: shade (auburn/copper/ginger/burgundy), vibrancy, and natural vs dyed appearance.",
    },
    "black hair": {
        "hair": "Describe the black hair: depth of color, blue or brown undertones, shine and health.",
    },
    "brown hair": {
        "hair": "Describe the brown hair: shade (light/medium/dark/chocolate/chestnut), warmth, highlights.",
    },
    "colored hair": {
        "hair": "Describe the colored/dyed hair: colors (pink/blue/purple/rainbow/ombre), vibrancy, and style.",
    },

    # Eye enhancements
    "blue eyes": {
        "eyes": "Describe the blue eyes: shade (pale/bright/navy/gray-blue), clarity, any unique patterns in the iris.",
    },
    "green eyes": {
        "eyes": "Describe the green eyes: shade (emerald/hazel-green/sea green), brightness, golden flecks if present.",
    },
    "brown eyes": {
        "eyes": "Describe the brown eyes: shade (light/amber/chocolate/near-black), warmth and depth.",
    },
    "heterochromia": {
        "eyes": "Describe the heterochromia: the different eye colors, contrast between them.",
    },

    # Face feature enhancements
    "freckles": {
        "face": "Describe the freckles: distribution (nose/cheeks/full face), density, color, and natural charm.",
    },
    "makeup": {
        "face": "Describe the makeup: foundation coverage, eye makeup (eyeshadow/liner/lashes), lip color, blush/contour, overall style (natural/dramatic/glamorous).",
    },
    "lipstick": {
        "face": "Describe the lipstick: color (red/pink/nude/bold), finish (matte/glossy/satin), and application.",
    },
    "beard": {
        "face": "Describe the beard: style (full/stubble/goatee/trimmed), length, grooming, color, and how it frames the face.",
    },
    "glasses": {
        "face": "Describe the glasses: frame style (round/square/cat-eye/aviator), frame color, how they sit on the face.",
    },

    # Expression enhancements
    "smile": {
        "expression": "Describe the smile: type (subtle/broad/toothy/smirk), warmth, whether it reaches the eyes, and the mood it conveys.",
    },
    "serious": {
        "expression": "Describe the serious expression: intensity, focus of gaze, set of the jaw, and the mood conveyed.",
    },
    "laughing": {
        "expression": "Describe the laughing expression: openness, joy conveyed, eye crinkles, natural vs posed feel.",
    },

    # Body/pose enhancements
    "sitting": {
        "pose": "Describe the sitting pose: posture (upright/relaxed/leaning), leg position, arm placement, and overall body language.",
    },
    "standing": {
        "pose": "Describe the standing pose: posture, weight distribution, arm position, leg stance, and overall presence.",
    },
    "lying down": {
        "pose": "Describe the lying pose: position (back/side/stomach), arm and leg arrangement, head position, relaxed or posed.",
    },
    "dynamic pose": {
        "pose": "Describe the dynamic pose: action captured, body movement, energy, and motion blur if present.",
    },

    # Lighting enhancements
    "backlight": {
        "lighting": "Describe the backlighting: rim light effect, silhouette elements, glow around edges, lens flare if present.",
    },
    "dramatic lighting": {
        "lighting": "Describe the dramatic lighting: strong shadows, high contrast, chiaroscuro effect, mood created.",
    },
    "soft lighting": {
        "lighting": "Describe the soft lighting: diffused quality, gentle shadows, flattering effect on skin.",
    },
    "golden hour": {
        "lighting": "Describe the golden hour lighting: warm orange/golden tones, long shadows, soft quality, magical atmosphere.",
    },
    "blue hour": {
        "lighting": "Describe the blue hour lighting: cool blue tones, twilight atmosphere, soft diffused quality.",
    },
    "studio lighting": {
        "lighting": "Describe the studio lighting: key/fill/rim light setup, controlled shadows, professional quality.",
    },

    # Background enhancements
    "bokeh": {
        "composition": "Describe the bokeh: shape of blur circles, depth separation, how it isolates the subject, dreamy quality.",
    },
    "simple background": {
        "environment": "Describe the simple background: color, gradient if present, studio or natural, texture.",
    },
}


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def get_animal_prompt(
    wildlife_type: Optional[str],
    section: str,
) -> Optional[str]:
    """
    Get animal-specific prompt for a section.

    Args:
        wildlife_type: Detected animal type from wildlife analyzer
        section: Section name (species/appearance/expression)

    Returns:
        Specific prompt or None if no match
    """
    if not wildlife_type:
        return None

    wildlife_lower = wildlife_type.lower()

    # Check exact match
    if wildlife_lower in ANIMAL_PROMPTS:
        prompts = ANIMAL_PROMPTS[wildlife_lower]
        return prompts.get(section)

    # Check partial match
    for animal_key, prompts in ANIMAL_PROMPTS.items():
        if animal_key in wildlife_lower or wildlife_lower in animal_key:
            return prompts.get(section)

    # Check category fallbacks
    for category, prompts in ANIMAL_CATEGORY_PROMPTS.items():
        if category in wildlife_lower:
            return prompts.get(section)

    return None


def get_clothing_prompt(
    tagger_results: TaggerResults,
    threshold: float = 0.5,
) -> Optional[Tuple[str, str]]:
    """
    Get clothing-specific prompt based on detected clothing.

    Args:
        tagger_results: Tagger outputs
        threshold: Confidence threshold

    Returns:
        Tuple of (clothing_type, prompt) or None
    """
    # Check deepfashion first
    if tagger_results.deepfashion:
        clothing_type = tagger_results.deepfashion.get("clothing_type", {})
        if isinstance(clothing_type, dict) and clothing_type:
            best_type = max(clothing_type, key=clothing_type.get)
            best_type_lower = best_type.lower()
            for key, prompt in CLOTHING_PROMPTS.items():
                if key in best_type_lower or best_type_lower in key:
                    return (best_type, prompt)

    # Check tagger tags
    clothing_tags = set()
    for tag, conf in tagger_results.wd14.items():
        if conf >= threshold:
            clothing_tags.add(tag.lower().replace("_", " "))

    for tag, conf in tagger_results.pixai.items():
        if conf >= threshold:
            clothing_tags.add(tag.lower().replace("_", " "))

    # Find matching clothing prompt
    for key, prompt in CLOTHING_PROMPTS.items():
        if key in clothing_tags:
            return (key, prompt)
        # Partial match
        for tag in clothing_tags:
            if key in tag:
                return (key, prompt)

    return None


def get_tag_enhancement(
    section: str,
    tagger_results: TaggerResults,
    threshold: float = 0.5,
) -> Optional[Tuple[str, str]]:
    """
    Get tag-driven enhancement for a section.

    Args:
        section: Section name to enhance
        tagger_results: Tagger outputs
        threshold: Confidence threshold

    Returns:
        Tuple of (trigger_tag, enhanced_prompt) or None
    """
    # Collect all tags
    all_tags: Dict[str, float] = {}
    for tag, conf in tagger_results.wd14.items():
        if conf >= threshold:
            all_tags[tag.lower().replace("_", " ")] = conf

    for tag, conf in tagger_results.pixai.items():
        if conf >= threshold:
            tag_lower = tag.lower().replace("_", " ")
            if tag_lower not in all_tags:
                all_tags[tag_lower] = conf

    for tag, conf in tagger_results.joytag.items():
        if conf >= threshold:
            tag_lower = tag.lower().replace("_", " ")
            if tag_lower not in all_tags:
                all_tags[tag_lower] = conf

    # Find matching enhancement
    best_match = None
    best_conf = 0.0

    for tag_key, enhancements in TAG_ENHANCEMENTS.items():
        if section in enhancements:
            # Check if tag is detected
            tag_key_lower = tag_key.lower()
            for detected_tag, conf in all_tags.items():
                if tag_key_lower in detected_tag or detected_tag in tag_key_lower:
                    if conf > best_conf:
                        best_match = (tag_key, enhancements[section])
                        best_conf = conf

    return best_match


def enhance_section_prompt(
    section: str,
    base_prompt: str,
    decisions: Decisions,
    tagger_results: TaggerResults,
    threshold: float = 0.5,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Enhance a section prompt with specific details.

    Args:
        section: Section name
        base_prompt: Original section prompt
        decisions: Decisions object
        tagger_results: Tagger outputs
        threshold: Confidence threshold

    Returns:
        Tuple of (enhanced_prompt, enhancement_info)
    """
    enhancement_info = None

    # For animal subjects, use animal-specific prompts
    if decisions.wildlife_detected and decisions.wildlife_type:
        animal_prompt = get_animal_prompt(decisions.wildlife_type, section)
        if animal_prompt:
            enhancement_info = {
                "type": "animal_specific",
                "animal": decisions.wildlife_type,
                "section": section,
            }
            return (animal_prompt, enhancement_info)

    # For clothing section, use clothing-specific prompts
    if section == "clothing":
        clothing_result = get_clothing_prompt(tagger_results, threshold)
        if clothing_result:
            clothing_type, clothing_prompt = clothing_result
            enhancement_info = {
                "type": "clothing_specific",
                "clothing": clothing_type,
            }
            return (clothing_prompt, enhancement_info)

    # Check for tag-driven enhancements
    tag_result = get_tag_enhancement(section, tagger_results, threshold)
    if tag_result:
        trigger_tag, enhanced_prompt = tag_result
        enhancement_info = {
            "type": "tag_enhanced",
            "trigger_tag": trigger_tag,
            "section": section,
        }
        return (enhanced_prompt, enhancement_info)

    # No enhancement, return original
    return (base_prompt, None)


def get_all_enhancements(
    active_sections: List[str],
    decisions: Decisions,
    tagger_results: TaggerResults,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Get all available enhancements for debugging/metadata.

    Args:
        active_sections: List of active section names
        decisions: Decisions object
        tagger_results: Tagger outputs
        threshold: Confidence threshold

    Returns:
        Dict mapping section to enhancement info
    """
    enhancements = {}

    for section in active_sections:
        _, info = enhance_section_prompt(
            section, "", decisions, tagger_results, threshold
        )
        if info:
            enhancements[section] = info

    return enhancements
