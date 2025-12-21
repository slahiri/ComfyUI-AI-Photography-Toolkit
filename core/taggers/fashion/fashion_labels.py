"""Fashion label mappings and vocabularies."""

# Fashionpedia 27 categories
FASHIONPEDIA_LABELS = [
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit, rompers",
    "cape",
    "glasses",
    "hat",
    "headband, head covering, hair accessory",
    "tie",
    "glove",
    "watch",
    "belt",
    "leg warmer",
    "tights, stockings",
    "sock",
    "shoe",
    "bag, wallet",
    "scarf",
    "umbrella",
]

# Convert to tag-friendly format
FASHIONPEDIA_TAG_MAP = {
    0: "shirt",
    1: "t-shirt",
    2: "sweater",
    3: "cardigan",
    4: "jacket",
    5: "vest",
    6: "pants",
    7: "shorts",
    8: "skirt",
    9: "coat",
    10: "dress",
    11: "jumpsuit",
    12: "cape",
    13: "glasses",
    14: "hat",
    15: "headwear",
    16: "tie",
    17: "gloves",
    18: "watch",
    19: "belt",
    20: "leg warmer",
    21: "tights",
    22: "socks",
    23: "shoes",
    24: "bag",
    25: "scarf",
    26: "umbrella",
}

# SegFormer clothing segmentation labels
SEGFORMER_CLOTHES_LABELS = [
    "background",
    "hat",
    "hair",
    "sunglasses",
    "upper-clothes",
    "skirt",
    "pants",
    "dress",
    "belt",
    "left-shoe",
    "right-shoe",
    "face",
    "left-leg",
    "right-leg",
    "left-arm",
    "right-arm",
    "bag",
    "scarf",
]

# Map segmentation labels to tag text
SEGFORMER_TAG_MAP = {
    1: "wearing hat",
    3: "wearing sunglasses",
    4: "upper clothing visible",
    5: "wearing skirt",
    6: "wearing pants",
    7: "wearing dress",
    8: "wearing belt",
    9: "shoes visible",
    10: "shoes visible",
    16: "carrying bag",
    17: "wearing scarf",
}

# Note: FashionCLIP vocabulary is now loaded from vocabularies.json via utils.py
# See: core/taggers/utils.py -> get_all_fashion_vocabulary()

# Wargon clothing classifier categories
WARGON_CATEGORIES = [
    "casual",
    "formal",
    "sportswear",
    "traditional",
    "streetwear",
    "party",
    "beachwear",
    "workwear",
]


def get_clothing_tag_text(label_id: int, label_map: dict, prefix: str = "wearing") -> str | None:
    """
    Convert label ID to tag text.

    Args:
        label_id: Numeric label
        label_map: Mapping from ID to base text
        prefix: Action prefix (e.g., "wearing", "carrying")

    Returns:
        Tag text like "wearing dress", or None if label_id is unknown
    """
    base = label_map.get(label_id)

    # Skip unknown label IDs - don't generate confusing "item_N" tags
    if base is None:
        return None

    # Items that use "carrying" instead of "wearing"
    carrying_items = {"bag", "umbrella", "backpack", "handbag"}

    if base in carrying_items:
        return f"carrying {base}"

    # Items that don't need prefix
    no_prefix_items = {"glasses", "sunglasses", "watch"}
    if base in no_prefix_items:
        return base

    return f"{prefix} {base}"
