"""
Scene-specific prompts for environment description.

Maps Places365 scene categories to descriptive prompts that guide
the LLM to describe scene-specific elements.
"""

from typing import Dict, Optional, Tuple


# Scene category groups with their specific prompts
SCENE_GROUPS: Dict[str, Dict] = {
    # =========================================================================
    # WATER / COASTAL
    # =========================================================================
    "water_coastal": {
        "scenes": [
            "beach", "coast", "coastline", "seashore", "shore", "beach house",
            "ocean", "sea", "cove", "bay", "harbor", "dock", "pier", "marina",
            "boardwalk", "promenade", "jetty", "lighthouse", "cliff",
        ],
        "prompt": """Describe the coastal/water setting: the beach or shoreline characteristics,
            water appearance (color, waves, clarity), sand or rock textures,
            horizon line, any maritime elements, and coastal atmosphere.""",
    },

    # =========================================================================
    # WATER / FRESHWATER
    # =========================================================================
    "water_freshwater": {
        "scenes": [
            "lake", "pond", "river", "stream", "creek", "waterfall",
            "hot spring", "swamp", "marsh", "wetland", "lagoon",
            "dam", "reservoir", "canal",
        ],
        "prompt": """Describe the water setting: the water body characteristics,
            reflections, surrounding vegetation, bank or shore details,
            water movement or stillness, and natural atmosphere.""",
    },

    # =========================================================================
    # FOREST / WOODLAND
    # =========================================================================
    "forest": {
        "scenes": [
            "forest", "forest path", "forest road", "rainforest", "jungle",
            "bamboo forest", "tree farm", "woodland", "grove", "thicket",
            "tree house", "orchard", "palm grove",
        ],
        "prompt": """Describe the forest setting: tree types and canopy,
            light filtering through foliage, forest floor details,
            depth and density, natural sounds implied, and woodland atmosphere.""",
    },

    # =========================================================================
    # MOUNTAIN / HIGHLAND
    # =========================================================================
    "mountain": {
        "scenes": [
            "mountain", "mountain path", "mountain snowy", "volcano",
            "valley", "canyon", "gorge", "cliff", "butte", "badlands",
            "hill", "hillside", "highland", "plateau", "ridge",
        ],
        "prompt": """Describe the mountain/highland setting: peaks and terrain,
            elevation sense, rocky surfaces, distant vistas,
            atmospheric perspective, and alpine or rugged character.""",
    },

    # =========================================================================
    # DESERT / ARID
    # =========================================================================
    "desert": {
        "scenes": [
            "desert", "desert road", "desert sand", "sand dune",
            "oasis", "badlands", "mesa", "canyon",
        ],
        "prompt": """Describe the desert setting: sand or rock formations,
            harsh light and shadows, heat shimmer if visible,
            sparse vegetation, horizon, and arid atmosphere.""",
    },

    # =========================================================================
    # FIELDS / MEADOWS / AGRICULTURAL
    # =========================================================================
    "fields": {
        "scenes": [
            "field", "field cultivated", "field wild", "meadow", "pasture",
            "wheat field", "corn field", "rice paddy", "vineyard",
            "farm", "farmland", "plantation", "grassland", "prairie",
            "savanna", "steppe",
        ],
        "prompt": """Describe the field/meadow setting: vegetation type and color,
            open space and horizon, agricultural elements if present,
            natural textures, seasonal indicators, and pastoral atmosphere.""",
    },

    # =========================================================================
    # GARDEN / PARK
    # =========================================================================
    "garden": {
        "scenes": [
            "garden", "botanical garden", "flower garden", "vegetable garden",
            "japanese garden", "zen garden", "park", "yard", "lawn",
            "courtyard", "patio", "terrace", "greenhouse", "topiary garden",
        ],
        "prompt": """Describe the garden/park setting: plants and flowers,
            landscaping and design elements, paths or structures,
            maintained vs natural areas, and cultivated atmosphere.""",
    },

    # =========================================================================
    # SNOW / ICE / WINTER
    # =========================================================================
    "winter": {
        "scenes": [
            "snowfield", "ski slope", "ski resort", "ice shelf", "glacier",
            "iceberg", "igloo", "ice skating rink outdoor", "tundra",
            "ski lodge", "chalet",
        ],
        "prompt": """Describe the winter/snow setting: snow coverage and texture,
            ice formations, cold light quality, winter atmosphere,
            contrast with darker elements, and frigid environment.""",
    },

    # =========================================================================
    # SKY / ATMOSPHERIC
    # =========================================================================
    "sky": {
        "scenes": [
            "sky", "cloudscape", "sunset", "sunrise", "aurora",
        ],
        "prompt": """Describe the sky-dominant scene: cloud formations and types,
            color gradients, light quality, atmospheric effects,
            celestial elements if visible, and expansive atmosphere.""",
    },

    # =========================================================================
    # URBAN STREET / OUTDOOR
    # =========================================================================
    "urban_street": {
        "scenes": [
            "street", "crosswalk", "sidewalk", "alley", "downtown",
            "plaza", "square", "pedestrian area", "market outdoor",
            "bazaar", "flea market", "parking lot", "toll plaza",
        ],
        "prompt": """Describe the urban street setting: road and pavement,
            buildings lining the area, signage, urban elements,
            pedestrian or vehicle activity implied, and city atmosphere.""",
    },

    # =========================================================================
    # URBAN ARCHITECTURE / BUILDINGS
    # =========================================================================
    "urban_architecture": {
        "scenes": [
            "building facade", "skyscraper", "tower", "castle", "palace",
            "church outdoor", "cathedral outdoor", "temple outdoor", "mosque outdoor",
            "ruins", "monument", "fountain", "bridge", "overpass", "viaduct",
            "gate", "arch", "dome",
        ],
        "prompt": """Describe the architectural setting: building style and era,
            structural features, materials and textures, scale and grandeur,
            architectural details, and urban presence.""",
    },

    # =========================================================================
    # RESIDENTIAL INDOOR
    # =========================================================================
    "indoor_home": {
        "scenes": [
            "bedroom", "bathroom", "kitchen", "living room", "dining room",
            "home office", "closet", "laundry room", "nursery", "playroom",
            "basement", "attic", "garage indoor", "entrance hall", "staircase",
            "dorm room", "apartment", "hotel room", "motel room",
        ],
        "prompt": """Describe the indoor home setting: room type and function,
            furniture and décor, lighting sources and quality,
            personal touches, materials and textures, and domestic atmosphere.""",
    },

    # =========================================================================
    # COMMERCIAL / HOSPITALITY INDOOR
    # =========================================================================
    "indoor_commercial": {
        "scenes": [
            "restaurant", "bar", "pub", "cafe", "coffee shop", "bakery",
            "food court", "dining hall", "banquet hall", "ballroom",
            "hotel lobby", "reception", "waiting room",
        ],
        "prompt": """Describe the commercial/hospitality setting: establishment type,
            seating and furniture, lighting and ambiance,
            décor style, and service-oriented atmosphere.""",
    },

    # =========================================================================
    # OFFICE / WORKPLACE
    # =========================================================================
    "indoor_office": {
        "scenes": [
            "office", "office building", "cubicle", "conference room",
            "meeting room", "boardroom", "reception desk", "server room",
            "computer room", "control room",
        ],
        "prompt": """Describe the office/workplace setting: workspace layout,
            furniture and equipment, lighting (natural/artificial),
            corporate elements, and professional atmosphere.""",
    },

    # =========================================================================
    # RETAIL / SHOPPING
    # =========================================================================
    "indoor_retail": {
        "scenes": [
            "shopping mall", "department store", "supermarket", "grocery store",
            "convenience store", "boutique", "shoe shop", "clothing store",
            "jewelry shop", "bookstore", "pharmacy", "gift shop",
        ],
        "prompt": """Describe the retail setting: store type and merchandise,
            display arrangements, lighting and signage,
            shopping environment, and commercial atmosphere.""",
    },

    # =========================================================================
    # CULTURAL / ENTERTAINMENT
    # =========================================================================
    "indoor_cultural": {
        "scenes": [
            "museum", "art gallery", "exhibition hall", "theater indoor",
            "movie theater", "concert hall", "auditorium", "opera house",
            "library", "archive", "reading room", "church indoor",
            "cathedral indoor", "temple indoor", "synagogue", "mosque indoor",
        ],
        "prompt": """Describe the cultural/entertainment setting: venue type,
            architectural features, displays or stage,
            seating, acoustics implied, and cultural atmosphere.""",
    },

    # =========================================================================
    # SPORTS / FITNESS
    # =========================================================================
    "sports": {
        "scenes": [
            "gym", "gymnasium", "weight room", "yoga studio", "dojo",
            "boxing ring", "wrestling ring", "basketball court indoor",
            "tennis court", "squash court", "bowling alley", "ice rink",
            "swimming pool indoor", "locker room", "stadium", "arena",
            "baseball field", "soccer field", "football field", "track",
            "golf course", "ski slope", "skate park",
        ],
        "prompt": """Describe the sports/fitness setting: facility type,
            equipment and playing surfaces, lighting,
            athletic elements, and energetic atmosphere.""",
    },

    # =========================================================================
    # TRANSPORTATION
    # =========================================================================
    "transportation": {
        "scenes": [
            "train station", "subway station", "bus station", "airport terminal",
            "airport", "hangar", "airplane cabin", "cockpit", "train interior",
            "bus interior", "subway interior", "car interior", "taxi",
            "boat deck", "ship deck", "ferry", "highway", "freeway",
            "tunnel", "parking garage", "gas station",
        ],
        "prompt": """Describe the transportation setting: vehicle or station type,
            interior or exterior details, transit elements,
            movement or waiting implied, and travel atmosphere.""",
    },

    # =========================================================================
    # INDUSTRIAL / UTILITY
    # =========================================================================
    "industrial": {
        "scenes": [
            "factory", "warehouse", "loading dock", "industrial area",
            "power plant", "wind farm", "oil rig", "refinery",
            "construction site", "scaffolding", "junkyard", "landfill",
            "mine", "quarry", "dam",
        ],
        "prompt": """Describe the industrial setting: facility type and function,
            machinery and structures, scale,
            working environment, and industrial atmosphere.""",
    },

    # =========================================================================
    # STUDIO / PHOTOGRAPHY
    # =========================================================================
    "studio": {
        "scenes": [
            "studio", "photo studio", "recording studio", "art studio",
            "television studio", "broadcast studio", "stage",
        ],
        "prompt": """Describe the studio setting: studio type and setup,
            lighting equipment, backdrop or set,
            professional elements, and creative atmosphere.""",
    },
}


def get_scene_prompt(scene_name: str) -> Tuple[str, str]:
    """
    Get the specific prompt for a scene.

    Args:
        scene_name: Scene name from Places365 (e.g., "beach", "forest")

    Returns:
        Tuple of (group_name, prompt) or ("default", default_prompt)
    """
    if not scene_name:
        return "default", _get_default_prompt()

    scene_lower = scene_name.lower().replace("_", " ")

    # First pass: exact match across all groups
    for group_name, group_data in SCENE_GROUPS.items():
        scenes = group_data["scenes"]
        if scene_lower in scenes:
            return group_name, group_data["prompt"]

    # Second pass: check if scene starts with a keyword (more specific)
    for group_name, group_data in SCENE_GROUPS.items():
        scenes = group_data["scenes"]
        for scene_keyword in scenes:
            # Match "beach house" to "beach" but not "office" to "home office"
            if scene_lower.startswith(scene_keyword + " ") or scene_lower.endswith(" " + scene_keyword):
                return group_name, group_data["prompt"]

    # Check for common keywords as fallback
    keyword_to_group = {
        "beach": "water_coastal",
        "ocean": "water_coastal",
        "sea": "water_coastal",
        "coast": "water_coastal",
        "lake": "water_freshwater",
        "river": "water_freshwater",
        "water": "water_freshwater",
        "forest": "forest",
        "tree": "forest",
        "wood": "forest",
        "jungle": "forest",
        "mountain": "mountain",
        "hill": "mountain",
        "valley": "mountain",
        "canyon": "mountain",
        "desert": "desert",
        "sand": "desert",
        "field": "fields",
        "meadow": "fields",
        "farm": "fields",
        "garden": "garden",
        "park": "garden",
        "snow": "winter",
        "ice": "winter",
        "ski": "winter",
        "sky": "sky",
        "cloud": "sky",
        "sunset": "sky",
        "street": "urban_street",
        "road": "urban_street",
        "alley": "urban_street",
        "plaza": "urban_street",
        "building": "urban_architecture",
        "tower": "urban_architecture",
        "bridge": "urban_architecture",
        "castle": "urban_architecture",
        "church": "urban_architecture",
        "bedroom": "indoor_home",
        "bathroom": "indoor_home",
        "kitchen": "indoor_home",
        "living": "indoor_home",
        "house": "indoor_home",
        "home": "indoor_home",
        "restaurant": "indoor_commercial",
        "bar": "indoor_commercial",
        "cafe": "indoor_commercial",
        "hotel": "indoor_commercial",
        "office": "indoor_office",
        "conference": "indoor_office",
        "shop": "indoor_retail",
        "store": "indoor_retail",
        "mall": "indoor_retail",
        "market": "indoor_retail",
        "museum": "indoor_cultural",
        "gallery": "indoor_cultural",
        "theater": "indoor_cultural",
        "library": "indoor_cultural",
        "church": "indoor_cultural",
        "gym": "sports",
        "stadium": "sports",
        "court": "sports",
        "pool": "sports",
        "train": "transportation",
        "airport": "transportation",
        "station": "transportation",
        "car": "transportation",
        "factory": "industrial",
        "warehouse": "industrial",
        "construction": "industrial",
        "studio": "studio",
    }

    for keyword, group in keyword_to_group.items():
        if keyword in scene_lower:
            return group, SCENE_GROUPS[group]["prompt"]

    return "default", _get_default_prompt()


def _get_default_prompt() -> str:
    """Get the default environment prompt."""
    return """Describe the environment and setting: location type,
        notable features, atmosphere, lighting conditions,
        and any contextual elements that establish the scene."""


def get_scene_group(scene_name: str) -> Optional[str]:
    """
    Get just the group name for a scene.

    Args:
        scene_name: Scene name from Places365

    Returns:
        Group name or None
    """
    group_name, _ = get_scene_prompt(scene_name)
    return group_name if group_name != "default" else None


def format_scene_prompt(scene_name: str) -> str:
    """
    Get a formatted, cleaned prompt for a scene.

    Args:
        scene_name: Scene name from Places365

    Returns:
        Cleaned prompt string
    """
    _, prompt = get_scene_prompt(scene_name)
    # Clean up whitespace
    lines = prompt.strip().split('\n')
    cleaned = ' '.join(line.strip() for line in lines if line.strip())
    return cleaned
