"""Sky & Weather Tagger - Detect cloud types, sun/moon, sky conditions, and weather.

Categories:
- Cloud Types: cumulus, stratus, cirrus, etc. (WMO classification)
- Sun: sunrise, sunset, golden hour, harsh sun, sun rays
- Moon: full moon, crescent, moonlit, lunar eclipse
- Sky Conditions: blue sky, overcast, dramatic, stormy
- Weather: sunny, rainy, foggy, snowy, stormy
- Celestial: stars, aurora, milky way

Uses CLIP zero-shot classification with comprehensive sky/weather vocabulary.
"""

import torch
from PIL import Image
from typing import List, Optional, Dict

from .base import BaseTagger, TagItem, TaggerResult


# Sky and weather vocabulary
SKY_WEATHER_VOCABULARY = {
    "cloud_types": [
        # High clouds (above 6km)
        "cirrus clouds - thin wispy ice clouds",
        "cirrocumulus clouds - small white puffs",
        "cirrostratus clouds - thin sheet covering sky",
        # Mid clouds (2-6km)
        "altocumulus clouds - white gray patches",
        "altostratus clouds - gray sheet covering sun",
        "nimbostratus clouds - dark rain clouds",
        # Low clouds (below 2km)
        "cumulus clouds - puffy white clouds",
        "stratocumulus clouds - lumpy gray layer",
        "stratus clouds - flat gray layer",
        "cumulonimbus clouds - tall thunderstorm clouds",
        # Other
        "contrails - airplane trails in sky",
        "fog or mist",
        "mammatus clouds - pouch-like formations",
        "lenticular clouds - lens shaped clouds",
    ],
    "sun": [
        "sunrise with orange sky",
        "sunset with red and orange sky",
        "golden hour warm sunlight",
        "blue hour twilight",
        "harsh midday sun",
        "sun rays through clouds",
        "sun rays through trees",
        "sun flare in photo",
        "lens flare from sun",
        "silhouette against sun",
        "backlit by sun",
        "sun reflection on water",
        "sun behind clouds",
        "sun peeking through clouds",
        "dappled sunlight",
    ],
    "moon": [
        "full moon in night sky",
        "crescent moon",
        "half moon",
        "gibbous moon",
        "new moon dark sky",
        "moonlit scene",
        "moonlight reflection on water",
        "moon behind clouds",
        "harvest moon - large orange moon",
        "blood moon - lunar eclipse",
        "supermoon - large bright moon",
        "moon and stars",
    ],
    "sky_conditions": [
        "clear blue sky",
        "bright blue sky with white clouds",
        "overcast gray sky",
        "dramatic cloudy sky",
        "stormy dark sky",
        "moody atmospheric sky",
        "pastel colored sky",
        "pink sky",
        "purple sky",
        "orange sky",
        "red sky",
        "gradient sky colors",
        "twilight sky",
        "dusk sky",
        "dawn sky",
        "hazy sky",
        "smoky sky",
    ],
    "weather": [
        "sunny clear weather",
        "partly cloudy weather",
        "cloudy overcast weather",
        "rainy weather with rain drops",
        "heavy rain downpour",
        "light rain drizzle",
        "thunderstorm with lightning",
        "snowy weather with snowfall",
        "blizzard heavy snow",
        "foggy misty weather",
        "dense fog low visibility",
        "windy weather",
        "stormy weather",
        "hail storm",
        "rainbow in sky",
        "double rainbow",
    ],
    "celestial": [
        "starry night sky",
        "stars visible in night sky",
        "milky way galaxy",
        "shooting star meteor",
        "meteor shower",
        "aurora borealis northern lights",
        "aurora australis southern lights",
        "night sky astrophotography",
        "constellation in sky",
        "planets visible in sky",
        "venus in sky",
        "comet in sky",
    ],
    "atmospheric": [
        "light rays - crepuscular rays",
        "god rays through clouds",
        "volumetric light in atmosphere",
        "atmospheric haze",
        "dust particles in light",
        "light beam through window",
        "sunbeam spotlight effect",
        "rainbow light refraction",
        "sun dog - parhelion",
        "light pillar phenomenon",
        "glory halo around shadow",
    ],
}

# Weather classification labels (for reference)
WEATHER_CLASSES = [
    "cloudy/overcast",
    "foggy/hazy",
    "rain/storm",
    "snow/frosty",
    "sun/clear",
]


class SkyWeatherTagger(BaseTagger):
    """Sky and weather detection using CLIP zero-shot classification."""

    def __init__(
        self,
        threshold: float = 0.18,
        top_k: int = 8,
        categories: Optional[List[str]] = None,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        Initialize sky/weather tagger.

        Args:
            threshold: Minimum confidence for tags
            top_k: Maximum tags to return
            categories: Which categories to detect (None = all)
            model_name: CLIP model to use
        """
        super().__init__()
        self.threshold = threshold
        self.top_k = top_k
        self.categories = categories or list(SKY_WEATHER_VOCABULARY.keys())
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._device = None
        self._text_embeddings = None

    def load(self) -> None:
        """Load model into memory (required by BaseTagger)."""
        self._load_model()
        self._is_loaded = self.model is not None

    def _load_model(self):
        """Load CLIP model and precompute text embeddings."""
        if self.model is not None:
            return

        try:
            from transformers import CLIPProcessor, CLIPModel

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = self.model.to(self._device)
            self.model.eval()

            # Precompute text embeddings
            self._precompute_text_embeddings()

        except Exception as e:
            print(f"[SkyWeather] Failed to load model: {e}")
            import traceback
            traceback.print_exc()

    def _precompute_text_embeddings(self):
        """Precompute text embeddings for all vocabulary."""
        all_prompts = []
        prompt_categories = []
        prompt_items = []

        for category in self.categories:
            if category not in SKY_WEATHER_VOCABULARY:
                continue

            for item in SKY_WEATHER_VOCABULARY[category]:
                # Create detection prompt
                if " - " in item:
                    # Has description, use just the main part for prompt
                    main_part = item.split(" - ")[0]
                    prompt = f"a photo showing {main_part}"
                else:
                    prompt = f"a photo showing {item}"

                all_prompts.append(prompt)
                prompt_categories.append(category)
                prompt_items.append(item)

        if not all_prompts:
            return

        # Batch process
        batch_size = 50
        all_embeddings = []

        for i in range(0, len(all_prompts), batch_size):
            batch = all_prompts[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_features)

        self._text_embeddings = {
            "items": prompt_items,
            "categories": prompt_categories,
            "embeddings": torch.cat(all_embeddings, dim=0),
        }

    def tag(self, image: Image.Image) -> TaggerResult:
        """Detect sky/weather elements in image."""
        self._load_model()

        if self.model is None or self._text_embeddings is None:
            return TaggerResult(tags=[], metadata={"error": "Model not loaded"})

        try:
            # Prepare image
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self._device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            text_features = self._text_embeddings["embeddings"]
            similarities = (image_features @ text_features.T)[0]

            # Get top matches
            top_k_count = min(self.top_k * 3, len(similarities))
            top_probs, top_indices = torch.topk(similarities, top_k_count)

            tags = []
            seen_tags = set()
            category_counts = {}

            for prob, idx in zip(top_probs, top_indices):
                # Normalize confidence
                conf = (prob.item() - 0.16) / 0.22
                conf = max(0, min(1, conf))

                if conf < self.threshold:
                    continue

                item = self._text_embeddings["items"][idx.item()]
                category = self._text_embeddings["categories"][idx.item()]

                # Clean item for tag
                tag_text = self._clean_item(item)

                # Skip duplicates (check core words)
                tag_key = self._get_tag_key(tag_text)
                if tag_key in seen_tags:
                    continue
                seen_tags.add(tag_key)

                tags.append(TagItem(
                    text=tag_text,
                    confidence=conf,
                    category=category,
                ))

                category_counts[category] = category_counts.get(category, 0) + 1

                if len(tags) >= self.top_k:
                    break

            # Generate meta-tags based on detections
            meta_tags = self._generate_meta_tags(tags, category_counts)
            tags.extend(meta_tags)

            # Sort by confidence
            tags.sort(key=lambda x: x.confidence, reverse=True)

            return TaggerResult(
                tags=tags,
                metadata={
                    "model": self.model_name,
                    "categories_detected": category_counts,
                    "count": len(tags),
                }
            )

        except Exception as e:
            print(f"[SkyWeather] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return TaggerResult(tags=[], metadata={"error": str(e)})

    def _clean_item(self, item: str) -> str:
        """Clean item text for use as tag."""
        # Remove description after " - "
        if " - " in item:
            item = item.split(" - ")[0]
        return item.strip()

    def _get_tag_key(self, tag: str) -> str:
        """Get key for deduplication."""
        # Extract main concept
        words = tag.lower().split()
        # Skip common words
        key_words = [w for w in words if w not in {"a", "the", "in", "with", "and", "or", "on"}]
        return " ".join(key_words[:3])

    def _generate_meta_tags(self, tags: List[TagItem], category_counts: dict) -> List[TagItem]:
        """Generate meta-tags based on detections."""
        meta_tags = []

        # Time of day detection
        tag_texts = [t.text.lower() for t in tags]

        if any("sunrise" in t or "dawn" in t for t in tag_texts):
            meta_tags.append(TagItem(
                text="time: sunrise",
                confidence=0.85,
                category="time_of_day",
            ))
        elif any("sunset" in t or "dusk" in t for t in tag_texts):
            meta_tags.append(TagItem(
                text="time: sunset",
                confidence=0.85,
                category="time_of_day",
            ))
        elif any("golden hour" in t for t in tag_texts):
            meta_tags.append(TagItem(
                text="time: golden hour",
                confidence=0.85,
                category="time_of_day",
            ))
        elif any("blue hour" in t or "twilight" in t for t in tag_texts):
            meta_tags.append(TagItem(
                text="time: blue hour",
                confidence=0.85,
                category="time_of_day",
            ))
        elif any("night" in t or "moon" in t or "star" in t for t in tag_texts):
            meta_tags.append(TagItem(
                text="time: night",
                confidence=0.85,
                category="time_of_day",
            ))

        # Weather summary
        if "weather" in category_counts:
            meta_tags.append(TagItem(
                text="contains weather elements",
                confidence=0.80,
                category="meta",
            ))

        # Celestial detection
        if "celestial" in category_counts or "moon" in category_counts:
            meta_tags.append(TagItem(
                text="celestial elements",
                confidence=0.80,
                category="meta",
            ))

        # Dramatic sky
        if any("dramatic" in t or "stormy" in t or "moody" in t for t in tag_texts):
            meta_tags.append(TagItem(
                text="dramatic sky",
                confidence=0.80,
                category="meta",
            ))

        return meta_tags

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self._text_embeddings is not None:
            del self._text_embeddings
            self._text_embeddings = None
        torch.cuda.empty_cache()
