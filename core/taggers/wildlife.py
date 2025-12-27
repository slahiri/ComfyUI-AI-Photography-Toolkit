"""Wildlife/Nature Tagger - Detect animals, pets, and natural environments.

Detects:
- Pets: dog, cat, hamster, rabbit, fish, bird
- Wildlife: deer, bear, wolf, fox, lion, elephant, etc.
- Birds: eagle, owl, parrot, penguin, flamingo, etc.
- Marine life: fish, whale, dolphin, shark, turtle, etc.
- Insects: butterfly, bee, spider, dragonfly, etc.
- Nature/Biomes: forest, jungle, desert, ocean, tundra, etc.

Uses CLIP zero-shot classification with comprehensive animal/nature vocabulary.
"""

import torch
from PIL import Image
from typing import List, Optional

from .base import BaseTagger, TagItem, TaggerResult


# Comprehensive animal and nature vocabulary
WILDLIFE_VOCABULARY = {
    "pets": [
        "pet dog", "pet cat", "puppy", "kitten",
        "golden retriever", "labrador", "german shepherd", "bulldog", "poodle",
        "persian cat", "siamese cat", "tabby cat", "maine coon",
        "pet rabbit", "pet hamster", "pet guinea pig", "pet bird",
        "pet parrot", "pet fish", "aquarium fish", "pet turtle",
        "pet snake", "pet lizard", "pet ferret",
    ],
    "dogs": [
        "dog", "puppy", "golden retriever", "labrador retriever",
        "german shepherd", "bulldog", "poodle", "beagle",
        "rottweiler", "yorkshire terrier", "boxer", "dachshund",
        "siberian husky", "great dane", "doberman", "shih tzu",
        "boston terrier", "pomeranian", "chihuahua", "corgi",
    ],
    "cats": [
        "cat", "kitten", "persian cat", "siamese cat",
        "maine coon", "ragdoll cat", "british shorthair", "bengal cat",
        "abyssinian cat", "sphynx cat", "scottish fold", "russian blue",
        "tabby cat", "calico cat", "black cat", "white cat",
        "orange cat", "tuxedo cat", "striped cat",
    ],
    "wildlife_mammals": [
        "wild animal", "wildlife", "mammal",
        "deer", "elk", "moose", "antelope", "gazelle",
        "bear", "grizzly bear", "polar bear", "black bear", "panda",
        "wolf", "fox", "red fox", "arctic fox", "coyote",
        "lion", "tiger", "leopard", "cheetah", "jaguar", "cougar",
        "elephant", "african elephant", "asian elephant",
        "giraffe", "zebra", "hippopotamus", "rhinoceros",
        "monkey", "chimpanzee", "gorilla", "orangutan", "baboon",
        "kangaroo", "koala", "wombat", "platypus",
        "squirrel", "chipmunk", "raccoon", "skunk", "beaver",
        "rabbit", "hare", "hedgehog", "badger", "otter",
        "seal", "sea lion", "walrus",
    ],
    "birds": [
        "bird", "songbird", "perching bird",
        "eagle", "bald eagle", "golden eagle", "hawk", "falcon",
        "owl", "barn owl", "snowy owl", "great horned owl",
        "parrot", "macaw", "cockatoo", "parakeet", "cockatiel",
        "penguin", "emperor penguin", "flamingo", "crane", "heron",
        "duck", "goose", "swan", "pelican", "seagull",
        "crow", "raven", "magpie", "jay", "blue jay",
        "sparrow", "robin", "cardinal", "finch", "canary",
        "hummingbird", "woodpecker", "toucan", "peacock", "turkey",
        "chicken", "rooster", "pigeon", "dove",
    ],
    "marine_life": [
        "fish", "tropical fish", "goldfish", "koi fish",
        "shark", "great white shark", "whale", "blue whale", "orca",
        "dolphin", "porpoise", "seal", "sea lion",
        "octopus", "squid", "jellyfish", "starfish", "sea urchin",
        "sea turtle", "coral", "coral reef", "clownfish",
        "crab", "lobster", "shrimp", "seahorse",
        "manta ray", "stingray", "eel", "salmon", "tuna",
    ],
    "reptiles_amphibians": [
        "reptile", "amphibian",
        "snake", "python", "cobra", "rattlesnake", "boa",
        "lizard", "gecko", "iguana", "chameleon", "komodo dragon",
        "crocodile", "alligator", "turtle", "tortoise",
        "frog", "tree frog", "toad", "salamander", "newt",
    ],
    "insects": [
        "insect", "bug",
        "butterfly", "monarch butterfly", "moth",
        "bee", "bumblebee", "honeybee", "wasp", "hornet",
        "dragonfly", "damselfly", "firefly", "ladybug", "beetle",
        "ant", "spider", "tarantula", "scorpion",
        "grasshopper", "cricket", "praying mantis", "cicada",
        "caterpillar", "mosquito", "fly", "housefly",
    ],
    "biomes": [
        "forest", "rainforest", "tropical rainforest", "temperate forest",
        "jungle", "woodland", "grove", "thicket",
        "desert", "sand dunes", "oasis", "arid landscape",
        "ocean", "sea", "coral reef", "deep sea",
        "mountain", "alpine", "mountain peak", "valley",
        "grassland", "savanna", "prairie", "meadow", "steppe",
        "tundra", "arctic", "antarctic", "glacier", "ice field",
        "wetland", "swamp", "marsh", "mangrove", "bog",
        "river", "stream", "lake", "pond", "waterfall",
        "beach", "coast", "cliff", "cave", "canyon",
    ],
    "plants": [
        "tree", "palm tree", "oak tree", "pine tree", "maple tree",
        "flower", "rose", "sunflower", "tulip", "orchid", "lily",
        "plant", "fern", "moss", "grass", "bamboo",
        "bush", "shrub", "cactus", "succulent",
        "mushroom", "fungus", "lichen",
        "leaf", "leaves", "foliage", "vegetation",
    ],
}


class WildlifeTagger(BaseTagger):
    """Wildlife and nature detection using CLIP zero-shot."""

    def __init__(
        self,
        threshold: float = 0.20,
        top_k: int = 10,
        categories: Optional[List[str]] = None,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        Initialize wildlife tagger.

        Args:
            threshold: Minimum confidence for tags
            top_k: Maximum total tags to return
            categories: Which categories to detect (None = all)
            model_name: CLIP model to use
        """
        super().__init__()
        self.threshold = threshold
        self.top_k = top_k
        self.categories = categories or list(WILDLIFE_VOCABULARY.keys())
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
            print(f"[Wildlife] Failed to load model: {e}")
            import traceback
            traceback.print_exc()

    def _precompute_text_embeddings(self):
        """Precompute text embeddings for all vocabulary."""
        all_prompts = []
        prompt_categories = []
        prompt_originals = []

        for category in self.categories:
            if category not in WILDLIFE_VOCABULARY:
                continue

            for item in WILDLIFE_VOCABULARY[category]:
                # Create detection prompt
                prompt = f"a photo of a {item}" if not item.startswith("a ") else f"a photo of {item}"
                all_prompts.append(prompt)
                prompt_categories.append(category)
                prompt_originals.append(item)

        if not all_prompts:
            return

        # Batch process in chunks to avoid memory issues
        batch_size = 100
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
            "prompts": prompt_originals,
            "categories": prompt_categories,
            "embeddings": torch.cat(all_embeddings, dim=0),
        }

    def tag(self, image: Image.Image) -> TaggerResult:
        """Detect wildlife and nature in image."""
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
            top_k_count = min(self.top_k * 3, len(similarities))  # Get more for filtering
            top_probs, top_indices = torch.topk(similarities, top_k_count)

            tags = []
            seen_tags = set()
            category_counts = {}

            for prob, idx in zip(top_probs, top_indices):
                # Normalize confidence (CLIP similarities are typically 0.2-0.4)
                conf = (prob.item() - 0.15) / 0.25  # Scale to more readable range
                conf = max(0, min(1, conf))  # Clamp to 0-1

                if conf < self.threshold:
                    continue

                # Get tag info
                tag_text = self._text_embeddings["prompts"][idx.item()]
                category = self._text_embeddings["categories"][idx.item()]

                # Skip duplicates
                if tag_text.lower() in seen_tags:
                    continue
                seen_tags.add(tag_text.lower())

                tags.append(TagItem(
                    text=tag_text,
                    confidence=conf,
                    category=category,
                ))

                category_counts[category] = category_counts.get(category, 0) + 1

                if len(tags) >= self.top_k:
                    break

            # Add meta-tags based on detections
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
            print(f"[Wildlife] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return TaggerResult(tags=[], metadata={"error": str(e)})

    def _generate_meta_tags(self, tags: List[TagItem], category_counts: dict) -> List[TagItem]:
        """Generate meta-tags based on detections."""
        meta_tags = []

        # Has animals?
        animal_categories = {"pets", "dogs", "cats", "wildlife_mammals", "birds",
                            "marine_life", "reptiles_amphibians", "insects"}
        if any(cat in category_counts for cat in animal_categories):
            meta_tags.append(TagItem(
                text="contains animals",
                confidence=0.9,
                category="meta",
            ))

        # Specific pet detection
        if "dogs" in category_counts or "cats" in category_counts or "pets" in category_counts:
            meta_tags.append(TagItem(
                text="pet photo",
                confidence=0.85,
                category="meta",
            ))

        # Wildlife detection
        if "wildlife_mammals" in category_counts or "birds" in category_counts:
            meta_tags.append(TagItem(
                text="wildlife photo",
                confidence=0.85,
                category="meta",
            ))

        # Nature detection
        if "biomes" in category_counts or "plants" in category_counts:
            meta_tags.append(TagItem(
                text="nature scene",
                confidence=0.85,
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
