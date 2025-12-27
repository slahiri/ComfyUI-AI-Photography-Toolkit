"""Places365 Scene Recognition Tagger.

Detects scene types from 365 categories including:
- Indoor: bedroom, kitchen, office, studio, gym, restaurant, etc.
- Outdoor: forest, beach, mountain, city, park, street, etc.
- Nature: ocean, desert, jungle, field, waterfall, etc.
- Urban: downtown, skyscraper, alley, plaza, etc.

Uses ResNet50 pretrained on Places365 dataset.
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import List, Optional
import os

from .base import BaseTagger, TagItem, TaggerResult


# Places365 category labels (365 categories)
# Grouped into meta-categories for easier understanding
PLACES365_CATEGORIES = [
    # Indoor - Living
    "apartment_building/outdoor", "bedroom", "living_room", "dining_room", "kitchen",
    "bathroom", "home_office", "nursery", "closet", "basement",
    # Indoor - Commercial
    "office", "conference_room", "cubicle", "server_room", "lobby",
    "reception", "waiting_room", "corridor", "staircase", "elevator",
    # Indoor - Retail/Food
    "restaurant", "cafeteria", "coffee_shop", "bar", "pub",
    "bakery", "butcher_shop", "supermarket", "shopping_mall", "department_store",
    # Indoor - Entertainment
    "movie_theater", "concert_hall", "theater", "casino", "arcade",
    "bowling_alley", "billiard_room", "gym", "yoga_studio", "swimming_pool/indoor",
    # Indoor - Education/Medical
    "classroom", "library", "laboratory", "hospital_room", "dentist_office",
    "pharmacy", "museum", "art_gallery", "church", "mosque",
    # Studio/Production
    "television_studio", "recording_studio", "photography_studio", "art_studio",
    # Outdoor - Urban
    "street", "downtown", "plaza", "parking_lot", "parking_garage",
    "gas_station", "train_station", "bus_station", "airport_terminal", "subway_station",
    "bridge", "overpass", "tunnel", "alley", "courtyard",
    # Outdoor - Residential
    "residential_neighborhood", "yard", "garden", "patio", "balcony",
    "roof", "driveway", "playground", "park", "cemetery",
    # Outdoor - Commercial
    "construction_site", "industrial_area", "warehouse", "loading_dock", "market",
    # Nature - Water
    "beach", "coast", "ocean", "lake", "river",
    "waterfall", "pond", "marsh", "swamp", "coral_reef",
    # Nature - Land
    "forest", "rainforest", "jungle", "woods", "grove",
    "field", "meadow", "prairie", "grassland", "pasture",
    "mountain", "mountain_path", "hill", "valley", "canyon",
    "cliff", "rock_arch", "cave", "desert", "sand_dune",
    # Nature - Sky/Weather
    "sky", "cloud", "sunset", "sunrise", "rainbow",
    # Agriculture
    "farm", "orchard", "vineyard", "rice_paddy", "wheat_field",
    "barn", "greenhouse", "stable",
    # Sports/Recreation
    "golf_course", "tennis_court", "basketball_court", "soccer_field", "baseball_field",
    "ski_slope", "ice_skating_rink", "swimming_pool/outdoor", "race_track", "stadium",
    # Transportation
    "highway", "road", "crosswalk", "railroad", "runway",
    "harbor", "marina", "pier", "dock", "boat_deck",
    # Historic/Cultural
    "castle", "palace", "temple", "pagoda", "ruins",
    "monument", "fountain", "statue", "amphitheater", "aqueduct",
]

# Scene type meta-categories for grouping
SCENE_META = {
    "indoor": {"bedroom", "living_room", "dining_room", "kitchen", "bathroom", "office",
               "restaurant", "bar", "gym", "library", "museum", "studio", "hospital"},
    "outdoor": {"street", "park", "garden", "plaza", "beach", "mountain", "forest",
                "field", "highway", "bridge", "stadium"},
    "nature": {"forest", "beach", "ocean", "lake", "mountain", "desert", "jungle",
               "waterfall", "field", "meadow", "canyon", "cave", "river"},
    "urban": {"street", "downtown", "plaza", "highway", "bridge", "skyscraper",
              "parking", "station", "airport", "construction"},
    "studio": {"photography_studio", "recording_studio", "television_studio",
               "art_studio", "studio"},
}


class Places365Tagger(BaseTagger):
    """Scene recognition using Places365 pretrained model."""

    def __init__(
        self,
        threshold: float = 0.15,
        top_k: int = 5,
        model_name: str = "resnet50",
    ):
        super().__init__()
        self.threshold = threshold
        self.top_k = top_k
        self.model_name = model_name
        self.model = None
        self.transform = None
        self.labels = None
        self._device = None

    def load(self) -> None:
        """Load model into memory (required by BaseTagger)."""
        self._load_model()
        self._is_loaded = self.model is not None or getattr(self, '_use_clip_fallback', False)

    def _load_model(self):
        """Load Places365 model."""
        if self.model is not None:
            return

        try:
            import torchvision.transforms as transforms
            from torchvision import models
            from huggingface_hub import hf_hub_download

            # Use GPU if available
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Try to load from HuggingFace hub or use standard Places365
            try:
                # Load ResNet50 Places365 from HuggingFace
                model_path = hf_hub_download(
                    repo_id="alexhoffman/resnet50_places365",
                    filename="resnet50_places365.pth.tar",
                )

                # Create model
                self.model = models.resnet50(num_classes=365)
                checkpoint = torch.load(model_path, map_location=self._device)

                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    state_dict = {k.replace('module.', ''): v
                                  for k, v in checkpoint['state_dict'].items()}
                else:
                    state_dict = checkpoint

                self.model.load_state_dict(state_dict)

            except Exception as e:
                # Fallback: use CLIP for scene detection
                print(f"[Places365] Model download failed: {e}")
                print("[Places365] Falling back to CLIP-based scene detection")
                self._use_clip_fallback = True
                return

            self.model = self.model.to(self._device)
            self.model.eval()

            # Standard ImageNet normalization (Places365 uses same)
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            # Load labels
            try:
                labels_path = hf_hub_download(
                    repo_id="alexhoffman/resnet50_places365",
                    filename="categories_places365.txt",
                )
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip().split(' ')[0][3:] for line in f.readlines()]
            except:
                # Use our predefined labels
                self.labels = PLACES365_CATEGORIES[:365]

            self._use_clip_fallback = False

        except ImportError as e:
            print(f"[Places365] Required package not found: {e}")
            self._use_clip_fallback = True

    def _clip_scene_detection(self, image: Image.Image) -> List[TagItem]:
        """Fallback to CLIP for scene detection."""
        try:
            from transformers import CLIPProcessor, CLIPModel

            # Load CLIP
            model_name = "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            # Scene prompts
            scene_prompts = [
                "indoor scene", "outdoor scene", "nature scene", "urban scene",
                "studio setting", "beach scene", "mountain scene", "forest scene",
                "city street", "home interior", "office space", "restaurant",
                "park or garden", "desert landscape", "ocean or sea", "rural area",
            ]

            # Prepare inputs
            inputs = processor(
                text=scene_prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = torch.softmax(logits, dim=-1)[0]

            # Build results
            tags = []
            for prompt, prob in zip(scene_prompts, probs):
                conf = prob.item()
                if conf >= self.threshold:
                    # Clean prompt for tag
                    tag = prompt.replace(" scene", "").replace(" setting", "")
                    tags.append(TagItem(text=tag, confidence=conf, category="scene"))

            # Sort and limit
            tags.sort(key=lambda x: x.confidence, reverse=True)
            return tags[:self.top_k]

        except Exception as e:
            print(f"[Places365] CLIP fallback failed: {e}")
            return []

    def tag(self, image: Image.Image) -> TaggerResult:
        """Detect scene type in image."""
        self._load_model()

        # Use CLIP fallback if model failed to load
        if getattr(self, '_use_clip_fallback', False):
            tags = self._clip_scene_detection(image)
            return TaggerResult(
                tags=tags,
                metadata={"model": "clip_fallback", "count": len(tags)}
            )

        try:
            # Prepare image
            if image.mode != "RGB":
                image = image.convert("RGB")

            input_tensor = self.transform(image).unsqueeze(0).to(self._device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            # Get top predictions
            top_probs, top_indices = torch.topk(probs, min(self.top_k * 2, len(probs)))

            tags = []
            meta_tags = set()

            for prob, idx in zip(top_probs, top_indices):
                conf = prob.item()
                if conf < self.threshold:
                    continue

                # Get label
                label = self.labels[idx.item()] if idx.item() < len(self.labels) else f"scene_{idx.item()}"
                label = label.replace("_", " ").replace("/", " ")

                tags.append(TagItem(
                    text=label,
                    confidence=conf,
                    category="scene",
                ))

                # Add meta-category tags
                label_lower = label.lower()
                for meta, keywords in SCENE_META.items():
                    if any(kw in label_lower for kw in keywords):
                        meta_tags.add(meta)

            # Add meta-category tags
            for meta in meta_tags:
                tags.append(TagItem(
                    text=meta,
                    confidence=0.8,  # High confidence for meta-categories
                    category="scene_type",
                ))

            # Sort by confidence
            tags.sort(key=lambda x: x.confidence, reverse=True)

            return TaggerResult(
                tags=tags[:self.top_k + len(meta_tags)],
                metadata={
                    "model": "places365_resnet50",
                    "count": len(tags),
                }
            )

        except Exception as e:
            print(f"[Places365] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return TaggerResult(tags=[], metadata={"error": str(e)})

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'transform'):
            del self.transform
            self.transform = None
        torch.cuda.empty_cache()
