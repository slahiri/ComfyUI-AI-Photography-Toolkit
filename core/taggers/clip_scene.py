"""CLIP Scene/Style Tagger - Zero-shot classification for photo attributes.

Detects:
- Photo style: selfie, portrait, landscape, macro, aerial, street, candid, posed
- Camera type: smartphone, DSLR, film camera, instant camera, webcam, drone
- Photo quality: professional, amateur, snapshot, artistic
- Subject type: person, pet, wildlife, product, food, architecture
- Mood/Atmosphere: romantic, dramatic, peaceful, energetic, moody
"""

import torch
from PIL import Image
from typing import List, Dict, Optional
from dataclasses import dataclass

from .base import BaseTagger, TagItem, TaggerResult


# Vocabulary categories for zero-shot classification
CLIP_VOCABULARIES = {
    "photo_style": [
        "selfie photo",
        "group selfie",
        "mirror selfie",
        "professional portrait",
        "casual portrait",
        "candid shot",
        "posed photo",
        "action shot",
        "landscape photography",
        "macro photography",
        "aerial photography",
        "drone shot",
        "street photography",
        "documentary photo",
        "fashion photography",
        "editorial photo",
        "product photography",
        "food photography",
        "architectural photography",
        "wildlife photography",
        "sports photography",
        "event photography",
        "wedding photography",
        "travel photography",
        "fine art photography",
        "abstract photography",
        "black and white photography",
        "long exposure photo",
        "HDR photo",
        "panoramic photo",
    ],
    "camera_type": [
        "smartphone photo",
        "iPhone photo",
        "Android phone photo",
        "DSLR photo",
        "mirrorless camera photo",
        "film camera photo",
        "instant camera photo",
        "polaroid photo",
        "vintage camera photo",
        "webcam photo",
        "security camera footage",
        "drone camera photo",
        "GoPro action camera photo",
        "professional studio camera",
        "medium format camera photo",
        "disposable camera photo",
        "point and shoot camera photo",
    ],
    "photo_quality": [
        "professional quality photo",
        "amateur photo",
        "casual snapshot",
        "artistic photo",
        "high quality photo",
        "low quality photo",
        "blurry photo",
        "sharp focused photo",
        "well lit photo",
        "poorly lit photo",
        "overexposed photo",
        "underexposed photo",
        "properly exposed photo",
        "noisy grainy photo",
        "clean clear photo",
    ],
    "subject_type": [
        "photo of a person",
        "photo of people",
        "photo of a pet",
        "photo of a dog",
        "photo of a cat",
        "photo of wildlife",
        "photo of a bird",
        "photo of an animal",
        "photo of food",
        "photo of a product",
        "photo of architecture",
        "photo of a building",
        "photo of a vehicle",
        "photo of a car",
        "photo of nature",
        "photo of plants",
        "photo of flowers",
        "photo of the sky",
        "photo of water",
        "photo of a landscape",
        "photo of a cityscape",
        "photo of an interior",
        "photo of art",
        "photo of text or documents",
        "photo of nothing specific",
    ],
    "mood": [
        "romantic mood",
        "dramatic mood",
        "peaceful mood",
        "energetic mood",
        "moody atmosphere",
        "happy cheerful mood",
        "sad melancholic mood",
        "mysterious mood",
        "nostalgic mood",
        "elegant mood",
        "casual relaxed mood",
        "tense mood",
        "playful mood",
        "serious mood",
        "dreamy mood",
        "dark mood",
        "bright cheerful mood",
        "warm cozy mood",
        "cold mood",
        "minimalist mood",
    ],
    "time_of_day": [
        "daytime photo",
        "nighttime photo",
        "golden hour photo",
        "blue hour photo",
        "sunrise photo",
        "sunset photo",
        "midday photo",
        "evening photo",
        "morning photo",
        "twilight photo",
    ],
    "season_weather": [
        "sunny weather",
        "cloudy weather",
        "rainy weather",
        "snowy weather",
        "foggy weather",
        "stormy weather",
        "spring season",
        "summer season",
        "autumn fall season",
        "winter season",
    ],
    "lens_characteristics": [
        # Focal length visual cues
        "wide angle lens distortion",
        "ultra wide angle perspective",
        "fisheye lens effect",
        "standard 50mm perspective",
        "portrait lens compression",
        "telephoto lens compression",
        "super telephoto reach",
        # Aperture/bokeh characteristics
        "shallow depth of field bokeh",
        "creamy smooth bokeh",
        "busy distracting bokeh",
        "deep focus sharp throughout",
        "background blur separation",
        "foreground blur",
        # Lens quality indicators
        "sharp corner to corner",
        "soft vignetting edges",
        "chromatic aberration fringing",
        "lens flare artifacts",
        "starburst light effect",
        # Specialty lenses
        "macro close-up lens",
        "tilt-shift miniature effect",
        "soft focus dreamy lens",
        "anamorphic lens flare",
    ],
    "focal_length_estimate": [
        # Specific focal length visual signatures
        "14mm ultra wide photo",
        "24mm wide angle photo",
        "35mm street photo perspective",
        "50mm standard lens photo",
        "85mm portrait lens photo",
        "135mm telephoto portrait",
        "200mm telephoto photo",
        "400mm super telephoto",
    ],
    # === BODY DETAIL VOCABULARIES ===
    "breast_size": [
        "flat chest",
        "small breasts",
        "medium breasts",
        "large breasts",
        "very large breasts",
        "huge breasts",
        "petite bust",
        "full bust",
        "busty figure",
        "well endowed",
    ],
    "body_exposure": [
        # Sideboob levels
        "no sideboob visible",
        "slight sideboob visible",
        "moderate sideboob visible",
        "significant sideboob visible",
        "extreme sideboob visible",
        # Cleavage levels
        "no cleavage visible",
        "slight cleavage visible",
        "moderate cleavage visible",
        "deep cleavage visible",
        "extreme cleavage visible",
        # Underboob
        "no underboob visible",
        "slight underboob visible",
        "underboob visible",
        # Other exposure
        "midriff exposed",
        "navel visible",
        "back exposed",
        "shoulders bare",
        "legs exposed",
        "thighs visible",
    ],
    "body_type": [
        "slim body type",
        "athletic body type",
        "curvy body type",
        "voluptuous body type",
        "petite body type",
        "tall and slender",
        "hourglass figure",
        "pear shaped body",
        "muscular build",
        "toned physique",
        "soft body",
        "thick thighs",
        "wide hips",
        "narrow waist",
    ],
    # === POSE VOCABULARIES ===
    "arm_pose": [
        # Arms up variations
        "arms at sides",
        "arms relaxed at sides",
        "one arm raised",
        "both arms raised",
        "arms raised above head",
        "arms stretched overhead",
        "arms behind head",
        "hands behind head",
        "arms crossed over chest",
        "arms folded",
        # Hand positions
        "hands on hips",
        "hand on hip",
        "hands in hair",
        "hand running through hair",
        "hands touching face",
        "hand on cheek",
        "hand on chin",
        "hands covering face",
        "hands covering eyes",
        "fingers spread over eyes",
        "hand over mouth",
        "hands clasped together",
        "hands in pockets",
        "pointing gesture",
        "peace sign",
        "waving hand",
        # Arms extended
        "arms outstretched",
        "reaching forward",
        "arms spread wide",
    ],
    "leg_pose": [
        "legs together",
        "legs slightly apart",
        "legs wide apart",
        "one leg forward",
        "weight on one leg",
        "crossed legs standing",
        "crossed legs sitting",
        "legs crossed at ankle",
        "one knee bent",
        "both knees bent",
        "kneeling on one knee",
        "kneeling on both knees",
        "legs tucked under",
        "legs extended forward",
        "legs spread sitting",
    ],
    "body_pose": [
        # Standing
        "standing straight",
        "standing relaxed",
        "standing with hip out",
        "contrapposto pose",
        "s-curve pose",
        "leaning forward",
        "leaning backward",
        "leaning to one side",
        "arched back pose",
        "twisted torso",
        # Sitting
        "sitting upright",
        "sitting relaxed",
        "sitting cross-legged",
        "sitting with legs apart",
        "sitting on edge",
        "lounging pose",
        "reclining pose",
        # Lying
        "lying on back",
        "lying on side",
        "lying on stomach",
        "propped on elbows",
        # Dynamic
        "walking pose",
        "mid-stride",
        "action pose",
        "dance pose",
        "stretching pose",
        "yoga pose",
    ],
    "head_pose": [
        "head straight forward",
        "head tilted left",
        "head tilted right",
        "head tilted back",
        "head tilted down",
        "chin up",
        "chin down",
        "head turned left",
        "head turned right",
        "looking over shoulder",
        "profile view of face",
        "three quarter face view",
    ],
    "gaze_direction": [
        "looking directly at camera",
        "looking at viewer",
        "eyes looking up",
        "eyes looking down",
        "eyes looking left",
        "eyes looking right",
        "eyes closed",
        "looking away from camera",
        "distant gaze",
        "eyes hidden",
        "looking over shoulder at camera",
        "side eye glance",
        "downcast eyes",
        "upward gaze",
    ],
    "facial_expression": [
        "neutral expression",
        "slight smile",
        "big smile",
        "laughing",
        "smirking",
        "pouting",
        "lips parted",
        "mouth open",
        "biting lip",
        "tongue out",
        "blowing kiss",
        "serious expression",
        "sultry expression",
        "seductive look",
        "playful expression",
        "surprised expression",
        "thinking expression",
        "confident expression",
    ],
}


class CLIPSceneTagger(BaseTagger):
    """Zero-shot scene/style classification using CLIP."""

    def __init__(
        self,
        threshold: float = 0.15,
        top_k_per_category: int = 3,
        categories: Optional[List[str]] = None,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        Initialize CLIP scene tagger.

        Args:
            threshold: Minimum confidence for tags
            top_k_per_category: Max tags per category
            categories: Which categories to use (None = all)
            model_name: CLIP model to use
        """
        super().__init__()
        self.threshold = threshold
        self.top_k_per_category = top_k_per_category
        self.categories = categories or list(CLIP_VOCABULARIES.keys())
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

            # Precompute text embeddings for all prompts
            self._precompute_text_embeddings()

        except Exception as e:
            print(f"[CLIPScene] Failed to load model: {e}")
            import traceback
            traceback.print_exc()

    def _precompute_text_embeddings(self):
        """Precompute text embeddings for all vocabulary prompts."""
        self._text_embeddings = {}

        for category in self.categories:
            if category not in CLIP_VOCABULARIES:
                continue

            prompts = CLIP_VOCABULARIES[category]
            inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            self._text_embeddings[category] = {
                "prompts": prompts,
                "embeddings": text_features,
            }

    def tag(self, image: Image.Image) -> TaggerResult:
        """Classify image attributes using CLIP zero-shot."""
        self._load_model()

        if self.model is None:
            return TaggerResult(tags=[], metadata={"error": "Model not loaded"})

        try:
            # Prepare image
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self._device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_tags = []
            category_results = {}

            # Compare with each category's text embeddings
            for category, data in self._text_embeddings.items():
                prompts = data["prompts"]
                text_features = data["embeddings"]

                # Compute similarities
                similarities = (image_features @ text_features.T)[0]
                probs = torch.softmax(similarities * 100, dim=0)  # Temperature scaling

                # Get top matches
                top_probs, top_indices = torch.topk(probs, min(self.top_k_per_category, len(prompts)))

                category_tags = []
                for prob, idx in zip(top_probs, top_indices):
                    conf = prob.item()
                    if conf < self.threshold:
                        continue

                    # Clean prompt for tag
                    prompt = prompts[idx.item()]
                    tag = self._clean_prompt(prompt)

                    category_tags.append(TagItem(
                        text=tag,
                        confidence=conf,
                        category=category,
                    ))

                all_tags.extend(category_tags)
                category_results[category] = len(category_tags)

            # Sort by confidence
            all_tags.sort(key=lambda x: x.confidence, reverse=True)

            return TaggerResult(
                tags=all_tags,
                metadata={
                    "model": self.model_name,
                    "categories": category_results,
                    "count": len(all_tags),
                }
            )

        except Exception as e:
            print(f"[CLIPScene] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return TaggerResult(tags=[], metadata={"error": str(e)})

    def _clean_prompt(self, prompt: str) -> str:
        """Clean prompt text into a clean tag."""
        # Remove common suffixes
        for suffix in [" photo", " photography", " shot", " mood", " atmosphere",
                       " weather", " season", " camera", " quality"]:
            prompt = prompt.replace(suffix, "")

        # Remove "photo of a" prefix
        prompt = prompt.replace("photo of a ", "")
        prompt = prompt.replace("photo of an ", "")
        prompt = prompt.replace("photo of ", "")

        return prompt.strip()

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
