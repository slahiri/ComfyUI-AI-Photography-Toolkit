"""DeepFashion-style attribute prediction.

Provides detailed fashion attribute detection including:
- Fabric type (cotton, silk, denim, leather, etc.)
- Pattern (solid, striped, floral, plaid, etc.)
- Texture (smooth, ribbed, quilted, etc.)
- Style attributes (casual, formal, sporty, etc.)
- Color attributes (primary, secondary, accent)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..base import BaseTagger, TagItem, TaggerResult


@dataclass
class FashionAttribute:
    """Fashion attribute with category."""
    name: str
    category: str  # fabric, pattern, texture, style, color, fit, neckline, sleeve, etc.
    confidence: float


class DeepFashionAttributeTagger(BaseTagger):
    """
    Fashion attribute prediction using CLIP zero-shot classification.

    Uses extensive vocabulary covering:
    - 20+ fabric types
    - 25+ patterns
    - 15+ textures
    - 30+ style attributes
    - Fit, neckline, sleeve types, etc.

    Based on DeepFashion attribute categories but using CLIP for flexibility.
    """

    TAGGER_NAME = "deepfashion_attributes"

    # Comprehensive attribute vocabularies
    FABRIC_TYPES = [
        "cotton fabric", "silk fabric", "satin fabric", "velvet fabric",
        "denim fabric", "leather material", "suede material", "lace fabric",
        "chiffon fabric", "linen fabric", "wool fabric", "cashmere fabric",
        "polyester fabric", "nylon fabric", "mesh fabric", "tulle fabric",
        "tweed fabric", "corduroy fabric", "jersey fabric", "fleece fabric",
        "organza fabric", "taffeta fabric", "crepe fabric", "jacquard fabric",
    ]

    PATTERNS = [
        "solid color", "striped pattern", "horizontal stripes", "vertical stripes",
        "plaid pattern", "checkered pattern", "gingham pattern", "tartan pattern",
        "floral pattern", "floral print", "botanical print", "tropical print",
        "polka dots", "geometric pattern", "abstract pattern", "paisley pattern",
        "animal print", "leopard print", "zebra print", "snake print",
        "camouflage pattern", "tie-dye pattern", "ombre pattern", "color block",
        "houndstooth pattern", "herringbone pattern", "argyle pattern",
        "toile pattern", "damask pattern", "ikat pattern", "batik pattern",
    ]

    TEXTURES = [
        "smooth texture", "ribbed texture", "quilted texture", "pleated texture",
        "ruched texture", "gathered texture", "smocked texture", "embroidered texture",
        "sequined texture", "beaded texture", "fringed texture", "ruffled texture",
        "crinkled texture", "distressed texture", "washed texture", "brushed texture",
        "matte finish", "shiny finish", "metallic finish", "glossy finish",
    ]

    STYLE_ATTRIBUTES = [
        "casual style", "formal style", "business style", "sporty style",
        "elegant style", "bohemian style", "vintage style", "retro style",
        "minimalist style", "maximalist style", "classic style", "modern style",
        "romantic style", "edgy style", "preppy style", "streetwear style",
        "athleisure style", "resort wear", "evening wear", "cocktail attire",
        "loungewear style", "workwear style", "party wear", "bridal style",
        "modest fashion", "avant-garde style", "grunge style", "punk style",
    ]

    FIT_TYPES = [
        "fitted silhouette", "loose fit", "oversized fit", "slim fit",
        "relaxed fit", "tailored fit", "bodycon fit", "A-line silhouette",
        "flared silhouette", "straight cut", "boxy fit", "draped silhouette",
    ]

    NECKLINES = [
        "round neckline", "V-neckline", "scoop neckline", "square neckline",
        "boat neckline", "off-shoulder neckline", "one-shoulder", "halter neckline",
        "sweetheart neckline", "cowl neckline", "turtleneck", "mock neck",
        "collar neckline", "mandarin collar", "plunging neckline", "keyhole neckline",
        "strapless", "crew neck", "henley neckline",
    ]

    SLEEVE_TYPES = [
        "sleeveless", "short sleeves", "cap sleeves", "three-quarter sleeves",
        "long sleeves", "bell sleeves", "puff sleeves", "balloon sleeves",
        "bishop sleeves", "dolman sleeves", "raglan sleeves", "flutter sleeves",
        "cold shoulder sleeves", "kimono sleeves", "lantern sleeves",
    ]

    LENGTH_TYPES = [
        "crop length", "cropped", "waist length", "hip length",
        "tunic length", "knee length", "midi length", "maxi length",
        "floor length", "mini length", "micro length", "tea length",
    ]

    COLORS = [
        "black color", "white color", "red color", "blue color", "navy blue",
        "green color", "emerald green", "olive green", "yellow color", "gold color",
        "orange color", "pink color", "blush pink", "hot pink", "purple color",
        "lavender color", "burgundy color", "maroon color", "brown color", "tan color",
        "beige color", "cream color", "ivory color", "gray color", "charcoal gray",
        "silver color", "coral color", "teal color", "turquoise color", "mint green",
        "peach color", "rose color", "champagne color", "nude color",
        "multicolored", "two-tone", "color blocked",
    ]

    def __init__(
        self,
        threshold: float = 0.15,
        top_k_per_category: int = 3,
        model_name: str = "patrickjohncyh/fashion-clip",
        **kwargs
    ):
        """
        Initialize DeepFashion attribute tagger.

        Args:
            threshold: Minimum confidence to include attribute
            top_k_per_category: Max attributes per category
            model_name: CLIP model to use (fashion-clip recommended)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.top_k_per_category = top_k_per_category
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        self._text_embeddings = {}

    def load(self) -> None:
        """Load FashionCLIP model and pre-compute text embeddings."""
        if self._is_loaded:
            return

        from transformers import CLIPProcessor, CLIPModel

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SID-DeepFashion] Loading {self.model_name}...")
        try:
            self._model = CLIPModel.from_pretrained(self.model_name)
        except Exception:
            # Fallback to base CLIP if fashion-clip not available
            print("[SID-DeepFashion] Fashion-CLIP not found, using base CLIP")
            self.model_name = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(self.model_name)

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        # Pre-compute text embeddings for all attributes
        self._precompute_embeddings()

        self._is_loaded = True
        print("[SID-DeepFashion] Loaded with pre-computed embeddings")

    def _precompute_embeddings(self) -> None:
        """Pre-compute text embeddings for all attribute vocabularies."""
        categories = {
            "fabric": self.FABRIC_TYPES,
            "pattern": self.PATTERNS,
            "texture": self.TEXTURES,
            "style": self.STYLE_ATTRIBUTES,
            "fit": self.FIT_TYPES,
            "neckline": self.NECKLINES,
            "sleeve": self.SLEEVE_TYPES,
            "length": self.LENGTH_TYPES,
            "color": self.COLORS,
        }

        for category, labels in categories.items():
            # Add context for better matching
            prompts = [f"clothing with {label}" for label in labels]

            inputs = self._processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)

            self._text_embeddings[category] = {
                "labels": labels,
                "embeddings": text_features
            }

    def _classify_category(
        self,
        image_features: torch.Tensor,
        category: str
    ) -> List[Tuple[str, float]]:
        """Classify image against a category's vocabulary."""
        if category not in self._text_embeddings:
            return []

        text_data = self._text_embeddings[category]
        text_embeddings = text_data["embeddings"]
        labels = text_data["labels"]

        # Compute similarities
        similarities = (image_features @ text_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=-1)  # Temperature scaling

        # Get top-k
        top_k = min(self.top_k_per_category, len(labels))
        top_probs, top_indices = probs.topk(top_k)

        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            if prob >= self.threshold:
                # Clean up label
                label = labels[idx]
                # Remove common suffixes for cleaner output
                for suffix in [" fabric", " material", " pattern", " texture",
                             " style", " color", " neckline", " silhouette"]:
                    if label.endswith(suffix) and not label.startswith(suffix.strip()):
                        label = label.replace(suffix, "")
                results.append((label, float(prob)))

        return results

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Predict fashion attributes for the image.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with detailed fashion attributes
        """
        if not self._is_loaded:
            self.load()

        tags = []
        metadata = {"model": self.model_name}
        attributes_by_category = {}

        try:
            # Process image
            rgb_image = image.convert("RGB")
            inputs = self._processor(images=rgb_image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Get image features
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                image_features = F.normalize(image_features, dim=-1)

            # Classify each category
            categories = ["fabric", "pattern", "texture", "style", "fit",
                         "neckline", "sleeve", "length", "color"]

            for category in categories:
                results = self._classify_category(image_features, category)

                if results:
                    attributes_by_category[category] = results

                    for label, confidence in results:
                        tags.append(TagItem(
                            text=label,
                            confidence=confidence,
                            category=f"fashion:{category}"
                        ))

            # Store summary in metadata
            metadata["attributes_detected"] = {
                cat: len(attrs) for cat, attrs in attributes_by_category.items()
            }
            metadata["total_attributes"] = len(tags)

            # Primary attributes
            if attributes_by_category:
                for cat in ["style", "pattern", "fabric"]:
                    if cat in attributes_by_category and attributes_by_category[cat]:
                        metadata[f"primary_{cat}"] = attributes_by_category[cat][0][0]

        except Exception as e:
            print(f"[SID-DeepFashion] Error: {e}")
            import traceback
            traceback.print_exc()
            metadata["error"] = str(e)

        return TaggerResult(
            tags=tags,
            metadata=metadata
        )

    def unload(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._text_embeddings.clear()
        self._device = None
        super().unload()


class FashionColorAnalyzer(BaseTagger):
    """
    Dedicated fashion color analysis.

    Extracts:
    - Primary color
    - Secondary colors
    - Color harmony type
    - Color temperature (warm/cool)
    """

    TAGGER_NAME = "fashion_color"

    # Color names mapping to RGB ranges
    COLOR_MAP = {
        "black": [(0, 0, 0), (40, 40, 40)],
        "white": [(230, 230, 230), (255, 255, 255)],
        "gray": [(80, 80, 80), (180, 180, 180)],
        "red": [(150, 0, 0), (255, 100, 100)],
        "burgundy": [(100, 0, 30), (150, 50, 80)],
        "pink": [(200, 100, 150), (255, 200, 230)],
        "orange": [(200, 100, 0), (255, 180, 100)],
        "yellow": [(200, 180, 0), (255, 255, 150)],
        "green": [(0, 100, 0), (150, 255, 150)],
        "olive": [(80, 100, 0), (150, 150, 80)],
        "teal": [(0, 100, 100), (100, 200, 200)],
        "blue": [(0, 0, 150), (100, 150, 255)],
        "navy": [(0, 0, 80), (50, 50, 150)],
        "purple": [(80, 0, 150), (180, 100, 255)],
        "brown": [(100, 60, 20), (180, 120, 80)],
        "beige": [(180, 160, 130), (240, 220, 190)],
        "cream": [(240, 230, 200), (255, 250, 240)],
    }

    def __init__(
        self,
        threshold: float = 0.1,
        num_colors: int = 5,
        min_percentage: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.num_colors = num_colors
        self.min_percentage = min_percentage
        self._is_loaded = True

    def load(self) -> None:
        self._is_loaded = True

    def _extract_dominant_colors(
        self,
        image: Image.Image,
        n_colors: int = 5
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """Extract dominant colors using k-means clustering."""
        import cv2
        from sklearn.cluster import KMeans

        # Resize for faster processing
        img = image.convert("RGB").resize((150, 150))
        pixels = np.array(img).reshape(-1, 3).astype(np.float32)

        # Remove very dark and very light pixels (likely background)
        brightness = pixels.mean(axis=1)
        mask = (brightness > 20) & (brightness < 245)
        filtered_pixels = pixels[mask]

        if len(filtered_pixels) < n_colors:
            filtered_pixels = pixels

        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)

        # Get cluster centers and their proportions
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        proportions = np.bincount(labels) / len(labels)

        # Sort by proportion
        sorted_idx = np.argsort(proportions)[::-1]

        return [(tuple(colors[i]), proportions[i]) for i in sorted_idx]

    def _name_color(self, rgb: Tuple[int, int, int]) -> str:
        """Find closest named color."""
        r, g, b = rgb
        min_dist = float('inf')
        best_name = "unknown"

        for name, (low, high) in self.COLOR_MAP.items():
            # Check if in range
            if (low[0] <= r <= high[0] and
                low[1] <= g <= high[1] and
                low[2] <= b <= high[2]):
                return name

            # Calculate distance to range center
            center = ((low[0] + high[0]) / 2,
                     (low[1] + high[1]) / 2,
                     (low[2] + high[2]) / 2)
            dist = np.sqrt((r - center[0])**2 +
                          (g - center[1])**2 +
                          (b - center[2])**2)

            if dist < min_dist:
                min_dist = dist
                best_name = name

        return best_name

    def _analyze_temperature(self, colors: List[Tuple]) -> str:
        """Determine if colors are warm or cool."""
        warm_score = 0
        cool_score = 0

        for (r, g, b), proportion in colors[:3]:
            # Warm colors: red, orange, yellow
            # Cool colors: blue, green, purple
            if r > b:
                warm_score += proportion
            else:
                cool_score += proportion

        if warm_score > cool_score * 1.3:
            return "warm tones"
        elif cool_score > warm_score * 1.3:
            return "cool tones"
        else:
            return "neutral tones"

    def tag(self, image: Image.Image) -> TaggerResult:
        """Analyze fashion colors in image."""
        tags = []
        metadata = {}

        try:
            # Extract dominant colors
            colors = self._extract_dominant_colors(image, n_colors=self.num_colors)

            # Name the colors
            named_colors = []
            for rgb, proportion in colors:
                name = self._name_color(rgb)
                if proportion >= self.min_percentage:
                    named_colors.append((name, proportion))

            # Add tags for colors
            if named_colors:
                primary = named_colors[0]
                tags.append(TagItem(
                    text=f"{primary[0]} primary color",
                    confidence=primary[1],
                    category="fashion:color"
                ))
                metadata["primary_color"] = primary[0]

                for name, prop in named_colors[1:3]:
                    tags.append(TagItem(
                        text=f"{name} accent",
                        confidence=prop,
                        category="fashion:color"
                    ))

            # Color temperature
            temp = self._analyze_temperature(colors)
            tags.append(TagItem(
                text=temp,
                confidence=0.7,
                category="fashion:color"
            ))
            metadata["color_temperature"] = temp

            # Monochrome check
            unique_names = set(name for name, _ in named_colors)
            if len(unique_names) == 1:
                tags.append(TagItem(
                    text="monochrome",
                    confidence=0.8,
                    category="fashion:color"
                ))
            elif len(unique_names) == 2:
                tags.append(TagItem(
                    text="two-tone",
                    confidence=0.7,
                    category="fashion:color"
                ))
            elif len(unique_names) >= 4:
                tags.append(TagItem(
                    text="multicolored",
                    confidence=0.7,
                    category="fashion:color"
                ))

            metadata["colors_detected"] = [name for name, _ in named_colors]

        except Exception as e:
            print(f"[SID-FashionColor] Error: {e}")
            metadata["error"] = str(e)

        return TaggerResult(tags=tags, metadata=metadata)

    def unload(self) -> None:
        pass
