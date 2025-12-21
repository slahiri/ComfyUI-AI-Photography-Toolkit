"""Composition analysis taggers - CADB and element detection."""

from PIL import Image
import numpy as np

from .base import BaseTagger, TagItem, TaggerResult
from .utils import download_hf_model


class CADBCompositionTagger(BaseTagger):
    """
    CADB Composition Classifier - 13-class composition detection.

    Detects composition types:
    - Rule of thirds
    - Center composition
    - Symmetry (horizontal/vertical)
    - Diagonal composition
    - Triangle composition
    - Pattern/texture
    - Frame within frame
    - Leading lines
    - S-curve
    - Golden ratio
    - Minimalist
    - Depth/layers
    - Fill frame

    Also detects composition elements like symmetry axes,
    leading line directions, etc.
    """

    TAGGER_NAME = "cadb_composition"
    MODEL_ID = "shuntagami/cadb-composition-classifier"

    # 13 composition classes
    COMPOSITION_CLASSES = [
        "rule of thirds",
        "center composition",
        "horizontal symmetry",
        "vertical symmetry",
        "diagonal composition",
        "triangle composition",
        "pattern composition",
        "frame within frame",
        "leading lines",
        "s-curve composition",
        "golden ratio",
        "minimalist composition",
        "layered depth",
    ]

    def __init__(
        self,
        threshold: float = 0.3,
        detect_elements: bool = True,
        **kwargs
    ):
        """
        Initialize CADB composition tagger.

        Args:
            threshold: Minimum confidence for composition tags
            detect_elements: Also detect composition elements
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.detect_elements = detect_elements
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load CADB model."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForImageClassification, AutoImageProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[SID-Composition] Loading CADB composition classifier...")

        # Try loading from HuggingFace
        try:
            model_path = download_hf_model(self.MODEL_ID, "composition")
            self._processor = AutoImageProcessor.from_pretrained(model_path)
            self._model = AutoModelForImageClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"[SID-Composition] CADB model not available: {e}")
            print("[SID-Composition] Using fallback composition analysis")
            self._model = None
            self._processor = None

        if self._model is not None:
            self._model.to(self._device)
            self._model.eval()

        self._is_loaded = True

    def _analyze_composition_elements(self, image: Image.Image) -> list[TagItem]:
        """Analyze composition elements using CV techniques."""
        import cv2

        tags = []
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Edge detection for line analysis
        edges = cv2.Canny(gray, 50, 150)

        # Hough line detection for leading lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=min(h, w)//4, maxLineGap=20)

        if lines is not None and len(lines) > 0:
            # Analyze line directions
            horizontal_lines = 0
            vertical_lines = 0
            diagonal_lines = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angle = abs(angle)

                if angle < 15 or angle > 165:
                    horizontal_lines += 1
                elif 75 < angle < 105:
                    vertical_lines += 1
                else:
                    diagonal_lines += 1

            total_lines = len(lines)

            if horizontal_lines > total_lines * 0.4:
                tags.append(TagItem(
                    text="horizontal lines",
                    confidence=0.7,
                    category="composition:elements"
                ))
            if vertical_lines > total_lines * 0.4:
                tags.append(TagItem(
                    text="vertical lines",
                    confidence=0.7,
                    category="composition:elements"
                ))
            if diagonal_lines > total_lines * 0.3:
                tags.append(TagItem(
                    text="diagonal lines",
                    confidence=0.7,
                    category="composition:elements"
                ))

            if total_lines > 5:
                tags.append(TagItem(
                    text="strong linear elements",
                    confidence=0.75,
                    category="composition:elements"
                ))

        # Symmetry detection
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)

        # Resize to match if needed
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]

        # Calculate symmetry score
        symmetry_diff = np.abs(left_half.astype(float) - right_half.astype(float))
        symmetry_score = 1 - (np.mean(symmetry_diff) / 255)

        if symmetry_score > 0.85:
            tags.append(TagItem(
                text="strong vertical symmetry",
                confidence=0.85,
                category="composition:symmetry"
            ))
        elif symmetry_score > 0.75:
            tags.append(TagItem(
                text="partial symmetry",
                confidence=0.7,
                category="composition:symmetry"
            ))

        # Horizontal symmetry
        top_half = gray[:h//2, :]
        bottom_half = cv2.flip(gray[h//2:, :], 0)

        min_h = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_h, :]
        bottom_half = bottom_half[:min_h, :]

        h_symmetry_diff = np.abs(top_half.astype(float) - bottom_half.astype(float))
        h_symmetry_score = 1 - (np.mean(h_symmetry_diff) / 255)

        if h_symmetry_score > 0.85:
            tags.append(TagItem(
                text="horizontal symmetry",
                confidence=0.85,
                category="composition:symmetry"
            ))

        # Rule of thirds analysis
        thirds_x = [w // 3, 2 * w // 3]
        thirds_y = [h // 3, 2 * h // 3]

        # Detect if key elements are near thirds lines
        # Use edge density as proxy for important elements
        edge_density = edges.astype(float) / 255

        # Check thirds intersections
        thirds_regions = [
            edge_density[thirds_y[0]-20:thirds_y[0]+20, thirds_x[0]-20:thirds_x[0]+20],
            edge_density[thirds_y[0]-20:thirds_y[0]+20, thirds_x[1]-20:thirds_x[1]+20],
            edge_density[thirds_y[1]-20:thirds_y[1]+20, thirds_x[0]-20:thirds_x[0]+20],
            edge_density[thirds_y[1]-20:thirds_y[1]+20, thirds_x[1]-20:thirds_x[1]+20],
        ]

        thirds_activity = sum(np.mean(r) for r in thirds_regions if r.size > 0)
        center_region = edge_density[h//3:2*h//3, w//3:2*w//3]
        center_activity = np.mean(center_region) if center_region.size > 0 else 0

        if thirds_activity > center_activity * 1.5:
            tags.append(TagItem(
                text="elements at thirds",
                confidence=0.75,
                category="composition:placement"
            ))

        # Negative space detection
        # Use luminance variance
        brightness = np.mean(gray)
        variance = np.var(gray)

        if variance < 1500:
            tags.append(TagItem(
                text="negative space",
                confidence=0.7,
                category="composition:space"
            ))
            tags.append(TagItem(
                text="minimalist elements",
                confidence=0.65,
                category="composition:style"
            ))

        # Frame within frame detection (dark borders)
        border_size = min(h, w) // 10
        border_regions = [
            gray[:border_size, :],  # top
            gray[-border_size:, :],  # bottom
            gray[:, :border_size],  # left
            gray[:, -border_size:],  # right
        ]
        border_darkness = np.mean([np.mean(r) for r in border_regions])
        center_brightness = np.mean(gray[h//4:3*h//4, w//4:3*w//4])

        if border_darkness < center_brightness * 0.7:
            tags.append(TagItem(
                text="natural framing",
                confidence=0.7,
                category="composition:framing"
            ))

        return tags

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Analyze image composition.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with composition tags
        """
        if not self._is_loaded:
            self.load()

        tags = []
        metadata = {}

        # Use CADB model if available
        if self._model is not None and self._processor is not None:
            import torch

            rgb_image = image.convert("RGB")
            inputs = self._processor(images=rgb_image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            # Get top compositions
            probs_np = probs.cpu().numpy()

            for idx, prob in enumerate(probs_np):
                if prob >= self.threshold and idx < len(self.COMPOSITION_CLASSES):
                    comp_name = self.COMPOSITION_CLASSES[idx]
                    tags.append(TagItem(
                        text=comp_name,
                        confidence=float(prob),
                        category="composition:type"
                    ))

            # Store top prediction
            top_idx = np.argmax(probs_np)
            if top_idx < len(self.COMPOSITION_CLASSES):
                metadata["primary_composition"] = self.COMPOSITION_CLASSES[top_idx]
                metadata["composition_confidence"] = float(probs_np[top_idx])

        # Detect composition elements
        if self.detect_elements:
            element_tags = self._analyze_composition_elements(image)
            tags.extend(element_tags)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "cadb-composition" if self._model else "cv-analysis",
                **metadata,
            }
        )

    def unload(self) -> None:
        """Release model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._device = None
        super().unload()


class CompositionElementsTagger(BaseTagger):
    """
    Lightweight composition elements detector using CV techniques.

    Detects:
    - Line types and directions
    - Symmetry (vertical/horizontal)
    - Rule of thirds alignment
    - Negative space
    - Natural framing
    - Depth indicators
    """

    TAGGER_NAME = "composition_elements"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self) -> None:
        """No model to load - uses OpenCV."""
        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """Analyze composition elements."""
        if not self._is_loaded:
            self.load()

        try:
            import cv2
        except ImportError:
            return TaggerResult(
                tags=[],
                metadata={"error": "OpenCV not available"}
            )

        # Reuse the element detection from CADB
        cadb = CADBCompositionTagger(detect_elements=True)
        cadb._is_loaded = True
        element_tags = cadb._analyze_composition_elements(image)

        return TaggerResult(
            tags=element_tags,
            metadata={"model": "cv-composition-elements"}
        )

    def unload(self) -> None:
        super().unload()
