"""RAM++ (Recognize Anything Plus) tagger - 6,400+ category recognition."""

from PIL import Image
import numpy as np

from .base import BaseTagger, TagItem, TaggerResult
from .utils import download_hf_model


class RAMPlusTagger(BaseTagger):
    """
    RAM++ (Recognize Anything Plus Model) tagger.

    State-of-the-art multi-label image tagging with 6,400+ categories
    including objects, scenes, attributes, and actions.

    Best general-purpose tagger with open-set recognition capability.

    Requires ~1.5GB VRAM, ~100ms inference.
    """

    TAGGER_NAME = "ram_plus"
    MODEL_ID = "xinyu1205/recognize-anything-plus-model"

    def __init__(
        self,
        threshold: float = 0.5,
        delete_tag_index: list[int] | None = None,
        **kwargs
    ):
        """
        Initialize RAM++ tagger.

        Args:
            threshold: Minimum confidence threshold for tags
            delete_tag_index: List of tag indices to exclude
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.delete_tag_index = delete_tag_index or []
        self._model = None
        self._transform = None
        self._device = None

    def load(self) -> None:
        """Load RAM++ model."""
        if self._is_loaded:
            return

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for RAM++")

        # Check if recognize-anything is installed
        try:
            from ram.models import ram_plus
            from ram import get_transform
        except ImportError as e:
            print(f"[SID-RAM++] recognize-anything package not found: {e}")
            print("[SID-RAM++] To install, run:")
            print("[SID-RAM++]   pip install git+https://github.com/xinyu1205/recognize-anything.git")
            raise ImportError(
                "recognize-anything package is required for RAM++. "
                "Install with: pip install git+https://github.com/xinyu1205/recognize-anything.git"
            )

        print("[SID-RAM++] Loading RAM++ model...")

        # Download model
        model_path = download_hf_model(self.MODEL_ID, "ram")

        # Check for model file
        model_file = model_path / "ram_plus_swin_large_14m.pth"
        if not model_file.exists():
            raise FileNotFoundError(
                f"RAM++ model file not found at {model_file}. "
                "Please re-download the model."
            )

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        print(f"[SID-RAM++] Loading from {model_file}")
        self._model = ram_plus(
            pretrained=str(model_file),
            image_size=384,
            vit="swin_l"
        )
        self._model.to(self._device)
        self._model.eval()

        # Get transform
        self._transform = get_transform(image_size=384)

        self._is_loaded = True
        print("[SID-RAM++] Model loaded successfully")

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Tag image with RAM++.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with multi-label tags
        """
        if not self._is_loaded:
            self.load()

        import torch

        # Convert to RGB
        rgb_image = image.convert("RGB")

        # Transform image
        image_tensor = self._transform(rgb_image).unsqueeze(0).to(self._device)

        # Inference
        with torch.no_grad():
            tags, tag_scores = self._model.generate_tag(image_tensor)

        # Parse results - tags is a string like "tag1 | tag2 | tag3"
        if isinstance(tags, str):
            tag_list = [t.strip() for t in tags.split("|")]
        else:
            tag_list = tags[0].split("|") if tags else []

        # Convert to TagItems
        result_tags = []
        for i, tag_text in enumerate(tag_list):
            tag_text = tag_text.strip()
            if not tag_text:
                continue

            # Get score if available
            confidence = 0.8  # Default confidence
            if tag_scores is not None and i < len(tag_scores):
                confidence = float(tag_scores[i])

            if confidence >= self.threshold:
                result_tags.append(TagItem(
                    text=tag_text,
                    confidence=confidence,
                    category="ram:general"
                ))

        return TaggerResult(
            tags=result_tags,
            metadata={
                "model": "ram-plus",
                "num_predictions": len(result_tags),
                "threshold": self.threshold,
            }
        )

    def unload(self) -> None:
        """Release RAM++ model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._transform is not None:
            del self._transform
            self._transform = None
        self._device = None
        super().unload()


class RAMPlusTaggerHF(BaseTagger):
    """
    RAM++ tagger using HuggingFace transformers (simpler installation).

    Alternative implementation that uses the transformers library
    for easier installation without the recognize-anything package.
    """

    TAGGER_NAME = "ram_plus_hf"
    MODEL_ID = "xinyu1205/recognize-anything-plus-model"

    def __init__(
        self,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize RAM++ tagger (HF version).

        Args:
            threshold: Minimum confidence threshold for tags
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load RAM++ model via transformers."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required for RAM++. "
                "Install with: pip install transformers torch"
            )

        # Download model
        model_path = download_hf_model(self.MODEL_ID, "ram")

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try loading with AutoModel
        try:
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = AutoModelForZeroShotImageClassification.from_pretrained(model_path)
            self._model.to(self._device)
            self._model.eval()
        except Exception:
            # Fall back to manual loading
            raise ImportError(
                "RAM++ requires the recognize-anything package. "
                "Install with: pip install git+https://github.com/xinyu1205/recognize-anything.git"
            )

        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """Tag image with RAM++ (HF version)."""
        if not self._is_loaded:
            self.load()

        import torch

        # Convert to RGB
        rgb_image = image.convert("RGB")

        # Process image
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Process outputs
        logits = outputs.logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()

        # Get tags above threshold
        result_tags = []
        labels = self._model.config.id2label

        for idx, prob in enumerate(probs):
            if prob >= self.threshold:
                tag_text = labels.get(idx, f"tag_{idx}")
                result_tags.append(TagItem(
                    text=tag_text,
                    confidence=float(prob),
                    category="ram:general"
                ))

        # Sort by confidence
        result_tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=result_tags,
            metadata={
                "model": "ram-plus-hf",
                "num_predictions": len(result_tags),
                "threshold": self.threshold,
            }
        )

    def unload(self) -> None:
        """Release RAM++ model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._device = None
        super().unload()
