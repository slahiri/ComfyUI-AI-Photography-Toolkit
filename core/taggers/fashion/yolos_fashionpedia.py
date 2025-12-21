"""YOLOS-Fashionpedia tagger - 27 fashion category detection."""

from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult
from ..utils import download_hf_model
from .fashion_labels import FASHIONPEDIA_TAG_MAP, get_clothing_tag_text


class YOLOSFashionpediaTagger(BaseTagger):
    """
    Fashion detection using YOLOS trained on Fashionpedia.

    Detects 27 fashion categories including clothing, accessories,
    and footwear with bounding boxes and confidence scores.

    Requires ~300MB VRAM, ~30ms inference.
    """

    TAGGER_NAME = "yolos_fashionpedia"
    MODEL_ID = "valentinafeve/yolos-fashionpedia"

    def __init__(
        self,
        threshold: float = 0.4,
        **kwargs
    ):
        """
        Initialize YOLOS-Fashionpedia tagger.

        Args:
            threshold: Minimum confidence threshold for detections
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load YOLOS model."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required for YOLOS-Fashionpedia. "
                "Install with: pip install transformers torch"
            )

        # Download model
        model_path = download_hf_model(self.MODEL_ID, "fashion")

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor and model
        self._processor = AutoImageProcessor.from_pretrained(model_path)
        self._model = AutoModelForObjectDetection.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Detect fashion items and generate tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with fashion detection tags
        """
        if not self._is_loaded:
            self.load()

        import torch

        # Convert to RGB
        rgb_image = image.convert("RGB")

        # Process image
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process detections
        target_sizes = torch.tensor([rgb_image.size[::-1]]).to(self._device)
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )[0]

        # Convert to tags
        tags = []
        detected_items = set()

        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy()
        ):
            label_id = int(label)
            confidence = float(score)

            # Get tag text (returns None for unknown label IDs)
            tag_text = get_clothing_tag_text(label_id, FASHIONPEDIA_TAG_MAP)

            # Skip unknown labels
            if tag_text is None:
                continue

            # Deduplicate - keep highest confidence for each item type
            if tag_text not in detected_items:
                detected_items.add(tag_text)
                tags.append(TagItem(
                    text=tag_text,
                    confidence=confidence,
                    category=f"fashion:{FASHIONPEDIA_TAG_MAP.get(label_id, 'item')}"
                ))

        # Sort by confidence
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "yolos-fashionpedia",
                "num_detections": len(tags),
                "threshold": self.threshold,
            }
        )

    def unload(self) -> None:
        """Release YOLOS model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._device = None
        super().unload()
