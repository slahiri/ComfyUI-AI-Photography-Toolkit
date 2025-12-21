"""YOLOv8n clothing detection tagger - fast and lightweight."""

from pathlib import Path
from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult
from ..utils import download_single_file


class YOLOv8ClothingTagger(BaseTagger):
    """
    Fast clothing detection using YOLOv8n.

    Lightweight model optimized for speed (~15ms inference).
    Requires ultralytics package and ~150MB VRAM.
    """

    TAGGER_NAME = "yolov8_clothing"
    MODEL_ID = "kesimeg/yolov8n-clothing-detection"
    MODEL_FILE = "best.pt"

    def __init__(
        self,
        threshold: float = 0.4,
        **kwargs
    ):
        """
        Initialize YOLOv8 clothing tagger.

        Args:
            threshold: Minimum confidence threshold for detections
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self._model = None

    def load(self) -> None:
        """Load YOLOv8 model."""
        if self._is_loaded:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOv8 clothing detection. "
                "Install with: pip install ultralytics"
            )

        # Download model weights
        model_path = download_single_file(
            self.MODEL_ID,
            self.MODEL_FILE,
            "fashion"
        )

        # Load model
        self._model = YOLO(str(model_path))
        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Detect clothing items and generate tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with clothing detection tags
        """
        if not self._is_loaded:
            self.load()

        # Convert to RGB
        rgb_image = image.convert("RGB")

        # Run inference
        results = self._model.predict(
            rgb_image,
            conf=self.threshold,
            verbose=False,
        )[0]

        # Convert to tags
        tags = []
        detected_items = set()

        for box in results.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names.get(class_id, f"item_{class_id}")

            # Create tag text
            tag_text = f"wearing {class_name}"

            # Deduplicate
            if tag_text not in detected_items:
                detected_items.add(tag_text)
                tags.append(TagItem(
                    text=tag_text,
                    confidence=confidence,
                    category=f"fashion:{class_name}"
                ))

        # Sort by confidence
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "yolov8n-clothing",
                "num_detections": len(tags),
                "threshold": self.threshold,
            }
        )

    def unload(self) -> None:
        """Release YOLOv8 model."""
        if self._model is not None:
            del self._model
            self._model = None
        super().unload()
