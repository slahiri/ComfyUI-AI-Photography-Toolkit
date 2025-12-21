"""SegFormer clothes tagger - semantic segmentation for clothing."""

import numpy as np
from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult
from ..utils import download_hf_model
from .fashion_labels import SEGFORMER_CLOTHES_LABELS, SEGFORMER_TAG_MAP


class SegFormerClothesTagger(BaseTagger):
    """
    Semantic segmentation for clothing using SegFormer.

    Segments image into clothing regions and generates tags
    based on the area covered by each clothing type.

    Requires ~400MB VRAM, ~60ms inference.
    """

    TAGGER_NAME = "segformer_clothes"
    MODEL_ID = "mattmdjaga/segformer_b2_clothes"

    def __init__(
        self,
        min_area_percent: float = 0.03,
        **kwargs
    ):
        """
        Initialize SegFormer clothes tagger.

        Args:
            min_area_percent: Minimum area percentage to generate a tag (0-1)
        """
        super().__init__(**kwargs)
        self.min_area_percent = min_area_percent
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load SegFormer model."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required for SegFormer. "
                "Install with: pip install transformers torch"
            )

        # Download model
        model_path = download_hf_model(self.MODEL_ID, "fashion")

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self._processor = SegformerImageProcessor.from_pretrained(model_path)
        self._model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Segment clothing and generate tags based on coverage.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with clothing segmentation tags
        """
        if not self._is_loaded:
            self.load()

        import torch
        import torch.nn.functional as F

        # Convert to RGB
        rgb_image = image.convert("RGB")
        original_size = rgb_image.size  # (width, height)

        # Process image
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Upsample to original size
        upsampled = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # (height, width)
            mode="bilinear",
            align_corners=False,
        )

        # Get segmentation mask
        seg_mask = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        # Calculate area percentages for each class
        total_pixels = seg_mask.size
        tags = []

        for class_id in range(len(SEGFORMER_CLOTHES_LABELS)):
            if class_id == 0:  # Skip background
                continue

            # Count pixels for this class
            class_pixels = np.sum(seg_mask == class_id)
            area_percent = class_pixels / total_pixels

            if area_percent >= self.min_area_percent:
                # Get tag text
                tag_text = SEGFORMER_TAG_MAP.get(class_id)
                if tag_text:
                    # Use area percentage as confidence proxy
                    confidence = min(0.95, 0.5 + area_percent * 2)

                    tags.append(TagItem(
                        text=tag_text,
                        confidence=confidence,
                        category=f"fashion:{SEGFORMER_CLOTHES_LABELS[class_id]}"
                    ))

        # Sort by confidence (area)
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "segformer-clothes",
                "num_segments": len(tags),
                "min_area_percent": self.min_area_percent,
            }
        )

    def unload(self) -> None:
        """Release SegFormer model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._device = None
        super().unload()
