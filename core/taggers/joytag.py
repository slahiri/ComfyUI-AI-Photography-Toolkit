"""JoyTag tagger - Photo-friendly Danbooru-style tags (5000+ tags)."""

from PIL import Image
import numpy as np
from pathlib import Path

from .base import BaseTagger, TagItem, TaggerResult
from .utils import download_hf_model


class JoyTagTagger(BaseTagger):
    """
    JoyTag tagger for photo-friendly Danbooru-style tags.

    Unlike WD14 which is trained primarily on anime/illustrations,
    JoyTag is trained on a mix of photos and artwork (5000+ tags),
    producing more relevant tags for photographs while maintaining
    Danbooru tag vocabulary compatibility.

    Model: fancyfeast/joytag
    Requires ~500MB VRAM, ~40ms inference.
    """

    TAGGER_NAME = "joytag"
    MODEL_ID = "fancyfeast/joytag"
    IMAGE_SIZE = 448  # From config.json

    def __init__(
        self,
        threshold: float = 0.4,
        max_tags: int = 50,
        **kwargs
    ):
        """
        Initialize JoyTag tagger.

        Args:
            threshold: Minimum confidence threshold for tags
            max_tags: Maximum number of tags to return
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_tags = max_tags
        self._session = None
        self._tags = None

    def load(self) -> None:
        """Load JoyTag ONNX model."""
        if self._is_loaded:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for JoyTag. "
                "Install with: pip install onnxruntime-gpu"
            )

        print("[SID-JoyTag] Loading JoyTag model...")

        # Download model
        model_path = download_hf_model(self.MODEL_ID, "joytag")

        # Load tag list
        tags_file = model_path / "top_tags.txt"
        if tags_file.exists():
            with open(tags_file, "r", encoding="utf-8") as f:
                self._tags = [line.strip() for line in f if line.strip()]
            print(f"[SID-JoyTag] Loaded {len(self._tags)} tags")
        else:
            raise FileNotFoundError(f"top_tags.txt not found at {tags_file}")

        # Load ONNX model
        onnx_path = model_path / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"model.onnx not found at {onnx_path}")

        # Create ONNX session with GPU if available
        providers = []
        if ort.get_device() == "GPU":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=providers
        )

        self._is_loaded = True
        print(f"[SID-JoyTag] Model loaded successfully (ONNX, {self.IMAGE_SIZE}x{self.IMAGE_SIZE})")

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """Prepare image for ONNX inference."""
        # Convert to RGB
        rgb_image = image.convert("RGB")

        # Resize to model's expected size
        rgb_image = rgb_image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(rgb_image, dtype=np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_array = (img_array - mean) / std

        # Transpose to CHW format and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Tag image with JoyTag.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with Danbooru-style tags
        """
        if not self._is_loaded:
            self.load()

        # Prepare image
        img_array = self._prepare_image(image)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name

        outputs = self._session.run([output_name], {input_name: img_array})
        logits = outputs[0][0]

        # Apply sigmoid for multi-label probabilities
        probs = 1 / (1 + np.exp(-logits))

        # Get tags above threshold
        result_tags = []

        if self._tags and len(self._tags) == len(probs):
            # Sort by confidence
            indices = np.argsort(probs)[::-1]

            for idx in indices[:self.max_tags]:
                prob = probs[idx]
                if prob >= self.threshold:
                    tag_text = self._tags[idx].replace("_", " ")
                    result_tags.append(TagItem(
                        text=tag_text,
                        confidence=float(prob),
                        category="general"
                    ))
        else:
            print(f"[SID-JoyTag] Warning: Tag count mismatch ({len(self._tags)} tags, {len(probs)} probs)")
            # Fallback: use indices
            indices = np.argsort(probs)[::-1]
            for idx in indices[:self.max_tags]:
                if probs[idx] >= self.threshold:
                    result_tags.append(TagItem(
                        text=f"tag_{idx}",
                        confidence=float(probs[idx]),
                        category="general"
                    ))

        return TaggerResult(
            tags=result_tags,
            metadata={
                "model": "joytag-onnx",
                "num_predictions": len(result_tags),
                "threshold": self.threshold,
                "total_tags": len(self._tags) if self._tags else 0,
                "image_size": self.IMAGE_SIZE,
            }
        )

    def unload(self) -> None:
        """Release JoyTag model."""
        if self._session is not None:
            del self._session
            self._session = None
        self._tags = None
        super().unload()
