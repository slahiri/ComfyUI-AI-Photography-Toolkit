"""Wargon clothing classifier - ViT-based style classification."""

from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult
from ..utils import download_hf_model


class WargonClothingTagger(BaseTagger):
    """
    Clothing style classification using Wargon ViT model.

    Classifies images into clothing style categories like
    casual, formal, sportswear, etc.

    Requires ~350MB VRAM, ~40ms inference.
    """

    TAGGER_NAME = "wargon_clothing"
    MODEL_ID = "wargoninnovation/wargon-clothing-classifier"

    def __init__(
        self,
        threshold: float = 0.3,
        top_k: int = 3,
        **kwargs
    ):
        """
        Initialize Wargon clothing classifier.

        Args:
            threshold: Minimum confidence threshold for tags
            top_k: Maximum number of style tags to return
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.top_k = top_k
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load Wargon classifier model."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForImageClassification, AutoImageProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required for Wargon classifier. "
                "Install with: pip install transformers torch"
            )

        # Download model
        model_path = download_hf_model(self.MODEL_ID, "fashion")

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self._processor = AutoImageProcessor.from_pretrained(model_path)
        self._model = AutoModelForImageClassification.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Classify clothing style and generate tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with style classification tags
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
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Get top-k predictions
        probs_np = probs.cpu().numpy()
        top_indices = probs_np.argsort()[::-1][:self.top_k]

        # Convert to tags
        tags = []
        for idx in top_indices:
            confidence = float(probs_np[idx])
            if confidence >= self.threshold:
                # Get label from model config
                label = self._model.config.id2label.get(int(idx), f"style_{idx}")

                # Format as style tag
                tag_text = f"{label} style"

                tags.append(TagItem(
                    text=tag_text,
                    confidence=confidence,
                    category="fashion:style"
                ))

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "wargon-clothing-classifier",
                "num_predictions": len(tags),
                "threshold": self.threshold,
            }
        )

    def unload(self) -> None:
        """Release Wargon model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._device = None
        super().unload()
