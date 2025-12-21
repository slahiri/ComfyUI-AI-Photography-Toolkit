"""FashionCLIP tagger - zero-shot fashion classification."""

from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult
from ..utils import download_hf_model, get_all_fashion_vocabulary


class FashionCLIPTagger(BaseTagger):
    """
    Zero-shot fashion classification using FashionCLIP.

    Uses CLIP fine-tuned on fashion data to classify images
    against a predefined vocabulary of fashion terms.

    Requires ~400MB VRAM, ~50ms inference.
    """

    TAGGER_NAME = "fashion_clip"
    MODEL_ID = "patrickjohncyh/fashion-clip"

    def __init__(
        self,
        threshold: float = 0.3,
        top_k: int = 5,
        vocabulary: list[str] | None = None,
        **kwargs
    ):
        """
        Initialize FashionCLIP tagger.

        Args:
            threshold: Minimum confidence threshold for tags
            top_k: Maximum number of tags to return
            vocabulary: Custom vocabulary (loads from config if None)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.top_k = top_k
        # Load vocabulary from config file
        self.vocabulary = vocabulary or get_all_fashion_vocabulary()
        self._model = None
        self._processor = None
        self._device = None
        self._text_features = None

    def load(self) -> None:
        """Load FashionCLIP model and precompute text embeddings."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required for FashionCLIP. "
                "Install with: pip install transformers torch"
            )

        # Download model
        model_path = download_hf_model(self.MODEL_ID, "fashion")

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self._processor = CLIPProcessor.from_pretrained(model_path)
        self._model = CLIPModel.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

        # Precompute text embeddings for vocabulary
        self._precompute_text_features()

        self._is_loaded = True

    def _precompute_text_features(self) -> None:
        """Precompute text embeddings for the vocabulary."""
        import torch

        # Prepare text prompts
        prompts = [f"a photo of {item}" for item in self.vocabulary]

        # Tokenize and encode
        text_inputs = self._processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}

        with torch.no_grad():
            self._text_features = self._model.get_text_features(**text_inputs)
            self._text_features = self._text_features / self._text_features.norm(
                dim=-1, keepdim=True
            )

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Classify image against fashion vocabulary.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with fashion classification tags
        """
        if not self._is_loaded:
            self.load()

        import torch

        # Convert to RGB
        rgb_image = image.convert("RGB")

        # Process image
        image_inputs = self._processor(
            images=rgb_image,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(self._device) for k, v in image_inputs.items()}

        # Get image features
        with torch.no_grad():
            image_features = self._model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (image_features @ self._text_features.T).squeeze(0)
            probs = similarity.softmax(dim=-1)

        # Get top-k predictions
        probs_np = probs.cpu().numpy()
        top_indices = probs_np.argsort()[::-1][:self.top_k]

        # Convert to tags
        tags = []
        for idx in top_indices:
            confidence = float(probs_np[idx])
            if confidence >= self.threshold:
                tags.append(TagItem(
                    text=self.vocabulary[idx],
                    confidence=confidence,
                    category="fashion:style"
                ))

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "fashion-clip",
                "num_predictions": len(tags),
                "threshold": self.threshold,
                "vocabulary_size": len(self.vocabulary),
            }
        )

    def unload(self) -> None:
        """Release FashionCLIP model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._text_features is not None:
            del self._text_features
            self._text_features = None
        self._device = None
        super().unload()
