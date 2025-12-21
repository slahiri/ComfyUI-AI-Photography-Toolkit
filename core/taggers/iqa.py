"""Image Quality Assessment tagger using IQA-PyTorch."""

from PIL import Image
import numpy as np

from .base import BaseTagger, TagItem, TaggerResult


class IQATagger(BaseTagger):
    """
    Image Quality Assessment using IQA-PyTorch (pyiqa).

    Provides multiple quality metrics:
    - NIMA: Aesthetic quality score (1-10)
    - MUSIQ: Technical quality (MOS scale)
    - BRISQUE: Blur/noise detection (lower = better)
    - CLIPIQA: CLIP-based quality assessment

    Install: pip install pyiqa
    """

    TAGGER_NAME = "iqa"

    # Score thresholds for generating tags
    AESTHETIC_THRESHOLDS = {
        "exceptional": 7.5,
        "excellent": 6.5,
        "good": 5.5,
        "average": 4.5,
        "below average": 3.5,
        "poor": 0,
    }

    TECHNICAL_THRESHOLDS = {
        "excellent technical quality": 75,
        "good technical quality": 60,
        "acceptable quality": 45,
        "low quality": 30,
        "poor quality": 0,
    }

    BRISQUE_THRESHOLDS = {
        "sharp and clean": 20,
        "good clarity": 35,
        "slight blur or noise": 50,
        "noticeable blur or noise": 70,
        "significant quality issues": 100,
    }

    def __init__(
        self,
        use_nima: bool = True,
        use_musiq: bool = True,
        use_brisque: bool = True,
        use_clipiqa: bool = False,
        **kwargs
    ):
        """
        Initialize IQA tagger.

        Args:
            use_nima: Enable NIMA aesthetic scoring
            use_musiq: Enable MUSIQ technical quality
            use_brisque: Enable BRISQUE blur/noise detection
            use_clipiqa: Enable CLIP-based IQA (slower)
        """
        super().__init__(**kwargs)
        self.use_nima = use_nima
        self.use_musiq = use_musiq
        self.use_brisque = use_brisque
        self.use_clipiqa = use_clipiqa

        self._models = {}
        self._device = None

    def load(self) -> None:
        """Load IQA models."""
        if self._is_loaded:
            return

        try:
            import pyiqa
            import torch
        except ImportError:
            raise ImportError(
                "pyiqa is required for IQA tagger. "
                "Install with: pip install pyiqa"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[SID-IQA] Loading IQA models...")

        # Load requested models
        if self.use_nima:
            print("[SID-IQA]   Loading NIMA (aesthetic)...")
            self._models["nima"] = pyiqa.create_metric("nima", device=self._device)

        if self.use_musiq:
            print("[SID-IQA]   Loading MUSIQ (technical)...")
            self._models["musiq"] = pyiqa.create_metric("musiq", device=self._device)

        if self.use_brisque:
            print("[SID-IQA]   Loading BRISQUE (blur/noise)...")
            self._models["brisque"] = pyiqa.create_metric("brisque", device=self._device)

        if self.use_clipiqa:
            print("[SID-IQA]   Loading CLIPIQA...")
            self._models["clipiqa"] = pyiqa.create_metric("clipiqa", device=self._device)

        print(f"[SID-IQA] Loaded {len(self._models)} models")
        self._is_loaded = True

    def _score_to_tag(self, score: float, thresholds: dict, prefix: str = "") -> tuple[str, float]:
        """Convert numeric score to descriptive tag."""
        for label, threshold in thresholds.items():
            if score >= threshold:
                tag = f"{prefix}{label}" if prefix else label
                # Normalize confidence based on position in range
                confidence = min(0.95, 0.5 + (score / 100) * 0.45)
                return tag, confidence
        return None, 0

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Analyze image quality.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with quality assessment tags
        """
        if not self._is_loaded:
            self.load()

        import torch
        from torchvision import transforms

        # Convert to RGB tensor
        rgb_image = image.convert("RGB")

        # Prepare tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(rgb_image).unsqueeze(0).to(self._device)

        tags = []
        metadata = {}

        # NIMA - Aesthetic quality (1-10 scale)
        if "nima" in self._models:
            with torch.no_grad():
                score = self._models["nima"](img_tensor).item()

            metadata["nima_score"] = round(score, 2)

            # Generate aesthetic tags
            if score >= 7.5:
                tags.append(TagItem(text="exceptional aesthetic quality", confidence=0.9, category="iqa:aesthetic"))
                tags.append(TagItem(text="visually stunning", confidence=0.85, category="iqa:aesthetic"))
            elif score >= 6.5:
                tags.append(TagItem(text="excellent aesthetic quality", confidence=0.85, category="iqa:aesthetic"))
                tags.append(TagItem(text="pleasing composition", confidence=0.8, category="iqa:aesthetic"))
            elif score >= 5.5:
                tags.append(TagItem(text="good aesthetic quality", confidence=0.8, category="iqa:aesthetic"))
            elif score >= 4.5:
                tags.append(TagItem(text="average aesthetic quality", confidence=0.7, category="iqa:aesthetic"))
            else:
                tags.append(TagItem(text="below average aesthetic", confidence=0.7, category="iqa:aesthetic"))

        # MUSIQ - Technical quality (0-100 scale typical)
        if "musiq" in self._models:
            with torch.no_grad():
                score = self._models["musiq"](img_tensor).item()

            # MUSIQ outputs vary, normalize to 0-100
            score = min(100, max(0, score))
            metadata["musiq_score"] = round(score, 2)

            if score >= 75:
                tags.append(TagItem(text="excellent technical quality", confidence=0.9, category="iqa:technical"))
                tags.append(TagItem(text="professional quality", confidence=0.85, category="iqa:technical"))
            elif score >= 60:
                tags.append(TagItem(text="good technical quality", confidence=0.8, category="iqa:technical"))
            elif score >= 45:
                tags.append(TagItem(text="acceptable technical quality", confidence=0.7, category="iqa:technical"))
            else:
                tags.append(TagItem(text="low technical quality", confidence=0.7, category="iqa:technical"))

        # BRISQUE - No-reference quality (lower = better, 0-100 typical)
        if "brisque" in self._models:
            with torch.no_grad():
                score = self._models["brisque"](img_tensor).item()

            metadata["brisque_score"] = round(score, 2)

            # BRISQUE: lower is better
            if score <= 20:
                tags.append(TagItem(text="sharp and clean", confidence=0.9, category="iqa:clarity"))
                tags.append(TagItem(text="excellent clarity", confidence=0.85, category="iqa:clarity"))
            elif score <= 35:
                tags.append(TagItem(text="good clarity", confidence=0.8, category="iqa:clarity"))
                tags.append(TagItem(text="minimal noise", confidence=0.75, category="iqa:clarity"))
            elif score <= 50:
                tags.append(TagItem(text="slight blur or noise", confidence=0.7, category="iqa:clarity"))
            elif score <= 70:
                tags.append(TagItem(text="noticeable blur", confidence=0.75, category="iqa:clarity"))
                tags.append(TagItem(text="visible noise", confidence=0.7, category="iqa:clarity"))
            else:
                tags.append(TagItem(text="significant blur", confidence=0.8, category="iqa:clarity"))
                tags.append(TagItem(text="heavy noise", confidence=0.75, category="iqa:clarity"))

        # CLIPIQA - CLIP-based quality
        if "clipiqa" in self._models:
            with torch.no_grad():
                score = self._models["clipiqa"](img_tensor).item()

            metadata["clipiqa_score"] = round(score, 3)

            # CLIPIQA is 0-1 scale
            if score >= 0.7:
                tags.append(TagItem(text="high perceived quality", confidence=0.85, category="iqa:perceived"))
            elif score >= 0.5:
                tags.append(TagItem(text="good perceived quality", confidence=0.75, category="iqa:perceived"))
            else:
                tags.append(TagItem(text="low perceived quality", confidence=0.7, category="iqa:perceived"))

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "iqa-pytorch",
                "models_used": list(self._models.keys()),
                **metadata,
            }
        )

    def unload(self) -> None:
        """Release IQA models."""
        for name, model in self._models.items():
            del model
        self._models = {}
        self._device = None
        super().unload()
