"""WD14 Tagger implementation using ONNX runtime."""

import os
import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import BaseTagger, TagItem, TaggerResult

import folder_paths

# Global verbose flag
_verbose = False


def set_verbose(verbose: bool) -> None:
    """Set verbose logging flag."""
    global _verbose
    _verbose = verbose


def _log(message: str) -> None:
    """Print message if verbose is enabled."""
    if _verbose:
        print(f"[SID-WD14] {message}")


# Available WD14 models
WD14_MODELS = {
    # V3 models (recommended)
    "wd-swinv2-tagger-v3": {
        "repo": "SmilingWolf/wd-swinv2-tagger-v3",
        "description": "SwinV2 V3 - Best accuracy (recommended)",
    },
    "wd-vit-tagger-v3": {
        "repo": "SmilingWolf/wd-vit-tagger-v3",
        "description": "ViT V3 - Good balance",
    },
    "wd-convnext-tagger-v3": {
        "repo": "SmilingWolf/wd-convnext-tagger-v3",
        "description": "ConvNext V3 - Fast inference",
    },
    "wd-eva02-large-tagger-v3": {
        "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
        "description": "EVA02 Large V3 - Highest accuracy (larger)",
    },
    # V2 models (legacy)
    "wd-v1-4-moat-tagger-v2": {
        "repo": "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "description": "MOAT V2 - Legacy, still good",
    },
    "wd-v1-4-swinv2-tagger-v2": {
        "repo": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "description": "SwinV2 V2 - Legacy",
    },
    "wd-v1-4-convnext-tagger-v2": {
        "repo": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "description": "ConvNext V2 - Legacy",
    },
    "wd-v1-4-vit-tagger-v2": {
        "repo": "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "description": "ViT V2 - Legacy",
    },
}

# Default model
DEFAULT_MODEL = "wd-swinv2-tagger-v3"


def get_models_dir() -> Path:
    """Get the directory for WD14 models."""
    # Check if ComfyUI has a dedicated folder for wd14_tagger
    if "wd14_tagger" in folder_paths.folder_names_and_paths:
        return Path(folder_paths.get_folder_paths("wd14_tagger")[0])

    # Fall back to our own models/taggers/wd14 directory
    models_dir = Path(folder_paths.models_dir) / "taggers" / "wd14"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


class WD14Tagger(BaseTagger):
    """
    WD14 Tagger for general-purpose image tagging.

    Uses SmilingWolf's WD14 models with ONNX runtime for fast inference.
    Outputs booru-style tags suitable for image generation prompts.
    """

    TAGGER_NAME = "wd14"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        exclude_tags: str = "",
        **kwargs,
    ):
        """
        Initialize WD14 tagger.

        Args:
            model_name: Which WD14 model variant to use
            general_threshold: Confidence threshold for general tags
            character_threshold: Confidence threshold for character tags
            replace_underscore: Replace underscores with spaces
            trailing_comma: Add trailing comma to each tag
            exclude_tags: Comma-separated list of tags to exclude
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.replace_underscore = replace_underscore
        self.trailing_comma = trailing_comma
        self.exclude_tags = set(
            s.strip().lower() for s in exclude_tags.split(",") if s.strip()
        )

        self._session = None
        self._tags_list = []
        self._general_index = None
        self._character_index = None
        self._input_size = 448  # Default, will be updated on load

    def _get_model_path(self) -> tuple[Path, Path]:
        """Get paths to model and tags files."""
        models_dir = get_models_dir()
        onnx_path = models_dir / f"{self.model_name}.onnx"
        csv_path = models_dir / f"{self.model_name}.csv"
        return onnx_path, csv_path

    def _download_model(self) -> None:
        """Download model from HuggingFace if not present."""
        onnx_path, csv_path = self._get_model_path()

        if onnx_path.exists() and csv_path.exists():
            _log(f"Model already downloaded: {self.model_name}")
            return

        if self.model_name not in WD14_MODELS:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {list(WD14_MODELS.keys())}"
            )

        repo_id = WD14_MODELS[self.model_name]["repo"]
        print(f"[SID-Toolkit] Downloading WD14 model: {self.model_name}...")
        _log(f"Repo: {repo_id}")

        from huggingface_hub import hf_hub_download

        # Download ONNX model
        _log("Downloading model.onnx...")
        hf_hub_download(
            repo_id=repo_id,
            filename="model.onnx",
            local_dir=get_models_dir(),
            local_dir_use_symlinks=False,
        )
        # Rename to model-specific name
        downloaded_onnx = get_models_dir() / "model.onnx"
        if downloaded_onnx.exists():
            downloaded_onnx.rename(onnx_path)

        # Download tags CSV
        _log("Downloading selected_tags.csv...")
        hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            local_dir=get_models_dir(),
            local_dir_use_symlinks=False,
        )
        # Rename to model-specific name
        downloaded_csv = get_models_dir() / "selected_tags.csv"
        if downloaded_csv.exists():
            downloaded_csv.rename(csv_path)

        print(f"[SID-Toolkit] WD14 model downloaded: {self.model_name}")

    def _load_tags(self, csv_path: Path) -> None:
        """Load tags from CSV file."""
        self._tags_list = []
        self._general_index = None
        self._character_index = None

        _log(f"Loading tags from {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for idx, row in enumerate(reader):
                # row format: tag_id, name, category, count
                tag_name = row[1]
                category = row[2]

                # Find category boundaries
                if self._general_index is None and category == "0":
                    self._general_index = idx
                elif self._character_index is None and category == "4":
                    self._character_index = idx

                # Process tag name
                if self.replace_underscore:
                    tag_name = tag_name.replace("_", " ")

                self._tags_list.append(tag_name)

        _log(f"Loaded {len(self._tags_list)} tags")
        _log(f"General tags start at index {self._general_index}")
        _log(f"Character tags start at index {self._character_index}")

    def load(self) -> None:
        """Load WD14 model and tags."""
        if self.is_loaded:
            _log("Model already loaded")
            return

        # Download if needed
        self._download_model()

        onnx_path, csv_path = self._get_model_path()

        _log(f"Loading ONNX model from {onnx_path}")

        # Import ONNX runtime
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for WD14 tagger. "
                "Install with: pip install onnxruntime-gpu"
            )

        # Get available providers
        available = ort.get_available_providers()
        _log(f"Available ONNX providers: {available}")

        # Prefer GPU if available
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        _log(f"Using providers: {providers}")

        # Load model
        print(f"[SID-Toolkit] Loading WD14 model: {self.model_name}")
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)

        # Get input size from model
        input_shape = self._session.get_inputs()[0].shape
        self._input_size = input_shape[1]  # Usually 448 or 512
        _log(f"Model input size: {self._input_size}")

        # Load tags
        self._load_tags(csv_path)

        self._is_loaded = True
        print(f"[SID-Toolkit] WD14 model loaded: {self.model_name}")

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize maintaining aspect ratio and pad with white
        size = self._input_size
        ratio = float(size) / max(image.size)
        new_size = tuple(int(x * ratio) for x in image.size)

        image = image.resize(new_size, Image.LANCZOS)

        # Create white square and paste image centered
        square = Image.new("RGB", (size, size), (255, 255, 255))
        paste_x = (size - new_size[0]) // 2
        paste_y = (size - new_size[1]) // 2
        square.paste(image, (paste_x, paste_y))

        # Convert to numpy array
        arr = np.array(square).astype(np.float32)
        # RGB -> BGR
        arr = arr[:, :, ::-1]
        # Add batch dimension
        arr = np.expand_dims(arr, 0)

        return arr

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Process image and return tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with detected tags
        """
        if not self.is_loaded:
            self.load()

        _log(f"Processing image: {image.size}")

        # Preprocess
        input_data = self._preprocess_image(image)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        _log("Running inference...")

        probs = self._session.run([output_name], {input_name: input_data})[0][0]

        _log(f"Got {len(probs)} predictions")

        # Process results
        tags = []

        # Rating tags (before general_index) - just get the highest
        if self._general_index is not None and self._general_index > 0:
            rating_probs = probs[:self._general_index]
            rating_idx = np.argmax(rating_probs)
            rating_tag = self._tags_list[rating_idx]
            rating_conf = float(rating_probs[rating_idx])
            tags.append(TagItem(
                text=rating_tag,
                confidence=rating_conf,
                category="rating"
            ))
            _log(f"Rating: {rating_tag} ({rating_conf:.3f})")

        # General tags
        if self._general_index is not None and self._character_index is not None:
            for idx in range(self._general_index, self._character_index):
                if probs[idx] > self.general_threshold:
                    tag_text = self._tags_list[idx]
                    if tag_text.lower() not in self.exclude_tags:
                        # Escape parentheses for prompt compatibility
                        tag_text = tag_text.replace("(", "\\(").replace(")", "\\)")
                        tags.append(TagItem(
                            text=tag_text,
                            confidence=float(probs[idx]),
                            category="general"
                        ))

        # Character tags
        if self._character_index is not None:
            for idx in range(self._character_index, len(self._tags_list)):
                if probs[idx] > self.character_threshold:
                    tag_text = self._tags_list[idx]
                    if tag_text.lower() not in self.exclude_tags:
                        tag_text = tag_text.replace("(", "\\(").replace(")", "\\)")
                        tags.append(TagItem(
                            text=tag_text,
                            confidence=float(probs[idx]),
                            category="character"
                        ))

        # Sort by confidence (descending)
        tags.sort(key=lambda x: x.confidence, reverse=True)

        _log(f"Found {len(tags)} tags above threshold")
        if tags:
            _log(f"Top 5: {[t.text for t in tags[:5]]}")

        return TaggerResult(
            tags=tags,
            metadata={
                "model": self.model_name,
                "general_threshold": self.general_threshold,
                "character_threshold": self.character_threshold,
            }
        )

    def unload(self) -> None:
        """Release model from memory."""
        _log("Unloading model...")
        if self._session is not None:
            del self._session
            self._session = None
        self._tags_list = []
        self._is_loaded = False
        super().unload()
        _log("Model unloaded")
