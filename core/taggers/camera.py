"""Camera and cinematography analysis taggers.

Includes:
- ShotTypeClassifier: Classifies shot scale (close-up, medium, full, long)
- CLIPCameraTagger: Zero-shot classification for camera angles, DOF, etc.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List

from .base import BaseTagger, TagItem, TaggerResult
from .utils import get_tagger_models_dir


class ShotTypeClassifier(BaseTagger):
    """
    Shot Type Classifier using MobileNetV3.

    Classifies images into 5 shot types:
    - Extreme Close-up (ECS): Very tight framing on small detail
    - Close-up (CS): Face or small object fills frame
    - Medium Shot (MS): Waist/knees up
    - Full Shot (FS): Full body visible
    - Long Shot (LS): Subject small in frame, environment dominant

    Based on: https://github.com/sssabet/Shot_Type_Classification
    Trained on MovieShots dataset.
    """

    TAGGER_NAME = "shot_type"

    # Shot type labels in model order
    SHOT_TYPES = [
        "long shot",
        "full shot",
        "medium shot",
        "close-up shot",
        "extreme close-up",
    ]

    # Friendly display names
    SHOT_TYPE_DISPLAY = {
        "long shot": "long shot",
        "full shot": "full body shot",
        "medium shot": "medium shot",
        "close-up shot": "close-up",
        "extreme close-up": "extreme close-up",
    }

    def __init__(
        self,
        threshold: float = 0.3,
        top_k: int = 2,
        **kwargs
    ):
        """
        Initialize shot type classifier.

        Args:
            threshold: Minimum confidence to include tag
            top_k: Maximum number of shot types to return
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.top_k = top_k
        self._model = None
        self._transform = None
        self._device = None

    def load(self) -> None:
        """Load MobileNetV3 model."""
        if self._is_loaded:
            return

        from torchvision import transforms
        from torchvision.models import mobilenet_v3_small

        # Setup device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check for local model weights
        models_dir = get_tagger_models_dir("shot_type")
        weights_path = models_dir / "mobilenetv3_shot_type.pth"

        # Create model architecture
        self._model = mobilenet_v3_small(weights=None)
        # Modify classifier for 5 classes
        self._model.classifier[-1] = torch.nn.Linear(
            self._model.classifier[-1].in_features, 5
        )

        if weights_path.exists():
            # Load trained weights
            print(f"[SID-ShotType] Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self._device)
            self._model.load_state_dict(state_dict)
        else:
            # Use ImageNet pretrained and hope for reasonable results
            # In production, we'd download the trained weights
            print("[SID-ShotType] No trained weights found, using pretrained backbone")
            print(f"[SID-ShotType] Place trained weights at: {weights_path}")
            # Load pretrained backbone
            from torchvision.models import MobileNet_V3_Small_Weights
            pretrained = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # Copy backbone weights only
            for name, param in pretrained.named_parameters():
                if not name.startswith('classifier'):
                    if name in dict(self._model.named_parameters()):
                        dict(self._model.named_parameters())[name].data.copy_(param.data)

        self._model.to(self._device)
        self._model.eval()

        # Standard ImageNet transforms
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self._is_loaded = True
        print("[SID-ShotType] Shot type classifier loaded")

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Classify shot type.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with shot type tags
        """
        if not self._is_loaded:
            self.load()

        tags = []
        metadata = {}

        try:
            # Preprocess image
            rgb_image = image.convert("RGB")
            input_tensor = self._transform(rgb_image).unsqueeze(0).to(self._device)

            # Inference
            with torch.no_grad():
                outputs = self._model(input_tensor)
                probs = torch.softmax(outputs, dim=-1)[0]

            probs_np = probs.cpu().numpy()

            # Get top predictions
            top_indices = np.argsort(probs_np)[::-1][:self.top_k]

            for idx in top_indices:
                conf = float(probs_np[idx])
                if conf >= self.threshold:
                    shot_type = self.SHOT_TYPES[idx]
                    display_name = self.SHOT_TYPE_DISPLAY.get(shot_type, shot_type)
                    tags.append(TagItem(
                        text=display_name,
                        confidence=conf,
                        category="composition:shot_type"
                    ))

            # Store primary shot type
            primary_idx = int(np.argmax(probs_np))
            metadata["primary_shot_type"] = self.SHOT_TYPES[primary_idx]
            metadata["shot_type_confidence"] = float(probs_np[primary_idx])

        except Exception as e:
            print(f"[SID-ShotType] Error: {e}")
            metadata["error"] = str(e)

        return TaggerResult(
            tags=tags,
            metadata={"model": "mobilenetv3_shot_type", **metadata}
        )

    def unload(self) -> None:
        """Release model."""
        if self._model is not None:
            del self._model
            self._model = None
        self._transform = None
        self._device = None
        super().unload()


class CLIPCameraTagger(BaseTagger):
    """
    Zero-shot camera attribute classifier using CLIP.

    Classifies:
    - Camera angles: low angle, high angle, eye level, bird's eye view, worm's eye view
    - Depth of field: shallow DOF, deep focus, bokeh
    - Camera movement hints: static, dynamic composition

    Uses OpenAI CLIP for flexible zero-shot classification.
    """

    TAGGER_NAME = "clip_camera"

    # Camera angle labels
    CAMERA_ANGLES = [
        "eye level shot",
        "low angle shot",
        "high angle shot",
        "bird's eye view",
        "worm's eye view",
        "dutch angle",
        "overhead shot",
    ]

    # Depth of field labels
    DOF_LABELS = [
        "shallow depth of field",
        "deep focus",
        "bokeh background",
        "sharp throughout",
        "selective focus",
    ]

    # Shot perspective
    PERSPECTIVE_LABELS = [
        "front view",
        "side view",
        "three-quarter view",
        "back view",
        "profile view",
    ]

    def __init__(
        self,
        threshold: float = 0.15,
        model_name: str = "openai/clip-vit-base-patch32",
        **kwargs
    ):
        """
        Initialize CLIP camera tagger.

        Args:
            threshold: Minimum confidence to include tag
            model_name: CLIP model to use
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load CLIP model."""
        if self._is_loaded:
            return

        from transformers import CLIPProcessor, CLIPModel

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SID-CLIPCamera] Loading {self.model_name}...")
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        self._is_loaded = True
        print("[SID-CLIPCamera] CLIP camera tagger loaded")

    def _classify_with_labels(
        self,
        image: Image.Image,
        labels: List[str],
        category: str,
        top_k: int = 2
    ) -> List[TagItem]:
        """Classify image against a set of labels."""
        tags = []

        # Prepare inputs
        inputs = self._processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()

        # Get top predictions
        top_indices = np.argsort(probs_np)[::-1][:top_k]

        for idx in top_indices:
            conf = float(probs_np[idx])
            if conf >= self.threshold:
                tags.append(TagItem(
                    text=labels[idx],
                    confidence=conf,
                    category=category
                ))

        return tags

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Classify camera attributes using CLIP.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with camera attribute tags
        """
        if not self._is_loaded:
            self.load()

        tags = []
        metadata = {}

        try:
            rgb_image = image.convert("RGB")

            # Classify camera angles
            angle_tags = self._classify_with_labels(
                rgb_image,
                self.CAMERA_ANGLES,
                "composition:camera_angle",
                top_k=2
            )
            tags.extend(angle_tags)

            if angle_tags:
                metadata["primary_angle"] = angle_tags[0].text

            # Classify depth of field
            dof_tags = self._classify_with_labels(
                rgb_image,
                self.DOF_LABELS,
                "composition:dof",
                top_k=1
            )
            tags.extend(dof_tags)

            if dof_tags:
                metadata["depth_of_field"] = dof_tags[0].text

            # Classify perspective
            persp_tags = self._classify_with_labels(
                rgb_image,
                self.PERSPECTIVE_LABELS,
                "composition:perspective",
                top_k=1
            )
            tags.extend(persp_tags)

            if persp_tags:
                metadata["perspective"] = persp_tags[0].text

        except Exception as e:
            print(f"[SID-CLIPCamera] Error: {e}")
            import traceback
            traceback.print_exc()
            metadata["error"] = str(e)

        return TaggerResult(
            tags=tags,
            metadata={"model": self.model_name, **metadata}
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
