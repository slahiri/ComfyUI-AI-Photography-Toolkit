"""NudeNet v2 tagger - NSFW detection with 16 classes."""

from PIL import Image
import numpy as np

from .base import BaseTagger, TagItem, TaggerResult
from .utils import download_single_file, get_tagger_models_dir


class NudeNetTagger(BaseTagger):
    """
    NudeNet v2 NSFW detector with 16 classes.

    Detection classes:
    EXPOSED:
        - FEMALE_BREAST_EXPOSED
        - FEMALE_GENITALIA_EXPOSED
        - MALE_GENITALIA_EXPOSED
        - BUTTOCKS_EXPOSED
        - ANUS_EXPOSED
        - FEET_EXPOSED
        - BELLY_EXPOSED
        - ARMPITS_EXPOSED

    COVERED:
        - FEMALE_BREAST_COVERED
        - BUTTOCKS_COVERED
        - BELLY_COVERED
        - FEET_COVERED

    OTHER:
        - FACE_FEMALE
        - FACE_MALE

    Model: notAI-tech/NudeNet
    """

    TAGGER_NAME = "nudenet"
    MODEL_ID = "notAI-tech/NudeNet"

    # Class definitions
    CLASSES = {
        0: "FEMALE_GENITALIA_COVERED",
        1: "FACE_FEMALE",
        2: "BUTTOCKS_EXPOSED",
        3: "FEMALE_BREAST_EXPOSED",
        4: "FEMALE_GENITALIA_EXPOSED",
        5: "MALE_BREAST_EXPOSED",
        6: "ANUS_EXPOSED",
        7: "FEET_EXPOSED",
        8: "BELLY_COVERED",
        9: "FEET_COVERED",
        10: "ARMPITS_COVERED",
        11: "ARMPITS_EXPOSED",
        12: "FACE_MALE",
        13: "BELLY_EXPOSED",
        14: "MALE_GENITALIA_EXPOSED",
        15: "ANUS_COVERED",
        16: "FEMALE_BREAST_COVERED",
        17: "BUTTOCKS_COVERED",
    }

    # Tag text mappings (more natural language)
    TAG_TEXT = {
        "FEMALE_GENITALIA_COVERED": "covered female genitalia",
        "FACE_FEMALE": "female face",
        "BUTTOCKS_EXPOSED": "exposed buttocks",
        "FEMALE_BREAST_EXPOSED": "exposed female breast",
        "FEMALE_GENITALIA_EXPOSED": "exposed female genitalia",
        "MALE_BREAST_EXPOSED": "exposed male chest",
        "ANUS_EXPOSED": "exposed anus",
        "FEET_EXPOSED": "exposed feet",
        "BELLY_COVERED": "covered belly",
        "FEET_COVERED": "covered feet",
        "ARMPITS_COVERED": "covered armpits",
        "ARMPITS_EXPOSED": "exposed armpits",
        "FACE_MALE": "male face",
        "BELLY_EXPOSED": "exposed belly",
        "MALE_GENITALIA_EXPOSED": "exposed male genitalia",
        "ANUS_COVERED": "covered anus",
        "FEMALE_BREAST_COVERED": "covered female breast",
        "BUTTOCKS_COVERED": "covered buttocks",
    }

    # Categories
    EXPOSED_CLASSES = {
        "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_EXPOSED",
        "ARMPITS_EXPOSED", "MALE_BREAST_EXPOSED"
    }

    COVERED_CLASSES = {
        "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
        "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED",
        "FEMALE_GENITALIA_COVERED", "ANUS_COVERED"
    }

    FACE_CLASSES = {"FACE_FEMALE", "FACE_MALE"}

    def __init__(
        self,
        threshold: float = 0.5,
        include_covered: bool = True,
        include_faces: bool = True,
        **kwargs
    ):
        """
        Initialize NudeNet tagger.

        Args:
            threshold: Detection confidence threshold
            include_covered: Include covered body part detections
            include_faces: Include face detections
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.include_covered = include_covered
        self.include_faces = include_faces
        self._detector = None

    def load(self) -> None:
        """Load NudeNet model."""
        if self._is_loaded:
            return

        try:
            from nudenet import NudeDetector
        except ImportError:
            raise ImportError(
                "nudenet is required for NudeNet tagger. "
                "Install with: pip install nudenet"
            )

        print("[SID-NudeNet] Loading NudeNet v2...")
        self._detector = NudeDetector()
        print("[SID-NudeNet] Model loaded")

        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Detect NSFW content in image.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with NSFW detection tags
        """
        if not self._is_loaded:
            self.load()

        import tempfile
        import os

        # Save image temporarily (NudeNet requires file path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            rgb_image = image.convert("RGB")
            rgb_image.save(tmp.name, "JPEG", quality=95)
            tmp_path = tmp.name

        try:
            # Run detection
            detections = self._detector.detect(tmp_path)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        # Process detections
        tags = []
        detection_counts = {}

        for detection in detections:
            class_name = detection.get("class", "")
            score = detection.get("score", 0)

            if score < self.threshold:
                continue

            # Filter by category
            if class_name in self.COVERED_CLASSES and not self.include_covered:
                continue
            if class_name in self.FACE_CLASSES and not self.include_faces:
                continue

            # Count detections per class
            if class_name not in detection_counts:
                detection_counts[class_name] = {"count": 0, "max_score": 0}
            detection_counts[class_name]["count"] += 1
            detection_counts[class_name]["max_score"] = max(
                detection_counts[class_name]["max_score"], score
            )

        # Convert to tags
        for class_name, data in detection_counts.items():
            tag_text = self.TAG_TEXT.get(class_name, class_name.lower().replace("_", " "))

            # Add count if multiple
            if data["count"] > 1:
                tag_text = f"{tag_text} (x{data['count']})"

            # Determine category
            if class_name in self.EXPOSED_CLASSES:
                category = "nsfw:exposed"
            elif class_name in self.COVERED_CLASSES:
                category = "nsfw:covered"
            elif class_name in self.FACE_CLASSES:
                category = "nsfw:face"
            else:
                category = "nsfw:other"

            tags.append(TagItem(
                text=tag_text,
                confidence=data["max_score"],
                category=category
            ))

        # Add overall safety tag
        exposed_count = sum(
            1 for c in detection_counts if c in self.EXPOSED_CLASSES
        )
        if exposed_count > 0:
            tags.append(TagItem(
                text="nsfw content",
                confidence=0.9,
                category="nsfw:rating"
            ))
        elif len(detection_counts) > 0:
            tags.append(TagItem(
                text="suggestive content",
                confidence=0.7,
                category="nsfw:rating"
            ))
        else:
            tags.append(TagItem(
                text="safe content",
                confidence=0.8,
                category="nsfw:rating"
            ))

        # Sort by confidence
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "nudenet-v2",
                "total_detections": len(detections),
                "filtered_detections": len(detection_counts),
                "exposed_count": exposed_count,
                "threshold": self.threshold,
            }
        )

    def unload(self) -> None:
        """Release NudeNet model."""
        if self._detector is not None:
            del self._detector
            self._detector = None
        super().unload()


class NudeNetONNXTagger(BaseTagger):
    """
    NudeNet using ONNX runtime for faster inference.

    Alternative implementation using ONNX model directly.
    """

    TAGGER_NAME = "nudenet_onnx"
    MODEL_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx"
    MODEL_NAME = "nudenet_640m.onnx"

    # Same class definitions as NudeNetTagger
    CLASSES = NudeNetTagger.CLASSES
    TAG_TEXT = NudeNetTagger.TAG_TEXT
    EXPOSED_CLASSES = NudeNetTagger.EXPOSED_CLASSES
    COVERED_CLASSES = NudeNetTagger.COVERED_CLASSES
    FACE_CLASSES = NudeNetTagger.FACE_CLASSES

    def __init__(
        self,
        threshold: float = 0.5,
        include_covered: bool = True,
        include_faces: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.include_covered = include_covered
        self.include_faces = include_faces
        self._session = None
        self._input_size = 640

    def load(self) -> None:
        """Load ONNX model."""
        if self._is_loaded:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. "
                "Install with: pip install onnxruntime-gpu"
            )

        # Download model
        models_dir = get_tagger_models_dir("nudenet")
        model_path = models_dir / self.MODEL_NAME

        if not model_path.exists():
            print(f"[SID-NudeNet] Downloading ONNX model...")
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, str(model_path))
            print(f"[SID-NudeNet] Downloaded to {model_path}")

        # Load ONNX session
        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        print("[SID-NudeNet] Loading ONNX model...")
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        print("[SID-NudeNet] ONNX model loaded")

        self._is_loaded = True

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ONNX inference."""
        # Resize
        rgb = image.convert("RGB")
        resized = rgb.resize((self._input_size, self._input_size))

        # Convert to numpy and normalize
        arr = np.array(resized).astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)

        return arr

    def tag(self, image: Image.Image) -> TaggerResult:
        """Run ONNX inference."""
        if not self._is_loaded:
            self.load()

        # Preprocess
        input_data = self._preprocess(image)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_data})

        # Process outputs (format depends on model version)
        # NudeNet ONNX outputs detection boxes with class and score
        detections = outputs[0][0]  # [N, 6] = [x1, y1, x2, y2, score, class]

        tags = []
        detection_counts = {}

        for det in detections:
            if len(det) >= 6:
                score = det[4]
                class_id = int(det[5])

                if score < self.threshold:
                    continue

                class_name = self.CLASSES.get(class_id, f"CLASS_{class_id}")

                # Filter
                if class_name in self.COVERED_CLASSES and not self.include_covered:
                    continue
                if class_name in self.FACE_CLASSES and not self.include_faces:
                    continue

                if class_name not in detection_counts:
                    detection_counts[class_name] = {"count": 0, "max_score": 0}
                detection_counts[class_name]["count"] += 1
                detection_counts[class_name]["max_score"] = max(
                    detection_counts[class_name]["max_score"], float(score)
                )

        # Convert to tags
        for class_name, data in detection_counts.items():
            tag_text = self.TAG_TEXT.get(class_name, class_name.lower().replace("_", " "))

            if data["count"] > 1:
                tag_text = f"{tag_text} (x{data['count']})"

            if class_name in self.EXPOSED_CLASSES:
                category = "nsfw:exposed"
            elif class_name in self.COVERED_CLASSES:
                category = "nsfw:covered"
            elif class_name in self.FACE_CLASSES:
                category = "nsfw:face"
            else:
                category = "nsfw:other"

            tags.append(TagItem(
                text=tag_text,
                confidence=data["max_score"],
                category=category
            ))

        # Safety rating
        exposed_count = sum(1 for c in detection_counts if c in self.EXPOSED_CLASSES)
        if exposed_count > 0:
            tags.append(TagItem(text="nsfw content", confidence=0.9, category="nsfw:rating"))
        elif detection_counts:
            tags.append(TagItem(text="suggestive content", confidence=0.7, category="nsfw:rating"))
        else:
            tags.append(TagItem(text="safe content", confidence=0.8, category="nsfw:rating"))

        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "nudenet-onnx",
                "total_detections": len(detections),
                "filtered_detections": len(detection_counts),
                "exposed_count": exposed_count,
            }
        )

    def unload(self) -> None:
        if self._session is not None:
            del self._session
            self._session = None
        super().unload()
