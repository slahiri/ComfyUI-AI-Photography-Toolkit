"""MediaPipe Pose tagger - lightweight pose estimation."""

from typing import Optional
import numpy as np
from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult


# MediaPipe 33 keypoint names
MEDIAPIPE_KEYPOINTS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]


class MediaPipePoseTagger(BaseTagger):
    """
    Lightweight pose estimation using MediaPipe.

    Detects 33 body keypoints and converts them to semantic pose tags.
    Very fast (~20ms) and low VRAM (~100MB).
    """

    TAGGER_NAME = "mediapipe_pose"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None

    def load(self) -> None:
        """Load MediaPipe Pose model."""
        if self._is_loaded:
            return

        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe is required for pose detection. "
                "Install with: pip install mediapipe"
            )

        mp_pose = mp.solutions.pose
        self._pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Detect pose and generate semantic tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with pose tags and keypoint metadata
        """
        if not self._is_loaded:
            self.load()

        # Convert to RGB numpy array
        rgb_image = image.convert("RGB")
        np_image = np.array(rgb_image)

        # Run pose detection
        results = self._pose.process(np_image)

        if not results.pose_landmarks:
            return TaggerResult(
                tags=[TagItem(text="no person detected", confidence=0.5, category="pose")],
                metadata={"detected": False}
            )

        # Extract keypoints as numpy array
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in landmarks
        ])

        # Convert keypoints to semantic tags
        from .pose_semantics import keypoints_to_tags
        tags = keypoints_to_tags(
            keypoints,
            MEDIAPIPE_KEYPOINTS,
            confidence_threshold=self.min_detection_confidence
        )

        return TaggerResult(
            tags=tags,
            metadata={
                "detected": True,
                "num_keypoints": len(landmarks),
                "model": "mediapipe"
            }
        )

    def unload(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        super().unload()
