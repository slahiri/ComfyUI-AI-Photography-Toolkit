"""DWPose Tagger - Convert skeleton keypoints to descriptive pose tags.

Uses DWPose (from controlnet_aux or comfyui_controlnet_aux) to detect body pose
and converts the keypoints into human-readable pose descriptions.

DWPose outputs:
- Body: 17 COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- Hands: 21 keypoints per hand
- Face: 68+ keypoints

This tagger converts these into tags like:
- "arms raised above head"
- "hands on hips"
- "standing with weight on left leg"
- "looking left"
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .base import BaseTagger, TagItem, TaggerResult


# COCO body keypoint indices
BODY_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass
class Keypoint:
    """A single keypoint with position and confidence."""
    x: float
    y: float
    confidence: float

    @property
    def is_valid(self) -> bool:
        """Check if keypoint is detected with sufficient confidence."""
        return self.confidence > 0.3

    def distance_to(self, other: "Keypoint") -> float:
        """Calculate distance to another keypoint."""
        if not self.is_valid or not other.is_valid:
            return float("inf")
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "Keypoint") -> float:
        """Calculate angle to another keypoint in degrees."""
        if not self.is_valid or not other.is_valid:
            return 0.0
        return np.degrees(np.arctan2(other.y - self.y, other.x - self.x))


class DWPoseTagger(BaseTagger):
    """Convert DWPose skeleton keypoints to descriptive pose tags."""

    def __init__(
        self,
        threshold: float = 0.6,
        use_dwpose: bool = True,
    ):
        """Initialize DWPose tagger.

        Args:
            threshold: Minimum confidence for output tags
            use_dwpose: Whether to run DWPose detection (requires controlnet_aux)
        """
        super().__init__()
        self.threshold = threshold
        self.use_dwpose = use_dwpose
        self._dwpose = None
        self._device = None

    def load(self) -> None:
        """Load DWPose model if available."""
        if not self.use_dwpose:
            self._is_loaded = True
            return

        try:
            # Try to import from controlnet_aux (ComfyUI)
            from controlnet_aux import DWposeDetector
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._dwpose = DWposeDetector.from_pretrained(
                "yolox_l.onnx",
                "dw-ll_ucoco_384.onnx",
            )
            self._is_loaded = True
            print("[DWPose] Loaded DWPose detector")

        except ImportError:
            try:
                # Try alternative import path
                from comfyui_controlnet_aux.src.controlnet_aux import DWposeDetector
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._dwpose = DWposeDetector.from_pretrained(
                    "yolox_l.onnx",
                    "dw-ll_ucoco_384.onnx",
                )
                self._is_loaded = True
                print("[DWPose] Loaded DWPose detector (alternative path)")

            except ImportError:
                print("[DWPose] DWPose not available - use tag_from_keypoints() instead")
                self._is_loaded = True  # Can still process pre-computed keypoints

    def tag(self, image: Image.Image) -> TaggerResult:
        """Detect pose and generate tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with pose tags
        """
        if self._dwpose is None:
            return TaggerResult(
                tags=[],
                metadata={"error": "DWPose not loaded - install controlnet_aux"}
            )

        try:
            # Run DWPose detection
            result = self._dwpose(image, output_type="np")

            # Extract keypoints from result
            # DWPose returns: {"bodies": {"candidate": [...], "subset": [...]}, "hands": [...], "faces": [...]}
            if isinstance(result, dict) and "bodies" in result:
                bodies = result["bodies"]
                if "candidate" in bodies and len(bodies["candidate"]) > 0:
                    keypoints = self._parse_dwpose_output(bodies)
                    return self.tag_from_keypoints(keypoints)

            return TaggerResult(
                tags=[],
                metadata={"error": "No pose detected"}
            )

        except Exception as e:
            print(f"[DWPose] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return TaggerResult(tags=[], metadata={"error": str(e)})

    def _parse_dwpose_output(self, bodies: Dict) -> Dict[str, Keypoint]:
        """Parse DWPose output into keypoint dict.

        Args:
            bodies: DWPose bodies output

        Returns:
            Dict mapping keypoint name to Keypoint object
        """
        keypoints = {}
        candidate = bodies.get("candidate", [])
        subset = bodies.get("subset", [[]])

        if len(subset) == 0 or len(candidate) == 0:
            return keypoints

        # Use first detected person
        person = subset[0]

        for name, idx in BODY_KEYPOINTS.items():
            if idx < len(person):
                point_idx = int(person[idx])
                if point_idx >= 0 and point_idx < len(candidate):
                    point = candidate[point_idx]
                    keypoints[name] = Keypoint(
                        x=float(point[0]),
                        y=float(point[1]),
                        confidence=float(point[2]) if len(point) > 2 else 0.5,
                    )
                else:
                    keypoints[name] = Keypoint(x=0, y=0, confidence=0)
            else:
                keypoints[name] = Keypoint(x=0, y=0, confidence=0)

        return keypoints

    def tag_from_keypoints(
        self,
        keypoints: Dict[str, Keypoint],
        image_height: float = 1.0,
    ) -> TaggerResult:
        """Generate pose tags from keypoints.

        Args:
            keypoints: Dict mapping keypoint name to Keypoint
            image_height: Image height for normalization

        Returns:
            TaggerResult with pose tags
        """
        tags: List[TagItem] = []

        # Analyze arm positions
        arm_tags = self._analyze_arms(keypoints, image_height)
        tags.extend(arm_tags)

        # Analyze leg positions
        leg_tags = self._analyze_legs(keypoints, image_height)
        tags.extend(leg_tags)

        # Analyze head/gaze
        head_tags = self._analyze_head(keypoints)
        tags.extend(head_tags)

        # Analyze body orientation
        body_tags = self._analyze_body(keypoints)
        tags.extend(body_tags)

        # Filter by threshold
        tags = [t for t in tags if t.confidence >= self.threshold]

        # Sort by confidence
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "source": "dwpose",
                "keypoints_detected": sum(1 for k in keypoints.values() if k.is_valid),
                "total_keypoints": len(keypoints),
            }
        )

    def _analyze_arms(
        self,
        kp: Dict[str, Keypoint],
        img_height: float,
    ) -> List[TagItem]:
        """Analyze arm positions."""
        tags = []

        # Get relevant keypoints
        l_shoulder = kp.get("left_shoulder", Keypoint(0, 0, 0))
        r_shoulder = kp.get("right_shoulder", Keypoint(0, 0, 0))
        l_elbow = kp.get("left_elbow", Keypoint(0, 0, 0))
        r_elbow = kp.get("right_elbow", Keypoint(0, 0, 0))
        l_wrist = kp.get("left_wrist", Keypoint(0, 0, 0))
        r_wrist = kp.get("right_wrist", Keypoint(0, 0, 0))
        nose = kp.get("nose", Keypoint(0, 0, 0))

        # Calculate shoulder center y for reference
        if l_shoulder.is_valid and r_shoulder.is_valid:
            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2
        else:
            return tags

        # Check if arms are raised above head
        head_y = nose.y if nose.is_valid else shoulder_y - img_height * 0.15

        left_raised = l_wrist.is_valid and l_wrist.y < head_y
        right_raised = r_wrist.is_valid and r_wrist.y < head_y

        if left_raised and right_raised:
            tags.append(TagItem(
                text="arms raised above head",
                confidence=0.9,
                category="arm_pose",
            ))
            tags.append(TagItem(
                text="both arms raised",
                confidence=0.85,
                category="arm_pose",
            ))
        elif left_raised or right_raised:
            tags.append(TagItem(
                text="one arm raised",
                confidence=0.85,
                category="arm_pose",
            ))

        # Check if hands near face (covering eyes, touching face)
        if nose.is_valid:
            face_center_y = nose.y
            face_threshold = img_height * 0.1

            l_near_face = l_wrist.is_valid and abs(l_wrist.y - face_center_y) < face_threshold
            r_near_face = r_wrist.is_valid and abs(r_wrist.y - face_center_y) < face_threshold

            if l_near_face and r_near_face:
                tags.append(TagItem(
                    text="hands near face",
                    confidence=0.85,
                    category="arm_pose",
                ))
                # Check if hands are in front of face (covering)
                if (l_wrist.x > shoulder_center_x - img_height * 0.15 and
                    r_wrist.x < shoulder_center_x + img_height * 0.15):
                    tags.append(TagItem(
                        text="hands covering face",
                        confidence=0.8,
                        category="arm_pose",
                    ))
            elif l_near_face or r_near_face:
                tags.append(TagItem(
                    text="hand touching face",
                    confidence=0.8,
                    category="arm_pose",
                ))

        # Check arms at sides
        if l_wrist.is_valid and r_wrist.is_valid:
            l_at_side = abs(l_wrist.y - l_shoulder.y) < img_height * 0.3 and l_wrist.y > l_shoulder.y
            r_at_side = abs(r_wrist.y - r_shoulder.y) < img_height * 0.3 and r_wrist.y > r_shoulder.y

            if l_at_side and r_at_side and not (left_raised or right_raised):
                tags.append(TagItem(
                    text="arms at sides",
                    confidence=0.75,
                    category="arm_pose",
                ))

        # Check arms crossed
        if l_wrist.is_valid and r_wrist.is_valid:
            # Arms crossed if wrists are on opposite sides of body
            if l_wrist.x > shoulder_center_x and r_wrist.x < shoulder_center_x:
                tags.append(TagItem(
                    text="arms crossed",
                    confidence=0.8,
                    category="arm_pose",
                ))

        # Check hands on hips
        l_hip = kp.get("left_hip", Keypoint(0, 0, 0))
        r_hip = kp.get("right_hip", Keypoint(0, 0, 0))

        if l_hip.is_valid and r_hip.is_valid:
            hip_threshold = img_height * 0.08
            l_on_hip = l_wrist.is_valid and l_wrist.distance_to(l_hip) < hip_threshold
            r_on_hip = r_wrist.is_valid and r_wrist.distance_to(r_hip) < hip_threshold

            if l_on_hip and r_on_hip:
                tags.append(TagItem(
                    text="hands on hips",
                    confidence=0.85,
                    category="arm_pose",
                ))
            elif l_on_hip or r_on_hip:
                tags.append(TagItem(
                    text="hand on hip",
                    confidence=0.85,
                    category="arm_pose",
                ))

        return tags

    def _analyze_legs(
        self,
        kp: Dict[str, Keypoint],
        img_height: float,
    ) -> List[TagItem]:
        """Analyze leg positions."""
        tags = []

        l_hip = kp.get("left_hip", Keypoint(0, 0, 0))
        r_hip = kp.get("right_hip", Keypoint(0, 0, 0))
        l_knee = kp.get("left_knee", Keypoint(0, 0, 0))
        r_knee = kp.get("right_knee", Keypoint(0, 0, 0))
        l_ankle = kp.get("left_ankle", Keypoint(0, 0, 0))
        r_ankle = kp.get("right_ankle", Keypoint(0, 0, 0))

        if not (l_hip.is_valid and r_hip.is_valid):
            return tags

        # Check stance width
        if l_ankle.is_valid and r_ankle.is_valid:
            hip_width = abs(l_hip.x - r_hip.x)
            ankle_width = abs(l_ankle.x - r_ankle.x)

            if ankle_width < hip_width * 0.5:
                tags.append(TagItem(
                    text="legs together",
                    confidence=0.8,
                    category="leg_pose",
                ))
            elif ankle_width > hip_width * 1.5:
                tags.append(TagItem(
                    text="legs apart",
                    confidence=0.8,
                    category="leg_pose",
                ))
                if ankle_width > hip_width * 2.5:
                    tags.append(TagItem(
                        text="wide stance",
                        confidence=0.75,
                        category="leg_pose",
                    ))

        # Check weight distribution
        if l_ankle.is_valid and r_ankle.is_valid and l_hip.is_valid and r_hip.is_valid:
            # Weight on one leg if hip is shifted significantly
            hip_center = (l_hip.x + r_hip.x) / 2
            l_hip_shift = l_hip.y - r_hip.y

            if abs(l_hip_shift) > img_height * 0.02:
                if l_hip_shift > 0:
                    tags.append(TagItem(
                        text="weight on right leg",
                        confidence=0.7,
                        category="leg_pose",
                    ))
                else:
                    tags.append(TagItem(
                        text="weight on left leg",
                        confidence=0.7,
                        category="leg_pose",
                    ))

        # Check if kneeling
        if l_knee.is_valid and r_knee.is_valid:
            avg_knee_y = (l_knee.y + r_knee.y) / 2
            avg_hip_y = (l_hip.y + r_hip.y) / 2

            # Kneeling if knees are close to where ankles would be
            if l_ankle.is_valid and r_ankle.is_valid:
                avg_ankle_y = (l_ankle.y + r_ankle.y) / 2
                if avg_knee_y > avg_ankle_y - img_height * 0.05:
                    tags.append(TagItem(
                        text="kneeling",
                        confidence=0.8,
                        category="body_pose",
                    ))

        return tags

    def _analyze_head(self, kp: Dict[str, Keypoint]) -> List[TagItem]:
        """Analyze head position and gaze."""
        tags = []

        nose = kp.get("nose", Keypoint(0, 0, 0))
        l_eye = kp.get("left_eye", Keypoint(0, 0, 0))
        r_eye = kp.get("right_eye", Keypoint(0, 0, 0))
        l_ear = kp.get("left_ear", Keypoint(0, 0, 0))
        r_ear = kp.get("right_ear", Keypoint(0, 0, 0))
        l_shoulder = kp.get("left_shoulder", Keypoint(0, 0, 0))
        r_shoulder = kp.get("right_shoulder", Keypoint(0, 0, 0))

        if not nose.is_valid:
            return tags

        # Head tilt (based on ear positions)
        if l_ear.is_valid and r_ear.is_valid:
            ear_diff = l_ear.y - r_ear.y
            if abs(ear_diff) > 10:
                if ear_diff > 0:
                    tags.append(TagItem(
                        text="head tilted right",
                        confidence=0.75,
                        category="head_pose",
                    ))
                else:
                    tags.append(TagItem(
                        text="head tilted left",
                        confidence=0.75,
                        category="head_pose",
                    ))

        # Head turn (based on eye-ear visibility)
        if l_shoulder.is_valid and r_shoulder.is_valid:
            shoulder_center = (l_shoulder.x + r_shoulder.x) / 2

            # Profile detection
            l_ear_visible = l_ear.is_valid and l_ear.confidence > 0.5
            r_ear_visible = r_ear.is_valid and r_ear.confidence > 0.5

            if l_ear_visible and not r_ear_visible:
                tags.append(TagItem(
                    text="head turned right",
                    confidence=0.8,
                    category="head_pose",
                ))
                tags.append(TagItem(
                    text="profile view",
                    confidence=0.7,
                    category="head_pose",
                ))
            elif r_ear_visible and not l_ear_visible:
                tags.append(TagItem(
                    text="head turned left",
                    confidence=0.8,
                    category="head_pose",
                ))
                tags.append(TagItem(
                    text="profile view",
                    confidence=0.7,
                    category="head_pose",
                ))

        # Looking over shoulder
        if l_shoulder.is_valid and r_shoulder.is_valid:
            shoulder_center = (l_shoulder.x + r_shoulder.x) / 2
            if nose.x < shoulder_center - 20 or nose.x > shoulder_center + 20:
                # Head significantly off-center from shoulders
                if (l_ear.is_valid and not r_ear.is_valid) or (r_ear.is_valid and not l_ear.is_valid):
                    tags.append(TagItem(
                        text="looking over shoulder",
                        confidence=0.75,
                        category="gaze",
                    ))

        return tags

    def _analyze_body(self, kp: Dict[str, Keypoint]) -> List[TagItem]:
        """Analyze overall body position."""
        tags = []

        l_shoulder = kp.get("left_shoulder", Keypoint(0, 0, 0))
        r_shoulder = kp.get("right_shoulder", Keypoint(0, 0, 0))
        l_hip = kp.get("left_hip", Keypoint(0, 0, 0))
        r_hip = kp.get("right_hip", Keypoint(0, 0, 0))
        l_ankle = kp.get("left_ankle", Keypoint(0, 0, 0))
        r_ankle = kp.get("right_ankle", Keypoint(0, 0, 0))

        # Check if standing, sitting, or lying
        if l_shoulder.is_valid and r_shoulder.is_valid and l_hip.is_valid and r_hip.is_valid:
            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            hip_y = (l_hip.y + r_hip.y) / 2

            # Torso is roughly vertical = standing/sitting
            # Torso is roughly horizontal = lying down
            shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
            hip_x = (l_hip.x + r_hip.x) / 2

            torso_angle = np.degrees(np.arctan2(hip_y - shoulder_y, hip_x - shoulder_x))

            if 60 < abs(torso_angle) < 120:
                # Vertical torso
                if l_ankle.is_valid and r_ankle.is_valid:
                    ankle_y = (l_ankle.y + r_ankle.y) / 2
                    if ankle_y > hip_y + 50:
                        tags.append(TagItem(
                            text="standing",
                            confidence=0.8,
                            category="body_pose",
                        ))
                    else:
                        tags.append(TagItem(
                            text="sitting",
                            confidence=0.75,
                            category="body_pose",
                        ))
                else:
                    # Can't see ankles, guess based on hip position
                    tags.append(TagItem(
                        text="upright pose",
                        confidence=0.7,
                        category="body_pose",
                    ))
            elif abs(torso_angle) < 30 or abs(torso_angle) > 150:
                # Horizontal torso
                tags.append(TagItem(
                    text="lying down",
                    confidence=0.75,
                    category="body_pose",
                ))

        # Check for contrapposto (S-curve)
        if (l_shoulder.is_valid and r_shoulder.is_valid and
            l_hip.is_valid and r_hip.is_valid):

            shoulder_tilt = l_shoulder.y - r_shoulder.y
            hip_tilt = l_hip.y - r_hip.y

            # Contrapposto: shoulders and hips tilt in opposite directions
            if shoulder_tilt * hip_tilt < 0 and abs(shoulder_tilt) > 5 and abs(hip_tilt) > 5:
                tags.append(TagItem(
                    text="contrapposto pose",
                    confidence=0.75,
                    category="body_pose",
                ))

        return tags

    def unload(self) -> None:
        """Unload DWPose model."""
        if self._dwpose is not None:
            del self._dwpose
            self._dwpose = None
