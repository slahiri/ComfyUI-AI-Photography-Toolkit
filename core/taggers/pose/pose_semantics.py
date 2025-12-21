"""Convert pose keypoints to semantic text tags."""

import numpy as np
from typing import Optional

from ..base import TagItem


# Keypoint indices for MediaPipe (can be adapted for other formats)
class KeypointIndices:
    """Standard keypoint indices for pose analysis."""
    # Face
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8

    # Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    # Hands (MediaPipe specific)
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20

    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


def keypoints_to_tags(
    keypoints: np.ndarray,
    keypoint_names: list[str],
    confidence_threshold: float = 0.5
) -> list[TagItem]:
    """
    Convert raw keypoints to semantic pose tags.

    Args:
        keypoints: Array of shape (N, 4) with [x, y, z, visibility] per keypoint
        keypoint_names: List of keypoint names
        confidence_threshold: Minimum visibility to consider a keypoint valid

    Returns:
        List of TagItem with semantic pose descriptions
    """
    tags = []
    idx = KeypointIndices

    # Check if we have enough visible keypoints
    visible_mask = keypoints[:, 3] >= confidence_threshold
    if not np.any(visible_mask):
        return [TagItem(text="pose unclear", confidence=0.3, category="pose")]

    # Analyze body orientation
    orientation_tag = _analyze_orientation(keypoints, idx, confidence_threshold)
    if orientation_tag:
        tags.append(orientation_tag)

    # Analyze arm position
    arm_tags = _analyze_arms(keypoints, idx, confidence_threshold)
    tags.extend(arm_tags)

    # Analyze body stance
    stance_tag = _analyze_stance(keypoints, idx, confidence_threshold)
    if stance_tag:
        tags.append(stance_tag)

    # Analyze head tilt
    head_tag = _analyze_head(keypoints, idx, confidence_threshold)
    if head_tag:
        tags.append(head_tag)

    # Overall pose style
    style_tag = _analyze_pose_style(keypoints, idx, confidence_threshold)
    if style_tag:
        tags.append(style_tag)

    # Add general pose tag
    tags.append(TagItem(text="person", confidence=0.95, category="pose:subject"))

    return tags


def _analyze_orientation(
    keypoints: np.ndarray,
    idx: KeypointIndices,
    threshold: float
) -> Optional[TagItem]:
    """Determine if person is facing camera, profile, or back."""
    # Get shoulder positions
    left_shoulder = keypoints[idx.LEFT_SHOULDER]
    right_shoulder = keypoints[idx.RIGHT_SHOULDER]
    nose = keypoints[idx.NOSE]

    # Check visibility
    shoulders_visible = (
        left_shoulder[3] >= threshold and
        right_shoulder[3] >= threshold
    )

    if not shoulders_visible:
        return None

    # Calculate shoulder width (in normalized coords)
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

    # Nose position relative to shoulders
    nose_visible = nose[3] >= threshold

    if shoulder_width < 0.05:
        # Very narrow shoulders = profile or back view
        if nose_visible:
            return TagItem(text="profile view", confidence=0.8, category="pose:orientation")
        else:
            return TagItem(text="back view", confidence=0.7, category="pose:orientation")
    elif shoulder_width > 0.15:
        # Wide shoulders = facing camera
        if nose_visible:
            return TagItem(text="facing camera", confidence=0.85, category="pose:orientation")
        else:
            return TagItem(text="facing away", confidence=0.7, category="pose:orientation")
    else:
        # Medium = three-quarter view
        return TagItem(text="three-quarter view", confidence=0.75, category="pose:orientation")


def _analyze_arms(
    keypoints: np.ndarray,
    idx: KeypointIndices,
    threshold: float
) -> list[TagItem]:
    """Analyze arm positions."""
    tags = []

    left_shoulder = keypoints[idx.LEFT_SHOULDER]
    right_shoulder = keypoints[idx.RIGHT_SHOULDER]
    left_elbow = keypoints[idx.LEFT_ELBOW]
    right_elbow = keypoints[idx.RIGHT_ELBOW]
    left_wrist = keypoints[idx.LEFT_WRIST]
    right_wrist = keypoints[idx.RIGHT_WRIST]
    left_hip = keypoints[idx.LEFT_HIP]
    right_hip = keypoints[idx.RIGHT_HIP]

    # Check if arms are visible
    left_arm_visible = (
        left_shoulder[3] >= threshold and
        left_elbow[3] >= threshold and
        left_wrist[3] >= threshold
    )
    right_arm_visible = (
        right_shoulder[3] >= threshold and
        right_elbow[3] >= threshold and
        right_wrist[3] >= threshold
    )

    if not left_arm_visible and not right_arm_visible:
        return tags

    # Check for raised arms (wrist above shoulder)
    left_raised = left_arm_visible and left_wrist[1] < left_shoulder[1] - 0.1
    right_raised = right_arm_visible and right_wrist[1] < right_shoulder[1] - 0.1

    if left_raised and right_raised:
        tags.append(TagItem(text="arms raised", confidence=0.85, category="pose:arms"))
    elif left_raised or right_raised:
        tags.append(TagItem(text="one arm raised", confidence=0.8, category="pose:arms"))

    # Check for hands on hips (wrist near hip level, elbow out)
    hips_visible = left_hip[3] >= threshold and right_hip[3] >= threshold
    if hips_visible:
        left_on_hip = (
            left_arm_visible and
            abs(left_wrist[1] - left_hip[1]) < 0.1 and
            left_elbow[0] < left_shoulder[0]
        )
        right_on_hip = (
            right_arm_visible and
            abs(right_wrist[1] - right_hip[1]) < 0.1 and
            right_elbow[0] > right_shoulder[0]
        )

        if left_on_hip and right_on_hip:
            tags.append(TagItem(text="hands on hips", confidence=0.8, category="pose:arms"))
        elif left_on_hip or right_on_hip:
            tags.append(TagItem(text="hand on hip", confidence=0.75, category="pose:arms"))

    # Check for crossed arms (wrists near opposite shoulders)
    if left_arm_visible and right_arm_visible:
        left_crossed = left_wrist[0] > (left_shoulder[0] + right_shoulder[0]) / 2
        right_crossed = right_wrist[0] < (left_shoulder[0] + right_shoulder[0]) / 2

        if left_crossed and right_crossed:
            wrists_close_to_body = (
                abs(left_wrist[1] - left_shoulder[1]) < 0.2 and
                abs(right_wrist[1] - right_shoulder[1]) < 0.2
            )
            if wrists_close_to_body:
                tags.append(TagItem(text="arms crossed", confidence=0.75, category="pose:arms"))

    # Default: arms at sides
    if not tags:
        arms_down = (
            (left_arm_visible and left_wrist[1] > left_hip[1] if hips_visible else left_wrist[1] > 0.6) or
            (right_arm_visible and right_wrist[1] > right_hip[1] if hips_visible else right_wrist[1] > 0.6)
        )
        if arms_down:
            tags.append(TagItem(text="arms at sides", confidence=0.7, category="pose:arms"))

    return tags


def _analyze_stance(
    keypoints: np.ndarray,
    idx: KeypointIndices,
    threshold: float
) -> Optional[TagItem]:
    """Analyze body stance (standing, sitting, etc.)."""
    left_hip = keypoints[idx.LEFT_HIP]
    right_hip = keypoints[idx.RIGHT_HIP]
    left_knee = keypoints[idx.LEFT_KNEE]
    right_knee = keypoints[idx.RIGHT_KNEE]
    left_ankle = keypoints[idx.LEFT_ANKLE]
    right_ankle = keypoints[idx.RIGHT_ANKLE]

    # Check visibility
    hips_visible = left_hip[3] >= threshold and right_hip[3] >= threshold
    knees_visible = left_knee[3] >= threshold and right_knee[3] >= threshold
    ankles_visible = left_ankle[3] >= threshold and right_ankle[3] >= threshold

    if not hips_visible:
        return None

    if knees_visible and ankles_visible:
        # Calculate leg angles
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        # Sitting: knees bent significantly, hips and knees at similar height
        hip_knee_diff = knee_y - hip_y
        knee_ankle_diff = ankle_y - knee_y

        if hip_knee_diff < 0.05 and knee_ankle_diff > 0.1:
            return TagItem(text="sitting", confidence=0.8, category="pose:stance")

        # Standing: relatively straight legs
        if hip_knee_diff > 0.1 and knee_ankle_diff > 0.1:
            # Check if weight on one leg (asymmetric)
            knee_diff = abs(left_knee[1] - right_knee[1])
            if knee_diff > 0.05:
                return TagItem(text="standing relaxed", confidence=0.75, category="pose:stance")
            else:
                return TagItem(text="standing straight", confidence=0.8, category="pose:stance")

    elif knees_visible:
        # Only upper body and knees visible
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2

        if knee_y - hip_y < 0.05:
            return TagItem(text="sitting", confidence=0.7, category="pose:stance")

    # Can't determine stance well
    return TagItem(text="standing", confidence=0.6, category="pose:stance")


def _analyze_head(
    keypoints: np.ndarray,
    idx: KeypointIndices,
    threshold: float
) -> Optional[TagItem]:
    """Analyze head tilt and gaze direction."""
    nose = keypoints[idx.NOSE]
    left_ear = keypoints[idx.LEFT_EAR]
    right_ear = keypoints[idx.RIGHT_EAR]
    left_eye = keypoints[idx.LEFT_EYE]
    right_eye = keypoints[idx.RIGHT_EYE]

    # Check visibility
    if nose[3] < threshold:
        return None

    ears_visible = left_ear[3] >= threshold and right_ear[3] >= threshold
    eyes_visible = left_eye[3] >= threshold and right_eye[3] >= threshold

    if ears_visible:
        # Check head tilt based on ear heights
        ear_diff = left_ear[1] - right_ear[1]
        if abs(ear_diff) > 0.03:
            if ear_diff > 0:
                return TagItem(text="head tilted right", confidence=0.7, category="pose:head")
            else:
                return TagItem(text="head tilted left", confidence=0.7, category="pose:head")

    if eyes_visible:
        # Eye line angle
        eye_diff = left_eye[1] - right_eye[1]
        if abs(eye_diff) > 0.02:
            if eye_diff > 0:
                return TagItem(text="head tilted", confidence=0.65, category="pose:head")

    return None


def _analyze_pose_style(
    keypoints: np.ndarray,
    idx: KeypointIndices,
    threshold: float
) -> Optional[TagItem]:
    """Determine overall pose style/mood."""
    # Count visible keypoints
    visible_count = np.sum(keypoints[:, 3] >= threshold)

    if visible_count < 10:
        return TagItem(text="partial view", confidence=0.7, category="pose:style")

    # Check for symmetry (formal vs casual)
    left_shoulder = keypoints[idx.LEFT_SHOULDER]
    right_shoulder = keypoints[idx.RIGHT_SHOULDER]
    left_hip = keypoints[idx.LEFT_HIP]
    right_hip = keypoints[idx.RIGHT_HIP]

    if all(kp[3] >= threshold for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        hip_diff = abs(left_hip[1] - right_hip[1])

        if shoulder_diff < 0.02 and hip_diff < 0.02:
            return TagItem(text="formal stance", confidence=0.7, category="pose:style")
        elif shoulder_diff > 0.04 or hip_diff > 0.04:
            return TagItem(text="casual stance", confidence=0.7, category="pose:style")

    # Full body visible
    if visible_count > 25:
        return TagItem(text="full body pose", confidence=0.8, category="pose:style")

    return None
