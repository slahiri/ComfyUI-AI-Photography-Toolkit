"""
Pose estimation tagger.

Uses DWPose or OpenPose for body keypoint detection.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_model = None
_detector = None


def _get_model_path() -> str:
    """Get the path to cache pose model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "pose")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load pose estimation model."""
    global _model, _detector

    if _model is not None:
        return _model, _detector

    try:
        # Try to use DWPose (preferred)
        try:
            from controlnet_aux import DWposeDetector
            _detector = DWposeDetector()
            _model = "dwpose"
            print("[Pose] Loaded DWPose detector")
            return _model, _detector
        except ImportError:
            pass

        # Try OpenPose
        try:
            from controlnet_aux import OpenposeDetector
            _detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            _model = "openpose"
            print("[Pose] Loaded OpenPose detector")
            return _model, _detector
        except ImportError:
            pass

        # Fallback to MediaPipe
        try:
            import mediapipe as mp
            _detector = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
            )
            _model = "mediapipe"
            print("[Pose] Loaded MediaPipe pose detector")
            return _model, _detector
        except ImportError:
            pass

        # No pose detector available
        print("[Pose] No pose detector available")
        print("[Pose] Install with: pip install controlnet-aux or pip install mediapipe")
        _model = "not_available"
        _detector = None
        return _model, _detector

    except Exception as e:
        print(f"[Pose] Error loading model: {e}")
        _model = "not_available"
        _detector = None
        return _model, _detector


def _analyze_pose_keypoints(keypoints: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    """
    Analyze pose keypoints to extract pose information.

    Standard pose keypoint indices (COCO format):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    Returns dict with pose analysis.
    """
    if not keypoints or len(keypoints) < 5:
        return {"detected": False}

    result = {
        "detected": True,
        "keypoint_count": len(keypoints),
        "visible_keypoints": len([k for k in keypoints if k[2] > 0.3]),
    }

    # Helper to safely get keypoint
    def get_kp(idx):
        if idx < len(keypoints):
            kp = keypoints[idx]
            if kp[2] > 0.3:  # confidence threshold
                return kp
        return None

    # Get key body parts
    nose = get_kp(0)
    left_shoulder = get_kp(5)
    right_shoulder = get_kp(6)
    left_elbow = get_kp(7)
    right_elbow = get_kp(8)
    left_wrist = get_kp(9)
    right_wrist = get_kp(10)
    left_hip = get_kp(11)
    right_hip = get_kp(12)
    left_knee = get_kp(13)
    right_knee = get_kp(14)
    left_ankle = get_kp(15)
    right_ankle = get_kp(16)

    # Calculate body center points
    shoulder_y = None
    if left_shoulder and right_shoulder:
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
        result["shoulders_visible"] = True

    hip_y = None
    if left_hip and right_hip:
        hip_y = (left_hip[1] + right_hip[1]) / 2
        hip_x = (left_hip[0] + right_hip[0]) / 2
        result["hips_visible"] = True

    knee_y = None
    if left_knee and right_knee:
        knee_y = (left_knee[1] + right_knee[1]) / 2
        result["knees_visible"] = True
    elif left_knee:
        knee_y = left_knee[1]
        result["knees_visible"] = True
    elif right_knee:
        knee_y = right_knee[1]
        result["knees_visible"] = True

    ankle_y = None
    if left_ankle and right_ankle:
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        result["ankles_visible"] = True
    elif left_ankle:
        ankle_y = left_ankle[1]
        result["ankles_visible"] = True
    elif right_ankle:
        ankle_y = right_ankle[1]
        result["ankles_visible"] = True

    # Determine pose type based on relative body part positions
    pose_type = "unknown"
    pose_confidence = 0.5

    if hip_y is not None and shoulder_y is not None:
        torso_height = abs(hip_y - shoulder_y)

        # Check for lying down (horizontal pose)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            hip_width = abs(left_hip[0] - right_hip[0])
            # If torso is more horizontal than vertical
            if torso_height < shoulder_width * 0.5:
                pose_type = "lying"
                pose_confidence = 0.8
                result["lying"] = pose_confidence

        # Standing vs sitting detection
        if pose_type == "unknown" and ankle_y is not None:
            hip_ankle_dist = abs(ankle_y - hip_y)
            hip_shoulder_dist = abs(hip_y - shoulder_y) if shoulder_y else 0.3

            # Normalize by torso height
            leg_ratio = hip_ankle_dist / hip_shoulder_dist if hip_shoulder_dist > 0 else 0

            if leg_ratio > 1.2:
                # Legs are extended - standing
                pose_type = "standing"
                pose_confidence = min(0.9, 0.6 + leg_ratio * 0.1)
                result["standing"] = pose_confidence
            elif leg_ratio < 0.6:
                # Legs are bent/tucked - sitting
                pose_type = "sitting"
                pose_confidence = 0.8
                result["sitting"] = pose_confidence
            else:
                # Intermediate - could be crouching or kneeling
                if knee_y and abs(knee_y - hip_y) < hip_shoulder_dist * 0.5:
                    pose_type = "kneeling"
                    pose_confidence = 0.7
                    result["kneeling"] = pose_confidence
                else:
                    pose_type = "crouching"
                    pose_confidence = 0.6
                    result["crouching"] = pose_confidence

        elif pose_type == "unknown" and knee_y is not None:
            # No ankles visible, use knees
            hip_knee_dist = abs(knee_y - hip_y) if hip_y else 0
            if hip_knee_dist > 0.2:
                pose_type = "standing"
                pose_confidence = 0.6
                result["standing"] = pose_confidence
            else:
                pose_type = "sitting"
                pose_confidence = 0.6
                result["sitting"] = pose_confidence

    result["pose_type"] = pose_type
    result["pose_confidence"] = pose_confidence

    # Check arm positions
    arms_raised = False
    arms_position = "neutral"

    if shoulder_y:
        # Check if wrists are above shoulders (arms raised)
        if left_wrist and left_wrist[1] < shoulder_y - 0.05:
            arms_raised = True
        if right_wrist and right_wrist[1] < shoulder_y - 0.05:
            arms_raised = True

        # Check if arms are extended outward
        if left_shoulder and left_wrist:
            left_arm_ext = abs(left_wrist[0] - left_shoulder[0])
            if left_arm_ext > 0.15:
                arms_position = "extended"
        if right_shoulder and right_wrist:
            right_arm_ext = abs(right_wrist[0] - right_shoulder[0])
            if right_arm_ext > 0.15:
                arms_position = "extended"

    if arms_raised:
        result["arms_raised"] = True
        result["arms_position"] = "raised"
    else:
        result["arms_position"] = arms_position

    # Check if hands are near face (e.g., touching face, thinking pose)
    if nose and (left_wrist or right_wrist):
        face_touch = False
        if left_wrist:
            dist_to_face = ((left_wrist[0] - nose[0])**2 + (left_wrist[1] - nose[1])**2)**0.5
            if dist_to_face < 0.15:
                face_touch = True
        if right_wrist:
            dist_to_face = ((right_wrist[0] - nose[0])**2 + (right_wrist[1] - nose[1])**2)**0.5
            if dist_to_face < 0.15:
                face_touch = True
        if face_touch:
            result["hand_near_face"] = True

    return result


def _run_mediapipe(detector: Any, pil_image: Image.Image) -> Dict[str, Any]:
    """Run MediaPipe pose detection."""
    import mediapipe as mp

    img_np = np.array(pil_image.convert("RGB"))
    results = detector.process(img_np)

    if not results.pose_landmarks:
        return {"detected": False}

    # Extract keypoints
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((landmark.x, landmark.y, landmark.visibility))

    return _analyze_pose_keypoints(keypoints)


def _run_controlnet_pose(detector: Any, pil_image: Image.Image, model_type: str) -> Dict[str, Any]:
    """Run ControlNet pose detection (DWPose or OpenPose)."""
    try:
        # DWPose can return keypoints directly with output_type parameter
        if model_type == "dwpose":
            # Try to get keypoints directly from DWPose
            try:
                # DWPose __call__ can accept output_type parameter
                # Try calling with detect_resolution for better accuracy
                pose_result = detector(
                    pil_image,
                    detect_resolution=512,
                    output_type="np",
                    include_body=True,
                    include_hand=False,
                    include_face=False,
                )

                # DWPose stores detected poses internally
                if hasattr(detector, 'detected_poses') and detector.detected_poses:
                    poses = detector.detected_poses
                    if len(poses) > 0:
                        # Get first detected person's body keypoints
                        body_keypoints = poses[0].body.keypoints if hasattr(poses[0], 'body') else None
                        if body_keypoints is not None and len(body_keypoints) > 0:
                            # Convert to list of (x, y, confidence) tuples
                            keypoints = []
                            for kp in body_keypoints:
                                if kp is not None:
                                    keypoints.append((float(kp.x), float(kp.y), float(kp.score) if hasattr(kp, 'score') else 1.0))
                                else:
                                    keypoints.append((0, 0, 0))

                            result = _analyze_pose_keypoints(keypoints)
                            result["model"] = model_type
                            result["keypoint_count"] = len([k for k in keypoints if k[2] > 0.3])
                            return result

                # Fallback: try to access pose data differently
                if hasattr(detector, 'pose_estimation') and detector.pose_estimation:
                    # Some versions store it here
                    return _extract_pose_from_detector(detector, model_type)

            except Exception as e:
                print(f"[Pose] DWPose keypoint extraction failed: {e}")

        # Fallback for OpenPose or if DWPose keypoint extraction failed
        # Just run detection and try to infer from the result
        try:
            pose_result = detector(pil_image)

            # Check if we got any pose data back
            if pose_result is not None:
                # For OpenPose, check if candidate keypoints are available
                if hasattr(detector, 'body_estimation'):
                    body_est = detector.body_estimation
                    if hasattr(body_est, 'candidate') and body_est.candidate is not None:
                        candidate = body_est.candidate
                        if len(candidate) > 0:
                            # Convert candidates to keypoints format
                            keypoints = [(c[0], c[1], c[2] if len(c) > 2 else 1.0) for c in candidate[:17]]
                            result = _analyze_pose_keypoints(keypoints)
                            result["model"] = model_type
                            return result

                return {
                    "detected": True,
                    "model": model_type,
                    "pose_image_available": True,
                    "note": "Pose detected but keypoints not extractable",
                }
        except Exception as e:
            print(f"[Pose] Detection error: {e}")

        return {"detected": False, "model": model_type}

    except Exception as e:
        print(f"[Pose] ControlNet detection error: {e}")
        import traceback
        traceback.print_exc()
        return {"detected": False, "error": str(e)}


def _extract_pose_from_detector(detector: Any, model_type: str) -> Dict[str, Any]:
    """Try to extract pose information from detector's internal state."""
    result = {"detected": True, "model": model_type}

    # Try various attributes that might hold pose data
    for attr in ['detected_poses', 'poses', 'keypoints', 'body_keypoints']:
        if hasattr(detector, attr):
            data = getattr(detector, attr)
            if data is not None and len(data) > 0:
                result["keypoint_source"] = attr
                break

    return result


def run_pose(image: Any) -> Dict[str, Any]:
    """
    Run pose estimation on an image.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with pose detection results:
            - detected: bool
            - pose_type: "standing", "sitting", "lying", etc.
            - keypoints: List of detected keypoints
            - body_parts: Dict of visible body parts
    """
    try:
        model_type, detector = _load_model()

        if model_type == "not_available" or detector is None:
            return {"detected": False, "error": "No pose detector available"}

        # Convert ComfyUI tensor to PIL
        if hasattr(image, 'cpu'):
            img_np = image[0].cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            if image.ndim == 4:
                image = image[0]
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=2)
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = Image.fromarray(np.array(image[0]))

        pil_image = pil_image.convert("RGB")

        # Run detection based on model type
        if model_type == "mediapipe":
            result = _run_mediapipe(detector, pil_image)
        elif model_type in ["dwpose", "openpose"]:
            result = _run_controlnet_pose(detector, pil_image, model_type)
        else:
            result = {"detected": False}

        result["model"] = model_type
        print(f"[Pose] Detection: {result.get('detected', False)}, model: {model_type}")
        return result

    except Exception as e:
        print(f"[Pose] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"detected": False, "error": str(e)}


def unload_model():
    """Unload pose model from memory."""
    global _model, _detector
    if _detector is not None:
        del _detector
    _model = None
    _detector = None
    print("[Pose] Model unloaded")
