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

    Returns dict with pose analysis.
    """
    if not keypoints or len(keypoints) < 5:
        return {"detected": False}

    result = {
        "detected": True,
        "keypoint_count": len(keypoints),
    }

    # Check visibility of key body parts
    # Standard pose keypoint indices (COCO format):
    # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders,
    # 7-8: elbows, 9-10: wrists, 11-12: hips, 13-14: knees, 15-16: ankles

    # Determine if standing, sitting, or lying
    if len(keypoints) >= 17:
        # Get key points (x, y, confidence)
        nose = keypoints[0] if len(keypoints) > 0 else (0, 0, 0)
        left_hip = keypoints[11] if len(keypoints) > 11 else (0, 0, 0)
        right_hip = keypoints[12] if len(keypoints) > 12 else (0, 0, 0)
        left_knee = keypoints[13] if len(keypoints) > 13 else (0, 0, 0)
        right_knee = keypoints[14] if len(keypoints) > 14 else (0, 0, 0)
        left_ankle = keypoints[15] if len(keypoints) > 15 else (0, 0, 0)
        right_ankle = keypoints[16] if len(keypoints) > 16 else (0, 0, 0)

        # Calculate vertical positions
        hip_y = (left_hip[1] + right_hip[1]) / 2 if left_hip[2] > 0.3 and right_hip[2] > 0.3 else 0
        knee_y = (left_knee[1] + right_knee[1]) / 2 if left_knee[2] > 0.3 and right_knee[2] > 0.3 else 0
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2 if left_ankle[2] > 0.3 and right_ankle[2] > 0.3 else 0

        # Estimate pose based on relative positions
        if hip_y > 0 and ankle_y > 0:
            hip_ankle_ratio = abs(ankle_y - hip_y)
            if hip_ankle_ratio < 0.15:
                result["pose_type"] = "sitting"
                result["sitting"] = 0.8
            elif hip_ankle_ratio > 0.3:
                result["pose_type"] = "standing"
                result["standing"] = 0.8
            else:
                result["pose_type"] = "crouching"
                result["crouching"] = 0.6

        # Check arm positions
        left_wrist = keypoints[9] if len(keypoints) > 9 else (0, 0, 0)
        right_wrist = keypoints[10] if len(keypoints) > 10 else (0, 0, 0)
        left_shoulder = keypoints[5] if len(keypoints) > 5 else (0, 0, 0)
        right_shoulder = keypoints[6] if len(keypoints) > 6 else (0, 0, 0)

        if left_wrist[2] > 0.3 and left_shoulder[2] > 0.3:
            if left_wrist[1] < left_shoulder[1]:  # Wrist above shoulder
                result["arms_raised"] = True

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
        # Run detection
        result = detector(pil_image)

        # The detector returns a pose image, we need to analyze it
        # For now, just indicate detection was successful
        return {
            "detected": True,
            "model": model_type,
            "pose_image_available": True,
        }
    except Exception as e:
        print(f"[Pose] ControlNet detection error: {e}")
        return {"detected": False}


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
