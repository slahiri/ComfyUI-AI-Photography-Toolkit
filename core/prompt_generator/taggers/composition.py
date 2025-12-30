"""
Composition analyzer implementation.

Analyzes image composition using rule of thirds, symmetry, leading lines, etc.
"""

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def _image_to_array(image: Any) -> np.ndarray:
    """Convert image to numpy array (RGB, 0-255, shape HxWx3)."""
    if hasattr(image, 'cpu'):
        img_np = image[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        return img_np
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.stack([image, image, image], axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                return image[:, :, :3]
            return image
        elif image.ndim == 4:
            return image[0]
        return image
    else:
        arr = np.array(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=2)
        return arr


def _analyze_rule_of_thirds(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze adherence to rule of thirds.

    Returns scores for each third intersection point.
    """
    h, w = img.shape[:2]
    gray = np.mean(img, axis=2)

    # Third positions
    third_h = h // 3
    third_w = w // 3

    # Calculate interest (variance) at each third intersection
    roi_size = min(h, w) // 10  # Region of interest size
    intersections = [
        (third_h, third_w),           # Top-left
        (third_h, 2 * third_w),       # Top-right
        (2 * third_h, third_w),       # Bottom-left
        (2 * third_h, 2 * third_w),   # Bottom-right
    ]

    scores = []
    for y, x in intersections:
        y1, y2 = max(0, y - roi_size), min(h, y + roi_size)
        x1, x2 = max(0, x - roi_size), min(w, x + roi_size)
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            # High variance = interesting content
            variance = np.var(roi)
            scores.append(variance)
        else:
            scores.append(0)

    # Normalize scores
    max_score = max(scores) if scores else 1
    if max_score == 0:
        max_score = 1

    normalized = [s / max_score for s in scores]

    # Overall rule of thirds score
    # Higher score if content is at intersection points rather than center
    center_roi = gray[h//3:2*h//3, w//3:2*w//3]
    center_variance = np.var(center_roi)

    thirds_avg = np.mean(scores)
    rule_of_thirds_score = min(1.0, thirds_avg / (center_variance + 1))

    return {
        "rule_of_thirds": rule_of_thirds_score,
        "top_left_interest": normalized[0],
        "top_right_interest": normalized[1],
        "bottom_left_interest": normalized[2],
        "bottom_right_interest": normalized[3],
    }


def _analyze_symmetry(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze horizontal and vertical symmetry.
    """
    h, w = img.shape[:2]
    gray = np.mean(img, axis=2)

    # Horizontal symmetry (left vs right)
    left_half = gray[:, :w//2]
    right_half = np.fliplr(gray[:, w//2:w//2 + left_half.shape[1]])

    if left_half.shape == right_half.shape:
        h_diff = np.mean(np.abs(left_half - right_half))
        h_symmetry = max(0, 1 - h_diff / 128)
    else:
        h_symmetry = 0.5

    # Vertical symmetry (top vs bottom)
    top_half = gray[:h//2, :]
    bottom_half = np.flipud(gray[h//2:h//2 + top_half.shape[0], :])

    if top_half.shape == bottom_half.shape:
        v_diff = np.mean(np.abs(top_half - bottom_half))
        v_symmetry = max(0, 1 - v_diff / 128)
    else:
        v_symmetry = 0.5

    return {
        "horizontal_symmetry": h_symmetry,
        "vertical_symmetry": v_symmetry,
        "overall_symmetry": (h_symmetry + v_symmetry) / 2,
    }


def _analyze_leading_lines(img: np.ndarray) -> Dict[str, float]:
    """
    Detect leading lines using edge detection.
    """
    try:
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=min(img.shape[:2]) // 5,
            maxLineGap=20
        )

        if lines is None:
            return {
                "leading_lines_detected": 0,
                "diagonal_lines": 0.0,
                "horizontal_lines": 0.0,
                "vertical_lines": 0.0,
            }

        diagonal_count = 0
        horizontal_count = 0
        vertical_count = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)

            if angle < 15 or angle > 165:
                horizontal_count += 1
            elif 75 < angle < 105:
                vertical_count += 1
            else:
                diagonal_count += 1

        total = len(lines)

        return {
            "leading_lines_detected": min(1.0, total / 20),  # Normalize
            "diagonal_lines": diagonal_count / total if total > 0 else 0,
            "horizontal_lines": horizontal_count / total if total > 0 else 0,
            "vertical_lines": vertical_count / total if total > 0 else 0,
        }

    except ImportError:
        return {
            "leading_lines_detected": 0,
            "diagonal_lines": 0.0,
            "horizontal_lines": 0.0,
            "vertical_lines": 0.0,
        }


def _analyze_balance(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze visual balance/weight distribution.
    """
    h, w = img.shape[:2]
    gray = np.mean(img, axis=2)

    # Calculate visual weight (brightness) in each quadrant
    quadrants = {
        "top_left": gray[:h//2, :w//2],
        "top_right": gray[:h//2, w//2:],
        "bottom_left": gray[h//2:, :w//2],
        "bottom_right": gray[h//2:, w//2:],
    }

    weights = {k: np.mean(v) for k, v in quadrants.items()}

    # Balance score (lower difference = better balance)
    left_weight = (weights["top_left"] + weights["bottom_left"]) / 2
    right_weight = (weights["top_right"] + weights["bottom_right"]) / 2
    h_balance = 1 - abs(left_weight - right_weight) / 255

    top_weight = (weights["top_left"] + weights["top_right"]) / 2
    bottom_weight = (weights["bottom_left"] + weights["bottom_right"]) / 2
    v_balance = 1 - abs(top_weight - bottom_weight) / 255

    return {
        "horizontal_balance": h_balance,
        "vertical_balance": v_balance,
        "overall_balance": (h_balance + v_balance) / 2,
    }


def _analyze_golden_ratio(img: np.ndarray) -> float:
    """
    Check if image dimensions follow golden ratio.
    """
    h, w = img.shape[:2]
    ratio = max(w, h) / min(w, h)
    golden = 1.618

    # Score based on how close to golden ratio
    diff = abs(ratio - golden)
    score = max(0, 1 - diff / 0.5)

    return score


def run_composition(image: Any) -> Dict[str, Any]:
    """
    Analyze image composition.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with composition analysis:
            - rule_of_thirds: 0-1 score
            - symmetry: horizontal, vertical, overall
            - leading_lines: detection scores
            - balance: visual weight distribution
            - golden_ratio: aspect ratio score
    """
    try:
        img = _image_to_array(image)

        result = {}

        # Rule of thirds
        thirds = _analyze_rule_of_thirds(img)
        result.update(thirds)

        # Symmetry
        symmetry = _analyze_symmetry(img)
        result.update(symmetry)

        # Leading lines
        lines = _analyze_leading_lines(img)
        result.update(lines)

        # Balance
        balance = _analyze_balance(img)
        result.update(balance)

        # Golden ratio
        result["golden_ratio"] = _analyze_golden_ratio(img)

        # Overall composition score
        overall = (
            result["rule_of_thirds"] * 0.3 +
            result["overall_symmetry"] * 0.2 +
            result["leading_lines_detected"] * 0.2 +
            result["overall_balance"] * 0.2 +
            result["golden_ratio"] * 0.1
        )
        result["composition_score"] = overall

        print(f"[Composition] Score: {overall:.2f}, RoT: {result['rule_of_thirds']:.2f}")
        return result

    except Exception as e:
        print(f"[Composition] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"composition_score": 0.5}
