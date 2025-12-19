# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - CV Analyzer

Fast computer vision analysis for human detection and feature extraction.
Uses YOLO for object detection and MediaPipe for face detection.
Replaces slow LLM-based detection with fast CV inference.

Author: Siddhartha Lahiri
License: MIT
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Lazy imports to avoid loading heavy libraries until needed
_yolo_model = None
_mediapipe_face = None
_mediapipe_pose = None


def _get_yolo_model():
    """Lazy load YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # Use nano model for speed
            _yolo_model = YOLO("yolov8n.pt")
            print("[CV-Analyzer] YOLO model loaded")
        except ImportError:
            print("[CV-Analyzer] WARNING: ultralytics not installed. Run: pip install ultralytics")
            return None
        except Exception as e:
            print(f"[CV-Analyzer] WARNING: Failed to load YOLO: {e}")
            return None
    return _yolo_model


def _get_mediapipe_face():
    """MediaPipe disabled - YOLO handles all detection."""
    # MediaPipe has protobuf compatibility issues on Windows
    # YOLO provides sufficient person detection
    return None


class CVAnalyzer:
    """
    Fast CV-based image analysis.

    Provides:
    - Human detection (YOLO + MediaPipe)
    - Feature extraction (colors, composition, lighting)
    - Shot type estimation
    """

    # YOLO class IDs
    PERSON_CLASS = 0

    # Shot type thresholds (face height as % of image height)
    SHOT_THRESHOLDS = {
        "extreme_close_up": 0.7,   # Face > 70% of frame
        "close_up": 0.5,           # Face 50-70%
        "medium_close_up": 0.35,   # Face 35-50%
        "medium_shot": 0.2,        # Face 20-35%
        "medium_full": 0.12,       # Face 12-20%
        "full_shot": 0.05,         # Face 5-12%
        "wide_shot": 0.0,          # Face < 5%
    }

    @classmethod
    def detect_humans(cls, image: np.ndarray) -> Dict[str, Any]:
        """
        Fast human detection using YOLO and MediaPipe.

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            {
                "has_human": bool,
                "human_count": int,
                "person_boxes": [...],  # YOLO detections
                "face_boxes": [...],    # MediaPipe detections
                "detection_time_ms": float
            }
        """
        start_time = time.time()

        result = {
            "has_human": False,
            "human_count": 0,
            "person_boxes": [],
            "face_boxes": [],
            "detection_time_ms": 0
        }

        h, w = image.shape[:2]

        # Try YOLO first (detects full body)
        yolo = _get_yolo_model()
        if yolo is not None:
            try:
                yolo_results = yolo(image, classes=[cls.PERSON_CLASS], verbose=False)
                for r in yolo_results:
                    for box in r.boxes:
                        if float(box.conf) > 0.5:  # Confidence threshold
                            xyxy = box.xyxy[0].cpu().numpy()
                            result["person_boxes"].append({
                                "bbox": xyxy.tolist(),
                                "confidence": float(box.conf),
                                "center": [(xyxy[0] + xyxy[2]) / 2 / w, (xyxy[1] + xyxy[3]) / 2 / h]
                            })
            except Exception as e:
                print(f"[CV-Analyzer] YOLO detection error: {e}")

        # Try MediaPipe face detection (backup + face positions)
        face_detector = _get_mediapipe_face()
        if face_detector is not None:
            try:
                # MediaPipe expects RGB
                if image.shape[2] == 4:  # RGBA
                    image_rgb = image[:, :, :3]
                else:
                    image_rgb = image

                mp_results = face_detector.process(image_rgb)
                if mp_results.detections:
                    for detection in mp_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        result["face_boxes"].append({
                            "bbox_relative": [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
                            "confidence": detection.score[0] if detection.score else 0.5,
                            "center": [bbox.xmin + bbox.width/2, bbox.ymin + bbox.height/2]
                        })
            except Exception as e:
                print(f"[CV-Analyzer] MediaPipe detection error: {e}")

        # Determine human presence
        person_count = len(result["person_boxes"])
        face_count = len(result["face_boxes"])

        result["has_human"] = person_count > 0 or face_count > 0
        result["human_count"] = max(person_count, face_count)
        result["detection_time_ms"] = (time.time() - start_time) * 1000

        return result

    @classmethod
    def extract_features(cls, image: np.ndarray, human_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract visual features from image.

        Args:
            image: RGB numpy array
            human_detection: Result from detect_humans()

        Returns:
            {
                "colors": {"dominant": [...], "temperature": "warm/cool/neutral"},
                "composition": {"type": "centered/rule_of_thirds/..."},
                "shot_type": "close_up/medium/wide/...",
                "lighting": {"brightness": "...", "contrast": "..."},
                "aspect_ratio": "portrait/landscape/square"
            }
        """
        import cv2

        start_time = time.time()
        h, w = image.shape[:2]

        features = {
            "colors": cls._analyze_colors(image),
            "composition": cls._analyze_composition(image, human_detection),
            "shot_type": cls._estimate_shot_type(image, human_detection),
            "lighting": cls._analyze_lighting(image),
            "aspect_ratio": cls._get_aspect_ratio(w, h),
            "feature_time_ms": 0
        }

        features["feature_time_ms"] = (time.time() - start_time) * 1000
        return features

    @classmethod
    def _analyze_colors(cls, image: np.ndarray) -> Dict[str, Any]:
        """Extract dominant colors and color temperature."""
        import cv2

        # Resize for faster processing
        small = cv2.resize(image, (100, 100))

        # Get dominant colors using k-means
        pixels = small.reshape(-1, 3).astype(np.float32)

        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
        except ImportError:
            # Fallback: simple average color + corners
            colors = [
                pixels.mean(axis=0).astype(int),
                small[0, 0],
                small[0, -1],
                small[-1, 0],
                small[-1, -1]
            ]

        # Convert to color names
        color_names = [cls._rgb_to_name(c) for c in colors]

        # Estimate color temperature
        avg_color = np.mean(pixels, axis=0)
        r, g, b = avg_color
        if r > b + 20:
            temperature = "warm"
        elif b > r + 20:
            temperature = "cool"
        else:
            temperature = "neutral"

        return {
            "dominant": list(dict.fromkeys(color_names))[:5],  # Unique, top 5
            "temperature": temperature
        }

    @classmethod
    def _rgb_to_name(cls, rgb) -> str:
        """Convert RGB to approximate color name."""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

        # Check grayscale first
        if abs(r - g) < 20 and abs(g - b) < 20:
            if max(r, g, b) > 200:
                return "white"
            elif max(r, g, b) < 50:
                return "black"
            else:
                return "gray"

        # Check dominant channel
        if r > g and r > b:
            if r > 200 and g > 150:
                return "orange" if g < 200 else "yellow"
            elif r > 150 and b > 100:
                return "pink" if b > 150 else "red"
            return "red" if r > 150 else "brown"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            if r > 150 and b > 150:
                return "purple"
            return "blue"

        # Mixed colors
        if r > 180 and g > 180:
            return "yellow"
        if r > 150 and g > 100 and b < 100:
            return "orange"
        if r > 100 and g < 80 and b > 100:
            return "purple"
        if g > 150 and b > 150:
            return "cyan"

        return "tan" if r > g > b else "gray"

    @classmethod
    def _analyze_composition(cls, image: np.ndarray, human_detection: Dict[str, Any]) -> Dict[str, str]:
        """Analyze image composition."""
        h, w = image.shape[:2]

        # Check if subject is centered
        composition_type = "unknown"

        if human_detection["has_human"]:
            # Use face or person center
            centers = []
            for face in human_detection.get("face_boxes", []):
                centers.append(face["center"])
            for person in human_detection.get("person_boxes", []):
                centers.append(person["center"])

            if centers:
                avg_x = sum(c[0] for c in centers) / len(centers)
                avg_y = sum(c[1] for c in centers) / len(centers)

                # Check if centered (within middle third)
                if 0.33 < avg_x < 0.66 and 0.25 < avg_y < 0.75:
                    composition_type = "centered"
                elif (0.25 < avg_x < 0.4) or (0.6 < avg_x < 0.75):
                    composition_type = "rule_of_thirds"
                else:
                    composition_type = "off_center"
        else:
            composition_type = "scenic"

        return {"type": composition_type}

    @classmethod
    def _estimate_shot_type(cls, image: np.ndarray, human_detection: Dict[str, Any]) -> str:
        """Estimate shot type based on face/body size in frame."""
        h, w = image.shape[:2]

        if not human_detection["has_human"]:
            return "scenic"

        # Use face size if available
        if human_detection["face_boxes"]:
            # Get largest face
            max_face_height = 0
            for face in human_detection["face_boxes"]:
                bbox = face["bbox_relative"]
                face_h = bbox[3]  # Relative height
                max_face_height = max(max_face_height, face_h)

            # Determine shot type from face size
            for shot_type, threshold in cls.SHOT_THRESHOLDS.items():
                if max_face_height >= threshold:
                    return shot_type

        # Fallback: use body size
        if human_detection["person_boxes"]:
            max_body_height = 0
            for person in human_detection["person_boxes"]:
                bbox = person["bbox"]
                body_h = (bbox[3] - bbox[1]) / h
                max_body_height = max(max_body_height, body_h)

            if max_body_height > 0.8:
                return "full_shot"
            elif max_body_height > 0.5:
                return "medium_full"
            elif max_body_height > 0.3:
                return "medium_shot"
            else:
                return "wide_shot"

        return "medium_shot"  # Default

    @classmethod
    def _analyze_lighting(cls, image: np.ndarray) -> Dict[str, str]:
        """Analyze image lighting characteristics."""
        import cv2

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)

        # Brightness
        if mean_brightness < 60:
            brightness = "dark"
        elif mean_brightness < 120:
            brightness = "low"
        elif mean_brightness < 180:
            brightness = "normal"
        elif mean_brightness < 220:
            brightness = "bright"
        else:
            brightness = "overexposed"

        # Contrast
        if std_contrast < 30:
            contrast = "low"
        elif std_contrast < 60:
            contrast = "medium"
        else:
            contrast = "high"

        return {
            "brightness": brightness,
            "contrast": contrast
        }

    @classmethod
    def _get_aspect_ratio(cls, width: int, height: int) -> str:
        """Determine aspect ratio category."""
        ratio = width / height

        if ratio > 1.6:
            return "ultrawide"
        elif ratio > 1.2:
            return "landscape"
        elif ratio > 0.85:
            return "square"
        elif ratio > 0.6:
            return "portrait"
        else:
            return "tall"

    @classmethod
    def analyze(cls, image: np.ndarray, mode: str = "standard") -> Dict[str, Any]:
        """
        Complete image analysis.

        Args:
            image: RGB numpy array
            mode: "quick" (detection only), "standard" (+ features)

        Returns:
            Complete analysis dictionary
        """
        result = {
            "mode": mode,
            "has_human": False,
            "human_count": 0,
            "prompt_type": "scene_only",
            "detection_time_ms": 0,
            "feature_time_ms": 0,
            "total_time_ms": 0
        }

        start_time = time.time()

        # Step 1: Human detection (always)
        detection = cls.detect_humans(image)
        result.update({
            "has_human": detection["has_human"],
            "human_count": detection["human_count"],
            "person_boxes": detection["person_boxes"],
            "face_boxes": detection["face_boxes"],
            "detection_time_ms": detection["detection_time_ms"],
            "prompt_type": "portrait" if detection["has_human"] else "scene_only"
        })

        # Step 2: Feature extraction (standard mode only)
        if mode == "standard":
            features = cls.extract_features(image, detection)
            result.update({
                "colors": features["colors"],
                "composition": features["composition"],
                "shot_type": features["shot_type"],
                "lighting": features["lighting"],
                "aspect_ratio": features["aspect_ratio"],
                "feature_time_ms": features["feature_time_ms"]
            })

        result["total_time_ms"] = (time.time() - start_time) * 1000

        return result


def check_dependencies() -> Dict[str, bool]:
    """Check if required CV dependencies are installed."""
    deps = {
        "ultralytics": False,
        "mediapipe": False,
        "opencv": False,
        "sklearn": False
    }

    try:
        import ultralytics
        deps["ultralytics"] = True
    except ImportError:
        pass

    try:
        import mediapipe
        deps["mediapipe"] = True
    except ImportError:
        pass

    try:
        import cv2
        deps["opencv"] = True
    except ImportError:
        pass

    try:
        from sklearn.cluster import KMeans
        deps["sklearn"] = True
    except ImportError:
        pass

    return deps


if __name__ == "__main__":
    # Test dependencies
    print("Checking CV dependencies...")
    deps = check_dependencies()
    for name, installed in deps.items():
        status = "OK" if installed else "MISSING"
        print(f"  {name}: {status}")

    if not deps["ultralytics"]:
        print("\nInstall YOLO: pip install ultralytics")
    if not deps["mediapipe"]:
        print("Install MediaPipe: pip install mediapipe")
