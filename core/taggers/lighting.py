"""Lighting and shadow analysis taggers.

Includes:
- IntrinsicLightingTagger: Analyzes lighting direction, quality, shadows
- Uses gradient analysis and histogram features for lighting classification
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import cv2

from .base import BaseTagger, TagItem, TaggerResult


class IntrinsicLightingTagger(BaseTagger):
    """
    Lighting analysis using intrinsic image concepts.

    Analyzes:
    - Light direction (left, right, top, front, back)
    - Light quality (soft/diffused vs hard/direct)
    - Shadow characteristics (soft shadows, hard shadows, no shadows)
    - Highlight detection (specular, rim light)
    - Overall lighting style (natural, studio, dramatic, flat, etc.)

    Uses gradient analysis and histogram features - no ML model required.
    Fast and reliable for photography analysis.
    """

    TAGGER_NAME = "lighting_intrinsic"

    # Lighting style classifications
    LIGHTING_STYLES = {
        "natural_light": "natural lighting",
        "studio_light": "studio lighting",
        "dramatic_light": "dramatic lighting",
        "flat_light": "flat lighting",
        "high_key": "high key lighting",
        "low_key": "low key lighting",
        "backlit": "backlit",
        "rim_light": "rim lighting",
        "split_light": "split lighting",
        "rembrandt": "Rembrandt lighting",
        "butterfly": "butterfly lighting",
        "loop": "loop lighting",
    }

    def __init__(
        self,
        threshold: float = 0.5,
        analyze_shadows: bool = True,
        analyze_highlights: bool = True,
        **kwargs
    ):
        """
        Initialize lighting analyzer.

        Args:
            threshold: Minimum confidence to include tag
            analyze_shadows: Whether to analyze shadow characteristics
            analyze_highlights: Whether to detect highlights/specular
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.analyze_shadows = analyze_shadows
        self.analyze_highlights = analyze_highlights
        self._is_loaded = True  # No model to load

    def load(self) -> None:
        """No model to load - uses CV analysis."""
        self._is_loaded = True

    def _analyze_gradients(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze image gradients to determine light direction."""
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Analyze brightness distribution for light direction
        h, w = gray.shape

        # Split image into regions
        left_half = gray[:, :w//2].mean()
        right_half = gray[:, w//2:].mean()
        top_half = gray[:h//2, :].mean()
        bottom_half = gray[h//2:, :].mean()
        center = gray[h//4:3*h//4, w//4:3*w//4].mean()

        # Determine primary light direction
        results = {}

        # Left vs right
        lr_diff = abs(left_half - right_half) / max(left_half, right_half, 1)
        if lr_diff > 0.1:
            if left_half > right_half:
                results["light from left"] = min(0.5 + lr_diff, 1.0)
            else:
                results["light from right"] = min(0.5 + lr_diff, 1.0)

        # Top vs bottom
        tb_diff = abs(top_half - bottom_half) / max(top_half, bottom_half, 1)
        if tb_diff > 0.1:
            if top_half > bottom_half:
                results["light from above"] = min(0.5 + tb_diff, 1.0)
            else:
                results["light from below"] = min(0.5 + tb_diff, 1.0)

        # Front light (center brighter than edges)
        edge_mean = (left_half + right_half + top_half + bottom_half) / 4
        if center > edge_mean * 1.1:
            results["front lighting"] = min(0.5 + (center - edge_mean) / center, 1.0)

        # Backlight detection (edges brighter than center, especially top)
        if edge_mean > center * 1.1 and top_half > center:
            results["backlit"] = min(0.5 + (edge_mean - center) / edge_mean, 1.0)

        # Gradient strength indicates light directionality
        grad_strength = magnitude.mean() / 255
        results["_gradient_strength"] = grad_strength

        return results

    def _analyze_contrast(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze contrast to determine light quality (soft vs hard)."""
        results = {}

        # Calculate local contrast using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        local_contrast = np.abs(laplacian).mean()

        # Global contrast from histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()

        # Find histogram spread
        nonzero = np.where(hist > 0.001)[0]
        if len(nonzero) > 0:
            hist_spread = nonzero[-1] - nonzero[0]
        else:
            hist_spread = 0

        # Standard deviation of brightness
        std_dev = gray.std()

        # Soft light: low local contrast, moderate global contrast
        # Hard light: high local contrast, high global contrast

        if local_contrast < 15 and std_dev < 50:
            results["soft diffused light"] = 0.8
            results["soft shadows"] = 0.7
        elif local_contrast > 30 and std_dev > 60:
            results["hard direct light"] = 0.8
            results["hard shadows"] = 0.75
        elif local_contrast < 20:
            results["soft light"] = 0.7
        else:
            results["mixed lighting"] = 0.6

        # High key vs low key
        mean_brightness = gray.mean()
        if mean_brightness > 180:
            results["high key lighting"] = min(0.5 + (mean_brightness - 180) / 75, 1.0)
        elif mean_brightness < 80:
            results["low key lighting"] = min(0.5 + (80 - mean_brightness) / 80, 1.0)

        results["_local_contrast"] = local_contrast
        results["_std_dev"] = std_dev
        results["_mean_brightness"] = mean_brightness

        return results

    def _detect_shadows(self, gray: np.ndarray, rgb: np.ndarray) -> Dict[str, float]:
        """Detect and classify shadows in the image."""
        results = {}

        # Find dark regions (potential shadows)
        dark_threshold = gray.mean() * 0.5
        dark_mask = gray < dark_threshold
        dark_ratio = dark_mask.sum() / dark_mask.size

        if dark_ratio < 0.05:
            results["minimal shadows"] = 0.8
        elif dark_ratio > 0.3:
            results["deep shadows"] = 0.75
            results["dramatic shadows"] = 0.7
        else:
            results["visible shadows"] = 0.7

        # Analyze shadow edges (soft vs hard)
        if dark_ratio > 0.05:
            # Dilate and compare to find shadow edges
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(dark_mask.astype(np.uint8), kernel, iterations=2)
            shadow_edge = dilated.astype(float) - dark_mask.astype(float)

            # Check gradient at shadow edges
            grad_at_edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            edge_sharpness = np.abs(grad_at_edge[shadow_edge > 0]).mean() if shadow_edge.sum() > 0 else 0

            if edge_sharpness > 30:
                results["sharp shadow edges"] = 0.7
            else:
                results["soft shadow edges"] = 0.7

        return results

    def _detect_highlights(self, gray: np.ndarray, rgb: np.ndarray) -> Dict[str, float]:
        """Detect specular highlights and rim lighting."""
        results = {}

        # Find bright regions
        bright_threshold = 240
        highlights = gray > bright_threshold
        highlight_ratio = highlights.sum() / highlights.size

        if highlight_ratio > 0.01:
            results["specular highlights"] = min(0.5 + highlight_ratio * 10, 1.0)

        # Check for rim lighting (bright edges around subject)
        h, w = gray.shape
        edge_region = np.zeros_like(gray, dtype=bool)
        edge_width = max(w // 10, 20)
        edge_region[:, :edge_width] = True
        edge_region[:, -edge_width:] = True
        edge_region[:edge_width, :] = True
        edge_region[-edge_width:, :] = True

        center_region = ~edge_region

        edge_brightness = gray[edge_region].mean()
        center_brightness = gray[center_region].mean()

        # Rim light: edges brighter than center
        if edge_brightness > center_brightness * 1.15:
            rim_strength = (edge_brightness - center_brightness) / edge_brightness
            results["rim lighting"] = min(0.5 + rim_strength, 1.0)

        # Check for catchlights (small very bright spots, common in eye/face photos)
        very_bright = gray > 250
        if very_bright.sum() > 0 and very_bright.sum() < gray.size * 0.005:
            # Small bright spots - likely catchlights
            results["catchlights visible"] = 0.6

        return results

    def _classify_lighting_style(
        self,
        direction_results: Dict,
        contrast_results: Dict,
        shadow_results: Dict,
        highlight_results: Dict
    ) -> Dict[str, float]:
        """Classify overall lighting style based on analysis."""
        results = {}

        mean_brightness = contrast_results.get("_mean_brightness", 128)
        local_contrast = contrast_results.get("_local_contrast", 20)
        gradient_strength = direction_results.get("_gradient_strength", 0.1)

        # Natural light characteristics
        if local_contrast < 25 and 100 < mean_brightness < 200:
            results["natural lighting"] = 0.7

        # Studio light - controlled, often with rim
        if "rim lighting" in highlight_results or "specular highlights" in highlight_results:
            if local_contrast > 20:
                results["studio lighting"] = 0.7

        # Dramatic lighting - high contrast, strong shadows
        if "deep shadows" in shadow_results and local_contrast > 30:
            results["dramatic lighting"] = 0.8

        # Flat lighting - low contrast, even
        if local_contrast < 15 and gradient_strength < 0.05:
            results["flat lighting"] = 0.75
            results["even lighting"] = 0.7

        # Split lighting - strong left/right difference
        if "light from left" in direction_results or "light from right" in direction_results:
            side_conf = max(
                direction_results.get("light from left", 0),
                direction_results.get("light from right", 0)
            )
            if side_conf > 0.7 and "deep shadows" in shadow_results:
                results["split lighting"] = 0.75

        # Rembrandt lighting - triangle of light
        if ("light from left" in direction_results or "light from right" in direction_results):
            if "light from above" in direction_results:
                if "visible shadows" in shadow_results:
                    results["Rembrandt lighting"] = 0.6

        # Butterfly lighting - light from above, centered
        if "light from above" in direction_results and "front lighting" in direction_results:
            results["butterfly lighting"] = 0.6

        # Golden hour approximation - warm tones would be detected separately
        if "soft light" in contrast_results and "natural lighting" in results:
            results["golden hour lighting"] = 0.5

        return results

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Analyze image lighting characteristics.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with lighting tags
        """
        tags = []
        metadata = {}

        try:
            # Convert to numpy
            rgb = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            # Run analyses
            direction_results = self._analyze_gradients(gray)
            contrast_results = self._analyze_contrast(gray)

            shadow_results = {}
            if self.analyze_shadows:
                shadow_results = self._detect_shadows(gray, rgb)

            highlight_results = {}
            if self.analyze_highlights:
                highlight_results = self._detect_highlights(gray, rgb)

            # Classify overall style
            style_results = self._classify_lighting_style(
                direction_results, contrast_results, shadow_results, highlight_results
            )

            # Combine all results
            all_results = {}
            for results in [direction_results, contrast_results, shadow_results,
                          highlight_results, style_results]:
                for key, conf in results.items():
                    if not key.startswith("_"):  # Skip internal metrics
                        all_results[key] = conf

            # Create tags
            for text, confidence in all_results.items():
                if confidence >= self.threshold:
                    tags.append(TagItem(
                        text=text,
                        confidence=confidence,
                        category="lighting"
                    ))

            # Store metadata
            metadata = {
                "mean_brightness": float(contrast_results.get("_mean_brightness", 0)),
                "local_contrast": float(contrast_results.get("_local_contrast", 0)),
                "gradient_strength": float(direction_results.get("_gradient_strength", 0)),
            }

            # Determine primary lighting style
            if style_results:
                primary_style = max(style_results.items(), key=lambda x: x[1])
                metadata["primary_lighting_style"] = primary_style[0]
                metadata["lighting_style_confidence"] = primary_style[1]

        except Exception as e:
            print(f"[SID-Lighting] Error: {e}")
            import traceback
            traceback.print_exc()
            metadata["error"] = str(e)

        return TaggerResult(
            tags=tags,
            metadata={"analyzer": "intrinsic_lighting", **metadata}
        )

    def unload(self) -> None:
        """No model to unload."""
        pass


class ShadowDetector(BaseTagger):
    """
    Dedicated shadow detection using deep learning.

    Uses a lightweight model to detect shadow regions and classify
    shadow characteristics.
    """

    TAGGER_NAME = "shadow_detector"

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self._model = None
        self._device = None

    def load(self) -> None:
        """Load shadow detection model."""
        if self._is_loaded:
            return

        # For now, use CV-based approach
        # Could be extended with MTMT or other shadow detection models
        self._is_loaded = True
        print("[SID-Shadow] Shadow detector initialized (CV-based)")

    def tag(self, image: Image.Image) -> TaggerResult:
        """Detect shadows in image."""
        if not self._is_loaded:
            self.load()

        tags = []
        metadata = {}

        try:
            rgb = np.array(image.convert("RGB"))

            # Convert to LAB for better shadow detection
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]

            # Adaptive thresholding for shadow regions
            blur = cv2.GaussianBlur(l_channel, (21, 21), 0)
            shadow_mask = l_channel < (blur * 0.7)

            # Calculate shadow coverage
            shadow_ratio = shadow_mask.sum() / shadow_mask.size

            if shadow_ratio < 0.05:
                tags.append(TagItem(text="no visible shadows", confidence=0.8, category="shadows"))
            elif shadow_ratio < 0.15:
                tags.append(TagItem(text="light shadows", confidence=0.75, category="shadows"))
            elif shadow_ratio < 0.30:
                tags.append(TagItem(text="moderate shadows", confidence=0.75, category="shadows"))
            else:
                tags.append(TagItem(text="heavy shadows", confidence=0.8, category="shadows"))
                tags.append(TagItem(text="strong shadow presence", confidence=0.7, category="shadows"))

            # Analyze shadow distribution
            h, w = shadow_mask.shape
            left_shadows = shadow_mask[:, :w//2].sum() / (h * w // 2)
            right_shadows = shadow_mask[:, w//2:].sum() / (h * w // 2)

            if abs(left_shadows - right_shadows) > 0.1:
                if left_shadows > right_shadows:
                    tags.append(TagItem(text="shadows on left", confidence=0.7, category="shadows"))
                else:
                    tags.append(TagItem(text="shadows on right", confidence=0.7, category="shadows"))

            metadata["shadow_coverage"] = float(shadow_ratio)
            metadata["shadow_distribution"] = {
                "left": float(left_shadows),
                "right": float(right_shadows)
            }

        except Exception as e:
            print(f"[SID-Shadow] Error: {e}")
            metadata["error"] = str(e)

        return TaggerResult(
            tags=tags,
            metadata={"analyzer": "shadow_detector", **metadata}
        )

    def unload(self) -> None:
        """Release resources."""
        if self._model is not None:
            del self._model
            self._model = None
        self._device = None
        super().unload()
