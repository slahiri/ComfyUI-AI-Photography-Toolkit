"""Instant taggers that don't require ML models - Color, Blur, etc."""

import colorsys
from collections import Counter
from typing import Optional

import numpy as np
from PIL import Image

from .base import BaseTagger, TagItem, TaggerResult


# Color name mapping for common colors
COLOR_NAMES = {
    # Reds
    (255, 0, 0): "red",
    (220, 20, 60): "crimson",
    (178, 34, 34): "firebrick",
    (139, 0, 0): "dark red",
    # Oranges
    (255, 165, 0): "orange",
    (255, 140, 0): "dark orange",
    (255, 127, 80): "coral",
    # Yellows
    (255, 255, 0): "yellow",
    (255, 215, 0): "gold",
    (240, 230, 140): "khaki",
    # Greens
    (0, 128, 0): "green",
    (34, 139, 34): "forest green",
    (0, 100, 0): "dark green",
    (144, 238, 144): "light green",
    (0, 255, 127): "spring green",
    (46, 139, 87): "sea green",
    # Cyans
    (0, 255, 255): "cyan",
    (0, 139, 139): "dark cyan",
    (32, 178, 170): "light sea green",
    (64, 224, 208): "turquoise",
    # Blues
    (0, 0, 255): "blue",
    (0, 0, 139): "dark blue",
    (65, 105, 225): "royal blue",
    (70, 130, 180): "steel blue",
    (135, 206, 235): "sky blue",
    (173, 216, 230): "light blue",
    (0, 0, 128): "navy",
    # Purples
    (128, 0, 128): "purple",
    (148, 0, 211): "dark violet",
    (238, 130, 238): "violet",
    (255, 0, 255): "magenta",
    (219, 112, 147): "pale violet red",
    (221, 160, 221): "plum",
    # Pinks
    (255, 192, 203): "pink",
    (255, 105, 180): "hot pink",
    (255, 20, 147): "deep pink",
    # Browns
    (165, 42, 42): "brown",
    (139, 69, 19): "saddle brown",
    (160, 82, 45): "sienna",
    (210, 180, 140): "tan",
    (245, 222, 179): "wheat",
    (222, 184, 135): "burlywood",
    # Grays
    (128, 128, 128): "gray",
    (169, 169, 169): "dark gray",
    (192, 192, 192): "silver",
    (211, 211, 211): "light gray",
    (105, 105, 105): "dim gray",
    # Black & White
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (245, 245, 245): "white smoke",
    (255, 250, 250): "snow",
    # Beiges/Creams
    (255, 248, 220): "cornsilk",
    (255, 235, 205): "blanched almond",
    (250, 235, 215): "antique white",
    (255, 228, 196): "bisque",
    (255, 222, 173): "navajo white",
}


def _closest_color_name(rgb: tuple[int, int, int]) -> str:
    """Find the closest named color."""
    min_dist = float('inf')
    closest = "unknown"

    for color_rgb, name in COLOR_NAMES.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
        if dist < min_dist:
            min_dist = dist
            closest = name

    return closest


def _get_color_temperature(rgb: tuple[int, int, int]) -> str:
    """Determine if color is warm or cool."""
    r, g, b = rgb
    # Warm colors have more red/yellow, cool colors have more blue
    warmth = (r * 1.0 + g * 0.5) - (b * 1.0 + g * 0.3)

    if warmth > 100:
        return "warm"
    elif warmth < -100:
        return "cool"
    else:
        return "neutral"


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert RGB to HSV."""
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hsv(r, g, b)


class ColorAnalyzer(BaseTagger):
    """
    Instant color analysis without ML models.

    Extracts dominant colors, color temperature, and palette characteristics.
    """

    TAGGER_NAME = "color"

    def __init__(
        self,
        num_colors: int = 5,
        quality: int = 10,
        **kwargs
    ):
        """
        Initialize color analyzer.

        Args:
            num_colors: Number of dominant colors to extract
            quality: Sampling quality (1=best, higher=faster)
        """
        super().__init__(**kwargs)
        self.num_colors = num_colors
        self.quality = quality

    def load(self) -> None:
        """No model to load."""
        self._is_loaded = True

    def _extract_palette(self, image: Image.Image) -> list[tuple[int, int, int]]:
        """Extract dominant colors using k-means-like approach."""
        # Resize for speed
        img = image.copy()
        img.thumbnail((150, 150))

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get pixels
        pixels = list(img.getdata())

        # Sample pixels
        if self.quality > 1:
            pixels = pixels[::self.quality]

        # Count colors (quantized to reduce noise)
        quantized = []
        for r, g, b in pixels:
            # Quantize to 32 levels
            qr = (r // 8) * 8
            qg = (g // 8) * 8
            qb = (b // 8) * 8
            quantized.append((qr, qg, qb))

        # Get most common
        counter = Counter(quantized)
        return [color for color, _ in counter.most_common(self.num_colors)]

    def tag(self, image: Image.Image) -> TaggerResult:
        """Analyze image colors."""
        if not self.is_loaded:
            self.load()

        tags = []
        palette = self._extract_palette(image)

        if not palette:
            return TaggerResult(tags=[], metadata={})

        # Analyze dominant color
        dominant = palette[0]
        dominant_name = _closest_color_name(dominant)
        tags.append(TagItem(
            text=f"{dominant_name} tones",
            confidence=0.9,
            category="color"
        ))

        # Color temperature (based on overall image)
        img_array = np.array(image.convert("RGB"))
        avg_color = tuple(int(x) for x in img_array.mean(axis=(0, 1)))
        temp = _get_color_temperature(avg_color)

        if temp == "warm":
            tags.append(TagItem(text="warm tones", confidence=0.85, category="color"))
            tags.append(TagItem(text="warm lighting", confidence=0.8, category="lighting"))
        elif temp == "cool":
            tags.append(TagItem(text="cool tones", confidence=0.85, category="color"))
            tags.append(TagItem(text="cool lighting", confidence=0.8, category="lighting"))
        else:
            tags.append(TagItem(text="neutral tones", confidence=0.8, category="color"))

        # Check for high/low saturation
        hsv_values = [_rgb_to_hsv(c) for c in palette]
        avg_saturation = sum(h[1] for h in hsv_values) / len(hsv_values)
        avg_value = sum(h[2] for h in hsv_values) / len(hsv_values)

        if avg_saturation > 0.6:
            tags.append(TagItem(text="vibrant colors", confidence=0.85, category="color"))
            tags.append(TagItem(text="saturated", confidence=0.8, category="color"))
        elif avg_saturation < 0.2:
            tags.append(TagItem(text="muted colors", confidence=0.85, category="color"))
            tags.append(TagItem(text="desaturated", confidence=0.8, category="color"))

        # Check for high/low key
        if avg_value > 0.7:
            tags.append(TagItem(text="high key", confidence=0.8, category="lighting"))
            tags.append(TagItem(text="bright", confidence=0.75, category="lighting"))
        elif avg_value < 0.3:
            tags.append(TagItem(text="low key", confidence=0.8, category="lighting"))
            tags.append(TagItem(text="dark", confidence=0.75, category="lighting"))

        # Check for monochromatic
        if len(set(_closest_color_name(c) for c in palette[:3])) == 1:
            tags.append(TagItem(text="monochromatic", confidence=0.85, category="color"))

        # Check contrast (difference between lightest and darkest)
        values = [_rgb_to_hsv(c)[2] for c in palette]
        contrast = max(values) - min(values)
        if contrast > 0.6:
            tags.append(TagItem(text="high contrast", confidence=0.8, category="color"))
        elif contrast < 0.2:
            tags.append(TagItem(text="low contrast", confidence=0.8, category="color"))

        return TaggerResult(
            tags=tags,
            metadata={
                "dominant_color": dominant_name,
                "palette": [_closest_color_name(c) for c in palette],
                "temperature": temp,
                "saturation": round(avg_saturation, 2),
                "brightness": round(avg_value, 2),
            }
        )

    def unload(self) -> None:
        """Nothing to unload."""
        self._is_loaded = False


class BlurDetector(BaseTagger):
    """
    Instant blur/sharpness detection using Laplacian variance.

    No ML model required - uses image gradients.
    """

    TAGGER_NAME = "blur"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self) -> None:
        """No model to load."""
        self._is_loaded = True

    def _laplacian_variance(self, image: Image.Image) -> float:
        """Calculate Laplacian variance as sharpness measure."""
        # Convert to grayscale numpy array
        gray = image.convert("L")
        arr = np.array(gray, dtype=np.float64)

        # Laplacian kernel
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])

        # Apply convolution (simplified - just use numpy)
        from scipy import ndimage
        try:
            filtered = ndimage.convolve(arr, laplacian)
            return float(filtered.var())
        except ImportError:
            # Fallback without scipy - manual convolution
            h, w = arr.shape
            result = np.zeros_like(arr)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    result[i, j] = (
                        arr[i-1, j] + arr[i+1, j] +
                        arr[i, j-1] + arr[i, j+1] -
                        4 * arr[i, j]
                    )
            return float(result.var())

    def tag(self, image: Image.Image) -> TaggerResult:
        """Detect blur/sharpness in image."""
        if not self.is_loaded:
            self.load()

        # Resize for consistent measurement
        img = image.copy()
        img.thumbnail((500, 500))

        variance = self._laplacian_variance(img)

        tags = []

        # Thresholds based on empirical testing
        if variance > 500:
            tags.append(TagItem(text="sharp", confidence=0.9, category="focus"))
            tags.append(TagItem(text="in focus", confidence=0.85, category="focus"))
            tags.append(TagItem(text="detailed", confidence=0.8, category="quality"))
        elif variance > 200:
            tags.append(TagItem(text="sharp", confidence=0.8, category="focus"))
            tags.append(TagItem(text="good focus", confidence=0.75, category="focus"))
        elif variance > 100:
            tags.append(TagItem(text="slightly soft", confidence=0.75, category="focus"))
        elif variance > 50:
            tags.append(TagItem(text="soft focus", confidence=0.8, category="focus"))
            tags.append(TagItem(text="shallow depth of field", confidence=0.7, category="focus"))
        else:
            tags.append(TagItem(text="blurry", confidence=0.85, category="focus"))
            tags.append(TagItem(text="out of focus", confidence=0.8, category="focus"))

        # Add bokeh tag for moderate blur with high variance in some areas
        # (indicates intentional shallow DOF)
        if 50 < variance < 300:
            tags.append(TagItem(text="bokeh", confidence=0.6, category="focus"))

        return TaggerResult(
            tags=tags,
            metadata={
                "sharpness_score": round(variance, 2),
                "is_sharp": variance > 200,
            }
        )

    def unload(self) -> None:
        """Nothing to unload."""
        self._is_loaded = False


class PhotographyTagger(BaseTagger):
    """
    Photography-specific tags based on image analysis.

    Combines multiple heuristics to generate photography-relevant tags.
    """

    TAGGER_NAME = "photography"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._color_analyzer = ColorAnalyzer()
        self._blur_detector = BlurDetector()

    def load(self) -> None:
        """Load sub-analyzers."""
        self._color_analyzer.load()
        self._blur_detector.load()
        self._is_loaded = True

    def _detect_aspect_ratio(self, image: Image.Image) -> list[TagItem]:
        """Detect aspect ratio, orientation, and format."""
        tags = []
        w, h = image.size
        ratio = w / h

        # Orientation
        if ratio > 1.2:
            tags.append(TagItem(text="landscape orientation", confidence=0.95, category="composition"))
            tags.append(TagItem(text="horizontal frame", confidence=0.9, category="framing"))
            if ratio > 2.2:
                tags.append(TagItem(text="panoramic", confidence=0.9, category="composition"))
                tags.append(TagItem(text="ultra-wide", confidence=0.85, category="framing"))
            elif ratio > 1.7:
                tags.append(TagItem(text="wide shot", confidence=0.8, category="composition"))
                tags.append(TagItem(text="16:9 aspect ratio", confidence=0.85, category="format"))
            elif 1.45 < ratio < 1.55:
                tags.append(TagItem(text="3:2 aspect ratio", confidence=0.85, category="format"))
            elif 1.3 < ratio < 1.4:
                tags.append(TagItem(text="4:3 aspect ratio", confidence=0.85, category="format"))
        elif ratio < 0.8:
            tags.append(TagItem(text="portrait orientation", confidence=0.95, category="composition"))
            tags.append(TagItem(text="vertical frame", confidence=0.9, category="framing"))
            if ratio < 0.5:
                tags.append(TagItem(text="vertical panoramic", confidence=0.8, category="composition"))
                tags.append(TagItem(text="9:16 aspect ratio", confidence=0.85, category="format"))
            elif 0.55 < ratio < 0.7:
                tags.append(TagItem(text="2:3 aspect ratio", confidence=0.85, category="format"))
        else:
            tags.append(TagItem(text="square format", confidence=0.9, category="composition"))
            tags.append(TagItem(text="1:1 aspect ratio", confidence=0.9, category="format"))

        # Resolution tags
        megapixels = (w * h) / 1_000_000
        if megapixels > 20:
            tags.append(TagItem(text="high resolution", confidence=0.9, category="quality"))
            tags.append(TagItem(text="detailed image", confidence=0.85, category="quality"))
        elif megapixels > 8:
            tags.append(TagItem(text="good resolution", confidence=0.8, category="quality"))
        elif megapixels > 2:
            tags.append(TagItem(text="medium resolution", confidence=0.8, category="quality"))
        elif megapixels < 1:
            tags.append(TagItem(text="low resolution", confidence=0.8, category="quality"))

        return tags

    def _detect_composition(self, image: Image.Image) -> list[TagItem]:
        """Detect composition patterns, framing, and shot types."""
        tags = []

        # Convert to grayscale for analysis
        gray = np.array(image.convert("L"), dtype=np.float64)
        h, w = gray.shape

        # Check for centered subject (higher contrast/activity in center)
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        edge_region = np.concatenate([
            gray[:h//4, :].flatten(),
            gray[3*h//4:, :].flatten(),
            gray[:, :w//4].flatten(),
            gray[:, 3*w//4:].flatten()
        ])

        center_var = center_region.var()
        edge_var = edge_region.var()

        if center_var > edge_var * 1.5:
            tags.append(TagItem(text="centered composition", confidence=0.75, category="composition"))
            tags.append(TagItem(text="centered framing", confidence=0.75, category="framing"))

        # Check for rule of thirds (activity at intersection points)
        third_h, third_w = h // 3, w // 3
        thirds_regions = [
            gray[max(0,third_h-20):third_h+20, max(0,third_w-20):third_w+20],
            gray[max(0,third_h-20):third_h+20, max(0,2*third_w-20):2*third_w+20],
            gray[max(0,2*third_h-20):2*third_h+20, max(0,third_w-20):third_w+20],
            gray[max(0,2*third_h-20):2*third_h+20, max(0,2*third_w-20):2*third_w+20],
        ]

        thirds_activity = sum(r.var() for r in thirds_regions if r.size > 0)
        if thirds_activity > center_var * 0.8:
            tags.append(TagItem(text="rule of thirds", confidence=0.6, category="composition"))

        # Check for symmetry (left-right similarity)
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        if min_width > 0:
            symmetry_diff = np.abs(left_half[:, :min_width] - right_half[:, :min_width]).mean()
            if symmetry_diff < 20:
                tags.append(TagItem(text="symmetrical composition", confidence=0.8, category="composition"))
                tags.append(TagItem(text="symmetry", confidence=0.75, category="composition"))

        # Check for top/bottom weighting (horizon line detection)
        top_half_mean = gray[:h//2, :].mean()
        bottom_half_mean = gray[h//2:, :].mean()

        if top_half_mean > bottom_half_mean + 30:
            tags.append(TagItem(text="bright sky", confidence=0.7, category="composition"))
            tags.append(TagItem(text="low horizon", confidence=0.65, category="framing"))
        elif bottom_half_mean > top_half_mean + 30:
            tags.append(TagItem(text="high horizon", confidence=0.65, category="framing"))
            tags.append(TagItem(text="ground emphasis", confidence=0.6, category="composition"))

        # Edge activity for framing type
        top_edge = gray[:h//8, :].var()
        bottom_edge = gray[7*h//8:, :].var()
        left_edge = gray[:, :w//8].var()
        right_edge = gray[:, 7*w//8:].var()

        total_edge_var = top_edge + bottom_edge + left_edge + right_edge
        if total_edge_var < center_var * 0.3:
            tags.append(TagItem(text="subject isolation", confidence=0.7, category="framing"))
            tags.append(TagItem(text="clean background", confidence=0.65, category="composition"))

        # Negative space detection
        img_mean = gray.mean()
        if img_mean > 180:  # Mostly bright
            tags.append(TagItem(text="negative space", confidence=0.7, category="composition"))
            tags.append(TagItem(text="minimalist framing", confidence=0.65, category="framing"))
        elif img_mean < 50:  # Mostly dark
            tags.append(TagItem(text="dark background", confidence=0.7, category="composition"))
            tags.append(TagItem(text="low-key lighting", confidence=0.7, category="lighting"))

        # Detect potential framing types based on activity distribution
        upper_third = gray[:h//3, :].var()
        middle_third = gray[h//3:2*h//3, :].var()
        lower_third = gray[2*h//3:, :].var()

        if upper_third > middle_third * 1.5 and upper_third > lower_third * 1.5:
            tags.append(TagItem(text="top-weighted composition", confidence=0.65, category="framing"))
        elif lower_third > middle_third * 1.5 and lower_third > upper_third * 1.5:
            tags.append(TagItem(text="bottom-weighted composition", confidence=0.65, category="framing"))

        # Add general framing tags
        tags.append(TagItem(text="photographic framing", confidence=0.8, category="framing"))

        return tags

    def _add_photography_style_tags(self) -> list[TagItem]:
        """Add general photography style tags."""
        return [
            TagItem(text="photograph", confidence=0.95, category="style"),
            TagItem(text="photography", confidence=0.9, category="style"),
        ]

    def tag(self, image: Image.Image) -> TaggerResult:
        """Generate photography-focused tags."""
        if not self.is_loaded:
            self.load()

        all_tags = []

        # Get color analysis
        color_result = self._color_analyzer.tag(image)
        all_tags.extend(color_result.tags)

        # Get blur/sharpness
        blur_result = self._blur_detector.tag(image)
        all_tags.extend(blur_result.tags)

        # Aspect ratio and resolution
        all_tags.extend(self._detect_aspect_ratio(image))

        # Composition analysis
        all_tags.extend(self._detect_composition(image))

        # Photography style tags
        all_tags.extend(self._add_photography_style_tags())

        # Combine metadata
        metadata = {
            **color_result.metadata,
            **blur_result.metadata,
            "width": image.size[0],
            "height": image.size[1],
        }

        return TaggerResult(tags=all_tags, metadata=metadata)

    def unload(self) -> None:
        """Unload sub-analyzers."""
        self._color_analyzer.unload()
        self._blur_detector.unload()
        self._is_loaded = False
