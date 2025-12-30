"""
Fashion Color analyzer implementation.

Extracts dominant colors from clothing/fashion regions.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


# Named color mapping
COLOR_NAMES = {
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (128, 128, 128): "gray",
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (255, 255, 0): "yellow",
    (255, 165, 0): "orange",
    (128, 0, 128): "purple",
    (255, 192, 203): "pink",
    (165, 42, 42): "brown",
    (0, 128, 128): "teal",
    (0, 0, 128): "navy",
    (245, 245, 220): "beige",
    (255, 215, 0): "gold",
    (192, 192, 192): "silver",
    (139, 69, 19): "chocolate",
    (255, 99, 71): "coral",
    (64, 224, 208): "turquoise",
    (75, 0, 130): "indigo",
    (230, 230, 250): "lavender",
    (128, 0, 0): "maroon",
    (128, 128, 0): "olive",
    (0, 128, 0): "dark green",
    (255, 127, 80): "salmon",
    (70, 130, 180): "steel blue",
}


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


def _rgb_to_name(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB to nearest color name."""
    min_dist = float('inf')
    closest_name = "unknown"

    for ref_rgb, name in COLOR_NAMES.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name


def _extract_dominant_colors(
    img: np.ndarray,
    n_colors: int = 5,
    mask: np.ndarray = None,
) -> List[Dict[str, Any]]:
    """
    Extract dominant colors using K-means clustering.
    """
    try:
        from sklearn.cluster import KMeans

        # Reshape image
        if mask is not None:
            pixels = img[mask].reshape(-1, 3)
        else:
            pixels = img.reshape(-1, 3)

        if len(pixels) < n_colors:
            return []

        # Sample for speed
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get cluster centers and counts
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)

        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]

        result = []
        total = len(labels)

        for idx in sorted_indices:
            rgb = tuple(colors[idx])
            percentage = counts[idx] / total
            name = _rgb_to_name(rgb)

            result.append({
                "rgb": rgb,
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "name": name,
                "percentage": percentage,
            })

        return result

    except ImportError:
        # Fallback without sklearn
        return _extract_colors_simple(img, n_colors, mask)


def _extract_colors_simple(
    img: np.ndarray,
    n_colors: int = 5,
    mask: np.ndarray = None,
) -> List[Dict[str, Any]]:
    """
    Simple color extraction without clustering.
    """
    if mask is not None:
        pixels = img[mask]
    else:
        pixels = img.reshape(-1, 3)

    if len(pixels) == 0:
        return []

    # Quantize colors to reduce unique values
    quantized = (pixels // 32) * 32  # 8 levels per channel

    # Count unique colors
    unique, counts = np.unique(quantized, axis=0, return_counts=True)

    # Sort by count
    sorted_indices = np.argsort(counts)[::-1][:n_colors]

    result = []
    total = len(pixels)

    for idx in sorted_indices:
        rgb = tuple(unique[idx])
        percentage = counts[idx] / total
        name = _rgb_to_name(rgb)

        result.append({
            "rgb": rgb,
            "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "name": name,
            "percentage": percentage,
        })

    return result


def _estimate_clothing_region(img: np.ndarray) -> np.ndarray:
    """
    Estimate clothing region (middle portion of image).

    Simple heuristic: assume clothing is in the middle-lower portion.
    """
    h, w = img.shape[:2]

    # Create mask for likely clothing region
    # Skip top 1/4 (usually head) and bottom 1/8 (usually cropped)
    mask = np.zeros((h, w), dtype=bool)
    mask[h//4:7*h//8, w//4:3*w//4] = True

    return mask


def _analyze_color_harmony(colors: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze color harmony of the palette.
    """
    if len(colors) < 2:
        return {"harmony_score": 0.5}

    # Convert to HSV for harmony analysis
    rgbs = [c["rgb"] for c in colors]

    try:
        import colorsys

        hues = []
        for rgb in rgbs:
            r, g, b = rgb[0] / 255, rgb[1] / 255, rgb[2] / 255
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hues.append(h * 360)

        # Check for complementary (opposite) colors
        complementary = 0
        for i, h1 in enumerate(hues):
            for h2 in hues[i+1:]:
                diff = abs(h1 - h2)
                if 150 < diff < 210:  # Near 180 degrees
                    complementary += 1

        # Check for analogous (similar) colors
        analogous = 0
        for i, h1 in enumerate(hues):
            for h2 in hues[i+1:]:
                diff = abs(h1 - h2)
                if diff < 30 or diff > 330:  # Within 30 degrees
                    analogous += 1

        n_pairs = len(hues) * (len(hues) - 1) / 2
        if n_pairs > 0:
            complementary_ratio = complementary / n_pairs
            analogous_ratio = analogous / n_pairs
        else:
            complementary_ratio = 0
            analogous_ratio = 0

        # Monochromatic check (all similar hues)
        if len(set(int(h / 30) for h in hues)) == 1:
            monochromatic = 0.8
        else:
            monochromatic = 0.2

        return {
            "harmony_score": (complementary_ratio + analogous_ratio + monochromatic) / 3,
            "complementary": complementary_ratio,
            "analogous": analogous_ratio,
            "monochromatic": monochromatic,
        }

    except Exception:
        return {"harmony_score": 0.5}


def run_fashion_color(image: Any) -> List[Dict[str, Any]]:
    """
    Extract fashion/clothing colors from image.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        List of color dicts with:
            - rgb: (r, g, b) tuple
            - hex: Hex color code
            - name: Color name
            - percentage: Proportion of image
    """
    try:
        img = _image_to_array(image)

        # Estimate clothing region
        mask = _estimate_clothing_region(img)

        # Extract colors from clothing region
        colors = _extract_dominant_colors(img, n_colors=5, mask=mask)

        if not colors:
            # Fallback to whole image
            colors = _extract_dominant_colors(img, n_colors=5)

        # Add harmony analysis
        if colors:
            harmony = _analyze_color_harmony(colors)
            # Add to first color entry for metadata
            colors[0]["harmony"] = harmony

        # Log results
        if colors:
            top_colors = ", ".join(c["name"] for c in colors[:3])
            print(f"[Fashion Color] Top colors: {top_colors}")

        return colors

    except Exception as e:
        print(f"[Fashion Color] Error: {e}")
        import traceback
        traceback.print_exc()
        return []
