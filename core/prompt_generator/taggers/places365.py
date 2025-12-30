"""
Places365 scene recognition analyzer.

Uses MIT Places365 model for scene/environment classification.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Lazy imports
_model = None
_labels = None
_transform = None


def _get_model_path() -> str:
    """Get the path to cache Places365 model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "places365")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _download_labels() -> List[str]:
    """Download and parse Places365 category labels."""
    cache_dir = _get_model_path()
    labels_path = os.path.join(cache_dir, "categories_places365.txt")

    if not os.path.exists(labels_path):
        import urllib.request

        url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
        print(f"[Places365] Downloading labels from {url}")
        urllib.request.urlretrieve(url, labels_path)

    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            # Format: /a/abbey 0
            parts = line.strip().split(" ")
            if parts:
                # Extract scene name from path like /a/abbey
                scene = parts[0].split("/")[-1].replace("_", " ")
                labels.append(scene)

    return labels


def _load_model():
    """Load Places365 model."""
    global _model, _labels, _transform

    if _model is not None:
        return _model, _labels, _transform

    try:
        import torch
        import torchvision.models as models
        from torchvision import transforms

        cache_dir = _get_model_path()

        # Download labels
        _labels = _download_labels()
        print(f"[Places365] Loaded {len(_labels)} scene categories")

        # Try to download pre-trained Places365 model
        # Using ResNet18 for efficiency
        model_path = os.path.join(cache_dir, "resnet18_places365.pth")

        if not os.path.exists(model_path):
            # Download from MIT CSAIL server (official host)
            import urllib.request
            # Primary URL: MIT CSAIL server
            urls = [
                "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
                "https://huggingface.co/csail-vision/places365/resolve/main/resnet18_places365.pth.tar",
            ]
            print(f"[Places365] Downloading model...")

            downloaded = False
            for url in urls:
                try:
                    print(f"[Places365] Trying: {url}")
                    urllib.request.urlretrieve(url, model_path)
                    downloaded = True
                    print(f"[Places365] Successfully downloaded from {url}")
                    break
                except Exception as e:
                    print(f"[Places365] Failed from {url}: {e}")
                    continue

            if not downloaded:
                print("[Places365] Could not download model from any source")
                # Fallback: use pre-trained ResNet18 with random head
                _model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                num_classes = 365
                _model.fc = torch.nn.Linear(_model.fc.in_features, num_classes)
                _model.eval()
                print("[Places365] Using ImageNet-pretrained base (not Places365 trained)")

                _transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                return _model, _labels, _transform

        # Load Places365 model
        _model = models.resnet18(num_classes=365)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'module.' prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        _model.load_state_dict(state_dict, strict=False)
        _model.eval()

        if torch.cuda.is_available():
            _model = _model.cuda()

        print("[Places365] Loaded ResNet18 Places365 model")

        # Set up transform
        _transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return _model, _labels, _transform

    except ImportError as e:
        print(f"[Places365] Missing dependency: {e}")
        return None, [], None
    except Exception as e:
        print(f"[Places365] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, [], None


def run_places365(
    image: Any,
    top_k: int = 5,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Classify scene using Places365.

    Args:
        image: Image tensor from ComfyUI or PIL Image
        top_k: Number of top predictions to return
        threshold: Minimum confidence threshold

    Returns:
        Dict mapping scene names to confidence scores
    """
    try:
        model, labels, transform = _load_model()

        if model is None:
            return {}

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
        else:
            pil_image = Image.fromarray(np.array(image[0]))

        # Preprocess
        import torch

        input_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        topk_probs, topk_indices = torch.topk(probs, min(top_k, len(labels)))

        result = {}
        for prob, idx in zip(topk_probs.cpu().numpy(), topk_indices.cpu().numpy()):
            if prob >= threshold and idx < len(labels):
                result[labels[idx]] = float(prob)

        if result:
            best_scene = max(result, key=result.get)
            print(f"[Places365] Top scene: {best_scene} ({result[best_scene]:.2f})")

        return result

    except Exception as e:
        print(f"[Places365] Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_scene_attributes(scene_name: str) -> Dict[str, Any]:
    """
    Get attributes for a scene category.

    Returns dict with:
        - indoor: bool
        - outdoor: bool
        - natural: bool
        - urban: bool
    """
    # Common indoor scenes
    indoor_keywords = [
        "bedroom", "bathroom", "kitchen", "living room", "office",
        "restaurant", "bar", "library", "museum", "gallery",
        "studio", "gym", "classroom", "corridor", "lobby",
    ]

    # Common outdoor scenes
    outdoor_keywords = [
        "beach", "forest", "mountain", "street", "highway",
        "park", "garden", "field", "desert", "ocean",
        "lake", "river", "sky", "rooftop", "bridge",
    ]

    # Natural scenes
    natural_keywords = [
        "forest", "mountain", "beach", "ocean", "lake",
        "river", "field", "meadow", "desert", "jungle",
        "waterfall", "cliff", "valley", "canyon",
    ]

    # Urban scenes
    urban_keywords = [
        "street", "highway", "building", "city", "downtown",
        "alley", "parking", "bridge", "tower", "skyscraper",
    ]

    scene_lower = scene_name.lower()

    return {
        "indoor": any(kw in scene_lower for kw in indoor_keywords),
        "outdoor": any(kw in scene_lower for kw in outdoor_keywords),
        "natural": any(kw in scene_lower for kw in natural_keywords),
        "urban": any(kw in scene_lower for kw in urban_keywords),
    }


def unload_model():
    """Unload Places365 model from memory."""
    global _model, _labels, _transform
    _model = None
    _labels = None
    _transform = None
    print("[Places365] Model unloaded")
