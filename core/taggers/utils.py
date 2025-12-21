"""Shared utilities for taggers."""

import json
from pathlib import Path
from typing import Optional

import folder_paths

# Cache for loaded vocabularies
_vocabulary_cache: dict = {}


def get_vocabularies_path() -> Path:
    """Get path to vocabularies.json config file."""
    return Path(__file__).parent / "vocabularies.json"


def load_vocabulary(category: str, subcategory: str = None) -> list[str]:
    """
    Load vocabulary from config file.

    Args:
        category: Main category (e.g., "fashion_clip", "pose", "photography")
        subcategory: Optional subcategory (e.g., "styles", "tops", "orientations")

    Returns:
        List of vocabulary terms
    """
    global _vocabulary_cache

    # Load and cache vocabularies
    if not _vocabulary_cache:
        vocab_path = get_vocabularies_path()
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                _vocabulary_cache = json.load(f)
        else:
            print(f"[SID-Tagger] Warning: vocabularies.json not found at {vocab_path}")
            return []

    # Get category data
    category_data = _vocabulary_cache.get(category, {})

    if subcategory:
        return category_data.get(subcategory, [])
    elif isinstance(category_data, list):
        return category_data
    elif isinstance(category_data, dict):
        # Flatten all subcategories into one list
        all_terms = []
        for key, value in category_data.items():
            if key.startswith("_"):  # Skip metadata keys
                continue
            if isinstance(value, list):
                all_terms.extend(value)
        return all_terms

    return []


def get_all_fashion_vocabulary() -> list[str]:
    """Get complete fashion vocabulary for FashionCLIP."""
    return load_vocabulary("fashion_clip")


def get_pose_vocabulary(subcategory: str = None) -> list[str]:
    """Get pose vocabulary terms."""
    return load_vocabulary("pose", subcategory)


def get_photography_vocabulary(subcategory: str = None) -> list[str]:
    """Get photography vocabulary terms."""
    return load_vocabulary("photography", subcategory)


def get_human_detection_tags() -> set[str]:
    """Get set of tags that indicate human presence."""
    tags = load_vocabulary("human_detection")
    return set(tag.lower() for tag in tags)


def get_tagger_models_dir(tagger_category: str) -> Path:
    """
    Get model directory for a tagger category.

    Args:
        tagger_category: Category name (e.g., "fashion", "pose", "wd14")

    Returns:
        Path to the tagger's model directory
    """
    models_dir = Path(folder_paths.models_dir) / "taggers" / tagger_category
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_hf_model(
    model_id: str,
    tagger_category: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "patrickjohncyh/fashion-clip")
        tagger_category: Category for storage (e.g., "fashion", "pose")
        revision: Specific revision/branch to download
        token: HuggingFace token for gated models

    Returns:
        Path to the downloaded model directory
    """
    from huggingface_hub import snapshot_download

    models_dir = get_tagger_models_dir(tagger_category)
    model_name = model_id.split("/")[-1]
    local_path = models_dir / model_name

    # Check if already downloaded
    if local_path.exists():
        # Verify it's a valid download (has some files)
        if any(local_path.iterdir()):
            return local_path

    print(f"[SID-Tagger] Downloading {model_id}...")

    kwargs = {
        "repo_id": model_id,
        "local_dir": local_path,
    }

    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token

    snapshot_download(**kwargs)

    print(f"[SID-Tagger] Downloaded to {local_path}")
    return local_path


def download_single_file(
    model_id: str,
    filename: str,
    tagger_category: str,
    token: Optional[str] = None,
) -> Path:
    """
    Download a single file from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID
        filename: File to download (e.g., "model.onnx", "best.pt")
        tagger_category: Category for storage
        token: HuggingFace token for gated models

    Returns:
        Path to the downloaded file
    """
    from huggingface_hub import hf_hub_download

    models_dir = get_tagger_models_dir(tagger_category)
    model_name = model_id.split("/")[-1]
    local_dir = models_dir / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / filename

    # Check if already downloaded
    if local_path.exists():
        return local_path

    print(f"[SID-Tagger] Downloading {model_id}/{filename}...")

    hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )

    print(f"[SID-Tagger] Downloaded to {local_path}")
    return local_path
