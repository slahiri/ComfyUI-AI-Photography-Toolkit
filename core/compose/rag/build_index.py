"""Build RAG index from scraped prompt-image pairs.

Run this script offline to generate the embeddings:
    python -m core.compose.rag.build_index --input E:/Prompt/scraped_data --output core/compose/rag/data

Or from the toolkit directory:
    python scripts/build_rag_index.py
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_scraped_data(data_path: Path) -> tuple:
    """Load scraped prompts and image paths.

    Args:
        data_path: Path to scraped_data directory

    Returns:
        Tuple of (prompts_list, images_dir)
    """
    prompts_file = data_path / "prompts_data.json"
    images_dir = data_path / "images"

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten the category structure
    prompts = []
    for category, items in data.items():
        for item in items:
            # Get local image path
            local_path = item.get("local_image_path", "")
            if local_path:
                # Convert relative path to absolute
                image_path = data_path / local_path.replace("scraped_data\\", "").replace("scraped_data/", "")
            else:
                image_path = None

            prompts.append({
                "prompt": item["prompt"],
                "category": category,
                "subcategory": item.get("subcategory", "default"),
                "image_path": str(image_path) if image_path else None,
            })

    return prompts, images_dir


def build_index(
    input_path: str,
    output_path: str,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> None:
    """Build RAG index from scraped data.

    Args:
        input_path: Path to scraped_data directory
        output_path: Path to output index directory
        model_name: CLIP model to use
        batch_size: Batch size for embedding
        device: Device to use (None = auto)
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"[RAG] Building index from {input_path}")
    print(f"[RAG] Output to {output_path}")
    print(f"[RAG] Model: {model_name}")

    # Load scraped data
    prompts, images_dir = load_scraped_data(input_path)
    print(f"[RAG] Found {len(prompts)} prompt-image pairs")

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RAG] Using device: {device}")

    # Load CLIP model
    print("[RAG] Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    # Get embedding dimension
    with torch.no_grad():
        dummy = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt")
        dummy = {k: v.to(device) for k, v in dummy.items()}
        dummy_out = model.get_image_features(**dummy)
        embedding_dim = dummy_out.shape[1]
    print(f"[RAG] Embedding dimension: {embedding_dim}")

    # Process images in batches
    embeddings = []
    valid_prompts = []
    failed = 0

    print("[RAG] Processing images...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="Embedding"):
        batch_prompts = prompts[i:i + batch_size]
        batch_images = []
        batch_valid_prompts = []

        for p in batch_prompts:
            image_path = p.get("image_path")
            if not image_path or not os.path.exists(image_path):
                failed += 1
                continue

            try:
                img = Image.open(image_path).convert("RGB")
                batch_images.append(img)
                batch_valid_prompts.append(p)
            except Exception as e:
                print(f"[RAG] Failed to load {image_path}: {e}")
                failed += 1
                continue

        if not batch_images:
            continue

        # Embed batch
        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                batch_embeddings = outputs.cpu().numpy()

            # Normalize
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + 1e-8
            batch_embeddings = batch_embeddings / norms

            embeddings.append(batch_embeddings)
            valid_prompts.extend(batch_valid_prompts)

        except Exception as e:
            print(f"[RAG] Batch embedding failed: {e}")
            failed += len(batch_images)

        # Close images to free memory
        for img in batch_images:
            img.close()

    if not embeddings:
        print("[RAG] No embeddings generated!")
        return

    # Concatenate embeddings
    all_embeddings = np.vstack(embeddings)
    print(f"[RAG] Generated {len(all_embeddings)} embeddings ({failed} failed)")

    # Remove image_path from prompts (not needed in shipped index)
    for p in valid_prompts:
        if "image_path" in p:
            del p["image_path"]

    # Save index
    from .index import save_index
    save_index(
        index_path=output_path,
        embeddings=all_embeddings,
        prompts=valid_prompts,
        model_name=model_name,
        version="1.0",
    )

    print(f"[RAG] Index built successfully!")
    print(f"[RAG]   - Items: {len(valid_prompts)}")
    print(f"[RAG]   - Embedding dim: {embedding_dim}")
    print(f"[RAG]   - File size: {all_embeddings.nbytes / 1024 / 1024:.2f} MB")

    # Cleanup
    del model, processor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from scraped data")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to scraped_data directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: core/compose/rag/data)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).parent / "data")

    build_index(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
