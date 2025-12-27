"""RAG index management - load and query pre-computed embeddings."""

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path

import numpy as np


@dataclass
class RAGItem:
    """A single item in the RAG index."""
    idx: int
    prompt: str
    category: str
    subcategory: str
    embedding: np.ndarray
    image_path: Optional[str] = None


@dataclass
class RAGIndex:
    """Pre-computed RAG index with embeddings and metadata."""
    embeddings: np.ndarray  # Shape: (N, embedding_dim)
    prompts: List[Dict[str, Any]]  # [{prompt, category, subcategory, image_path}, ...]
    model_name: str
    embedding_dim: int
    version: str

    @property
    def size(self) -> int:
        """Number of items in the index."""
        return len(self.prompts)

    def get_item(self, idx: int) -> RAGItem:
        """Get a single item by index."""
        meta = self.prompts[idx]
        return RAGItem(
            idx=idx,
            prompt=meta["prompt"],
            category=meta["category"],
            subcategory=meta.get("subcategory", "default"),
            embedding=self.embeddings[idx],
            image_path=meta.get("image_path"),
        )

    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return list(set(p["category"] for p in self.prompts))

    def get_category_indices(self, category: str) -> List[int]:
        """Get indices for a specific category."""
        return [i for i, p in enumerate(self.prompts) if p["category"] == category]

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        category_filter: Optional[str] = None,
    ) -> List[tuple]:
        """Search for similar items.

        Args:
            query_embedding: Query vector (embedding_dim,)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            category_filter: Only search within this category

        Returns:
            List of (idx, similarity, RAGItem) tuples, sorted by similarity descending
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Get candidate indices
        if category_filter:
            candidates = self.get_category_indices(category_filter)
            if not candidates:
                return []
            candidate_embeddings = self.embeddings[candidates]
        else:
            candidates = list(range(self.size))
            candidate_embeddings = self.embeddings

        # Normalize embeddings (should already be normalized, but ensure)
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
        normalized_embeddings = candidate_embeddings / norms

        # Compute similarities (cosine similarity via dot product of normalized vectors)
        similarities = normalized_embeddings @ query_norm

        # Get top-k indices
        if threshold > 0:
            valid_mask = similarities >= threshold
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                return []
            similarities = similarities[valid_indices]
            candidates = [candidates[i] for i in valid_indices]

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in sorted_indices:
            idx = candidates[i]
            sim = float(similarities[i])
            item = self.get_item(idx)
            results.append((idx, sim, item))

        return results


def get_index_path() -> Path:
    """Get path to the RAG index data directory."""
    return Path(__file__).parent / "data"


def load_index(index_path: Optional[Path] = None) -> Optional[RAGIndex]:
    """Load a pre-computed RAG index.

    Args:
        index_path: Path to index directory (default: core/compose/rag/data)

    Returns:
        RAGIndex if found, None otherwise
    """
    if index_path is None:
        index_path = get_index_path()

    embeddings_file = index_path / "embeddings.npy"
    prompts_file = index_path / "prompts.json"
    config_file = index_path / "config.json"

    # Check if index exists
    if not embeddings_file.exists() or not prompts_file.exists():
        return None

    try:
        # Load embeddings
        embeddings = np.load(str(embeddings_file))

        # Load prompts metadata
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        # Load config (optional)
        config = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

        return RAGIndex(
            embeddings=embeddings,
            prompts=prompts,
            model_name=config.get("model_name", "openai/clip-vit-base-patch32"),
            embedding_dim=embeddings.shape[1],
            version=config.get("version", "unknown"),
        )

    except Exception as e:
        print(f"[RAG] Failed to load index: {e}")
        return None


def save_index(
    index_path: Path,
    embeddings: np.ndarray,
    prompts: List[Dict[str, Any]],
    model_name: str,
    version: str = "1.0",
) -> None:
    """Save a RAG index to disk.

    Args:
        index_path: Directory to save index
        embeddings: Embedding array (N, dim)
        prompts: List of prompt metadata dicts
        model_name: Name of embedding model used
        version: Index version string
    """
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(str(index_path / "embeddings.npy"), embeddings.astype(np.float32))

    # Save prompts
    with open(index_path / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    # Save config
    config = {
        "model_name": model_name,
        "embedding_dim": embeddings.shape[1],
        "num_items": len(prompts),
        "version": version,
    }
    with open(index_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[RAG] Saved index with {len(prompts)} items to {index_path}")
