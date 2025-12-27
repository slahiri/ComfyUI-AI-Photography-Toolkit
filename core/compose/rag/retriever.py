"""RAG retriever - embed images and query the index."""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from .index import RAGIndex, RAGItem, load_index


@dataclass
class RAGResult:
    """A single RAG retrieval result."""
    prompt: str
    category: str
    subcategory: str
    similarity: float
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "category": self.category,
            "subcategory": self.subcategory,
            "similarity": self.similarity,
        }


class RAGRetriever:
    """Retrieves similar reference images and their prompts.

    Uses CLIP for image embedding and cosine similarity for retrieval.
    """

    # Category mapping from scraped data to canonical categories
    CATEGORY_MAP = {
        "medium": "style_medium",
        "subject": "subject_details",
        "clothing": "subject_details",
        "outdoor": "environment",
        "indoor": "environment",
        "camera": "composition",
        "lighting": "lighting",
        "action-pose": "action_pose",
        "post-processing": "style_medium",
        "others": "uncategorized",
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """Initialize the retriever.

        Args:
            model_name: CLIP model to use for embeddings
            device: Device to run model on (None = auto)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._index: Optional[RAGIndex] = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of model and index."""
        if self._initialized:
            return self._index is not None

        # Load index first (no GPU needed)
        self._index = load_index()
        if self._index is None:
            print("[RAG] No index found - RAG disabled")
            self._initialized = True
            return False

        # Load CLIP model
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

            self._initialized = True
            print(f"[RAG] Initialized with {self._index.size} reference images")
            return True

        except ImportError:
            print("[RAG] transformers not available - RAG disabled")
            self._initialized = True
            return False
        except Exception as e:
            print(f"[RAG] Failed to load CLIP model: {e}")
            self._initialized = True
            return False

    def embed_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Embed an image using CLIP.

        Args:
            image: PIL Image to embed

        Returns:
            Embedding vector (512,) or None if failed
        """
        if not self._ensure_initialized():
            return None

        try:
            import torch

            # Preprocess image
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embedding
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy()[0]

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding

        except Exception as e:
            print(f"[RAG] Embedding failed: {e}")
            return None

    def retrieve(
        self,
        image: Image.Image,
        top_k: int = 3,
        threshold: float = 0.7,
        categories: Optional[List[str]] = None,
    ) -> List[RAGResult]:
        """Retrieve similar reference images and their prompts.

        Args:
            image: Query image
            top_k: Number of results per category
            threshold: Minimum similarity threshold
            categories: List of categories to search (None = all)

        Returns:
            List of RAGResult objects
        """
        if not self._ensure_initialized() or self._index is None:
            return []

        # Embed query image
        embedding = self.embed_image(image)
        if embedding is None:
            return []

        results = []

        # Determine categories to search
        if categories is None:
            categories = self._index.get_categories()

        # Search each category
        for category in categories:
            matches = self._index.search(
                query_embedding=embedding,
                top_k=top_k,
                threshold=threshold,
                category_filter=category,
            )

            for idx, similarity, item in matches:
                results.append(RAGResult(
                    prompt=item.prompt,
                    category=item.category,
                    subcategory=item.subcategory,
                    similarity=similarity,
                    image_path=item.image_path,
                ))

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)

        return results

    def retrieve_by_category(
        self,
        image: Image.Image,
        top_k: int = 3,
        threshold: float = 0.7,
    ) -> Dict[str, List[RAGResult]]:
        """Retrieve results grouped by category.

        Args:
            image: Query image
            top_k: Number of results per category
            threshold: Minimum similarity threshold

        Returns:
            Dict mapping category to list of results
        """
        if not self._ensure_initialized() or self._index is None:
            return {}

        # Embed query image
        embedding = self.embed_image(image)
        if embedding is None:
            return {}

        results_by_category: Dict[str, List[RAGResult]] = {}

        # Search each category
        for category in self._index.get_categories():
            matches = self._index.search(
                query_embedding=embedding,
                top_k=top_k,
                threshold=threshold,
                category_filter=category,
            )

            if matches:
                results_by_category[category] = [
                    RAGResult(
                        prompt=item.prompt,
                        category=item.category,
                        subcategory=item.subcategory,
                        similarity=similarity,
                        image_path=item.image_path,
                    )
                    for idx, similarity, item in matches
                ]

        return results_by_category

    def map_to_canonical_category(self, rag_category: str) -> str:
        """Map RAG category to canonical category name."""
        return self.CATEGORY_MAP.get(rag_category, "uncategorized")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if RAG is available (index exists)."""
        if not self._initialized:
            # Quick check without loading model
            index = load_index()
            return index is not None
        return self._index is not None


# Global instance (lazy initialized)
_rag_retriever: Optional[RAGRetriever] = None


def get_rag_retriever() -> RAGRetriever:
    """Get or create the global RAG retriever instance."""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
    return _rag_retriever


def unload_rag_retriever() -> None:
    """Unload the global RAG retriever."""
    global _rag_retriever
    if _rag_retriever is not None:
        _rag_retriever.unload()
        _rag_retriever = None
