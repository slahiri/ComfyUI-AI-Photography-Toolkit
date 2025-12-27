"""Extract tokens from RAG (visual retrieval) results.

Uses pre-computed image embeddings to find similar reference images
and extract their associated prompt terms as tokens.
"""

from typing import List, Optional
from PIL import Image

from .base import ImageToken, TokenType, TokenSource, TokenBatch


def extract_from_rag(
    image: Image.Image,
    top_k: int = 3,
    threshold: float = 0.7,
    categories: Optional[List[str]] = None,
) -> List[ImageToken]:
    """Extract tokens from RAG retrieval.

    Args:
        image: PIL Image to query
        top_k: Number of results per category
        threshold: Minimum similarity threshold
        categories: Optional list of categories to search

    Returns:
        List of ImageToken objects from RAG matches
    """
    try:
        from ..rag import get_rag_retriever
    except ImportError:
        return []

    retriever = get_rag_retriever()

    # Check if RAG is available
    if not retriever.is_available:
        return []

    # Retrieve similar reference images
    results = retriever.retrieve(
        image=image,
        top_k=top_k,
        threshold=threshold,
        categories=categories,
    )

    tokens = []
    for result in results:
        # Map RAG category to canonical category
        canonical_category = retriever.map_to_canonical_category(result.category)

        tokens.append(ImageToken(
            text=result.prompt,
            confidence=result.similarity,
            source=TokenSource.RAG,
            token_type=TokenType.TAG,
            metadata={
                "rag_category": result.category,
                "rag_subcategory": result.subcategory,
                "canonical_category": canonical_category,
                "similarity": result.similarity,
            }
        ))

    return tokens


def extract_from_rag_by_category(
    image: Image.Image,
    top_k: int = 3,
    threshold: float = 0.7,
) -> TokenBatch:
    """Extract tokens from RAG grouped by category.

    Args:
        image: PIL Image to query
        top_k: Number of results per category
        threshold: Minimum similarity threshold

    Returns:
        TokenBatch with RAG tokens
    """
    try:
        from ..rag import get_rag_retriever
    except ImportError:
        return TokenBatch()

    retriever = get_rag_retriever()

    if not retriever.is_available:
        return TokenBatch()

    # Retrieve by category
    results_by_category = retriever.retrieve_by_category(
        image=image,
        top_k=top_k,
        threshold=threshold,
    )

    batch = TokenBatch()

    for category, results in results_by_category.items():
        canonical_category = retriever.map_to_canonical_category(category)

        for result in results:
            batch.add(ImageToken(
                text=result.prompt,
                confidence=result.similarity,
                source=TokenSource.RAG,
                token_type=TokenType.TAG,
                metadata={
                    "rag_category": result.category,
                    "rag_subcategory": result.subcategory,
                    "canonical_category": canonical_category,
                    "similarity": result.similarity,
                }
            ))

    return batch
