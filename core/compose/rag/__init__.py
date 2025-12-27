"""RAG (Retrieval Augmented Generation) module for visual prompt enhancement.

Uses pre-computed image embeddings to find similar reference images
and retrieve their associated prompt terms.

Data files (shipped with package):
- embeddings.npy: Pre-computed image embeddings
- prompts.json: Prompt metadata {prompt, category, subcategory}
"""

from .retriever import RAGRetriever, get_rag_retriever, unload_rag_retriever
from .index import RAGIndex, load_index, get_index_path

__all__ = [
    "RAGRetriever",
    "get_rag_retriever",
    "unload_rag_retriever",
    "RAGIndex",
    "load_index",
    "get_index_path",
]
