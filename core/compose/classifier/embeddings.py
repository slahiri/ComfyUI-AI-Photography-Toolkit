"""Embedding-based classification (Layer 4).

Layer 4: Semantic similarity classification using sentence-transformers.
Uses pre-computed category centroids to classify tokens based on meaning,
not just exact keyword matches.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np

from .categories import CanonicalCategory, SubjectDetailType
from .base import TokenClassification
from ..tokenizer.base import ImageToken


# =============================================================================
# Category Examples for Centroid Computation
# =============================================================================

# Representative examples for each category - used to compute centroids
CATEGORY_EXAMPLES: Dict[CanonicalCategory, List[str]] = {
    CanonicalCategory.QUALITY_BOOSTERS: [
        "high quality", "masterpiece", "best quality", "detailed",
        "sharp", "professional", "excellent", "photorealistic",
        "8k", "ultra detailed", "high resolution", "crisp",
    ],
    CanonicalCategory.SUBJECT: [
        "woman", "man", "girl", "boy", "person", "people",
        "car", "vehicle", "animal", "cat", "dog", "bird",
        "object", "no humans", "solo", "couple", "group",
    ],
    CanonicalCategory.SUBJECT_DETAILS: [
        "brown hair", "blue eyes", "long hair", "short hair",
        "dress", "shirt", "pants", "bikini", "jewelry",
        "slim", "athletic", "breasts", "makeup", "tattoo",
        "fair skin", "tan", "freckles", "smile", "serious",
    ],
    CanonicalCategory.ACTION_POSE: [
        "standing", "sitting", "lying down", "walking", "running",
        "looking at viewer", "looking away", "arms crossed",
        "hand on hip", "holding", "dancing", "posing",
        "cowboy shot", "from behind", "profile view",
    ],
    CanonicalCategory.ENVIRONMENT: [
        "indoor", "outdoor", "bedroom", "studio", "street",
        "beach", "forest", "mountain", "city", "sky",
        "window", "door", "wall", "background", "nature",
        "building", "house", "office", "garden", "park",
    ],
    CanonicalCategory.LIGHTING: [
        "soft light", "natural light", "studio lighting",
        "dramatic lighting", "backlight", "rim light",
        "warm tones", "cool tones", "high contrast",
        "shadows", "bright", "dark", "golden hour",
    ],
    CanonicalCategory.STYLE_MEDIUM: [
        "photograph", "digital art", "illustration", "painting",
        "realistic", "anime", "3d render", "sketch",
        "vintage", "modern", "cinematic", "artistic",
    ],
    CanonicalCategory.COMPOSITION: [
        "close up", "medium shot", "full body", "portrait",
        "landscape", "wide shot", "dutch angle", "low angle",
        "centered", "rule of thirds", "symmetry", "front view",
    ],
    CanonicalCategory.TECHNICAL: [
        "blur", "noise", "grain", "watermark", "text",
        "low resolution", "jpeg artifacts", "cropped",
        "nsfw", "safe content", "explicit", "sensitive",
    ],
}

# Subject detail subcategory examples
SUBJECT_DETAIL_EXAMPLES: Dict[SubjectDetailType, List[str]] = {
    SubjectDetailType.HAIR: [
        "long hair", "short hair", "brown hair", "blonde hair",
        "black hair", "red hair", "curly hair", "straight hair",
        "ponytail", "braid", "bun", "bangs", "wavy hair",
    ],
    SubjectDetailType.EYES: [
        "blue eyes", "brown eyes", "green eyes", "looking at viewer",
        "looking away", "closed eyes", "eye contact", "gaze",
    ],
    SubjectDetailType.FACE: [
        "smile", "serious", "expression", "lips", "makeup",
        "lipstick", "blush", "freckles", "beauty mark",
    ],
    SubjectDetailType.BODY: [
        "slim", "athletic", "curvy", "tall", "petite",
        "breasts", "abs", "muscles", "thighs", "shoulders",
    ],
    SubjectDetailType.CLOTHING: [
        "dress", "shirt", "pants", "bikini", "swimsuit",
        "jacket", "skirt", "underwear", "suit", "uniform",
    ],
    SubjectDetailType.SKIN: [
        "fair skin", "tan", "dark skin", "pale", "freckled",
        "smooth skin", "complexion", "asian", "caucasian",
    ],
    SubjectDetailType.ACCESSORIES: [
        "glasses", "jewelry", "necklace", "earrings", "watch",
        "hat", "scarf", "bag", "ring", "bracelet",
    ],
}


class EmbeddingClassifier:
    """Layer 4: Semantic similarity classifier using embeddings.

    Uses sentence-transformers to embed tokens and compare against
    pre-computed category centroids.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.5):
        """Initialize the embedding classifier.

        Args:
            model_name: Sentence-transformers model to use
            threshold: Minimum similarity to accept classification
        """
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.category_centroids: Dict[CanonicalCategory, np.ndarray] = {}
        self.subcategory_centroids: Dict[SubjectDetailType, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the model and centroids.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._compute_centroids()
            self._initialized = True
            return True
        except ImportError:
            # sentence-transformers not installed
            return False
        except Exception as e:
            print(f"Warning: Failed to initialize embedding classifier: {e}")
            return False

    def _compute_centroids(self):
        """Compute category centroids from examples."""
        if self.model is None:
            return

        # Compute main category centroids
        for category, examples in CATEGORY_EXAMPLES.items():
            embeddings = self.model.encode(examples)
            self.category_centroids[category] = np.mean(embeddings, axis=0)

        # Compute subcategory centroids
        for subcategory, examples in SUBJECT_DETAIL_EXAMPLES.items():
            embeddings = self.model.encode(examples)
            self.subcategory_centroids[subcategory] = np.mean(embeddings, axis=0)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def classify(self, token: ImageToken) -> Optional[TokenClassification]:
        """Classify a token using embedding similarity.

        Args:
            token: Token to classify

        Returns:
            TokenClassification if similarity above threshold, None otherwise
        """
        if not self._ensure_initialized():
            return None

        # Embed the token text
        token_embedding = self.model.encode([token.text])[0]

        # Find best matching category
        best_category = None
        best_similarity = 0.0
        all_similarities = {}

        for category, centroid in self.category_centroids.items():
            similarity = self._cosine_similarity(token_embedding, centroid)
            all_similarities[category.value] = round(similarity, 3)

            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        # Check threshold
        if best_similarity < self.threshold:
            return None

        # If subject details, find subcategory
        subcategory = None
        if best_category == CanonicalCategory.SUBJECT_DETAILS:
            subcategory = self._find_subcategory(token_embedding)

        # Find secondary categories (others above threshold * 0.8)
        secondary = [
            cat for cat, centroid in self.category_centroids.items()
            if cat != best_category
            and self._cosine_similarity(token_embedding, centroid) > self.threshold * 0.8
        ]

        return TokenClassification(
            token=token,
            primary_category=best_category,
            secondary_categories=secondary[:2],  # Max 2 secondary
            subcategory=subcategory,
            confidence=best_similarity,
            classifier_layer=4,
            metadata={
                "classification_method": "embedding_similarity",
                "similarity": round(best_similarity, 3),
                "all_similarities": all_similarities,
            }
        )

    def _find_subcategory(self, embedding: np.ndarray) -> Optional[SubjectDetailType]:
        """Find best matching subject detail subcategory."""
        best_subcategory = None
        best_similarity = 0.0

        for subcategory, centroid in self.subcategory_centroids.items():
            similarity = self._cosine_similarity(embedding, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_subcategory = subcategory

        return best_subcategory if best_similarity > 0.4 else None

    def classify_batch(self, tokens: List[ImageToken]) -> List[Optional[TokenClassification]]:
        """Classify multiple tokens efficiently.

        Args:
            tokens: List of tokens to classify

        Returns:
            List of classifications (None for tokens below threshold)
        """
        if not self._ensure_initialized():
            return [None] * len(tokens)

        # Batch encode all tokens
        texts = [t.text for t in tokens]
        embeddings = self.model.encode(texts)

        results = []
        for token, embedding in zip(tokens, embeddings):
            # Find best category
            best_category = None
            best_similarity = 0.0

            for category, centroid in self.category_centroids.items():
                similarity = self._cosine_similarity(embedding, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category

            if best_similarity < self.threshold:
                results.append(None)
                continue

            # Find subcategory if needed
            subcategory = None
            if best_category == CanonicalCategory.SUBJECT_DETAILS:
                subcategory = self._find_subcategory(embedding)

            results.append(TokenClassification(
                token=token,
                primary_category=best_category,
                subcategory=subcategory,
                confidence=best_similarity,
                classifier_layer=4,
                metadata={
                    "classification_method": "embedding_similarity",
                    "similarity": round(best_similarity, 3),
                }
            ))

        return results


# Global instance (lazy initialized)
_embedding_classifier: Optional[EmbeddingClassifier] = None


def get_embedding_classifier() -> EmbeddingClassifier:
    """Get or create the global embedding classifier instance."""
    global _embedding_classifier
    if _embedding_classifier is None:
        _embedding_classifier = EmbeddingClassifier()
    return _embedding_classifier


def classify_by_embedding(token: ImageToken) -> Optional[TokenClassification]:
    """Layer 4: Classify token using embedding similarity.

    Args:
        token: Token to classify

    Returns:
        TokenClassification if similarity above threshold, None otherwise
    """
    classifier = get_embedding_classifier()
    return classifier.classify(token)


def classify_batch_by_embedding(tokens: List[ImageToken]) -> List[Optional[TokenClassification]]:
    """Classify multiple tokens using embeddings (more efficient).

    Args:
        tokens: List of tokens to classify

    Returns:
        List of classifications
    """
    classifier = get_embedding_classifier()
    return classifier.classify_batch(tokens)
