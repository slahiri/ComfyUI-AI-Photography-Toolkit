"""
Semantic inclusion analyzer for prompt validation.

Uses embeddings to determine if detected tags are semantically
reflected in the generated output prompt.
"""

import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_cache_path() -> str:
    """Get cache path for model."""
    cache_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".cache", "sentence_transformers"
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load sentence transformer model."""
    global _model

    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer

        cache_dir = _get_cache_path()
        _model = SentenceTransformer(_model_name, cache_folder=cache_dir)
        print(f"[InclusionAnalyzer] Loaded {_model_name}")
        return _model

    except ImportError:
        print("[InclusionAnalyzer] sentence-transformers not installed, using fallback")
        _model = "not_available"
        return _model
    except Exception as e:
        print(f"[InclusionAnalyzer] Error loading model: {e}")
        _model = "not_available"
        return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    """Split text into overlapping chunks for better matching."""
    words = text.split()
    chunks = []

    # Full text as one chunk
    chunks.append(text)

    # Sentence-level chunks
    sentences = text.replace(".", ".|").replace(",", ",|").split("|")
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            chunks.append(sentence)

    # Sliding window chunks
    step = chunk_size // 2
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 30:
            chunks.append(chunk)

    return chunks


def analyze_inclusion(
    injected_tags: List[Tuple[str, float]],
    output_prompt: str,
    similarity_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Analyze semantic inclusion of tags in output prompt.

    Args:
        injected_tags: List of (tag, confidence) tuples that were injected
        output_prompt: The generated output prompt
        similarity_threshold: Minimum similarity to consider "included"

    Returns:
        Dict with inclusion analysis results
    """
    if not injected_tags or not output_prompt:
        return {
            "available": False,
            "reason": "No tags or output to analyze",
            "inclusion_score": 0.0,
            "tags": [],
        }

    model = _load_model()

    if model == "not_available":
        # Fallback to simple keyword matching
        return _fallback_analysis(injected_tags, output_prompt)

    try:
        # Create expanded tag phrases for better matching
        tag_phrases = []
        for tag, conf in injected_tags:
            # Create variations for better semantic matching
            phrases = _expand_tag_to_phrases(tag)
            tag_phrases.append({
                "tag": tag,
                "confidence": conf,
                "phrases": phrases,
            })

        # Get embeddings for all tag phrases
        all_phrases = []
        phrase_to_tag = {}
        for tp in tag_phrases:
            for phrase in tp["phrases"]:
                all_phrases.append(phrase)
                phrase_to_tag[phrase] = tp["tag"]

        # Get output chunks
        output_chunks = _chunk_text(output_prompt)

        # Encode everything
        phrase_embeddings = model.encode(all_phrases, convert_to_numpy=True)
        chunk_embeddings = model.encode(output_chunks, convert_to_numpy=True)

        # Compute max similarity for each tag
        tag_results = []
        for tp in tag_phrases:
            tag = tp["tag"]
            best_similarity = 0.0
            best_match = ""
            best_phrase = ""

            for phrase in tp["phrases"]:
                phrase_idx = all_phrases.index(phrase)
                phrase_emb = phrase_embeddings[phrase_idx]

                for chunk_idx, chunk_emb in enumerate(chunk_embeddings):
                    sim = _cosine_similarity(phrase_emb, chunk_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = output_chunks[chunk_idx][:100]
                        best_phrase = phrase

            included = best_similarity >= similarity_threshold
            tag_results.append({
                "tag": tag,
                "confidence": round(tp["confidence"], 3),
                "included": included,
                "similarity": round(best_similarity, 3),
                "matched_phrase": best_phrase if included else None,
                "matched_text": best_match[:80] + "..." if included and len(best_match) > 80 else best_match if included else None,
            })

        # Calculate overall inclusion score
        included_count = sum(1 for t in tag_results if t["included"])
        total_count = len(tag_results)

        # Weighted by original confidence
        weighted_included = sum(
            t["confidence"] for t in tag_results if t["included"]
        )
        weighted_total = sum(t["confidence"] for t in tag_results)

        inclusion_score = weighted_included / weighted_total if weighted_total > 0 else 0.0

        return {
            "available": True,
            "method": "semantic_embedding",
            "model": _model_name,
            "similarity_threshold": similarity_threshold,
            "inclusion_score": round(inclusion_score, 3),
            "included_count": included_count,
            "total_count": total_count,
            "included_ratio": round(included_count / total_count, 3) if total_count > 0 else 0.0,
            "tags": tag_results,
            "summary": {
                "included": [t["tag"] for t in tag_results if t["included"]],
                "missing": [t["tag"] for t in tag_results if not t["included"]],
            }
        }

    except Exception as e:
        print(f"[InclusionAnalyzer] Error: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_analysis(injected_tags, output_prompt)


def _expand_tag_to_phrases(tag: str) -> List[str]:
    """
    Expand a tag into multiple phrases for better semantic matching.

    Args:
        tag: Original tag string

    Returns:
        List of phrase variations
    """
    phrases = [tag]

    # Add "with X" and "has X" variations for features
    if any(word in tag.lower() for word in ["hair", "eyes", "skin", "lips"]):
        phrases.append(f"with {tag}")
        phrases.append(f"has {tag}")

    # Add descriptive variations
    tag_lower = tag.lower()

    # Color expansions
    color_synonyms = {
        "blonde": ["golden", "fair", "light", "honey"],
        "brunette": ["brown", "chestnut", "dark"],
        "red hair": ["auburn", "ginger", "copper"],
        "blue eyes": ["azure", "sapphire", "cerulean"],
        "green eyes": ["emerald", "jade"],
        "brown eyes": ["chocolate", "hazel", "amber"],
    }

    for key, synonyms in color_synonyms.items():
        if key in tag_lower:
            for syn in synonyms:
                phrases.append(tag_lower.replace(key.split()[0], syn))

    # Expression expansions
    expression_synonyms = {
        "smile": ["smiling", "grinning", "cheerful", "happy expression"],
        "serious": ["stern", "solemn", "thoughtful"],
        "laugh": ["laughing", "joyful", "amused"],
    }

    for key, synonyms in expression_synonyms.items():
        if key in tag_lower:
            phrases.extend(synonyms)

    # Lighting expansions
    lighting_synonyms = {
        "golden hour": ["warm light", "sunset lighting", "golden light"],
        "soft lighting": ["diffused light", "gentle lighting", "soft light"],
        "dramatic": ["high contrast", "moody", "chiaroscuro"],
        "backlight": ["rim light", "silhouette", "backlit"],
    }

    for key, synonyms in lighting_synonyms.items():
        if key in tag_lower:
            phrases.extend(synonyms)

    # Weather/atmosphere expansions
    atmosphere_synonyms = {
        "bokeh": ["blurred background", "shallow depth", "out of focus background"],
        "fog": ["mist", "haze", "foggy", "misty"],
        "rain": ["rainy", "wet", "raindrops"],
        "wind": ["windy", "windswept", "breeze", "blowing"],
    }

    for key, synonyms in atmosphere_synonyms.items():
        if key in tag_lower:
            phrases.extend(synonyms)

    return list(set(phrases))  # Remove duplicates


def _fallback_analysis(
    injected_tags: List[Tuple[str, float]],
    output_prompt: str,
) -> Dict[str, Any]:
    """
    Fallback analysis using keyword matching with some fuzzy logic.

    Args:
        injected_tags: List of (tag, confidence) tuples
        output_prompt: The generated output prompt

    Returns:
        Dict with basic inclusion analysis
    """
    output_lower = output_prompt.lower()
    tag_results = []

    for tag, conf in injected_tags:
        tag_lower = tag.lower()

        # Check exact match
        exact_match = tag_lower in output_lower

        # Check word-by-word match
        words = tag_lower.replace("_", " ").split()
        word_matches = sum(1 for word in words if word in output_lower)
        word_ratio = word_matches / len(words) if words else 0

        # Check synonyms
        synonym_match = False
        phrases = _expand_tag_to_phrases(tag)
        for phrase in phrases:
            if phrase.lower() in output_lower:
                synonym_match = True
                break

        included = exact_match or word_ratio >= 0.5 or synonym_match
        similarity = 1.0 if exact_match else (0.7 if synonym_match else word_ratio * 0.6)

        tag_results.append({
            "tag": tag,
            "confidence": round(conf, 3),
            "included": included,
            "similarity": round(similarity, 3),
            "matched_phrase": None,
            "matched_text": None,
        })

    included_count = sum(1 for t in tag_results if t["included"])
    total_count = len(tag_results)

    weighted_included = sum(t["confidence"] for t in tag_results if t["included"])
    weighted_total = sum(t["confidence"] for t in tag_results)

    inclusion_score = weighted_included / weighted_total if weighted_total > 0 else 0.0

    return {
        "available": True,
        "method": "keyword_fallback",
        "model": None,
        "similarity_threshold": 0.5,
        "inclusion_score": round(inclusion_score, 3),
        "included_count": included_count,
        "total_count": total_count,
        "included_ratio": round(included_count / total_count, 3) if total_count > 0 else 0.0,
        "tags": tag_results,
        "summary": {
            "included": [t["tag"] for t in tag_results if t["included"]],
            "missing": [t["tag"] for t in tag_results if not t["included"]],
        }
    }


def unload_model():
    """Unload the model from memory."""
    global _model
    if _model is not None and _model != "not_available":
        del _model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _model = None
    print("[InclusionAnalyzer] Model unloaded")
