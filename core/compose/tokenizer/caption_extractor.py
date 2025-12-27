"""Extract tokens from caption sources (Florence, VLM).

Captions are natural language descriptions that need to be parsed
into sentences, clauses, and phrases for classification.
"""

from typing import Dict, List, Any, Optional, Tuple
import re

from .base import ImageToken, TokenType, TokenSource, TokenBatch
from ..classifier.categories import is_meta_pattern, get_florence_key_category


# Sentence splitting pattern
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# Clause splitting patterns
CLAUSE_SPLIT = re.compile(r'\s*(?:,\s+(?:and|with|while|which|who|that|where)\s+|,\s+|\s+and\s+|\s+with\s+)\s*', re.IGNORECASE)

# Patterns to clean up extracted text
CLEANUP_PATTERNS = [
    (re.compile(r'^(?:she|he|they|it)\s+(?:is|are|has|have)\s+', re.IGNORECASE), ''),
    (re.compile(r'^(?:the|a|an)\s+(?:woman|man|person|image|photo)\s+(?:is|shows|depicts|features)\s+', re.IGNORECASE), ''),
    (re.compile(r'^\s*(?:and|with|also|additionally)\s+', re.IGNORECASE), ''),
]


def clean_text(text: str) -> str:
    """Clean extracted text by removing common prefixes."""
    result = text.strip()
    for pattern, replacement in CLEANUP_PATTERNS:
        result = pattern.sub(replacement, result)
    return result.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []

    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_clauses(sentence: str) -> List[str]:
    """Split a sentence into clauses."""
    if not sentence:
        return []

    clauses = CLAUSE_SPLIT.split(sentence)
    return [c.strip() for c in clauses if c.strip() and len(c.strip()) > 3]


def extract_from_florence_caption(
    metadata: Dict[str, Any],
    split_into_clauses: bool = False
) -> List[ImageToken]:
    """Extract tokens from florence_caption.

    Args:
        metadata: Full metadata dict
        split_into_clauses: If True, split sentences into clauses

    Returns:
        List of ImageToken objects (SENTENCE or PHRASE type)
    """
    tokens = []
    caption = metadata.get("florence_caption", "")

    if not caption or not isinstance(caption, str):
        return tokens

    sentences = split_sentences(caption)

    for sentence in sentences:
        # Check for META patterns
        if is_meta_pattern(sentence):
            tokens.append(ImageToken(
                text=sentence,
                confidence=0.85,
                source=TokenSource.FLORENCE_CAPTION,
                token_type=TokenType.SENTENCE,
                metadata={"filtered": True, "reason": "META pattern"}
            ))
            continue

        if split_into_clauses:
            clauses = split_clauses(sentence)
            for clause in clauses:
                cleaned = clean_text(clause)
                if cleaned and len(cleaned) > 5:
                    tokens.append(ImageToken(
                        text=cleaned,
                        confidence=0.85,
                        source=TokenSource.FLORENCE_CAPTION,
                        token_type=TokenType.PHRASE,
                        metadata={"original_sentence": sentence}
                    ))
        else:
            tokens.append(ImageToken(
                text=sentence,
                confidence=0.85,
                source=TokenSource.FLORENCE_CAPTION,
                token_type=TokenType.SENTENCE
            ))

    return tokens


def extract_from_florence_description(
    metadata: Dict[str, Any],
    split_into_clauses: bool = False
) -> List[ImageToken]:
    """Extract tokens from florence_description.

    florence_description is typically more detailed than florence_caption.
    """
    tokens = []
    description = metadata.get("florence_description", "")

    if not description or not isinstance(description, str):
        return tokens

    sentences = split_sentences(description)

    for sentence in sentences:
        # Check for META patterns
        if is_meta_pattern(sentence):
            tokens.append(ImageToken(
                text=sentence,
                confidence=0.85,
                source=TokenSource.FLORENCE_DESCRIPTION,
                token_type=TokenType.SENTENCE,
                metadata={"filtered": True, "reason": "META pattern"}
            ))
            continue

        if split_into_clauses:
            clauses = split_clauses(sentence)
            for clause in clauses:
                cleaned = clean_text(clause)
                if cleaned and len(cleaned) > 5:
                    tokens.append(ImageToken(
                        text=cleaned,
                        confidence=0.85,
                        source=TokenSource.FLORENCE_DESCRIPTION,
                        token_type=TokenType.PHRASE,
                        metadata={"original_sentence": sentence}
                    ))
        else:
            tokens.append(ImageToken(
                text=sentence,
                confidence=0.85,
                source=TokenSource.FLORENCE_DESCRIPTION,
                token_type=TokenType.SENTENCE
            ))

    return tokens


def extract_from_florence_analyze(metadata: Dict[str, Any]) -> List[ImageToken]:
    """Extract tokens from florence_analyze structured output.

    florence_analyze is a dict of key-value pairs like:
    {"camera_angle": "front view", "clothing": "white dress", ...}

    Each key maps to a category via FLORENCE_KEY_TO_CATEGORY.
    """
    tokens = []
    analyze_data = metadata.get("florence_analyze", {})

    if not isinstance(analyze_data, dict):
        # Sometimes florence_analyze is a string - try to parse it
        if isinstance(analyze_data, str):
            analyze_data = _parse_florence_analyze_string(analyze_data)

    if not analyze_data:
        return tokens

    for key, value in analyze_data.items():
        if not value:
            continue

        # Convert value to string
        value_str = str(value).strip()

        # Skip NA values
        if value_str.upper() in ("NA", "N/A", "NONE", ""):
            continue

        # Get the category this key maps to
        category = get_florence_key_category(key)

        tokens.append(ImageToken(
            text=value_str,
            confidence=0.9,  # florence_analyze is reliable
            source=TokenSource.FLORENCE_ANALYZE,
            token_type=TokenType.KEY_VALUE,
            metadata={
                "key": key,
                "mapped_category": category.value if category else None
            }
        ))

    return tokens


def _parse_florence_analyze_string(text: str) -> Dict[str, str]:
    """Parse florence_analyze when it's a string instead of dict.

    Format: "key1: value1, key2: value2" or "key1: value1; key2: value2"
    """
    result = {}

    # Split by common delimiters
    pairs = re.split(r'[,;]\s*', text)

    for pair in pairs:
        if ':' in pair:
            parts = pair.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip()
                if key and value:
                    result[key] = value

    return result


def extract_from_vlm_description(
    metadata: Dict[str, Any],
    split_into_clauses: bool = False
) -> List[ImageToken]:
    """Extract tokens from VLM description.

    VLM descriptions come from vision-language models like Qwen-VL.
    """
    tokens = []
    description = metadata.get("vlm_description", "")

    if not description or not isinstance(description, str):
        return tokens

    sentences = split_sentences(description)

    for sentence in sentences:
        # Check for META patterns
        if is_meta_pattern(sentence):
            tokens.append(ImageToken(
                text=sentence,
                confidence=0.9,  # VLM is generally reliable
                source=TokenSource.VLM_DESCRIPTION,
                token_type=TokenType.SENTENCE,
                metadata={"filtered": True, "reason": "META pattern"}
            ))
            continue

        if split_into_clauses:
            clauses = split_clauses(sentence)
            for clause in clauses:
                cleaned = clean_text(clause)
                if cleaned and len(cleaned) > 5:
                    tokens.append(ImageToken(
                        text=cleaned,
                        confidence=0.9,
                        source=TokenSource.VLM_DESCRIPTION,
                        token_type=TokenType.PHRASE,
                        metadata={"original_sentence": sentence}
                    ))
        else:
            tokens.append(ImageToken(
                text=sentence,
                confidence=0.9,
                source=TokenSource.VLM_DESCRIPTION,
                token_type=TokenType.SENTENCE
            ))

    return tokens


def extract_from_florence_mixed(metadata: Dict[str, Any]) -> List[ImageToken]:
    """Extract tokens from florence_mixed_caption.

    This combines caption with tags, so we handle it specially.
    Format often includes both prose and tag lists.
    """
    tokens = []
    mixed = metadata.get("florence_mixed_caption", "")

    if not mixed or not isinstance(mixed, str):
        return tokens

    # Split into prose and tag sections
    # Tags are often after double newline or in comma-separated format
    parts = mixed.split('\n\n')

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Check if this looks like a tag list (comma-separated, short items)
        if ',' in part and all(len(item.strip()) < 30 for item in part.split(',')):
            # This is a tag list
            tags = [t.strip() for t in part.split(',') if t.strip()]
            for tag in tags:
                if tag and len(tag) > 1:
                    tokens.append(ImageToken(
                        text=tag,
                        confidence=0.85,
                        source=TokenSource.FLORENCE_MIXED,
                        token_type=TokenType.TAG,
                        metadata={"from_mixed": True}
                    ))
        else:
            # This is prose - treat as sentences
            sentences = split_sentences(part)
            for sentence in sentences:
                if is_meta_pattern(sentence):
                    tokens.append(ImageToken(
                        text=sentence,
                        confidence=0.85,
                        source=TokenSource.FLORENCE_MIXED,
                        token_type=TokenType.SENTENCE,
                        metadata={"filtered": True, "reason": "META pattern"}
                    ))
                else:
                    tokens.append(ImageToken(
                        text=sentence,
                        confidence=0.85,
                        source=TokenSource.FLORENCE_MIXED,
                        token_type=TokenType.SENTENCE
                    ))

    return tokens


def extract_all_caption_tokens(
    metadata: Dict[str, Any],
    split_into_clauses: bool = False,
    sources: Optional[List[str]] = None
) -> TokenBatch:
    """Extract tokens from all caption sources.

    Args:
        metadata: Full metadata dictionary
        split_into_clauses: Whether to split sentences into clauses
        sources: Optional list of sources to extract from.
                 Defaults to all Florence + VLM sources

    Returns:
        TokenBatch containing all extracted tokens
    """
    batch = TokenBatch()
    batch.image_info = metadata.get("image_info", {})

    if sources is None:
        sources = [
            "florence_caption",
            "florence_description",
            "florence_analyze",
            "florence_mixed",
            "vlm_description"
        ]

    for source in sources:
        if source == "florence_caption":
            tokens = extract_from_florence_caption(metadata, split_into_clauses)
        elif source == "florence_description":
            tokens = extract_from_florence_description(metadata, split_into_clauses)
        elif source == "florence_analyze":
            tokens = extract_from_florence_analyze(metadata)
        elif source == "florence_mixed":
            tokens = extract_from_florence_mixed(metadata)
        elif source == "vlm_description":
            tokens = extract_from_vlm_description(metadata, split_into_clauses)
        else:
            continue

        batch.add_all(tokens)

    return batch
