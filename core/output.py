"""Output cleaning and formatting utilities."""

import re


def clean_caption(text: str) -> str:
    """
    Clean and normalize caption output.

    Removes:
    - Thinking tags (<think>...</think>)
    - Markdown formatting
    - Common LLM prefixes ("Here's the caption:", etc.)
    - Extra whitespace

    Args:
        text: Raw caption from model

    Returns:
        Cleaned caption string
    """
    if not text:
        return ""

    # Remove thinking tags (common in some models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)

    # Remove markdown code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove common LLM prefixes
    prefixes = [
        r"^Here['']?s?\s*(?:is\s*)?(?:the\s*)?(?:a\s*)?(?:caption|description|prompt)[:\s]*",
        r"^(?:The\s*)?(?:image\s*)?(?:shows|depicts|features|contains)[:\s]*",
        r"^(?:This\s*)?(?:image\s*)?(?:is\s*)?(?:a\s*)?(?:photo|picture|photograph)\s*of[:\s]*",
        r"^(?:Caption|Description|Prompt)[:\s]*",
    ]
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def truncate_to_words(text: str, max_words: int) -> str:
    """
    Truncate text to a maximum number of words.

    Attempts to end at a sentence boundary if possible.

    Args:
        text: Input text
        max_words: Maximum word count

    Returns:
        Truncated text
    """
    words = text.split()

    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])

    # Try to end at sentence boundary
    last_period = truncated.rfind(". ")
    if last_period > len(truncated) * 0.7:  # Only if near the end
        return truncated[:last_period + 1]

    return truncated + "..."


def format_as_tags(text: str) -> str:
    """
    Format caption as comma-separated tags.

    Args:
        text: Caption text

    Returns:
        Comma-separated tag string
    """
    # Already comma-separated
    if "," in text and text.count(",") > 3:
        return text

    # Convert sentences to tags
    # Remove periods and split on common delimiters
    text = text.replace(".", ",")
    text = re.sub(r"\s*,\s*", ", ", text)

    return text.strip().rstrip(",")
