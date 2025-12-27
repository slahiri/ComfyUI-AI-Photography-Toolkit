"""Validation rules for prompt checking.

Defines patterns for detecting issues like conflicts, duplicates,
and quality problems.
"""

from typing import Dict, List, Set, Tuple
import re


# =============================================================================
# Conflict Detection Rules
# =============================================================================

# Groups of mutually exclusive terms - only one should appear
CONFLICT_GROUPS: List[Set[str]] = [
    # Gender conflicts
    {"woman", "man", "girl", "boy"},
    {"female", "male"},
    {"she", "he"},

    # Age conflicts
    {"young", "old", "elderly", "mature", "teen", "child", "adult"},

    # Position conflicts
    {"standing", "sitting", "lying", "kneeling", "crouching", "squatting"},

    # Direction conflicts
    {"facing viewer", "facing away", "from behind", "from side", "profile"},

    # Eye state conflicts
    {"open eyes", "closed eyes", "half-closed eyes"},

    # Mouth state conflicts
    {"open mouth", "closed mouth", "parted lips"},

    # Hair length conflicts
    {"long hair", "short hair", "medium hair", "bald"},

    # Time of day conflicts
    {"daytime", "nighttime", "sunset", "sunrise", "noon", "midnight"},

    # Indoor/outdoor conflicts
    {"indoor", "outdoor", "inside", "outside"},

    # Lighting conflicts
    {"bright", "dark", "dim"},
    {"high key", "low key"},
]


def find_conflicts(tokens: List[str]) -> List[Tuple[str, str, str]]:
    """Find conflicting tokens in a list.

    Args:
        tokens: List of token strings

    Returns:
        List of (token1, token2, conflict_type) tuples
    """
    conflicts = []
    tokens_lower = [t.lower() for t in tokens]

    for group in CONFLICT_GROUPS:
        found = []
        for token in tokens_lower:
            # Skip long tokens (likely sentences, not individual terms)
            if len(token) > 50:
                continue

            for term in group:
                # Use word boundary matching to avoid false positives
                # e.g., "cowboy" should not match "boy"
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, token):
                    # Don't match if it's part of a compound word
                    if term in ["boy", "girl", "man", "woman"]:
                        # Check it's not part of compound like "cowboy", "showgirl"
                        if f"cow{term}" in token or f"show{term}" in token:
                            continue
                    found.append((token, term))
                    break

        if len(found) > 1:
            # Multiple terms from same conflict group
            group_name = "/".join(sorted(group)[:3]) + "..."
            for i in range(len(found)):
                for j in range(i + 1, len(found)):
                    # Don't report if tokens are too similar (likely same phrase)
                    if found[i][0] == found[j][0]:
                        continue
                    conflicts.append((found[i][0], found[j][0], group_name))

    return conflicts


# =============================================================================
# Duplicate Detection
# =============================================================================

# Words that are OK to repeat
REPEAT_OK_WORDS = {
    "the", "a", "an", "and", "or", "with", "in", "on", "at", "to", "of",
    "is", "are", "has", "have", "very", "extremely", "highly",
    "her", "his", "she", "he", "they", "their", "it", "its",
    "this", "that", "these", "those", "be", "been", "being",
    "for", "from", "by", "as", "into", "like", "through",
}


def find_duplicates(text: str) -> List[Tuple[str, int]]:
    """Find duplicate words/phrases in text.

    Args:
        text: Prompt text

    Returns:
        List of (word, count) tuples for duplicates
    """
    # Split into words
    words = re.findall(r'\b\w+\b', text.lower())

    # Count occurrences
    counts: Dict[str, int] = {}
    for word in words:
        if word not in REPEAT_OK_WORDS and len(word) > 2:
            counts[word] = counts.get(word, 0) + 1

    # Return duplicates (count > 1)
    duplicates = [(word, count) for word, count in counts.items() if count > 1]
    return sorted(duplicates, key=lambda x: x[1], reverse=True)


def find_phrase_duplicates(text: str) -> List[Tuple[str, int]]:
    """Find duplicate phrases (2-3 word sequences) in text.

    Args:
        text: Prompt text

    Returns:
        List of (phrase, count) tuples for duplicates
    """
    words = re.findall(r'\b\w+\b', text.lower())
    phrase_counts: Dict[str, int] = {}

    # Check 2-word and 3-word phrases
    for n in [2, 3]:
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            # Skip if all words are common
            if all(w in REPEAT_OK_WORDS for w in words[i:i + n]):
                continue
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    duplicates = [(phrase, count) for phrase, count in phrase_counts.items() if count > 1]
    return sorted(duplicates, key=lambda x: x[1], reverse=True)


def remove_duplicate_words(text: str) -> str:
    """Remove duplicate comma-separated segments while preserving sentences.

    Only removes exact duplicate comma-separated phrases, not individual
    words within flowing text to preserve grammar.

    Args:
        text: Prompt text

    Returns:
        Text with duplicate segments removed
    """
    # Split by commas but preserve sentence structure
    segments = text.split(", ")
    seen: Set[str] = set()
    result = []

    for segment in segments:
        # Normalize for comparison
        segment_normalized = segment.lower().strip()

        # Skip if too short (likely incomplete)
        if len(segment_normalized) < 3:
            result.append(segment)
            continue

        # Check if this segment (or very similar) was seen
        if segment_normalized not in seen:
            result.append(segment)
            seen.add(segment_normalized)
        # Also add if it's a sentence (has verb-like structure)
        elif any(c in segment for c in ".!?"):
            result.append(segment)

    return ", ".join(result)


# =============================================================================
# Quality Checks
# =============================================================================

# Common quality issues patterns
QUALITY_ISSUES: List[Tuple[str, str, str]] = [
    # (pattern, issue_message, suggestion)
    (r'\b(\w+)\s+\1\b', "Repeated word", "Remove duplicate word"),
    (r',\s*,', "Double comma", "Remove extra comma"),
    (r'\.\s*\.', "Double period", "Remove extra period"),
    (r'\s{2,}', "Multiple spaces", "Reduce to single space"),
    (r'^\s*,', "Leading comma", "Remove leading comma"),
    (r',\s*$', "Trailing comma", "Remove trailing comma"),
    (r'\(\s*\)', "Empty parentheses", "Remove empty parentheses"),
]


def check_quality_issues(text: str) -> List[Tuple[str, str, str]]:
    """Check for common quality issues.

    Args:
        text: Prompt text

    Returns:
        List of (matched_text, issue_message, suggestion) tuples
    """
    issues = []
    for pattern, message, suggestion in QUALITY_ISSUES:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            issues.append((match.group(), message, suggestion))
    return issues


def fix_quality_issues(text: str) -> str:
    """Auto-fix common quality issues.

    Args:
        text: Prompt text

    Returns:
        Fixed text
    """
    # Remove repeated words
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    # Fix punctuation
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)

    # Fix leading/trailing punctuation
    text = re.sub(r'^\s*,\s*', '', text)
    text = re.sub(r'\s*,\s*$', '', text)

    # Remove empty parentheses
    text = re.sub(r'\(\s*\)', '', text)

    return text.strip()


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_clip_tokens(text: str) -> int:
    """Estimate CLIP token count for text.

    CLIP tokenizer roughly:
    - 1 token per ~4 characters for common words
    - More tokens for rare words/names

    Args:
        text: Prompt text

    Returns:
        Estimated token count
    """
    # Simple estimation: words + some overhead for subword tokenization
    words = text.split()
    base_tokens = len(words)

    # Add tokens for long words (likely split)
    for word in words:
        if len(word) > 8:
            base_tokens += (len(word) - 8) // 4

    return base_tokens
