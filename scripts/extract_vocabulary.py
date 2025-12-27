#!/usr/bin/env python3
"""Extract vocabulary from scraped prompts data and generate dictionary entries.

This script analyzes the scraped prompts data and outputs vocabulary
that can be added to the classifier dictionaries.

Usage:
    python scripts/extract_vocabulary.py [input_path] [--output vocab.json]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


# Map scraped categories to canonical categories
CATEGORY_MAP = {
    "medium": "style_medium",
    "subject": "subject",  # Mix of subject and subject_details
    "clothing": "subject_details",  # Subcategory: clothing
    "outdoor": "environment",
    "indoor": "environment",
    "camera": "technical",
    "lighting": "lighting",
    "action-pose": "action_pose",
    "post-processing": "style_medium",
    "others": "quality_boosters",  # Mood terms
}


def clean_prompt(prompt: str) -> str:
    """Clean and normalize a prompt term."""
    # Remove special characters
    prompt = prompt.replace("�", " - ")
    prompt = prompt.replace("&", "and")
    # Clean up whitespace
    prompt = " ".join(prompt.split())
    return prompt.strip().lower()


def extract_vocabulary(data: dict) -> dict:
    """Extract vocabulary organized by canonical category."""
    vocabulary = defaultdict(set)

    for source_category, items in data.items():
        if not items:
            continue

        canonical_cat = CATEGORY_MAP.get(source_category, "uncategorized")

        for item in items:
            prompt = item.get("prompt", "")
            if not prompt:
                continue

            cleaned = clean_prompt(prompt)
            if len(cleaned) >= 2:
                vocabulary[canonical_cat].add(cleaned)

                # Also add variations
                # "Golden hour sunlight" -> also add "golden hour"
                words = cleaned.split()
                if len(words) > 2:
                    # Add first 2-word phrase
                    vocabulary[canonical_cat].add(" ".join(words[:2]))

    # Convert sets to sorted lists
    return {cat: sorted(list(terms)) for cat, terms in vocabulary.items()}


def generate_dictionary_code(vocabulary: dict) -> str:
    """Generate Python code for adding vocabulary to dictionaries."""
    lines = [
        "# Auto-generated vocabulary from scraped prompts data",
        "# Add these to CATEGORY_KEYWORDS in categories.py",
        "",
    ]

    for category, terms in vocabulary.items():
        lines.append(f"# {category.upper()}: {len(terms)} terms")
        lines.append(f"{category.upper()}_VOCABULARY = {{")

        # Format terms in groups of 5
        for i in range(0, len(terms), 5):
            group = terms[i:i+5]
            formatted = ", ".join(f'"{t}"' for t in group)
            lines.append(f"    {formatted},")

        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocabulary from scraped prompts data"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=r"E:\Prompt\scraped_data\prompts_data.json",
        help="Path to prompts_data.json"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "python"],
        default="python",
        help="Output format"
    )

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from: {input_path}", file=sys.stderr)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract vocabulary
    vocabulary = extract_vocabulary(data)

    # Report stats
    total_terms = sum(len(terms) for terms in vocabulary.values())
    print(f"\nExtracted {total_terms} terms across {len(vocabulary)} categories:", file=sys.stderr)
    for cat, terms in vocabulary.items():
        print(f"  {cat}: {len(terms)} terms", file=sys.stderr)

    # Generate output
    if args.format == "json":
        output = json.dumps(vocabulary, indent=2, ensure_ascii=False)
    else:
        output = generate_dictionary_code(vocabulary)

    # Write output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nWritten to: {output_path}", file=sys.stderr)
    else:
        print("\n" + "=" * 60, file=sys.stderr)
        print(output)


if __name__ == "__main__":
    main()
