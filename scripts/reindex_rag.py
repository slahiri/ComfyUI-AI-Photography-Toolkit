#!/usr/bin/env python3
"""Rebuild RAG index from scraped prompt data.

Usage:
    python scripts/reindex_rag.py [input_path] [--device cuda|cpu]

Examples:
    python scripts/reindex_rag.py
    python scripts/reindex_rag.py E:\Prompt\scraped_data
    python scripts/reindex_rag.py E:\Prompt\scraped_data --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild RAG index from scraped prompt data"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=r"E:\Prompt\scraped_data",
        help="Path to scraped data folder (default: E:\\Prompt\\scraped_data)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for embedding (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)",
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model to use (default: openai/clip-vit-base-patch32)",
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Output path is always the RAG data directory
    output_path = PROJECT_ROOT / "core" / "compose" / "rag" / "data"

    print("=" * 60)
    print("RAG Index Rebuilder")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device or 'auto'}")
    print("=" * 60)

    # Import and run the build
    from core.compose.rag.build_index import build_index

    success = build_index(
        input_path=input_path,
        output_path=output_path,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    if success:
        print("\nIndex rebuilt successfully!")
        sys.exit(0)
    else:
        print("\nIndex build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
