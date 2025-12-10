"""
Iterative Prompt Improvement Workflow

Run batches of 5 images, analyze results, improve prompts, repeat until diminishing returns.

Usage:
    # Interactive mode (requires terminal input)
    python iterative_improve.py --folder ./test_images/ --provider anthropic --api-key KEY

    # Auto mode (non-interactive, runs N batches automatically)
    python iterative_improve.py --folder ./test_images/ --provider anthropic --api-key KEY --auto 10

    # Resume from previous session
    python iterative_improve.py --folder ./test_images/ --provider anthropic --api-key KEY --resume ./improvement_sessions/session_WOMAN_20251210_101551

    # With subject filter
    python iterative_improve.py --folder ./test_images/ --subject WOMAN --provider anthropic --api-key KEY
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from standalone components file (no ComfyUI dependencies)
from prompt_components import COMPONENTS, SUBJECT_COMPONENTS, DETAIL_MODES


# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 5
OUTPUT_DIR = "./improvement_sessions"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================================
# Working Prompts (editable during session)
# ============================================================================

# Start with copy of original prompts
WORKING_PROMPTS = deepcopy(COMPONENTS)


# ============================================================================
# Utilities
# ============================================================================

def load_image_as_base64(image_path: str) -> Tuple[str, int, int]:
    from PIL import Image
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        width, height = img.size
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        base64_str = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str, width, height


def find_images(folder: str) -> List[str]:
    folder = Path(folder)
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted([str(p) for p in images])


def create_client(provider: str, api_key: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    elif provider == "grok":
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    raise ValueError(f"Unknown provider: {provider}")


def get_default_model(provider: str) -> str:
    return {"anthropic": "claude-sonnet-4-5-20250929", "openai": "gpt-4o", "grok": "grok-2-vision-1212"}.get(provider)


def parse_json_response(text: str) -> dict:
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    return {"raw": text, "parse_error": True}


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_component(client, model: str, provider: str, base64_image: str,
                      component_key: str, prompt: str) -> Tuple[dict, float]:
    """Run single component analysis."""

    system = "You are an expert visual analyst. Output valid JSON only."
    start = time.time()

    try:
        if provider == "anthropic":
            resp = client.messages.create(
                model=model, max_tokens=1000, temperature=0.3, system=system,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            raw = resp.content[0].text
        else:
            resp = client.chat.completions.create(
                model=model, max_tokens=1000, temperature=0.3,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]}
                ]
            )
            raw = resp.choices[0].message.content

        return parse_json_response(raw), time.time() - start
    except Exception as e:
        return {"error": str(e)}, time.time() - start


def detect_subject(client, model: str, provider: str, base64_image: str) -> str:
    """Detect subject type."""
    result, _ = analyze_component(
        client, model, provider, base64_image,
        "subject_detection", WORKING_PROMPTS["subject_detection"]["prompt"]
    )
    return result.get("subject_type", "OTHER").upper()


def run_batch(
    client, model: str, provider: str,
    image_paths: List[str],
    components: List[str],
    target_subject: Optional[str] = None,
) -> List[dict]:
    """Run a batch of images through specified components."""

    # Make a local copy of components to avoid any reference issues
    components = list(components)

    results = []

    for img_path in image_paths:
        print(f"\n  Processing: {Path(img_path).name}")

        try:
            base64_img, w, h = load_image_as_base64(img_path)

            # Detect subject
            subject = detect_subject(client, model, provider, base64_img)
            print(f"    Subject: {subject}")

            # Skip if not target subject
            if target_subject and subject != target_subject.upper():
                print(f"    [Skipped - not {target_subject}]")
                continue

            # Run each component
            img_results = {
                "image": Path(img_path).name,
                "subject": subject,
                "components": {}
            }

            for comp_key in components:
                if comp_key not in WORKING_PROMPTS:
                    continue

                print(f"    {comp_key}...", end=" ", flush=True)
                result, duration = analyze_component(
                    client, model, provider, base64_img,
                    comp_key, WORKING_PROMPTS[comp_key]["prompt"]
                )
                img_results["components"][comp_key] = result
                print(f"{duration:.1f}s")

            results.append(img_results)

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"image": Path(img_path).name, "error": str(e)})

    return results


def calculate_quality_score(results: List[dict], components: List[str]) -> dict:
    """Calculate quality metrics from batch results."""

    scores = {
        "total_images": len(results),
        "successful": 0,
        "parse_errors": 0,
        "empty_fields": 0,
        "component_scores": {},
    }

    for comp in components:
        scores["component_scores"][comp] = {
            "success": 0,
            "parse_error": 0,
            "empty_prompt_desc": 0,
            "has_error_field": 0,
        }

    for result in results:
        if "error" in result:
            continue

        scores["successful"] += 1

        for comp, data in result.get("components", {}).items():
            if comp not in scores["component_scores"]:
                continue

            cs = scores["component_scores"][comp]

            if data.get("parse_error"):
                cs["parse_error"] += 1
                scores["parse_errors"] += 1
            else:
                cs["success"] += 1

            if data.get("error"):
                cs["has_error_field"] += 1

            # Check for empty prompt_description
            if not data.get("prompt_description"):
                cs["empty_prompt_desc"] += 1
                scores["empty_fields"] += 1

    return scores


def display_batch_results(results: List[dict], components: List[str]):
    """Display results in a readable format."""

    print("\n" + "="*70)
    print("BATCH RESULTS")
    print("="*70)

    for result in results:
        if "error" in result:
            print(f"\n[{result['image']}] ERROR: {result['error']}")
            continue

        print(f"\n[{result['image']}] Subject: {result['subject']}")

        for comp, data in result.get("components", {}).items():
            print(f"\n  {comp}:")

            if data.get("parse_error"):
                print(f"    [PARSE ERROR]")
                print(f"    Raw: {str(data.get('raw', ''))[:100]}...")
                continue

            if data.get("error"):
                print(f"    [ERROR] {data['error']}")
                continue

            # Show key fields based on component
            if comp == "framing":
                print(f"    Shot: {data.get('shot_type', '?')}, Pose: {data.get('subject_pose', '?')}")
                print(f"    Color: {data.get('color_mode', '?')}, Fill: {data.get('frame_fill_percent', '?')}%")
            elif comp == "ethnicity":
                print(f"    Ethnicity: {data.get('ethnicity', '?')}")
                print(f"    Skin: {data.get('skin_tone_base', '?')} ({data.get('skin_tone_undertone', '?')})")
                print(f"    Gender: {data.get('gender', '?')}, Age: {data.get('age_range', '?')}")
            elif comp == "hair":
                print(f"    Arrangement: {data.get('arrangement', '?')}")
                print(f"    Color: {data.get('base_color', '?')}, Texture: {data.get('texture', '?')}")
                print(f"    Crown: {data.get('crown_description', '?')}")
            elif comp == "eyes":
                print(f"    Color: {data.get('color', '?')}, Shape: {data.get('shape', '?')}")
                print(f"    Gaze: {data.get('gaze', '?')}, Expression: {data.get('expression', '?')}")
            elif comp == "clothing":
                garments = data.get("garments", [])
                if garments:
                    for g in garments[:2]:
                        print(f"    {g.get('type', '?')}: {g.get('color', '?')} {g.get('material', '?')}")
            elif comp == "body_pose":
                print(f"    Posture: {data.get('posture', '?')}")
                print(f"    Body angle: {data.get('body_angle', '?')}")

            # Show prompt_description if present
            desc = data.get("prompt_description", "")
            if desc:
                print(f"    Prompt: {desc[:80]}{'...' if len(desc) > 80 else ''}")


def display_quality_metrics(scores: dict, iteration: int):
    """Display quality metrics."""

    print("\n" + "="*70)
    print(f"QUALITY METRICS - Iteration {iteration}")
    print("="*70)

    print(f"\nOverall:")
    print(f"  Images processed: {scores['successful']}/{scores['total_images']}")
    print(f"  Parse errors: {scores['parse_errors']}")
    print(f"  Empty fields: {scores['empty_fields']}")

    print(f"\nBy Component:")
    for comp, cs in scores["component_scores"].items():
        total = cs["success"] + cs["parse_error"]
        if total == 0:
            continue
        success_rate = cs["success"] / total * 100 if total > 0 else 0
        print(f"  {comp}:")
        print(f"    Success rate: {success_rate:.0f}% ({cs['success']}/{total})")
        if cs["empty_prompt_desc"] > 0:
            print(f"    Empty prompt_description: {cs['empty_prompt_desc']}")


def edit_prompt_interactive(component_key: str) -> bool:
    """Interactive prompt editor."""

    if component_key not in WORKING_PROMPTS:
        print(f"Unknown component: {component_key}")
        return False

    current = WORKING_PROMPTS[component_key]["prompt"]

    print(f"\n{'='*70}")
    print(f"EDITING: {component_key}")
    print(f"{'='*70}")
    print(f"\nCurrent prompt ({len(current)} chars):")
    print("-"*40)
    print(current)
    print("-"*40)

    print("\nOptions:")
    print("  1. Replace entire prompt (paste new, then type END on new line)")
    print("  2. Append to prompt")
    print("  3. Find/replace text")
    print("  4. Cancel")

    choice = input("\nChoice [1-4]: ").strip()

    if choice == "1":
        print("\nPaste new prompt, then type END on a new line:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        new_prompt = "\n".join(lines)
        if new_prompt.strip():
            WORKING_PROMPTS[component_key]["prompt"] = new_prompt
            print(f"[OK] Prompt replaced ({len(new_prompt)} chars)")
            return True

    elif choice == "2":
        print("\nText to append (type END on new line when done):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        append_text = "\n".join(lines)
        if append_text.strip():
            WORKING_PROMPTS[component_key]["prompt"] = current + "\n\n" + append_text
            print(f"[OK] Text appended")
            return True

    elif choice == "3":
        find = input("Find text: ")
        replace = input("Replace with: ")
        if find and find in current:
            WORKING_PROMPTS[component_key]["prompt"] = current.replace(find, replace)
            print(f"[OK] Replaced '{find}' with '{replace}'")
            return True
        else:
            print(f"Text not found: '{find}'")

    print("No changes made")
    return False


def save_session(session_dir: str, iteration: int, results: List[dict],
                 scores: dict, prompts_changed: List[str]):
    """Save session state."""

    os.makedirs(session_dir, exist_ok=True)

    # Save iteration results
    iter_file = os.path.join(session_dir, f"iteration_{iteration:02d}.json")
    with open(iter_file, "w", encoding="utf-8") as f:
        json.dump({
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "scores": scores,
            "prompts_changed": prompts_changed,
        }, f, indent=2, ensure_ascii=False)

    # Save current working prompts
    prompts_file = os.path.join(session_dir, f"prompts_iter_{iteration:02d}.json")
    prompts_export = {}
    for key, comp in WORKING_PROMPTS.items():
        prompts_export[key] = {"name": comp["name"], "prompt": comp["prompt"]}

    with open(prompts_file, "w", encoding="utf-8") as f:
        json.dump(prompts_export, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {iter_file}")
    print(f"[Saved] {prompts_file}")


def load_prompts_from_file(filepath: str):
    """Load prompts from a JSON file."""
    global WORKING_PROMPTS

    with open(filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    for key, data in loaded.items():
        if key in WORKING_PROMPTS:
            if isinstance(data, dict):
                WORKING_PROMPTS[key]["prompt"] = data.get("prompt", data)
            else:
                WORKING_PROMPTS[key]["prompt"] = data

    print(f"[Loaded] Prompts from {filepath}")


# ============================================================================
# Persistent Tracking
# ============================================================================

def get_tracking_file(folder: str) -> str:
    """Get path to tracking file for a folder."""
    folder_hash = abs(hash(os.path.abspath(folder))) % 100000
    return os.path.join(OUTPUT_DIR, f"tracking_{folder_hash}.json")


def load_processed_images(folder: str) -> set:
    """Load set of already-processed image filenames."""
    tracking_file = get_tracking_file(folder)
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed", []))
        except:
            pass
    return set()


def save_processed_images(folder: str, processed: set):
    """Save set of processed image filenames."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tracking_file = get_tracking_file(folder)
    with open(tracking_file, "w", encoding="utf-8") as f:
        json.dump({
            "folder": os.path.abspath(folder),
            "processed": list(processed),
            "count": len(processed),
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)


def add_to_processed(folder: str, image_names: List[str]):
    """Add image names to processed set."""
    processed = load_processed_images(folder)
    for name in image_names:
        processed.add(name)
    save_processed_images(folder, processed)


def get_unprocessed_images(folder: str, all_images: List[str]) -> List[str]:
    """Filter to only unprocessed images."""
    processed = load_processed_images(folder)
    unprocessed = [img for img in all_images if Path(img).name not in processed]
    return unprocessed


def reset_tracking(folder: str):
    """Reset tracking for a folder."""
    tracking_file = get_tracking_file(folder)
    if os.path.exists(tracking_file):
        os.remove(tracking_file)
        print(f"[Reset] Tracking cleared for {folder}")


# ============================================================================
# Main Workflow
# ============================================================================

def run_auto_session(
    folder: str,
    provider: str,
    api_key: str,
    num_batches: int = 10,
    model: Optional[str] = None,
    target_subject: str = "WOMAN",
    components: Optional[List[str]] = None,
    load_prompts: Optional[str] = None,
    reset: bool = False,
):
    """Run automatic (non-interactive) batch session."""

    print("\n" + "="*70)
    print("AUTO BATCH SESSION (Non-Interactive)")
    print("="*70)
    print(f"\nFolder: {folder}")
    print(f"Provider: {provider}")
    print(f"Target Subject: {target_subject}")
    print(f"Batches to run: {num_batches}")

    # Reset tracking if requested
    if reset:
        reset_tracking(folder)

    # Find all images
    all_images = find_images(folder)
    print(f"Total images in folder: {len(all_images)}")

    # Filter to unprocessed only
    unprocessed = get_unprocessed_images(folder, all_images)
    already_processed = len(all_images) - len(unprocessed)
    print(f"Already processed: {already_processed}")
    print(f"Remaining unprocessed: {len(unprocessed)}")

    if len(unprocessed) == 0:
        print("\nAll images have been processed!")
        print("Use --reset to start fresh.")
        return

    # Default components - make a COPY
    if components is None:
        components = list(SUBJECT_COMPONENTS.get(target_subject.upper(), {}).get("Standard", []))
    else:
        components = list(components)
    print(f"Components: {', '.join(components)}")

    # Create client
    model = model or get_default_model(provider)
    print(f"Model: {model}")
    client = create_client(provider, api_key)

    # Load prompts if specified
    if load_prompts:
        load_prompts_from_file(load_prompts)

    # Session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"auto_{target_subject}_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session dir: {session_dir}")

    # Shuffle unprocessed images
    random.shuffle(unprocessed)
    image_queue = list(unprocessed)

    all_scores = []
    all_results = []

    print("\n" + "="*70)
    print(f"Starting {num_batches} batches...")
    print("="*70)

    for iteration in range(1, num_batches + 1):
        print(f"\n{'#'*70}")
        print(f"# BATCH {iteration}/{num_batches}")
        print(f"{'#'*70}")

        # Check if we have enough images
        if len(image_queue) < BATCH_SIZE:
            print(f"\nOnly {len(image_queue)} images remaining.")
            if len(image_queue) == 0:
                print("No more unprocessed images. Stopping.")
                break
            # Use remaining images
            batch = image_queue
            image_queue = []
        else:
            batch = image_queue[:BATCH_SIZE]
            image_queue = image_queue[BATCH_SIZE:]

        print(f"\nProcessing {len(batch)} images:")
        for img in batch:
            print(f"  - {Path(img).name}")

        # Run batch
        print(f"\nRunning analysis...")
        results = run_batch(client, model, provider, batch, components, target_subject)
        all_results.extend(results)

        # Track processed images
        processed_names = [Path(img).name for img in batch]
        add_to_processed(folder, processed_names)

        # Calculate metrics
        scores = calculate_quality_score(results, components)
        display_quality_metrics(scores, iteration)
        all_scores.append(scores)

        # Save iteration
        save_session(session_dir, iteration, results, scores, [])

        # Show progress
        remaining = len(image_queue)
        total_processed = already_processed + (iteration * BATCH_SIZE)
        print(f"\n[Progress] {total_processed}/{len(all_images)} images processed, {remaining} remaining in queue")

    # Final summary
    print("\n" + "="*70)
    print("AUTO SESSION COMPLETE")
    print("="*70)
    print(f"Batches completed: {len(all_scores)}")
    print(f"Total images analyzed: {len(all_results)}")

    if all_scores:
        total_parse_errors = sum(s['parse_errors'] for s in all_scores)
        total_empty = sum(s['empty_fields'] for s in all_scores)
        print(f"Total parse errors: {total_parse_errors}")
        print(f"Total empty fields: {total_empty}")

    print(f"\nResults saved in: {session_dir}")
    print(f"Tracking file: {get_tracking_file(folder)}")


def run_improvement_session(
    folder: str,
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    target_subject: str = "WOMAN",
    components: Optional[List[str]] = None,
    load_prompts: Optional[str] = None,
    resume_session: Optional[str] = None,
    reset: bool = False,
):
    """Run interactive improvement session."""

    print("\n" + "="*70)
    print("ITERATIVE PROMPT IMPROVEMENT SESSION")
    print("="*70)
    print(f"\nFolder: {folder}")
    print(f"Provider: {provider}")
    print(f"Target Subject: {target_subject}")

    # Reset tracking if requested
    if reset:
        reset_tracking(folder)

    # Find all images
    all_images = find_images(folder)
    print(f"Total images in folder: {len(all_images)}")

    # Filter to unprocessed only
    unprocessed = get_unprocessed_images(folder, all_images)
    already_processed = len(all_images) - len(unprocessed)
    print(f"Already processed: {already_processed}")
    print(f"Remaining unprocessed: {len(unprocessed)}")

    if len(all_images) == 0:
        print("No images found!")
        return

    if len(unprocessed) == 0:
        print("\nAll images have been processed!")
        print("Use --reset to start fresh, or run without tracking.")
        return

    # Default components for WOMAN - make a COPY to avoid reference issues
    if components is None:
        components = list(SUBJECT_COMPONENTS.get(target_subject.upper(), {}).get("Standard", []))
    else:
        components = list(components)  # Ensure we have a copy
    print(f"Components: {', '.join(components)}")

    # Create client
    model = model or get_default_model(provider)
    print(f"Model: {model}")
    client = create_client(provider, api_key)

    # Load prompts if specified
    if load_prompts:
        load_prompts_from_file(load_prompts)

    # Resume from session if specified
    if resume_session and os.path.exists(resume_session):
        # Load prompts from last iteration
        prompt_files = sorted(Path(resume_session).glob("prompts_iter_*.json"))
        if prompt_files:
            load_prompts_from_file(str(prompt_files[-1]))
            print(f"[Resumed] Loaded prompts from {prompt_files[-1].name}")

    # Session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"session_{target_subject}_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session dir: {session_dir}")

    # Shuffle UNPROCESSED images only
    random.shuffle(unprocessed)
    image_queue = list(unprocessed)

    iteration = 1
    all_scores = []

    print("\n" + "="*70)
    print("Starting improvement loop...")
    print("Run batches of 5, review results, improve prompts, repeat.")
    print("="*70)

    while True:
        print(f"\n{'#'*70}")
        print(f"# ITERATION {iteration}")
        print(f"{'#'*70}")

        # Get next batch
        if len(image_queue) < BATCH_SIZE:
            print(f"\nOnly {len(image_queue)} images remaining.")
            if len(image_queue) == 0:
                print("All unprocessed images have been analyzed!")
                print("Use --reset to start fresh.")
                break
            # Use remaining images
            batch = image_queue
            image_queue = []
        else:
            batch = image_queue[:BATCH_SIZE]
            image_queue = image_queue[BATCH_SIZE:]

        print(f"\nBatch {iteration}: {len(batch)} images")
        for img in batch:
            print(f"  - {Path(img).name}")

        # Run batch
        print(f"\nRunning analysis...")
        results = run_batch(client, model, provider, batch, components, target_subject)

        # Track processed images
        processed_names = [Path(img).name for img in batch]
        add_to_processed(folder, processed_names)

        # Display results
        display_batch_results(results, components)

        # Calculate and display metrics
        scores = calculate_quality_score(results, components)
        display_quality_metrics(scores, iteration)
        all_scores.append(scores)

        # Show trend if multiple iterations
        if len(all_scores) > 1:
            print(f"\n{'='*70}")
            print("TREND (parse errors / empty fields):")
            for i, s in enumerate(all_scores, 1):
                print(f"  Iter {i}: {s['parse_errors']} parse errors, {s['empty_fields']} empty fields")

            # Check for diminishing returns
            if len(all_scores) >= 3:
                recent = all_scores[-3:]
                if all(s['parse_errors'] == 0 and s['empty_fields'] <= 2 for s in recent):
                    print("\n[!] Quality appears stable. Consider stopping.")

        # Save this iteration
        save_session(session_dir, iteration, results, scores, [])

        # Interactive menu
        print(f"\n{'='*70}")
        print("OPTIONS:")
        print("  [1-9] Edit component prompt (1=framing, 2=ethnicity, 3=hair, etc.)")
        print("  [c]   Continue to next batch")
        print("  [r]   Re-run same batch with current prompts")
        print("  [l]   List components")
        print("  [v]   View a component's current prompt")
        print("  [s]   Save & show final prompts")
        print("  [q]   Quit")

        while True:
            choice = input("\nChoice: ").strip().lower()

            if choice == "c":
                iteration += 1
                break

            elif choice == "r":
                print("\nRe-running same batch...")
                image_queue = batch + image_queue  # Put batch back
                break

            elif choice == "l":
                print("\nComponents:")
                for i, comp in enumerate(components, 1):
                    print(f"  {i}. {comp}")

            elif choice == "v":
                comp = input("Component name: ").strip()
                if comp in WORKING_PROMPTS:
                    print(f"\n{WORKING_PROMPTS[comp]['prompt']}")
                else:
                    print(f"Unknown: {comp}")

            elif choice == "s":
                # Save final prompts
                final_file = os.path.join(session_dir, "FINAL_PROMPTS.json")
                prompts_export = {}
                for key, comp in WORKING_PROMPTS.items():
                    prompts_export[key] = {"name": comp["name"], "prompt": comp["prompt"]}
                with open(final_file, "w", encoding="utf-8") as f:
                    json.dump(prompts_export, f, indent=2, ensure_ascii=False)
                print(f"\n[Saved] {final_file}")
                print("\nTo use these prompts, copy them to sid_zimage_prompt_generator_advanced_v2.py")

            elif choice == "q":
                print("\nSession ended.")
                print(f"Results saved in: {session_dir}")
                return

            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(components):
                    comp_key = components[idx]
                    if edit_prompt_interactive(comp_key):
                        print(f"\nPrompt for '{comp_key}' updated. Use 'r' to re-run batch.")
                else:
                    print(f"Invalid number. Range: 1-{len(components)}")

            else:
                # Try as component name
                if choice in WORKING_PROMPTS:
                    if edit_prompt_interactive(choice):
                        print(f"\nPrompt updated. Use 'r' to re-run batch.")
                else:
                    print("Invalid choice")


def main():
    parser = argparse.ArgumentParser(description="Iterative prompt improvement")

    parser.add_argument("--folder", "-f", required=True, help="Folder with test images")
    parser.add_argument("--provider", "-p", default="anthropic", choices=["anthropic", "openai", "grok"])
    parser.add_argument("--api-key", "-k", help="API key (or use env var)")
    parser.add_argument("--model", "-m", help="Model name")
    parser.add_argument("--subject", "-s", default="WOMAN", help="Target subject type")
    parser.add_argument("--components", "-c", nargs="+", help="Specific components to test")
    parser.add_argument("--load-prompts", help="Load prompts from JSON file")

    # New options
    parser.add_argument("--auto", type=int, metavar="N", help="Run N batches automatically (non-interactive)")
    parser.add_argument("--reset", action="store_true", help="Reset tracking, re-process all images")
    parser.add_argument("--resume", help="Resume from previous session directory")
    parser.add_argument("--status", action="store_true", help="Show tracking status and exit")

    args = parser.parse_args()

    # Status check
    if args.status:
        all_images = find_images(args.folder)
        processed = load_processed_images(args.folder)
        print(f"\nFolder: {args.folder}")
        print(f"Total images: {len(all_images)}")
        print(f"Processed: {len(processed)}")
        print(f"Remaining: {len(all_images) - len(processed)}")
        print(f"Tracking file: {get_tracking_file(args.folder)}")
        return

    # Get API key
    api_key = args.api_key
    if not api_key:
        env_vars = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "grok": "GROK_API_KEY"}
        api_key = os.environ.get(env_vars.get(args.provider, ""))

    if not api_key:
        parser.error("API key required")

    # Auto mode vs interactive mode
    if args.auto:
        run_auto_session(
            folder=args.folder,
            provider=args.provider,
            api_key=api_key,
            num_batches=args.auto,
            model=args.model,
            target_subject=args.subject,
            components=args.components,
            load_prompts=args.load_prompts,
            reset=args.reset,
        )
    else:
        run_improvement_session(
            folder=args.folder,
            provider=args.provider,
            api_key=api_key,
            model=args.model,
            target_subject=args.subject,
            components=args.components,
            load_prompts=args.load_prompts,
            resume_session=args.resume,
            reset=args.reset,
        )


if __name__ == "__main__":
    main()
