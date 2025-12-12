"""
SID_ZImagePromptEnhancer Node

Universal prompt enhancement using cloud APIs or local LLMs.
Simple 3-instruction system: Enhance, Replace, Add.
Detail levels: Quick, Standard, Detailed, Extreme.

- Quick: Describe and update the prompt (basic cleanup)
- Standard: Basic LLM enhancement
- Detailed/Extreme: Deep enhancement using prompt generator logic

Connect SID_LLM_API or SID_LLM_Local (text models) to provide the LLM.
"""

import gc
import hashlib
import re
import sys
import time
import threading
from typing import Optional

from comfy_api.latest import io as comfy_io

from .llm_providers.llm_model_type import LLMModelConfig


# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")

# In-memory cache for seed-based results
_enhancer_cache: dict[str, str] = {}
_negative_cache: dict[str, str] = {}  # Separate cache for negative prompts
_MAX_CACHE_SIZE = 100  # Maximum number of cached results
_CACHE_CLEANUP_THRESHOLD = 80  # Trigger cleanup when cache reaches this size


def clear_enhancer_cache():
    """Clear the enhancer cache and run garbage collection."""
    global _enhancer_cache, _negative_cache
    _enhancer_cache.clear()
    _negative_cache.clear()
    gc.collect()
    print("[PromptEnhancer] Cache cleared and garbage collected")


def get_enhancer_cache_info() -> dict:
    """Get information about the current cache state."""
    return {
        "positive_cache_size": len(_enhancer_cache),
        "negative_cache_size": len(_negative_cache),
        "max_size": _MAX_CACHE_SIZE,
        "positive_memory_bytes": sum(len(k) + len(v) for k, v in _enhancer_cache.items()),
        "negative_memory_bytes": sum(len(k) + len(v) for k, v in _negative_cache.items()),
    }


# =============================================================================
# Progress Spinner for Console Feedback
# =============================================================================

class ProgressSpinner:
    """
    A console progress spinner that shows activity during LLM calls.
    Shows elapsed time and a spinning animation.
    """

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Processing", prefix: str = "[PromptEnhancer]"):
        self.message = message
        self.prefix = prefix
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _spinner_loop(self):
        """Background thread that displays the spinner."""
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            spinner = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
            sys.stdout.write(f"\r{self.prefix} {spinner} {self.message}... {elapsed:.1f}s")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    def start(self):
        """Start the spinner."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        """Stop the spinner and print completion."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        elapsed = time.time() - self._start_time if self._start_time else 0
        icon = "✓" if success else "✗"
        sys.stdout.write(f"\r{self.prefix} {icon} {self.message} ({elapsed:.1f}s)        \n")
        sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, *args):
        self.stop(success=exc_type is None)


# =============================================================================
# Detail Level Configurations
# =============================================================================

DETAIL_LEVELS = ["Quick", "Standard", "Detailed", "Extreme"]

# Length increase limits per detail level (as percentage of original)
LENGTH_LIMITS = {
    "Quick": (0, 20),      # 0-20% increase (max 1.2x original)
    "Standard": (10, 30),  # 10-30% increase (max 1.3x original)
    "Detailed": (20, 40),  # 20-40% increase (max 1.4x original)
    "Extreme": (30, 40),   # 30-40% increase (max 1.4x original)
}

# Repetition penalty for LLM calls (reduces repetitive text)
REPETITION_PENALTY = 1.15

# Quick: Just describe and cleanup
QUICK_SYSTEM = """You are a prompt editor. Your task is to clean up and clarify the prompt.
- Fix grammar and spelling
- Remove redundant words
- Ensure clarity
CRITICAL: Keep ALL original details intact. Only fix language issues.
LENGTH CONSTRAINT: Output must be approximately the same length as input (0-20% increase max).
Output ONLY the cleaned prompt."""

# Standard: Basic enhancement
STANDARD_SYSTEM = """You are a prompt engineer. Your task is to enhance image generation prompts.

CRITICAL RULES - YOU MUST FOLLOW:
1. PRESERVE every single detail from the original prompt - do NOT remove or replace anything
2. Keep all specific descriptions exactly as written (measurements, colors, textures, positions)
3. Only ADD new complementary details between existing content
4. Enhancement means ADDITION, never replacement or summarization
5. LENGTH CONSTRAINT: Output must be 10-30% longer than input, NO MORE

Output ONLY the enhanced prompt - no explanations."""

# Detailed/Extreme: Deep enhancement using component analysis (mirrors prompt generator)
DEEP_SYSTEM = """You are an expert prompt engineer specializing in AI image generation.

## CRITICAL PRESERVATION RULES - MUST FOLLOW:
1. **NEVER DELETE** - Every word, phrase, and detail from the original prompt MUST appear in your output
2. **NEVER REPLACE** - Do not substitute original descriptions with your own interpretations
3. **NEVER SUMMARIZE** - Do not condense or paraphrase existing content
4. **ONLY ADD** - Insert new details BETWEEN or AFTER existing content to enrich it
5. **PRESERVE SPECIFICS** - Keep exact measurements, colors, textures, positions, brand names, technical specs
6. **LENGTH CONSTRAINT** - Output must be 20-40% longer than input, NO MORE

## Enhancement Method:
For each section of the original prompt, you may INSERT additional details:
- Between sentences: Add complementary visual details
- After descriptions: Add texture, material, or lighting nuances
- Around technical specs: Add how those specs affect the visual result

## What You CAN Add (sparingly):
- Key micro-details (skin texture, fabric weave, light reflections)
- Brief atmospheric elements
- Essential technical photography terms

## What You CANNOT Do:
- Remove ANY existing description
- Replace "espresso brown" with "dark brown" or any synonym
- Change camera/lens specs the user specified
- Simplify or condense detailed descriptions
- Rewrite sections in your own words
- Exceed the length constraint

Output ONLY the enhanced prompt - no analysis, no markdown, no explanations."""

# Extreme: Maximum detail with synthesis pass
EXTREME_SYSTEM = """You are an expert prompt engineer. Enhance while respecting length limits.

RULES:
1. PRESERVE all original content - every word and detail
2. Only ADD brief, high-impact details
3. LENGTH CONSTRAINT: Output must be 30-40% longer than input, NO MORE
4. Focus on quality over quantity

Current prompt to enhance:
{prompt}

Output ONLY the final prompt with additions:"""


# =============================================================================
# ComfyUI Node
# =============================================================================

class SID_ZImagePromptEnhancer(comfy_io.ComfyNode):
    """
    Universal Prompt Enhancer for Z-Image workflows.

    Simple 3-instruction system:
    - Enhance: Instructions to improve/enhance the prompt
    - Replace: Elements to find and replace
    - Add: Elements to add to the prompt

    Detail levels:
    - Quick: Describe and cleanup (basic)
    - Standard: Basic LLM enhancement
    - Detailed: Deep component-based enhancement
    - Extreme: Maximum detail with synthesis pass

    Works with cloud APIs or local LLMs.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="SID_ZImagePromptEnhancer",
            display_name="SID Z-Image Prompt Enhancer",
            category="SID Photography Toolkit/Prompt",
            description="Prompt enhancement with Enhance/Replace/Add instructions",
            inputs=[
                # LLM Model Input
                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="Connect SID_LLM_API or SID_LLM_Local (text models)"
                ),

                # Prompt Input
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    tooltip="The prompt to enhance"
                ),

                # Negative Prompt Input
                comfy_io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Optional: Negative prompt to enhance. If empty, one will be generated from the positive prompt."
                ),

                # Detail Level
                comfy_io.Combo.Input(
                    "detail_level",
                    options=DETAIL_LEVELS,
                    default="Standard",
                    tooltip="Quick: cleanup, Standard: enhance, Detailed: deep analysis, Extreme: maximum detail"
                ),

                # Three instruction boxes
                comfy_io.String.Input(
                    "enhance",
                    multiline=True,
                    default="",
                    display_name="Enhance",
                    tooltip="Instructions to improve the prompt"
                ),
                comfy_io.String.Input(
                    "replace",
                    multiline=True,
                    default="",
                    display_name="Replace",
                    tooltip="Elements to replace (e.g., 'change background to forest')"
                ),
                comfy_io.String.Input(
                    "add",
                    multiline=True,
                    default="",
                    display_name="Add",
                    tooltip="Elements to add (e.g., 'add bokeh effect')"
                ),

                # Seed Control
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Random seed for reproducibility"
                ),
            ],
            outputs=[
                comfy_io.String.Output(
                    "enhanced_prompt",
                    display_name="enhanced_prompt",
                ),
                comfy_io.String.Output(
                    "enhanced_negative_prompt",
                    display_name="enhanced_negative_prompt",
                    tooltip="Enhanced negative prompt (generated or enhanced from input)",
                ),
                comfy_io.String.Output(
                    "original_prompt",
                    display_name="original_prompt",
                ),
            ],
        )

    @classmethod
    def _get_client(cls, llm_model: LLMModelConfig):
        """Get LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=600.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)

        elif provider in ["openai", "openai_compatible", "grok", "gemini", "groq", "together", "openrouter", "fireworks", "cerebras", "huggingface", "mistral", "deepseek", "ollama", "lmstudio", "custom"]:
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None
            )

        elif provider == "local":
            return None

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def _call_llm(cls, client, llm_model: LLMModelConfig, system_prompt: str, user_prompt: str, spinner_msg: str = "Calling LLM") -> str:
        """Make LLM call with progress spinner and repetition penalty."""
        if llm_model.provider.lower() == "local":
            return cls._call_local_llm(llm_model, system_prompt, user_prompt, spinner_msg)

        with ProgressSpinner(spinner_msg, prefix="[PromptEnhancer]"):
            if hasattr(client, 'messages'):
                # Anthropic API - doesn't support frequency_penalty directly
                response = client.messages.create(
                    model=llm_model.model,
                    max_tokens=llm_model.max_tokens,
                    temperature=llm_model.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text
            else:
                # OpenAI-compatible API - add frequency_penalty to reduce repetition
                # Get repetition_penalty from config, fallback to default
                rep_penalty = (llm_model.extra_params or {}).get("repetition_penalty", REPETITION_PENALTY)
                freq_penalty = rep_penalty - 1.0  # OpenAI uses 0-2 range, convert from 1.0-2.0
                response = client.chat.completions.create(
                    model=llm_model.model,
                    max_tokens=llm_model.max_tokens,
                    temperature=llm_model.temperature,
                    frequency_penalty=freq_penalty,
                    presence_penalty=0.1,  # Small penalty for reusing any token
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

    @classmethod
    def _call_local_llm(cls, llm_model: LLMModelConfig, system_prompt: str, user_prompt: str, spinner_msg: str = "Generating") -> str:
        """Call local text model (spinner handled by LocalModelClient)."""
        from .llm_providers.sid_llm_local import LocalModelClient

        print(f"[PromptEnhancer] {spinner_msg} (local model)...")
        extra = llm_model.extra_params or {}
        client = LocalModelClient(
            model_name=llm_model.model,
            quantization=extra.get("quantization", "4-bit"),
            device=extra.get("device", "auto"),
            attention_mode=extra.get("attention_mode", "auto"),
            keep_model_loaded=extra.get("keep_model_loaded", True),
            top_p=extra.get("top_p", 0.9),
            repetition_penalty=extra.get("repetition_penalty", 1.2),
            use_torch_compile=extra.get("use_torch_compile", False),
        )
        return client.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=llm_model.max_tokens,
            temperature=llm_model.temperature,
        )

    @classmethod
    def _generate_cache_key(cls, seed: int, prompt: str, detail_level: str, enhance: str, replace: str, add: str, model: str) -> str:
        """Generate a unique cache key based on inputs and seed."""
        key_data = f"{seed}|{prompt}|{detail_level}|{enhance}|{replace}|{add}|{model}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    @classmethod
    def execute(
        cls,
        llm_model: LLMModelConfig,
        prompt: str,
        negative_prompt: str,
        detail_level: str,
        enhance: str,
        replace: str,
        add: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        """Execute prompt enhancement."""
        global _enhancer_cache, _negative_cache
        start_time = time.time()

        try:
            # Header
            print("")
            print("=" * 60)
            print("[PromptEnhancer] Starting prompt enhancement")
            print("=" * 60)

            if not prompt.strip():
                print("[PromptEnhancer] Empty prompt, skipping")
                return comfy_io.NodeOutput(prompt, "", prompt)

            # If no instructions provided, passthrough without LLM call
            if not enhance.strip() and not replace.strip() and not add.strip():
                print("[PromptEnhancer] No instructions (enhance/replace/add), passthrough")
                print("=" * 60)
                print("")
                # Pass through prompt and negative as-is
                return comfy_io.NodeOutput(prompt, negative_prompt, prompt)

            # Generate cache key
            cache_key = cls._generate_cache_key(
                seed, prompt, detail_level, enhance, replace, add, llm_model.model
            )

            # Generate negative cache key (includes the input negative prompt to differentiate)
            negative_cache_key = cls._generate_cache_key(
                seed, prompt + negative_prompt, detail_level, enhance, replace, add, llm_model.model + "_neg"
            )

            # Check cache for both positive and negative
            if cache_key in _enhancer_cache and negative_cache_key in _negative_cache:
                cached_positive = _enhancer_cache[cache_key]
                cached_negative = _negative_cache[negative_cache_key]
                elapsed = time.time() - start_time
                print(f"[PromptEnhancer] Cache HIT (seed: {seed})")
                print(f"[PromptEnhancer] Returning cached result ({len(cached_positive.split())} words positive, {len(cached_negative.split())} words negative)")
                print(f"[PromptEnhancer] Total time: {elapsed:.3f}s")
                print("=" * 60)
                print("")
                return comfy_io.NodeOutput(cached_positive, cached_negative, prompt)

            print(f"[PromptEnhancer] Cache MISS (seed: {seed})")

            # Show input stats
            word_count = len(prompt.split())
            print(f"[PromptEnhancer] Input: {word_count} words")
            print(f"[PromptEnhancer] Mode: {detail_level}")
            print(f"[PromptEnhancer] Provider: {llm_model.provider}")
            print(f"[PromptEnhancer] Model: {llm_model.model}")
            print(f"[PromptEnhancer] Seed: {seed}")

            # Check if local model supports text
            if llm_model.provider.lower() == "local":
                if not llm_model.extra_params.get("supports_text", False):
                    raise ValueError(
                        f"Model '{llm_model.model}' is a vision-only model. "
                        "Please use a (Text) model for prompt enhancement."
                    )

            # Build user instructions
            user_instructions = []
            if enhance.strip():
                user_instructions.append(f"ENHANCE: {enhance.strip()}")
                print(f"[PromptEnhancer] Enhance instruction: {enhance.strip()[:50]}...")
            if replace.strip():
                user_instructions.append(f"REPLACE: {replace.strip()}")
                print(f"[PromptEnhancer] Replace instruction: {replace.strip()[:50]}...")
            if add.strip():
                user_instructions.append(f"ADD: {add.strip()}")
                print(f"[PromptEnhancer] Add instruction: {add.strip()[:50]}...")

            instructions_text = "\n".join(user_instructions) if user_instructions else ""

            # Get client
            print("[PromptEnhancer] Initializing LLM client...")
            client = cls._get_client(llm_model)

            # Process based on detail level
            print("-" * 60)
            if detail_level == "Quick":
                print("[PromptEnhancer] Quick Mode: Cleaning up prompt...")
                enhanced = cls._quick_enhance(client, llm_model, prompt, instructions_text)

            elif detail_level == "Standard":
                print("[PromptEnhancer] Standard Mode: Enhancing prompt...")
                enhanced = cls._standard_enhance(client, llm_model, prompt, instructions_text)

            elif detail_level == "Detailed":
                print("[PromptEnhancer] Detailed Mode: Deep component analysis...")
                enhanced = cls._deep_enhance(client, llm_model, prompt, instructions_text)

            elif detail_level == "Extreme":
                print("[PromptEnhancer] Extreme Mode: Two-pass deep enhancement...")
                enhanced = cls._extreme_enhance(client, llm_model, prompt, instructions_text)

            else:
                enhanced = cls._standard_enhance(client, llm_model, prompt, instructions_text)

            # Store positive in cache (with size limit and cleanup)
            if len(_enhancer_cache) >= _MAX_CACHE_SIZE:
                # Remove oldest 20% of entries when full
                num_to_remove = max(1, _MAX_CACHE_SIZE // 5)
                keys_to_remove = list(_enhancer_cache.keys())[:num_to_remove]
                for key in keys_to_remove:
                    del _enhancer_cache[key]
                print(f"[PromptEnhancer] Positive cache cleanup: removed {num_to_remove} old entries")
                gc.collect()
            _enhancer_cache[cache_key] = enhanced
            print(f"[PromptEnhancer] Positive cached (cache size: {len(_enhancer_cache)})")

            # Process negative prompt
            print("-" * 60)
            if negative_prompt.strip():
                # Enhance the provided negative prompt
                print(f"[PromptEnhancer] Enhancing provided negative prompt ({len(negative_prompt.split())} words)...")
                enhanced_negative = cls._enhance_negative_prompt(client, llm_model, negative_prompt, enhanced, detail_level)
            else:
                # Generate negative prompt from the enhanced positive prompt
                print("[PromptEnhancer] Generating negative prompt from positive...")
                enhanced_negative = cls._generate_negative_prompt(client, llm_model, enhanced, detail_level)

            # Store negative in cache (with size limit and cleanup)
            if len(_negative_cache) >= _MAX_CACHE_SIZE:
                num_to_remove = max(1, _MAX_CACHE_SIZE // 5)
                keys_to_remove = list(_negative_cache.keys())[:num_to_remove]
                for key in keys_to_remove:
                    del _negative_cache[key]
                print(f"[PromptEnhancer] Negative cache cleanup: removed {num_to_remove} old entries")
                gc.collect()
            _negative_cache[negative_cache_key] = enhanced_negative
            print(f"[PromptEnhancer] Negative cached (cache size: {len(_negative_cache)})")

            # Clear references to potentially large objects
            del client
            del instructions_text

            # Results
            elapsed = time.time() - start_time
            output_words = len(enhanced.split())
            neg_words = len(enhanced_negative.split())
            word_diff = output_words - word_count
            diff_str = f"+{word_diff}" if word_diff >= 0 else str(word_diff)

            print("-" * 60)
            print(f"[PromptEnhancer] Complete!")
            print(f"[PromptEnhancer] Positive: {word_count} words -> {output_words} words ({diff_str})")
            print(f"[PromptEnhancer] Negative: {neg_words} words")
            print(f"[PromptEnhancer] Total time: {elapsed:.1f}s")
            print("=" * 60)
            print("")

            return comfy_io.NodeOutput(enhanced, enhanced_negative, prompt)

        except Exception as e:
            print(f"[PromptEnhancer] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return comfy_io.NodeOutput(prompt, "", prompt)

    @classmethod
    def _enforce_length_limit(cls, enhanced: str, original: str, detail_level: str) -> str:
        """Enforce length limits by truncating if necessary."""
        original_words = len(original.split())
        enhanced_words = len(enhanced.split())

        # Get max allowed increase for this detail level
        _, max_increase = LENGTH_LIMITS.get(detail_level, (0, 50))
        max_words = int(original_words * (1 + max_increase / 100))

        if enhanced_words <= max_words:
            return enhanced

        # Need to truncate - try to find a natural break point
        words = enhanced.split()
        truncated = words[:max_words]

        # Try to end at a sentence boundary (., !, or ,)
        result = ' '.join(truncated)
        for i in range(len(result) - 1, max(0, len(result) - 100), -1):
            if result[i] in '.!,':
                result = result[:i+1]
                break

        print(f"[PromptEnhancer] Length enforced: {enhanced_words} -> {len(result.split())} words (max: {max_words})")
        return result

    @classmethod
    def _quick_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, instructions: str) -> str:
        """Quick: Describe and cleanup."""
        original_words = len(prompt.split())
        max_words = int(original_words * 1.2)  # 0-20% increase

        user_prompt = f"""Clean up and clarify this prompt (KEEP IT UNDER {max_words} WORDS):

{prompt}"""

        if instructions:
            user_prompt += f"\n\nAdditional instructions:\n{instructions}"

        user_prompt += f"\n\nCleaned prompt (max {max_words} words):"

        result = cls._call_llm(client, llm_model, QUICK_SYSTEM, user_prompt, "Quick cleanup")
        result = cls._clean_response(result, prompt)
        return cls._enforce_length_limit(result, prompt, "Quick")

    @classmethod
    def _standard_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, instructions: str) -> str:
        """Standard: Basic LLM enhancement."""
        original_words = len(prompt.split())
        max_words = int(original_words * 1.3)  # 10-30% increase

        user_prompt = f"""Enhance this prompt (TARGET: {max_words} words max):

Original word count: {original_words}
Maximum output: {max_words} words

{prompt}"""

        if instructions:
            user_prompt += f"\n\nUser instructions:\n{instructions}"

        user_prompt += f"\n\nEnhanced prompt (max {max_words} words):"

        result = cls._call_llm(client, llm_model, STANDARD_SYSTEM, user_prompt, "Standard enhancement")
        result = cls._clean_response(result, prompt)
        return cls._enforce_length_limit(result, prompt, "Standard")

    @classmethod
    def _deep_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, instructions: str) -> str:
        """Detailed: Deep component-based enhancement."""
        original_words = len(prompt.split())
        max_words = int(original_words * 1.4)  # 20-40% increase

        user_prompt = f"""Enhance this prompt while respecting length limits.

WORD LIMITS:
- Original: {original_words} words
- Maximum output: {max_words} words (20-40% increase)
- Do NOT exceed {max_words} words

=== ORIGINAL PROMPT ===
{prompt}
=== END ORIGINAL PROMPT ===

Add only HIGH-IMPACT details. Quality over quantity."""

        if instructions:
            user_prompt += f"\n\nUser instructions:\n{instructions}"

        user_prompt += f"\n\nEnhanced prompt (max {max_words} words):"

        result = cls._call_llm(client, llm_model, DEEP_SYSTEM, user_prompt, "Deep enhancement")
        result = cls._clean_response(result, prompt)
        return cls._enforce_length_limit(result, prompt, "Detailed")

    @classmethod
    def _extreme_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, instructions: str) -> str:
        """Extreme: Single pass with strict length limit."""
        original_words = len(prompt.split())
        max_words = int(original_words * 1.4)  # 30-40% increase

        user_prompt = f"""Enhance this prompt with maximum detail while respecting length limits.

STRICT WORD LIMITS:
- Original: {original_words} words
- Maximum output: {max_words} words (30-40% increase)
- Do NOT exceed {max_words} words under any circumstances

=== ORIGINAL PROMPT ===
{prompt}
=== END ORIGINAL PROMPT ===

Add only the most impactful micro-details. Every word must count."""

        if instructions:
            user_prompt += f"\n\nUser instructions:\n{instructions}"

        user_prompt += f"\n\nEnhanced prompt (STRICT MAX {max_words} words):"

        # Single pass only for Extreme (removed second pass to control length)
        result = cls._call_llm(client, llm_model, EXTREME_SYSTEM.format(prompt=""), user_prompt, "Extreme enhancement")
        result = cls._clean_response(result, prompt)
        return cls._enforce_length_limit(result, prompt, "Extreme")

    @classmethod
    def _clean_response(cls, response: str, original: str) -> str:
        """Clean up the LLM response."""
        if not response:
            return original

        # Remove common prefixes
        prefixes = [
            r'^(?:Here\'?s?|This is|The|An?) (?:the |an? )?(?:enhanced|improved|modified|refined|cleaned) (?:prompt|version)[:\s]*\n*',
            r'^(?:Enhanced|Improved|Modified|Refined|Cleaned) (?:prompt|version)[:\s]*\n*',
            r'^(?:Sure|Certainly|Of course)[,!]?[^:]*[:\s]*\n*',
        ]
        for p in prefixes:
            response = re.sub(p, '', response, flags=re.IGNORECASE | re.MULTILINE)

        # Remove markdown code blocks
        code_match = re.search(r'```(?:\w+)?\s*([\s\S]*?)```', response)
        if code_match:
            response = code_match.group(1)

        # Remove surrounding quotes
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1]

        # Remove trailing explanations
        explanations = [
            r'\n\n(?:This enhanced|The enhanced|I\'ve enhanced|Note:|Key changes:|Changes made:|I have)[\s\S]*$',
            r'\n\n(?:Let me know|Feel free|I hope|This version)[\s\S]*$',
        ]
        for p in explanations:
            response = re.sub(p, '', response, flags=re.IGNORECASE)

        return response.strip() or original

    @classmethod
    def _generate_negative_prompt(cls, client, llm_model: LLMModelConfig, positive_prompt: str, detail_level: str) -> str:
        """Generate a negative prompt based on the enhanced positive prompt."""
        # Build system prompt based on detail level
        if detail_level == "Quick":
            system_prompt = """You are an expert at creating negative prompts for AI image generation.
Generate a concise negative prompt that avoids common image quality issues.
Output ONLY the negative prompt - no explanations, no markdown, no quotes."""

        elif detail_level == "Standard":
            system_prompt = """You are an expert at creating negative prompts for AI image generation.
Based on the positive prompt, generate a negative prompt that:
1. Avoids quality issues (blur, noise, artifacts, distortion)
2. Excludes anatomical errors (extra limbs, deformed features)
3. Prevents style conflicts with the intended aesthetic
Output ONLY the negative prompt - no explanations, no markdown, no quotes."""

        else:  # Detailed or Extreme
            system_prompt = """You are an expert at creating comprehensive negative prompts for AI image generation.

Based on the positive prompt, generate a detailed negative prompt that addresses:

## QUALITY ISSUES
- Image artifacts, blur, noise, pixelation, compression artifacts
- Low resolution, watermarks, text, signatures

## ANATOMICAL ERRORS
- Extra limbs, missing limbs, deformed hands, extra fingers
- Mutated features, distorted faces, asymmetric eyes
- Unnatural body proportions, floating limbs

## STYLE CONFLICTS
- Elements that would conflict with the intended aesthetic
- Inappropriate lighting or color issues for the scene
- Clashing artistic styles

## COMPOSITION ISSUES
- Bad cropping, cut off body parts, awkward framing
- Cluttered backgrounds, distracting elements

Output ONLY the negative prompt as a comma-separated list - no explanations, no markdown, no quotes."""

        user_prompt = f"""Based on this positive prompt, generate an appropriate negative prompt:

POSITIVE PROMPT:
{positive_prompt}

Generate a negative prompt that will help avoid quality issues and errors while maintaining the intended style. Output ONLY the negative prompt:"""

        try:
            result = cls._call_llm(client, llm_model, system_prompt, user_prompt, "Generating negative prompt")
            return cls._clean_negative_response(result)
        except Exception as e:
            print(f"[PromptEnhancer] Error generating negative prompt: {e}")
            return cls._get_default_negative_prompt(detail_level)

    @classmethod
    def _enhance_negative_prompt(cls, client, llm_model: LLMModelConfig, negative_prompt: str, positive_prompt: str, detail_level: str) -> str:
        """Enhance a provided negative prompt based on the positive prompt context."""
        if detail_level == "Quick":
            system_prompt = """You are a prompt editor. Clean up and clarify the negative prompt.
- Fix grammar and spelling
- Remove redundant terms
- Ensure clarity
CRITICAL: Keep ALL original terms intact. Only fix language issues.
Output ONLY the cleaned negative prompt."""

        elif detail_level == "Standard":
            system_prompt = """You are a prompt engineer specializing in negative prompts for AI image generation.
Enhance the negative prompt while preserving all original terms.
You may add complementary negative terms that align with avoiding issues in the positive prompt context.
Output ONLY the enhanced negative prompt - no explanations."""

        else:  # Detailed or Extreme
            system_prompt = """You are an expert prompt engineer specializing in negative prompts for AI image generation.

RULES:
1. PRESERVE every term from the original negative prompt
2. ADD complementary negative terms based on the positive prompt context
3. Include terms to avoid: quality issues, anatomical errors, style conflicts
4. The output should be LONGER than the input, containing ALL original terms plus additions

Output ONLY the enhanced negative prompt as a comma-separated list - no explanations, no markdown, no quotes."""

        user_prompt = f"""Enhance this negative prompt based on the positive prompt context:

POSITIVE PROMPT (for context):
{positive_prompt}

NEGATIVE PROMPT TO ENHANCE:
{negative_prompt}

Enhanced negative prompt:"""

        try:
            result = cls._call_llm(client, llm_model, system_prompt, user_prompt, "Enhancing negative prompt")
            cleaned = cls._clean_negative_response(result)
            # If enhancement failed or returned something too short, return original
            if not cleaned or len(cleaned) < len(negative_prompt) // 2:
                return negative_prompt
            return cleaned
        except Exception as e:
            print(f"[PromptEnhancer] Error enhancing negative prompt: {e}")
            return negative_prompt

    @classmethod
    def _clean_negative_response(cls, response: str) -> str:
        """Clean up the negative prompt LLM response."""
        if not response:
            return ""

        # Remove common prefixes
        prefixes = [
            r'^(?:Here\'?s?|This is|The|A) (?:the |a )?negative prompt[:\s]*\n*',
            r'^Negative prompt[:\s]*\n*',
            r'^(?:Sure|Certainly|Of course)[,!]?[^:]*[:\s]*\n*',
            r'^(?:Enhanced|Improved) negative prompt[:\s]*\n*',
        ]
        for p in prefixes:
            response = re.sub(p, '', response, flags=re.IGNORECASE)

        # Remove markdown code blocks
        match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', response)
        if match:
            response = match.group(1)

        # Remove surrounding quotes
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1]

        # Remove trailing explanations
        explanations = [
            r'\n\n(?:This|The|I\'ve|Note:|This negative)[\s\S]*$',
            r'\n\n(?:Let me|Feel free|I hope)[\s\S]*$',
        ]
        for p in explanations:
            response = re.sub(p, '', response, flags=re.IGNORECASE)

        return response.strip()

    @classmethod
    def _get_default_negative_prompt(cls, detail_level: str) -> str:
        """Return a sensible default negative prompt based on detail level."""
        if detail_level == "Quick":
            return "blur, low quality, distorted"

        elif detail_level == "Standard":
            return "blur, noise, artifacts, distortion, low quality, deformed, extra limbs, bad anatomy"

        else:  # Detailed or Extreme
            return ("blur, noise, artifacts, distortion, low quality, pixelated, watermark, text, signature, "
                    "deformed, extra limbs, missing limbs, extra fingers, mutated hands, bad anatomy, "
                    "asymmetric eyes, distorted face, unnatural proportions, floating limbs, "
                    "bad cropping, cut off, cluttered background")
