"""RefineEngine - Multi-pass VLM questioning for detailed prompts."""

from typing import Optional

from PIL import Image

from .modules import ModuleSelector, REFINE_CONFIG
from .combiner import ResponseCombiner


class RefineEngine:
    """
    Multi-pass VLM questioning engine for Refine mode.

    Supports configurable pass counts (2-14) and two execution modes:
    - Sequential: Run modules one after another, each builds on context
    - Parallel: Run all modules without context sharing (faster)
    """

    # Available pass counts
    PASS_COUNTS = ["2", "4", "6", "8", "10", "12", "14"]

    # Legacy modes for backwards compatibility
    MODES = ["sequential", "parallel"]

    def __init__(self, vlm_wrapper, config: Optional[dict] = None):
        """
        Initialize RefineEngine.

        Args:
            vlm_wrapper: Loaded VLM model wrapper (from ModelFactory)
            config: Optional config override (defaults to refine_modules.json)
        """
        self.vlm = vlm_wrapper
        self.config = config or REFINE_CONFIG
        self.selector = ModuleSelector(self.config)
        self.combiner = ResponseCombiner(self.config)

    # Detail level settings: (instruction, max_tokens_per_module)
    # Note: With consolidated modules (6 instead of 22), each module needs more tokens
    DETAIL_LEVELS = {
        "brief": ("Be concise. 2-3 sentences per aspect.", 512),
        "moderate": ("Provide moderate detail. 3-5 sentences per aspect.", 1024),
        "detailed": ("Provide rich detail for each aspect listed.", 2048),
        "comprehensive": ("Provide comprehensive, exhaustive detail for every aspect.", 4096),
    }

    def run(
        self,
        image: Image.Image,
        metadata: dict,
        target_format: str = "z-image",
        mode: str = "parallel",
        pass_count: str = "6",
        detail_level: str = "detailed",
        temperature: float = 0.3,
        tag_threshold: float = 0.35,
        verbose: bool = False,
    ) -> str:
        """
        Run multi-pass refinement.

        Args:
            image: PIL Image
            metadata: Metadata dict from ImageAnalysis
            target_format: Target model format (z-image, sdxl, flux, pony, etc.)
            mode: "sequential" (with context) or "parallel" (no context, faster)
            pass_count: Number of VLM passes - "2", "4", "6", "8", "10", "12", "14"
            detail_level: "brief", "moderate", "detailed", or "comprehensive"
            temperature: Generation temperature
            tag_threshold: Minimum confidence to include tags (0.1-0.9)
            verbose: Enable logging

        Returns:
            Combined prompt string
        """
        # Store threshold for use in module prompt building
        self._tag_threshold = tag_threshold
        # Get detail level settings
        detail_config = self.DETAIL_LEVELS.get(detail_level, self.DETAIL_LEVELS["detailed"])
        self._detail_instruction = detail_config[0]
        self._module_max_tokens = detail_config[1]

        if verbose:
            print(f"[RefineEngine] Detail: {detail_level} (max {self._module_max_tokens} tokens/module)")

        # Validate mode
        if mode not in self.MODES:
            print(f"[RefineEngine] Unknown mode '{mode}', using 'parallel'")
            mode = "parallel"

        # Validate pass count
        if pass_count not in self.PASS_COUNTS:
            print(f"[RefineEngine] Unknown pass_count '{pass_count}', using '6'")
            pass_count = "6"

        # Get modules from pass template
        pass_templates = self.config.get("pass_templates", {})
        modules = pass_templates.get(pass_count, pass_templates.get("6", []))

        if verbose:
            print(f"[RefineEngine] Mode: {mode}, Passes: {pass_count}, Detail: {detail_level}")
            print(f"[RefineEngine] Selected modules: {modules}")

        if not modules:
            print("[RefineEngine] No modules selected, using fallback")
            return self._run_fallback(image, metadata, temperature, verbose)

        # Run based on mode
        if mode == "sequential":
            responses, prompts_used = self._run_sequential(
                image, metadata, modules, temperature, verbose
            )
        else:  # parallel or adaptive
            responses, prompts_used = self._run_parallel(
                image, metadata, modules, temperature, verbose
            )

        # Combine responses
        result = self.combiner.combine(responses, target_format, verbose)

        # Store synthesis info for caller to access
        self.last_run_info = {
            "mode": mode,
            "pass_count": pass_count,
            "detail_level": detail_level,
            "modules_run": modules,
            "questions_and_answers": [
                {"module": module, "question": prompts_used.get(module, ""), "answer": responses.get(module, "")}
                for module in modules
            ],
            "final_prompt": result,
        }

        return result

    def _run_sequential(
        self,
        image: Image.Image,
        metadata: dict,
        modules: list[str],
        temperature: float,
        verbose: bool,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Run modules sequentially, building context.

        Each module sees the previous module's output as additional context.
        Most thorough but slowest.

        Returns:
            (responses, prompts_used) - both are dicts keyed by module name
        """
        import time
        import comfy.utils

        responses = {}
        prompts_used = {}
        accumulated_context = []
        start_time = time.time()

        if verbose:
            print(f"[RefineEngine] Running {len(modules)} modules sequentially (with context)...")
            print(f"[RefineEngine] Modules: {', '.join(modules)}")

        # Create progress bar for module completion
        pbar = comfy.utils.ProgressBar(len(modules))

        for i, module_name in enumerate(modules):
            module_start = time.time()

            if verbose:
                print(f"[RefineEngine] >> Starting [{i+1}/{len(modules)}]: {module_name}")

            # Build prompt with tag context
            base_prompt = self.selector.build_prompt(module_name, metadata, self._tag_threshold)

            # Store the base prompt (question) before adding instructions
            prompts_used[module_name] = base_prompt

            # Add detail level instruction
            base_prompt = f"{base_prompt}\n\n{self._detail_instruction}"

            # Add accumulated context from previous passes
            if accumulated_context:
                context_str = "\n\nPrevious observations:\n" + "\n".join(accumulated_context[-3:])  # Last 3
                prompt = base_prompt + context_str
                if verbose:
                    print(f"[RefineEngine]    Adding context from {len(accumulated_context)} previous modules")
            else:
                prompt = base_prompt

            if verbose:
                prompt_preview = base_prompt[:100].replace('\n', ' ')
                print(f"[RefineEngine]    Prompt: \"{prompt_preview}...\"")

            # Generate response (use detail-level token limit)
            response = self._call_vlm(image, prompt, self._module_max_tokens, temperature)
            elapsed = time.time() - module_start

            if response:
                responses[module_name] = response
                # Add to context for next pass (summarized)
                summary = response[:200] + "..." if len(response) > 200 else response
                accumulated_context.append(f"[{module_name}]: {summary}")

                if verbose:
                    print(f"[RefineEngine] << Completed [{i+1}/{len(modules)}]: {module_name} ({len(response)} chars, {elapsed:.1f}s)")
            else:
                if verbose:
                    print(f"[RefineEngine] << Completed [{i+1}/{len(modules)}]: {module_name} (empty response, {elapsed:.1f}s)")

            # Update progress bar after each module completes
            pbar.update(1)

        total_time = time.time() - start_time
        if verbose:
            print(f"[RefineEngine] All {len(modules)} modules completed in {total_time:.1f}s")
            print(f"[RefineEngine] Responses collected: {len(responses)}/{len(modules)}")

        return responses, prompts_used

    def _run_parallel(
        self,
        image: Image.Image,
        metadata: dict,
        modules: list[str],
        temperature: float,
        verbose: bool,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Run modules in parallel for speed.

        No context sharing between modules.
        Fastest option.

        Returns:
            (responses, prompts_used) - both are dicts keyed by module name
        """
        import time
        import comfy.utils

        responses = {}
        prompts_used = {}
        start_time = time.time()

        # Build all prompts first
        prompts = {}
        for module_name in modules:
            base_prompt = self.selector.build_prompt(module_name, metadata, self._tag_threshold)
            prompts_used[module_name] = base_prompt  # Store base question
            prompts[module_name] = f"{base_prompt}\n\n{self._detail_instruction}"
            if verbose:
                prompt_preview = base_prompt[:100].replace('\n', ' ')
                print(f"[RefineEngine] Prepared {module_name}: \"{prompt_preview}...\"")

        if verbose:
            print(f"[RefineEngine] Running {len(modules)} modules (no context sharing)...")
            print(f"[RefineEngine] Modules: {', '.join(modules)}")

        # Create progress bar for module completion
        pbar = comfy.utils.ProgressBar(len(modules))

        # Run modules sequentially (GPU can only handle one VLM call at a time)
        # "Parallel" mode means no context sharing, not actual parallel execution
        for i, module_name in enumerate(modules):
            module_start = time.time()

            if verbose:
                print(f"[RefineEngine] >> Starting [{i+1}/{len(modules)}]: {module_name}")

            try:
                response = self._call_vlm(image, prompts[module_name], self._module_max_tokens, temperature)
                elapsed = time.time() - module_start

                if response:
                    responses[module_name] = response
                    if verbose:
                        print(f"[RefineEngine] << Completed [{i+1}/{len(modules)}]: {module_name} ({len(response)} chars, {elapsed:.1f}s)")
                else:
                    if verbose:
                        print(f"[RefineEngine] << Completed [{i+1}/{len(modules)}]: {module_name} (empty response, {elapsed:.1f}s)")
            except Exception as e:
                print(f"[RefineEngine] Error in {module_name}: {e}")

            # Update progress bar after each module completes
            pbar.update(1)

        total_time = time.time() - start_time
        if verbose:
            print(f"[RefineEngine] All {len(modules)} modules completed in {total_time:.1f}s")
            print(f"[RefineEngine] Responses collected: {len(responses)}/{len(modules)}")

        return responses, prompts_used

    def _call_vlm(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Make a single VLM call.

        Args:
            image: PIL Image
            prompt: Question prompt
            max_tokens: Max response tokens
            temperature: Generation temperature

        Returns:
            VLM response string
        """
        try:
            from ..models.base import GenerationConfig, CaptionMode

            gen_config = GenerationConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
            )

            # Get system prompt
            system_prompt = self.config.get(
                "system_prompt",
                "You are an expert image analyst. Describe exactly what you observe with precision and confidence."
            )

            response = self.vlm.generate(
                image=image,
                mode=CaptionMode.DETAILED,
                config=gen_config,
                custom_prompt=prompt,
                system_prompt=system_prompt,
            )

            return response.strip() if response else ""

        except Exception as e:
            print(f"[RefineEngine] VLM call error: {e}")
            return ""

    def _run_fallback(
        self,
        image: Image.Image,
        metadata: dict,
        temperature: float,
        verbose: bool,
    ) -> str:
        """Fallback when no modules selected - basic description."""
        prompt = """Describe this image in detail for AI image generation.
Include:
- Subject appearance (if person: age, features, expression, hair)
- Clothing and accessories
- Pose and body language
- Lighting quality and direction
- Composition and framing
- Background and setting
- Overall mood and style

Write as comma-separated descriptive phrases."""

        response = self._call_vlm(image, prompt, self._module_max_tokens, temperature)
        return response if response else "Error: Could not generate description"

    def get_available_modules(self) -> list[str]:
        """Get list of all available module names."""
        return list(self.config.get("modules", {}).keys())

    def get_mode_info(self, mode: str) -> dict:
        """Get information about a specific mode."""
        modes_config = self.config.get("modes", {})
        return modes_config.get(mode, {})
