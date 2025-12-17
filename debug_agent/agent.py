# -*- coding: utf-8 -*-
"""
Prompt Debug Agent

Agentic evaluation system using Claude Opus 4.5 with tool use.
"""

import json
import base64
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image
import io

from .knowledge import KnowledgeBase
from .tools import AGENT_TOOLS, ToolExecutor
from .storage import ResultsStorage
from .system_info import get_system_info


class PromptDebugAgent:
    """
    Intelligent agent for evaluating prompt generation quality.

    Uses Claude Opus 4.5 with tool use for:
    - Analyzing source and output images
    - Evaluating prompts against best practices
    - Providing model-specific recommendations
    - Optionally searching for latest techniques
    - Accumulating learned insights
    """

    EVALUATOR_MODEL = "claude-opus-4-5-20251101"

    SYSTEM_PROMPT = """You are an expert prompt evaluation agent for Z-Image prompting and vision model prompt engineering.

Your task is to evaluate the quality of a generated prompt by:
1. Analyzing the source image to understand what needs to be described
2. Evaluating how well the prompt captures the source image
3. Analyzing the output image to see how well it matches
4. Comparing against Z-Image best practices
5. Providing model-specific recommendations

You have access to tools for:
- Reading the local knowledge base (vocabulary, best practices, model tips)
- Searching the web for latest prompting techniques (if Tavily key provided)
- Updating the knowledge base with new insights
- Evaluating the prompt structure

IMPORTANT GUIDELINES:
- Be thorough but concise in your analysis
- Focus on actionable improvements
- Score objectively using the rubric in the knowledge base
- Provide specific examples when suggesting improvements
- Consider the model's known strengths and weaknesses

When you have completed your evaluation, provide your response as JSON with this EXACT structure:
```json
{
  "scores": {
    "source_alignment": {"score": 8.5, "reason": "..."},
    "output_quality": {"score": 7.0, "reason": "..."},
    "structure": {"score": 8.0, "reason": "..."},
    "vocabulary": {"score": 7.5, "reason": "..."},
    "overall": {"score": 7.8, "reason": "..."}
  },
  "analysis": {
    "detected_elements": ["..."],
    "mentioned_in_prompt": ["..."],
    "missing_elements": ["..."],
    "hallucinated_elements": ["..."]
  },
  "recommendations": ["recommendation 1", "recommendation 2", "..."],
  "improved_prompt": "Your complete rewritten prompt that addresses all issues and follows best practices. This should be a full, ready-to-use prompt."
}
```

CRITICAL: The "improved_prompt" field MUST contain a complete, improved version of the prompt that can be used directly. Do not leave it empty or put placeholder text.
"""

    def __init__(
        self,
        anthropic_api_key: str,
        tavily_api_key: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        storage: Optional[ResultsStorage] = None,
    ):
        """
        Initialize the debug agent.

        Args:
            anthropic_api_key: API key for Anthropic Claude
            tavily_api_key: Optional API key for Tavily search
            knowledge_base: Optional KnowledgeBase instance
            storage: Optional ResultsStorage instance
        """
        self.anthropic_key = anthropic_api_key
        self.tavily_key = tavily_api_key

        # Initialize Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        # Initialize Tavily client if key provided
        self.tavily = None
        if tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily = TavilyClient(api_key=tavily_api_key)
            except ImportError:
                print("[DebugAgent] Tavily not installed. Web search disabled.")

        # Initialize knowledge base and storage
        self.knowledge = knowledge_base or KnowledgeBase()
        self.storage = storage or ResultsStorage()

        # Context for current evaluation
        self.context: Dict[str, Any] = {}

    def evaluate(
        self,
        source_image: Any,
        output_image: Any,
        prompt: str,
        model_config: Dict[str, Any],
        generator_settings: Optional[Dict[str, Any]] = None,
        save_results: bool = True,
        update_knowledge: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a full evaluation of the prompt.

        Args:
            source_image: Source image (PIL Image, tensor, or base64)
            output_image: Output image (PIL Image, tensor, or base64)
            prompt: The generated prompt to evaluate
            model_config: Configuration of the model that generated the prompt
            generator_settings: Optional generator settings (mode, length, etc.)
            save_results: Whether to save results to disk
            update_knowledge: Whether to update knowledge base with insights

        Returns:
            Full evaluation dictionary
        """
        start_time = time.time()

        # Convert images to base64
        source_b64 = self._image_to_base64(source_image)
        output_b64 = self._image_to_base64(output_image)

        # Set up context
        self.context = {
            "source_image": source_b64,
            "output_image": output_b64,
            "prompt": prompt,
            "model_config": model_config,
            "generator_settings": generator_settings or {},
            "save_results": save_results,
            "update_knowledge": update_knowledge,
        }

        # Initialize tool executor
        tool_executor = ToolExecutor(
            knowledge=self.knowledge,
            context=self.context,
            tavily_client=self.tavily
        )

        # Get system info
        system_info = get_system_info()

        # Build initial message
        messages = self._build_initial_message(source_b64, output_b64, prompt, model_config)

        # Run agent loop
        print("[DebugAgent] Starting evaluation...")
        evaluation = self._run_agent_loop(messages, tool_executor)

        # Calculate timing
        evaluation["timing"] = {
            "total_seconds": round(time.time() - start_time, 2)
        }

        # Add metadata
        evaluation["system_info"] = system_info
        evaluation["model_config"] = model_config
        evaluation["generator_settings"] = generator_settings or {}

        # Save results if requested
        if save_results:
            print("[DebugAgent] Saving results...")
            save_info = self.storage.save_full_session(
                source_image=source_image,
                output_image=output_image,
                prompt=prompt,
                model_config=model_config,
                evaluation=evaluation,
                generator_settings=generator_settings,
                system_info=system_info
            )
            evaluation["saved_to"] = save_info["path"]
            print(f"[DebugAgent] Results saved to: {save_info['path']}")

        # Update knowledge base if requested
        if update_knowledge:
            self._update_learned_insights(evaluation)

        print(f"[DebugAgent] Evaluation complete in {evaluation['timing']['total_seconds']:.1f}s")

        return evaluation

    def _image_to_base64(self, image: Any) -> str:
        """Convert image to base64 string."""
        if isinstance(image, str):
            # Already base64
            return image

        # Convert to PIL Image if needed
        if hasattr(image, 'cpu'):
            # PyTorch tensor
            import numpy as np
            if len(image.shape) == 4:
                image = image[0]
            np_image = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(np_image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_initial_message(
        self,
        source_b64: str,
        output_b64: str,
        prompt: str,
        model_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build the initial message with images and context."""
        user_content = [
            {
                "type": "text",
                "text": "Please evaluate this prompt generation. The first image is the SOURCE (original input), the second is the OUTPUT (generated result)."
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": source_b64
                }
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": output_b64
                }
            },
            {
                "type": "text",
                "text": f"""
PROMPT TO EVALUATE:
{prompt}

MODEL CONFIGURATION:
- Provider: {model_config.get('provider', 'unknown')}
- Model: {model_config.get('model', 'unknown')}
- Temperature: {model_config.get('temperature', 'unknown')}
- Extra params: {json.dumps(model_config.get('extra_params', {}), indent=2)}

Please:
1. First, read the knowledge base (zimage_best_practices and zimage_vocabulary)
2. Analyze the source image thoroughly
3. Evaluate the prompt against best practices
4. Compare source and output images
5. Get model-specific tips for {model_config.get('model', 'the model')}
6. Generate comprehensive evaluation with scores and recommendations

Provide your final evaluation as a structured JSON response.
"""
            }
        ]

        return [{"role": "user", "content": user_content}]

    def _run_agent_loop(
        self,
        messages: List[Dict[str, Any]],
        tool_executor: ToolExecutor,
        max_iterations: int = 15
    ) -> Dict[str, Any]:
        """
        Run the agent loop until completion.

        Args:
            messages: Initial messages
            tool_executor: ToolExecutor instance
            max_iterations: Maximum number of iterations

        Returns:
            Final evaluation dictionary
        """
        agent_steps = []

        for i in range(max_iterations):
            print(f"[DebugAgent] Step {i + 1}...")

            # Call Claude
            response = self.client.messages.create(
                model=self.EVALUATOR_MODEL,
                max_tokens=8192,
                system=self.SYSTEM_PROMPT,
                tools=AGENT_TOOLS,
                messages=messages
            )

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract final result from response
                return self._extract_final_result(response, agent_steps)

            # Process tool calls
            tool_results = []
            assistant_content = response.content

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    print(f"[DebugAgent] Calling tool: {tool_name}")

                    # Execute tool
                    step_start = time.time()
                    result = tool_executor.execute(tool_name, tool_input)
                    step_duration = time.time() - step_start

                    # Record step
                    agent_steps.append({
                        "step": i + 1,
                        "tool": tool_name,
                        "input": tool_input,
                        "status": result.get("status", "success"),
                        "duration_ms": round(step_duration * 1000)
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        # Max iterations reached
        return {
            "status": "incomplete",
            "reason": "max_iterations_reached",
            "agent_steps": agent_steps
        }

    def _extract_final_result(
        self,
        response,
        agent_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract the final evaluation from Claude's response.

        Args:
            response: Claude API response
            agent_steps: List of agent steps taken

        Returns:
            Evaluation dictionary
        """
        # Find text content in response
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        # Try to parse JSON from response
        evaluation = {}

        # Look for JSON block
        json_match = None
        if "```json" in text_content:
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text_content)
        elif "{" in text_content and "}" in text_content:
            # Try to find JSON object
            import re
            json_match = re.search(r'\{[\s\S]*\}', text_content)

        if json_match:
            try:
                evaluation = json.loads(json_match.group(1) if "```json" in text_content else json_match.group(0))
            except json.JSONDecodeError:
                evaluation = {"raw_response": text_content}
        else:
            evaluation = {"raw_response": text_content}

        # Add agent steps
        evaluation["agent_steps"] = {
            "steps_completed": agent_steps,
            "total_steps": len(agent_steps)
        }

        # Ensure we have the evaluator info
        if "evaluator" not in evaluation:
            evaluation["evaluator"] = {
                "provider": "anthropic",
                "model": self.EVALUATOR_MODEL
            }

        return evaluation

    def _update_learned_insights(self, evaluation: Dict[str, Any]):
        """Update knowledge base with insights from this evaluation."""
        try:
            insight = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": self.context.get("model_config", {}).get("model", "unknown"),
                "overall_score": evaluation.get("evaluation", {}).get("scores", {}).get("overall", {}).get("score"),
            }

            # Extract key learnings
            if "recommendations" in evaluation:
                insight["recommendations"] = evaluation["recommendations"].get("prompt_improvements", [])[:3]

            self.knowledge.add_learned_insight(insight)
            print("[DebugAgent] Updated learned insights")
        except Exception as e:
            print(f"[DebugAgent] Failed to update insights: {e}")

    def get_knowledge_status(self) -> Dict[str, Any]:
        """Get status of the knowledge base."""
        return self.knowledge.get_status()
