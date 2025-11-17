"""
LLM-based executor for dependency analysis and step execution.

This module provides dependency analysis and step execution using any LLM backend
(local or cloud) through the unified LLM client abstraction.

Replaces gemini_executor.py with a backend-agnostic implementation.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .workflow_types import Plan, Step, StepType
from .llm_client import BaseLLMClient, LLMConfig, create_llm_client, load_llm_config_from_yaml

logger = logging.getLogger(__name__)


class BatchDefinition:
    """Represents a batch of steps to execute."""

    def __init__(self, batch_id: int, parallel: bool, steps: List[int], description: str):
        self.batch_id = batch_id
        self.parallel = parallel
        self.step_indices = steps
        self.description = description

    def __repr__(self):
        mode = "parallel" if self.parallel else "sequential"
        return f"Batch {self.batch_id} ({mode}): {len(self.step_indices)} steps - {self.description}"


class DependencyAnalyzer:
    """
    Analyzes plan dependencies using LLM to create execution batches.

    Now supports multiple LLM backends (local, Gemini, OpenAI) through
    the unified LLM client abstraction.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, config_path: Optional[Path] = None):
        """
        Initialize dependency analyzer.

        Args:
            llm_client: LLM client instance (if None, creates from config)
            config_path: Path to config.yaml (if None, uses default location)
        """
        if llm_client is None:
            # Load configuration
            if config_path is None:
                script_dir = Path(__file__).parent.parent
                config_path = script_dir / 'config' / 'config.yaml'

            try:
                llm_config = load_llm_config_from_yaml(str(config_path))
                llm_client = create_llm_client(llm_config)
                logger.info(f"Initialized DependencyAnalyzer with {llm_config.backend} backend")
            except Exception as e:
                logger.warning(f"Failed to load LLM config: {e}. Using default local config.")
                llm_client = create_llm_client(LLMConfig(backend="local"))

        self.llm_client = llm_client

    def analyze_plan(self, plan: Plan) -> List[BatchDefinition]:
        """
        Analyze a plan and return batch definitions.

        Args:
            plan: The research plan to analyze

        Returns:
            List of BatchDefinition objects in execution order
        """
        logger.info("=== Starting Dependency Analysis with LLM ===")

        # Load the prompt template
        script_dir = Path(__file__).parent.parent
        prompt_path = script_dir / 'config' / 'parallel_prompt.md'

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except FileNotFoundError:
            logger.error(f"parallel_prompt.md not found at {prompt_path}")
            return self._fallback_heuristic_batching(plan)

        # Convert plan to JSON for LLM
        plan_json = self._plan_to_json(plan)
        user_input = f"Research Plan to Analyze:\n\n```json\n{json.dumps(plan_json, indent=2)}\n```"

        try:
            logger.info("Sending plan to LLM for dependency analysis...")

            # Use LLM client abstraction
            response_text = self.llm_client.generate_sync(
                prompt=user_input,
                system_prompt=system_prompt,
                json_mode=True
            )

            logger.debug(f"LLM Response: {response_text[:200]}...")

            # Clean markdown if present
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            elif response_text.strip().startswith("```"):
                # Handle generic code blocks
                response_text = response_text.strip()[3:-3].strip()

            batch_data = json.loads(response_text)

            # Convert to BatchDefinition objects
            batches = []
            for batch_info in batch_data.get("batches", []):
                batch = BatchDefinition(
                    batch_id=batch_info["batch_id"],
                    parallel=batch_info["parallel"],
                    steps=batch_info["steps"],
                    description=batch_info["description"]
                )
                batches.append(batch)
                logger.info(f"  {batch}")

            logger.info(f"LLM Analysis: {batch_data.get('reasoning', 'N/A')}")
            logger.info(f"Expected Speedup: {batch_data.get('expected_speedup', 'N/A')}")
            logger.info(f"Created {len(batches)} batches")

            return batches

        except Exception as e:
            logger.error(f"LLM dependency analysis failed: {e}")
            logger.info("Falling back to heuristic batching")
            return self._fallback_heuristic_batching(plan)

    def _plan_to_json(self, plan: Plan) -> Dict[str, Any]:
        """Convert Plan object to JSON dict."""
        return {
            "locale": plan.locale,
            "has_enough_context": plan.has_enough_context,
            "thought": plan.thought,
            "title": plan.title,
            "steps": [
                {
                    "need_search": step.need_search,
                    "title": step.title,
                    "description": step.description,
                    "step_type": step.step_type.value
                }
                for step in plan.steps
            ]
        }

    def _fallback_heuristic_batching(self, plan: Plan) -> List[BatchDefinition]:
        """
        Fallback heuristic batching strategy.

        Groups all research steps in one parallel batch,
        then processing steps sequentially.
        """
        logger.info("Using heuristic batching strategy")

        research_indices = []
        processing_indices = []

        for i, step in enumerate(plan.steps):
            if step.step_type == StepType.RESEARCH:
                research_indices.append(i)
            else:
                processing_indices.append(i)

        batches = []

        if research_indices:
            batches.append(BatchDefinition(
                batch_id=1,
                parallel=True,
                steps=research_indices,
                description="All research steps in parallel"
            ))

        # Add processing steps individually
        for idx, proc_idx in enumerate(processing_indices):
            batches.append(BatchDefinition(
                batch_id=len(batches) + 1,
                parallel=False,
                steps=[proc_idx],
                description=f"Processing step: {plan.steps[proc_idx].title}"
            ))

        return batches


class StepExecutor:
    """
    Executes individual steps using any LLM backend.

    Renamed from GeminiStepExecutor to reflect backend-agnostic design.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, config_path: Optional[Path] = None):
        """
        Initialize step executor.

        Args:
            llm_client: LLM client instance (if None, creates from config)
            config_path: Path to config.yaml (if None, uses default location)
        """
        if llm_client is None:
            # Load configuration
            if config_path is None:
                script_dir = Path(__file__).parent.parent
                config_path = script_dir / 'config' / 'config.yaml'

            try:
                llm_config = load_llm_config_from_yaml(str(config_path))
                llm_client = create_llm_client(llm_config)
                logger.info(f"Initialized StepExecutor with {llm_config.backend} backend")
            except Exception as e:
                logger.warning(f"Failed to load LLM config: {e}. Using default local config.")
                llm_client = create_llm_client(LLMConfig(backend="local"))

        self.llm_client = llm_client

    async def execute_step(self, step: Step, context: List[str]) -> str:
        """
        Execute a single step using LLM.

        Args:
            step: The step to execute
            context: Previous step results for context

        Returns:
            Execution result string
        """
        # Build prompt based on step type
        if step.step_type == StepType.RESEARCH:
            prompt = self._build_research_prompt(step, context)
        else:
            prompt = self._build_processing_prompt(step, context)

        logger.info(f"Executing: {step.title}")

        try:
            # Use LLM client abstraction (async)
            result = await self.llm_client.generate(prompt=prompt)

            logger.debug(f"Result preview: {result[:150]}...")
            return result

        except Exception as e:
            error_msg = f"ERROR: LLM call failed - {str(e)}"
            logger.error(f"Step '{step.title}' failed: {e}")
            return error_msg

    def _build_research_prompt(self, step: Step, context: List[str]) -> str:
        """Build prompt for research-type steps."""
        prompt = f"""You are a research assistant. Your task is to provide comprehensive information for the following research requirement.

**Task**: {step.title}

**Requirements**: {step.description}

**Instructions**:
- Provide detailed, factual information based on your knowledge
- Include specific data points, numbers, and facts where relevant
- Organize information clearly
- Be comprehensive but concise

"""

        if context:
            prompt += f"\n**Previous Research Context**:\n"
            for i, ctx in enumerate(context[-3:], 1):  # Last 3 for context
                prompt += f"\n{i}. {ctx[:200]}...\n"

        prompt += "\nProvide your research findings:"

        return prompt

    def _build_processing_prompt(self, step: Step, context: List[str]) -> str:
        """Build prompt for processing-type steps."""
        prompt = f"""You are a data analyst. Your task is to analyze and process the provided data.

**Task**: {step.title}

**Requirements**: {step.description}

**Data to Process**:
"""

        if context:
            for i, ctx in enumerate(context, 1):
                prompt += f"\n--- Dataset {i} ---\n{ctx}\n"
        else:
            prompt += "\n(No previous data available)\n"

        prompt += f"""
**Instructions**:
- Carefully analyze all provided data
- Follow the requirements exactly
- Show your work/calculations if relevant
- Provide clear, structured output
- Be thorough and accurate

Provide your analysis:"""

        return prompt


# Backward compatibility aliases
GeminiStepExecutor = StepExecutor  # For existing code that imports GeminiStepExecutor


def load_plan_from_json(json_path: Path) -> Plan:
    """
    Load a plan from a JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Plan object
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    steps = []
    for i, step_data in enumerate(data["steps"]):
        step = Step(
            title=step_data["title"],
            description=step_data["description"],
            step_type=StepType(step_data["step_type"]),
            need_search=step_data["need_search"],
            step_id=f"step-{i+1}"
        )
        steps.append(step)

    plan = Plan(
        title=data["title"],
        locale=data["locale"],
        thought=data["thought"],
        steps=steps,
        has_enough_context=data.get("has_enough_context", False)
    )

    return plan
