"""
Real implementation with Gemini API for dependency analysis and step execution.

This module replaces the mock execution with real LLM calls using Google's Gemini API.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from .workflow_types import Plan, Step, StepType

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GEMINI_API_KEY not found. Real execution will fail.")


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
    """Analyzes plan dependencies using LLM to create execution batches."""
    
    def __init__(self):
        self.model = genai.GenerativeModel(
            # 'models/gemini-2.0-flash-exp',
            'models/gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
    
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
        
        # Create model with system instruction
        model_with_system = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        try:
            logger.info("Sending plan to LLM for dependency analysis...")
            response = model_with_system.generate_content(user_input)
            
            # Parse response
            response_text = response.text
            logger.debug(f"LLM Response: {response_text[:200]}...")
            
            # Clean markdown if present
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            
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


class GeminiStepExecutor:
    """Executes individual steps using Gemini API."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    
    async def execute_step(self, step: Step, context: List[str], step_idx: int = None) -> str:
        """
        Execute a single step using Gemini API.
        
        Args:
            step: The step to execute
            context: Previous step results for context
            step_idx: Current step index for reference
        
        Returns:
            Execution result string
        """
        # Build prompt based on step type
        if step.step_type == StepType.RESEARCH:
            prompt = self._build_research_prompt(step, context, step_idx)
        else:
            prompt = self._build_processing_prompt(step, context, step_idx)
        
        logger.info(f"Executing: {step.title}")
        
        try:
            # Run in thread pool to not block async event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            result = response.text
            logger.debug(f"Result preview: {result[:150]}...")
            return result
            
        except Exception as e:
            error_msg = f"ERROR: Gemini API call failed - {str(e)}"
            logger.error(f"Step '{step.title}' failed: {e}")
            return error_msg
    
    def _build_research_prompt(self, step: Step, context: List[str], step_idx: int = None) -> str:
        """Build prompt for research-type steps."""
        
        # Detect if this step references specific items (City 1, City 2, etc.)
        step_specific_instruction = ""
        if step_idx is not None and ("City 1" in step.description or "City 2" in step.description or "City 3" in step.description):
            # Extract which city number this is
            city_num = None
            if "City 1" in step.description or "city 1" in step.title.lower():
                city_num = 1
            elif "City 2" in step.description or "city 2" in step.title.lower():
                city_num = 2
            elif "City 3" in step.description or "city 3" in step.title.lower():
                city_num = 3
            
            if city_num and context:
                step_specific_instruction = f"""
**IMPORTANT**: This step asks you to research "City {city_num}". Look at Step 0's results below to identify which specific city "City {city_num}" refers to. 
- If Step 0 identifies a ranked list of cities, "City {city_num}" means the city ranked #{city_num} in that list.
- Extract the actual city name from Step 0's results and use it for your research.
- Replace "City {city_num}" with the actual city name in your response.
"""
        
        prompt = f"""You are a research assistant. Your task is to provide comprehensive information for the following research requirement.

**Task**: {step.title}

**Requirements**: {step.description}
{step_specific_instruction}
**Instructions**:
- Provide detailed, factual information based on your knowledge
- Include specific data points, numbers, and facts where relevant
- Organize information clearly
- Be comprehensive but concise

"""
        
        if context:
            prompt += f"\n**Previous Research Results (Use these as your data source)**:\n"
            # Include full context from previous steps, not truncated
            for i, ctx in enumerate(context, 1):
                prompt += f"\n--- Result from Step {i-1} ---\n{ctx}\n"
        
        prompt += "\nProvide your research findings:"
        
        return prompt
    
    def _build_processing_prompt(self, step: Step, context: List[str], step_idx: int = None) -> str:
        """Build prompt for processing-type steps."""
        prompt = f"""You are a data analyst. Your task is to analyze and process the provided data.

**Task**: {step.title}

**Requirements**: {step.description}

**Available Data from Previous Steps**:
"""
        
        if context:
            for i, ctx in enumerate(context, 1):
                prompt += f"\n--- Result from Step {i-1} ---\n{ctx}\n"
        else:
            prompt += "\n(No previous data available)\n"
        
        prompt += f"""
**CRITICAL INSTRUCTIONS for Data Mapping**:
- The datasets above are numbered according to the step that produced them (Step 0, Step 1, Step 2, etc.)
- If your task mentions "City 1", "City 2", or "City 3", you MUST:
  1. First look at Step 0's result to identify which actual cities these refer to (e.g., the 1st, 2nd, and 3rd cities in the ranked list)
  2. Then look at Steps 1, 2, and 3 to find the research about those specific cities
  3. Match the data correctly: Step 1 data → City ranked #1 in Step 0, Step 2 data → City ranked #2, etc.
- If your task references specific steps or cities, cite the actual city names from Step 0, not generic placeholders
- Be extremely careful to use data from the correct steps
- Follow the requirements exactly
- Show your work/calculations if relevant
- Provide clear, structured output with actual city names, not "City 1", "City 2", etc.
- Be thorough and accurate

Provide your analysis:"""
        
        return prompt


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
