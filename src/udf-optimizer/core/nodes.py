"""
Graph nodes for parallel execution in DeerFlow-style workflow.

This module implements the core execution nodes including:
- parallel_research_team_node: Orchestrates parallel execution of independent steps
- _execute_single_step: Executes a single step independently
"""

import asyncio
import logging
import time
from typing import Literal, Tuple, Optional, List
from functools import partial

from .workflow_types import State, Step, Configuration, ExecutionResult, StepType
from .gemini_executor import DependencyAnalyzer, GeminiStepExecutor, BatchDefinition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Semaphore for rate limiting
PARALLEL_LIMIT: Optional[asyncio.Semaphore] = None


def initialize_rate_limiter(max_concurrent: int):
    """Initialize the global rate limiter semaphore."""
    global PARALLEL_LIMIT
    PARALLEL_LIMIT = asyncio.Semaphore(max_concurrent)


async def _execute_single_step(
    state: State,
    config: Configuration,
    step: Step,
    step_idx: int,
    agent_type: Literal["researcher", "coder"]
) -> Tuple[int, str, str]:
    """
    Execute a single step independently (can be called in parallel).
    
    This function executes one step and returns its result. It's designed to be
    called concurrently with other steps via asyncio.gather().
    
    Args:
        state: Current workflow state
        config: Runtime configuration
        step: The step object to execute
        step_idx: Index for result tracking
        agent_type: "researcher" or "coder"
    
    Returns:
        Tuple of (step_index, result_content, agent_name)
    
    Raises:
        TimeoutError: If step execution exceeds timeout
        Exception: For other execution errors
    """
    async with PARALLEL_LIMIT:  # Rate limiting
        start_time = time.time()
        
        try:
            logger.info(f"[Step {step_idx}] Starting execution: {step.title} (type: {agent_type})")
            
            # Get context from previous steps
            context = [obs for obs in state.observations if obs.strip()]
            
            # Execute using real Gemini API
            result_content = await _mock_agent_execution(step, agent_type, config, context)
            
            duration = time.time() - start_time
            logger.info(f"[Step {step_idx}] Completed in {duration:.2f}s")
            
            return (step_idx, result_content, agent_type)
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"ERROR: Timeout after {duration:.2f}s for step '{step.title}'"
            logger.error(f"[Step {step_idx}] {error_msg}")
            return (step_idx, error_msg, agent_type)
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"ERROR: {str(e)} in step '{step.title}'"
            logger.error(f"[Step {step_idx}] Failed after {duration:.2f}s: {str(e)}")
            return (step_idx, error_msg, agent_type)


async def _mock_agent_execution(
    step: Step,
    agent_type: str,
    config: Configuration,
    context: List[str]
) -> str:
    """
    Execute a step using real Gemini API.
    
    This replaces the previous mock implementation with actual LLM calls.
    
    Args:
        step: Step to execute
        agent_type: Type of agent to use (currently ignored, uses Gemini)
        config: Configuration settings
        context: Previous step results for context
    
    Returns:
        Execution result from Gemini
    """
    executor = GeminiStepExecutor()
    result = await executor.execute_step(step, context)
    return result


async def _execute_with_retry(
    state: State,
    config: Configuration,
    step: Step,
    step_idx: int,
    agent_type: str
) -> Tuple[int, str, str]:
    """
    Execute a step with retry logic.
    
    Implements exponential backoff retry strategy for transient failures.
    
    Args:
        state: Current workflow state
        config: Runtime configuration
        step: Step to execute
        step_idx: Step index
        agent_type: Type of agent
    
    Returns:
        Execution result tuple
    """
    last_error = None
    
    for attempt in range(config.max_retries + 1):
        try:
            result = await asyncio.wait_for(
                _execute_single_step(state, config, step, step_idx, agent_type),
                timeout=config.task_timeout_seconds
            )
            
            # If result indicates error but we should retry
            if config.retry_on_failure and "ERROR:" in result[1] and attempt < config.max_retries:
                backoff = config.retry_backoff_seconds[min(attempt, len(config.retry_backoff_seconds) - 1)]
                logger.warning(f"[Step {step_idx}] Retry {attempt + 1}/{config.max_retries} after {backoff}s")
                await asyncio.sleep(backoff)
                continue
            
            return result
            
        except asyncio.TimeoutError as e:
            last_error = e
            if attempt < config.max_retries and config.retry_on_failure:
                backoff = config.retry_backoff_seconds[min(attempt, len(config.retry_backoff_seconds) - 1)]
                logger.warning(f"[Step {step_idx}] Timeout, retry {attempt + 1}/{config.max_retries} after {backoff}s")
                await asyncio.sleep(backoff)
            else:
                return (step_idx, f"ERROR: Max retries exceeded (timeout)", agent_type)
        
        except Exception as e:
            last_error = e
            if attempt < config.max_retries and config.retry_on_failure:
                backoff = config.retry_backoff_seconds[min(attempt, len(config.retry_backoff_seconds) - 1)]
                logger.warning(f"[Step {step_idx}] Error: {e}, retry {attempt + 1}/{config.max_retries} after {backoff}s")
                await asyncio.sleep(backoff)
            else:
                return (step_idx, f"ERROR: Max retries exceeded - {str(e)}", agent_type)
    
    return (step_idx, f"ERROR: {str(last_error)}", agent_type)


async def parallel_research_team_node(
    state: State,
    config: Configuration
) -> dict:
    """
    Execute research steps in parallel batches based on LLM dependency analysis.
    
    This is the core parallelization node that:
    1. Uses LLM to analyze dependencies and create optimal batches
    2. Executes each batch (parallel or sequential as determined by LLM)
    3. Aggregates results after each batch
    4. Returns command to proceed to planner
    
    Args:
        state: Current workflow state
        config: Parallelization configuration
    
    Returns:
        Dict with next node ("planner"), updated state, and batch_metrics
    """
    logger.info("=== Starting Parallel Research Team Node ===")
    
    if not state.current_plan:
        logger.error("No plan found in state")
        return {"next": "planner", "state": state, "batch_metrics": []}
    
    plan = state.current_plan
    incomplete_steps = plan.get_incomplete_steps()
    
    if not incomplete_steps:
        logger.info("All steps complete, proceeding to planner")
        return {"next": "planner", "state": state, "batch_metrics": []}
    
    logger.info(f"Found {len(incomplete_steps)} incomplete steps")
    
    # Initialize rate limiter if not already done
    if PARALLEL_LIMIT is None:
        initialize_rate_limiter(config.max_concurrent_tasks)
    
    # Use LLM to analyze dependencies and create batches
    analyzer = DependencyAnalyzer()
    batches = analyzer.analyze_plan(plan)
    
    logger.info(f"LLM created {len(batches)} batches for execution")
    
    # Track batch metrics
    batch_metrics = []
    
    # Execute batches in order
    for batch_idx, batch in enumerate(batches):
        logger.info(f"\n{'='*60}")
        logger.info(f"Executing {batch}")
        logger.info(f"{'='*60}")
        
        # Get steps for this batch
        batch_steps = [incomplete_steps[idx] for idx in batch.step_indices if idx < len(incomplete_steps)]
        
        # Time the batch execution
        batch_start = time.time()
        
        if batch.parallel:
            # Execute all steps in batch concurrently
            await _execute_batch_parallel(state, config, batch_steps, batch.step_indices)
        else:
            # Execute steps sequentially
            await _execute_batch_sequential(state, config, batch_steps, batch.step_indices)
        
        batch_end = time.time()
        batch_duration = batch_end - batch_start
        
        # Record batch metrics
        batch_metrics.append({
            "batch_id": batch_idx,
            "parallel": batch.parallel,
            "step_count": len(batch_steps),
            "start_time": batch_start,
            "end_time": batch_end,
            "duration": batch_duration,
            "description": f"Batch {batch_idx} ({'parallel' if batch.parallel else 'sequential'}): {len(batch_steps)} steps"
        })
        
        logger.info(f"Batch {batch_idx} completed in {batch_duration:.2f}s")
    
    logger.info("=== All batches completed ===")
    return {"next": "planner", "state": state, "batch_metrics": batch_metrics}


async def _execute_batch_parallel(
    state: State,
    config: Configuration,
    steps: List[Step],
    step_indices: List[int]
) -> None:
    """Execute a batch of steps in parallel."""
    tasks = []
    
    for idx, step in zip(step_indices, steps):
        agent_type = "researcher" if step.step_type == StepType.RESEARCH else "coder"
        task = _execute_with_retry(state, config, step, idx, agent_type)
        tasks.append(task)
    
    # Execute all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch execution error: {result}")
            state.observations.append(f"ERROR: {str(result)}")
        else:
            step_idx, content, agent_type = result
            # Assign result to the correct step in the batch
            if i < len(steps):
                steps[i].execution_res = content
            state.observations.append(content)
            logger.debug(f"Added result for step {step_idx}")


async def _execute_batch_sequential(
    state: State,
    config: Configuration,
    steps: List[Step],
    step_indices: List[int]
) -> None:
    """Execute a batch of steps sequentially."""
    for idx, step in zip(step_indices, steps):
        agent_type = "researcher" if step.step_type == StepType.RESEARCH else "coder"
        
        try:
            result = await _execute_with_retry(state, config, step, idx, agent_type)
            step_idx, content, agent_type = result
            
            # Assign result directly to the current step
            step.execution_res = content
            state.observations.append(content)
            logger.debug(f"Added result for step {step_idx}")
            
        except Exception as e:
            logger.error(f"Sequential execution error for step {idx}: {e}")
            state.observations.append(f"ERROR: {str(e)}")


def research_team_node_sequential(state: State, config: Configuration) -> dict:
    """
    Fallback sequential execution for when parallelization is disabled.
    
    This provides backward compatibility by executing steps one at a time
    in the original sequential manner.
    
    Args:
        state: Current workflow state
        config: Configuration (parallelization disabled)
    
    Returns:
        Dict with next node and updated state
    """
    logger.info("Using sequential execution (parallelization disabled)")
    
    if not state.current_plan:
        return {"next": "planner", "state": state}
    
    plan = state.current_plan
    incomplete_steps = plan.get_incomplete_steps()
    
    if not incomplete_steps:
        return {"next": "planner", "state": state}
    
    # Execute first incomplete step only
    step = incomplete_steps[0]
    agent_type = "researcher" if step.step_type == StepType.RESEARCH else "coder"
    
    # Synchronous execution (simulated)
    result = f"Sequential result for '{step.title}'"
    step.execution_res = result
    state.add_observation(result)
    
    # Return to research_team for next iteration
    return {"next": "research_team", "state": state}
