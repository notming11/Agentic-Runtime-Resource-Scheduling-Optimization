"""
Sequential execution node - executes all steps one at a time without parallelization.

This module provides a true sequential execution path for comparison with
the parallel execution system.
"""

import asyncio
import logging
import time
from typing import List, Tuple

from .workflow_types import State, Step, Configuration, StepType
from .gemini_executor import GeminiStepExecutor

logger = logging.getLogger(__name__)


async def sequential_execution_node(
    state: State,
    config: Configuration
) -> dict:
    """
    Execute all steps sequentially (one at a time) without parallelization.
    
    This provides a baseline for performance comparison with parallel execution.
    Each step is executed completely before moving to the next step.
    
    Args:
        state: Current workflow state
        config: Configuration settings
    
    Returns:
        Dict with next node, updated state, and step_metrics
    """
    logger.info("=== Starting Sequential Execution Node ===")
    
    if not state.current_plan:
        logger.error("No plan found in state")
        return {"next": "planner", "state": state, "step_metrics": []}
    
    plan = state.current_plan
    incomplete_steps = plan.get_incomplete_steps()
    
    if not incomplete_steps:
        logger.info("All steps complete")
        return {"next": "planner", "state": state, "step_metrics": []}
    
    logger.info(f"Found {len(incomplete_steps)} incomplete steps")
    logger.info("Executing steps sequentially (NO parallelization)")
    
    executor = GeminiStepExecutor()
    
    # Track step metrics
    step_metrics = []
    
    # Execute each step one at a time
    for i, step in enumerate(incomplete_steps):
        step_start_time = time.time()
        
        logger.info(f"\n[Step {i}] Starting: {step.title}")
        
        try:
            # Get context from previous steps
            context = [obs for obs in state.observations if obs.strip()]
            
            # Execute step
            result = await executor.execute_step(step, context)
            
            # Store result
            step.execution_res = result
            state.observations.append(result)
            
            step_duration = time.time() - step_start_time
            logger.info(f"[Step {i}] Completed in {step_duration:.2f}s")
            
            # Record step metrics
            step_metrics.append({
                "step_id": i,
                "step_title": step.title,
                "start_time": step_start_time,
                "end_time": time.time(),
                "duration": step_duration,
                "success": "ERROR" not in result,
                "error_message": "" if "ERROR" not in result else result
            })
            
            # Log success/failure
            if "ERROR" in result:
                logger.warning(f"[Step {i}] Completed with error")
            else:
                logger.info(f"[Step {i}] Successfully completed")
                
        except Exception as e:
            step_duration = time.time() - step_start_time
            error_msg = f"ERROR: {str(e)}"
            logger.error(f"[Step {i}] Failed after {step_duration:.2f}s: {e}")
            
            step.execution_res = error_msg
            state.observations.append(error_msg)
            
            # Record failed step metrics
            step_metrics.append({
                "step_id": i,
                "step_title": step.title,
                "start_time": step_start_time,
                "end_time": time.time(),
                "duration": step_duration,
                "success": False,
                "error_message": error_msg
            })
    
    logger.info("=== Sequential execution completed ===")
    return {"next": "planner", "state": state, "step_metrics": step_metrics}
