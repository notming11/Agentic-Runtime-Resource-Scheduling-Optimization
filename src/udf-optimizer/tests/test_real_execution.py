"""
Main script for real Gemini API integration with parallel execution.

This script demonstrates the full system with:
- Real plan loaded from example_response_1.txt
- LLM-based dependency analysis using parallel_prompt.md
- Gemini API for step execution
- Parallel batch execution based on LLM analysis
"""

import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime

from core import State, Configuration, parallel_research_team_node, load_plan_from_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_report_to_markdown(plan, state, execution_time: float, output_path: Path):
    """Save execution results to a markdown report file."""
    
    # Calculate metrics
    completed_steps = len([s for s in plan.steps if s.execution_res])
    total_steps = len(plan.steps)
    sequential_estimate = total_steps * 10
    speedup = sequential_estimate / execution_time if execution_time > 0 else 0
    
    # Generate markdown content
    report = f"""# Execution Report: {plan.title}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

- **Total Steps**: {total_steps}
- **Completed Steps**: {completed_steps}
- **Success Rate**: {(completed_steps/total_steps)*100:.1f}%
- **Execution Time**: {execution_time:.2f}s
- **Estimated Sequential Time**: {sequential_estimate:.2f}s
- **Speedup**: {speedup:.2f}x

## Plan Overview

**Locale**: {plan.locale}

**Thought Process**:
{plan.thought}

---

## Step-by-Step Results

"""
    
    # Add each step's results
    for i, step in enumerate(plan.steps):
        status = "✅ **Completed**" if step.execution_res and "ERROR" not in step.execution_res else "❌ **Failed**" if step.execution_res else "⏸️ **Not Executed**"
        
        report += f"""### Step {i}: {step.title}

**Status**: {status}  
**Type**: {step.step_type.value}  
**Description**: {step.description}

"""
        
        if step.execution_res:
            # Clean up the result for markdown
            result_text = step.execution_res.strip()
            report += f"""**Result**:
```
{result_text}
```

"""
        else:
            report += "**Result**: Not executed\n\n"
        
        report += "---\n\n"
    
    # Add performance section
    report += f"""## Performance Metrics

### Execution Breakdown
- **Parallel execution time**: {execution_time:.2f} seconds
- **Estimated sequential time**: {sequential_estimate:.2f} seconds (assuming 10s per step)
- **Time saved**: {sequential_estimate - execution_time:.2f} seconds
- **Speedup factor**: {speedup:.2f}x

### Observations
Total observations captured: {len(state.observations)}

"""
    
    # Add raw observations
    if state.observations:
        report += "### Raw Observations\n\n"
        for i, obs in enumerate(state.observations, 1):
            report += f"{i}. {obs[:200]}{'...' if len(obs) > 200 else ''}\n\n"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return output_path


async def execute_real_workflow():
    """Execute the full workflow with real LLM integration."""
    
    logger.info("="*80)
    logger.info("PARALLEL EXECUTION SYSTEM - REAL LLM INTEGRATION")
    logger.info("="*80)
    
    # Load the real plan
    script_dir = Path(__file__).parent
    plan_path = script_dir / "examples" / "example_response_1.txt"
    
    logger.info(f"\n1. Loading plan from: {plan_path}")
    plan = load_plan_from_json(plan_path)
    
    logger.info(f"   Plan: {plan.title}")
    logger.info(f"   Total steps: {len(plan.steps)}")
    logger.info(f"   Research steps: {sum(1 for s in plan.steps if s.need_search)}")
    logger.info(f"   Processing steps: {sum(1 for s in plan.steps if not s.need_search)}")
    
    # Create initial state
    state = State(
        messages=[],
        observations=[],
        current_plan=plan
    )
    
    # Create configuration (optimize for speed)
    config = Configuration(
        enabled=True,  # Enable parallelization
        max_concurrent_tasks=5,  # Limit concurrent API calls
        task_timeout_seconds=120,  # 2 minutes per task
        max_retries=2,
        retry_backoff_seconds=[3, 10, 30],
        retry_on_failure=True,
        failure_mode="partial_completion",
        dependency_strategy="llm_based"
    )
    
    logger.info(f"\n2. Configuration:")
    logger.info(f"   Max concurrent tasks: {config.max_concurrent_tasks}")
    logger.info(f"   Task timeout: {config.task_timeout_seconds}s")
    logger.info(f"   Max retries: {config.max_retries}")
    
    # Execute workflow
    logger.info(f"\n3. Starting execution with LLM dependency analysis...\n")
    
    start_time = time.time()
    
    try:
        result = await parallel_research_team_node(state, config)
        
        execution_time = time.time() - start_time
        
        logger.info(f"\n{'='*80}")
        logger.info("EXECUTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total execution time: {execution_time:.2f}s")
        logger.info(f"Steps completed: {len([s for s in plan.steps if s.execution_res])}/{len(plan.steps)}")
        logger.info(f"Total observations: {len(state.observations)}")
        
        # Show results summary
        logger.info(f"\n4. Results Summary:")
        for i, step in enumerate(plan.steps):
            if step.execution_res:
                result_preview = step.execution_res[:100].replace('\n', ' ')
                status = "✓" if "ERROR" not in step.execution_res else "✗"
                logger.info(f"   {status} Step {i}: {step.title}")
                logger.info(f"      → {result_preview}...")
        
        # Calculate theoretical sequential time (assuming 10s per step)
        sequential_time = len(plan.steps) * 10
        speedup = sequential_time / execution_time
        
        logger.info(f"\n5. Performance Analysis:")
        logger.info(f"   Parallel execution: {execution_time:.2f}s")
        logger.info(f"   Sequential estimate: {sequential_time:.2f}s")
        logger.info(f"   Speedup: {speedup:.2f}x")
        
        # Save results to markdown report
        report_path = script_dir / "examples" / "example_report.md"
        saved_path = save_report_to_markdown(plan, state, execution_time, report_path)
        
        logger.info(f"\n6. Report Saved:")
        logger.info(f"   📄 {saved_path}")
        logger.info(f"   Open this file to view the complete execution report")
        
        return state, execution_time
        
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise


def main():
    """Entry point for the application."""
    try:
        asyncio.run(execute_real_workflow())
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
