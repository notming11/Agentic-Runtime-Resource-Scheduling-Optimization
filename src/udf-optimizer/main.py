"""
Main demonstration script for DeerFlow parallel execution.

This script demonstrates the parallelization implementation described in the
"Parallelization Implementation Report for DeerFlow". It shows:

1. Creating a research plan with independent and dependent steps
2. Executing steps in parallel batches
3. Comparing sequential vs parallel execution times
4. Demonstrating the speedup achieved

Run with: python main.py
"""

import asyncio
import logging
import time
from pathlib import Path

from core import State, Plan, Step, StepType, load_configuration, get_example_config, build_workflow_graph, initialize_rate_limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_example_plan_simple() -> Plan:
    """
    Create a simple research plan with 5 independent research steps.
    
    This mimics the example from the report:
    "Research top attractions in 10 cities" -> 10 parallel steps + analysis + report
    
    Simplified to 5 cities for demonstration.
    """
    steps = []
    
    cities = ["Tokyo", "Paris", "London", "New York", "Dubai"]
    
    # Create research steps for each city (all independent)
    for i, city in enumerate(cities):
        step = Step(
            title=f"Research {city} attractions",
            description=f"Use web search to find top tourist sites in {city}",
            step_type=StepType.RESEARCH,
            need_search=True,
            step_id=f"step-{i+1}"
        )
        steps.append(step)
    
    # Add analysis step (depends on all research)
    analysis_step = Step(
        title="Analyze common patterns",
        description="Examine all city data for themes and patterns",
        step_type=StepType.PROCESSING,
        need_search=False,
        step_id="step-6",
        dependencies=[f"step-{i+1}" for i in range(len(cities))]
    )
    steps.append(analysis_step)
    
    # Add report step (depends on analysis)
    report_step = Step(
        title="Generate report",
        description="Based on analysis, create comprehensive summary",
        step_type=StepType.PROCESSING,
        need_search=False,
        step_id="step-7",
        dependencies=["step-6"]
    )
    steps.append(report_step)
    
    plan = Plan(
        title="Multi-City Research Plan",
        locale="en-US",
        thought="Research multiple cities in parallel, then analyze and report",
        steps=steps,
        has_enough_context=False
    )
    
    return plan


def create_example_plan_complex() -> Plan:
    """
    Create a complex plan with 10 cities to match the report example.
    
    Expected speedup: ~10x for research phase
    """
    steps = []
    
    cities = [
        "Tokyo", "Paris", "London", "New York", "Dubai",
        "Singapore", "Rome", "Barcelona", "Sydney", "Bangkok"
    ]
    
    # 10 independent research steps
    for i, city in enumerate(cities):
        step = Step(
            title=f"Research {city} attractions",
            description=f"Use web search to find top tourist sites, hotels, and restaurants in {city}",
            step_type=StepType.RESEARCH,
            need_search=True,
            step_id=f"step-{i+1}"
        )
        steps.append(step)
    
    # Analysis step (depends on all 10 research steps)
    analysis_step = Step(
        title="Analyze common patterns across all cities",
        description="Examine all city data for themes, patterns, and recommendations",
        step_type=StepType.PROCESSING,
        need_search=False,
        step_id="step-11",
        dependencies=[f"step-{i+1}" for i in range(len(cities))]
    )
    steps.append(analysis_step)
    
    # Report step (depends on analysis)
    report_step = Step(
        title="Generate comprehensive report",
        description="Based on analysis, create detailed summary with recommendations",
        step_type=StepType.PROCESSING,
        need_search=False,
        step_id="step-12",
        dependencies=["step-11"]
    )
    steps.append(report_step)
    
    plan = Plan(
        title="10-City Research Plan",
        locale="en-US",
        thought="Research 10 cities in parallel to maximize efficiency, then analyze patterns and generate report",
        steps=steps,
        has_enough_context=False
    )
    
    return plan


async def execute_workflow(state: State, config) -> State:
    """
    Execute the complete workflow using the graph builder.
    
    This is a simplified execution that directly calls the parallel node.
    In a real implementation, this would traverse the full graph.
    """
    logger.info("=" * 80)
    logger.info("STARTING WORKFLOW EXECUTION")
    logger.info("=" * 80)
    
    # Initialize rate limiter
    initialize_rate_limiter(config.max_concurrent_tasks)
    
    # Build graph
    graph = build_workflow_graph(config)
    
    # Execute research_team node (the parallel execution node)
    research_team_func = graph.nodes["research_team"]
    
    # Execute (handle both sync and async)
    if asyncio.iscoroutinefunction(research_team_func):
        result = await research_team_func(state)
    else:
        result = research_team_func(state)
    
    updated_state = result.get("state", state)
    
    logger.info("=" * 80)
    logger.info("WORKFLOW EXECUTION COMPLETE")
    logger.info("=" * 80)
    
    return updated_state


def print_results(state: State, execution_time: float):
    """Print execution results in a readable format."""
    print("\n" + "=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)
    
    if state.current_plan:
        plan = state.current_plan
        print(f"\nPlan: {plan.title}")
        print(f"Total Steps: {len(plan.steps)}")
        
        completed = sum(1 for step in plan.steps if step.is_complete())
        print(f"Completed: {completed}/{len(plan.steps)}")
        
        print("\n--- Step Results ---")
        for i, step in enumerate(plan.steps, 1):
            status = "✓" if step.is_complete() else "✗"
            print(f"\n{status} Step {i}: {step.title}")
            print(f"   Type: {step.step_type.value}")
            if step.execution_res:
                # Truncate long results
                result_preview = step.execution_res[:100]
                if len(step.execution_res) > 100:
                    result_preview += "..."
                print(f"   Result: {result_preview}")
    
    print(f"\n--- Performance ---")
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    
    print("\n" + "=" * 80 + "\n")


async def demo_parallel_execution():
    """Demonstrate parallel execution with the example from the report."""
    print("\n" + "=" * 80)
    print("DEERFLOW PARALLELIZATION DEMO")
    print("=" * 80)
    print("\nThis demo implements the architecture from the Parallelization Report.")
    print("It shows parallel execution of independent research steps.\n")
    
    # Load configuration
    config = get_example_config("speed_optimized")
    config.enabled = True  # Ensure parallelization is enabled
    
    print(f"Configuration:")
    print(f"  - Parallelization: {'ENABLED' if config.enabled else 'DISABLED'}")
    print(f"  - Max Concurrent Tasks: {config.max_concurrent_tasks}")
    print(f"  - Task Timeout: {config.task_timeout_seconds}s")
    print(f"  - Failure Mode: {config.failure_mode}")
    
    # Create example plan
    print("\n--- Creating Research Plan ---")
    plan = create_example_plan_complex()  # 10 cities
    
    print(f"Plan: {plan.title}")
    print(f"Total Steps: {len(plan.steps)}")
    print(f"  - Research steps: {sum(1 for s in plan.steps if s.step_type == StepType.RESEARCH)}")
    print(f"  - Processing steps: {sum(1 for s in plan.steps if s.step_type == StepType.PROCESSING)}")
    
    # Create initial state
    state = State(current_plan=plan)
    state.add_message("user", "Research top attractions in 10 cities")
    
    # Execute workflow
    print("\n--- Executing Workflow (Parallel Mode) ---\n")
    start_time = time.time()
    
    final_state = await execute_workflow(state, config)
    
    execution_time = time.time() - start_time
    
    # Print results
    print_results(final_state, execution_time)
    
    # Calculate theoretical speedup
    num_parallel_steps = sum(1 for s in plan.steps if s.step_type == StepType.RESEARCH)
    theoretical_sequential_time = num_parallel_steps * 3  # Assuming ~3s per step
    
    print(f"--- Speedup Analysis ---")
    print(f"Parallel steps executed: {num_parallel_steps}")
    print(f"Theoretical sequential time: ~{theoretical_sequential_time:.0f}s")
    print(f"Actual parallel time: {execution_time:.2f}s")
    if execution_time > 0:
        speedup = theoretical_sequential_time / execution_time
        print(f"Speedup: {speedup:.1f}x")
    
    print("\n✓ Demo completed successfully!")


async def demo_comparison():
    """Compare parallel vs sequential execution."""
    print("\n" + "=" * 80)
    print("PARALLEL VS SEQUENTIAL COMPARISON")
    print("=" * 80)
    
    # Create plan
    plan = create_example_plan_simple()  # 5 cities for faster demo
    
    print(f"\nPlan: {plan.title}")
    print(f"Steps: {len(plan.steps)}")
    
    # Test 1: Parallel execution
    print("\n--- Test 1: PARALLEL EXECUTION ---\n")
    config_parallel = get_example_config("speed_optimized")
    config_parallel.enabled = True
    
    state_parallel = State(current_plan=plan)
    
    start = time.time()
    await execute_workflow(state_parallel, config_parallel)
    parallel_time = time.time() - start
    
    print(f"\n✓ Parallel execution completed in {parallel_time:.2f}s")
    
    # Test 2: Sequential execution
    print("\n--- Test 2: SEQUENTIAL EXECUTION ---\n")
    
    # Reset plan
    plan_seq = create_example_plan_simple()
    config_sequential = get_example_config("sequential_fallback")
    config_sequential.enabled = False
    
    state_sequential = State(current_plan=plan_seq)
    
    start = time.time()
    # Note: Sequential would execute one step at a time in a loop
    # This is simplified for demo
    print("(Simulated sequential execution)")
    sequential_time = len([s for s in plan_seq.steps if s.step_type == StepType.RESEARCH]) * 3
    
    print(f"\n✓ Sequential execution would take ~{sequential_time:.2f}s")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"Parallel Time:   {parallel_time:.2f}s")
    print(f"Sequential Time: ~{sequential_time:.2f}s")
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print(f"Speedup:         {speedup:.1f}x")
    time_saved = sequential_time - parallel_time
    print(f"Time Saved:      {time_saved:.2f}s ({time_saved/sequential_time*100:.1f}%)")
    print("=" * 80 + "\n")


async def main():
    """Main entry point."""
    try:
        # Run main demo
        await demo_parallel_execution()
        
        # Optional: Run comparison
        print("\n" + "=" * 80)
        response = input("Run parallel vs sequential comparison? (y/n): ")
        if response.lower() == 'y':
            await demo_comparison()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
