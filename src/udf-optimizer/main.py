"""
Main entry point for the UDF Optimizer with interactive user input.

This script allows users to:
1. Load a plan from a file or create one interactively
2. Choose execution mode (parallel vs sequential)
3. Configure performance settings
4. Execute and save results

Run with: python main.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

from core import (
    State, 
    Configuration, 
    parallel_research_team_node, 
    sequential_execution_node,
    load_plan_from_json
)
from compare_performance import (
    run_parallel_execution,
    run_sequential_execution,
    generate_llm_analysis,
    save_performance_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 80)
    print("UDF OPTIMIZER - PARALLEL EXECUTION SYSTEM")
    print("=" * 80)
    print("\nThis system optimizes workflow execution using parallel processing")
    print("and LLM-based dependency analysis.\n")


def list_available_plans() -> list[Path]:
    """List all available plan files in the examples directory."""
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        return []
    
    # Find all .txt files that look like plans
    plan_files = list(examples_dir.glob("example_response*.txt"))
    return sorted(plan_files)


def get_plan_choice() -> Path:
    """Prompt user to select a plan file."""
    print("\n--- Select Plan ---")
    
    available_plans = list_available_plans()
    
    if not available_plans:
        print("No plan files found in examples/ directory.")
        print("Please create a plan file (e.g., examples/example_response_1.txt)")
        sys.exit(1)
    
    print("\nAvailable plans:")
    for i, plan_path in enumerate(available_plans, 1):
        print(f"  {i}. {plan_path.name}")
    
    print(f"  {len(available_plans) + 1}. Enter custom path")
    
    while True:
        try:
            choice = input(f"\nSelect plan (1-{len(available_plans) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(available_plans):
                return available_plans[choice_num - 1]
            elif choice_num == len(available_plans) + 1:
                custom_path = input("Enter path to plan file: ").strip()
                custom_path = Path(custom_path)
                if custom_path.exists():
                    return custom_path
                else:
                    print(f"File not found: {custom_path}")
            else:
                print(f"Please enter a number between 1 and {len(available_plans) + 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def get_execution_mode() -> str:
    """Prompt user to select execution mode."""
    print("\n--- Select Execution Mode ---")
    print("  1. Parallel execution (optimized with LLM dependency analysis)")
    print("  2. Sequential execution (one step at a time)")
    print("  3. Compare both (run parallel and sequential, generate comparison report)")
    
    while True:
        try:
            choice = input("\nSelect mode (1-3): ").strip()
            if choice in ["1", "2", "3"]:
                return {"1": "parallel", "2": "sequential", "3": "compare"}[choice]
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def get_configuration() -> Configuration:
    """Prompt user for configuration settings or use defaults."""
    print("\n--- Configuration ---")
    print("Use default optimized settings? (y/n)")
    
    use_defaults = input("Choice [y]: ").strip().lower() or "y"
    
    if use_defaults == "y":
        config = Configuration(
            enabled=True,
            max_concurrent_tasks=5,
            task_timeout_seconds=120,
            max_retries=2,
            retry_backoff_seconds=[3, 10, 30],
            retry_on_failure=True,
            failure_mode="partial_completion",
            dependency_strategy="llm_based"
        )
        print("\nUsing default configuration:")
    else:
        print("\n--- Custom Configuration ---")
        
        try:
            max_concurrent = int(input("Max concurrent tasks [5]: ").strip() or "5")
            timeout = int(input("Task timeout in seconds [120]: ").strip() or "120")
            max_retries = int(input("Max retries [2]: ").strip() or "2")
            
            config = Configuration(
                enabled=True,
                max_concurrent_tasks=max_concurrent,
                task_timeout_seconds=timeout,
                max_retries=max_retries,
                retry_backoff_seconds=[3, 10, 30],
                retry_on_failure=True,
                failure_mode="partial_completion",
                dependency_strategy="llm_based"
            )
            print("\nUsing custom configuration:")
        except ValueError:
            print("Invalid input, using defaults")
            config = Configuration(
                enabled=True,
                max_concurrent_tasks=5,
                task_timeout_seconds=120,
                max_retries=2,
                retry_backoff_seconds=[3, 10, 30],
                retry_on_failure=True,
                failure_mode="partial_completion",
                dependency_strategy="llm_based"
            )
    
    print(f"  - Max concurrent tasks: {config.max_concurrent_tasks}")
    print(f"  - Task timeout: {config.task_timeout_seconds}s")
    print(f"  - Max retries: {config.max_retries}")
    print(f"  - Failure mode: {config.failure_mode}")
    
    return config


def get_output_path() -> Path:
    """Prompt user for output report path."""
    print("\n--- Output Configuration ---")
    
    default_name = f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    default_path = Path(__file__).parent / "examples" / default_name
    
    print(f"Default output: {default_path}")
    custom = input("Use custom output path? (y/n) [n]: ").strip().lower() or "n"
    
    if custom == "y":
        custom_path = input("Enter output path: ").strip()
        return Path(custom_path)
    else:
        return default_path


async def execute_parallel(plan_path: Path, config: Configuration, output_path: Path):
    """Execute plan in parallel mode."""
    logger.info(f"\nLoading plan from: {plan_path}")
    plan = load_plan_from_json(plan_path)
    
    logger.info(f"Plan: {plan.title}")
    logger.info(f"Total steps: {len(plan.steps)}")
    
    state = State(
        messages=[],
        observations=[],
        current_plan=plan
    )
    
    logger.info("\nExecuting in PARALLEL mode...\n")
    
    import time
    start_time = time.time()
    
    result = await parallel_research_team_node(state, config)
    
    execution_time = time.time() - start_time
    
    # Save report
    from tests.test_real_execution import save_report_to_markdown
    save_report_to_markdown(plan, state, execution_time, output_path)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EXECUTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Execution time: {execution_time:.2f}s")
    logger.info(f"Steps completed: {len([s for s in plan.steps if s.execution_res])}/{len(plan.steps)}")
    logger.info(f"Report saved: {output_path}\n")


async def execute_sequential(plan_path: Path, config: Configuration, output_path: Path):
    """Execute plan in sequential mode."""
    logger.info(f"\nLoading plan from: {plan_path}")
    plan = load_plan_from_json(plan_path)
    
    logger.info(f"Plan: {plan.title}")
    logger.info(f"Total steps: {len(plan.steps)}")
    
    state = State(
        messages=[],
        observations=[],
        current_plan=plan
    )
    
    logger.info("\nExecuting in SEQUENTIAL mode...\n")
    
    import time
    start_time = time.time()
    
    result = await sequential_execution_node(state, config)
    
    execution_time = time.time() - start_time
    
    # Save report
    from tests.test_real_execution import save_report_to_markdown
    save_report_to_markdown(plan, state, execution_time, output_path)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EXECUTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Execution time: {execution_time:.2f}s")
    logger.info(f"Steps completed: {len([s for s in plan.steps if s.execution_res])}/{len(plan.steps)}")
    logger.info(f"Report saved: {output_path}\n")


async def execute_comparison(plan_path: Path, config: Configuration, output_path: Path):
    """Execute both modes and generate comparison report."""
    logger.info(f"\nLoading plan from: {plan_path}")
    plan = load_plan_from_json(plan_path)
    
    logger.info(f"Plan: {plan.title}")
    logger.info(f"Total steps: {len(plan.steps)}")
    
    logger.info("\nRunning comparison between parallel and sequential execution...")
    logger.info("This will take approximately twice as long as a single execution.\n")
    
    # Run both executions
    logger.info("=== Running Parallel Execution ===")
    parallel_metrics = await run_parallel_execution(plan, config)
    
    logger.info("\n=== Running Sequential Execution ===")
    sequential_metrics = await run_sequential_execution(plan, config)
    
    # Generate analysis
    logger.info("\n=== Generating LLM Analysis ===")
    analysis = await generate_llm_analysis(parallel_metrics, sequential_metrics)
    
    # Save report
    save_performance_report(parallel_metrics, sequential_metrics, analysis, output_path)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPARISON COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Parallel time: {parallel_metrics.total_duration:.2f}s")
    logger.info(f"Sequential time: {sequential_metrics.total_duration:.2f}s")
    logger.info(f"Speedup: {sequential_metrics.total_duration / parallel_metrics.total_duration:.2f}x")
    logger.info(f"Report saved: {output_path}\n")


async def main():
    """Main entry point with user interaction."""
    try:
        print_banner()
        
        # Get user choices
        plan_path = get_plan_choice()
        execution_mode = get_execution_mode()
        config = get_configuration()
        output_path = get_output_path()
        
        # Execute based on mode
        if execution_mode == "parallel":
            await execute_parallel(plan_path, config, output_path)
        elif execution_mode == "sequential":
            await execute_sequential(plan_path, config, output_path)
        elif execution_mode == "compare":
            await execute_comparison(plan_path, config, output_path)
        
        print("\n✓ Execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
