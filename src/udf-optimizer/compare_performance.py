"""
Performance Comparison Test - Parallel vs Sequential Execution

This script runs the same plan through both parallel and sequential execution
modes, captures detailed metrics, and generates a comprehensive performance report.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import copy

import google.generativeai as genai
from dotenv import load_dotenv
import os

from core import (
    State, Configuration, Plan,
    parallel_research_team_node,
    sequential_execution_node,
    load_plan_from_json,
    initialize_rate_limiter
)

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StepMetric:
    """Metrics for a single step execution."""
    step_id: int
    step_title: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    batch_id: int = -1
    error_message: str = ""


@dataclass
class BatchMetric:
    """Metrics for a batch execution."""
    batch_id: int
    parallel: bool
    step_count: int
    start_time: float
    end_time: float
    duration: float
    description: str


@dataclass
class ExecutionMetrics:
    """Complete metrics for an execution run."""
    mode: str  # "parallel" or "sequential"
    total_steps: int
    completed_steps: int
    failed_steps: int
    total_duration: float
    step_metrics: List[StepMetric]
    batch_metrics: List[BatchMetric]
    observations_count: int
    speedup_vs_baseline: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['step_metrics'] = [asdict(m) for m in self.step_metrics]
        data['batch_metrics'] = [asdict(m) for m in self.batch_metrics]
        return data


class ExecutionTimer:
    """Context manager for timing execution."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.duration = time.time() - self.start_time
        logger.info(f"{self.name} completed in {self.duration:.2f}s")


async def run_parallel_execution(plan: Plan, config: Configuration) -> tuple[State, ExecutionMetrics]:
    """Run parallel execution and capture metrics."""
    
    logger.info("\n" + "="*80)
    logger.info("PARALLEL EXECUTION MODE")
    logger.info("="*80)
    
    # Create fresh state
    state = State(
        messages=[],
        observations=[],
        current_plan=copy.deepcopy(plan)
    )
    
    # Initialize rate limiter
    initialize_rate_limiter(config.max_concurrent_tasks)
    
    # Track metrics
    step_metrics = []
    batch_metrics_list = []
    
    with ExecutionTimer("Parallel execution") as timer:
        # Execute with parallel node
        result = await parallel_research_team_node(state, config)
        # Extract batch metrics from result
        batch_metrics_list = result.get("batch_metrics", [])
    
    # Collect step metrics
    for i, step in enumerate(state.current_plan.steps):
        is_success = step.execution_res and "ERROR" not in step.execution_res
        step_metrics.append(StepMetric(
            step_id=i,
            step_title=step.title,
            start_time=0,  # Not individually tracked in parallel
            end_time=0,
            duration=0,
            success=is_success,
            error_message="" if is_success else step.execution_res
        ))
    
    # Convert batch metrics dicts to BatchMetric objects
    batch_metrics = [BatchMetric(**bm) for bm in batch_metrics_list]
    
    metrics = ExecutionMetrics(
        mode="parallel",
        total_steps=len(state.current_plan.steps),
        completed_steps=len([s for s in state.current_plan.steps if s.execution_res]),
        failed_steps=len([s for s in state.current_plan.steps if s.execution_res and "ERROR" in s.execution_res]),
        total_duration=timer.duration,
        step_metrics=step_metrics,
        batch_metrics=batch_metrics,
        observations_count=len(state.observations)
    )
    
    return state, metrics


async def run_sequential_execution(plan: Plan, config: Configuration) -> tuple[State, ExecutionMetrics]:
    """Run sequential execution and capture metrics."""
    
    logger.info("\n" + "="*80)
    logger.info("SEQUENTIAL EXECUTION MODE")
    logger.info("="*80)
    
    # Create fresh state
    state = State(
        messages=[],
        observations=[],
        current_plan=copy.deepcopy(plan)
    )
    
    # Track metrics
    step_metrics_list = []
    
    with ExecutionTimer("Sequential execution") as timer:
        # Execute with sequential node
        result = await sequential_execution_node(state, config)
        # Extract step metrics from result
        step_metrics_list = result.get("step_metrics", [])
    
    # Convert step metrics dicts to StepMetric objects
    step_metrics = [StepMetric(**sm) for sm in step_metrics_list]
    
    metrics = ExecutionMetrics(
        mode="sequential",
        total_steps=len(state.current_plan.steps),
        completed_steps=len([s for s in state.current_plan.steps if s.execution_res]),
        failed_steps=len([s for s in state.current_plan.steps if s.execution_res and "ERROR" in s.execution_res]),
        total_duration=timer.duration,
        step_metrics=step_metrics,
        batch_metrics=[],  # Sequential mode doesn't use batches
        observations_count=len(state.observations)
    )
    
    return state, metrics


def generate_llm_analysis(
    plan: Plan,
    parallel_metrics: ExecutionMetrics,
    sequential_metrics: ExecutionMetrics,
    parallel_state: State,
    sequential_state: State
) -> str:
    """Use LLM to analyze and summarize the performance comparison."""
    
    logger.info("\nGenerating LLM analysis of performance comparison...")
    
    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "ERROR: GEMINI_API_KEY not found. Cannot generate LLM analysis."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    
    # Build analysis prompt
    speedup = sequential_metrics.total_duration / parallel_metrics.total_duration if parallel_metrics.total_duration > 0 else 0
    
    prompt = f"""Analyze the following performance comparison between parallel and sequential execution of a research plan and provide insights.

# Plan Overview
**Title**: {plan.title}
**Total Steps**: {len(plan.steps)}
**Research Steps**: {sum(1 for s in plan.steps if s.need_search)}
**Processing Steps**: {sum(1 for s in plan.steps if not s.need_search)}

# Performance Metrics

## Parallel Execution
- **Total Duration**: {parallel_metrics.total_duration:.2f} seconds
- **Completed Steps**: {parallel_metrics.completed_steps}/{parallel_metrics.total_steps}
- **Failed Steps**: {parallel_metrics.failed_steps}
- **Success Rate**: {(parallel_metrics.completed_steps/parallel_metrics.total_steps)*100:.1f}%

## Sequential Execution
- **Total Duration**: {sequential_metrics.total_duration:.2f} seconds
- **Completed Steps**: {sequential_metrics.completed_steps}/{sequential_metrics.total_steps}
- **Failed Steps**: {sequential_metrics.failed_steps}
- **Success Rate**: {(sequential_metrics.completed_steps/sequential_metrics.total_steps)*100:.1f}%

## Comparison
- **Speedup**: {speedup:.2f}x
- **Time Saved**: {sequential_metrics.total_duration - parallel_metrics.total_duration:.2f} seconds
- **Efficiency Gain**: {((sequential_metrics.total_duration - parallel_metrics.total_duration) / sequential_metrics.total_duration * 100):.1f}%

# Analysis Requirements

Please provide:

1. **Executive Summary**: Brief overview of the performance comparison (2-3 sentences)

2. **Key Findings**: 
   - What was the actual speedup achieved?
   - How does this compare to the theoretical maximum?
   - Were there any bottlenecks or issues?

3. **Architectural Insights**:
   - Why did parallel execution achieve this speedup?
   - What factors limited greater speedup?
   - How does the LLM-based dependency analysis contribute?

4. **Recommendations**:
   - When should users prefer parallel vs sequential execution?
   - How could performance be further improved?
   - Any configuration adjustments suggested?

5. **Conclusion**: Overall assessment of the parallel execution system's effectiveness

Please be analytical, data-driven, and provide actionable insights. Format your response in clear markdown sections.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return f"ERROR: LLM analysis failed - {str(e)}"


def save_performance_report(
    plan: Plan,
    parallel_metrics: ExecutionMetrics,
    sequential_metrics: ExecutionMetrics,
    parallel_state: State,
    sequential_state: State,
    llm_analysis: str,
    output_path: Path
):
    """Generate and save the comprehensive performance report."""
    
    speedup = sequential_metrics.total_duration / parallel_metrics.total_duration if parallel_metrics.total_duration > 0 else 0
    efficiency_gain = ((sequential_metrics.total_duration - parallel_metrics.total_duration) / sequential_metrics.total_duration * 100) if sequential_metrics.total_duration > 0 else 0
    
    report = f"""# Performance Comparison Report: Parallel vs Sequential Execution

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Plan**: {plan.title}  
**Total Steps**: {len(plan.steps)}

---

## 📊 Executive Summary

| Metric | Parallel | Sequential | Improvement |
|--------|----------|------------|-------------|
| **Total Duration** | {parallel_metrics.total_duration:.2f}s | {sequential_metrics.total_duration:.2f}s | **{speedup:.2f}x faster** |
| **Completed Steps** | {parallel_metrics.completed_steps}/{parallel_metrics.total_steps} | {sequential_metrics.completed_steps}/{sequential_metrics.total_steps} | - |
| **Success Rate** | {(parallel_metrics.completed_steps/parallel_metrics.total_steps)*100:.1f}% | {(sequential_metrics.completed_steps/sequential_metrics.total_steps)*100:.1f}% | - |
| **Failed Steps** | {parallel_metrics.failed_steps} | {sequential_metrics.failed_steps} | - |
| **Time Saved** | - | - | **{sequential_metrics.total_duration - parallel_metrics.total_duration:.2f}s** |
| **Efficiency Gain** | - | - | **{efficiency_gain:.1f}%** |

---

## 🤖 LLM Analysis

{llm_analysis}

---

## 📈 Detailed Metrics

### Parallel Execution
- **Mode**: Parallel with LLM-based dependency analysis
- **Total Duration**: {parallel_metrics.total_duration:.2f} seconds
- **Completed Steps**: {parallel_metrics.completed_steps}
- **Failed Steps**: {parallel_metrics.failed_steps}
- **Observations Captured**: {parallel_metrics.observations_count}

### Sequential Execution
- **Mode**: Sequential (one step at a time)
- **Total Duration**: {sequential_metrics.total_duration:.2f} seconds
- **Completed Steps**: {sequential_metrics.completed_steps}
- **Failed Steps**: {sequential_metrics.failed_steps}
- **Observations Captured**: {sequential_metrics.observations_count}

---

## 🔍 Step-by-Step Comparison

| Step | Title | Parallel Status | Sequential Status |
|------|-------|----------------|-------------------|
"""
    
    # Add step comparison
    for i in range(len(plan.steps)):
        step = plan.steps[i]
        p_metric = parallel_metrics.step_metrics[i] if i < len(parallel_metrics.step_metrics) else None
        s_metric = sequential_metrics.step_metrics[i] if i < len(sequential_metrics.step_metrics) else None
        
        p_status = "✅" if p_metric and p_metric.success else "❌" if p_metric else "⏸️"
        s_status = "✅" if s_metric and s_metric.success else "❌" if s_metric else "⏸️"
        
        report += f"| {i} | {step.title[:50]}... | {p_status} | {s_status} |\n"
    
    report += f"""
---

## 💡 Key Insights

### Performance Analysis
- **Actual Speedup**: {speedup:.2f}x
- **Theoretical Maximum**: {len(plan.steps)}x (if all steps were independent)
- **Efficiency**: {(speedup / len(plan.steps) * 100):.1f}% of theoretical maximum

### Bottlenecks
- Sequential processing steps (data collation, calculation, ranking)
- API rate limits and response times
- Network latency for LLM calls

### Advantages of Parallel Execution
- Independent research steps execute concurrently
- LLM-based dependency analysis optimizes batching
- Efficient resource utilization with semaphore-based rate limiting

---

## 📋 Raw Data

### Parallel Execution Timeline
```json
{json.dumps(parallel_metrics.to_dict(), indent=2)}
```

### Sequential Execution Timeline
```json
{json.dumps(sequential_metrics.to_dict(), indent=2)}
```

---

## 🎯 Recommendations

Based on this performance comparison:

1. **Use Parallel Execution When**:
   - Plan has multiple independent research steps
   - Time-to-completion is critical
   - API rate limits allow concurrent requests

2. **Use Sequential Execution When**:
   - Steps have complex dependencies
   - Debugging execution flow
   - API rate limits are restrictive
   - Cost optimization is priority (fewer API calls)

3. **Optimization Opportunities**:
   - Increase `max_concurrent_tasks` if rate limits allow
   - Optimize LLM prompts for faster responses
   - Cache frequently accessed data
   - Implement request batching for similar queries

---

**Report Generated by**: UDF Optimizer Performance Comparison System  
**Version**: 2.0  
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n📄 Performance report saved to: {output_path}")
    return output_path


async def main():
    """Main execution function."""
    
    logger.info("="*80)
    logger.info("PERFORMANCE COMPARISON: PARALLEL VS SEQUENTIAL EXECUTION")
    logger.info("="*80)
    
    script_dir = Path(__file__).parent
    
    # Load plan
    plan_path = script_dir / "examples" / "example_response_short.txt"
    logger.info(f"\n📋 Loading plan: {plan_path}")
    plan = load_plan_from_json(plan_path)
    logger.info(f"   Plan: {plan.title}")
    logger.info(f"   Total steps: {len(plan.steps)}")
    
    # Configuration
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
    
    try:
        # Run parallel execution
        parallel_state, parallel_metrics = await run_parallel_execution(plan, config)
        
        # Wait a bit between runs to avoid rate limits
        logger.info("\n⏳ Waiting 10 seconds before sequential run...")
        await asyncio.sleep(10)
        
        # Run sequential execution
        sequential_state, sequential_metrics = await run_sequential_execution(plan, config)
        
        # Generate LLM analysis
        llm_analysis = generate_llm_analysis(
            plan,
            parallel_metrics,
            sequential_metrics,
            parallel_state,
            sequential_state
        )
        
        # Save comprehensive report
        report_path = script_dir / "examples" / "example_performance_report.md"
        save_performance_report(
            plan,
            parallel_metrics,
            sequential_metrics,
            parallel_state,
            sequential_state,
            llm_analysis,
            report_path
        )
        
        # Print summary
        speedup = sequential_metrics.total_duration / parallel_metrics.total_duration if parallel_metrics.total_duration > 0 else 0
        logger.info("\n" + "="*80)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*80)
        logger.info(f"Parallel Duration:   {parallel_metrics.total_duration:.2f}s")
        logger.info(f"Sequential Duration: {sequential_metrics.total_duration:.2f}s")
        logger.info(f"Speedup:            {speedup:.2f}x")
        logger.info(f"Time Saved:         {sequential_metrics.total_duration - parallel_metrics.total_duration:.2f}s")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Comparison test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
