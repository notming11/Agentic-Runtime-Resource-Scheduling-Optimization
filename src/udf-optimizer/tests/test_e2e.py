"""
End-to-End Comparison Test - Validates full parallel vs sequential workflow.

This test:
1. Runs parallel execution and collects metrics
2. Runs sequential execution and collects metrics  
3. Generates LLM-analyzed performance report
4. Validates all metrics are properly collected

Tests both short and long example plans.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Configuration, load_plan_from_json
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


async def test_full_comparison(plan_file: str = "example_response_short.txt"):
    """
    End-to-end test of parallel vs sequential comparison.
    
    Args:
        plan_file: Name of the plan file in examples/ directory
    
    Tests that:
    - Parallel execution completes and collects batch metrics
    - Sequential execution completes and collects step metrics
    - LLM analysis generates successfully
    - Performance report is saved
    - All metrics are valid
    """
    
    logger.info("\n" + "="*80)
    logger.info(f"END-TO-END COMPARISON TEST - {plan_file}")
    logger.info("="*80 + "\n")
    
    # Load plan
    plan_path = Path(__file__).parent.parent / "examples" / plan_file
    logger.info(f"📋 Loading plan: {plan_path}")
    
    if not plan_path.exists():
        logger.error(f"❌ Plan file not found: {plan_path}")
        return False
    
    plan = load_plan_from_json(str(plan_path))
    logger.info(f"✓ Plan loaded: {plan.title} ({len(plan.steps)} steps)\n")
    
    # Configuration
    config = Configuration(
        enabled=True,
        max_concurrent_tasks=3,
        task_timeout_seconds=120
    )
    
    try:
        # ============================================================
        # STEP 1: Parallel Execution
        # ============================================================
        logger.info("="*80)
        logger.info("STEP 1: Running Parallel Execution")
        logger.info("="*80)
        
        parallel_state, parallel_metrics = await run_parallel_execution(plan, config)
        
        logger.info(f"\n✓ Parallel execution completed")
        logger.info(f"  Duration: {parallel_metrics.total_duration:.2f}s")
        logger.info(f"  Steps: {parallel_metrics.completed_steps}/{parallel_metrics.total_steps}")
        logger.info(f"  Failed: {parallel_metrics.failed_steps}")
        logger.info(f"  Batches: {len(parallel_metrics.batch_metrics)}")
        
        # Validate parallel metrics
        assert parallel_metrics.mode == "parallel", "Mode should be 'parallel'"
        assert parallel_metrics.total_steps == len(plan.steps), f"Should have {len(plan.steps)} steps"
        assert len(parallel_metrics.batch_metrics) > 0, "Should have batch metrics"
        
        # Validate each batch metric has all required fields
        for i, bm in enumerate(parallel_metrics.batch_metrics):
            assert hasattr(bm, 'batch_id'), f"Batch {i} missing batch_id"
            assert hasattr(bm, 'parallel'), f"Batch {i} missing parallel flag"
            assert hasattr(bm, 'step_count'), f"Batch {i} missing step_count"
            assert hasattr(bm, 'start_time'), f"Batch {i} missing start_time"
            assert hasattr(bm, 'end_time'), f"Batch {i} missing end_time"
            assert hasattr(bm, 'duration'), f"Batch {i} missing duration"
            assert hasattr(bm, 'description'), f"Batch {i} missing description"
            assert bm.duration > 0, f"Batch {i} duration should be > 0"
            logger.info(f"  Batch {bm.batch_id}: {bm.duration:.2f}s ({bm.step_count} steps, {'parallel' if bm.parallel else 'sequential'})")
        
        logger.info(f"\n✓ Parallel metrics validation passed")
        
        # ============================================================
        # STEP 2: Sequential Execution
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Running Sequential Execution")
        logger.info("="*80)
        
        # Wait between runs to avoid rate limits
        logger.info("⏳ Waiting 10 seconds to avoid rate limits...")
        await asyncio.sleep(10)
        
        sequential_state, sequential_metrics = await run_sequential_execution(plan, config)
        
        logger.info(f"\n✓ Sequential execution completed")
        logger.info(f"  Duration: {sequential_metrics.total_duration:.2f}s")
        logger.info(f"  Steps: {sequential_metrics.completed_steps}/{sequential_metrics.total_steps}")
        logger.info(f"  Failed: {sequential_metrics.failed_steps}")
        logger.info(f"  Step metrics: {len(sequential_metrics.step_metrics)}")
        
        # Validate sequential metrics
        assert sequential_metrics.mode == "sequential", "Mode should be 'sequential'"
        assert sequential_metrics.total_steps == len(plan.steps), f"Should have {len(plan.steps)} steps"
        assert len(sequential_metrics.step_metrics) > 0, "Should have step metrics"
        assert len(sequential_metrics.batch_metrics) == 0, "Sequential mode should have no batch metrics"
        
        # Validate each step metric has all required fields
        for i, sm in enumerate(sequential_metrics.step_metrics):
            assert hasattr(sm, 'step_id'), f"Step {i} missing step_id"
            assert hasattr(sm, 'step_title'), f"Step {i} missing step_title"
            assert hasattr(sm, 'start_time'), f"Step {i} missing start_time"
            assert hasattr(sm, 'end_time'), f"Step {i} missing end_time"
            assert hasattr(sm, 'duration'), f"Step {i} missing duration"
            assert hasattr(sm, 'success'), f"Step {i} missing success"
            assert sm.duration > 0, f"Step {i} duration should be > 0"
            logger.info(f"  Step {sm.step_id}: {sm.duration:.2f}s ({'✓' if sm.success else '✗'})")
        
        logger.info(f"\n✓ Sequential metrics validation passed")
        
        # ============================================================
        # STEP 3: Performance Comparison
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Generating Performance Comparison")
        logger.info("="*80)
        
        speedup = sequential_metrics.total_duration / parallel_metrics.total_duration if parallel_metrics.total_duration > 0 else 0
        time_saved = sequential_metrics.total_duration - parallel_metrics.total_duration
        
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"  Parallel Duration:   {parallel_metrics.total_duration:.2f}s")
        logger.info(f"  Sequential Duration: {sequential_metrics.total_duration:.2f}s")
        logger.info(f"  Speedup:            {speedup:.2f}x")
        logger.info(f"  Time Saved:         {time_saved:.2f}s ({time_saved/60:.1f} minutes)")
        
        assert speedup > 0, "Speedup should be positive"
        
        # ============================================================
        # STEP 4: Generate LLM Analysis and Report
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Generating LLM Analysis and Report")
        logger.info("="*80)
        
        logger.info("\n📝 Generating LLM analysis...")
        llm_analysis = generate_llm_analysis(
            plan,
            parallel_metrics,
            sequential_metrics,
            parallel_state,
            sequential_state
        )
        
        assert llm_analysis is not None, "LLM analysis should not be None"
        assert len(llm_analysis) > 0, "LLM analysis should not be empty"
        logger.info(f"✓ LLM analysis generated ({len(llm_analysis)} characters)")
        
        # Save report with plan-specific filename
        plan_name = Path(plan_file).stem  # Get filename without extension
        report_filename = f"test_performance_report_{plan_name}.md"
        report_path = Path(__file__).parent.parent / "examples" / report_filename
        logger.info(f"\n💾 Saving performance report to: {report_path}")
        
        save_performance_report(
            plan,
            parallel_metrics,
            sequential_metrics,
            parallel_state,
            sequential_state,
            llm_analysis,
            report_path
        )
        
        assert report_path.exists(), "Report file should be created"
        
        # Validate report content
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert len(report_content) > 0, "Report should not be empty"
        assert "Parallel" in report_content, "Report should mention parallel execution"
        assert "Sequential" in report_content, "Report should mention sequential execution"
        assert f"{speedup:.2f}" in report_content, "Report should include speedup metric"
        
        logger.info(f"✓ Report saved successfully ({len(report_content)} characters)")
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info(f"TEST SUMMARY - {plan_file}")
        logger.info("="*80)
        logger.info("✓ Parallel execution: PASSED")
        logger.info("✓ Sequential execution: PASSED")
        logger.info("✓ Metrics validation: PASSED")
        logger.info("✓ LLM analysis: PASSED")
        logger.info("✓ Report generation: PASSED")
        logger.info("="*80)
        logger.info("🎉 ALL TESTS PASSED")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run E2E tests with both short and long plans."""
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING E2E TESTS WITH MULTIPLE PLANS")
    logger.info("="*80)
    
    results = []
    
    # Test 1: Short plan
    logger.info("\n" + "🔹 TEST 1: SHORT PLAN (10 steps)")
    success = await test_full_comparison("example_response_short.txt")
    results.append(("Short Plan (10 steps)", success))
    
    if success:
        # Wait between tests to avoid rate limits
        logger.info("\n⏳ Waiting 15 seconds before long plan test...")
        await asyncio.sleep(15)
    
    # Test 2: Long plan
    logger.info("\n" + "🔹 TEST 2: LONG PLAN (30 steps)")
    success = await test_full_comparison("example_response_long.txt")
    results.append(("Long Plan (30 steps)", success))
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    logger.info(f"\n{total_passed}/{total_tests} test plans passed")
    logger.info("="*80 + "\n")
    
    return all(r[1] for r in results)


if __name__ == "__main__":
    # Run all tests by default
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
