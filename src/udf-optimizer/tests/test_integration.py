"""
Integration Tests - Test parallel and sequential execution separately.

These tests validate:
- Parallel execution with batch metrics collection
- Sequential execution with step metrics collection
- Metrics structure and validation
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Configuration, load_plan_from_json
from compare_performance import run_parallel_execution, run_sequential_execution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


async def test_parallel_execution():
    """Test parallel execution and batch metrics collection."""
    
    logger.info("\n" + "="*80)
    logger.info("Testing Parallel Execution")
    logger.info("="*80)
    
    # Load plan
    plan_path = Path(__file__).parent.parent / "examples" / "example_response_short.txt"
    if not plan_path.exists():
        logger.error(f"Plan file not found: {plan_path}")
        return False
    
    plan = load_plan_from_json(str(plan_path))
    logger.info(f"Plan loaded: {plan.title} ({len(plan.steps)} steps)")
    
    # Configuration
    config = Configuration(
        enabled=True,
        max_concurrent_tasks=3,
        task_timeout_seconds=120
    )
    
    try:
        # Run parallel execution
        state, metrics = await run_parallel_execution(plan, config)
        
        # Validate results
        assert metrics is not None, "Metrics should not be None"
        assert metrics.mode == "parallel", "Mode should be 'parallel'"
        assert metrics.total_steps == len(plan.steps), f"Should have {len(plan.steps)} steps"
        assert len(metrics.batch_metrics) > 0, "Should have batch metrics"
        
        logger.info(f"✓ Parallel execution completed")
        logger.info(f"  Duration: {metrics.total_duration:.2f}s")
        logger.info(f"  Steps: {metrics.completed_steps}/{metrics.total_steps}")
        logger.info(f"  Batches: {len(metrics.batch_metrics)}")
        
        # Validate batch metrics structure
        for i, bm in enumerate(metrics.batch_metrics):
            assert hasattr(bm, 'batch_id'), f"Batch {i} missing batch_id"
            assert hasattr(bm, 'parallel'), f"Batch {i} missing parallel flag"
            assert hasattr(bm, 'step_count'), f"Batch {i} missing step_count"
            assert hasattr(bm, 'start_time'), f"Batch {i} missing start_time"
            assert hasattr(bm, 'end_time'), f"Batch {i} missing end_time"
            assert hasattr(bm, 'duration'), f"Batch {i} missing duration"
            assert hasattr(bm, 'description'), f"Batch {i} missing description"
            assert bm.duration > 0, f"Batch {i} duration should be > 0"
            
            logger.info(f"  Batch {bm.batch_id}: {bm.duration:.2f}s ({bm.step_count} steps, {'parallel' if bm.parallel else 'sequential'})")
        
        logger.info("✓ All batch metrics validated")
        return True
        
    except Exception as e:
        logger.error(f"✗ Parallel execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sequential_execution():
    """Test sequential execution and step metrics collection."""
    
    logger.info("\n" + "="*80)
    logger.info("Testing Sequential Execution")
    logger.info("="*80)
    
    # Load plan
    plan_path = Path(__file__).parent.parent / "examples" / "example_response_short.txt"
    if not plan_path.exists():
        logger.error(f"Plan file not found: {plan_path}")
        return False
    
    plan = load_plan_from_json(str(plan_path))
    logger.info(f"Plan loaded: {plan.title} ({len(plan.steps)} steps)")
    
    # Configuration
    config = Configuration(
        enabled=True,
        max_concurrent_tasks=1,
        task_timeout_seconds=120
    )
    
    try:
        # Run sequential execution
        state, metrics = await run_sequential_execution(plan, config)
        
        # Validate results
        assert metrics is not None, "Metrics should not be None"
        assert metrics.mode == "sequential", "Mode should be 'sequential'"
        assert metrics.total_steps == len(plan.steps), f"Should have {len(plan.steps)} steps"
        assert len(metrics.step_metrics) > 0, "Should have step metrics"
        assert len(metrics.batch_metrics) == 0, "Sequential mode should have no batch metrics"
        
        logger.info(f"✓ Sequential execution completed")
        logger.info(f"  Duration: {metrics.total_duration:.2f}s")
        logger.info(f"  Steps: {metrics.completed_steps}/{metrics.total_steps}")
        
        # Validate step metrics structure
        for i, sm in enumerate(metrics.step_metrics):
            assert hasattr(sm, 'step_id'), f"Step {i} missing step_id"
            assert hasattr(sm, 'step_title'), f"Step {i} missing step_title"
            assert hasattr(sm, 'start_time'), f"Step {i} missing start_time"
            assert hasattr(sm, 'end_time'), f"Step {i} missing end_time"
            assert hasattr(sm, 'duration'), f"Step {i} missing duration"
            assert hasattr(sm, 'success'), f"Step {i} missing success"
            assert sm.duration > 0, f"Step {i} duration should be > 0"
            
            logger.info(f"  Step {sm.step_id}: {sm.duration:.2f}s ({'✓' if sm.success else '✗'})")
        
        logger.info("✓ All step metrics validated")
        return True
        
    except Exception as e:
        logger.error(f"✗ Sequential execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests."""
    
    logger.info("\n" + "="*80)
    logger.info("INTEGRATION TESTS")
    logger.info("="*80)
    
    results = []
    
    # Test parallel execution
    result = await test_parallel_execution()
    results.append(("Parallel Execution", result))
    
    # Wait before sequential to avoid rate limits
    if result:
        logger.info("\n⏳ Waiting 10 seconds before sequential test...")
        await asyncio.sleep(10)
    
    # Test sequential execution
    result = await test_sequential_execution()
    results.append(("Sequential Execution", result))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    logger.info(f"\n{total_passed}/{total_tests} tests passed")
    logger.info("="*80 + "\n")
    
    return all(r[1] for r in results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
