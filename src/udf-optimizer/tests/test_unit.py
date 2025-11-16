"""
Unit Tests - Test dataclasses and components without LLM execution.

These tests validate:
- Dataclass instantiation and field validation
- Configuration objects
- Basic imports and structure
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compare_performance import StepMetric, BatchMetric, ExecutionMetrics
from core import Configuration, load_plan_from_json


def test_step_metric_instantiation():
    """Test StepMetric dataclass can be instantiated with correct fields."""
    step_metric = StepMetric(
        step_id=0,
        step_title="Test Step",
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        success=True,
        batch_id=0,
        error_message=""
    )
    
    assert step_metric.step_id == 0
    assert step_metric.step_title == "Test Step"
    assert step_metric.duration == 1.0
    assert step_metric.success == True
    print("✓ StepMetric instantiation test passed")


def test_batch_metric_instantiation():
    """Test BatchMetric dataclass can be instantiated with correct fields."""
    batch_metric = BatchMetric(
        batch_id=0,
        parallel=True,
        step_count=3,
        start_time=0.0,
        end_time=10.0,
        duration=10.0,
        description="Test batch"
    )
    
    assert batch_metric.batch_id == 0
    assert batch_metric.parallel == True
    assert batch_metric.step_count == 3
    assert batch_metric.duration == 10.0
    assert batch_metric.description == "Test batch"
    print("✓ BatchMetric instantiation test passed")


def test_execution_metrics_instantiation():
    """Test ExecutionMetrics dataclass with nested metrics."""
    step_metric = StepMetric(
        step_id=0,
        step_title="Test",
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        success=True
    )
    
    batch_metric = BatchMetric(
        batch_id=0,
        parallel=True,
        step_count=1,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        description="Test"
    )
    
    metrics = ExecutionMetrics(
        mode="parallel",
        total_steps=10,
        completed_steps=10,
        failed_steps=0,
        total_duration=50.0,
        step_metrics=[step_metric],
        batch_metrics=[batch_metric],
        observations_count=10
    )
    
    assert metrics.mode == "parallel"
    assert len(metrics.step_metrics) == 1
    assert len(metrics.batch_metrics) == 1
    assert metrics.total_duration == 50.0
    print("✓ ExecutionMetrics instantiation test passed")


def test_configuration_creation():
    """Test Configuration object creation."""
    config = Configuration(
        enabled=True,
        max_concurrent_tasks=5,
        task_timeout_seconds=120,
        max_retries=2
    )
    
    assert config.enabled == True
    assert config.max_concurrent_tasks == 5
    assert config.task_timeout_seconds == 120
    assert config.max_retries == 2
    print("✓ Configuration creation test passed")


def test_plan_loading():
    """Test that plan can be loaded from JSON file."""
    plan_path = Path(__file__).parent.parent / "examples" / "example_response_short.txt"
    
    if not plan_path.exists():
        print(f"⚠ Skipping plan loading test - file not found: {plan_path}")
        return
    
    plan = load_plan_from_json(str(plan_path))
    
    assert plan is not None
    assert hasattr(plan, 'title')
    assert hasattr(plan, 'steps')
    assert len(plan.steps) > 0
    print(f"✓ Plan loading test passed (loaded {len(plan.steps)} steps)")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("UNIT TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_step_metric_instantiation,
        test_batch_metric_instantiation,
        test_execution_metrics_instantiation,
        test_configuration_creation,
        test_plan_loading
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
