# Test Suite

This directory contains the test suite for the parallel execution optimization system.

## Test Files

### 1. `validate.py` - Smoke Test
**Purpose:** Quick validation that all components can be imported and basic configuration works.

**What it tests:**
- All imports work without errors
- Plan can be loaded from JSON
- DependencyAnalyzer and GeminiStepExecutor can be instantiated
- API key is configured
- Configuration objects can be created

**When to run:**
- Before running main execution
- After making changes to imports or core structure
- As a quick sanity check

**Run:** `python tests/validate.py`

**No LLM calls, runs instantly.**

---

### 2. `test_unit.py` - Unit Tests
**Purpose:** Test individual components and dataclasses without executing workflows.

**What it tests:**
- StepMetric dataclass instantiation
- BatchMetric dataclass instantiation
- ExecutionMetrics dataclass with nested metrics
- Configuration object creation
- Plan loading from JSON

**When to run:**
- After modifying dataclass definitions
- When debugging metric collection issues
- As part of CI/CD pipeline

**Run:** `python tests/test_unit.py`

**No LLM calls, runs instantly.**

---

### 3. `test_integration.py` - Integration Tests
**Purpose:** Test parallel and sequential execution modes separately with real LLM calls.

**What it tests:**
- **Parallel execution:**
  - Batch metrics collection
  - Parallel batching logic
  - Batch metric field validation
  
- **Sequential execution:**
  - Step metrics collection
  - Step-by-step execution
  - Step metric field validation

**When to run:**
- After modifying execution logic
- When debugging batch/step metrics
- To validate individual execution modes

**Run:** `python tests/test_integration.py`

**⚠️ Makes LLM API calls (~20 steps total), takes ~2-3 minutes.**

---

### 4. `test_e2e.py` - End-to-End Test
**Purpose:** Complete workflow test from parallel execution through report generation.

**What it tests:**
- Parallel execution with batch metrics
- Sequential execution with step metrics
- Performance comparison calculation
- LLM-generated analysis
- Report file generation and validation
- Speedup metrics accuracy

**Test plans:**
- **Short plan:** `example_response_short.txt` (10 steps) - quick validation
- **Long plan:** `example_response_long.txt` (30 steps) - comprehensive test

**When to run:**
- Before releasing changes
- To validate complete workflow
- When debugging comparison system
- To test scalability with different plan sizes

**Run:** `python tests/test_e2e.py`

**Run single plan:**
```python
# In Python script or interactive session:
import asyncio
from tests.test_e2e import test_full_comparison

# Test with short plan
asyncio.run(test_full_comparison("example_response_short.txt"))

# Test with long plan
asyncio.run(test_full_comparison("example_response_long.txt"))
```

**⚠️ Makes extensive LLM API calls:**
- Short plan: ~20 steps × 2 modes = 40 LLM calls, takes ~5-8 minutes
- Long plan: ~60 steps × 2 modes = 120 LLM calls, takes ~15-25 minutes
- Both plans: ~160 LLM calls total, takes ~20-30 minutes

---

## Test Plan Files

Tests use two plan files with different complexity levels:

### `example_response_short.txt` (10 steps)
- 5 research steps (1 sequential, 3 parallel, 1 sequential)
- 5 processing steps (all currently sequential)
- **Use case:** Quick validation, integration tests, fast feedback
- **Execution time:** ~5-8 minutes for full comparison

### `example_response_long.txt` (30 steps)
- More complex plan with multiple parallel opportunities
- **Use case:** Scalability testing, comprehensive validation
- **Execution time:** ~15-25 minutes for full comparison

Both plans are chosen because:
- Contain both research and processing steps
- Demonstrate realistic dependency patterns
- Test parallel batching logic
- Provide different complexity levels for testing

---

## Running Tests

### Run Individual Tests
```bash
# Quick validation (no LLM calls)
python tests/validate.py

# Unit tests (no LLM calls)
python tests/test_unit.py

# Integration tests (requires API key, ~2-3 min)
python tests/test_integration.py

# End-to-end test (requires API key, ~5-8 min)
python tests/test_e2e.py
```

### Run All Tests
```bash
# Run in order (recommended)
python tests/validate.py && \
python tests/test_unit.py && \
python tests/test_integration.py && \
python tests/test_e2e.py
```

---

### Test Outputs

### Console Logs
All tests output detailed logs showing:
- Step/batch execution progress
- Timing for each step/batch
- Success/failure status
- Validation results

### Generated Files
- `examples/test_performance_report_example_response_short.md` - Short plan comparison report
- `examples/test_performance_report_example_response_long.md` - Long plan comparison report
- Each report contains LLM analysis and performance comparison for its respective plan

---

## Environment Requirements

### For validate.py and test_unit.py:
- Python 3.11+
- Required packages installed
- No API key needed

### For test_integration.py and test_e2e.py:
- Python 3.11+
- Required packages installed
- `GEMINI_API_KEY` in `.env` file
- Active internet connection

---

## Interpreting Results

### Success Indicators
- All assertions pass ✓
- No exceptions raised
- Exit code 0
- Metrics properly structured
- Report file created (for e2e test)

### Common Failures

**Import Errors:**
- Run `validate.py` to check imports
- Verify all packages installed: `pip install -r requirements.txt`

**API Key Errors:**
- Check `.env` file exists
- Verify `GEMINI_API_KEY` is set
- Test with: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GEMINI_API_KEY')[:10])"`

**Batch Metrics Missing Fields:**
- Indicates nodes.py not returning proper metrics structure
- Check nodes.py returns dict with "batch_metrics" key
- Verify BatchMetric dataclass has all required fields

**Sequential Metrics Missing:**
- Indicates sequential_executor.py not returning metrics
- Check sequential_executor.py returns dict with "step_metrics" key
- Verify StepMetric dataclass has all required fields

---

## Test Development Guidelines

### Adding New Tests

1. **Unit tests** - Add to `test_unit.py` if testing individual components
2. **Integration tests** - Add to `test_integration.py` if testing execution modes
3. **E2E tests** - Add to `test_e2e.py` if testing full workflow

### Test Naming
- Unit tests: `test_<component>_<aspect>()`
- Integration tests: `test_<mode>_execution()`
- E2E tests: `test_<workflow>_<scenario>()`

### Assertions
- Use descriptive assertion messages
- Validate all critical fields
- Check for expected errors

---

## Troubleshooting

### Test Hangs
- Check for infinite loops in execution logic
- Verify timeout settings in Configuration
- Check network connectivity for API calls

### Incorrect Metrics
- Add debug logging to nodes.py and sequential_executor.py
- Print metrics before assertion to see actual structure
- Verify dataclass field names match exactly

### Rate Limiting
- Tests include 10-second delays between executions
- If still hitting limits, increase delays in test code
- Consider using smaller test plans

---

## Quick Reference

| Test | LLM Calls | Duration | Purpose |
|------|-----------|----------|---------|
| validate.py | ❌ | <1s | Smoke test |
| test_unit.py | ❌ | <1s | Unit tests |
| test_integration.py | ✅ | ~2-3min | Execution modes |
| test_e2e.py (short) | ✅ | ~5-8min | Full workflow (10 steps) |
| test_e2e.py (long) | ✅ | ~15-25min | Full workflow (30 steps) |
| test_e2e.py (both) | ✅ | ~20-30min | Complete validation |
