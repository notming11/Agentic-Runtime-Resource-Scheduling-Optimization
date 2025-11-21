# RSM End-to-End Testing Guide

## Overview

The `test_e2e.py` file provides comprehensive end-to-end testing for the Autellix Resource Scheduler Module (RSM). It validates the complete system integration across all four partitions:

1. **Frontend** - Process table and session management
2. **Scheduler** - PLAS/ATLAS priority scheduling
3. **Load Balancer** - Data locality-aware routing
4. **Engine Manager** - Multi-engine orchestration

## Test Scenarios

The E2E test runs four comprehensive scenarios:

### Scenario 1: Single-Threaded Chatbot (PLAS)
- **Purpose:** Validate PLAS (Program-Level Attained Service) scheduling
- **Pattern:** Sequential requests (typical chatbot workflow)
- **Validates:**
  - Process table registration
  - Session lifecycle management
  - Basic request routing
  - Service time tracking

### Scenario 2: Multi-Threaded Research Agent (ATLAS)
- **Purpose:** Validate ATLAS scheduling with Map-Reduce pattern
- **Pattern:** Plan → Parallel Research → Aggregate
- **Validates:**
  - Thread dependency tracking
  - Critical path estimation
  - Parallel request execution
  - Priority calculation based on service time

### Scenario 3: Load Balancing Validation
- **Purpose:** Validate load distribution across engines
- **Pattern:** Multiple concurrent sessions
- **Validates:**
  - Request distribution across engines
  - Load balancing fairness
  - Concurrent session handling

### Scenario 4: Cache Locality Validation
- **Purpose:** Validate KV cache affinity routing
- **Pattern:** Sequential long requests (>2048 tokens)
- **Validates:**
  - Cache-aware routing for long requests
  - Session affinity preservation
  - Cache hit rate optimization

## Running the Test

### Basic Usage

```bash
cd /workspace/Huawei/src/rsm/tests
python test_e2e.py
```

### Quick Test (Reduced Load)

To run a faster version for quick validation:

```python
# In test_e2e.py, modify the main() function:
results = await run_e2e_test(
    num_engines=2,  # Use fewer engines
    skip_long_scenarios=True  # Run shorter scenarios
)
```

### Requirements

The test uses **mock engines** by default, so it works without vLLM or GPU:
- Python 3.8+
- asyncio
- RSM module dependencies

For **production testing with real vLLM engines**:
- vLLM installed
- GPU(s) available
- Update `AutellixConfig.model` to real model path (e.g., `"meta-llama/Llama-2-7b-hf"`)

## Test Output

### Console Output

The test provides detailed logging:

```
================================================================================
AUTELLIX RSM - END-TO-END TEST SUITE
================================================================================

Initializing Autellix system with 4 engines...
✓ System initialized

================================================================================
SCENARIO 1: Single-Threaded Chatbot (PLAS Scheduling)
================================================================================

Started session: session_abc123
Submitting 10 sequential requests...

Request  1/10: latency=0.523s, tokens=98, engine=engine_0, priority=1
Request  2/10: latency=0.487s, tokens=102, engine=engine_0, priority=1
...

Scenario completed in 5.23s
Average latency: 0.501s

...
```

### Performance Report

The test generates a comprehensive Markdown report at:
```
/workspace/Huawei/src/rsm/examples/e2e_test_report.md
```

Report includes:
- **Executive Summary** - All scenarios at a glance
- **Detailed Metrics** - Per-scenario performance breakdown
- **Engine Distribution** - Load balancing visualization
- **Cache Statistics** - Locality hit rates
- **PLAS vs ATLAS Comparison** - Scheduling mode analysis
- **System Statistics** - Global metrics across all partitions

### Example Report Section

```markdown
## Multi-Threaded Research Agent

**Scheduling Mode:** ATLAS

### Performance Metrics

- **Total Duration:** 3.45s
- **Total Requests:** 7
- **Average Latency:** 0.493s
- **Min Latency:** 0.412s
- **Max Latency:** 0.587s
- **Total Tokens:** 1250

### Engine Distribution

| Engine | Requests | Percentage |
|--------|----------|------------|
| engine_0 | 2 | 28.6% |
| engine_1 | 3 | 42.9% |
| engine_2 | 2 | 28.6% |

### Session Statistics

- **service_time:** 2.31s
- **total_calls:** 7
- **active_threads:** 6
```

## Metrics Collected

### Request-Level Metrics
- `latency` - End-to-end request latency
- `tokens` - Tokens generated
- `engine_id` - Assigned engine
- `priority` - Scheduler priority
- `service_time` - Actual execution time
- `waiting_time` - Time in queue
- `thread_id` - Thread identifier (for ATLAS)

### Scenario-Level Metrics
- `total_duration` - Wall-clock time
- `avg_latency` - Mean request latency
- `min/max_latency` - Latency bounds
- `engine_distribution` - Load distribution
- `cache_hit_rate` - Cache locality effectiveness

### System-Level Metrics
- **Frontend:** Total programs, active calls, completed calls
- **Scheduler:** Enqueues, dequeues, priority distributions
- **Load Balancer:** Total requests, locality hits, routing decisions
- **Engines:** Throughput, latencies, completion rates

## Validation Checks

The test includes comprehensive assertions:

✓ **Basic Validations:**
- All requests complete successfully
- Valid latencies and token counts
- Engines assigned to all requests

✓ **Scenario-Specific Validations:**
- Cache locality: Requests route to same engine
- Load balancing: Distribution fairness (balance ratio > 0.5)
- ATLAS: Thread dependencies honored
- PLAS: Sequential execution order

✓ **System Validations:**
- Process table consistency
- Scheduler queue integrity
- Load balancer statistics accuracy

## Customization

### Adjust Test Parameters

```python
# Modify scenario parameters
async def main():
    results = await run_e2e_test(
        num_engines=4,           # Number of engines
        skip_long_scenarios=False # Skip lengthy tests
    )
```

### Add Custom Scenarios

```python
async def scenario_custom(system: AutellixSystem) -> ScenarioMetrics:
    """Custom test scenario."""
    metrics = ScenarioMetrics(
        scenario_name="Custom Test",
        mode="PLAS",
        total_duration=0.0,
        total_requests=0
    )

    # Your test logic here
    async with system.create_client() as client:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Test"}]
        )
        # Collect metrics...

    metrics.compute_aggregates()
    return metrics
```

### Modify Report Generation

The `generate_performance_report()` function can be customized to include:
- Custom visualizations
- Additional metrics
- Comparative analysis
- Trend graphs

## Troubleshooting

### Common Issues

**Issue:** `ImportError: No module named 'rsm'`
- **Solution:** Run from correct directory or add to Python path

**Issue:** Engines fail to start
- **Solution:** Check GPU availability or use mock engines

**Issue:** Timeouts during test
- **Solution:** Increase timeout values or reduce test load

**Issue:** Cache locality test fails
- **Solution:** Ensure `cache_token_threshold` is set correctly (default: 2048)

### Debug Mode

Enable detailed logging:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Integration with CI/CD

### Quick Smoke Test
```bash
# Fast test for CI pipelines
python test_e2e.py --quick
```

### Full Regression Test
```bash
# Complete test suite
python test_e2e.py --full
```

### Expected Runtime
- **Quick Test:** ~30-60 seconds
- **Full Test:** ~3-5 minutes (with mock engines)
- **Production Test:** ~10-15 minutes (with real vLLM)

## Comparison with UDF Optimizer E2E Test

| Aspect | UDF Optimizer | RSM |
|--------|---------------|-----|
| **Focus** | Task parallelization | Request scheduling |
| **Metrics** | Batch speedup, task latencies | Scheduler priorities, load distribution |
| **Comparison** | Parallel vs Sequential | PLAS vs ATLAS |
| **Key Validation** | Dependency resolution | Cache locality, fairness |
| **Report** | Speedup analysis | Scheduling analysis |

## Next Steps

After running the E2E test:

1. **Review Report** - Analyze performance characteristics
2. **Benchmark Baselines** - Establish expected metrics for regression testing
3. **Optimize Configuration** - Tune scheduler parameters based on results
4. **Production Testing** - Run with real vLLM engines and workloads
5. **Continuous Monitoring** - Integrate metrics into observability stack

## References

- **RSM README:** `/workspace/Huawei/src/rsm/README.md`
- **CLAUDE.md:** `/workspace/Huawei/CLAUDE.md`
- **Autellix Paper:** `/workspace/Huawei/autellix.pdf`
- **UDF E2E Test:** `/workspace/Huawei/src/udf-optimizer/tests/test_e2e.py`
