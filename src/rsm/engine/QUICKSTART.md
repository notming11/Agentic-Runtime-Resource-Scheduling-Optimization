# Quick Start Guide - Multi-Engine Orchestration

Get started with the Autellix RSM multi-engine orchestration system in 5 minutes.

## Prerequisites

```bash
# Python 3.8+
python --version

# Optional: vLLM for real LLM serving
pip install vllm

# Optional: CUDA for GPU acceleration
nvidia-smi
```

## 5-Minute Quick Start

### 1. Basic Setup (Mock Mode)

Works without vLLM - great for testing the orchestration logic:

```python
import asyncio
from rsm.engine import MultiEngineManager, EngineConfig

async def basic_example():
    # Create manager
    manager = MultiEngineManager()

    # Add 2 mock engines
    manager.add_engine(EngineConfig(
        engine_id="engine_0",
        model="mock-model",  # Uses mock for testing
        gpu_id=0
    ))
    manager.add_engine(EngineConfig(
        engine_id="engine_1",
        model="mock-model",
        gpu_id=1
    ))

    # Start engines
    manager.start()

    # Submit request
    future = await manager.submit_request(
        session_id="my_session",
        prompt="Explain AI in simple terms",
        sampling_params={"max_tokens": 100}
    )

    # Get result
    result = await future
    print(f"Response: {result.text}")
    print(f"Latency: {result.latency:.3f}s")

    # Cleanup
    manager.stop()

# Run
asyncio.run(basic_example())
```

### 2. With Real vLLM (Production)

Using actual vLLM engines for real LLM serving:

```python
import asyncio
from rsm.engine import MultiEngineManager, EngineConfig

async def production_example():
    manager = MultiEngineManager()

    # Add real vLLM engines
    for gpu_id in range(4):  # 4 GPUs
        manager.add_engine(EngineConfig(
            engine_id=f"engine_{gpu_id}",
            model="meta-llama/Llama-2-7b-hf",  # Real model
            gpu_id=gpu_id,
            max_num_seqs=128,
            max_model_len=4096
        ))

    manager.start()

    # Submit multiple requests
    futures = []
    for i in range(10):
        future = await manager.submit_request(
            session_id=f"workflow_{i}",
            prompt=f"Question {i}: What is machine learning?",
            sampling_params={
                "temperature": 0.7,
                "max_tokens": 256,
                "top_p": 0.9
            }
        )
        futures.append(future)

    # Wait for all results
    results = await asyncio.gather(*futures)

    # Show stats
    stats = manager.get_stats()
    print(f"Completed: {stats['completed_requests']}/{stats['total_requests']}")
    print(f"Avg latency: {stats['average_latency']:.3f}s")

    manager.stop()

asyncio.run(production_example())
```

### 3. Request Cancellation

```python
async def cancellation_example():
    manager = MultiEngineManager()
    manager.add_engine(EngineConfig("engine_0", "mock-model", 0))
    manager.start()

    # Submit long request
    future = await manager.submit_request(
        session_id="test",
        prompt="Generate a very long response..." * 100,
        sampling_params={"max_tokens": 1000}
    )

    # Get request ID
    requests = manager.lifecycle.get_session_requests("test")
    request_id = requests[0].request_id

    # Cancel after 0.5s
    await asyncio.sleep(0.5)
    await manager.cancel_request(request_id)

    try:
        result = await future
        print(f"Result: {result.finish_reason}")  # Should be 'cancelled'
    except asyncio.CancelledError:
        print("Request cancelled successfully")

    manager.stop()

asyncio.run(cancellation_example())
```

### 4. KV Cache Locality

Leverage cache affinity for multi-turn conversations:

```python
async def cache_locality_example():
    manager = MultiEngineManager(cache_token_threshold=2048)

    # Add engines
    for i in range(2):
        manager.add_engine(EngineConfig(f"engine_{i}", "mock-model", i))

    manager.start()

    session_id = "conversation_1"

    # First request (long prompt - establishes cache)
    long_context = "Context: " + "background info " * 1000  # >2048 tokens
    prompt1 = long_context + "Question 1: What is AI?"

    result1 = await (await manager.submit_request(
        session_id=session_id,
        prompt=prompt1,
        sampling_params={"max_tokens": 100}
    ))

    # Second request (reuses context - should route to same engine)
    prompt2 = long_context + "Question 2: What is ML?"

    result2 = await (await manager.submit_request(
        session_id=session_id,
        prompt=prompt2,
        sampling_params={"max_tokens": 100}
    ))

    # Check cache stats
    cache_stats = manager.kv_cache.get_cache_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

    # Show which engines were used
    session_engines = manager.kv_cache.get_session_engines(session_id)
    print(f"Engines used: {session_engines}")  # Should prefer one engine

    manager.stop()

asyncio.run(cache_locality_example())
```

### 5. Monitoring

```python
async def monitoring_example():
    manager = MultiEngineManager()

    # Add engines
    for i in range(3):
        manager.add_engine(EngineConfig(f"engine_{i}", "mock-model", i))

    manager.start()

    # Submit requests
    for i in range(20):
        await manager.submit_request(
            session_id=f"session_{i % 5}",
            prompt=f"Request {i}",
            sampling_params={"max_tokens": 50}
        )

    # Monitor engine status
    print("\nEngine Status:")
    for status in manager.get_all_engine_status():
        print(f"\n{status['engine_id']}:")
        print(f"  Status: {status['status']}")
        print(f"  Queue depth: {status['load']['queue_depth']}")
        print(f"  Active requests: {status['load']['active_requests']}")

    # Overall stats
    stats = manager.get_stats()
    print(f"\nOverall Stats:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Active engines: {stats['active_engines']}/{stats['total_engines']}")
    print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")

    manager.stop()

asyncio.run(monitoring_example())
```

## Running Tests

```bash
# Integration tests
cd src/rsm/tests
python test_multi_engine.py

# Demo application
cd src/rsm/examples
python multi_engine_demo.py
```

## Common Patterns

### Pattern 1: Session-Based Workflows

```python
async def workflow_pattern():
    manager = MultiEngineManager()
    # ... setup ...

    session_id = "user_workflow_123"

    # Step 1: Planning
    plan_result = await (await manager.submit_request(
        session_id=session_id,
        prompt="Create a plan for...",
        sampling_params={"max_tokens": 500}
    ))

    # Step 2: Execution (same session, may use cache)
    exec_result = await (await manager.submit_request(
        session_id=session_id,
        prompt=f"Execute this plan: {plan_result.text}",
        sampling_params={"max_tokens": 1000}
    ))

    # Step 3: Review
    review_result = await (await manager.submit_request(
        session_id=session_id,
        prompt=f"Review results: {exec_result.text}",
        sampling_params={"max_tokens": 300}
    ))

    return review_result.text
```

### Pattern 2: Batch Processing

```python
async def batch_pattern():
    manager = MultiEngineManager()
    # ... setup ...

    # Submit batch
    tasks = [
        manager.submit_request(
            session_id=f"task_{i}",
            prompt=f"Process item {i}",
            sampling_params={"max_tokens": 100}
        )
        for i in range(100)
    ]

    # Wait for all
    futures = await asyncio.gather(*tasks)
    results = await asyncio.gather(*futures)

    return results
```

### Pattern 3: With Timeout

```python
async def timeout_pattern():
    manager = MultiEngineManager()
    # ... setup ...

    future = await manager.submit_request(
        session_id="test",
        prompt="Your prompt",
        sampling_params={"max_tokens": 500}
    )

    try:
        result = await asyncio.wait_for(future, timeout=10.0)
        return result.text
    except asyncio.TimeoutError:
        # Cancel if timeout
        requests = manager.lifecycle.get_session_requests("test")
        if requests:
            await manager.cancel_request(requests[0].request_id)
        raise
```

## Configuration Tips

### For Throughput

```python
# More engines, smaller batch sizes
for i in range(8):  # Use all GPUs
    manager.add_engine(EngineConfig(
        engine_id=f"engine_{i}",
        model="your-model",
        gpu_id=i,
        max_num_seqs=64,  # Smaller batch
        max_model_len=2048  # Shorter context
    ))
```

### For Latency

```python
# Fewer engines, larger batch sizes, cache locality
for i in range(2):  # Use fewer GPUs
    manager.add_engine(EngineConfig(
        engine_id=f"engine_{i}",
        model="your-model",
        gpu_id=i,
        max_num_seqs=256,  # Larger batch
        max_model_len=8192  # Longer context
    ))

# Lower cache threshold to use affinity more often
manager = MultiEngineManager(cache_token_threshold=1024)
```

### For Long Conversations

```python
# Prioritize cache locality
manager = MultiEngineManager(
    cache_token_threshold=512  # Lower threshold
)

# Use larger context window
for i in range(4):
    manager.add_engine(EngineConfig(
        engine_id=f"engine_{i}",
        model="your-model",
        gpu_id=i,
        max_model_len=16384,  # Very long context
        max_num_seqs=32  # Smaller batch for memory
    ))
```

## Troubleshooting

### Engine Won't Start

```python
# Check if engine is alive
if not engine.is_alive():
    status = engine.get_status()
    print(f"Engine status: {status}")

# Check CUDA
import os
os.environ["CUDA_VISIBLE_DEVICES"]  # Should match gpu_id
```

### Requests Not Completing

```python
# Check engine status
for status in manager.get_all_engine_status():
    if not status['is_alive']:
        print(f"{status['engine_id']} is not alive!")
    if status['status'] == 'error':
        print(f"{status['engine_id']} has error!")

# Check request state
requests = manager.lifecycle.get_session_requests("your_session")
for req in requests:
    print(f"{req.request_id}: {req.state.value}")
```

### High Latency

```python
# Check load distribution
stats = manager.get_stats()
print(f"Cache hit rate: {stats['cache_stats']['hit_rate']}")

for status in manager.get_all_engine_status():
    load = status['load']
    print(f"{status['engine_id']}: "
          f"queue={load['queue_depth']}, "
          f"active={load['active_requests']}")
```

## Next Steps

1. **Read full documentation**: See `engine/README.md`
2. **Run tests**: `python tests/test_multi_engine.py`
3. **Try demos**: `python examples/multi_engine_demo.py`
4. **Integration guide**: See `PARTITION_4_SUMMARY.md`

## Need Help?

- Check `engine/README.md` for detailed API documentation
- See `examples/multi_engine_demo.py` for more examples
- Review `tests/test_multi_engine.py` for testing patterns
- Read `PARTITION_4_SUMMARY.md` for architecture details
