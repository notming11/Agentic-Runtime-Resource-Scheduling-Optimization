"""
Multi-Engine Orchestration Demo

This example demonstrates how to use the Multi-Engine Manager to:
1. Set up multiple vLLM engines
2. Submit requests with load balancing
3. Leverage KV cache locality for long requests
4. Cancel requests and sessions
5. Monitor engine status and performance

This demo uses mock engines (works without vLLM installed) but can easily
be adapted to use real vLLM engines by changing the model names.
"""

import asyncio
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.multi_engine_manager import MultiEngineManager
from engine.engine_process import EngineConfig


async def demo_basic_usage():
    """Demonstrate basic multi-engine usage"""
    print("="*70)
    print("DEMO 1: Basic Multi-Engine Usage")
    print("="*70)

    # Create manager
    manager = MultiEngineManager()

    # Add two engines (using mock model for demo)
    print("\n1. Adding engines...")
    configs = [
        EngineConfig(
            engine_id="engine_A",
            model="mock-7b",  # Use real model path for production
            gpu_id=0,
            max_num_seqs=128
        ),
        EngineConfig(
            engine_id="engine_B",
            model="mock-7b",
            gpu_id=1,
            max_num_seqs=128
        )
    ]

    for config in configs:
        manager.add_engine(config)
        print(f"   Added {config.engine_id} on GPU {config.gpu_id}")

    # Start manager
    print("\n2. Starting engines...")
    manager.start()
    print("   Waiting for engines to initialize...")
    time.sleep(2.0)

    # Check status
    print("\n3. Engine status:")
    for status in manager.get_all_engine_status():
        print(f"   {status['engine_id']}: {status['status']} "
              f"(alive: {status['is_alive']})")

    # Submit requests
    print("\n4. Submitting requests...")
    futures = []
    for i in range(5):
        future = await manager.submit_request(
            session_id=f"session_{i}",
            prompt=f"Explain quantum computing in simple terms. Request {i}.",
            sampling_params={
                "temperature": 0.7,
                "max_tokens": 100
            }
        )
        futures.append((f"session_{i}", future))
        print(f"   ✓ Submitted request for session_{i}")

    # Wait for results
    print("\n5. Waiting for results...")
    for session_id, future in futures:
        try:
            result = await asyncio.wait_for(future, timeout=10.0)
            print(f"   ✓ {session_id}: {result.text[:60]}...")
            print(f"      Latency: {result.latency:.3f}s, "
                  f"Tokens: {result.completion_tokens}")
        except Exception as e:
            print(f"   ✗ {session_id}: {e}")

    # Show statistics
    print("\n6. Statistics:")
    stats = manager.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Completed: {stats['completed_requests']}")
    print(f"   Average latency: {stats['average_latency']:.3f}s")

    # Cleanup
    print("\n7. Shutting down...")
    manager.stop()
    print("   ✓ All engines stopped")


async def demo_cache_locality():
    """Demonstrate KV cache locality for long requests"""
    print("\n" + "="*70)
    print("DEMO 2: KV Cache Locality")
    print("="*70)

    manager = MultiEngineManager(cache_token_threshold=2048)

    # Add engines
    print("\n1. Setting up engines...")
    for i in range(2):
        config = EngineConfig(
            engine_id=f"engine_{i}",
            model="mock-7b",
            gpu_id=i
        )
        manager.add_engine(config)

    manager.start()
    time.sleep(2.0)

    # Submit short requests (should use least-loaded routing)
    print("\n2. Submitting SHORT requests (≤2048 tokens)...")
    print("   These should use LEAST-LOADED routing")
    session_id = "short_session"

    for i in range(3):
        short_prompt = f"Short prompt number {i}"
        await manager.submit_request(
            session_id=session_id,
            prompt=short_prompt,
            sampling_params={"max_tokens": 50}
        )
        print(f"   ✓ Short request {i} submitted")

    # Give requests time to route
    await asyncio.sleep(0.5)

    # Check routing
    requests = manager.lifecycle.get_session_requests(session_id, active_only=False)
    engines_used_short = set(r.engine_id for r in requests if r.engine_id)
    print(f"   Engines used for short requests: {engines_used_short}")

    # Submit long requests (should use cache locality)
    print("\n3. Submitting LONG requests (>2048 tokens)...")
    print("   These should use CACHE AFFINITY routing")
    session_id_long = "long_session"

    long_prompt_base = "word " * 3000  # ~3000 tokens

    for i in range(3):
        long_prompt = long_prompt_base + f" Request {i}"
        await manager.submit_request(
            session_id=session_id_long,
            prompt=long_prompt,
            sampling_params={"max_tokens": 50}
        )
        print(f"   ✓ Long request {i} submitted")

    # Give requests time to route
    await asyncio.sleep(0.5)

    # Check routing - long requests should prefer same engine
    requests_long = manager.lifecycle.get_session_requests(
        session_id_long,
        active_only=False
    )
    engines_used_long = [r.engine_id for r in requests_long if r.engine_id]
    print(f"   Engines used for long requests: {engines_used_long}")

    # Check cache statistics
    print("\n4. KV Cache Statistics:")
    cache_stats = manager.kv_cache.get_cache_stats()
    print(f"   Total requests: {cache_stats['total_requests']}")
    print(f"   Cache hits: {cache_stats['total_hits']}")
    print(f"   Cache misses: {cache_stats['total_misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")

    # Show which engines have cache for the long session
    session_engines = manager.kv_cache.get_session_engines(session_id_long)
    print(f"\n   Session '{session_id_long}' has cache on: {session_engines}")

    # Cleanup
    manager.stop()


async def demo_cancellation():
    """Demonstrate request and session cancellation"""
    print("\n" + "="*70)
    print("DEMO 3: Request Cancellation")
    print("="*70)

    manager = MultiEngineManager()

    # Setup
    print("\n1. Setting up engines...")
    for i in range(2):
        manager.add_engine(EngineConfig(
            engine_id=f"engine_{i}",
            model="mock-7b",
            gpu_id=i
        ))
    manager.start()
    time.sleep(2.0)

    # Submit a long-running request
    print("\n2. Submitting long-running request...")
    session_id = "cancel_demo"
    future = await manager.submit_request(
        session_id=session_id,
        prompt="Generate a very long response..." * 100,
        sampling_params={"max_tokens": 500}
    )
    print("   ✓ Request submitted")

    # Get request ID
    requests = manager.lifecycle.get_session_requests(session_id)
    request_id = requests[0].request_id
    print(f"   Request ID: {request_id}")

    # Wait a bit, then cancel
    print("\n3. Waiting 0.5s, then cancelling...")
    await asyncio.sleep(0.5)

    cancelled = await manager.cancel_request(request_id)
    print(f"   Cancellation sent: {cancelled}")

    # Try to get result
    try:
        result = await asyncio.wait_for(future, timeout=2.0)
        print(f"   Result: {result.finish_reason}")
    except asyncio.CancelledError:
        print("   ✓ Request was cancelled")
    except asyncio.TimeoutError:
        print("   ✓ Request timed out (cancelled)")

    # Demo session cancellation
    print("\n4. Demo: Cancel entire session")
    session_id = "multi_cancel"

    # Submit multiple requests
    print(f"   Submitting 3 requests for {session_id}...")
    futures = []
    for i in range(3):
        f = await manager.submit_request(
            session_id=session_id,
            prompt=f"Request {i}" * 50,
            sampling_params={"max_tokens": 100}
        )
        futures.append(f)

    # Cancel whole session
    print(f"   Cancelling session {session_id}...")
    cancelled_ids = await manager.cancel_session(session_id)
    print(f"   Cancelled {len(cancelled_ids)} requests")

    # Check results
    for i, f in enumerate(futures):
        try:
            result = await asyncio.wait_for(f, timeout=1.0)
            print(f"   Request {i}: {result.finish_reason}")
        except asyncio.CancelledError:
            print(f"   ✓ Request {i} cancelled")
        except Exception as e:
            print(f"   Request {i}: {type(e).__name__}")

    # Cleanup
    manager.stop()


async def demo_monitoring():
    """Demonstrate monitoring and statistics"""
    print("\n" + "="*70)
    print("DEMO 4: Monitoring and Statistics")
    print("="*70)

    manager = MultiEngineManager()

    # Setup
    print("\n1. Setting up engines...")
    for i in range(3):  # Use 3 engines
        manager.add_engine(EngineConfig(
            engine_id=f"engine_{i}",
            model="mock-7b",
            gpu_id=i
        ))
    manager.start()
    time.sleep(2.0)

    # Submit many requests
    print("\n2. Submitting 20 requests across engines...")
    futures = []
    for i in range(20):
        f = await manager.submit_request(
            session_id=f"session_{i % 5}",  # 5 different sessions
            prompt=f"Request {i}: Tell me about AI" * 10,
            sampling_params={"max_tokens": 50}
        )
        futures.append(f)

    # Monitor engine status
    print("\n3. Engine Status:")
    for status in manager.get_all_engine_status():
        print(f"   {status['engine_id']}:")
        print(f"      Status: {status['status']}")
        print(f"      Load: {status['load']}")
        print(f"      Metrics: {status['metrics']}")

    # Wait for some completions
    print("\n4. Waiting for completions...")
    await asyncio.sleep(2.0)

    # Show detailed statistics
    print("\n5. Detailed Statistics:")
    stats = manager.get_stats()

    print(f"\n   Overall:")
    print(f"      Total requests: {stats['total_requests']}")
    print(f"      Completed: {stats['completed_requests']}")
    print(f"      Failed: {stats['failed_requests']}")
    print(f"      Cancelled: {stats['cancelled_requests']}")
    print(f"      Average latency: {stats['average_latency']:.3f}s")

    print(f"\n   Lifecycle:")
    lc_stats = stats['lifecycle_stats']
    print(f"      Active requests: {lc_stats['active_requests']}")
    print(f"      Pending: {lc_stats['pending_requests']}")
    print(f"      Running: {lc_stats['running_requests']}")

    print(f"\n   Cache:")
    cache_stats = stats['cache_stats']
    print(f"      Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"      Total sessions: {cache_stats['total_sessions']}")

    # Show per-engine metrics
    print(f"\n   Per-Engine Metrics:")
    for engine_id, engine in manager.engines.items():
        metrics = engine.get_metrics()
        print(f"      {engine_id}:")
        print(f"         Processed: {metrics['requests_processed']}")
        print(f"         Failed: {metrics['requests_failed']}")
        print(f"         Avg latency: {metrics['average_latency']:.3f}s")

    # Wait for all to finish
    for f in futures:
        try:
            await asyncio.wait_for(f, timeout=5.0)
        except:
            pass

    # Final stats
    print("\n6. Final Statistics:")
    final_stats = manager.get_stats()
    print(f"   Completion rate: {final_stats['completed_requests']}/{final_stats['total_requests']} "
          f"({100*final_stats['completed_requests']/max(final_stats['total_requests'],1):.1f}%)")

    # Cleanup
    manager.stop()


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("MULTI-ENGINE ORCHESTRATION DEMONSTRATION")
    print("Autellix RSM - Partition 4 Implementation")
    print("="*70)

    # Run each demo
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("KV Cache Locality", demo_cache_locality),
        ("Cancellation", demo_cancellation),
        ("Monitoring", demo_monitoring),
    ]

    for name, demo_func in demos:
        print(f"\n\nStarting: {name}")
        print("-" * 70)
        try:
            asyncio.run(demo_func())
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
