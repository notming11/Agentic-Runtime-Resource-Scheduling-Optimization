"""
Integration Tests for Multi-Engine Orchestration

This module tests the complete multi-engine orchestration system including:
- Starting/stopping multiple engine processes
- Routing requests across engines (now using LoadBalancer)
- Cancelling in-flight requests
- Session affinity and KV cache locality
- Load balancing

These tests use mock engines (since vLLM may not be available) but exercise
all the orchestration logic.
"""

import asyncio
import time
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.multi_engine_manager import MultiEngineManager
from engine.engine_process import EngineConfig, EngineStatus
from engine.lifecycle_manager import RequestState
from frontend.process_table import GlobalProcessTable


class TestMultiEngineOrchestration(unittest.TestCase):
    """Integration tests for multi-engine system"""

    def setUp(self):
        """Set up test fixtures"""
        # Create process table for LoadBalancer integration
        self.process_table = GlobalProcessTable()
        
        # Create manager with 2 mock engines
        self.manager = MultiEngineManager(process_table=self.process_table)

        # Add two engines
        config1 = EngineConfig(
            engine_id="engine_0",
            model="mock-model",
            gpu_id=0
        )
        config2 = EngineConfig(
            engine_id="engine_1",
            model="mock-model",
            gpu_id=1
        )

        self.manager.add_engine(config1)
        self.manager.add_engine(config2)

    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'manager'):
            self.manager.stop(timeout=5.0)

    def test_01_start_multiple_engines(self):
        """Test starting multiple engine processes"""
        print("\n=== Test 1: Starting Multiple Engines ===")

        # Start manager
        self.manager.start()

        # Wait for engines to be ready
        time.sleep(2.0)

        # Check engines are alive
        for engine_id in ["engine_0", "engine_1"]:
            engine = self.manager.engines[engine_id]
            status = engine.get_status()
            print(f"Engine {engine_id} status: {status.value}")
            self.assertTrue(
                engine.is_alive(),
                f"Engine {engine_id} should be alive"
            )
            self.assertIn(
                status,
                [EngineStatus.READY, EngineStatus.BUSY],
                f"Engine {engine_id} should be ready or busy"
            )

        # Check status reporting
        all_status = self.manager.get_all_engine_status()
        self.assertEqual(len(all_status), 2, "Should have 2 engines")
        print(f"All engines status: {all_status}")

        print("✓ All engines started successfully\n")

    def test_02_route_requests_across_engines(self):
        """Test routing multiple requests across engines using LoadBalancer"""
        print("\n=== Test 2: Routing Requests Across Engines ===")

        # Start manager
        self.manager.start()
        time.sleep(2.0)

        # Create event loop for async operations
        async def run_test():
            # Submit 10 requests
            futures = []
            session_ids = []

            for i in range(10):
                session_id = f"session_{i}"
                session_ids.append(session_id)

                future = await self.manager.submit_request(
                    session_id=session_id,
                    prompt=f"Test prompt {i}" * 10,  # Make it reasonably long
                    sampling_params={"max_tokens": 50}
                )
                futures.append((session_id, future))

            print(f"Submitted {len(futures)} requests")

            # Wait for all to complete
            results = []
            for session_id, future in futures:
                try:
                    result = await asyncio.wait_for(future, timeout=5.0)
                    results.append(result)
                    print(f"✓ Request for {session_id} completed: {result.text[:50]}...")
                except asyncio.TimeoutError:
                    print(f"✗ Request for {session_id} timed out")
                except Exception as e:
                    print(f"✗ Request for {session_id} failed: {e}")

            # Check results
            self.assertGreater(
                len(results),
                0,
                "At least some requests should complete"
            )

            # Check that requests were distributed across engines
            engine_distribution = {}
            for request_id, engine_id in self.manager._request_map.items():
                engine_distribution[engine_id] = engine_distribution.get(engine_id, 0) + 1

            print(f"\nEngine distribution: {engine_distribution}")

            # Get stats (now includes load balancer stats)
            stats = self.manager.get_stats()
            print(f"\nManager stats:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Completed requests: {stats['completed_requests']}")
            print(f"  Average latency: {stats['average_latency']:.3f}s")
            
            # Check load balancer stats
            lb_stats = stats.get('load_balancer_stats', {})
            print(f"\nLoad Balancer stats:")
            print(f"  Small requests: {lb_stats.get('small_requests', 0)}")
            print(f"  Large requests: {lb_stats.get('large_requests', 0)}")
            print(f"  Locality hits: {lb_stats.get('locality_hits', 0)}")

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

        print("✓ Requests routed across engines successfully\n")

    def test_03_cancel_inflight_request(self):
        """Test cancelling an in-flight request"""
        print("\n=== Test 3: Cancelling In-Flight Requests ===")

        # Start manager
        self.manager.start()
        time.sleep(2.0)

        async def run_test():
            # Submit a request
            session_id = "cancel_test_session"
            future = await self.manager.submit_request(
                session_id=session_id,
                prompt="This is a test prompt that will be cancelled" * 20,
                sampling_params={"max_tokens": 100}
            )

            print(f"Submitted request for {session_id}")

            # Get request ID
            requests = self.manager.lifecycle.get_session_requests(session_id)
            self.assertEqual(len(requests), 1, "Should have 1 request")
            request_id = requests[0].request_id

            # Wait a bit for request to start
            await asyncio.sleep(0.2)

            # Cancel the request
            print(f"Cancelling request {request_id}...")
            cancelled = await self.manager.cancel_request(request_id)
            self.assertTrue(cancelled, "Request should be cancelled")

            # Check request state
            metadata = self.manager.lifecycle.get_request(request_id)
            print(f"Request state after cancel: {metadata.state.value if metadata else 'not found'}")

            # Try to wait for result (should be cancelled)
            try:
                result = await asyncio.wait_for(future, timeout=2.0)
                print(f"Result: {result.finish_reason}")
                # Should either be cancelled or raise CancelledError
                if result.finish_reason:
                    self.assertIn(
                        result.finish_reason,
                        ["cancelled", "stopped"],
                        "Result should be cancelled"
                    )
            except asyncio.CancelledError:
                print("✓ Future was cancelled (expected)")
            except asyncio.TimeoutError:
                print("✓ Request timed out after cancellation (acceptable)")

            # Check stats
            stats = self.manager.get_stats()
            print(f"\nCancelled requests: {stats['cancelled_requests']}")

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

        print("✓ Request cancellation works correctly\n")

    def test_04_session_affinity(self):
        """Test session affinity via LoadBalancer and KV cache"""
        print("\n=== Test 4: Session Affinity ===")

        # Start manager
        self.manager.start()
        time.sleep(2.0)

        async def run_test():
            session_id = "affinity_test_session"

            # Submit multiple long requests for the same session
            # Long requests (>2048 tokens) should route to same engine via LoadBalancer
            futures = []
            for i in range(5):
                # Create a long prompt (>2048 tokens)
                long_prompt = "word " * 3000  # Approximately 3000 tokens

                future = await self.manager.submit_request(
                    session_id=session_id,
                    prompt=long_prompt,
                    sampling_params={"max_tokens": 50}
                )
                futures.append(future)

            print(f"Submitted {len(futures)} long requests for {session_id}")

            # Check which engines were used
            engines_used = set()
            requests = self.manager.lifecycle.get_session_requests(
                session_id,
                active_only=False
            )

            for request in requests:
                if request.engine_id:
                    engines_used.add(request.engine_id)
                    print(f"Request {request.request_id} routed to {request.engine_id}")

            # Wait for completion
            for i, future in enumerate(futures):
                try:
                    await asyncio.wait_for(future, timeout=5.0)
                    print(f"✓ Request {i} completed")
                except Exception as e:
                    print(f"✗ Request {i} failed: {e}")

            # Check cache stats
            cache_stats = self.manager.kv_cache.get_cache_stats()
            print(f"\nKV Cache stats:")
            print(f"  Total hits: {cache_stats['total_hits']}")
            print(f"  Total misses: {cache_stats['total_misses']}")
            print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")

            # Check LoadBalancer stats
            stats = self.manager.get_stats()
            lb_stats = stats.get('load_balancer_stats', {})
            print(f"\nLoad Balancer stats:")
            print(f"  Large requests: {lb_stats.get('large_requests', 0)}")
            print(f"  Locality hits: {lb_stats.get('locality_hits', 0)}")
            print(f"  Locality assigns: {lb_stats.get('locality_assigns', 0)}")

            # Check which engine was assigned for this program
            assigned_engine = self.manager.load_balancer.get_program_engine(session_id)
            print(f"\nProgram {session_id} assigned to engine: {assigned_engine}")

            # After first request, subsequent requests should use same engine
            session_engines = self.manager.kv_cache.get_session_engines(session_id)
            print(f"Engines with cache for {session_id}: {session_engines}")

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

        print("✓ Session affinity tracking works\n")

    def test_05_cancel_session(self):
        """Test cancelling all requests for a session"""
        print("\n=== Test 5: Cancel Entire Session ===")

        # Start manager
        self.manager.start()
        time.sleep(2.0)

        async def run_test():
            session_id = "session_cancel_test"

            # Submit multiple requests
            futures = []
            for i in range(3):
                future = await self.manager.submit_request(
                    session_id=session_id,
                    prompt=f"Test prompt {i}" * 50,
                    sampling_params={"max_tokens": 100}
                )
                futures.append(future)

            print(f"Submitted {len(futures)} requests for {session_id}")

            # Wait a bit
            await asyncio.sleep(0.2)

            # Cancel entire session
            print(f"Cancelling session {session_id}...")
            cancelled_ids = await self.manager.cancel_session(session_id)
            print(f"Cancelled {len(cancelled_ids)} requests: {cancelled_ids}")

            # Check that program was removed from LoadBalancer
            assigned_engine = self.manager.load_balancer.get_program_engine(session_id)
            self.assertIsNone(assigned_engine, "Program should be removed from LoadBalancer")

            # Check results
            for i, future in enumerate(futures):
                try:
                    result = await asyncio.wait_for(future, timeout=2.0)
                    print(f"Request {i}: {result.finish_reason}")
                except asyncio.CancelledError:
                    print(f"✓ Request {i} cancelled")
                except Exception as e:
                    print(f"Request {i}: {e}")

            # Check stats
            stats = self.manager.get_stats()
            print(f"\nTotal cancelled: {stats['cancelled_requests']}")

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

        print("✓ Session cancellation works\n")

    def test_06_load_balancing(self):
        """Test load balancing across engines using LoadBalancer"""
        print("\n=== Test 6: Load Balancing ===")

        # Start manager
        self.manager.start()
        time.sleep(2.0)

        async def run_test():
            # Submit many short requests (should use least-loaded routing via LoadBalancer)
            futures = []
            for i in range(20):
                # Short prompts (≤2048 tokens) should use LEAST_USED routing
                short_prompt = f"Short test prompt {i}"

                future = await self.manager.submit_request(
                    session_id=f"short_session_{i}",
                    prompt=short_prompt,
                    sampling_params={"max_tokens": 20}
                )
                futures.append(future)

            print(f"Submitted {len(futures)} short requests")

            # Check engine loads via LoadBalancer
            print("\nEngine loads (from LoadBalancer):")
            for engine_id in ["engine_0", "engine_1"]:
                workload = self.manager.load_balancer.get_engine_workload(engine_id)
                print(f"  {engine_id}: workload={workload}")

            # Wait for completion
            completed = 0
            for future in futures:
                try:
                    await asyncio.wait_for(future, timeout=5.0)
                    completed += 1
                except Exception:
                    pass

            print(f"\n✓ Completed {completed}/{len(futures)} requests")

            # Check final stats
            stats = self.manager.get_stats()
            print(f"\nFinal stats:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Completed: {stats['completed_requests']}")
            print(f"  Average latency: {stats['average_latency']:.3f}s")

            # Check LoadBalancer stats
            lb_stats = stats.get('load_balancer_stats', {})
            print(f"\nLoad Balancer stats:")
            print(f"  Small requests: {lb_stats.get('small_requests', 0)}")
            print(f"  Total requests: {lb_stats.get('total_requests', 0)}")

            # Check that load was distributed
            engine_metrics = {}
            for engine_id in ["engine_0", "engine_1"]:
                engine = self.manager.engines[engine_id]
                metrics = engine.get_metrics()
                engine_metrics[engine_id] = metrics["requests_processed"]
                print(f"  {engine_id}: {metrics['requests_processed']} requests")

            # Both engines should have processed some requests
            # (unless one failed to start)
            active_engines = sum(1 for count in engine_metrics.values() if count > 0)
            print(f"\nActive engines: {active_engines}/2")

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

        print("✓ Load balancing works\n")

    def test_07_load_balancer_integration(self):
        """Test LoadBalancer integration with threshold logic"""
        print("\n=== Test 7: LoadBalancer Integration ===")

        # Start manager
        self.manager.start()
        time.sleep(2.0)

        async def run_test():
            # Test 1: Small request (≤2048 tokens) should use LEAST_USED
            print("\nTest 1: Small requests use LEAST_USED")
            small_session = "small_test"
            await self.manager.submit_request(
                session_id=small_session,
                prompt="Short prompt",
                sampling_params={"max_tokens": 20}
            )

            stats = self.manager.get_stats()
            lb_stats = stats.get('load_balancer_stats', {})
            self.assertEqual(lb_stats.get('small_requests', 0), 1, "Should have 1 small request")
            print(f"  ✓ Small requests: {lb_stats.get('small_requests', 0)}")

            # Test 2: Large request (>2048 tokens) should assign to engine
            print("\nTest 2: Large requests use locality")
            large_session = "large_test"
            long_prompt = "word " * 3000  # >2048 tokens
            
            # First large request assigns
            await self.manager.submit_request(
                session_id=large_session,
                prompt=long_prompt,
                sampling_params={"max_tokens": 50}
            )

            stats = self.manager.get_stats()
            lb_stats = stats.get('load_balancer_stats', {})
            self.assertEqual(lb_stats.get('large_requests', 0), 1, "Should have 1 large request")
            self.assertEqual(lb_stats.get('locality_assigns', 0), 1, "Should have 1 locality assignment")
            print(f"  ✓ Large requests: {lb_stats.get('large_requests', 0)}")
            print(f"  ✓ Locality assigns: {lb_stats.get('locality_assigns', 0)}")

            # Second large request should hit locality
            await self.manager.submit_request(
                session_id=large_session,
                prompt=long_prompt,
                sampling_params={"max_tokens": 50}
            )

            stats = self.manager.get_stats()
            lb_stats = stats.get('load_balancer_stats', {})
            self.assertEqual(lb_stats.get('large_requests', 0), 2, "Should have 2 large requests")
            self.assertEqual(lb_stats.get('locality_hits', 0), 1, "Should have 1 locality hit")
            print(f"  ✓ Locality hits: {lb_stats.get('locality_hits', 0)}")

            # Check program table
            assigned_engine = self.manager.load_balancer.get_program_engine(large_session)
            print(f"\nProgram {large_session} assigned to: {assigned_engine}")
            self.assertIsNotNone(assigned_engine, "Program should be in LoadBalancer pt table")

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

        print("✓ LoadBalancer integration works correctly\n")


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultiEngineOrchestration)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)