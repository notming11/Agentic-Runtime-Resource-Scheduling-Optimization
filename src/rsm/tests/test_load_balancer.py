"""
Tests for Autellix Load Balancer (Algorithm 2).
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_balancer import LoadBalancer, RequestSize


class MockProcessTable:
    """Mock process table for testing."""
    def __init__(self):
        self.programs = {}
    
    def create_program(self, pid):
        self.programs[pid] = {"pid": pid}
    
    def get_program(self, pid):
        return self.programs.get(pid)


class TestRequestClassification(unittest.TestCase):
    """Test request size classification."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
    
    def test_small_request_threshold(self):
        """Test that requests <= 2048 tokens are small."""
        # Exactly 2048 should be small
        self.lb.register_engine("engine_0")
        engine = self.lb.route_request("prog1", num_tokens=2048)
        stats = self.lb.get_stats()
        self.assertEqual(stats['small_requests'], 1)
        self.assertEqual(stats['large_requests'], 0)
    
    def test_large_request_threshold(self):
        """Test that requests > 2048 tokens are large."""
        self.lb.register_engine("engine_0")
        engine = self.lb.route_request("prog1", num_tokens=2049)
        stats = self.lb.get_stats()
        self.assertEqual(stats['small_requests'], 0)
        self.assertEqual(stats['large_requests'], 1)
    
    def test_boundary_conditions(self):
        """Test exact boundary at 2048 tokens."""
        self.lb.register_engine("engine_0")
        
        # 2048 should be SMALL
        self.lb.route_request("prog1", num_tokens=2048)
        
        # 2049 should be LARGE
        self.lb.route_request("prog2", num_tokens=2049)
        
        stats = self.lb.get_stats()
        self.assertEqual(stats['small_requests'], 1)
        self.assertEqual(stats['large_requests'], 1)


class TestSmallRequestLoadBalancing(unittest.TestCase):
    """Test that small requests use load balancing (LEAST_USED)."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
        
        # Register 3 engines
        self.lb.register_engine("engine_0")
        self.lb.register_engine("engine_1")
        self.lb.register_engine("engine_2")
    
    def test_small_requests_use_least_used(self):
        """Test that small requests go to least used engine."""
        pid = "program_1"
        
        # Make engine_0 heavily loaded
        self.lb._engines["engine_0"].active_requests = 10
        
        # Make engine_1 lightly loaded
        self.lb._engines["engine_1"].active_requests = 1
        
        # Make engine_2 moderately loaded
        self.lb._engines["engine_2"].active_requests = 5
        
        # Small request should go to engine_1 (least used)
        engine = self.lb.route_request(pid, num_tokens=1000)
        self.assertEqual(engine, "engine_1")
    
    def test_small_requests_distribute_load(self):
        """Test that consecutive small requests distribute across engines."""
        # All engines start with 0 load
        
        # Send 6 small requests
        engines_used = []
        for i in range(6):
            engine = self.lb.route_request(f"program_{i}", num_tokens=500)
            engines_used.append(engine)
            # Don't complete requests, so workload accumulates
        
        # Should distribute across multiple engines
        unique_engines = set(engines_used)
        self.assertGreater(len(unique_engines), 1, 
                          "Small requests should distribute across engines")


class TestLargeRequestLocality(unittest.TestCase):
    """Test that large requests use locality (pt table)."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
        
        # Register engines
        self.lb.register_engine("engine_0")
        self.lb.register_engine("engine_1")
        self.lb.register_engine("engine_2")
    
    def test_large_requests_build_locality(self):
        """Test that large requests from same program go to same engine."""
        pid = "program_1"
        
        # First large request assigns to some engine
        engine1 = self.lb.route_request(pid, num_tokens=3000)
        self.assertIsNotNone(engine1)
        
        # Check that program is now in pt table
        self.assertEqual(self.lb.get_program_engine(pid), engine1)
        
        # Subsequent large requests should go to same engine
        for _ in range(5):
            engine = self.lb.route_request(pid, num_tokens=3500)
            self.assertEqual(engine, engine1, 
                           "Large requests should use locality")
    
    def test_large_request_assigns_to_least_used(self):
        """Test that first large request for program uses LEAST_USED."""
        pid = "program_1"
        
        # Make engine_1 least loaded
        self.lb._engines["engine_0"].active_requests = 10
        self.lb._engines["engine_1"].active_requests = 1
        self.lb._engines["engine_2"].active_requests = 5
        
        # First large request should go to engine_1
        engine = self.lb.route_request(pid, num_tokens=3000)
        self.assertEqual(engine, "engine_1")
        
        # Program should now be assigned to engine_1 in pt table
        self.assertEqual(self.lb.get_program_engine(pid), "engine_1")
    
    def test_different_programs_different_engines(self):
        """Test that different programs can be assigned to different engines."""
        pid1 = "program_1"
        pid2 = "program_2"
        
        # Route large requests for different programs
        engine1 = self.lb.route_request(pid1, num_tokens=3000)
        engine2 = self.lb.route_request(pid2, num_tokens=3000)
        
        # Both should get engines
        self.assertIsNotNone(engine1)
        self.assertIsNotNone(engine2)
        
        # Programs should remember their engines
        self.assertEqual(self.lb.get_program_engine(pid1), engine1)
        self.assertEqual(self.lb.get_program_engine(pid2), engine2)
    
    def test_locality_hit_tracking(self):
        """Test that locality hits are tracked correctly."""
        pid = "program_1"
        
        # First large request - creates assignment
        self.lb.route_request(pid, num_tokens=3000)
        stats = self.lb.get_stats()
        self.assertEqual(stats['locality_assigns'], 1)
        self.assertEqual(stats['locality_hits'], 0)
        
        # Next 5 large requests should hit locality
        for _ in range(5):
            self.lb.route_request(pid, num_tokens=3500)
        
        stats = self.lb.get_stats()
        self.assertEqual(stats['locality_assigns'], 1)
        self.assertEqual(stats['locality_hits'], 5)


class TestMixedRequestPatterns(unittest.TestCase):
    """Test mixed small and large requests."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
        self.lb.register_engine("engine_0")
        self.lb.register_engine("engine_1")
    
    def test_small_then_large_requests(self):
        """Test small requests followed by large requests from same program."""
        pid = "program_1"
        
        # Send small requests (should load balance)
        small_engines = set()
        for _ in range(3):
            engine = self.lb.route_request(pid, num_tokens=1000)
            small_engines.add(engine)
        
        # Program should NOT be in pt table yet (only large requests add to pt)
        # Note: small requests don't affect pt table
        
        # Send large request (should assign to engine)
        large_engine = self.lb.route_request(pid, num_tokens=3000)
        
        # Now program should be in pt table
        self.assertEqual(self.lb.get_program_engine(pid), large_engine)
        
        # Next large request should use same engine
        engine = self.lb.route_request(pid, num_tokens=3500)
        self.assertEqual(engine, large_engine)
    
    def test_large_then_small_requests(self):
        """Test large requests followed by small requests from same program."""
        pid = "program_1"
        
        # Send large request (assigns to engine)
        large_engine = self.lb.route_request(pid, num_tokens=3000)
        
        # Make another engine less loaded
        self.lb._engines[large_engine].active_requests = 10
        other_engine = "engine_0" if large_engine == "engine_1" else "engine_1"
        self.lb._engines[other_engine].active_requests = 1
        
        # Small request should ignore locality and use least loaded
        small_engine = self.lb.route_request(pid, num_tokens=1000)
        self.assertEqual(small_engine, other_engine,
                        "Small requests should ignore program's pt assignment")


class TestEngineManagement(unittest.TestCase):
    """Test engine registration and management."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
    
    def test_register_engine(self):
        """Test engine registration."""
        self.lb.register_engine("engine_0")
        self.assertIn("engine_0", self.lb._engines)
    
    def test_unregister_engine(self):
        """Test engine removal."""
        self.lb.register_engine("engine_0")
        self.lb.unregister_engine("engine_0")
        self.assertNotIn("engine_0", self.lb._engines)
    
    def test_unregister_engine_clears_program_assignments(self):
        """Test that removing engine clears program assignments."""
        self.lb.register_engine("engine_0")
        self.lb.register_engine("engine_1")
        
        pid = "program_1"
        
        # Assign program to engine_0 via large request
        self.lb.route_request(pid, num_tokens=3000)
        assigned_engine = self.lb.get_program_engine(pid)
        
        # Remove that engine
        self.lb.unregister_engine(assigned_engine)
        
        # Program should no longer be assigned
        self.assertIsNone(self.lb.get_program_engine(pid))


class TestRequestCompletion(unittest.TestCase):
    """Test request completion tracking."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
        self.lb.register_engine("engine_0")
    
    def test_complete_request_decrements_workload(self):
        """Test that completing a request reduces engine workload."""
        pid = "program_1"
        
        # Route a request
        engine = self.lb.route_request(pid, num_tokens=1000)
        initial_workload = self.lb.get_engine_workload(engine)
        
        # Complete the request
        self.lb.complete_request(engine, pid)
        final_workload = self.lb.get_engine_workload(engine)
        
        self.assertEqual(final_workload, initial_workload - 1)
    
    def test_workload_affects_least_used_selection(self):
        """Test that workload affects LEAST_USED selection."""
        self.lb.register_engine("engine_1")
        
        # Make engine_0 more loaded
        self.lb._engines["engine_0"].active_requests = 5
        self.lb._engines["engine_1"].active_requests = 2
        
        # Small request should go to engine_1
        engine = self.lb.route_request("prog1", num_tokens=1000)
        self.assertEqual(engine, "engine_1")


class TestProgramRemoval(unittest.TestCase):
    """Test program removal and cleanup."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
        self.lb.register_engine("engine_0")
    
    def test_remove_program_clears_pt_entry(self):
        """Test that removing a program clears its pt table entry."""
        pid = "program_1"
        
        # Assign program via large request
        self.lb.route_request(pid, num_tokens=3000)
        self.assertIsNotNone(self.lb.get_program_engine(pid))
        
        # Remove program
        self.lb.remove_program(pid)
        self.assertIsNone(self.lb.get_program_engine(pid))
    
    def test_remove_program_clears_engine_tracking(self):
        """Test that removing a program clears engine's program set."""
        pid = "program_1"
        
        # Route request
        engine = self.lb.route_request(pid, num_tokens=3000)
        self.assertIn(pid, self.lb._engines[engine].programs_assigned)
        
        # Remove program
        self.lb.remove_program(pid)
        self.assertNotIn(pid, self.lb._engines[engine].programs_assigned)


class TestStatistics(unittest.TestCase):
    """Test statistics tracking."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
        self.lb.register_engine("engine_0")
        self.lb.register_engine("engine_1")
    
    def test_request_counting(self):
        """Test that requests are counted correctly."""
        # Route 3 small and 2 large requests
        for i in range(3):
            self.lb.route_request(f"prog{i}", num_tokens=500)
        
        for i in range(2):
            self.lb.route_request(f"prog{i+3}", num_tokens=3000)
        
        stats = self.lb.get_stats()
        self.assertEqual(stats['total_requests'], 5)
        self.assertEqual(stats['small_requests'], 3)
        self.assertEqual(stats['large_requests'], 2)
    
    def test_program_table_size(self):
        """Test that pt table size is tracked."""
        # Route large requests for 3 different programs
        for i in range(3):
            self.lb.route_request(f"program_{i}", num_tokens=3000)
        
        stats = self.lb.get_stats()
        self.assertEqual(stats['programs_in_table'], 3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        self.process_table = MockProcessTable()
        self.lb = LoadBalancer(self.process_table)
    
    def test_route_with_no_engines(self):
        """Test routing when no engines are registered."""
        result = self.lb.route_request("program_1", num_tokens=1000)
        self.assertIsNone(result)
    
    def test_zero_tokens(self):
        """Test with zero tokens (should be small)."""
        self.lb.register_engine("engine_0")
        self.lb.route_request("prog1", num_tokens=0)
        
        stats = self.lb.get_stats()
        self.assertEqual(stats['small_requests'], 1)
    
    def test_engine_removed_while_program_assigned(self):
        """Test handling when assigned engine is removed."""
        self.lb.register_engine("engine_0")
        self.lb.register_engine("engine_1")
        
        pid = "program_1"
        
        # Assign program to engine_0
        engine = self.lb.route_request(pid, num_tokens=3000)
        self.assertEqual(engine, "engine_0")
        
        # Remove engine_0
        self.lb.unregister_engine("engine_0")
        
        # Next large request should reassign to engine_1
        engine = self.lb.route_request(pid, num_tokens=3500)
        self.assertEqual(engine, "engine_1")
        self.assertEqual(self.lb.get_program_engine(pid), "engine_1")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)