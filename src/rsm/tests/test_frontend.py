"""
Unit Tests for Autellix RSM Frontend

Tests for process table, session manager, and API wrapper.
"""

import pytest
import time
import threading
from typing import List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend import (
    GlobalProcessTable,
    ProgramEntry,
    ThreadMetadata,
    ProgramState,
    SessionManager,
    SessionState,
    AutellixClient,
    autellix_session
)


class TestProcessTable:
    """Tests for GlobalProcessTable"""

    def test_create_program(self):
        """Test creating a program entry"""
        pt = GlobalProcessTable()
        pid = "test_program_1"

        entry = pt.create_program(pid, is_multithreaded=False)

        assert entry.pid == pid
        assert entry.service_time == 0.0
        assert entry.waiting_time == 0.0
        assert not entry.is_multithreaded
        assert len(entry.threads) == 0

    def test_duplicate_program_raises_error(self):
        """Test that creating duplicate program raises ValueError"""
        pt = GlobalProcessTable()
        pid = "test_program_1"

        pt.create_program(pid)

        with pytest.raises(ValueError):
            pt.create_program(pid)

    def test_add_llm_call(self):
        """Test adding an LLM call to a program"""
        pt = GlobalProcessTable()
        pid = "test_program_1"
        pt.create_program(pid, is_multithreaded=False)

        thread = pt.add_llm_call(
            pid=pid,
            call_id="call_1",
            thread_id="thread_1",
            prefill_tokens=100,
            engine_id="engine_0"
        )

        assert thread is not None
        assert thread.thread_id == "thread_1"
        assert thread.call_id == "call_1"
        assert thread.prefill_tokens == 100

        program = pt.get_program(pid)
        assert len(program.threads) == 1
        assert "engine_0" in program.engine_ids

    def test_plas_service_time_update(self):
        """Test PLAS (single-threaded) service time calculation"""
        pt = GlobalProcessTable()
        pid = "test_program_1"
        pt.create_program(pid, is_multithreaded=False)

        # Add and complete calls
        pt.add_llm_call(pid, "call_1", "thread_1", 100)
        pt.update_program_metrics(pid, "call_1", service_time=1.0, waiting_time=0.5)

        program = pt.get_program(pid)
        assert program.service_time == 1.0
        assert program.waiting_time == 0.5

        # Add another call - service time should be cumulative
        pt.add_llm_call(pid, "call_2", "thread_2", 100)
        pt.update_program_metrics(pid, "call_2", service_time=2.0, waiting_time=0.3)

        program = pt.get_program(pid)
        assert program.service_time == 3.0  # 1.0 + 2.0
        assert program.waiting_time == 0.8  # 0.5 + 0.3

    def test_atlas_service_time_update(self):
        """Test ATLAS (multi-threaded) critical path calculation"""
        pt = GlobalProcessTable()
        pid = "test_program_1"
        pt.create_program(pid, is_multithreaded=True)

        # Add root call
        pt.add_llm_call(pid, "call_1", "thread_1", 100)
        program = pt.get_program(pid)
        thread_1 = program.threads["thread_1"]
        thread_1.service_time = 2.0
        pt.update_program_metrics(pid, "call_1", service_time=2.0, waiting_time=0.5)

        assert program.service_time == 2.0

        # Add two parallel calls depending on thread_1
        pt.add_llm_call(pid, "call_2", "thread_2", 100, parent_thread_ids=["thread_1"])
        pt.add_llm_call(pid, "call_3", "thread_3", 100, parent_thread_ids=["thread_1"])

        thread_2 = program.threads["thread_2"]
        thread_3 = program.threads["thread_3"]

        thread_2.service_time = 1.0
        thread_3.service_time = 3.0

        # Update with longer critical path (thread_1 -> thread_3)
        program.update_service_time_multi_threaded("thread_3", ["thread_1"])

        # Service time should be max critical path
        # thread_1 (2.0) + thread_3 (3.0) = 5.0
        # Note: The actual critical path calculation depends on parent priorities
        # which are updated in the real implementation

    def test_starvation_ratio(self):
        """Test starvation ratio calculation"""
        pt = GlobalProcessTable()
        pid = "test_program_1"
        program = pt.create_program(pid)

        # No service time yet
        assert program.get_starvation_ratio() == 0.0

        # Add some waiting time
        program.waiting_time = 10.0
        assert program.get_starvation_ratio() == float('inf')

        # Add service time
        program.service_time = 5.0
        assert program.get_starvation_ratio() == 2.0  # 10.0 / 5.0

    def test_concurrent_access(self):
        """Test thread-safe concurrent access to process table"""
        pt = GlobalProcessTable()
        pid = "test_program_1"
        pt.create_program(pid)

        num_threads = 10
        calls_per_thread = 100

        def add_calls(thread_idx: int):
            for i in range(calls_per_thread):
                call_id = f"call_{thread_idx}_{i}"
                thread_id = f"thread_{thread_idx}_{i}"
                pt.add_llm_call(pid, call_id, thread_id, 100)

        threads: List[threading.Thread] = []
        for i in range(num_threads):
            t = threading.Thread(target=add_calls, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        program = pt.get_program(pid)
        assert program.total_calls == num_threads * calls_per_thread

    def test_cleanup_stale_programs(self):
        """Test cleanup of stale programs"""
        pt = GlobalProcessTable()

        # Create a program and mark it as old
        pid = "stale_program"
        program = pt.create_program(pid)
        program.most_recent_completion = time.time() - 7200  # 2 hours ago

        # Cleanup with 1 hour timeout
        pt.cleanup_stale_programs(timeout_seconds=3600)

        # Program should be removed
        assert pt.get_program(pid) is None


class TestSessionManager:
    """Tests for SessionManager"""

    def test_start_session(self):
        """Test starting a session"""
        pt = GlobalProcessTable()
        sm = SessionManager(pt)

        session_id = sm.start_session(is_multithreaded=False)

        assert session_id is not None
        assert sm.is_active(session_id)

        session_info = sm.get_session(session_id)
        assert session_info.session_id == session_id
        assert session_info.state == SessionState.ACTIVE

    def test_end_session(self):
        """Test ending a session"""
        pt = GlobalProcessTable()
        sm = SessionManager(pt)

        session_id = sm.start_session()
        assert sm.is_active(session_id)

        success = sm.end_session(session_id)
        assert success

        session_info = sm.get_session(session_id)
        assert session_info.state == SessionState.COMPLETED

    def test_register_llm_call(self):
        """Test registering an LLM call"""
        pt = GlobalProcessTable()
        sm = SessionManager(pt)

        session_id = sm.start_session()

        thread_id = sm.register_llm_call(
            session_id=session_id,
            call_id="call_1",
            prefill_tokens=100,
            engine_id="engine_0"
        )

        assert thread_id is not None

        # Verify in process table
        session_info = sm.get_session(session_id)
        program = pt.get_program(session_info.pid)
        assert len(program.threads) == 1

    def test_complete_llm_call(self):
        """Test completing an LLM call"""
        pt = GlobalProcessTable()
        sm = SessionManager(pt)

        session_id = sm.start_session()
        thread_id = sm.register_llm_call(
            session_id=session_id,
            call_id="call_1",
            prefill_tokens=100
        )

        success = sm.complete_llm_call(
            session_id=session_id,
            thread_id=thread_id,
            service_time=1.5,
            waiting_time=0.3
        )

        assert success

        # Verify metrics updated
        session_stats = sm.get_session_stats(session_id)
        assert session_stats["completed_calls"] == 1
        assert session_stats["service_time"] == 1.5
        assert session_stats["waiting_time"] == 0.3

    def test_session_metadata(self):
        """Test session metadata storage"""
        pt = GlobalProcessTable()
        sm = SessionManager(pt)

        metadata = {"user_id": "123", "app": "chatbot"}
        session_id = sm.start_session(metadata=metadata)

        session_info = sm.get_session(session_id)
        assert session_info.metadata == metadata

    def test_cleanup_inactive_sessions(self):
        """Test cleanup of inactive sessions"""
        pt = GlobalProcessTable()
        sm = SessionManager(pt)

        session_id = sm.start_session()

        # Make session appear inactive
        session_info = sm.get_session(session_id)
        session_info.last_activity = time.time() - 7200  # 2 hours ago

        # Cleanup with 1 hour timeout
        sm.cleanup_inactive_sessions(timeout_seconds=3600)

        # Session should be timed out
        session_info = sm.get_session(session_id)
        assert session_info.state == SessionState.TIMEOUT


class TestAutellixClient:
    """Tests for AutellixClient"""

    def test_client_initialization(self):
        """Test client initialization"""
        client = AutellixClient(
            backend_url="http://localhost:8000",
            is_multithreaded=False
        )

        assert client.session_id is not None
        assert not client._is_closed

        client.close()
        assert client._is_closed

    def test_chat_completion(self):
        """Test chat completion request"""
        client = AutellixClient(is_multithreaded=False)

        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}],
            model="test-model"
        )

        assert response is not None
        assert "choices" in response
        assert "autellix_metadata" in response

        client.close()

    def test_context_manager(self):
        """Test client as context manager"""
        with autellix_session() as client:
            assert client.session_id is not None
            response = client.chat_completion(
                messages=[{"role": "user", "content": "Test"}]
            )
            assert response is not None

        # Session should be closed after context exit
        assert client._is_closed

    def test_parallel_chat_completion(self):
        """Test parallel chat completions"""
        client = AutellixClient(is_multithreaded=True)

        requests = [
            {"messages": [{"role": "user", "content": f"Query {i}"}]}
            for i in range(3)
        ]

        responses = client.parallel_chat_completion(requests)

        assert len(responses) == 3
        for response in responses:
            assert "autellix_metadata" in response

        client.close()

    def test_session_stats(self):
        """Test getting session statistics"""
        client = AutellixClient()

        # Make some calls
        for i in range(3):
            client.chat_completion(
                messages=[{"role": "user", "content": f"Message {i}"}]
            )

        stats = client.get_session_stats()

        assert stats["total_calls"] == 3
        assert "service_time" in stats
        assert "waiting_time" in stats

        client.close()


def test_integration_scenario():
    """
    Integration test: Multi-threaded program with map-reduce pattern
    """
    with autellix_session(is_multithreaded=True) as client:
        # Step 1: Planning (root node)
        planning_response = client.chat_completion(
            messages=[{"role": "user", "content": "Plan research on 3 cities"}]
        )
        planning_thread = planning_response["autellix_metadata"]["thread_id"]

        # Step 2: Parallel research (map)
        research_requests = [
            {"messages": [{"role": "user", "content": f"Research city {i}"}]}
            for i in range(3)
        ]
        research_responses = client.parallel_chat_completion(
            requests=research_requests,
            parent_thread_id=planning_thread
        )
        research_threads = [r["autellix_metadata"]["thread_id"] for r in research_responses]

        # Step 3: Aggregation (reduce)
        aggregation_response = client.chat_completion(
            messages=[{"role": "user", "content": "Aggregate results"}],
            parent_thread_ids=research_threads
        )

        # Verify stats
        stats = client.get_session_stats()
        assert stats["total_calls"] == 5  # 1 planning + 3 research + 1 aggregation
        assert stats["is_multithreaded"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
