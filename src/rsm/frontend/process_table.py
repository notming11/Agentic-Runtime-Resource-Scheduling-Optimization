"""
Global Process Table for Autellix RSM

This module implements the global process table that tracks runtime metadata
for all active programs, as described in Section 4.2.1 of the Autellix paper.

The process table enables program-level scheduling by maintaining:
- Service time (critical path length)
- Waiting time (for anti-starvation)
- Engine affinity (for KV-cache locality)
- Thread metadata (active LLM calls)
- Starvation counters
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class ProgramState(Enum):
    """Program execution states"""
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ThreadMetadata:
    """
    Metadata for a single thread (active LLM call) within a program.

    Each thread corresponds to an active LLM call. For multi-threaded programs,
    multiple threads may be active simultaneously.

    Attributes:
        thread_id: Unique identifier for this thread
        call_id: Unique identifier for the LLM call
        arrival_time: When this LLM call arrived at the engine (timestamp)
        start_time: When this LLM call started executing (None if not started)
        waiting_time: Time spent waiting in scheduler queues (seconds)
        service_time: Time spent executing on model executor (seconds)
        prefill_tokens: Number of input tokens
        decode_tokens: Number of output tokens generated so far
        engine_id: ID of the engine processing this call
        parent_thread_ids: IDs of parent threads in the DAG (for ATLAS)
        priority: Current priority value (for scheduling)
        queue_index: Current queue index in multilevel scheduler
    """
    thread_id: str
    call_id: str
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    waiting_time: float = 0.0
    service_time: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    engine_id: Optional[str] = None
    parent_thread_ids: List[str] = field(default_factory=list)
    priority: float = 0.0
    queue_index: int = 0

    def update_waiting_time(self) -> float:
        """Update and return current waiting time"""
        if self.start_time is None:
            self.waiting_time = time.time() - self.arrival_time
        return self.waiting_time

    def update_service_time(self, additional_time: float):
        """Add to service time (called after each decode step)"""
        self.service_time += additional_time


@dataclass
class ProgramEntry:
    """
    Entry in the global process table for a single program.

    As described in Autellix Section 4.2.1, each program entry tracks:
    - Service time: For single-threaded programs, cumulative execution time.
                   For multi-threaded programs, longest critical path.
    - Waiting time: Total time spent in scheduler queues
    - Engine IDs: Engines currently running this program's calls
    - Thread metadata: Active LLM calls and their statistics

    Attributes:
        pid: Unique program identifier (session ID)
        service_time: Cumulative service time (single-threaded) or
                     max critical path length (multi-threaded)
        waiting_time: Total waiting time across all LLM calls
        engine_ids: Set of engine IDs processing this program
        threads: Dictionary mapping thread_id to ThreadMetadata
        most_recent_arrival: Timestamp of most recent LLM call arrival
        most_recent_completion: Timestamp of most recent LLM call completion
        state: Current program state
        created_at: When this program entry was created
        is_multithreaded: Whether this is a multi-threaded program
        total_calls: Total number of LLM calls submitted
        completed_calls: Number of completed LLM calls
    """
    pid: str
    service_time: float = 0.0
    waiting_time: float = 0.0
    engine_ids: Set[str] = field(default_factory=set)
    threads: Dict[str, ThreadMetadata] = field(default_factory=dict)
    most_recent_arrival: Optional[float] = None
    most_recent_completion: Optional[float] = None
    state: ProgramState = ProgramState.RUNNING
    created_at: float = field(default_factory=time.time)
    is_multithreaded: bool = False
    total_calls: int = 0
    completed_calls: int = 0

    def get_starvation_ratio(self) -> float:
        """
        Calculate wait-to-service ratio for anti-starvation mechanism.

        From Autellix Section 4.2.2:
        A program is promoted to highest priority queue when:
            W_total / T_total >= β (starvation threshold)

        Returns:
            Ratio of total waiting time to total service time
        """
        if self.service_time == 0:
            return float('inf') if self.waiting_time > 0 else 0.0
        return self.waiting_time / self.service_time

    def update_service_time_single_threaded(self, call_service_time: float):
        """
        Update service time for single-threaded programs (PLAS).

        From Autellix Equation 1:
            p(c_j) = Σ_{k<j, c_k.id=c_i.id} t_k

        Service time is cumulative execution time of all completed calls.
        """
        self.service_time += call_service_time

    def update_service_time_multi_threaded(self, thread_id: str, parent_ids: List[str]):
        """
        Update service time for multi-threaded programs (ATLAS).

        From Autellix Equation 2:
            p(c_j) = max_{c_k ∈ P(c_j)} {p(c_k) + t_k}

        Service time is the longest observed critical path.
        Only update if this thread's critical path is longer.
        """
        if thread_id not in self.threads:
            return

        thread = self.threads[thread_id]

        # Calculate this thread's critical path
        if not parent_ids:
            # Root node
            critical_path = thread.service_time
        else:
            # Find max parent critical path
            max_parent_path = 0.0
            for parent_id in parent_ids:
                if parent_id in self.threads:
                    parent_thread = self.threads[parent_id]
                    max_parent_path = max(max_parent_path,
                                        parent_thread.priority + parent_thread.service_time)
            critical_path = max_parent_path

        # Update thread's priority (critical path to this point)
        thread.priority = critical_path

        # Update program's service time if this is the longest critical path
        self.service_time = max(self.service_time, critical_path)

    def add_thread(self, thread: ThreadMetadata):
        """Add a new thread (LLM call) to this program"""
        self.threads[thread.thread_id] = thread
        self.total_calls += 1
        self.most_recent_arrival = thread.arrival_time
        if thread.engine_id:
            self.engine_ids.add(thread.engine_id)

    def remove_thread(self, thread_id: str) -> Optional[ThreadMetadata]:
        """Remove and return a completed thread"""
        thread = self.threads.pop(thread_id, None)
        if thread:
            self.completed_calls += 1
            self.most_recent_completion = time.time()
        return thread

    def get_active_thread_count(self) -> int:
        """Return number of currently active threads"""
        return len(self.threads)


class GlobalProcessTable:
    """
    Global, thread-safe process table for tracking all running programs.

    This implements the process table described in Autellix Section 4.2.1.
    The table is accessed by both the scheduler (for prioritization) and
    the load balancer (for engine routing decisions).

    Thread-safety is critical as the table is accessed concurrently by:
    - Frontend (adding/removing programs)
    - Scheduler (reading priorities, updating metrics)
    - Load balancer (reading engine assignments)
    """

    def __init__(self):
        """Initialize the global process table"""
        self._table: Dict[str, ProgramEntry] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested access

    def create_program(self, pid: str, is_multithreaded: bool = False) -> ProgramEntry:
        """
        Create a new program entry in the process table.

        Args:
            pid: Unique program identifier
            is_multithreaded: Whether this program uses multi-threading

        Returns:
            The created ProgramEntry

        Raises:
            ValueError: If program with this PID already exists
        """
        with self._lock:
            if pid in self._table:
                raise ValueError(f"Program {pid} already exists in process table")

            entry = ProgramEntry(
                pid=pid,
                is_multithreaded=is_multithreaded
            )
            self._table[pid] = entry
            return entry

    def remove_program(self, pid: str) -> Optional[ProgramEntry]:
        """
        Remove a program from the process table.

        Called when a program completes or fails.

        Args:
            pid: Program identifier

        Returns:
            The removed ProgramEntry, or None if not found
        """
        with self._lock:
            return self._table.pop(pid, None)

    def get_program(self, pid: str) -> Optional[ProgramEntry]:
        """
        Get a program entry by PID.

        Args:
            pid: Program identifier

        Returns:
            ProgramEntry if found, None otherwise
        """
        with self._lock:
            return self._table.get(pid)

    def update_program_metrics(self, pid: str, call_id: str,
                              service_time: float, waiting_time: float):
        """
        Update program metrics when an LLM call completes.

        This is called by the scheduler when a call finishes (Algorithm 1, Line 17).

        Args:
            pid: Program identifier
            call_id: LLM call identifier
            service_time: Time spent executing this call
            waiting_time: Time this call spent waiting
        """
        with self._lock:
            entry = self._table.get(pid)
            if not entry:
                return

            # Update waiting time
            entry.waiting_time += waiting_time

            # Update service time based on program type
            if entry.is_multithreaded:
                # Find thread and update using ATLAS
                thread = None
                for t in entry.threads.values():
                    if t.call_id == call_id:
                        thread = t
                        break

                if thread:
                    thread.update_service_time(service_time)
                    entry.update_service_time_multi_threaded(
                        thread.thread_id,
                        thread.parent_thread_ids
                    )
            else:
                # PLAS: cumulative service time
                entry.update_service_time_single_threaded(service_time)

    def add_llm_call(self, pid: str, call_id: str, thread_id: str,
                     prefill_tokens: int, engine_id: Optional[str] = None,
                     parent_thread_ids: Optional[List[str]] = None) -> Optional[ThreadMetadata]:
        """
        Register a new LLM call for a program.

        Args:
            pid: Program identifier
            call_id: Unique LLM call identifier
            thread_id: Unique thread identifier
            prefill_tokens: Number of input tokens
            engine_id: Engine assigned to this call
            parent_thread_ids: Parent threads in DAG (for ATLAS)

        Returns:
            Created ThreadMetadata, or None if program not found
        """
        with self._lock:
            entry = self._table.get(pid)
            if not entry:
                return None

            thread = ThreadMetadata(
                thread_id=thread_id,
                call_id=call_id,
                prefill_tokens=prefill_tokens,
                engine_id=engine_id,
                parent_thread_ids=parent_thread_ids or []
            )

            # Set initial priority from program's current service time
            thread.priority = entry.service_time

            entry.add_thread(thread)
            return thread

    def complete_llm_call(self, pid: str, thread_id: str) -> Optional[ThreadMetadata]:
        """
        Mark an LLM call as completed and remove it from active threads.

        Args:
            pid: Program identifier
            thread_id: Thread identifier

        Returns:
            Removed ThreadMetadata, or None if not found
        """
        with self._lock:
            entry = self._table.get(pid)
            if not entry:
                return None

            return entry.remove_thread(thread_id)

    def get_program_priority(self, pid: str) -> float:
        """
        Get the current priority (service time) for a program.

        Used by scheduler to assign LLM calls to priority queues.

        Args:
            pid: Program identifier

        Returns:
            Program's service time (priority value)
        """
        with self._lock:
            entry = self._table.get(pid)
            return entry.service_time if entry else 0.0

    def get_starvation_ratio(self, pid: str) -> float:
        """
        Get the wait-to-service ratio for anti-starvation mechanism.

        Args:
            pid: Program identifier

        Returns:
            Starvation ratio (W_total / T_total)
        """
        with self._lock:
            entry = self._table.get(pid)
            return entry.get_starvation_ratio() if entry else 0.0

    def get_engine_ids(self, pid: str) -> Set[str]:
        """
        Get the set of engines currently processing this program.

        Used by load balancer for locality-aware routing.

        Args:
            pid: Program identifier

        Returns:
            Set of engine IDs
        """
        with self._lock:
            entry = self._table.get(pid)
            return entry.engine_ids.copy() if entry else set()

    def list_programs(self) -> List[str]:
        """
        Get list of all active program IDs.

        Returns:
            List of PIDs
        """
        with self._lock:
            return list(self._table.keys())

    def get_stats(self) -> Dict:
        """
        Get global statistics about the process table.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_programs = len(self._table)
            total_active_calls = sum(len(e.threads) for e in self._table.values())
            total_completed_calls = sum(e.completed_calls for e in self._table.values())

            return {
                "total_programs": total_programs,
                "total_active_calls": total_active_calls,
                "total_completed_calls": total_completed_calls,
                "programs": {
                    pid: {
                        "service_time": entry.service_time,
                        "waiting_time": entry.waiting_time,
                        "active_threads": len(entry.threads),
                        "completed_calls": entry.completed_calls,
                        "starvation_ratio": entry.get_starvation_ratio(),
                        "engines": list(entry.engine_ids),
                        "state": entry.state.value
                    }
                    for pid, entry in self._table.items()
                }
            }

    def cleanup_stale_programs(self, timeout_seconds: float = 3600):
        """
        Remove programs that haven't had activity for a long time.

        Prevents memory leaks from abandoned sessions.

        Args:
            timeout_seconds: Inactivity timeout in seconds
        """
        with self._lock:
            current_time = time.time()
            stale_pids = []

            for pid, entry in self._table.items():
                # Check if program has been inactive
                last_activity = entry.most_recent_completion or entry.most_recent_arrival
                if last_activity and (current_time - last_activity) > timeout_seconds:
                    # No active threads
                    if len(entry.threads) == 0:
                        stale_pids.append(pid)

            # Remove stale programs
            for pid in stale_pids:
                self._table.pop(pid, None)
