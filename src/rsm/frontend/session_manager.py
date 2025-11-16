"""
Session Manager for Autellix RSM

This module implements session lifecycle management for the stateful API,
as described in Section 5 of the Autellix paper.

The session manager:
- Creates unique session IDs for programs
- Manages session lifecycle (start_session, end_session)
- Integrates with the global process table
- Tracks session-to-program mappings
"""

import uuid
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .process_table import GlobalProcessTable, ProgramEntry


class SessionState(Enum):
    """Session lifecycle states"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SessionInfo:
    """
    Information about a session.

    Attributes:
        session_id: Unique session identifier
        pid: Program ID in process table (same as session_id)
        created_at: Timestamp when session was created
        last_activity: Timestamp of last activity
        state: Current session state
        is_multithreaded: Whether this session uses multi-threading
        metadata: Additional user-provided metadata
    """
    session_id: str
    pid: str
    created_at: float
    last_activity: float
    state: SessionState = SessionState.ACTIVE
    is_multithreaded: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SessionManager:
    """
    Manages session lifecycle for the stateful Autellix API.

    From Autellix Section 5:
    - Upon program initialization, automatically issues start_session request
    - Returns unique session identifier
    - Creates corresponding entry in process table
    - Subsequent LLM calls annotated with session ID
    - end_session removes entry from process table

    This class is thread-safe and can be shared across multiple API instances.
    """

    def __init__(self, process_table: GlobalProcessTable):
        """
        Initialize the session manager.

        Args:
            process_table: Global process table for tracking programs
        """
        self.process_table = process_table
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = threading.RLock()

        # Statistics
        self._total_sessions_created = 0
        self._total_sessions_completed = 0
        self._total_sessions_failed = 0

    def start_session(
        self,
        is_multithreaded: bool = False,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new session for an agentic program.

        This is called automatically when a user's program initializes
        the Autellix API wrapper.

        From Autellix Section 5:
        - Creates unique session ID
        - Adds entry to process table
        - Returns session ID to client

        Args:
            is_multithreaded: Whether this program uses multi-threading (ATLAS vs PLAS)
            session_id: Optional custom session ID (if None, generates UUID)
            metadata: Optional metadata about the session

        Returns:
            Unique session identifier

        Raises:
            ValueError: If session_id already exists
        """
        with self._lock:
            # Generate or validate session ID
            if session_id is None:
                session_id = self._generate_session_id()
            elif session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")

            # Create program entry in process table
            # PID is same as session ID
            pid = session_id
            try:
                self.process_table.create_program(
                    pid=pid,
                    is_multithreaded=is_multithreaded
                )
            except ValueError as e:
                raise ValueError(f"Failed to create session: {e}")

            # Create session info
            current_time = time.time()
            session_info = SessionInfo(
                session_id=session_id,
                pid=pid,
                created_at=current_time,
                last_activity=current_time,
                is_multithreaded=is_multithreaded,
                metadata=metadata or {}
            )

            self._sessions[session_id] = session_info
            self._total_sessions_created += 1

            return session_id

    def end_session(
        self,
        session_id: str,
        state: SessionState = SessionState.COMPLETED
    ) -> bool:
        """
        End a session and remove it from the process table.

        This is called when:
        - Program completes successfully (COMPLETED)
        - Program encounters an error (FAILED)
        - Session times out (TIMEOUT)

        From Autellix Section 5:
        - Removes entry from process table
        - Cleans up session state

        Args:
            session_id: Session identifier
            state: Final session state

        Returns:
            True if session was successfully ended, False if not found
        """
        with self._lock:
            session_info = self._sessions.get(session_id)
            if not session_info:
                return False

            # Update session state
            session_info.state = state
            session_info.last_activity = time.time()

            # Remove from process table
            self.process_table.remove_program(session_info.pid)

            # Update statistics
            if state == SessionState.COMPLETED:
                self._total_sessions_completed += 1
            elif state == SessionState.FAILED:
                self._total_sessions_failed += 1

            # Remove from active sessions
            # Note: We keep session_info in memory for potential debugging/logging
            # In production, you might move to a separate completed_sessions dict

            return True

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session information.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo if found, None otherwise
        """
        with self._lock:
            return self._sessions.get(session_id)

    def update_activity(self, session_id: str):
        """
        Update last activity timestamp for a session.

        Called when the session submits a new LLM call or receives a response.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            session_info = self._sessions.get(session_id)
            if session_info:
                session_info.last_activity = time.time()

    def is_active(self, session_id: str) -> bool:
        """
        Check if a session is active.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists and is active
        """
        with self._lock:
            session_info = self._sessions.get(session_id)
            return session_info is not None and session_info.state == SessionState.ACTIVE

    def list_active_sessions(self) -> list[str]:
        """
        Get list of all active session IDs.

        Returns:
            List of active session IDs
        """
        with self._lock:
            return [
                sid for sid, info in self._sessions.items()
                if info.state == SessionState.ACTIVE
            ]

    def cleanup_inactive_sessions(self, timeout_seconds: float = 3600):
        """
        Clean up sessions that have been inactive for too long.

        This prevents memory leaks from abandoned sessions.

        Args:
            timeout_seconds: Inactivity timeout in seconds (default: 1 hour)
        """
        with self._lock:
            current_time = time.time()
            inactive_sessions = []

            for session_id, session_info in self._sessions.items():
                if session_info.state == SessionState.ACTIVE:
                    inactive_time = current_time - session_info.last_activity
                    if inactive_time > timeout_seconds:
                        inactive_sessions.append(session_id)

            # End inactive sessions
            for session_id in inactive_sessions:
                self.end_session(session_id, state=SessionState.TIMEOUT)

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific session.

        Combines session info with program metrics from process table.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics, or None if not found
        """
        with self._lock:
            session_info = self._sessions.get(session_id)
            if not session_info:
                return None

            program_entry = self.process_table.get_program(session_info.pid)

            stats = {
                "session_id": session_id,
                "state": session_info.state.value,
                "created_at": session_info.created_at,
                "last_activity": session_info.last_activity,
                "duration": time.time() - session_info.created_at,
                "is_multithreaded": session_info.is_multithreaded,
                "metadata": session_info.metadata
            }

            if program_entry:
                stats.update({
                    "service_time": program_entry.service_time,
                    "waiting_time": program_entry.waiting_time,
                    "total_calls": program_entry.total_calls,
                    "completed_calls": program_entry.completed_calls,
                    "active_threads": program_entry.get_active_thread_count(),
                    "starvation_ratio": program_entry.get_starvation_ratio(),
                    "engine_ids": list(program_entry.engine_ids)
                })

            return stats

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global statistics across all sessions.

        Returns:
            Dictionary with global statistics
        """
        with self._lock:
            active_count = sum(
                1 for s in self._sessions.values()
                if s.state == SessionState.ACTIVE
            )

            return {
                "total_sessions_created": self._total_sessions_created,
                "total_sessions_completed": self._total_sessions_completed,
                "total_sessions_failed": self._total_sessions_failed,
                "active_sessions": active_count,
                "process_table_stats": self.process_table.get_stats()
            }

    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID.

        Uses UUID4 for uniqueness. In production, you might want to add
        additional context like timestamps or node IDs for distributed systems.

        Returns:
            Unique session identifier
        """
        return f"session_{uuid.uuid4().hex[:16]}"

    def register_llm_call(
        self,
        session_id: str,
        call_id: str,
        thread_id: Optional[str] = None,
        prefill_tokens: int = 0,
        engine_id: Optional[str] = None,
        parent_thread_ids: Optional[list[str]] = None
    ) -> Optional[str]:
        """
        Register a new LLM call for a session.

        This is called by the API wrapper when a new LLM call is submitted.

        Args:
            session_id: Session identifier
            call_id: Unique LLM call identifier
            thread_id: Thread ID (generated if None)
            prefill_tokens: Number of input tokens
            engine_id: Engine assigned to this call
            parent_thread_ids: Parent threads (for ATLAS)

        Returns:
            Thread ID, or None if session not found
        """
        with self._lock:
            session_info = self._sessions.get(session_id)
            if not session_info or session_info.state != SessionState.ACTIVE:
                return None

            # Generate thread ID if not provided
            if thread_id is None:
                thread_id = f"thread_{uuid.uuid4().hex[:12]}"

            # Update session activity
            session_info.last_activity = time.time()

            # Add to process table
            thread_metadata = self.process_table.add_llm_call(
                pid=session_info.pid,
                call_id=call_id,
                thread_id=thread_id,
                prefill_tokens=prefill_tokens,
                engine_id=engine_id,
                parent_thread_ids=parent_thread_ids
            )

            return thread_id if thread_metadata else None

    def complete_llm_call(
        self,
        session_id: str,
        thread_id: str,
        service_time: float,
        waiting_time: float
    ) -> bool:
        """
        Mark an LLM call as completed.

        This is called by the scheduler when a call finishes.

        Args:
            session_id: Session identifier
            thread_id: Thread identifier
            service_time: Time spent executing
            waiting_time: Time spent waiting

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            session_info = self._sessions.get(session_id)
            if not session_info:
                return False

            # Update session activity
            session_info.last_activity = time.time()

            # Get call_id from thread
            program_entry = self.process_table.get_program(session_info.pid)
            if not program_entry:
                return False

            thread = program_entry.threads.get(thread_id)
            if not thread:
                return False

            call_id = thread.call_id

            # Update metrics in process table
            self.process_table.update_program_metrics(
                pid=session_info.pid,
                call_id=call_id,
                service_time=service_time,
                waiting_time=waiting_time
            )

            # Remove thread from process table
            self.process_table.complete_llm_call(
                pid=session_info.pid,
                thread_id=thread_id
            )

            return True
