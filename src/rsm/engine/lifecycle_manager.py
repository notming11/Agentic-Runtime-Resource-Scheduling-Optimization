"""
Request Lifecycle Manager for Autellix RSM

This module implements request state management and cancellation support
for LLM calls in the multi-engine system.

Key responsibilities:
- Track request states (pending, running, completed, cancelled)
- Handle request cancellation (stop in-flight LLM calls)
- Manage session-to-request mappings
- Provide request lifecycle hooks
"""

import time
import threading
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum


class RequestState(Enum):
    """
    States of an LLM request in its lifecycle.

    PENDING: Request queued but not yet executing
    RUNNING: Request is currently executing on an engine
    COMPLETED: Request finished successfully
    CANCELLED: Request was cancelled before completion
    FAILED: Request failed due to an error
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class RequestMetadata:
    """
    Metadata for a single LLM request.

    Attributes:
        request_id: Unique request identifier
        session_id: Associated session/program ID
        engine_id: Engine assigned to process this request (None if not assigned)
        call_id: LLM call identifier (for process table tracking)
        thread_id: Thread identifier (for ATLAS/PLAS)
        state: Current request state
        created_at: When this request was created
        started_at: When this request started executing (None if not started)
        completed_at: When this request completed (None if not completed)
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens generated
        error_message: Error message if request failed
        cancellation_reason: Reason for cancellation (if cancelled)
    """
    request_id: str
    session_id: str
    engine_id: Optional[str] = None
    call_id: Optional[str] = None
    thread_id: Optional[str] = None
    state: RequestState = RequestState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_message: Optional[str] = None
    cancellation_reason: Optional[str] = None

    def get_waiting_time(self) -> float:
        """Calculate time spent waiting (before execution starts)"""
        if self.started_at is None:
            return time.time() - self.created_at
        return self.started_at - self.created_at

    def get_execution_time(self) -> float:
        """Calculate time spent executing"""
        if self.started_at is None:
            return 0.0
        if self.completed_at is None:
            return time.time() - self.started_at
        return self.completed_at - self.started_at

    def get_total_time(self) -> float:
        """Calculate total time from creation to completion"""
        if self.completed_at is None:
            return time.time() - self.created_at
        return self.completed_at - self.created_at


class RequestLifecycleManager:
    """
    Manages the lifecycle of LLM requests across multiple engines.

    This class provides:
    - Request registration and state tracking
    - Cancellation support for in-flight requests
    - Session-level cancellation (cancel all requests for a session)
    - Lifecycle hooks for monitoring and integration
    - Thread-safe operations

    Integration points:
    - Frontend: Creates requests, monitors state
    - Scheduler: Updates request states
    - MultiEngineManager: Routes requests, handles cancellation
    """

    def __init__(self):
        """Initialize the request lifecycle manager"""
        # request_id -> RequestMetadata
        self._active_requests: Dict[str, RequestMetadata] = {}

        # session_id -> List[request_id]
        self._session_requests: Dict[str, List[str]] = {}

        # engine_id -> Set[request_id]
        self._engine_requests: Dict[str, Set[str]] = {}

        # Completed requests (kept for history/debugging)
        self._completed_requests: Dict[str, RequestMetadata] = {}

        # Lifecycle hooks
        self._on_state_change_hooks: List[Callable] = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "cancelled_requests": 0,
            "failed_requests": 0
        }

    def start_request(
        self,
        request_id: str,
        session_id: str,
        engine_id: Optional[str] = None,
        call_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        prompt_tokens: int = 0
    ) -> RequestMetadata:
        """
        Register a new request in the system.

        Args:
            request_id: Unique request identifier
            session_id: Associated session/program ID
            engine_id: Engine assigned to this request (optional)
            call_id: LLM call identifier
            thread_id: Thread identifier (for ATLAS/PLAS)
            prompt_tokens: Number of input tokens

        Returns:
            Created RequestMetadata

        Raises:
            ValueError: If request_id already exists
        """
        with self._lock:
            if request_id in self._active_requests:
                raise ValueError(f"Request {request_id} already exists")

            # Create request metadata
            metadata = RequestMetadata(
                request_id=request_id,
                session_id=session_id,
                engine_id=engine_id,
                call_id=call_id,
                thread_id=thread_id,
                prompt_tokens=prompt_tokens
            )

            # Register request
            self._active_requests[request_id] = metadata

            # Add to session mapping
            if session_id not in self._session_requests:
                self._session_requests[session_id] = []
            self._session_requests[session_id].append(request_id)

            # Add to engine mapping if engine assigned
            if engine_id:
                if engine_id not in self._engine_requests:
                    self._engine_requests[engine_id] = set()
                self._engine_requests[engine_id].add(request_id)

            # Update stats
            self._stats["total_requests"] += 1

            # Trigger hooks
            self._trigger_state_change(request_id, RequestState.PENDING)

            return metadata

    def update_request_state(
        self,
        request_id: str,
        new_state: RequestState,
        engine_id: Optional[str] = None,
        completion_tokens: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update the state of a request.

        Args:
            request_id: Request identifier
            new_state: New state
            engine_id: Engine ID (if state is RUNNING)
            completion_tokens: Number of tokens generated (if COMPLETED)
            error_message: Error message (if FAILED)

        Returns:
            True if update successful, False if request not found
        """
        with self._lock:
            metadata = self._active_requests.get(request_id)
            if not metadata:
                # Check completed requests
                metadata = self._completed_requests.get(request_id)
                if not metadata:
                    return False

            old_state = metadata.state
            metadata.state = new_state

            # Update timestamps based on state
            if new_state == RequestState.RUNNING:
                metadata.started_at = time.time()
                if engine_id:
                    # Update engine assignment
                    if metadata.engine_id and metadata.engine_id != engine_id:
                        # Remove from old engine
                        if metadata.engine_id in self._engine_requests:
                            self._engine_requests[metadata.engine_id].discard(request_id)

                    metadata.engine_id = engine_id

                    # Add to new engine
                    if engine_id not in self._engine_requests:
                        self._engine_requests[engine_id] = set()
                    self._engine_requests[engine_id].add(request_id)

            elif new_state in [RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED]:
                metadata.completed_at = time.time()

                if completion_tokens is not None:
                    metadata.completion_tokens = completion_tokens

                if error_message:
                    metadata.error_message = error_message

                # Move to completed requests
                if request_id in self._active_requests:
                    del self._active_requests[request_id]
                    self._completed_requests[request_id] = metadata

                # Remove from engine mapping
                if metadata.engine_id and metadata.engine_id in self._engine_requests:
                    self._engine_requests[metadata.engine_id].discard(request_id)

                # Update stats
                if new_state == RequestState.COMPLETED:
                    self._stats["completed_requests"] += 1
                elif new_state == RequestState.CANCELLED:
                    self._stats["cancelled_requests"] += 1
                elif new_state == RequestState.FAILED:
                    self._stats["failed_requests"] += 1

            # Trigger hooks
            self._trigger_state_change(request_id, new_state, old_state)

            return True

    def cancel_request(
        self,
        request_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel a request.

        This marks the request as cancelled. The engine is responsible
        for actually stopping the LLM call execution.

        Args:
            request_id: Request identifier
            reason: Cancellation reason (optional)

        Returns:
            True if request was cancelled, False if not found or already completed
        """
        with self._lock:
            metadata = self._active_requests.get(request_id)
            if not metadata:
                return False

            # Can only cancel pending or running requests
            if metadata.state in [RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED]:
                return False

            # Set cancellation reason
            metadata.cancellation_reason = reason or "User cancelled"

            # Update state
            return self.update_request_state(request_id, RequestState.CANCELLED)

    def cancel_session(
        self,
        session_id: str,
        reason: Optional[str] = None
    ) -> List[str]:
        """
        Cancel all active requests for a session.

        Useful for cancelling an entire workflow when it's no longer needed.

        Args:
            session_id: Session identifier
            reason: Cancellation reason (optional)

        Returns:
            List of cancelled request IDs
        """
        with self._lock:
            if session_id not in self._session_requests:
                return []

            cancelled_requests = []
            reason = reason or f"Session {session_id} cancelled"

            for request_id in self._session_requests[session_id]:
                if self.cancel_request(request_id, reason):
                    cancelled_requests.append(request_id)

            return cancelled_requests

    def complete_request(
        self,
        request_id: str,
        completion_tokens: int = 0
    ) -> bool:
        """
        Mark a request as completed successfully.

        Args:
            request_id: Request identifier
            completion_tokens: Number of tokens generated

        Returns:
            True if successful, False otherwise
        """
        return self.update_request_state(
            request_id,
            RequestState.COMPLETED,
            completion_tokens=completion_tokens
        )

    def fail_request(
        self,
        request_id: str,
        error_message: str
    ) -> bool:
        """
        Mark a request as failed.

        Args:
            request_id: Request identifier
            error_message: Error message

        Returns:
            True if successful, False otherwise
        """
        return self.update_request_state(
            request_id,
            RequestState.FAILED,
            error_message=error_message
        )

    def get_request(self, request_id: str) -> Optional[RequestMetadata]:
        """
        Get request metadata.

        Checks both active and completed requests.

        Args:
            request_id: Request identifier

        Returns:
            RequestMetadata if found, None otherwise
        """
        with self._lock:
            return (self._active_requests.get(request_id) or
                   self._completed_requests.get(request_id))

    def get_session_requests(
        self,
        session_id: str,
        active_only: bool = True
    ) -> List[RequestMetadata]:
        """
        Get all requests for a session.

        Args:
            session_id: Session identifier
            active_only: If True, only return active requests

        Returns:
            List of RequestMetadata
        """
        with self._lock:
            if session_id not in self._session_requests:
                return []

            requests = []
            for request_id in self._session_requests[session_id]:
                metadata = self.get_request(request_id)
                if metadata:
                    if active_only:
                        if metadata.state in [RequestState.PENDING, RequestState.RUNNING]:
                            requests.append(metadata)
                    else:
                        requests.append(metadata)

            return requests

    def get_engine_requests(
        self,
        engine_id: str
    ) -> List[RequestMetadata]:
        """
        Get all active requests on an engine.

        Args:
            engine_id: Engine identifier

        Returns:
            List of RequestMetadata
        """
        with self._lock:
            if engine_id not in self._engine_requests:
                return []

            requests = []
            for request_id in self._engine_requests[engine_id]:
                metadata = self._active_requests.get(request_id)
                if metadata:
                    requests.append(metadata)

            return requests

    def get_pending_requests(self) -> List[RequestMetadata]:
        """
        Get all pending requests (not yet started).

        Returns:
            List of RequestMetadata
        """
        with self._lock:
            return [
                metadata for metadata in self._active_requests.values()
                if metadata.state == RequestState.PENDING
            ]

    def get_running_requests(self) -> List[RequestMetadata]:
        """
        Get all running requests.

        Returns:
            List of RequestMetadata
        """
        with self._lock:
            return [
                metadata for metadata in self._active_requests.values()
                if metadata.state == RequestState.RUNNING
            ]

    def cleanup_session(self, session_id: str):
        """
        Clean up request tracking for a session.

        Should be called when a session ends.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            if session_id in self._session_requests:
                # Remove request IDs from session mapping
                # Keep completed requests in history
                del self._session_requests[session_id]

    def register_state_change_hook(
        self,
        callback: Callable[[str, RequestState, Optional[RequestState]], None]
    ):
        """
        Register a callback for request state changes.

        Callback signature: callback(request_id, new_state, old_state)

        Args:
            callback: Function to call on state change
        """
        with self._lock:
            self._on_state_change_hooks.append(callback)

    def _trigger_state_change(
        self,
        request_id: str,
        new_state: RequestState,
        old_state: Optional[RequestState] = None
    ):
        """
        Trigger registered state change hooks.

        Args:
            request_id: Request identifier
            new_state: New state
            old_state: Previous state (optional)
        """
        for hook in self._on_state_change_hooks:
            try:
                hook(request_id, new_state, old_state)
            except Exception:
                # Ignore errors in hooks
                pass

    def get_stats(self) -> Dict:
        """
        Get statistics about request lifecycle.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            active_count = len(self._active_requests)
            pending_count = sum(
                1 for r in self._active_requests.values()
                if r.state == RequestState.PENDING
            )
            running_count = sum(
                1 for r in self._active_requests.values()
                if r.state == RequestState.RUNNING
            )

            return {
                "total_requests": self._stats["total_requests"],
                "completed_requests": self._stats["completed_requests"],
                "cancelled_requests": self._stats["cancelled_requests"],
                "failed_requests": self._stats["failed_requests"],
                "active_requests": active_count,
                "pending_requests": pending_count,
                "running_requests": running_count,
                "active_sessions": len(self._session_requests),
                "engines_in_use": len([e for e in self._engine_requests.values() if e])
            }

    def cleanup_old_requests(self, max_age_seconds: float = 3600):
        """
        Remove old completed requests from history.

        Args:
            max_age_seconds: Maximum age for keeping completed requests (default: 1 hour)
        """
        with self._lock:
            current_time = time.time()
            old_requests = []

            for request_id, metadata in self._completed_requests.items():
                if metadata.completed_at:
                    age = current_time - metadata.completed_at
                    if age > max_age_seconds:
                        old_requests.append(request_id)

            # Remove old requests
            for request_id in old_requests:
                del self._completed_requests[request_id]
