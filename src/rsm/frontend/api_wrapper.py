"""
Stateful API Wrapper for Autellix RSM

This module implements the stateful API layer that extends OpenAI's Chat Completion
and vLLM's Python APIs, as described in Section 5 of the Autellix paper.

The API wrapper:
- Provides a stateful interface that appears stateless to developers
- Automatically manages sessions (start_session on init, end_session on cleanup)
- Annotates LLM calls with session, program, and thread IDs
- Tracks timing information for scheduling
- Integrates with process table and session manager
"""

import uuid
import time
import atexit
import warnings
from typing import Optional, Dict, Any, List, Iterator, Union
from contextlib import contextmanager

from .session_manager import SessionManager, SessionState
from .process_table import GlobalProcessTable


class AutellixClient:
    """
    Stateful client for Autellix RSM.

    From Autellix Section 5:
    - Users import Autellix's library into their Python applications
    - Upon program initialization, automatically issues start_session request
    - Returns unique session identifier
    - Subsequent LLM calls annotated with session ID
    - end_session called on cleanup

    This class wraps the underlying LLM engine (vLLM, OpenAI, etc.) and
    transparently injects session and program context.

    Example:
        ```python
        # Single-threaded program (uses PLAS)
        client = AutellixClient(backend_url="http://localhost:8000")
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        client.close()

        # Multi-threaded program (uses ATLAS)
        client = AutellixClient(
            backend_url="http://localhost:8000",
            is_multithreaded=True
        )
        # ... parallel calls ...
        ```
    """

    # Global instances for singleton process table and session manager
    _global_process_table: Optional[GlobalProcessTable] = None
    _global_session_manager: Optional[SessionManager] = None

    @classmethod
    def _get_global_instances(cls):
        """Get or create global process table and session manager instances."""
        if cls._global_process_table is None:
            cls._global_process_table = GlobalProcessTable()
        if cls._global_session_manager is None:
            cls._global_session_manager = SessionManager(cls._global_process_table)
        return cls._global_process_table, cls._global_session_manager

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        is_multithreaded: bool = False,
        session_metadata: Optional[Dict[str, Any]] = None,
        auto_start_session: bool = True
    ):
        """
        Initialize Autellix client.

        Args:
            backend_url: URL of the Autellix backend server
            is_multithreaded: Whether this program uses multi-threading (ATLAS vs PLAS)
            session_metadata: Optional metadata to attach to the session
            auto_start_session: Whether to automatically start session on init
        """
        self.backend_url = backend_url
        self.is_multithreaded = is_multithreaded

        # Get global instances
        self.process_table, self.session_manager = self._get_global_instances()

        # Session state
        self.session_id: Optional[str] = None
        self._is_closed = False

        # Start session automatically
        if auto_start_session:
            self.session_id = self.session_manager.start_session(
                is_multithreaded=is_multithreaded,
                metadata=session_metadata
            )

        # Register cleanup handler
        atexit.register(self.close)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        thread_id: Optional[str] = None,
        parent_thread_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Send a chat completion request to the Autellix backend.

        This method automatically:
        1. Annotates the request with session_id, thread_id
        2. Registers the call in the process table
        3. Tracks timing information
        4. Routes to appropriate engine via load balancer

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses
            thread_id: Optional thread ID (for multi-threaded programs)
            parent_thread_ids: Parent thread IDs (for ATLAS scheduling)
            **kwargs: Additional parameters to pass to backend

        Returns:
            Response dictionary or iterator (if streaming)

        Raises:
            RuntimeError: If session is not active
        """
        if not self.session_id or self._is_closed:
            raise RuntimeError("Client session is not active. Call start_session() first.")

        # Generate call ID and thread ID
        call_id = self._generate_call_id()
        if thread_id is None:
            thread_id = self._generate_thread_id()

        # Calculate input tokens (simplified estimate)
        prefill_tokens = self._estimate_tokens(messages)

        # Register call with session manager
        registered_thread_id = self.session_manager.register_llm_call(
            session_id=self.session_id,
            call_id=call_id,
            thread_id=thread_id,
            prefill_tokens=prefill_tokens,
            parent_thread_ids=parent_thread_ids
        )

        if not registered_thread_id:
            raise RuntimeError(f"Failed to register LLM call for session {self.session_id}")

        # Prepare request with Autellix metadata
        request_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            # Autellix-specific fields
            "autellix_session_id": self.session_id,
            "autellix_call_id": call_id,
            "autellix_thread_id": registered_thread_id,
            "autellix_parent_threads": parent_thread_ids or [],
            "autellix_is_multithreaded": self.is_multithreaded,
            **kwargs
        }

        # Record request timestamp
        request_start_time = time.time()

        # Send request to backend
        # In real implementation, this would use requests library or async HTTP client
        # For now, we'll return a mock response structure
        response = self._send_request(request_data)

        # Record completion time
        completion_time = time.time()
        service_time = completion_time - request_start_time

        # Update session activity
        self.session_manager.update_activity(self.session_id)

        return response

    def parallel_chat_completion(
        self,
        requests: List[Dict[str, Any]],
        parent_thread_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Submit multiple chat completion requests in parallel.

        This is a convenience method for multi-threaded programs.
        Each request will be assigned a unique thread_id, and all will
        have the same parent_thread_id for ATLAS scheduling.

        Args:
            requests: List of request dictionaries (same format as chat_completion)
            parent_thread_id: Common parent thread ID for all requests

        Returns:
            List of response dictionaries

        Raises:
            RuntimeError: If session is not active or program is not multithreaded
        """
        if not self.is_multithreaded:
            warnings.warn(
                "parallel_chat_completion called on single-threaded client. "
                "Consider using is_multithreaded=True for ATLAS scheduling."
            )

        # Submit all requests
        responses = []
        for req in requests:
            parent_ids = [parent_thread_id] if parent_thread_id else None
            response = self.chat_completion(
                parent_thread_ids=parent_ids,
                **req
            )
            responses.append(response)

        return responses

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current session.

        Returns:
            Dictionary with session statistics

        Raises:
            RuntimeError: If no active session
        """
        if not self.session_id:
            raise RuntimeError("No active session")

        stats = self.session_manager.get_session_stats(self.session_id)
        if not stats:
            raise RuntimeError(f"Session {self.session_id} not found")

        return stats

    def close(self):
        """
        Close the client and end the session.

        This is called automatically on program exit via atexit handler,
        but can also be called manually for explicit cleanup.
        """
        if self._is_closed or not self.session_id:
            return

        # End session
        self.session_manager.end_session(
            self.session_id,
            state=SessionState.COMPLETED
        )

        self._is_closed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # Error occurred
            if self.session_id:
                self.session_manager.end_session(
                    self.session_id,
                    state=SessionState.FAILED
                )
        else:
            # Normal completion
            self.close()

    def _send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to Autellix backend.

        In a real implementation, this would:
        1. Make HTTP request to backend server
        2. Handle streaming responses
        3. Parse and return response

        For now, returns a mock response.

        Args:
            request_data: Request payload

        Returns:
            Response dictionary
        """
        # TODO: Implement actual HTTP client
        # Example using requests library:
        # import requests
        # response = requests.post(
        #     f"{self.backend_url}/v1/chat/completions",
        #     json=request_data
        # )
        # return response.json()

        # Mock response for now
        return {
            "id": request_data["autellix_call_id"],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Mock response from Autellix backend"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": request_data.get("autellix_prefill_tokens", 0),
                "completion_tokens": 10,
                "total_tokens": request_data.get("autellix_prefill_tokens", 0) + 10
            },
            # Autellix-specific metadata
            "autellix_metadata": {
                "session_id": request_data["autellix_session_id"],
                "thread_id": request_data["autellix_thread_id"],
                "engine_id": "engine_0"
            }
        }

    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        return f"call_{uuid.uuid4().hex[:16]}"

    def _generate_thread_id(self) -> str:
        """Generate unique thread ID."""
        return f"thread_{uuid.uuid4().hex[:12]}"

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate number of tokens in messages.

        This is a simplified estimate. In production, you would use
        the actual tokenizer for the model.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Rough estimate: ~4 characters per token
        return total_chars // 4


@contextmanager
def autellix_session(
    backend_url: str = "http://localhost:8000",
    is_multithreaded: bool = False,
    session_metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for Autellix sessions.

    Automatically handles session start and cleanup.

    Example:
        ```python
        with autellix_session(is_multithreaded=True) as client:
            response = client.chat_completion(
                messages=[{"role": "user", "content": "Hello!"}]
            )
        # Session automatically closed
        ```

    Args:
        backend_url: URL of Autellix backend
        is_multithreaded: Whether to use ATLAS (True) or PLAS (False)
        session_metadata: Optional session metadata

    Yields:
        AutellixClient instance
    """
    client = AutellixClient(
        backend_url=backend_url,
        is_multithreaded=is_multithreaded,
        session_metadata=session_metadata,
        auto_start_session=True
    )
    try:
        yield client
    finally:
        client.close()


class AutellixOpenAIAdapter:
    """
    Adapter that makes AutellixClient compatible with OpenAI's API.

    This allows existing code using OpenAI's client to work with minimal changes.

    Example:
        ```python
        # Instead of:
        # from openai import OpenAI
        # client = OpenAI()

        # Use:
        from autellix import AutellixOpenAIAdapter
        client = AutellixOpenAIAdapter(backend_url="http://localhost:8000")

        # Same interface
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        ```
    """

    class ChatCompletions:
        """Chat completions endpoint adapter."""

        def __init__(self, client: AutellixClient):
            self.client = client

        def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 1.0,
            max_tokens: Optional[int] = None,
            stream: bool = False,
            **kwargs
        ):
            """Create a chat completion (OpenAI-compatible interface)."""
            return self.client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )

    class Chat:
        """Chat endpoint adapter."""

        def __init__(self, client: AutellixClient):
            self.completions = AutellixOpenAIAdapter.ChatCompletions(client)

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        is_multithreaded: bool = False,
        **kwargs
    ):
        """
        Initialize OpenAI-compatible adapter.

        Args:
            backend_url: Autellix backend URL
            is_multithreaded: Whether to use ATLAS
            **kwargs: Additional arguments passed to AutellixClient
        """
        self._client = AutellixClient(
            backend_url=backend_url,
            is_multithreaded=is_multithreaded,
            **kwargs
        )
        self.chat = self.Chat(self._client)

    def close(self):
        """Close the underlying client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._client.__exit__(exc_type, exc_val, exc_tb)
