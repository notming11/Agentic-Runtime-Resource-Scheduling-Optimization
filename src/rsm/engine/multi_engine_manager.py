"""
Multi-Engine Manager for Autellix RSM

This module implements the orchestration layer that coordinates multiple vLLM
engine instances, handles request routing, and manages the complete lifecycle
of LLM calls.

Key responsibilities:
- Manage multiple EngineProcess instances
- Route requests from load balancer to appropriate engines
- Collect and aggregate results asynchronously
- Handle cancellation across engines
- Provide monitoring and status reporting
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import concurrent.futures

from .engine_process import (
    EngineProcess,
    EngineConfig,
    EngineStatus,
    LLMRequest,
    LLMResponse,
    create_engine_process
)
from .kv_cache_coordinator import KVCacheCoordinator
from .lifecycle_manager import RequestLifecycleManager, RequestState


@dataclass
class EngineLoad:
    """
    Represents the current load on an engine.

    Attributes:
        engine_id: Engine identifier
        queue_depth: Number of requests in queue
        active_requests: Number of currently executing requests
        requests_per_second: Recent throughput
        average_latency: Average response latency
        available: Whether engine is available for new requests
    """
    engine_id: str
    queue_depth: int = 0
    active_requests: int = 0
    requests_per_second: float = 0.0
    average_latency: float = 0.0
    available: bool = True

    def get_load_score(self) -> float:
        """
        Calculate load score for routing decisions.

        Lower score = less loaded.
        """
        if not self.available:
            return float('inf')

        # Weighted combination of queue depth and active requests
        # Queue depth is more important as it represents backlog
        return (2.0 * self.queue_depth + self.active_requests)


class MultiEngineManager:
    """
    Manages multiple vLLM engine instances and coordinates request routing.

    This is the main orchestration component that:
    1. Starts/stops multiple engine processes
    2. Routes requests to appropriate engines using load balancing and cache affinity
    3. Collects results asynchronously
    4. Handles request cancellation
    5. Provides monitoring and statistics

    Integration:
    - Receives requests from frontend/scheduler
    - Uses KVCacheCoordinator for routing decisions
    - Uses RequestLifecycleManager for state tracking
    - Returns results via futures/callbacks
    """

    def __init__(
        self,
        engine_configs: Optional[List[EngineConfig]] = None,
        cache_token_threshold: int = 2048
    ):
        """
        Initialize the multi-engine manager.

        Args:
            engine_configs: List of engine configurations (can add later)
            cache_token_threshold: Token threshold for cache locality routing
        """
        # Engine management
        self.engines: Dict[str, EngineProcess] = {}
        self.engine_loads: Dict[str, EngineLoad] = {}

        # Coordinators
        self.kv_cache = KVCacheCoordinator(cache_token_threshold)
        self.lifecycle = RequestLifecycleManager()

        # Request routing
        # request_id -> engine_id
        self._request_map: Dict[str, str] = {}

        # Result futures
        # request_id -> Future
        self._result_futures: Dict[str, asyncio.Future] = {}

        # Result callbacks
        # request_id -> callback
        self._result_callbacks: Dict[str, Callable] = {}

        # Background tasks
        self._result_collector_task: Optional[asyncio.Task] = None
        self._load_monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Thread safety
        self._lock = threading.RLock()

        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "cancelled_requests": 0,
            "total_latency": 0.0
        }

        # Add initial engines if provided
        if engine_configs:
            for config in engine_configs:
                self.add_engine(config)

    def start(self):
        """
        Start the multi-engine manager.

        This starts all engine processes and background tasks.
        """
        if self._running:
            return

        self._running = True

        # Start event loop in separate thread
        self._start_event_loop()

        # Start all engines
        for engine in self.engines.values():
            if not engine.is_alive():
                engine.start()

        # Start background tasks
        self._start_background_tasks()

    def stop(self, timeout: float = 30.0):
        """
        Stop the multi-engine manager gracefully.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self._running:
            return

        self._running = False

        # Stop background tasks
        if self._loop:
            # Cancel tasks
            if self._result_collector_task:
                self._result_collector_task.cancel()
            if self._load_monitor_task:
                self._load_monitor_task.cancel()

        # Stop all engines
        for engine in self.engines.values():
            engine.stop(timeout=timeout / len(self.engines))

        # Stop event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)

    def add_engine(self, config: EngineConfig) -> bool:
        """
        Add a new engine to the manager.

        Args:
            config: Engine configuration

        Returns:
            True if engine was added successfully
        """
        with self._lock:
            if config.engine_id in self.engines:
                return False

            # Create engine process
            engine = EngineProcess(config)
            self.engines[config.engine_id] = engine

            # Initialize load tracking
            self.engine_loads[config.engine_id] = EngineLoad(
                engine_id=config.engine_id
            )

            # Register with KV cache coordinator
            self.kv_cache.register_engine(config.engine_id)

            # Start engine if manager is running
            if self._running:
                engine.start()

            return True

    def remove_engine(self, engine_id: str, timeout: float = 30.0) -> bool:
        """
        Remove an engine from the manager.

        Args:
            engine_id: Engine identifier
            timeout: Maximum time to wait for shutdown

        Returns:
            True if engine was removed successfully
        """
        with self._lock:
            if engine_id not in self.engines:
                return False

            # Stop engine
            engine = self.engines[engine_id]
            engine.stop(timeout=timeout)

            # Remove from tracking
            del self.engines[engine_id]
            del self.engine_loads[engine_id]

            # Unregister from KV cache coordinator
            self.kv_cache.unregister_engine(engine_id)

            return True

    async def submit_request(
        self,
        session_id: str,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        priority: float = 0.0,
        callback: Optional[Callable[[LLMResponse], None]] = None
    ) -> asyncio.Future:
        """
        Submit a request for processing.

        Args:
            session_id: Session/program identifier
            prompt: Input prompt
            sampling_params: Sampling parameters (temperature, max_tokens, etc.)
            request_id: Request identifier (generated if None)
            priority: Request priority
            callback: Optional callback for result

        Returns:
            Future that will resolve to LLMResponse
        """
        # Generate request ID if needed
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        # Default sampling params
        if sampling_params is None:
            sampling_params = {
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9
            }

        # Count prompt tokens (rough estimate)
        prompt_tokens = len(prompt.split())

        # Select engine using load balancer and cache affinity
        engine_id = self._select_engine(session_id, prompt_tokens)
        if not engine_id:
            raise RuntimeError("No available engines")

        # Create request
        request = LLMRequest(
            request_id=request_id,
            session_id=session_id,
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority
        )

        # Register with lifecycle manager
        self.lifecycle.start_request(
            request_id=request_id,
            session_id=session_id,
            engine_id=engine_id,
            prompt_tokens=prompt_tokens
        )

        # Create future for result
        future = asyncio.Future()
        self._result_futures[request_id] = future

        if callback:
            self._result_callbacks[request_id] = callback

        # Track routing
        self._request_map[request_id] = engine_id

        # Submit to engine
        engine = self.engines[engine_id]
        success = engine.submit_request(request)

        if not success:
            # Failed to submit
            self.lifecycle.fail_request(request_id, "Failed to submit to engine")
            future.set_exception(RuntimeError("Failed to submit to engine"))
            return future

        # Update state
        self.lifecycle.update_request_state(
            request_id,
            RequestState.RUNNING,
            engine_id=engine_id
        )

        # Record cache usage
        self.kv_cache.record_cache_usage(
            session_id=session_id,
            engine_id=engine_id,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt
        )

        # Update stats
        self._stats["total_requests"] += 1

        return future

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a request.

        Args:
            request_id: Request identifier

        Returns:
            True if request was cancelled
        """
        with self._lock:
            # Get engine
            engine_id = self._request_map.get(request_id)
            if not engine_id or engine_id not in self.engines:
                return False

            # Cancel in lifecycle manager
            if not self.lifecycle.cancel_request(request_id):
                return False

            # Cancel in engine
            engine = self.engines[engine_id]
            engine.cancel_request(request_id)

            # Resolve future
            if request_id in self._result_futures:
                future = self._result_futures[request_id]
                if not future.done():
                    future.cancel()

            # Update stats
            self._stats["cancelled_requests"] += 1

            return True

    async def cancel_session(self, session_id: str) -> List[str]:
        """
        Cancel all requests for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of cancelled request IDs
        """
        cancelled = []
        requests = self.lifecycle.get_session_requests(session_id, active_only=True)

        for request_metadata in requests:
            if await self.cancel_request(request_metadata.request_id):
                cancelled.append(request_metadata.request_id)

        return cancelled

    def get_engine_status(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status for a specific engine.

        Args:
            engine_id: Engine identifier

        Returns:
            Dictionary with engine status, or None if not found
        """
        with self._lock:
            if engine_id not in self.engines:
                return None

            engine = self.engines[engine_id]
            load = self.engine_loads.get(engine_id)

            return {
                "engine_id": engine_id,
                "status": engine.get_status().value,
                "is_alive": engine.is_alive(),
                "load": asdict(load) if load else None,
                "metrics": engine.get_metrics(),
                "active_requests": len(self.lifecycle.get_engine_requests(engine_id))
            }

    def get_all_engine_status(self) -> List[Dict[str, Any]]:
        """
        Get status for all engines.

        Returns:
            List of engine status dictionaries
        """
        return [
            self.get_engine_status(engine_id)
            for engine_id in self.engines.keys()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall manager statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            avg_latency = 0.0
            if self._stats["completed_requests"] > 0:
                avg_latency = (
                    self._stats["total_latency"] /
                    self._stats["completed_requests"]
                )

            return {
                **self._stats,
                "average_latency": avg_latency,
                "active_engines": sum(
                    1 for e in self.engines.values() if e.is_alive()
                ),
                "total_engines": len(self.engines),
                "lifecycle_stats": self.lifecycle.get_stats(),
                "cache_stats": self.kv_cache.get_cache_stats()
            }

    def _select_engine(
        self,
        session_id: str,
        prompt_tokens: int
    ) -> Optional[str]:
        """
        Select best engine for a request using hybrid routing policy.

        Implements Autellix's routing policy:
        - Short calls (≤ threshold): Least-loaded engine
        - Long calls (> threshold): Engine with cache affinity, or least-loaded

        Args:
            session_id: Session identifier
            prompt_tokens: Number of prompt tokens

        Returns:
            Selected engine ID, or None if no engines available
        """
        with self._lock:
            if not self.engines:
                return None

            # Get available engines
            available_engines = [
                eid for eid, e in self.engines.items()
                if e.is_alive() and self.engine_loads[eid].available
            ]

            if not available_engines:
                return None

            # Try cache affinity for long calls
            affinity_engine = self.kv_cache.get_best_engine_for_session(
                session_id,
                available_engines,
                prompt_tokens
            )

            if affinity_engine:
                return affinity_engine

            # Fall back to least-loaded
            return self._get_least_loaded_engine(available_engines)

    def _get_least_loaded_engine(
        self,
        available_engines: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Get the least loaded engine.

        Args:
            available_engines: List of engines to consider (all if None)

        Returns:
            Engine ID with lowest load
        """
        if available_engines is None:
            available_engines = list(self.engines.keys())

        if not available_engines:
            return None

        # Find engine with lowest load score
        best_engine = None
        best_score = float('inf')

        for engine_id in available_engines:
            if engine_id not in self.engine_loads:
                continue

            load = self.engine_loads[engine_id]
            score = load.get_load_score()

            if score < best_score:
                best_score = score
                best_engine = engine_id

        return best_engine

    def _start_event_loop(self):
        """Start asyncio event loop in separate thread"""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be ready
        while self._loop is None:
            time.sleep(0.01)

    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        if not self._loop:
            return

        # Result collector
        self._result_collector_task = asyncio.run_coroutine_threadsafe(
            self._collect_results(),
            self._loop
        ).result()

        # Load monitor
        self._load_monitor_task = asyncio.run_coroutine_threadsafe(
            self._monitor_loads(),
            self._loop
        ).result()

    async def _collect_results(self):
        """Background task to collect results from engines"""
        while self._running:
            for engine_id, engine in list(self.engines.items()):
                # Get result (non-blocking)
                result = engine.get_result(timeout=0.01)

                if result:
                    await self._handle_result(result)

            await asyncio.sleep(0.01)

    async def _handle_result(self, result: LLMResponse):
        """
        Handle a result from an engine.

        Args:
            result: LLM response
        """
        request_id = result.request_id

        # Update lifecycle
        if result.error:
            self.lifecycle.fail_request(request_id, result.error)
            self._stats["failed_requests"] += 1
        elif result.finish_reason == "cancelled":
            self.lifecycle.cancel_request(request_id)
            self._stats["cancelled_requests"] += 1
        else:
            self.lifecycle.complete_request(
                request_id,
                completion_tokens=result.completion_tokens
            )
            self._stats["completed_requests"] += 1
            self._stats["total_latency"] += result.latency

        # Resolve future
        if request_id in self._result_futures:
            future = self._result_futures[request_id]
            if not future.done():
                if result.error:
                    future.set_exception(RuntimeError(result.error))
                else:
                    future.set_result(result)
            del self._result_futures[request_id]

        # Call callback
        if request_id in self._result_callbacks:
            callback = self._result_callbacks[request_id]
            try:
                callback(result)
            except Exception:
                pass  # Ignore callback errors
            del self._result_callbacks[request_id]

        # Cleanup routing map
        if request_id in self._request_map:
            del self._request_map[request_id]

    async def _monitor_loads(self):
        """Background task to monitor engine loads"""
        while self._running:
            for engine_id, engine in list(self.engines.items()):
                # Get metrics
                metrics = engine.get_metrics()
                status = engine.get_status()

                # Update load
                load = self.engine_loads[engine_id]
                load.queue_depth = metrics.get("queue_depth", 0)
                load.active_requests = len(
                    self.lifecycle.get_engine_requests(engine_id)
                )
                load.average_latency = metrics.get("average_latency", 0.0)
                load.available = (
                    status == EngineStatus.READY or
                    status == EngineStatus.BUSY
                )

                # Calculate requests per second (approximate)
                # This would need more sophisticated tracking in production
                load.requests_per_second = metrics.get("requests_processed", 0) / max(
                    time.time() - engine.config.port if hasattr(engine.config, 'port') else 1.0,
                    1.0
                )

            await asyncio.sleep(1.0)
