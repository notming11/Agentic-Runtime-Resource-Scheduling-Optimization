"""
Engine Process Wrapper for Autellix RSM

This module implements a wrapper around vLLM AsyncLLMEngine instances
to enable multi-engine orchestration through process-based isolation.

Each EngineProcess runs in a separate Python process and communicates
via queues for request submission and result collection.
"""

import asyncio
import multiprocessing as mp
import time
import uuid
import threading
from typing import Dict, Optional, List, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import queue


class EngineStatus(Enum):
    """Engine process status"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class LLMRequest:
    """
    Request to be processed by an LLM engine.

    Attributes:
        request_id: Unique request identifier
        session_id: Session/program identifier
        prompt: Input prompt
        sampling_params: Sampling parameters (temperature, max_tokens, etc.)
        stream: Whether to stream results
        priority: Request priority (for queue ordering)
    """
    request_id: str
    session_id: str
    prompt: str
    sampling_params: Dict[str, Any]
    stream: bool = False
    priority: float = 0.0


@dataclass
class LLMResponse:
    """
    Response from an LLM engine.

    Attributes:
        request_id: Request identifier
        session_id: Session identifier
        text: Generated text
        tokens: List of generated tokens (if available)
        finish_reason: Reason for completion (e.g., "stop", "length")
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        error: Error message (if failed)
        latency: Time taken to generate (seconds)
    """
    request_id: str
    session_id: str
    text: str
    tokens: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    latency: float = 0.0


@dataclass
class EngineConfig:
    """
    Configuration for an engine process.

    Attributes:
        engine_id: Unique engine identifier
        model: Model name/path
        gpu_id: GPU device ID (for CUDA_VISIBLE_DEVICES)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum model context length
        max_num_seqs: Maximum number of concurrent sequences
        port: Optional port for OpenAI-compatible server
        trust_remote_code: Whether to trust remote code
        dtype: Model dtype (e.g., "float16", "bfloat16")
    """
    engine_id: str
    model: str
    gpu_id: int = 0
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    max_num_seqs: int = 256
    port: Optional[int] = None
    trust_remote_code: bool = False
    dtype: str = "auto"


class EngineProcess:
    """
    Wrapper around vLLM AsyncLLMEngine that runs in a separate process.

    This class provides:
    - Process-based isolation for multiple engines
    - Request queue management
    - Result collection
    - Cancellation support
    - Status monitoring

    The engine runs in its own process and communicates via multiprocessing queues.
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize the engine process.

        Args:
            config: Engine configuration
        """
        self.config = config
        self.engine_id = config.engine_id

        # Communication queues
        self.request_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.status_queue: mp.Queue = mp.Queue()

        # Process handle
        self.process: Optional[mp.Process] = None

        # Cancellation tracking (shared across processes)
        self._manager = mp.Manager()
        self.cancel_set: Dict[str, bool] = self._manager.dict()

        # Engine metrics (shared)
        self._metrics = self._manager.dict()
        self._metrics.update({
            "requests_processed": 0,
            "requests_cancelled": 0,
            "requests_failed": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "average_latency": 0.0,
            "queue_depth": 0
        })

        # Local state
        self._status = EngineStatus.INITIALIZING
        self._lock = threading.Lock()

    def start(self):
        """
        Start the engine process.

        This spawns a new process that initializes the vLLM engine
        and begins processing requests.
        """
        if self.process and self.process.is_alive():
            raise RuntimeError(f"Engine {self.engine_id} is already running")

        # Create and start process
        self.process = mp.Process(
            target=self._run_engine_loop,
            args=(
                self.config,
                self.request_queue,
                self.result_queue,
                self.status_queue,
                self.cancel_set,
                self._metrics
            ),
            daemon=False
        )
        self.process.start()

        # Wait for engine to be ready
        self._wait_for_ready(timeout=300)  # 5 minutes for model loading

    def stop(self, timeout: float = 30.0):
        """
        Stop the engine process gracefully.

        Args:
            timeout: Maximum time to wait for shutdown (seconds)
        """
        if not self.process:
            return

        # Send stop signal
        try:
            self.request_queue.put(("STOP", None), timeout=5.0)
        except queue.Full:
            pass

        # Wait for process to terminate
        self.process.join(timeout=timeout)

        if self.process.is_alive():
            # Force terminate
            self.process.terminate()
            self.process.join(timeout=5.0)

        self._status = EngineStatus.STOPPED

    def submit_request(
        self,
        request: LLMRequest
    ) -> bool:
        """
        Submit a request to the engine.

        Args:
            request: LLM request

        Returns:
            True if request was queued, False otherwise
        """
        if self._status != EngineStatus.READY:
            return False

        try:
            self.request_queue.put(("REQUEST", request), timeout=1.0)
            self._metrics["queue_depth"] += 1
            return True
        except queue.Full:
            return False

    def cancel_request(self, request_id: str):
        """
        Cancel a request.

        Adds the request ID to the cancellation set. The engine process
        will check this set and abort the request if it's still running.

        Args:
            request_id: Request identifier
        """
        self.cancel_set[request_id] = True

    def get_result(self, timeout: float = 0.1) -> Optional[LLMResponse]:
        """
        Get a result from the result queue (non-blocking).

        Args:
            timeout: Maximum time to wait for result (seconds)

        Returns:
            LLMResponse if available, None otherwise
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_status(self) -> EngineStatus:
        """
        Get current engine status.

        Returns:
            EngineStatus
        """
        # Check for status updates from process
        try:
            while True:
                status = self.status_queue.get_nowait()
                self._status = status
        except queue.Empty:
            pass

        return self._status

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get engine metrics.

        Returns:
            Dictionary with metrics
        """
        return dict(self._metrics)

    def is_alive(self) -> bool:
        """
        Check if engine process is alive.

        Returns:
            True if process is running
        """
        return self.process is not None and self.process.is_alive()

    def _wait_for_ready(self, timeout: float = 300.0):
        """
        Wait for engine to become ready.

        Args:
            timeout: Maximum time to wait (seconds)

        Raises:
            TimeoutError: If engine doesn't become ready in time
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            if status == EngineStatus.READY:
                return
            elif status == EngineStatus.ERROR:
                raise RuntimeError(f"Engine {self.engine_id} failed to initialize")
            time.sleep(1.0)

        raise TimeoutError(f"Engine {self.engine_id} didn't become ready in {timeout}s")

    @staticmethod
    def _run_engine_loop(
        config: EngineConfig,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        status_queue: mp.Queue,
        cancel_set: Dict[str, bool],
        metrics: Dict[str, Any]
    ):
        """
        Main loop that runs in the engine process.

        This function:
        1. Initializes the vLLM engine
        2. Processes requests from the queue
        3. Handles cancellations
        4. Sends results back

        Args:
            config: Engine configuration
            request_queue: Queue for incoming requests
            result_queue: Queue for outgoing results
            status_queue: Queue for status updates
            cancel_set: Set of cancelled request IDs
            metrics: Shared metrics dictionary
        """
        # Set GPU device
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        # Initialize engine
        try:
            # Try to import vLLM
            try:
                from vllm import AsyncLLMEngine, SamplingParams
                from vllm.engine.arg_utils import AsyncEngineArgs
                has_vllm = True
            except ImportError:
                has_vllm = False

            if has_vllm:
                # Real vLLM engine
                engine_args = AsyncEngineArgs(
                    model=config.model,
                    tensor_parallel_size=config.tensor_parallel_size,
                    max_model_len=config.max_model_len,
                    max_num_seqs=config.max_num_seqs,
                    trust_remote_code=config.trust_remote_code,
                    dtype=config.dtype
                )
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                status_queue.put(EngineStatus.READY)
            else:
                # Mock engine for testing (when vLLM not available)
                engine = None
                status_queue.put(EngineStatus.READY)

        except Exception as e:
            status_queue.put(EngineStatus.ERROR)
            result_queue.put(LLMResponse(
                request_id="error",
                session_id="error",
                text="",
                error=f"Failed to initialize engine: {str(e)}"
            ))
            return

        # Main processing loop
        asyncio.run(EngineProcess._process_requests(
            engine,
            config,
            request_queue,
            result_queue,
            status_queue,
            cancel_set,
            metrics,
            has_vllm
        ))

    @staticmethod
    async def _process_requests(
        engine: Any,
        config: EngineConfig,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        status_queue: mp.Queue,
        cancel_set: Dict[str, bool],
        metrics: Dict[str, Any],
        has_vllm: bool
    ):
        """
        Async loop for processing requests.

        Args:
            engine: vLLM AsyncLLMEngine instance (or None for mock)
            config: Engine configuration
            request_queue: Queue for incoming requests
            result_queue: Queue for outgoing results
            status_queue: Queue for status updates
            cancel_set: Set of cancelled request IDs
            metrics: Shared metrics dictionary
            has_vllm: Whether vLLM is available
        """
        active_requests: Dict[str, asyncio.Task] = {}

        while True:
            # Check for new requests (non-blocking)
            try:
                msg_type, data = request_queue.get_nowait()

                if msg_type == "STOP":
                    # Graceful shutdown
                    break
                elif msg_type == "REQUEST":
                    request: LLMRequest = data
                    metrics["queue_depth"] = max(0, metrics["queue_depth"] - 1)

                    # Check if already cancelled
                    if request.request_id in cancel_set:
                        del cancel_set[request.request_id]
                        metrics["requests_cancelled"] += 1
                        result_queue.put(LLMResponse(
                            request_id=request.request_id,
                            session_id=request.session_id,
                            text="",
                            finish_reason="cancelled"
                        ))
                        continue

                    # Process request
                    task = asyncio.create_task(
                        EngineProcess._process_single_request(
                            engine,
                            request,
                            result_queue,
                            cancel_set,
                            metrics,
                            has_vllm
                        )
                    )
                    active_requests[request.request_id] = task

            except queue.Empty:
                pass

            # Check for cancelled requests
            for request_id in list(cancel_set):
                if request_id in active_requests:
                    # Cancel the task
                    active_requests[request_id].cancel()
                    del active_requests[request_id]
                    del cancel_set[request_id]
                    metrics["requests_cancelled"] += 1

            # Update status
            if active_requests:
                status_queue.put(EngineStatus.BUSY)
            else:
                status_queue.put(EngineStatus.READY)

            # Small delay to prevent busy-waiting
            await asyncio.sleep(0.01)

        # Cleanup: cancel all active requests
        for task in active_requests.values():
            task.cancel()

    @staticmethod
    async def _process_single_request(
        engine: Any,
        request: LLMRequest,
        result_queue: mp.Queue,
        cancel_set: Set[str],
        metrics: Dict[str, Any],
        has_vllm: bool
    ):
        """
        Process a single LLM request.

        Args:
            engine: vLLM AsyncLLMEngine instance (or None for mock)
            request: LLM request
            result_queue: Queue for results
            cancel_set: Set of cancelled request IDs
            metrics: Shared metrics dictionary
            has_vllm: Whether vLLM is available
        """
        start_time = time.time()

        try:
            if has_vllm and engine:
                # Real vLLM processing
                from vllm import SamplingParams

                # Convert sampling params
                sampling_params = SamplingParams(**request.sampling_params)

                # Generate
                request_id = request.request_id
                results = []

                async for output in engine.generate(
                    request.prompt,
                    sampling_params,
                    request_id
                ):
                    # Check for cancellation
                    if request_id in cancel_set:
                        await engine.abort(request_id)
                        del cancel_set[request_id]
                        return

                    results.append(output)

                # Get final output
                if results:
                    final_output = results[-1]
                    generated_text = final_output.outputs[0].text
                    finish_reason = final_output.outputs[0].finish_reason
                    completion_tokens = len(final_output.outputs[0].token_ids)
                    prompt_tokens = len(final_output.prompt_token_ids)
                else:
                    generated_text = ""
                    finish_reason = "error"
                    completion_tokens = 0
                    prompt_tokens = 0

            else:
                # Mock processing (for testing without vLLM)
                await asyncio.sleep(0.1)  # Simulate processing time
                generated_text = f"Mock response for: {request.prompt[:50]}..."
                finish_reason = "stop"
                completion_tokens = 10
                prompt_tokens = len(request.prompt.split())

            # Create response
            latency = time.time() - start_time
            response = LLMResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                text=generated_text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=latency
            )

            # Update metrics
            metrics["requests_processed"] += 1
            metrics["total_prompt_tokens"] += prompt_tokens
            metrics["total_completion_tokens"] += completion_tokens

            # Update average latency (exponential moving average)
            alpha = 0.1
            metrics["average_latency"] = (
                alpha * latency +
                (1 - alpha) * metrics["average_latency"]
            )

            # Send result
            result_queue.put(response)

        except asyncio.CancelledError:
            # Request was cancelled
            result_queue.put(LLMResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                text="",
                finish_reason="cancelled"
            ))
            raise

        except Exception as e:
            # Request failed
            metrics["requests_failed"] += 1
            result_queue.put(LLMResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                text="",
                error=str(e)
            ))


def create_engine_process(
    model: str,
    engine_id: Optional[str] = None,
    gpu_id: int = 0,
    **kwargs
) -> EngineProcess:
    """
    Factory function to create an engine process with sensible defaults.

    Args:
        model: Model name/path
        engine_id: Engine identifier (generated if None)
        gpu_id: GPU device ID
        **kwargs: Additional engine configuration

    Returns:
        EngineProcess instance
    """
    if engine_id is None:
        engine_id = f"engine_{uuid.uuid4().hex[:8]}"

    config = EngineConfig(
        engine_id=engine_id,
        model=model,
        gpu_id=gpu_id,
        **kwargs
    )

    return EngineProcess(config)
