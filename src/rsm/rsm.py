"""
Autellix RSM - Complete System Integration

This module integrates all four partitions into a working system:
- Partition 1: Frontend (Process Table, Session Manager, API Wrapper)
- Partition 2: Scheduler (ATLAS with PLAS/ATLAS modes)
- Partition 3: Load Balancer (Data Locality-Aware)
- Partition 4: Multi-Engine Manager (vLLM Orchestration)

Architecture:
    User Program (API Wrapper)
           ↓
    Session Manager → Process Table
           ↓
    Scheduler (ATLAS) ← Process Table
           ↓
    Load Balancer ← Process Table
           ↓
    Multi-Engine Manager
           ↓
    vLLM Engines (GPU)
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

# Partition 1: Frontend
from frontend import (
    GlobalProcessTable,
    SessionManager,
    AutellixClient,
    autellix_session
)

# Partition 2: Scheduler
import scheduler.atlas_scheduler as atlas_scheduler
from scheduler.context import ProgramContext

# Partition 3: Load Balancer
from load_balancer import LoadBalancer

# Partition 4: Multi-Engine Manager
from engine import (
    MultiEngineManager,
    RequestState
)
from engine.engine_process import EngineConfig, LLMRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AutellixConfig:
    """Configuration for the complete Autellix system."""
    
    # Scheduler config
    num_priority_levels: int = 8
    base_quantum: int = 512
    adaptation_threshold: float = 3.0
    adaptation_check_interval: int = 5000
    starvation_threshold: int = 5000
    starvation_ratio: float = 3.0
    
    # Load balancer config
    cache_token_threshold: int = 2048
    
    # Engine config
    model: str = "meta-llama/Llama-2-7b-hf"
    num_engines: int = 4
    max_num_seqs: int = 256
    max_model_len: int = 4096


class AutellixSystem:
    """
    Complete Autellix RSM system integrating all partitions.
    
    This class orchestrates the entire request lifecycle:
    1. User submits request via API wrapper
    2. Session manager registers call in process table
    3. Scheduler assigns priority based on program service time
    4. Load balancer routes to appropriate engine
    5. Engine executes and returns result
    6. System updates process table metrics
    """
    
    def __init__(self, config: AutellixConfig):
        """
        Initialize the complete Autellix system.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Partition 1: Process management
        logger.info("Initializing Partition 1: Frontend")
        self.process_table = GlobalProcessTable()
        self.session_manager = SessionManager(self.process_table)
        
        # Partition 2: Scheduler
        logger.info("Initializing Partition 2: Scheduler")
        scheduler_config = {
            'num_priority_levels': config.num_priority_levels,
            'base_quantum': config.base_quantum,
            'adaptation_threshold': config.adaptation_threshold,
            'adaptation_check_interval': config.adaptation_check_interval,
            'starvation_threshold': config.starvation_threshold,
            'starvation_ratio': config.starvation_ratio
        }
        self.scheduler = atlas_scheduler.ATLASScheduler(scheduler_config, self.process_table)
        
        # Partition 3: Load balancer
        logger.info("Initializing Partition 3: Load Balancer")
        self.load_balancer = LoadBalancer(self.process_table)
        
        # Partition 4: Multi-engine manager
        logger.info("Initializing Partition 4: Multi-Engine Manager")
        self.engine_manager = MultiEngineManager(
            cache_token_threshold=config.cache_token_threshold
        )
        
        # Register engines
        for i in range(config.num_engines):
            engine_config = EngineConfig(
                engine_id=f"engine_{i}",
                model=config.model,
                gpu_id=i,
                max_num_seqs=config.max_num_seqs,
                max_model_len=config.max_model_len
            )
            self.engine_manager.add_engine(engine_config)
            self.load_balancer.register_engine(f"engine_{i}")
        
        # Background tasks
        self._running = False
        self._scheduler_task = None
        self._starvation_task = None
        
        logger.info("Autellix system initialized successfully")
    
    def start(self):
        """Start the Autellix system."""
        if self._running:
            return
        
        logger.info("Starting Autellix system")
        self._running = True
        
        # Start engine manager
        self.engine_manager.start()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Autellix system started")
    
    def stop(self, timeout: float = 30.0):
        """Stop the Autellix system gracefully."""
        logger.info("Stopping Autellix system")
        self._running = False
        
        # Stop background tasks
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._starvation_task:
            self._starvation_task.cancel()
        
        # Stop engine manager
        self.engine_manager.stop(timeout=timeout)
        
        logger.info("Autellix system stopped")
    
    async def submit_request(
        self,
        session_id: str,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        parent_thread_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Submit a request through the complete Autellix pipeline.
        
        This demonstrates the full integration:
        1. Register with session manager
        2. Get priority from scheduler
        3. Route via load balancer
        4. Execute on engine
        5. Update metrics
        
        Args:
            session_id: Session identifier
            prompt: Input prompt
            sampling_params: Sampling parameters
            thread_id: Thread identifier (for multi-threaded programs)
            parent_thread_ids: Parent threads (for ATLAS)
        
        Returns:
            Response dictionary
        """
        # Default params
        if sampling_params is None:
            sampling_params = {
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9
            }
        
        # Estimate tokens
        prompt_tokens = len(prompt.split())
        
        # Step 1: Register call with session manager
        call_id = f"call_{time.time()}"
        if thread_id is None:
            thread_id = f"thread_{time.time()}"
        
        registered_thread = self.session_manager.register_llm_call(
            session_id=session_id,
            call_id=call_id,
            thread_id=thread_id,
            prefill_tokens=prompt_tokens,
            parent_thread_ids=parent_thread_ids
        )
        
        if not registered_thread:
            raise RuntimeError(f"Failed to register call for session {session_id}")
        
        # Step 2: Register with scheduler
        self.scheduler.on_request_arrival(
            request_id=call_id,
            program_id=session_id,
            thread_id=thread_id
        )
        
        # Step 3: Get priority from scheduler
        # (In production, scheduler would run in background)
        program_priority = self.process_table.get_program_priority(session_id)
        
        # Step 4: Route via load balancer
        engine_id = self.load_balancer.route_request(
            pid=session_id,
            num_tokens=prompt_tokens
        )
        
        if not engine_id:
            raise RuntimeError("No available engines")
        
        logger.info(
            f"Request {call_id} routed to {engine_id} "
            f"(priority={program_priority}, tokens={prompt_tokens})"
        )
        
        # Step 5: Submit to engine manager
        start_time = time.time()
        
        future = await self.engine_manager.submit_request(
            session_id=session_id,
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=call_id
        )
        
        # Wait for result
        result = await future
        
        # Step 6: Update metrics
        service_time = time.time() - start_time
        waiting_time = result.latency - service_time  # Approximation
        
        # Update scheduler
        self.scheduler.update_service(
            request_id=call_id,
            tokens_processed=result.completion_tokens
        )
        
        # Update session manager
        self.session_manager.complete_llm_call(
            session_id=session_id,
            thread_id=thread_id,
            service_time=service_time,
            waiting_time=max(0, waiting_time)
        )
        
        # Update load balancer
        self.load_balancer.complete_request(engine_id, session_id)
        
        logger.info(
            f"Request {call_id} completed: "
            f"service={service_time:.2f}s, tokens={result.completion_tokens}"
        )
        
        return {
            "text": result.text,
            "tokens": result.completion_tokens,
            "latency": result.latency,
            "engine_id": engine_id,
            "priority": program_priority,
            "metadata": {
                "session_id": session_id,
                "thread_id": thread_id,
                "service_time": service_time,
                "waiting_time": waiting_time
            }
        }
    
    def _start_background_tasks(self):
        """Start background monitoring and scheduling tasks."""
        # Anti-starvation check
        async def check_starvation():
            while self._running:
                self.scheduler.check_and_promote_starved_programs()
                await asyncio.sleep(1.0)
        
        self._starvation_task = asyncio.create_task(check_starvation())
        
        logger.info("Background tasks started")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "frontend": {
                "process_table": self.process_table.get_stats(),
                "sessions": self.session_manager.get_global_stats()
            },
            "scheduler": self.scheduler.queue_manager.get_statistics(),
            "load_balancer": self.load_balancer.get_stats(),
            "engines": self.engine_manager.get_stats()
        }
    
    def create_client(
        self,
        is_multithreaded: bool = False,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> 'IntegratedAutellixClient':
        """
        Create an API client connected to this system.
        
        Args:
            is_multithreaded: Whether to use ATLAS scheduling
            session_metadata: Optional session metadata
        
        Returns:
            IntegratedAutellixClient instance
        """
        return IntegratedAutellixClient(
            system=self,
            is_multithreaded=is_multithreaded,
            session_metadata=session_metadata
        )


class IntegratedAutellixClient:
    """
    Client that integrates with the complete Autellix system.
    
    This provides the same API as AutellixClient but routes requests
    through the complete integrated system.
    """
    
    def __init__(
        self,
        system: AutellixSystem,
        is_multithreaded: bool = False,
        session_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize integrated client.
        
        Args:
            system: AutellixSystem instance
            is_multithreaded: Whether to use ATLAS scheduling
            session_metadata: Optional session metadata
        """
        self.system = system
        self.is_multithreaded = is_multithreaded
        
        # Start session
        self.session_id = system.session_manager.start_session(
            is_multithreaded=is_multithreaded,
            metadata=session_metadata
        )
        
        logger.info(f"Started session {self.session_id}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
        thread_id: Optional[str] = None,
        parent_thread_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request through integrated system.
        
        Args:
            messages: Message list
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            thread_id: Thread ID
            parent_thread_ids: Parent threads
            **kwargs: Additional parameters
        
        Returns:
            Response dictionary
        """
        # Combine messages into prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        sampling_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        return await self.system.submit_request(
            session_id=self.session_id,
            prompt=prompt,
            sampling_params=sampling_params,
            thread_id=thread_id,
            parent_thread_ids=parent_thread_ids
        )
    
    async def parallel_chat_completion(
        self,
        requests: List[Dict[str, Any]],
        parent_thread_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Submit multiple requests in parallel.
        
        Args:
            requests: List of request dictionaries
            parent_thread_id: Common parent thread
        
        Returns:
            List of responses
        """
        tasks = []
        parent_ids = [parent_thread_id] if parent_thread_id else None
        
        for req in requests:
            task = self.chat_completion(
                parent_thread_ids=parent_ids,
                **req
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self.system.session_manager.get_session_stats(self.session_id)
    
    def close(self):
        """Close the session."""
        self.system.session_manager.end_session(self.session_id)
        logger.info(f"Closed session {self.session_id}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()


# ============================================================================
# Example Usage Patterns
# ============================================================================

async def example_single_threaded():
    """Example: Single-threaded chatbot (PLAS scheduling)."""
    print("\n" + "="*80)
    print("Example 1: Single-Threaded Chatbot (PLAS)")
    print("="*80)
    
    # Initialize system
    config = AutellixConfig(num_engines=2)
    system = AutellixSystem(config)
    system.start()
    
    try:
        # Create client
        async with system.create_client(is_multithreaded=False) as client:
            # Make sequential calls
            for i in range(3):
                response = await client.chat_completion(
                    messages=[
                        {"role": "user", "content": f"Question {i+1}: What is AI?"}
                    ],
                    max_tokens=100
                )
                print(f"\nResponse {i+1}:")
                print(f"  Text: {response['text'][:100]}...")
                print(f"  Latency: {response['latency']:.3f}s")
                print(f"  Engine: {response['engine_id']}")
            
            # Show stats
            stats = client.get_session_stats()
            print(f"\nSession Stats:")
            print(f"  Service time: {stats['service_time']:.2f}s")
            print(f"  Total calls: {stats['total_calls']}")
    
    finally:
        system.stop()


async def example_multi_threaded():
    """Example: Multi-threaded agent (ATLAS scheduling)."""
    print("\n" + "="*80)
    print("Example 2: Multi-Threaded Research Agent (ATLAS)")
    print("="*80)
    
    # Initialize system
    config = AutellixConfig(num_engines=4)
    system = AutellixSystem(config)
    system.start()
    
    try:
        async with system.create_client(is_multithreaded=True) as client:
            # Step 1: Planning
            print("\nStep 1: Planning")
            plan_response = await client.chat_completion(
                messages=[
                    {"role": "user", "content": "Plan research on 3 cities"}
                ]
            )
            plan_thread = plan_response['metadata']['thread_id']
            print(f"  Planned on {plan_response['engine_id']}")
            
            # Step 2: Parallel research (Map)
            print("\nStep 2: Parallel Research")
            research_requests = [
                {
                    "messages": [
                        {"role": "user", "content": f"Research city {i}"}
                    ]
                }
                for i in range(3)
            ]
            
            research_responses = await client.parallel_chat_completion(
                requests=research_requests,
                parent_thread_id=plan_thread
            )
            
            for i, resp in enumerate(research_responses):
                print(f"  Research {i}: {resp['engine_id']}, "
                      f"latency={resp['latency']:.3f}s")
            
            # Step 3: Aggregation (Reduce)
            print("\nStep 3: Aggregation")
            thread_ids = [r['metadata']['thread_id'] for r in research_responses]
            final_response = await client.chat_completion(
                messages=[
                    {"role": "user", "content": "Aggregate research results"}
                ],
                parent_thread_ids=thread_ids
            )
            print(f"  Aggregated on {final_response['engine_id']}")
            
            # Show stats
            stats = client.get_session_stats()
            print(f"\nSession Stats:")
            print(f"  Service time: {stats['service_time']:.2f}s")
            print(f"  Total calls: {stats['total_calls']}")
            print(f"  Active threads: {stats['active_threads']}")
    
    finally:
        system.stop()


async def example_load_distribution():
    """Example: Load distribution across engines."""
    print("\n" + "="*80)
    print("Example 3: Load Distribution")
    print("="*80)
    
    # Initialize system
    config = AutellixConfig(num_engines=4)
    system = AutellixSystem(config)
    system.start()
    
    try:
        # Create multiple clients
        clients = [
            system.create_client(is_multithreaded=False)
            for _ in range(3)
        ]
        
        # Submit requests from different sessions
        tasks = []
        for i, client in enumerate(clients):
            for j in range(2):
                task = client.chat_completion(
                    messages=[
                        {"role": "user", 
                         "content": f"Session {i}, Request {j}"}
                    ],
                    max_tokens=100
                )
                tasks.append(task)
        
        # Wait for all
        responses = await asyncio.gather(*tasks)
        
        # Show distribution
        engine_counts = {}
        for resp in responses:
            engine_id = resp['engine_id']
            engine_counts[engine_id] = engine_counts.get(engine_id, 0) + 1
        
        print("\nEngine Distribution:")
        for engine_id, count in sorted(engine_counts.items()):
            print(f"  {engine_id}: {count} requests")
        
        # Cleanup
        for client in clients:
            client.close()
    
    finally:
        system.stop()


async def example_system_stats():
    """Example: System-wide statistics."""
    print("\n" + "="*80)
    print("Example 4: System Statistics")
    print("="*80)
    
    # Initialize system
    config = AutellixConfig(num_engines=4)
    system = AutellixSystem(config)
    system.start()
    
    try:
        # Create client and submit requests
        async with system.create_client() as client:
            for i in range(5):
                await client.chat_completion(
                    messages=[{"role": "user", "content": f"Request {i}"}]
                )
        
        # Get system stats
        stats = system.get_system_stats()
        
        print("\nProcess Table:")
        pt_stats = stats['frontend']['process_table']
        print(f"  Total programs: {pt_stats['total_programs']}")
        print(f"  Active calls: {pt_stats['total_active_calls']}")
        print(f"  Completed calls: {pt_stats['total_completed_calls']}")
        
        print("\nScheduler:")
        sched_stats = stats['scheduler']
        print(f"  Total enqueues: {sched_stats['total_enqueues']}")
        print(f"  Total dequeues: {sched_stats['total_dequeues']}")
        
        print("\nLoad Balancer:")
        lb_stats = stats['load_balancer']
        print(f"  Total requests: {lb_stats['total_requests']}")
        print(f"  Small requests: {lb_stats['small_requests']}")
        print(f"  Large requests: {lb_stats['large_requests']}")
        print(f"  Locality hits: {lb_stats['locality_hits']}")
        
        print("\nEngines:")
        engine_stats = stats['engines']
        print(f"  Completed: {engine_stats['completed_requests']}")
        print(f"  Average latency: {engine_stats['average_latency']:.3f}s")
    
    finally:
        system.stop()


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("AUTELLIX RSM - COMPLETE SYSTEM INTEGRATION")
    print("="*80)
    
    await example_single_threaded()
    await example_multi_threaded()
    await example_load_distribution()
    await example_system_stats()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())