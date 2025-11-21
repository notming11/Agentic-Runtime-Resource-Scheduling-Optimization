import threading
from collections import deque
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
import logging
import time
from dataclasses import field

from .priority_queue import PriorityQueueManager
from .context import ProgramContext, ThreadContext
from .mlfq_base import MLFQBase
from frontend.process_table import GlobalProcessTable

class ATLASScheduler(MLFQBase):
    
    """ATLAS Scheduler implementing adaptive MLFQ with thread-level tracking."""
    procress_table: GlobalProcessTable
    queue_manager: PriorityQueueManager
    program_contexts: Dict[str, ProgramContext]
    adaptation_threshold: float
    adaptation_check_interval: int
    starvation_threshold: int
    starvation_ratio: float
    
    def __init__(self, config: dict, process_table: GlobalProcessTable):
        super().__init__(config)
        self.process_table = process_table
        self.queue_manager = PriorityQueueManager(self.num_levels)
        self.program_contexts: Dict[str, ProgramContext] = {}
        
        # Adaptation parameters
        self.adaptation_threshold = config.get('adaptation_threshold', 3.0)
        self.adaptation_check_interval = config.get('adaptation_check_interval', 5000)
        
        # Anti-starvation
        self.starvation_threshold = config.get('starvation_threshold', 5000)
        self.starvation_ratio = config.get('starvation_ratio', 3.0)
        
    def schedule_next(self) -> Optional[str]:
        # Find highest priority non-empty queue
        priority = self.queue_manager.get_next_nonempty_level()
        if priority is None:
            return None
    
        # Dequeue program (round-robin within priority)
        program = self.queue_manager.dequeue_program(priority)
        if not program or not program.pending_requests:
            return None
    
        # Check adaptation if interval elapsed
        if self._should_check_adaptation(program):
            self._update_scheduling_mode(program)
    
        # Select request based on mode (program vs thread)
        request_id = self._select_request_from_program(program)
    
        # Re-enqueue if more work remains
        if program.pending_requests:
            self.queue_manager.enqueue_program(program, program.current_priority)
    
        return request_id
    
    def _select_request_from_program(self, program: ProgramContext) -> str:
        # Dispatch based on scheduling mode
        if program.scheduling_mode == 'thread':
            return self._select_by_thread(program)
        return self._select_by_program(program)

    def _select_by_program(self, program: ProgramContext) -> str:
        # PLAS: simple FIFO within program
        return program.pending_requests.popleft()

    def _select_by_thread(self, program: ProgramContext) -> str:
        # ATLAS: prioritize thread with least service
        available_threads = [
            t for t in program.threads.values() 
            if t.pending_requests
        ]
    
        if not available_threads:
            return program.pending_requests.popleft()
    
        # Sort by service time ascending
        available_threads.sort(key=lambda t: t.thread_service)
        chosen_thread = available_threads[0]
        request_id = chosen_thread.pending_requests.popleft()
        program.pending_requests.remove(request_id)
    
        return request_id
    
    def update_service(self, request_id: str, tokens_processed: int):
        # Retrieve metadata from process table
        program_id = self.process_table.get_program_for_request(request_id)
        thread_id = self.process_table.get_thread_for_request(request_id)
    
        program = self.program_contexts.get(program_id)
        if not program:
            return
    
        # Update program-level service (PLAS)
        program.cumulative_service += tokens_processed
    
        # Update thread-level service (ATLAS)
        if program.scheduling_mode == 'thread' and thread_id in program.threads:
            thread = program.threads[thread_id]
            thread.thread_service += tokens_processed
        
            # Update critical path estimate (max across threads)
            program.cumulative_service = max(
                program.cumulative_service,
                thread.thread_service
            )
    
        # Check quantum exhaustion
        self._check_quantum_exhaustion(program)

        # Unregister request mapping now that it's complete
        self.process_table.unregister_request(request_id)
    
    def _check_quantum_exhaustion(self, program: ProgramContext):
        quantum = self.get_quantum_for_priority(program.current_priority)
    
        # Demote if quantum exceeded
        if program.cumulative_service >= quantum:
            new_priority = min(
                program.current_priority + 1, 
                self.num_levels - 1
            )
        
            # Move to lower priority queue
            self.queue_manager.move_program(
                program.program_id, 
                new_priority
            )
        
            program.current_priority = new_priority
            program.cumulative_service = 0  # Reset for new quantum
        
            # self._log_priority_change(
            #     program.program_id, 
            #     program.current_priority - 1, 
            #     new_priority
            # )
            
    def _should_check_adaptation(self, program: ProgramContext) -> bool:
        now_ms = time.time() * 1000
        elapsed = now_ms - program.last_adaptation_check
        return elapsed > self.adaptation_check_interval

    def _update_scheduling_mode(self, program: ProgramContext):
        program.last_adaptation_check = time.time() * 1000
    
        # Need multiple threads to compare
        if len(program.threads) < 2:
            program.scheduling_mode = 'program'
            return
    
        # Calculate variance ratio across threads
        thread_services = [
            t.thread_service for t in program.threads.values()
        ]
    
        if not thread_services or all(s == 0 for s in thread_services):
            program.scheduling_mode = 'program'
            return
    
        max_service = max(thread_services)
        avg_service = sum(thread_services) / len(thread_services)
    
        if avg_service == 0:
            program.scheduling_mode = 'program'
            return
    
        variance_ratio = max_service / avg_service
    
        # High variance → thread-level, Low variance → program-level
        if variance_ratio > self.adaptation_threshold:
            program.scheduling_mode = 'thread'
            # self._log_mode_switch(program.program_id, 'thread', variance_ratio)
        else:
            program.scheduling_mode = 'program'
            # self._log_mode_switch(program.program_id, 'program', variance_ratio)
            
    def on_request_arrival(self, request_id: str, program_id: str, thread_id: str):
        # Get or create program context
        if program_id not in self.program_contexts:
            program = ProgramContext(
            program_id=program_id,
                cumulative_service=0,
                current_priority=0,
                pending_requests=deque(),
                waiting_since=time.time(),
                threads={},
                scheduling_mode='program',
                last_adaptation_check=time.time() * 1000
            )
            self.program_contexts[program_id] = program
        else:
            program = self.program_contexts[program_id]
    
        # Get or create thread context
        if thread_id not in program.threads:
            thread = ThreadContext(
                thread_id=thread_id,
                program_id=program_id,
                thread_service=0,
                pending_requests=deque(),
                last_active=time.time()
            )
            program.threads[thread_id] = thread
        else:
            thread = program.threads[thread_id]
    
        # Add to both queues
        program.pending_requests.append(request_id)
        thread.pending_requests.append(request_id)

        # Register request mapping in process table for later lookups
        self.process_table.register_request(request_id, program_id, thread_id)

        # Calculate priority from cumulative service
        priority = self.calculate_priority_from_service(program.cumulative_service)
        program.current_priority = priority

        # Enqueue in priority queue manager
        self.queue_manager.enqueue_program(program, priority)
        
    def check_and_promote_starved_programs(self):
        now_ms = time.time() * 1000
    
        for program in self.program_contexts.values():
            if not program.pending_requests:
                continue
        
        wait_time_ms = now_ms - (program.waiting_since * 1000)
        
        # Calculate wait/service ratio for program-level starvation
        if program.cumulative_service > 0:
            ratio = wait_time_ms / program.cumulative_service
            
            if ratio > self.starvation_ratio and program.current_priority > 0:
                # Promote to highest priority
                new_priority = 0
                self.queue_manager.move_program(program.program_id, new_priority)
                program.current_priority = new_priority
                program.waiting_since = time.time()
                
                # self._log_promotion(program.program_id, 'starvation')