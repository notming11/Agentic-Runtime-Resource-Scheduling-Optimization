import threading
from collections import deque
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
import logging
import time
from dataclasses import field

class ThreadContext:
    """Thread-level tracking (used in ATLAS mode)"""
    thread_id: str
    program_id: str
    thread_service: int = 0              # Tokens processed by this thread
    pending_requests: deque = field(default_factory=deque)
    last_active: float = field(default_factory=time.time)
    
    def __init__(self, thread_id: str, program_id: str, last_active: float):
        self.thread_id = thread_id
        self.program_id = program_id
        self.last_active = last_active
    
    def get_thread_id(self):
        return self.thread_id
    
    def get_program_id(self):
        return self.program_id
    
    def get_thread_service(self):
        return self.thread_service
    
    def get_pending_request(self):
        return self.pending_requests
    
    def get_last_active(self):
        return self.last_active
    
        
    
class ProgramContext:
    """Program-level tracking (always used)"""
    program_id: str
    cumulative_service: int = 0          # Total tokens across all threads
    current_priority: int = 0            # Current MLFQ level
    pending_requests: deque = field(default_factory=deque)
    waiting_since: float = field(default_factory=time.time)
    current_quantum_remaining: int = 0
    current_priority: int 
    
    # ATLAS-specific fields
    threads: Dict[str, ThreadContext] = field(default_factory=dict)
    scheduling_mode: str = 'thread', 'program'
    last_adaptation_check: float = field(default_factory=lambda: time.time() * 1000)
    
    # Anti-starvation fields
    is_promoted: bool = False
    promotion_expires_at: int = 0
    promotion_duration: int = 0
    
    def __init__(self, program_id: str, waiting_since: float, threads: Dict[str, ThreadContext], priority: int):
        self.program_id = program_id
        self.waiting_since = waiting_since
        self.threads = threads
        self.priority = priority
        
    