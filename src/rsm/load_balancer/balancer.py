"""
Main load balancer implementation for Autellix.

Implements Algorithm 2 from the Autellix paper:
- Small requests (<=2048 tokens): Use LEAST_USED (load balance)
- Large requests (>2048 tokens): Use locality (route to program's engine)
"""

import threading
from typing import Dict, Optional, List

from .types import RequestSize
from .engine_info import EngineInfo


class LoadBalancer:
    """
    Load balancer implementing Algorithm 2 from Autellix paper.
    
    Key insight: Small requests benefit from load balancing across engines,
    while large requests benefit from locality (reusing KV-cache from same program).
    """
    
    # From Algorithm 2: threshold is 2048 tokens
    SMALL_REQUEST_THRESHOLD = 2048
    
    def __init__(self, process_table):
        """
        Initialize load balancer.
        
        Args:
            process_table: GlobalProcessTable for program state tracking
        """
        self.process_table = process_table
        self._engines: Dict[str, EngineInfo] = {}
        self._program_table: Dict[str, str] = {}  # pt[c.pid] = engine_id
        self._lock = threading.RLock()
        
        # Statistics
        self._total_requests = 0
        self._small_requests = 0
        self._large_requests = 0
        self._locality_hits = 0  # Large requests using existing assignment
        self._locality_assigns = 0  # Large requests creating new assignment
    
    def register_engine(self, engine_id: str):
        """
        Register a new engine.
        
        Args:
            engine_id: Unique engine identifier
        """
        with self._lock:
            if engine_id not in self._engines:
                self._engines[engine_id] = EngineInfo(engine_id=engine_id)
    
    def unregister_engine(self, engine_id: str):
        """
        Remove an engine.
        
        Args:
            engine_id: Engine identifier
        """
        with self._lock:
            self._engines.pop(engine_id, None)
            # Remove from program assignments
            to_remove = [pid for pid, eid in self._program_table.items() 
                        if eid == engine_id]
            for pid in to_remove:
                del self._program_table[pid]
    
    def _least_used_engine(self) -> Optional[str]:
        """
        LEAST_USED procedure from Algorithm 2.
        
        Query engine workloads in parallel and return the least utilized engine.
        
        Returns:
            Engine ID with minimum workload, or None if no engines
        """
        if not self._engines:
            return None
        
        # Find engine with minimum workload (active requests)
        return min(self._engines.items(), 
                  key=lambda x: x[1].workload())[0]
    
    def route_request(self, pid: str, num_tokens: int) -> Optional[str]:
        """
        LOAD_BALANCER procedure from Algorithm 2.
        
        Args:
            pid: Program identifier (c.pid in the algorithm)
            num_tokens: Number of prefill tokens (LEN(c.tokens))
            
        Returns:
            Selected engine ID, or None if no engines available
        """
        with self._lock:
            self._total_requests += 1
            
            if not self._engines:
                return None
            
            assigned_engine = None
            
            # Line 2: if LEN(c.tokens) <= 2048 then (Small request)
            if num_tokens <= self.SMALL_REQUEST_THRESHOLD:
                self._small_requests += 1
                # Line 3: assigned_engine = LEAST_USED(Engines)
                assigned_engine = self._least_used_engine()
            
            # Line 4-12: else (Large request)
            else:
                self._large_requests += 1
                # Line 5: if c.pid ∈ pt then (Program already assigned)
                if pid in self._program_table:
                    # Line 6: assigned_engine = pt[c.pid]
                    assigned_engine = self._program_table[pid]
                    # Verify engine still exists
                    if assigned_engine not in self._engines:
                        # Engine was removed, need to reassign
                        assigned_engine = None
                        del self._program_table[pid]
                    else:
                        self._locality_hits += 1
                
                # Line 7-10: else (Program not yet assigned)
                if assigned_engine is None:
                    # Line 9: assigned_engine = LEAST_USED(Engines)
                    assigned_engine = self._least_used_engine()
                    if assigned_engine:
                        # Line 10: pt[c.pid] = assigned_engine
                        self._program_table[pid] = assigned_engine
                        self._locality_assigns += 1
            
            # Update engine state
            if assigned_engine:
                engine = self._engines[assigned_engine]
                engine.active_requests += 1
                engine.programs_assigned.add(pid)
            
            # Line 13: return assigned_engine
            return assigned_engine
    
    def complete_request(self, engine_id: str, pid: str):
        """
        Notify that a request has completed.
        
        Args:
            engine_id: Engine that processed the request
            pid: Program identifier
        """
        with self._lock:
            if engine_id in self._engines:
                engine = self._engines[engine_id]
                if engine.active_requests > 0:
                    engine.active_requests -= 1
    
    def remove_program(self, pid: str):
        """
        Remove program from tracking (e.g., session ended).
        
        Args:
            pid: Program identifier
        """
        with self._lock:
            self._program_table.pop(pid, None)
            
            # Remove from all engines' assignment tracking
            for engine in self._engines.values():
                engine.programs_assigned.discard(pid)
    
    def get_program_engine(self, pid: str) -> Optional[str]:
        """
        Get the assigned engine for a program (from pt table).
        
        Args:
            pid: Program identifier
            
        Returns:
            Engine ID if program is assigned, None otherwise
        """
        with self._lock:
            return self._program_table.get(pid)
    
    def get_engine_workload(self, engine_id: str) -> Optional[int]:
        """
        Get workload for an engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Number of active requests, or None if engine not found
        """
        with self._lock:
            engine = self._engines.get(engine_id)
            return engine.workload() if engine else None
    
    def get_stats(self) -> Dict:
        """
        Get load balancer statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "small_requests": self._small_requests,
                "large_requests": self._large_requests,
                "locality_hits": self._locality_hits,
                "locality_assigns": self._locality_assigns,
                "programs_in_table": len(self._program_table),
                "engines": {
                    eid: {
                        "active_requests": engine.active_requests,
                        "programs_assigned": len(engine.programs_assigned),
                        "workload": engine.workload()
                    }
                    for eid, engine in self._engines.items()
                }
            }
