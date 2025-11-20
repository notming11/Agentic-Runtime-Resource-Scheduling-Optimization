"""
Priority Queue Manager for Autellix ATLAS Scheduler

Implements discretized prioritization (paper §4.2.2). Programs assigned to queues
based on cumulative service, not always Q₁ like traditional MLFQ.

Anti-starvation: Promoted programs tracked with time-bounded expiration.
Cumulative service NOT reset (per paper), programs return to service-based priority after expiration.
"""

import threading
from collections import deque
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
import logging
import time

from .context import ProgramContext

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

@dataclass
class PriorityQueueConfig:
    """Configuration for priority queue ranges and time quanta."""
    queue_index: int                # 0 = highest priority
    service_range_low: float        # Q_lo_i (inclusive)
    service_range_high: float       # Q_hi_i (exclusive)
    time_quantum: int               # tokens allowed before demotion
    
    def contains_service(self, service_time: float) -> bool:
        """Check if service time falls in this queue's range."""
        return self.service_range_low <= service_time < self.service_range_high

class PriorityQueueManager:
    """
    Thread-safe priority queue manager implementing Autellix's discretized prioritization.
    
    Key Innovation: Programs assigned based on cumulative service, prevents program-level HoL blocking.
    Anti-Starvation: Time-bounded promotions to Q₀, cumulative service preserved.
    """
    
    def __init__(self, 
                 num_levels: int = 8,
                 base_quantum: int = 512,
                 service_range_multiplier: float = 2.0,
                 default_promotion_duration: float = 10.0):
        """
        Initialize priority queue manager.
        
        Priority ranges (exponential):
        Q₀: [0, base_quantum), Q₁: [base_quantum, 2*base_quantum), ...
        """
        self.num_levels = num_levels
        self.base_quantum = base_quantum
        self.multiplier = service_range_multiplier
        self.default_promotion_duration = default_promotion_duration
        
        self.queue_configs = self._initialize_queue_configs()
        self.program_queues: List[deque] = [deque() for _ in range(num_levels)]
        self.program_to_priority: Dict[str, int] = {}
        self.queued_programs: Set[str] = set()
        self.lock = threading.Lock()
        
        self.stats = {
            'total_enqueues': 0,
            'total_dequeues': 0,
            'total_moves': 0,
            'total_demotions': 0,
            'total_promotions': 0,
            'total_anti_starvation_promotions': 0,
            'total_promotion_expirations': 0,
            'programs_per_level': [0] * num_levels,
            'service_range_violations': 0,
            'currently_promoted': 0,
        }
        
        logger.info(f"PriorityQueueManager initialized: {num_levels} levels, "
                   f"base_quantum={base_quantum}, promotion_duration={default_promotion_duration}s")
        
        self._log_queue_configs()
        
    def _initialize_queue_configs(self) -> List[PriorityQueueConfig]:
        """Initialize priority queue configurations with exponential ranges."""
        configs = []
        
        for i in range(self.num_levels):
            if i == 0:
                low = 0.0
            else:
                low = self.base_quantum * (self.multiplier ** (i - 1))
            
            if i == self.num_levels - 1:
                high = float('inf')
            else:
                high = self.base_quantum * (self.multiplier ** i)
            
            quantum = int(self.base_quantum * (self.multiplier ** i))
            
            config = PriorityQueueConfig(
                queue_index=i,
                service_range_low=low,
                service_range_high=high,
                time_quantum=quantum
            )
            configs.append(config)
        
        return configs
    
    def _log_queue_configs(self):
        """Log priority queue configuration."""
        logger.info("Priority Queue Configuration:")
        for config in self.queue_configs:
            logger.info(f"  Q{config.queue_index}: "
                       f"service [{config.service_range_low:.0f}, "
                       f"{config.service_range_high:.0f}), "
                       f"quantum={config.time_quantum} tokens")
    
    def enqueue_program(self, 
                       program: 'ProgramContext', 
                       explicit_priority: Optional[int] = None,
                       is_anti_starvation: bool = False,
                       promotion_duration: Optional[float] = None) -> bool:
        """
        Add program to appropriate priority queue.
        
        Traditional MLFQ: Always enqueue to Q₁
        Autellix: Enqueue to Qᵢ where cumulative_service ∈ [Q_lo_i, Q_hi_i)
        
        Anti-starvation: Set time-bounded promotion, cumulative_service preserved.
        """
        with self.lock:
            program_id = program.program_id
            
            if program_id in self.queued_programs:
                current_priority = self.program_to_priority.get(program_id)
                logger.warning(f"Program {program_id} already queued at Q{current_priority}")
                return False
            
            # Determine target queue
            if explicit_priority is not None and is_anti_starvation:
                # Anti-starvation promotion
                if not 0 <= explicit_priority < self.num_levels:
                    logger.error(f"Invalid explicit priority {explicit_priority}")
                    return False
                
                priority = explicit_priority
                duration = promotion_duration if promotion_duration is not None else self.default_promotion_duration
                
                program.is_promoted = True
                program.promotion_expires_at = time.time() + duration
                program.promotion_duration = duration
                
                self.stats['total_anti_starvation_promotions'] += 1
                self.stats['currently_promoted'] += 1
                
                logger.info(f"Program {program_id} promoted to Q{priority} "
                          f"(anti-starvation: service={program.cumulative_service:.0f}, duration={duration:.1f}s)")
                          
            elif explicit_priority is not None:
                # Explicit override (testing)
                if not 0 <= explicit_priority < self.num_levels:
                    logger.error(f"Invalid explicit priority {explicit_priority}")
                    return False
                priority = explicit_priority
                logger.info(f"Program {program_id} assigned to Q{priority} (explicit override)")
                          
            elif program.is_promoted and time.time() < program.promotion_expires_at:
                # Still under promotion protection
                priority = 0
                remaining = program.promotion_expires_at - time.time()
                logger.debug(f"Program {program_id} still promoted to Q{priority} "
                           f"(service={program.cumulative_service:.0f}, expires in {remaining:.1f}s)")
                           
            else:
                # Check if promotion expired
                if program.is_promoted:
                    program.is_promoted = False
                    self.stats['total_promotion_expirations'] += 1
                    self.stats['currently_promoted'] -= 1
                    logger.info(f"Program {program_id} promotion expired, returning to service-based priority "
                              f"(service={program.cumulative_service:.0f})")
                
                # Normal service-based assignment
                priority = self._get_queue_for_service(program.cumulative_service)
                logger.debug(f"Program {program_id} assigned to Q{priority} (service={program.cumulative_service:.0f})")
            
            # Add to queue
            self.program_queues[priority].append(program)
            self.program_to_priority[program_id] = priority
            self.queued_programs.add(program_id)
            
            # Assign quantum
            program.current_quantum_remaining = self.queue_configs[priority].time_quantum
            program.current_priority = priority
            
            # Update statistics
            self.stats['total_enqueues'] += 1
            self.stats['programs_per_level'][priority] += 1
            
            return True
        
    def dequeue_program(self, priority: int) -> Optional['ProgramContext']:
        """Remove and return next program from priority level (round-robin)."""
        if not 0 <= priority < self.num_levels:
            logger.error(f"Invalid priority {priority}")
            return None
        
        with self.lock:
            queue = self.program_queues[priority]
            
            if not queue:
                return None
            
            program = queue.popleft()
            program_id = program.program_id
            
            del self.program_to_priority[program_id]
            self.queued_programs.remove(program_id)
            
            self.stats['total_dequeues'] += 1
            self.stats['programs_per_level'][priority] -= 1
            
            logger.debug(f"Dequeued program {program_id} from Q{priority} ({len(queue)} remaining)")
            return program
    
    def get_next_nonempty_level(self) -> Optional[int]:
        """Find highest priority level with waiting programs."""
        with self.lock:
            for level in range(self.num_levels):
                if self.program_queues[level]:
                    queue_size = len(self.program_queues[level])
                    logger.debug(f"Next work at Q{level} ({queue_size} programs)")
                    return level
            
            logger.debug("No work in any queue")
            return None
    
    def move_program(self, program: ProgramContext, new_priority: int, reason: str = "unknown") -> bool:
        """
        Move program between priority levels (demotion/promotion).
        For anti-starvation, prefer enqueue_program with is_anti_starvation=True.
        """
        program_id = program.program_id
        if not 0 <= new_priority < self.num_levels:
            logger.error(f"Invalid new_priority {new_priority}")
            return False
        
        with self.lock:
            if program_id not in self.program_to_priority:
                logger.warning(f"Cannot move program {program_id}: not in queues")
                return False
            
            old_priority = self.program_to_priority[program_id]
            
            if old_priority == new_priority:
                logger.debug(f"Program {program_id} already at Q{new_priority}")
                return True
            
            # Find and remove from old queue
            old_queue = self.program_queues[old_priority]
            
            for i, p in enumerate(old_queue):
                if p.program_id == program_id:
                    program = old_queue[i]
                    del old_queue[i]
                    break
            
            if program is None:
                logger.error(f"Program {program_id} not found in Q{old_priority}")
                del self.program_to_priority[program_id]
                self.queued_programs.remove(program_id)
                self.stats['service_range_violations'] += 1
                return False
            
            # Add to new queue
            self.program_queues[new_priority].append(program)
            self.program_to_priority[program_id] = new_priority
            program.current_priority = new_priority
            program.current_quantum_remaining = self.queue_configs[new_priority].time_quantum
            
            # Update statistics
            self.stats['total_moves'] += 1
            self.stats['programs_per_level'][old_priority] -= 1
            self.stats['programs_per_level'][new_priority] += 1
            
            if new_priority > old_priority:
                self.stats['total_demotions'] += 1
                direction = "demoted"
            else:
                self.stats['total_promotions'] += 1
                direction = "promoted"
            
            logger.info(f"Program {program_id} {direction} from Q{old_priority} to Q{new_priority} "
                       f"(reason: {reason}, service={program.cumulative_service:.0f})")
            
            return True
      
    ##--------------------------------------------------------------
    ## Unitilities (e.g. Getter)
    ##--------------------------------------------------------------
        
    def get_program_priority(self, program_id: str) -> Optional[int]:
        """Get current priority level of program."""
        with self.lock:
            return self.program_to_priority.get(program_id)
    
    def get_queue_quantum(self, priority: int) -> Optional[int]:
        """Get time quantum for priority level."""
        if not 0 <= priority < self.num_levels:
            return None
        with self.lock:
            return self.queue_configs[priority].time_quantum
    
    def is_program_queued(self, program_id: str) -> bool:
        """Check if program is currently queued."""
        with self.lock:
            return program_id in self.queued_programs
    
    def get_queue_sizes(self) -> List[int]:
        """Get number of programs at each priority level."""
        with self.lock:
            return [len(queue) for queue in self.program_queues]
    
    def get_total_programs(self) -> int:
        """Get total number of queued programs."""
        with self.lock:
            return len(self.queued_programs)
    
    def get_promoted_programs_count(self) -> int:
        """Get number of programs under anti-starvation promotion."""
        with self.lock:
            return self.stats['currently_promoted']
    
    def validate_consistency(self) -> bool:
        """
        Check internal consistency, accounting for anti-starvation promotions.
        Promoted programs may be in Q₀ even if service suggests lower queue.
        """
        with self.lock:
            actual_programs = {}
            seen_programs = set()
            violations = []
            promoted_ok = []
            
            for priority, queue in enumerate(self.program_queues):
                for program in queue:
                    program_id = program.program_id
                    
                    # Check duplicates
                    if program_id in seen_programs:
                        logger.error(f"Duplicate program {program_id} in multiple queues")
                        return False
                    
                    seen_programs.add(program_id)
                    actual_programs[program_id] = priority
                    
                    # Check if in correct queue
                    expected_priority = self._get_queue_for_service(program.cumulative_service)
                    
                    if priority != expected_priority:
                        current_time = time.time()
                        if (program.is_promoted and 
                            current_time < program.promotion_expires_at and
                            priority == 0):
                            # OK - anti-starvation override
                            remaining = program.promotion_expires_at - current_time
                            promoted_ok.append(
                                f"Program {program_id} in Q{priority} due to anti-starvation "
                                f"(service={program.cumulative_service:.0f}, expires in {remaining:.1f}s)"
                            )
                        else:
                            # Real violation
                            violations.append(
                                f"Program {program_id} in Q{priority} but service={program.cumulative_service:.0f} "
                                f"should be Q{expected_priority} (promoted={program.is_promoted})"
                            )
            
            # Verify tracking consistency
            if actual_programs != self.program_to_priority:
                logger.error("program_to_priority mismatch")
                return False
            
            if seen_programs != self.queued_programs:
                logger.error("queued_programs set mismatch")
                return False
            
            # Log promotions (info level)
            if promoted_ok:
                logger.info(f"{len(promoted_ok)} programs legitimately promoted")
                for msg in promoted_ok[:5]:
                    logger.info(f"  {msg}")
            
            # Log violations (warning level)
            if violations:
                logger.warning(f"Found {len(violations)} service range violations")
                for violation in violations[:5]:
                    logger.warning(f"  {violation}")
                return False
            
            # Sync statistics
            actual_counts = [len(queue) for queue in self.program_queues]
            if actual_counts != self.stats['programs_per_level']:
                logger.warning("Statistics out of sync, updating")
                self.stats['programs_per_level'] = actual_counts
            
            return True
        
    def get_debug_info(self) -> Dict:
        """Get detailed debug information about queue state."""
        with self.lock:
            info = {
                'num_levels': self.num_levels,
                'base_quantum': self.base_quantum,
                'total_programs': len(self.queued_programs),
                'programs_per_level': [len(q) for q in self.program_queues],
                'statistics': self.stats.copy(),
                'program_locations': self.program_to_priority.copy(),
                'queue_configs': [],
                'level_details': []
            }
            
            # Per-level details
            for level, queue in enumerate(self.program_queues):
                config = self.queue_configs[level]
                
                programs_info = []
                for program in queue:
                    programs_info.append({
                        'program_id': program.program_id,
                        'cumulative_service': program.cumulative_service,
                        'current_priority': program.current_priority,
                        'quantum_remaining': getattr(program, 'current_quantum_remaining', None),
                        'is_promoted': getattr(program, 'is_promoted', False),
                        'promotion_expires_at': getattr(program, 'promotion_expires_at', 0.0)
                    })
                
                level_info = {
                    'priority': level,
                    'size': len(queue),
                    'service_range': [config.service_range_low, config.service_range_high],
                    'time_quantum': config.time_quantum,
                    'programs': programs_info
                }
                info['level_details'].append(level_info)
                
                info['queue_configs'].append({
                    'queue': level,
                    'service_range': f"[{config.service_range_low:.0f}, {config.service_range_high:.0f})",
                    'quantum': config.time_quantum
                })
            
            return info
        
    def get_statistics(self) -> Dict:
        """Get operational statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats['total_programs'] = len(self.queued_programs)
            stats['queue_sizes'] = [len(q) for q in self.program_queues]
            
            if stats['total_moves'] > 0:
                stats['promotion_rate'] = stats['total_promotions'] / stats['total_moves']
                stats['demotion_rate'] = stats['total_demotions'] / stats['total_moves']
            
            return stats
    
    def reset_statistics(self):
        """Reset operational statistics."""
        with self.lock:
            self.stats = {
                'total_enqueues': 0,
                'total_dequeues': 0,
                'total_moves': 0,
                'total_demotions': 0,
                'total_promotions': 0,
                'total_anti_starvation_promotions': 0,
                'total_promotion_expirations': 0,
                'programs_per_level': [len(q) for q in self.program_queues],
                'service_range_violations': 0,
                'currently_promoted': 0,
            }
            logger.info("Statistics reset")
            
    def clear_all_queues(self):
        """Remove all programs from all queues."""
        with self.lock:
            for queue in self.program_queues:
                queue.clear()
            
            self.program_to_priority.clear()
            self.queued_programs.clear()
            self.stats['programs_per_level'] = [0] * self.num_levels
            
            logger.warning("All queues cleared")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        with self.lock:
            queue_sizes = [len(q) for q in self.program_queues]
            return (f"PriorityQueueManager(levels={self.num_levels}, "
                   f"base_quantum={self.base_quantum}, "
                   f"total_programs={len(self.queued_programs)}, "
                   f"distribution={queue_sizes})")
    
    def __str__(self) -> str:
        """Human-readable representation."""
        with self.lock:
            lines = [f"Priority Queue Manager ({self.num_levels} levels):"]
            for i, queue in enumerate(self.program_queues):
                config = self.queue_configs[i]
                lines.append(f"  Q{i}: {len(queue)} programs, quantum={config.time_quantum}, "
                           f"range=[{config.service_range_low:.0f}, {config.service_range_high:.0f})")
            return "\n".join(lines)