# Class for multi-level feedback queue
from collections import deque
from typing import List
import math

class PriorityLevel:
    """Represents one level in the MLFQ"""
    level: int          #(0 = highest priority)
    time_quantum: int   #quantum for each llm calls in this level
    queue: deque
    
    def __init__(self, level, quantum):
        self.level = level
        self.time_quantum = quantum
        self.queue = deque()
        
    def is_empty(self):
        """Check if this level has any programs waiting"""
        return len(self.queue) == 0
    
    def size(self) -> int:
        """Get number of programs at this level"""
        return len(self.queue)
    
    def enqueue(self, program):
        """Add program to this level's queue"""
        self.queue.append(program)
    
    def dequeue(self):
        """Remove and return next program (FIFO)"""
        if self.queue:
            return self.queue.popleft()
        return None
    
class MLFQBase:
    """Base infrastructure for MLFQ"""
    priority_levels: List[PriorityLevel]     #Liist of priority levels
    num_levels: int = 8                     #number of priority levels
    base_quantum: int = 512                 #base quantum for priority levels
    starvation_threshold: int = 5000        #threshold for starvation
    
    def __init__(self, config):
        self.num_levels = config.get('num_priority_levels', 8)
        self.base_quantum = config.get('base_quantum', 512)
        self.starvation_threshold = config.get('starvation_threshold', 5000)
        
        self.priority_levels = []
        for level in range(self.num_levels):
            quantum = self.base_quantum * (2 ** level)
            self.priority_levels.append(PriorityLevel(level, quantum))
            
    def get_quantum_for_priority(self, priority: int):
        """get quantum for a priority level"""
        return self.priority_levels[priority].quantum
    
    def calculate_priority_from_service(self, service_time: int):
        if service_time < self.base_quantum:
            return 0
        
        level = int(math.log2(service_time/self.base_quantum)) + 1
        return min(level, self.num_levels-1)
    
    def get_next_nonempty_level(self) -> int | None:
        """
        Find the highest priority level with waiting programs.
        
        Returns:
            Priority level index, or None if all levels empty
        """
        for level in range(self.num_levels):
            if not self.priority_levels[level].is_empty():
                return level
        return None
    
    