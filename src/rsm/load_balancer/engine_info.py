"""Engine information tracking."""

from dataclasses import dataclass, field
from typing import Set


@dataclass
class EngineInfo:
    """
    Engine tracking for load balancing.
    
    Attributes:
        engine_id: Unique identifier
        active_requests: Current number of running requests
        programs_assigned: PIDs assigned to this engine for locality
    """
    engine_id: str
    active_requests: int = 0
    programs_assigned: Set[str] = field(default_factory=set)
    
    def workload(self) -> int:
        """Workload metric - count of active requests."""
        return self.active_requests