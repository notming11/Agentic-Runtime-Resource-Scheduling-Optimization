"""
Engine Module for Autellix RSM

This module implements multi-engine orchestration and call management
for the Resource Scheduler Module, as described in Partition 4.

Components:
- MultiEngineManager: Coordinates multiple vLLM engine instances
- KVCacheCoordinator: Tracks KV cache affinity for routing decisions
- RequestLifecycleManager: Manages request states and cancellation
- EngineProcess: Wrapper around vLLM AsyncLLMEngine instances
"""

from .kv_cache_coordinator import KVCacheCoordinator
from .lifecycle_manager import RequestLifecycleManager, RequestState
from .engine_process import EngineProcess
from .multi_engine_manager import MultiEngineManager

__all__ = [
    "KVCacheCoordinator",
    "RequestLifecycleManager",
    "RequestState",
    "EngineProcess",
    "MultiEngineManager",
]
