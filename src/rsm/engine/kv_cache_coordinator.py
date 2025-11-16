"""
KV Cache Coordinator for Autellix RSM

This module implements CPU-based KV cache affinity tracking (not memory management).
It tracks which session/program has KV cache on which engine and calculates cache
hit rates for routing decisions.

Based on Autellix Section 4.3 - Data Locality-Aware Load Balancing:
- Short calls (≤2048 tokens): Least-loaded engine
- Long calls (>2048 tokens): Same engine as previous program call
"""

import time
import threading
import hashlib
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CacheEntry:
    """
    Tracks KV cache metadata for a session on a specific engine.

    Attributes:
        session_id: Session identifier
        engine_id: Engine where cache exists
        prompt_hash: Hash of the prompt (for deduplication)
        token_count: Number of tokens in cached prompt
        last_access: Timestamp of last access
        access_count: Number of times this cache was accessed
    """
    session_id: str
    engine_id: str
    prompt_hash: str
    token_count: int
    last_access: float = field(default_factory=time.time)
    access_count: int = 0

    def update_access(self):
        """Update access statistics"""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class EngineStats:
    """
    Statistics for an engine's KV cache usage.

    Attributes:
        engine_id: Engine identifier
        total_sessions: Number of unique sessions with cache on this engine
        total_cache_hits: Total number of cache hits
        total_requests: Total number of requests routed to this engine
        average_token_count: Average token count of cached prompts
    """
    engine_id: str
    total_sessions: int = 0
    total_cache_hits: int = 0
    total_requests: int = 0
    average_token_count: float = 0.0

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.total_cache_hits / self.total_requests


class KVCacheCoordinator:
    """
    CPU-based KV cache affinity tracking for routing decisions.

    This class does NOT manage actual GPU memory or KV cache implementation.
    It only tracks metadata to inform the load balancer's routing decisions.

    Key responsibilities:
    - Track which session has cache on which engine
    - Calculate cache hit rates for routing decisions
    - Provide affinity recommendations to load balancer
    - Maintain statistics for monitoring
    """

    def __init__(self, cache_token_threshold: int = 2048):
        """
        Initialize the KV cache coordinator.

        Args:
            cache_token_threshold: Token count threshold for cache locality
                                  (default: 2048 from Autellix paper)
        """
        self.cache_token_threshold = cache_token_threshold

        # session_id -> {engine_id -> CacheEntry}
        self._session_cache_map: Dict[str, Dict[str, CacheEntry]] = defaultdict(dict)

        # engine_id -> set of session_ids with cache on this engine
        self._engine_sessions: Dict[str, Set[str]] = defaultdict(set)

        # Statistics tracking
        self._engine_stats: Dict[str, EngineStats] = {}
        self._cache_stats = {
            "total_hits": 0,
            "total_misses": 0,
            "total_requests": 0
        }

        # Thread safety
        self._lock = threading.RLock()

    def record_cache_usage(
        self,
        session_id: str,
        engine_id: str,
        prompt_tokens: int,
        prompt_text: Optional[str] = None
    ):
        """
        Track where a session's KV cache exists.

        Called by the engine/load balancer after routing a request.

        Args:
            session_id: Session identifier
            engine_id: Engine where this request was routed
            prompt_tokens: Number of tokens in the prompt
            prompt_text: Optional prompt text for hashing
        """
        with self._lock:
            # Generate prompt hash for deduplication
            prompt_hash = self._hash_prompt(prompt_text) if prompt_text else ""

            # Create or update cache entry
            if engine_id not in self._session_cache_map[session_id]:
                # New cache entry
                entry = CacheEntry(
                    session_id=session_id,
                    engine_id=engine_id,
                    prompt_hash=prompt_hash,
                    token_count=prompt_tokens
                )
                self._session_cache_map[session_id][engine_id] = entry
                self._engine_sessions[engine_id].add(session_id)

                # Update engine stats
                if engine_id not in self._engine_stats:
                    self._engine_stats[engine_id] = EngineStats(engine_id=engine_id)
                self._engine_stats[engine_id].total_sessions += 1
            else:
                # Update existing cache entry
                entry = self._session_cache_map[session_id][engine_id]
                entry.update_access()

                # Update token count (take max for conservative estimate)
                entry.token_count = max(entry.token_count, prompt_tokens)

            # Update engine stats
            if engine_id in self._engine_stats:
                stats = self._engine_stats[engine_id]
                stats.total_requests += 1

                # Update average token count
                total_tokens = sum(
                    e.token_count
                    for e in self._session_cache_map[session_id].values()
                )
                stats.average_token_count = total_tokens / len(
                    self._session_cache_map[session_id]
                )

    def get_best_engine_for_session(
        self,
        session_id: str,
        available_engines: List[str],
        prompt_tokens: int
    ) -> Optional[str]:
        """
        Return engine with best cache affinity for this session.

        Implements Autellix's hybrid routing policy:
        - Short calls (≤ threshold): Return None (use least-loaded)
        - Long calls (> threshold): Return engine with best affinity

        Args:
            session_id: Session identifier
            available_engines: List of currently available engine IDs
            prompt_tokens: Number of tokens in this request

        Returns:
            Engine ID with best affinity, or None if should use least-loaded
        """
        with self._lock:
            # Short calls: use least-loaded routing
            if prompt_tokens <= self.cache_token_threshold:
                return None

            # Long calls: prefer engine with existing cache
            if session_id not in self._session_cache_map:
                # No cache history, use least-loaded
                return None

            session_caches = self._session_cache_map[session_id]

            # Find best available engine with cache
            best_engine = None
            best_score = -1.0

            for engine_id in available_engines:
                if engine_id in session_caches:
                    cache_entry = session_caches[engine_id]

                    # Score based on:
                    # 1. Recency (more recent = better)
                    # 2. Access count (more accesses = better)
                    # 3. Token count (more tokens cached = better)
                    recency_score = 1.0 / (time.time() - cache_entry.last_access + 1.0)
                    access_score = cache_entry.access_count
                    token_score = cache_entry.token_count / self.cache_token_threshold

                    # Weighted combination (recency is most important)
                    score = (0.5 * recency_score +
                            0.3 * access_score +
                            0.2 * token_score)

                    if score > best_score:
                        best_score = score
                        best_engine = engine_id

            # Track hit/miss
            if best_engine:
                self._cache_stats["total_hits"] += 1
                if best_engine in self._engine_stats:
                    self._engine_stats[best_engine].total_cache_hits += 1
            else:
                self._cache_stats["total_misses"] += 1

            self._cache_stats["total_requests"] += 1

            return best_engine

    def estimate_cache_hit_rate(
        self,
        session_id: str,
        engine_id: str
    ) -> float:
        """
        Estimate hit rate if this session were routed to this engine.

        Args:
            session_id: Session identifier
            engine_id: Engine identifier

        Returns:
            Estimated cache hit rate (0.0 to 1.0)
        """
        with self._lock:
            # If session has cache on this engine, estimate high hit rate
            if (session_id in self._session_cache_map and
                engine_id in self._session_cache_map[session_id]):
                cache_entry = self._session_cache_map[session_id][engine_id]

                # Higher hit rate for recently accessed caches
                recency = time.time() - cache_entry.last_access
                if recency < 60:  # Less than 1 minute
                    return 0.9
                elif recency < 300:  # Less than 5 minutes
                    return 0.7
                elif recency < 1800:  # Less than 30 minutes
                    return 0.5
                else:
                    return 0.3

            # No cache history
            return 0.0

    def remove_session(self, session_id: str):
        """
        Remove all cache tracking for a session.

        Called when a session ends.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            if session_id not in self._session_cache_map:
                return

            # Remove from engine sessions
            for engine_id in self._session_cache_map[session_id]:
                if engine_id in self._engine_sessions:
                    self._engine_sessions[engine_id].discard(session_id)

                # Update engine stats
                if engine_id in self._engine_stats:
                    self._engine_stats[engine_id].total_sessions -= 1

            # Remove session cache map
            del self._session_cache_map[session_id]

    def get_engine_sessions(self, engine_id: str) -> Set[str]:
        """
        Get all sessions that have cache on this engine.

        Args:
            engine_id: Engine identifier

        Returns:
            Set of session IDs
        """
        with self._lock:
            return self._engine_sessions.get(engine_id, set()).copy()

    def get_session_engines(self, session_id: str) -> List[str]:
        """
        Get all engines that have cache for this session.

        Args:
            session_id: Session identifier

        Returns:
            List of engine IDs
        """
        with self._lock:
            if session_id not in self._session_cache_map:
                return []
            return list(self._session_cache_map[session_id].keys())

    def get_cache_stats(self) -> Dict:
        """
        Get global cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._cache_stats["total_requests"]
            hit_rate = 0.0
            if total_requests > 0:
                hit_rate = self._cache_stats["total_hits"] / total_requests

            return {
                "total_hits": self._cache_stats["total_hits"],
                "total_misses": self._cache_stats["total_misses"],
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "total_sessions": len(self._session_cache_map),
                "engine_stats": {
                    engine_id: {
                        "total_sessions": stats.total_sessions,
                        "total_requests": stats.total_requests,
                        "total_cache_hits": stats.total_cache_hits,
                        "hit_rate": stats.get_hit_rate(),
                        "average_token_count": stats.average_token_count
                    }
                    for engine_id, stats in self._engine_stats.items()
                }
            }

    def cleanup_stale_sessions(self, timeout_seconds: float = 3600):
        """
        Remove cache tracking for sessions that haven't been accessed recently.

        Args:
            timeout_seconds: Inactivity timeout in seconds (default: 1 hour)
        """
        with self._lock:
            current_time = time.time()
            stale_sessions = []

            for session_id, engine_caches in self._session_cache_map.items():
                # Check most recent access across all engines
                most_recent = max(
                    (cache.last_access for cache in engine_caches.values()),
                    default=0
                )

                if (current_time - most_recent) > timeout_seconds:
                    stale_sessions.append(session_id)

            # Remove stale sessions
            for session_id in stale_sessions:
                self.remove_session(session_id)

    def _hash_prompt(self, prompt: str) -> str:
        """
        Generate hash of prompt for deduplication.

        Args:
            prompt: Prompt text

        Returns:
            SHA256 hash (first 16 chars)
        """
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def register_engine(self, engine_id: str):
        """
        Register a new engine in the coordinator.

        Args:
            engine_id: Engine identifier
        """
        with self._lock:
            if engine_id not in self._engine_stats:
                self._engine_stats[engine_id] = EngineStats(engine_id=engine_id)
            if engine_id not in self._engine_sessions:
                self._engine_sessions[engine_id] = set()

    def unregister_engine(self, engine_id: str):
        """
        Unregister an engine from the coordinator.

        Removes all cache tracking for this engine.

        Args:
            engine_id: Engine identifier
        """
        with self._lock:
            # Remove from session cache maps
            for session_id in list(self._session_cache_map.keys()):
                if engine_id in self._session_cache_map[session_id]:
                    del self._session_cache_map[session_id][engine_id]

                # Remove session if no more engines
                if not self._session_cache_map[session_id]:
                    del self._session_cache_map[session_id]

            # Remove engine sessions
            if engine_id in self._engine_sessions:
                del self._engine_sessions[engine_id]

            # Remove engine stats
            if engine_id in self._engine_stats:
                del self._engine_stats[engine_id]
