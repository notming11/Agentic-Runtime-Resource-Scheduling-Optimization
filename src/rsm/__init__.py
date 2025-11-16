"""
Autellix Resource Scheduler Module (RSM)

A program-aware scheduling system for LLM-driven agentic workflows,
inspired by Autellix (Luo et al., 2025).

This module provides:
- Stateful API layer for managing agentic programs
- Global process table for tracking program runtime metadata
- Session management for program lifecycle
- PLAS/ATLAS scheduling algorithms
- Data locality-aware load balancing

Based on: "Autellix: An Efficient Serving Engine for LLM Agents as General Programs"
"""

from .frontend import (
    # Main API
    AutellixClient,
    AutellixOpenAIAdapter,
    autellix_session,
    # Process Management
    GlobalProcessTable,
    SessionManager,
    # Data Classes
    ProgramEntry,
    ThreadMetadata,
    SessionInfo,
    # Enums
    ProgramState,
    SessionState,
)

__version__ = "1.0.0"
__author__ = "RSM Development Team"
__paper__ = "Autellix: An Efficient Serving Engine for LLM Agents as General Programs (Luo et al., 2025)"

__all__ = [
    # Main API
    "AutellixClient",
    "AutellixOpenAIAdapter",
    "autellix_session",
    # Process Management
    "GlobalProcessTable",
    "SessionManager",
    # Data Classes
    "ProgramEntry",
    "ThreadMetadata",
    "SessionInfo",
    # Enums
    "ProgramState",
    "SessionState",
]
