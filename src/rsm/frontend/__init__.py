"""
Autellix RSM Frontend Module

This module provides the stateful API layer for Autellix RSM, including:
- Process table for tracking program runtime metadata
- Session manager for lifecycle management
- API wrappers extending OpenAI/vLLM interfaces

Based on the Autellix paper: "An Efficient Serving Engine for LLM Agents as General Programs"
"""

from .process_table import (
    GlobalProcessTable,
    ProgramEntry,
    ThreadMetadata,
    ProgramState
)

from .session_manager import (
    SessionManager,
    SessionInfo,
    SessionState
)

from .api_wrapper import (
    AutellixClient,
    AutellixOpenAIAdapter,
    autellix_session
)

__all__ = [
    # Process Table
    "GlobalProcessTable",
    "ProgramEntry",
    "ThreadMetadata",
    "ProgramState",
    # Session Manager
    "SessionManager",
    "SessionInfo",
    "SessionState",
    # API Wrapper
    "AutellixClient",
    "AutellixOpenAIAdapter",
    "autellix_session",
]

__version__ = "1.0.0"
