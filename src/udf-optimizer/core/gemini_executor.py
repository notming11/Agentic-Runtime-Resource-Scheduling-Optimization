"""
Backward compatibility wrapper for gemini_executor.py

This module now imports from executor.py which supports multiple LLM backends
(local, Gemini, OpenAI, etc.). The API remains the same for backward compatibility.

MIGRATION NOTE:
- For new code, use `from .executor import DependencyAnalyzer, StepExecutor` instead
- This file exists only for backward compatibility with existing code
- Original Gemini-only implementation backed up in gemini_executor_backup.py
"""

import warnings
import logging

# Import from new executor module
from .executor import (
    BatchDefinition,
    DependencyAnalyzer,
    StepExecutor as GeminiStepExecutor,  # Renamed for compatibility
    load_plan_from_json
)

logger = logging.getLogger(__name__)

# Deprecation warning (only show once)
warnings.simplefilter('once', DeprecationWarning)
warnings.warn(
    "gemini_executor.py is deprecated. Please use executor.py instead. "
    "The new module supports multiple LLM backends (local, Gemini, OpenAI) "
    "configured via config.yaml or environment variables.",
    DeprecationWarning,
    stacklevel=2
)

logger.info("gemini_executor.py: Using new multi-backend executor implementation")

# Export all for backward compatibility
__all__ = [
    'BatchDefinition',
    'DependencyAnalyzer',
    'GeminiStepExecutor',
    'load_plan_from_json'
]
