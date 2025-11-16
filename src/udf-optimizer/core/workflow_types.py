"""
Type definitions for DeerFlow parallelization implementation.

This module defines the core data structures used throughout the parallel
execution system, including Steps, Plans, and State objects.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class StepType(str, Enum):
    """Type of step in the research plan."""
    RESEARCH = "research"
    PROCESSING = "processing"


@dataclass
class Step:
    """
    Represents a single step in a research plan.
    
    Attributes:
        title: Human-readable step name
        description: Detailed description of what the step should accomplish
        step_type: Type of step (research or processing)
        need_search: Whether this step requires web search
        execution_res: Result after execution (None if not yet executed)
        dependencies: List of step IDs this step depends on (for parallel execution)
        step_id: Unique identifier for this step
    """
    title: str
    description: str
    step_type: StepType
    need_search: bool
    execution_res: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    step_id: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if this step has been executed."""
        return self.execution_res is not None


@dataclass
class Plan:
    """
    Represents a complete research plan with multiple steps.
    
    Attributes:
        title: Overall plan title
        locale: Language locale (e.g., "en-US")
        thought: Planner's reasoning about the approach
        steps: List of steps to execute
        has_enough_context: Whether sufficient context exists to answer without research
    """
    title: str
    locale: str
    thought: str
    steps: List[Step]
    has_enough_context: bool = False
    
    def get_incomplete_steps(self) -> List[Step]:
        """Get all steps that haven't been executed yet."""
        return [step for step in self.steps if not step.is_complete()]
    
    def get_step_by_id(self, step_id: str) -> Optional[Step]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None


@dataclass
class State:
    """
    Represents the execution state of a workflow.
    
    This tracks all information accumulated during workflow execution,
    including messages, observations, and the current plan.
    
    Attributes:
        messages: List of conversation messages
        observations: List of results from executed steps
        current_plan: The active research plan
        resources: Available resources (e.g., documents, URLs)
        metadata: Additional metadata for execution
    """
    messages: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    current_plan: Optional[Plan] = None
    resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_observation(self, observation: str) -> None:
        """Add an observation from a completed step."""
        self.observations.append(observation)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})


@dataclass
class Configuration:
    """
    Configuration for parallelization behavior.
    
    Attributes:
        enabled: Whether parallelization is enabled
        max_concurrent_tasks: Maximum number of tasks to run in parallel
        max_tasks_per_second: Rate limit for launching tasks
        task_timeout_seconds: Timeout for individual tasks
        batch_timeout_seconds: Timeout for entire batches
        retry_on_failure: Whether to retry failed tasks
        max_retries: Maximum number of retry attempts
        retry_backoff_seconds: Exponential backoff delays
        failure_mode: How to handle failures ('fail_fast', 'partial_completion')
        dependency_strategy: Method for analyzing dependencies
    """
    enabled: bool = True
    max_concurrent_tasks: int = 10
    max_tasks_per_second: float = 5.0
    task_timeout_seconds: int = 300
    batch_timeout_seconds: int = 900
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_backoff_seconds: List[int] = field(default_factory=lambda: [2, 10, 30])
    failure_mode: Literal["fail_fast", "partial_completion"] = "partial_completion"
    dependency_strategy: Literal["llm_based", "heuristic", "explicit"] = "llm_based"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Configuration":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class ExecutionResult:
    """Result from executing a single step."""
    
    def __init__(
        self,
        step_id: str,
        success: bool,
        content: Optional[str] = None,
        error: Optional[str] = None,
        duration_seconds: float = 0.0
    ):
        self.step_id = step_id
        self.success = success
        self.content = content
        self.error = error
        self.duration_seconds = duration_seconds
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"ExecutionResult({self.step_id}: {status})"
