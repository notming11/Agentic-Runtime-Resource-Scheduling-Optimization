"""
Graph builder for DeerFlow-style parallel workflow.

This module constructs the workflow graph with parallel execution capabilities.
Implements the changes described in the parallelization report.
"""

import logging
from typing import Literal, Callable, Dict, Any

from .workflow_types import State, Configuration
from .nodes import parallel_research_team_node, research_team_node_sequential

logger = logging.getLogger(__name__)


class WorkflowGraph:
    """
    Represents a workflow graph with nodes and edges.
    
    This is a simplified version of LangGraph's graph structure for demonstration.
    In a real implementation, this would use LangGraph's StateGraph.
    """
    
    def __init__(self, config: Configuration):
        self.config = config
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable] = {}
        self.entry_point: str = "start"
        
    def add_node(self, name: str, func: Callable):
        """Add a node to the graph."""
        self.nodes[name] = func
        logger.debug(f"Added node: {name}")
    
    def add_edge(self, from_node: str, to_node: str):
        """Add a direct edge between two nodes."""
        self.edges[from_node] = to_node
        logger.debug(f"Added edge: {from_node} -> {to_node}")
    
    def add_conditional_edges(
        self,
        from_node: str,
        condition_func: Callable,
        edge_map: Dict[str, str]
    ):
        """Add conditional edges based on a condition function."""
        self.conditional_edges[from_node] = (condition_func, edge_map)
        logger.debug(f"Added conditional edges from {from_node}")
    
    def set_entry_point(self, node: str):
        """Set the entry point of the graph."""
        self.entry_point = node
        logger.debug(f"Set entry point: {node}")
    
    def compile(self):
        """Compile the graph for execution."""
        logger.info(f"Compiled graph with {len(self.nodes)} nodes")
        return self


def build_parallel_workflow_graph(config: Configuration) -> WorkflowGraph:
    """
    Build the workflow graph with parallel execution enabled.
    
    This implements the new architecture from the parallelization report:
    - research_team node uses parallel execution
    - Direct edge from research_team to planner (no conditional routing)
    - Removes the need for continue_to_running_research_team()
    
    Args:
        config: Configuration with parallelization settings
    
    Returns:
        Compiled workflow graph
    """
    logger.info("Building parallel workflow graph")
    
    graph = WorkflowGraph(config)
    
    # Define nodes (simplified version - in real implementation these would be full agents)
    def coordinator_node(state: State) -> dict:
        """Initial coordinator node."""
        logger.info("Coordinator node: Processing user request")
        state.add_message("system", "Coordinator initialized")
        return {"next": "planner", "state": state}
    
    def planner_node(state: State) -> dict:
        """Planner node that creates or validates the plan."""
        logger.info("Planner node: Checking plan status")
        
        if state.current_plan is None:
            logger.info("No plan exists, would create plan here")
            return {"next": "research_team", "state": state}
        
        incomplete = state.current_plan.get_incomplete_steps()
        if incomplete:
            logger.info(f"{len(incomplete)} steps remaining")
            return {"next": "research_team", "state": state}
        else:
            logger.info("All steps complete, proceeding to reporter")
            return {"next": "reporter", "state": state}
    
    def reporter_node(state: State) -> dict:
        """Final reporter node that synthesizes results."""
        logger.info("Reporter node: Generating final report")
        state.add_message("assistant", "Final report generated from all observations")
        return {"next": "END", "state": state}
    
    # Create the research_team node with parallel or sequential execution
    if config.enabled:
        async def research_team_node(state: State) -> dict:
            """Research team node with parallel execution."""
            return await parallel_research_team_node(state, config)
    else:
        def research_team_node(state: State) -> dict:
            """Research team node with sequential execution (fallback)."""
            return research_team_node_sequential(state, config)
    
    # Add all nodes to graph
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("research_team", research_team_node)
    graph.add_node("reporter", reporter_node)
    
    # Add edges - NEW ARCHITECTURE: Direct edge from research_team to planner
    # This replaces the old conditional routing logic
    graph.add_edge("coordinator", "planner")
    graph.add_edge("research_team", "planner")  # KEY CHANGE: Direct edge, not conditional
    graph.add_edge("reporter", "END")
    
    # Planner has conditional routing to determine next step
    def planner_routing(result: dict) -> Literal["research_team", "reporter"]:
        """Route from planner based on plan completion."""
        if result.get("next") == "reporter":
            return "reporter"
        return "research_team"
    
    graph.add_conditional_edges(
        "planner",
        planner_routing,
        {"research_team": "research_team", "reporter": "reporter"}
    )
    
    # Set entry point
    graph.set_entry_point("coordinator")
    
    return graph.compile()


def build_sequential_workflow_graph(config: Configuration) -> WorkflowGraph:
    """
    Build the traditional sequential workflow graph.
    
    This is the old architecture for backward compatibility:
    - research_team uses conditional routing
    - Routes to individual researcher/coder nodes
    - Executes one step at a time
    
    Args:
        config: Configuration with parallelization disabled
    
    Returns:
        Compiled workflow graph
    """
    logger.info("Building sequential workflow graph (legacy mode)")
    
    config.enabled = False
    graph = WorkflowGraph(config)
    
    # Use simplified sequential execution
    # In real implementation, this would have researcher_node and coder_node separately
    
    def coordinator_node(state: State) -> dict:
        state.add_message("system", "Sequential coordinator initialized")
        return {"next": "planner", "state": state}
    
    def planner_node(state: State) -> dict:
        if state.current_plan is None:
            return {"next": "research_team", "state": state}
        incomplete = state.current_plan.get_incomplete_steps()
        if incomplete:
            return {"next": "research_team", "state": state}
        else:
            return {"next": "reporter", "state": state}
    
    def research_team_node(state: State) -> dict:
        return research_team_node_sequential(state, config)
    
    def reporter_node(state: State) -> dict:
        state.add_message("assistant", "Sequential final report")
        return {"next": "END", "state": state}
    
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("research_team", research_team_node)
    graph.add_node("reporter", reporter_node)
    
    graph.add_edge("coordinator", "planner")
    
    # OLD ARCHITECTURE: Conditional routing from research_team
    # Returns to research_team after each step
    def research_team_routing(result: dict) -> Literal["planner", "research_team"]:
        """Old conditional routing logic."""
        # After executing one step, loop back or go to planner
        if result.get("next") == "planner":
            return "planner"
        return "research_team"
    
    graph.add_conditional_edges(
        "research_team",
        research_team_routing,
        {"planner": "planner", "research_team": "research_team"}
    )
    
    # Planner routing
    def planner_routing(result: dict) -> Literal["research_team", "reporter"]:
        if result.get("next") == "reporter":
            return "reporter"
        return "research_team"
    
    graph.add_conditional_edges(
        "planner",
        planner_routing,
        {"research_team": "research_team", "reporter": "reporter"}
    )
    
    graph.add_edge("reporter", "END")
    graph.set_entry_point("coordinator")
    
    return graph.compile()


def build_workflow_graph(config: Configuration) -> WorkflowGraph:
    """
    Build the appropriate workflow graph based on configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Compiled workflow graph (parallel or sequential)
    """
    if config.enabled:
        logger.info("✓ Parallelization ENABLED - Using parallel execution graph")
        return build_parallel_workflow_graph(config)
    else:
        logger.info("✗ Parallelization DISABLED - Using sequential execution graph")
        return build_sequential_workflow_graph(config)
