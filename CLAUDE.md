# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a modular runtime framework for accelerating agentic AI workflows through two complementary optimization modules:

1. **UDF (User-Defined Function) Optimizer** - A dependency-aware task parallelization layer that converts sequential agent plans into concurrent execution batches
2. **Resource Scheduler Module (RSM)** - A program-aware scheduling system that optimizes LLM call scheduling, load balancing, and fairness across concurrent workflows

These modules are designed to integrate transparently with existing agentic frameworks (DeerFlow, LangGraph, CrewAI) via an adapter pattern, requiring no modification to core agent logic.

## Repository Structure

```
src/
├── udf-optimizer/          # Task parallelization module
│   ├── README.md          # User-facing documentation
│   └── TECHNICAL_GUIDE.md # Complete design specification (79KB)
└── rsm/                   # Resource scheduler module
    └── README.md          # Architecture and component documentation
```

## Current Development Status

**IMPORTANT:** This repository is currently in the **specification/design phase**. Both modules have comprehensive documentation but **no implementation code yet**. The READMEs and TECHNICAL_GUIDE.md contain complete design specifications, architecture details, and pseudocode, but actual Python implementations do not exist.

When working in this codebase:
- You are likely implementing components from scratch based on specifications
- Reference the detailed technical guides for implementation guidance
- The TECHNICAL_GUIDE.md in udf-optimizer/ is particularly comprehensive (79KB)

## UDF Optimizer Architecture

### Core Components

1. **Task Abstraction Layer** - Framework-agnostic interface for mapping tasks to a common schema
2. **Dependency Analyzer** - Builds DAG of task relationships using:
   - **LLM-Based Analysis** (recommended): Uses language models to understand task relationships
   - **Heuristic Analysis**: Rule-based inference using task metadata
   - **Explicit Dependencies**: Framework-provided declarations
3. **Parallel Executor** - Runs independent batches asynchronously using `asyncio`
4. **Configuration Manager** - Controls concurrency limits, dependency strategies, error recovery

### Execution Model

```
Planner → [UDF Module] → Executor
Sequential Plan → Dependency Graph → Parallel Batches
```

Tasks are organized into batches separated by synchronization barriers to preserve data dependencies. Independent tasks within a batch execute concurrently.

### Key Performance Characteristics

- **4-10x speedup** for typical multi-task workflows with independent tasks
- Designed for **I/O-bound operations** (API calls, web searches, LLM requests)
- Limited speedup for CPU-bound tasks
- Memory usage scales with concurrent task count

## Resource Scheduler Module (RSM) Architecture

### Core Components

1. **Process Table** - Global in-memory registry tracking per-program runtime metadata:
   - Service time (critical path length)
   - Waiting time
   - Engine affinity (for KV-cache locality)
   - Thread metadata and starvation counters

2. **ATLAS Scheduler** - Adaptive Thread-Level Attained Service:
   - Estimates critical path online using longest cumulative service time
   - Priority formula: `p(c_j) = max(p(c_k) + t_k)` for parent calls c_k
   - Prioritizes calls on longer critical paths to minimize makespan

3. **Multilevel Queues** - Discretizes program priorities into K queues:
   - Each queue represents a service time range
   - FIFO within queues, preemption between queues
   - Expired calls demoted one level

4. **Anti-Starvation Policy** - Prevents starvation via wait-to-service ratio:
   - Monitors: `W_total / T_total ≥ β`
   - Promotes entire program to highest priority queue when threshold exceeded
   - Ensures synchronized fairness across threads

5. **Load Balancer** - Data locality-aware routing:
   - **Short calls (≤2048 tokens)**: Least-loaded engine
   - **Long calls (>2048 tokens)**: Same engine as previous program call
   - Preserves KV-cache locality for 1.4× higher throughput

### Scheduler Flow

```
Program Invoked → LLM Calls Queued → ATLAS Scheduler
  → Multilevel Queues → Load Balancer → Engine Execution
```

### Mathematical Foundations

The RSM implements concepts from **Autellix (Luo et al., 2025)**, particularly:
- Online critical path estimation without explicit DAG traversal
- Program-level scheduling (not just call-level)
- Hybrid routing policy balancing cache locality and load distribution

## Integration Pattern

Both modules use the **Adapter Pattern** to integrate with existing frameworks:

```
Agent Framework
├── Planner
│   ↓
│   [UDF Module] ← Parallel plan generation
│   ↓
├── Executor
│   ↓
└── [RSM Backend] ← Schedules/load-balances LLM calls
```

Integration requires:
1. **Task Adapter** - Maps framework tasks to generic interface
2. **Executor Adapter** - Wraps execution logic
3. **Module Hook** - Intercepts plan before execution
4. **Configuration** - Sets preferences (typically YAML)

Estimated integration effort: ~200 lines of code per framework.

## Configuration

### UDF Optimizer Configuration

```yaml
parallelization:
  enabled: true
  max_concurrent_tasks: 10              # Maximum simultaneous tasks
  dependency_strategy: "llm_based"      # or "heuristic", "explicit"
  fallback_strategy: "heuristic"
  task_timeout_seconds: 300
  retry_on_failure: true
  failure_mode: "partial_completion"    # or "fail_fast"
```

### RSM Configuration Parameters

- **Starvation threshold (β)**: Wait-to-service ratio for promotion
- **Queue count (K)**: Number of priority levels
- **Time quantum**: Scheduling epoch duration
- **Cache locality threshold**: Token count for routing decision (default: 2048)

## Implementation Guidelines

When implementing components in this repository:

### Python Version
- Requires Python 3.8+
- Use `asyncio` for concurrent I/O operations

### UDF Optimizer Implementation Notes

1. **Dependency Analysis**:
   - LLM-based analysis requires API access (OpenAI, Anthropic, etc.)
   - Implement graceful fallback to heuristic analysis
   - Cache dependency analysis results when possible

2. **Error Handling**:
   - Must gracefully degrade to sequential execution on failure
   - Implement task-level isolation (one failure doesn't break others)
   - Use circuit breaker pattern for systemic failures
   - Provide detailed logging for debugging

3. **Result Aggregation**:
   - Implement memory-efficient aggregation for large result sets
   - May require result summarization for LLM context limits
   - Consider vector database integration for large contexts

### RSM Implementation Notes

1. **Process Table**:
   - In-memory registry with efficient concurrent access
   - Update atomically on call completion
   - Consider distributed state for multi-node deployments

2. **Scheduler Integration**:
   - Integrate with vLLM or similar LLM serving backends
   - Implement preemption support for queue prioritization
   - Track engine-level KV-cache state for routing

3. **Load Balancing**:
   - Monitor engine load metrics in real-time
   - Implement affinity tracking per program
   - Handle engine failures with graceful degradation

## Testing Strategy

When implementing tests:

1. **UDF Optimizer**:
   - Test dependency detection accuracy across strategies
   - Verify correct batch creation for various DAG structures
   - Measure speedup ratios for I/O-bound vs CPU-bound workloads
   - Test error handling and fallback mechanisms

2. **RSM**:
   - Verify priority calculation correctness
   - Test anti-starvation promotion triggers
   - Validate KV-cache locality preservation
   - Benchmark throughput improvements vs round-robin

## Documentation References

- **UDF Optimizer**: See `src/udf-optimizer/TECHNICAL_GUIDE.md` for complete 79KB specification
- **RSM**: See `src/rsm/README.md` for architecture details and mathematical formulations
- **Main README**: See root `README.md` for high-level overview

## Design Philosophy

Both modules follow these principles:

1. **Framework Agnostic** - Work with any agent system via adapters
2. **Transparent Integration** - No changes to core agent logic required
3. **Graceful Degradation** - Fall back to sequential/standard behavior on errors
4. **Production Ready** - Robust error handling, monitoring, observability
5. **Performance Focused** - Optimize for I/O-bound LLM operations

## Future Roadmap

Based on module READMEs:

- [ ] Core implementation of UDF parallelization engine
- [ ] LLM-based dependency analysis implementation
- [ ] Integration with DeerFlow framework
- [ ] Hierarchical result aggregation
- [ ] Vector database integration for large contexts
- [ ] RSM distributed scheduling across multiple nodes
- [ ] Monitoring dashboards and metrics collection
