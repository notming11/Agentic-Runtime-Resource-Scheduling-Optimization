# 🧩 Agentic AI Optimization Framework
**UDF Optimization + Resource Scheduler Module (RSM)**  
_A modular runtime framework for accelerating agentic AI workflows_

---

## 🚀 Overview


1. **User-Defined Function (UDF) Optimization Module** — a dependency-aware task parallelization layer that converts sequential agent plans into concurrent execution batches.
2. **Resource Scheduler Module (RSM)** — a program-aware scheduling system inspired by **Autellix**, designed to optimize LLM call scheduling, load balancing, and fairness across concurrent workflows.

Together, these modules improve workflow throughput, reduce latency, and enhance resource utilization in LLM-driven, multi-agent systems such as **DeerFlow**, **LangGraph**, or **CrewAI**.

---

## 🏗️ Architecture

### 1️⃣ User-Defined Function (UDF) Module
**Goal:** Identify independent tasks and execute them concurrently.

**Core components:**
- **Task Abstraction Layer:** Framework-agnostic interface for mapping tasks (e.g., “Steps” in DeerFlow) to a common schema.
- **Dependency Analyzer:** Builds a DAG of task relationships via explicit, heuristic, or LLM-based analysis.
- **Parallel Executor:** Runs independent batches asynchronously using `asyncio` for I/O-bound tasks.
- **Configuration Manager:** Controls concurrency limits, dependency strategies, and error recovery.

**Execution Model:**
Planner → [UDF Module] → Executor
Sequential Plan → Dependency Graph → Parallel Batches

Each batch runs concurrently, separated by synchronization barriers to preserve data dependencies.  

---

### 2️⃣ Resource Scheduler Module (RSM)
**Goal:** Optimize the execution of LLM calls across multiple engines.

#### 🧮 Core Components
- **Process Table:** Tracks runtime metadata (service time, wait time, engine affinity) for all active programs.
- **ATLAS Scheduler:** Estimates the critical path of each program’s DAG using _Attained Service_ to prioritize LLM calls.
- **Multilevel Queues:** Discretize program priorities into tiers for fine-grained scheduling and preemption.
- **Anti-Starvation Policy:** Promotes long-running or delayed programs when their wait/service ratio exceeds a configurable threshold.
- **Load Balancer:** Distributes calls across engines, preserving KV-cache locality for long sequences and balancing load for short ones.

#### 🔁 Scheduler Flow
Program Invoked → LLM Calls Queued → ATLAS Scheduler
→ Multilevel Queues → Load Balancer → Engine Execution


---

## ⚙️ Integration

Both modules integrate transparently with existing agentic frameworks via an **Adapter Pattern**:

Agent Framework

│

├── Planner

│ ↓

│ [ UDF Module ] ← Parallel plan generation

│ ↓

├── Executor

│ ↓

└── [ RSM Backend ] ← Schedules and load-balances LLM calls


No modification to core agent logic is required; the modules intercept between planning and execution.

---

## 🧪 Example Usage 
udf_optimizer:
```python
from Agentic_system import planner, executor

def udf_optimizer():
# Analyze and parallelize the agent plan
    plan = planner.generate_plan(user_request)
    parallel_plan = ParallelPlanner().optimize(plan)

    executor.execute(parallel_plan)
    
```

RSM:
```python
from Agent_system import executor

def scheduler():
    llm_calls = executor.llm_calls()

    prioritized = atlas(llm_calls)
    load_balance = balancer(prioritized)

    load_balance.execute()
```

