# ⚙️ Resource Scheduler Module (RSM)
_A program-aware scheduling and load balancing system for LLM-driven agentic workflows._

---

## 🧩 Overview

The **Resource Scheduler Module (RSM)** is a runtime layer that manages the **scheduling, prioritization, and distribution of LLM calls** across multiple engines.  
It extends concepts from **Autellix (Luo et al., 2025)** and is designed to optimize **multi-agent workflows** where large language model (LLM) calls represent the dominant execution cost.

By treating each agentic workflow as a **program composed of interdependent tasks** (i.e., a dynamic DAG), the RSM reduces head-of-line blocking, preserves data locality, and ensures fairness across concurrent sessions.

---

## 🎯 Objectives

- Optimize throughput and latency in multi-agent workflows.
- Maintain fairness using **Anti-Starvation mechanisms**.
- Exploit **KV-cache locality** for long LLM calls.
- Support scalable, distributed LLM serving using vLLM or similar backends.

---

## 🏗️ Architecture

### System Overview
Program Invoked

│

▼

[Process Table] ← tracks program metadata

│

▼

[ATLAS Scheduler] ← prioritizes critical paths

│

▼

[Multilevel Queues] ← manages discrete priority classes

│

▼

[Load Balancer] ← routes to least-loaded / local engine

│

▼

LLM Engine(s)


The RSM executes in three main stages:  
1. **Program Tracking:** Maintain runtime metadata in the Process Table.  
2. **Scheduling:** Use ATLAS and multilevel queues to prioritize active programs.  
3. **Load Balancing:** Route LLM calls across engines to minimize recomputation.

---

## ⚙️ Core Components

### 1️⃣ Process Table
A global in-memory registry that tracks per-program runtime metrics:

| Field | Description |
|--------|-------------|
| **PID** | Unique program identifier |
| **Service Time** | Longest observed cumulative runtime (critical path length) |
| **Waiting Time** | Total time spent in scheduler queues |
| **Engine ID(s)** | Assigned LLM engine(s) for locality preservation |
| **Thread Metadata** | Active LLM calls (arrival time, wait time, service time) |
| **Starvation Counters** | Ratio of wait time to service time (for promotion triggers) |

This table enables **program-level scheduling** without explicit DAG traversal.

---

### 2️⃣ Adaptive Thread-Level Attained Service (ATLAS)

The ATLAS policy estimates the **critical path** of a program’s DAG *online* by tracking the longest cumulative service time among all threads in the same program.

**Key idea:**  
> Calls belonging to longer critical paths are prioritized to minimize overall makespan.

**Priority formula:**

$
p(c_j) =
\begin{cases}
0, & \text{if } c_j \text{ is a root node} \\
\max_{c_k \in P(c_j)} (p(c_k) + t_k), & \text{otherwise}
\end{cases}
$

In practice, each program maintains a single scalar `service_time` value in the process table, updated whenever a call finishes. This provides a fast, online approximation of the critical path length.

---

### 3️⃣ Multilevel Program-Based Scheduler

To prevent head-of-line blocking, RSM discretizes program priorities into **K queues** `{Q1, Q2, …, QK}`.  
Each queue represents a service time range `[Qlo_i, Qhi_i)`.

**Scheduling policy:**
- New LLM calls are inserted into a queue based on their program’s cumulative service time.
- Within each queue, calls execute **FIFO**.
- Higher-priority queues preempt lower ones.
- Expired calls (after time quantum) are **demoted** one level down.

This design provides predictable fairness while preserving GPU utilization.

---

### 4️⃣ Anti-Starvation Mechanism

Discrete priority systems can starve long-running programs.  
RSM prevents this by monitoring the **wait-to-service ratio**:

$
\frac{W_{total}}{T_{total}} \ge \beta
$

If a program exceeds this threshold `β`, its next call is **promoted to the highest priority queue (Q1)**.  
All active threads for that program are promoted together, ensuring synchronized fairness.

---

### 5️⃣ Load Balancer (Data Locality–Aware Routing)

To minimize KV-cache recomputation, RSM’s load balancer distinguishes between **short** and **long** LLM calls:

| Input Length | Routing Policy | Rationale |
|---------------|----------------|------------|
| ≤ 2048 tokens | Least-loaded engine | Cache reuse negligible |
| > 2048 tokens | Same engine as previous program call | Preserve KV-cache locality |

This hybrid routing policy yields up to **1.4× higher throughput** over standard round-robin load balancing.

---

## 🧠 Pseudocode Reference

```python
def scheduler(queues, process_table):
    for call in incoming_llm_calls:
        # Assign queue based on cumulative service time
        q_idx = determine_queue(call.pid, process_table)
        queues[q_idx].append(call)

    for q in queues:  # high → low priority
        for c in q:
            if c.finished:
                update_process_table(c)
            elif c.quanta_expired():
                demote(c)
            elif starvation_ratio(c.pid) >= beta:
                promote(c)
