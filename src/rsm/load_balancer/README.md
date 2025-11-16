# Autellix Load Balancer (Partition 3)

Implementation of the data locality-aware load balancer from the Autellix paper (Algorithm 2).

## Overview

The load balancer routes LLM inference requests across multiple vLLM engine instances using a simple but effective policy:

- **Small requests (≤2048 prefill tokens)**: Load balance using `LEAST_USED` - distribute across engines to avoid hotspots
- **Large requests (>2048 prefill tokens)**: Use locality - route to program's assigned engine to exploit KV-cache reuse

This design balances throughput (via load distribution) with efficiency (via KV-cache reuse for programs with long context).

## File Structure

```
load_balancer/
├── __init__.py           # Module exports and package initialization
├── types.py              # Type definitions and enums
├── engine_info.py        # Engine state tracking
└── balancer.py           # Main LoadBalancer implementation

tests/
└── test_load_balancer.py # Comprehensive test suite
```

## File Descriptions

### `load_balancer/__init__.py`

Package initialization file that exports the main classes:
- `LoadBalancer` - Main load balancer class
- `EngineInfo` - Engine state dataclass
- `RequestSize` - Request classification enum

**Purpose**: Provides clean imports for other modules using the load balancer.

**Example**:
```python
from load_balancer import LoadBalancer
```

---

### `load_balancer/types.py`

Type definitions and enumerations used throughout the load balancer.

**Contents**:
- `RequestSize` enum with two values:
  - `SMALL` - Requests with ≤2048 prefill tokens (use load balancing)
  - `LARGE` - Requests with >2048 prefill tokens (use locality)

**Purpose**: Centralizes type definitions to avoid circular imports and improve maintainability.

---

### `load_balancer/engine_info.py`

Data structure for tracking individual engine state.

**Key Class**: `EngineInfo`

**Attributes**:
- `engine_id`: Unique identifier for the engine
- `active_requests`: Current number of running requests (workload metric)
- `programs_assigned`: Set of program IDs (PIDs) assigned to this engine

**Methods**:
- `workload()`: Returns the current workload (active request count)

**Purpose**: Encapsulates engine state for the `LEAST_USED` selection algorithm.

---

### `load_balancer/balancer.py`

Main load balancer implementation following Algorithm 2 from the Autellix paper.

**Key Class**: `LoadBalancer`

**Core Methods**:
- `register_engine(engine_id)`: Add a new engine to the pool
- `unregister_engine(engine_id)`: Remove an engine from the pool
- `route_request(pid, num_tokens)`: Route a request to an appropriate engine (Algorithm 2)
- `complete_request(engine_id, pid)`: Decrement workload when request finishes
- `remove_program(pid)`: Clean up program state when session ends
- `get_stats()`: Return statistics about load distribution

**Algorithm Implementation**:

```python
def route_request(pid, num_tokens):
    if num_tokens <= 2048:  # Small request
        return LEAST_USED(engines)
    else:  # Large request
        if pid in program_table:
            return program_table[pid]
        else:
            engine = LEAST_USED(engines)
            program_table[pid] = engine
            return engine
```

**Internal State**:
- `_engines`: Dictionary of `EngineInfo` objects indexed by engine ID
- `_program_table`: Program-to-engine assignments (`pt` in Algorithm 2)
- Statistics counters for monitoring

**Purpose**: Implements the complete load balancing logic with thread-safe operations.

---

### `tests/test_load_balancer.py`

Comprehensive test suite verifying correctness of the load balancer.

**Test Classes**:

1. **`TestRequestClassification`**
   - Verifies 2048 token threshold
   - Tests boundary conditions (2048 vs 2049 tokens)

2. **`TestSmallRequestLoadBalancing`**
   - Confirms small requests use `LEAST_USED`
   - Tests load distribution across engines

3. **`TestLargeRequestLocality`**
   - Confirms large requests use program table (`pt`)
   - Tests locality hit tracking
   - Verifies first large request uses `LEAST_USED` then assigns

4. **`TestMixedRequestPatterns`**
   - Tests programs with both small and large requests
   - Verifies small requests ignore `pt` table
   - Verifies large requests maintain locality

5. **`TestEngineManagement`**
   - Tests engine registration/unregistration
   - Verifies cleanup of program assignments

6. **`TestRequestCompletion`**
   - Tests workload decrement on completion
   - Verifies workload affects routing decisions

7. **`TestProgramRemoval`**
   - Tests cleanup when programs/sessions end
   - Verifies `pt` table entries are cleared

8. **`TestStatistics`**
   - Tests statistics tracking
   - Verifies request counting

9. **`TestEdgeCases`**
   - Tests error conditions (no engines, removed engines)
   - Tests boundary cases (zero tokens)

**Running Tests**:
```bash
python -m pytest tests/test_load_balancer.py -v
# or
python tests/test_load_balancer.py
```

**Purpose**: Ensures correctness and provides regression protection.

---

## Integration with Other Partitions

### Partition 1: Frontend & Process Management

The load balancer receives the `GlobalProcessTable` in its constructor for potential future integration:

```python
from frontend.process_table import GlobalProcessTable
from load_balancer import LoadBalancer

process_table = GlobalProcessTable()
load_balancer = LoadBalancer(process_table)
```

### Partition 2: Scheduler

The scheduler queries the load balancer to route requests:

```python
from scheduler.plas_scheduler import PLASScheduler
from load_balancer import LoadBalancer

scheduler = PLASScheduler(process_table)
load_balancer = LoadBalancer(process_table)

# When scheduler needs to route a request
engine_id = load_balancer.route_request(
    pid=request.pid,
    num_tokens=request.prefill_tokens
)
```

### Partition 4: Engine Manager

The engine manager registers engines and updates workload:

```python
from engine.multi_engine_manager import MultiEngineManager
from load_balancer import LoadBalancer

engine_manager = MultiEngineManager(num_engines=4)
load_balancer = LoadBalancer(process_table)

# Register all engines
for engine in engine_manager.engines:
    load_balancer.register_engine(engine.id)

# After request completion
load_balancer.complete_request(engine_id, pid)
```

---

## Usage Example

```python
from load_balancer import LoadBalancer
from frontend.process_table import GlobalProcessTable

# Initialize
process_table = GlobalProcessTable()
lb = LoadBalancer(process_table)

# Register engines
lb.register_engine("engine_0")
lb.register_engine("engine_1")
lb.register_engine("engine_2")

# Route a small request (uses LEAST_USED)
engine = lb.route_request(
    pid="session_123",
    num_tokens=1500  # Small request
)
print(f"Small request routed to: {engine}")

# Route a large request (uses locality)
engine = lb.route_request(
    pid="session_123",
    num_tokens=3000  # Large request - will assign to engine
)
print(f"Large request routed to: {engine}")

# Subsequent large requests use same engine
engine = lb.route_request(
    pid="session_123",
    num_tokens=3500  # Will reuse assigned engine
)
print(f"Second large request routed to: {engine}")

# Complete requests to update workload
lb.complete_request(engine, "session_123")

# Get statistics
stats = lb.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Programs in table: {stats['programs_in_table']}")
print(f"Locality hits: {stats['locality_hits']}")
```

---

## Key Design Decisions

### 1. Thread Safety
All public methods use `threading.RLock()` for thread-safe access, enabling concurrent calls from multiple components (scheduler, engine manager).

### 2. Simple Workload Metric
Workload is measured as the count of active requests per engine. This simple metric is sufficient for the load balancing algorithm and avoids complex metrics.

### 3. Program Table (`pt`)
The `_program_table` dictionary implements the `pt` data structure from Algorithm 2, mapping program IDs to their assigned engines for locality.

### 4. No Decode Token Estimation
Following the Autellix paper exactly, only **prefill tokens** are used for classification. The decode phase length is not considered.

### 5. Graceful Engine Removal
When an engine is unregistered while programs are assigned to it, the load balancer automatically reassigns those programs on their next request.

---

## Algorithm 2 Reference

From the Autellix paper:

```
procedure LOAD_BALANCER(Call c, Table pt, List Engines)
    if LEN(c.tokens) ≤ 2048 then              ▷ Small request
        assigned_engine = LEAST_USED(Engines)
    else
        if c.pid ∈ pt then                     ▷ Program already assigned
            assigned_engine = pt[c.pid]
        else
            assigned_engine = LEAST_USED(Engines)
            pt[c.pid] = assigned_engine        ▷ Assign program to engine
        end if
    end if
    return assigned_engine
end procedure
```

---

## Statistics and Monitoring

The `get_stats()` method returns comprehensive statistics:

```python
{
    "total_requests": 100,
    "small_requests": 60,
    "large_requests": 40,
    "locality_hits": 35,          # Large requests using existing assignment
    "locality_assigns": 5,        # Large requests creating new assignment
    "programs_in_table": 10,      # Number of programs assigned to engines
    "engines": {
        "engine_0": {
            "active_requests": 15,
            "programs_assigned": 3,
            "workload": 15
        },
        ...
    }
}
```

---

## Future Extensions

Potential enhancements not in the current implementation:

1. **Advanced Workload Metrics**: Incorporate queue depth, memory usage, or GPU utilization
2. **Dynamic Threshold**: Adapt the 2048 token threshold based on system load
3. **Engine Affinity Hints**: Allow manual program-to-engine assignments
4. **Load Prediction**: Use historical data to predict request completion times
5. **Multi-tier Routing**: Add region/datacenter awareness for distributed deployments

---

## References

- **Autellix Paper**: "Autellix: A Fair Multi-Program LLM Serving System with Program-Level Scheduling"
- **Algorithm 2**: Load balancing algorithm (Section 4.2.3)
- **Related Work**: PLAS/ATLAS schedulers (Partition 2), vLLM engine integration (Partition 4)

---

## License

This implementation is part of the Autellix RSM project and follows the same license as the main project.