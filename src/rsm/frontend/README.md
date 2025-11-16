# Autellix RSM Frontend

This module implements the stateful API layer and process management for the Autellix Resource Scheduler Module (RSM), based on the paper "Autellix: An Efficient Serving Engine for LLM Agents as General Programs" (Luo et al., 2025).

## Architecture Overview

The frontend consists of three main components:

### 1. Process Table (`process_table.py`)

Global process table that tracks runtime metadata for all active programs (Autellix Section 4.2.1).

**Key Features:**
- Thread-safe, in-memory registry of running programs
- Tracks per-program metrics:
  - **Service Time**: Cumulative execution time (PLAS) or longest critical path (ATLAS)
  - **Waiting Time**: Total time in scheduler queues (for anti-starvation)
  - **Engine Affinity**: Engines processing this program (for KV-cache locality)
  - **Thread Metadata**: Active LLM calls with arrival, waiting, and service times
  - **Starvation Counters**: Wait-to-service ratio for promotion triggers

**Classes:**
- `GlobalProcessTable`: Main process table with thread-safe operations
- `ProgramEntry`: Entry for a single program
- `ThreadMetadata`: Metadata for a single LLM call (thread)
- `ProgramState`: Enum for program states (RUNNING, WAITING, COMPLETED, FAILED)

### 2. Session Manager (`session_manager.py`)

Manages session lifecycle for the stateful API (Autellix Section 5).

**Key Features:**
- Creates unique session IDs for programs
- Manages session lifecycle (`start_session`, `end_session`)
- Integrates with process table
- Tracks session-to-program mappings
- Handles session cleanup and timeout

**Classes:**
- `SessionManager`: Main session lifecycle manager
- `SessionInfo`: Information about a session
- `SessionState`: Enum for session states (ACTIVE, COMPLETED, FAILED, TIMEOUT)

### 3. API Wrapper (`api_wrapper.py`)

Stateful API layer extending OpenAI/vLLM Python APIs (Autellix Section 5).

**Key Features:**
- Provides stateful interface that appears stateless to developers
- Automatically manages sessions on initialization
- Annotates LLM calls with session, program, and thread IDs
- Tracks timing information for scheduling
- OpenAI-compatible adapter for easy migration

**Classes:**
- `AutellixClient`: Main stateful client
- `AutellixOpenAIAdapter`: OpenAI-compatible adapter
- `autellix_session`: Context manager for sessions

## Usage Examples

### Example 1: Single-Threaded Chatbot (PLAS Scheduling)

```python
from rsm.frontend import AutellixClient

# Create client for single-threaded program
client = AutellixClient(
    backend_url="http://localhost:8000",
    is_multithreaded=False,  # Use PLAS scheduling
    session_metadata={"app": "chatbot"}
)

# Make LLM calls
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    model="llama-3.1-8b"
)

# Get session statistics
stats = client.get_session_stats()
print(f"Service time: {stats['service_time']:.2f}s")
print(f"Waiting time: {stats['waiting_time']:.2f}s")

# Close session
client.close()
```

### Example 2: Multi-Threaded Agent (ATLAS Scheduling)

```python
from rsm.frontend import autellix_session

# Use context manager for automatic cleanup
with autellix_session(is_multithreaded=True) as client:
    # Step 1: Planning
    planning_response = client.chat_completion(
        messages=[{"role": "user", "content": "Research 3 cities"}]
    )
    planning_thread = planning_response["autellix_metadata"]["thread_id"]

    # Step 2: Parallel execution (Map)
    research_requests = [
        {"messages": [{"role": "user", "content": f"Research city {i}"}]}
        for i in range(3)
    ]
    research_responses = client.parallel_chat_completion(
        requests=research_requests,
        parent_thread_id=planning_thread
    )

    # Step 3: Aggregation (Reduce)
    thread_ids = [r["autellix_metadata"]["thread_id"] for r in research_responses]
    final_response = client.chat_completion(
        messages=[{"role": "user", "content": "Aggregate results"}],
        parent_thread_ids=thread_ids
    )

# Session automatically closed
```

### Example 3: OpenAI-Compatible Adapter

```python
from rsm.frontend import AutellixOpenAIAdapter

# Use OpenAI-compatible adapter for easy migration
with AutellixOpenAIAdapter(backend_url="http://localhost:8000") as client:
    # Same interface as OpenAI client
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.7
    )
```

### Example 4: Direct Process Table Access

```python
from rsm.frontend import GlobalProcessTable

# Create process table
process_table = GlobalProcessTable()

# Create a program
pid = "program_123"
process_table.create_program(pid, is_multithreaded=True)

# Add LLM calls
thread = process_table.add_llm_call(
    pid=pid,
    call_id="call_1",
    thread_id="thread_1",
    prefill_tokens=100,
    engine_id="engine_0"
)

# Update metrics after completion
process_table.update_program_metrics(
    pid=pid,
    call_id="call_1",
    service_time=1.5,
    waiting_time=0.3
)

# Get program statistics
program = process_table.get_program(pid)
print(f"Service time: {program.service_time:.2f}s")
print(f"Starvation ratio: {program.get_starvation_ratio():.2f}")

# Cleanup
process_table.remove_program(pid)
```

## Key Concepts

### PLAS vs ATLAS Scheduling

**PLAS (Program-Level Attained Service)** - For single-threaded programs:
- Priority based on cumulative service time: `p(c_j) = Σ t_k`
- Shorter programs complete first
- Simple, efficient for sequential workflows

**ATLAS (Adaptive Thread-Level Attained Service)** - For multi-threaded programs:
- Priority based on critical path: `p(c_j) = max{p(c_k) + t_k}`
- Estimates longest path through DAG online
- Minimizes program makespan
- Prevents straggler threads from delaying completion

### Anti-Starvation Mechanism

Programs are promoted to highest priority when:
```
W_total / T_total ≥ β
```
Where:
- `W_total`: Total waiting time
- `T_total`: Total service time
- `β`: Starvation threshold (configurable)

### Data Locality

Process table tracks engine assignments for KV-cache locality:
- Short calls (≤2048 tokens): Route to least-loaded engine
- Long calls (>2048 tokens): Route to program's engine for cache reuse

## API Reference

### AutellixClient

```python
client = AutellixClient(
    backend_url: str = "http://localhost:8000",
    is_multithreaded: bool = False,
    session_metadata: Optional[Dict] = None,
    auto_start_session: bool = True
)
```

**Methods:**
- `chat_completion(messages, model, temperature, max_tokens, stream, thread_id, parent_thread_ids, **kwargs)` - Send chat completion request
- `parallel_chat_completion(requests, parent_thread_id)` - Send parallel requests
- `get_session_stats()` - Get session statistics
- `close()` - Close session

### GlobalProcessTable

```python
process_table = GlobalProcessTable()
```

**Methods:**
- `create_program(pid, is_multithreaded)` - Create program entry
- `remove_program(pid)` - Remove program entry
- `get_program(pid)` - Get program entry
- `add_llm_call(pid, call_id, thread_id, prefill_tokens, engine_id, parent_thread_ids)` - Register LLM call
- `complete_llm_call(pid, thread_id)` - Mark call as completed
- `update_program_metrics(pid, call_id, service_time, waiting_time)` - Update metrics
- `get_program_priority(pid)` - Get program priority
- `get_starvation_ratio(pid)` - Get starvation ratio
- `get_stats()` - Get global statistics

### SessionManager

```python
session_manager = SessionManager(process_table)
```

**Methods:**
- `start_session(is_multithreaded, session_id, metadata)` - Start new session
- `end_session(session_id, state)` - End session
- `get_session(session_id)` - Get session info
- `is_active(session_id)` - Check if session is active
- `register_llm_call(session_id, call_id, thread_id, prefill_tokens, engine_id, parent_thread_ids)` - Register call
- `complete_llm_call(session_id, thread_id, service_time, waiting_time)` - Complete call
- `get_session_stats(session_id)` - Get session statistics

## Testing

Run tests with pytest:

```bash
cd src/rsm
pytest tests/test_frontend.py -v
```

Run examples:

```bash
cd src/rsm
python examples/basic_usage.py
```

## Performance Considerations

### Thread Safety
- All components use `threading.RLock()` for thread-safe concurrent access
- Safe to use from multiple threads or async contexts

### Memory Management
- Process table automatically cleaned up on session end
- Stale sessions cleaned up via `cleanup_inactive_sessions()`
- Consider implementing periodic cleanup for long-running deployments

### Scalability
- Process table uses in-memory dictionary (O(1) lookups)
- Thread metadata stored per active LLM call
- Memory scales with number of concurrent programs and active calls

## Integration with Backend

The frontend generates requests with Autellix-specific metadata:

```python
{
    "messages": [...],
    "model": "llama-3.1-8b",
    # Standard fields
    "temperature": 0.7,
    "max_tokens": 100,

    # Autellix metadata
    "autellix_session_id": "session_abc123",
    "autellix_call_id": "call_xyz789",
    "autellix_thread_id": "thread_def456",
    "autellix_parent_threads": ["thread_abc123"],
    "autellix_is_multithreaded": True
}
```

Backend components (scheduler, load balancer) use this metadata for:
- Priority assignment (PLAS/ATLAS)
- Queue selection (multilevel queues)
- Engine routing (locality-aware)
- Anti-starvation promotion

## References

- Autellix Paper: "An Efficient Serving Engine for LLM Agents as General Programs" (Luo et al., 2025)
- Section 4.2.1: Process Table
- Section 5: Frontend Implementation
- Algorithm 1: Program-Aware Scheduler
- Equation 1: PLAS Priority
- Equation 2: ATLAS Priority
