# Real LLM Integration - Quick Start Guide

## Overview
This system now uses **real Gemini API calls** for:
1. **Dependency Analysis**: LLM analyzes the plan and creates optimal execution batches
2. **Step Execution**: Each research/processing step is executed using Gemini 2.0 Flash

## Setup

### 1. Install Dependencies
```bash
pip install google-generativeai python-dotenv
```

### 2. Configure API Key
Create a `.env` file in this directory:
```bash
GEMINI_API_KEY=your_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the System
```bash
python main_real.py
```

## Architecture

### Flow Diagram
```
example_response_1.txt (input plan)
    ↓
load_plan_from_json()
    ↓
DependencyAnalyzer.analyze_plan()
    ├─ Reads parallel_prompt.md
    ├─ Calls Gemini API
    └─ Returns BatchDefinition[] 
    ↓
parallel_research_team_node()
    ├─ Executes each batch (parallel/sequential)
    └─ Uses GeminiStepExecutor for each step
        ↓
GeminiStepExecutor.execute_step()
    ├─ Builds context-aware prompt
    ├─ Calls Gemini API
    └─ Returns execution result
```

### Key Components

#### 1. `gemini_executor.py`
- **DependencyAnalyzer**: Uses Gemini to analyze dependencies
  - System instruction: `parallel_prompt.md`
  - Input: Plan JSON
  - Output: Batches with parallel/sequential flags
  
- **GeminiStepExecutor**: Executes individual steps
  - Research steps: General information gathering
  - Processing steps: Data analysis with previous context
  
- **load_plan_from_json()**: Parses JSON into Plan object

#### 2. `nodes.py` (Updated)
- **parallel_research_team_node()**: Main orchestration
  - Calls DependencyAnalyzer to get batches
  - Executes batches in order
  - Aggregates results between batches
  
- **_execute_batch_parallel()**: Parallel execution using asyncio.gather
- **_execute_batch_sequential()**: Sequential execution with context passing

#### 3. `main_real.py`
- End-to-end demo script
- Loads example_response_1.txt
- Configures for speed (5 concurrent tasks)
- Shows execution metrics and speedup

## Example Execution

```
=== Starting Dependency Analysis with LLM ===
Sending plan to LLM for dependency analysis...
  Batch 1 (parallel): 7 steps - All research steps can run independently
  Batch 2 (sequential): 1 step - Collate all data (depends on research)
  Batch 3 (sequential): 1 step - Calculate costs (depends on collation)
  Batch 4 (sequential): 1 step - Rank cities (depends on calculation)

LLM Analysis: Research steps have no dependencies...
Expected Speedup: ~3-4x (7 parallel research + 3 sequential processing)
Created 4 batches

============================================================
Executing Batch 1 (parallel): 7 steps - All research steps...
============================================================
[Step 0] Starting execution: Identify Top 10 Most Visited Cities
[Step 1] Starting execution: Research Cities 1-3...
[Step 2] Starting execution: Research Cities 4-6...
...
[All steps complete in ~15s instead of 70s sequential]

Total execution time: 42.15s
Sequential estimate: 100.00s
Speedup: 2.37x
```

## Configuration

### Speed vs Reliability
Adjust in `main_real.py`:

```python
# For maximum speed (may hit rate limits)
config = Configuration(
    max_concurrent_tasks=10,
    max_retries=1
)

# For maximum reliability (slower)
config = Configuration(
    max_concurrent_tasks=3,
    max_retries=3,
    failure_mode="fail_fast"
)
```

### Timeout Tuning
- `task_timeout_seconds`: Max time per step (default: 120s)
- Research steps typically take 10-20s
- Processing steps may take longer with large context

## Testing

### 1. Test with Mock (No API Key Required)
```bash
python main.py
```
Uses the original mock execution for testing architecture.

### 2. Test with Real API
```bash
python main_real.py
```
Requires GEMINI_API_KEY in .env file.

### 3. Custom Plans
Create your own JSON plan following example_response_1.txt format:
```python
from gemini_executor import load_plan_from_json
plan = load_plan_from_json(Path("your_plan.json"))
```

## Troubleshooting

### "GEMINI_API_KEY not found"
- Create `.env` file in src/udf-optimizer/
- Add: `GEMINI_API_KEY=your_key`
- Restart Python process

### "Rate limit exceeded"
- Reduce `max_concurrent_tasks` to 3
- Increase `retry_backoff_seconds`
- Check API quota at Google AI Studio

### "Dependency analysis failed"
- Check `parallel_prompt.md` exists
- LLM will fallback to heuristic batching
- Check logs for error details

### Slow Execution
- Increase `max_concurrent_tasks` (if not hitting rate limits)
- Reduce `task_timeout_seconds` for faster failures
- Use "fail_fast" mode to stop on errors early

## Performance Tips

1. **Batch Size**: LLM creates optimal batches automatically
2. **Concurrency**: Start with 5, increase if no rate limit errors
3. **Retries**: Use 2-3 for production, 1 for testing
4. **Context**: Processing steps use last 3 research results for context

## Next Steps

1. **Custom Agents**: Replace GeminiStepExecutor with domain-specific agents
2. **Tool Integration**: Add web_search, crawl, python_repl tools
3. **Streaming**: Use streaming responses for longer tasks
4. **Caching**: Cache LLM responses to avoid redundant API calls
5. **Monitoring**: Add Prometheus/Grafana for production metrics

## Credits

Based on "Parallelization Implementation Report for DeerFlow" (2025-11-16)
Implements LLM-based dependency analysis strategy from TECHNICAL_GUIDE.md
