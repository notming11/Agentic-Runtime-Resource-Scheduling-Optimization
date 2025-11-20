# Quick Start Guide - DeerFlow Parallelization

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Currently the implementation only supports the gemini API, this can easily be configured to use other service provieders by implementing a provider executor under /core

```bash
# Copy template
cp .env.example .env

# Edit .env and add your key
GEMINI_API_KEY=your_api_key_here
```

Get API key from: https://makersuite.google.com/app/apikey

### 3. Run Validation
```bash
cd src\udf-optimizer

python tests/validate.py
```

### 4. Run the System
```bash
cd src\udf-optimizer

# RUN e2e tests
python tests/test_e2e.py
```
---

## 📊 What You'll See

```
Plan: 10-City Research Plan
Total Steps: 12 (10 research + 2 processing)

Parallel Execution Time: 4.01s
Sequential Would Take: ~30s
Speedup: 7.5x

✓ All 12 steps completed successfully!
```

---

## 🎯 Key Features Demonstrated

1. **Parallel Execution**
   - All 10 city research steps run simultaneously
   - No waiting for one city before starting the next

2. **Intelligent Batching**
   - Analysis step waits for all research to complete
   - Report step waits for analysis
   - Dependencies respected automatically

3. **Real-Time Monitoring**
   - See each step start and complete
   - Track execution time
   - View results

4. **Error Handling**
   - Retry failed tasks automatically
   - Continue even if one step fails
   - Circuit breaker for systemic issues

---

## 📝 Example Code

### Create a Research Plan
```python
from workflow_types import Plan, Step, StepType

steps = [
    Step(
        title="Research Tokyo",
        description="Find top attractions",
        step_type=StepType.RESEARCH,
        need_search=True,
        step_id="step-1"
    ),
    # ... more cities ...
]

plan = Plan(
    title="Multi-City Research",
    locale="en-US",
    thought="Research cities in parallel",
    steps=steps
)
```

### Execute in Parallel
```python
from config_manager import get_example_config
from builder import build_workflow_graph
from workflow_types import State

# Get fast configuration
config = get_example_config("speed_optimized")

# Create state with plan
state = State(current_plan=plan)

# Build and execute workflow
graph = build_workflow_graph(config)
final_state = await execute_workflow(state, config)
```

---

## 🔧 Customization

### Edit Configuration
```yaml
# config.yaml
parallelization:
  enabled: true
  max_concurrent_tasks: 15    # Adjust this
  task_timeout_seconds: 300
  retry_on_failure: true
```

### Create Custom Plans
```python
# In main.py
def create_my_custom_plan():
    steps = [
        # Add your research steps
    ]
    return Plan(title="My Plan", steps=steps, ...)
```

---

## 📈 Performance Tips

1. **More Independent Steps = Greater Speedup**
   - 5 independent steps → 3-4x speedup
   - 10 independent steps → 7-8x speedup
   - 20 independent steps → 15-20x speedup

2. **Network I/O Benefits Most**
   - Web searches ✅
   - API calls ✅
   - Database queries ✅
   - CPU-intensive tasks ⚠️

3. **Configure for Your Use Case**
   - Fast networks → increase concurrency
   - Slow APIs → decrease concurrency
   - Rate limits → adjust tasks_per_second

---

## 🎓 Understanding the Output

### Execution Log
```
INFO:nodes:[Step 0] Starting execution: Research Tokyo
INFO:nodes:[Step 1] Starting execution: Research Paris
...
INFO:nodes:[Step 0] Completed in 4.00s
INFO:nodes:Batch completed in 4.01s
```

**What this means:**
- All steps started simultaneously
- Batch waits for slowest step (4.00s)
- Total batch time: 4.01s (not sum of all steps!)

### Results Summary
```
✓ Step 1: Research Tokyo attractions
   Result: Research completed for 'Research Tokyo'...

--- Performance ---
Total Execution Time: 4.01 seconds
Speedup: 7.5x
```

---

## 🐛 Troubleshooting

### "ImportError: cannot import name 'GenericAlias'"
**Fix:** File naming conflict resolved - using `workflow_types.py`

### "No module named 'yaml'"
**Fix:** `pip install pyyaml`

### Slower than expected?
**Check:**
- `max_concurrent_tasks` setting
- Network latency
- Task dependencies (are they actually independent?)

---

## 🎯 Next Steps

1. **Explore the Code**
   - `main.py` - Start here
   - `nodes.py` - Parallel execution logic
   - `builder.py` - Graph construction

2. **Modify Examples**
   - Change number of cities
   - Add different step types
   - Adjust configuration

3. **Integrate with DeerFlow**
   - See `IMPLEMENTATION_README.md`
   - Follow integration guide
   - Connect real agents

---

## 📚 More Information

- **Full Documentation:** `IMPLEMENTATION_README.md`
- **Technical Details:** `TECHNICAL_GUIDE.md`
- **Implementation Report:** Original specification
- **Summary:** `IMPLEMENTATION_SUMMARY.md`

---
