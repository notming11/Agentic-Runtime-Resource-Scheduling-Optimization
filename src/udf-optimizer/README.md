# UDF Optimizer - Parallel Execution System

Production-ready parallelization system for agent workflows with LLM-based dependency analysis and Gemini API integration.

## 📁 Project Structure

```
udf-optimizer/
├── core/                    # Core system components
│   ├── __init__.py         # Package exports
│   ├── workflow_types.py   # Data structures (Plan, Step, State, Config)
│   ├── nodes.py            # Execution orchestration
│   ├── builder.py          # Graph construction
│   ├── config_manager.py   # Configuration management
│   └── gemini_executor.py  # LLM integration (Gemini API)
│
├── config/                  # Configuration files
│   ├── parallel_prompt.md  # LLM system instruction for dependency analysis
│   └── config.yaml         # Runtime configuration presets
│
├── examples/                # Example files and tests
│   ├── example_response_1.txt  # Sample 10-step research plan (JSON)
│   ├── example_response.txt    # Alternative example plan
│   ├── example_prompt.txt      # Example user prompt
│   └── test_main.py            # Original Gemini API reference
│
├── tests/                   # Test and validation
│   └── validate.py         # System validation script
│
├── docs/                    # Documentation
│   ├── README.md           # This file (main documentation)
│   ├── QUICKSTART.md       # Quick start guide
│   ├── REAL_INTEGRATION_GUIDE.md      # Setup and configuration
│   ├── REAL_INTEGRATION_SUMMARY.md    # Technical implementation
│   ├── TECHNICAL_GUIDE.md             # Architecture details
│   ├── IMPLEMENTATION_README.md       # Implementation notes
│   ├── IMPLEMENTATION_SUMMARY.md      # Feature summary
│   ├── ARCHITECTURE_DIAGRAMS.md       # Visual diagrams
│   └── CHECKLIST.md                   # Development checklist
│
├── main.py                  # Demo with mock execution
├── main_real.py            # Demo with real Gemini API
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── .env                   # Your API key (gitignored)
└── .gitignore            # Git ignore rules
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Copy template
cp .env.example .env

# Edit .env and add your key
GEMINI_API_KEY=your_api_key_here
```

Get API key from: https://makersuite.google.com/app/apikey

### 3. Run Validation
```bash
python tests/validate.py
```

### 3. Run the System
```bash
# Real LLM execution (requires API key)
python main_real.py

# Or mock execution (no API key needed)
python main.py

# Run performance comparison (parallel vs sequential)
python compare_performance.py
```

## 📊 Performance Example

**10-Step Tourist Research Plan:**
- **Sequential**: ~100s (10s per step)
- **Parallel**: ~45s (2.2x speedup)
- **Efficiency**: Automatic LLM-based batching

## 🎯 Key Features

### 1. **LLM-Based Dependency Analysis**
- Automatically determines optimal task batching
- No hardcoded dependency rules
- Conservative parallelization (safe by default)

### 2. **Real Gemini API Integration**
- Dependency analysis using `config/parallel_prompt.md`
- Step execution with context awareness
- Research vs Processing task differentiation

### 3. **Production-Ready Architecture**
- Async/await for non-blocking I/O
- Rate limiting and retry logic
- Comprehensive error handling
- Detailed logging and metrics

### 4. **Flexible Configuration**
```python
from core import Configuration

# Speed preset
config = Configuration.from_preset("speed")

# Custom configuration
config = Configuration(
    max_concurrent_tasks=5,
    task_timeout_seconds=120,
    max_retries=2
)
```

## 📖 Documentation

- **[docs/README.md](docs/README.md)** - Full documentation
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Getting started guide
- **[docs/REAL_INTEGRATION_GUIDE.md](docs/REAL_INTEGRATION_GUIDE.md)** - Setup and troubleshooting
- **[docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md)** - Architecture deep dive

## 🧪 Testing

### Validation Test
```bash
python tests/validate.py
```
Checks all imports, configurations, and API key setup.

### Mock Execution (Fast)
```bash
python main.py
```
Tests architecture with simulated delays (~4s).

### Real Execution (Realistic)
```bash
python main_real.py
```
Full LLM integration with actual API calls (~45s).

### Performance Comparison
```bash
python compare_performance.py
```
Runs both parallel and sequential execution, captures metrics, and generates an LLM-analyzed performance report to `examples/example_performance_report.md`.

## 🔧 Configuration Presets

| Preset | Concurrent | Timeout | Retries | Use Case |
|--------|-----------|---------|---------|----------|
| **speed** | 5 | 120s | 2 | Development |
| **balanced** | 5 | 120s | 2 | Production |
| **reliability** | 3 | 180s | 3 | Critical |
| **cost** | 2 | 90s | 1 | Budget |

## 📦 Core Components

### `core/workflow_types.py`
Data structures: `Plan`, `Step`, `State`, `Configuration`, `StepType`

### `core/nodes.py`
- `parallel_research_team_node()` - Main orchestrator
- `_execute_batch_parallel()` - Concurrent execution
- `_execute_batch_sequential()` - Sequential with context

### `core/gemini_executor.py`
- `DependencyAnalyzer` - LLM-based dependency analysis
- `GeminiStepExecutor` - Step execution with Gemini
- `load_plan_from_json()` - Plan parser

### `core/builder.py`
- `build_parallel_workflow_graph()` - Parallel graph
- `build_sequential_workflow_graph()` - Sequential fallback

### `core/config_manager.py`
- `ConfigurationManager` - YAML config loader
- Preset configurations (speed, balanced, reliability, cost)

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found"
Create `.env` file with your API key.

### "Rate limit exceeded"
Reduce `max_concurrent_tasks` to 3 or lower.

### "Dependency analysis failed"
Check `config/parallel_prompt.md` exists. System will fallback to heuristic batching.

### "Import errors"
Run `pip install -r requirements.txt` to install dependencies.

## 💡 Usage Example

```python
from pathlib import Path
from core import (
    load_plan_from_json,
    parallel_research_team_node,
    State,
    Configuration
)
import asyncio

# Load plan
plan = load_plan_from_json(Path("examples/example_response_1.txt"))

# Create state
state = State(messages=[], observations=[], current_plan=plan)

# Configure
config = Configuration.from_preset("balanced")

# Execute
async def run():
    result = await parallel_research_team_node(state, config)
    return result

asyncio.run(run())
```

## 🌟 What's New

### v2.0 - Real LLM Integration
- ✅ Gemini 2.0 Flash for dependency analysis
- ✅ Gemini 2.0 Flash for step execution
- ✅ Context-aware prompting
- ✅ Production error handling
- ✅ Organized folder structure

### Architecture Improvements
- ✅ Modular `core/` package
- ✅ Separate `config/`, `examples/`, `tests/`, `docs/`
- ✅ Clean imports with `__init__.py`
- ✅ Relative imports within core modules

## 📄 Dependencies

```txt
google-generativeai  # Gemini API
python-dotenv        # Environment variables
pyyaml              # Configuration files
```

## 🚀 Next Steps

### For Production
1. Add web_search, crawl, python_repl tools
2. Implement streaming responses
3. Add response caching
4. Set up monitoring (Prometheus/Grafana)
5. Add checkpoint system for recovery

### For Experimentation
1. Try different prompting strategies
2. Test various concurrency levels
3. A/B test LLM vs heuristic analysis
4. Benchmark different plans

## 📝 License

See LICENSE file.

## 🙏 Credits

Implementation based on "Parallelization Implementation Report for DeerFlow" (2025-11-16).

---

**Status**: ✅ Production Ready | **Version**: 2.0 | **Last Updated**: November 2025
