# Local LLM Setup Guide for UDF Optimizer

This guide explains how to configure the UDF Optimizer to use local LLM inference instead of cloud APIs like Gemini.

## Overview

The UDF Optimizer now supports multiple LLM backends:
- **Local LLM servers** (vLLM, llama.cpp, Ollama, text-generation-webui, LocalAI)
- **Google Gemini** (original implementation)
- **OpenAI** (cloud or compatible)

All backends use a unified interface, so you can switch between them by just changing the configuration.

## Quick Start

### 1. Install Dependencies

```bash
# For local LLM support (OpenAI-compatible API client)
pip install openai

# Optional: For Gemini support
pip install google-generativeai

# For YAML configuration loading
pip install pyyaml
```

### 2. Configure Backend

Edit `src/udf-optimizer/config/config.yaml`:

```yaml
llm:
  backend: "local"  # Change this!
  local_api_base: "http://localhost:8000/v1"
  local_model: "your-model-name"
```

### 3. Run

```bash
python examples/demo_workflow.py
```

That's it! The system will now use your local LLM instead of Gemini.

---

## Supported Local LLM Servers

### Option 1: vLLM (Recommended)

**Why vLLM?**
- Fastest inference for production
- Built-in OpenAI-compatible API
- Excellent batching and throughput
- GPU acceleration

**Installation:**
```bash
pip install vllm
```

**Start server:**
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key your-api-key  # Optional
```

**Configuration:**
```yaml
llm:
  backend: "local"
  local_api_base: "http://localhost:8000/v1"
  local_model: "meta-llama/Llama-3.1-8B-Instruct"
  local_api_key: "your-api-key"  # If you set --api-key
```

**Recommended models:**
- `meta-llama/Llama-3.1-8B-Instruct` (good balance)
- `meta-llama/Llama-3.1-70B-Instruct` (best quality, needs more GPU)
- `Qwen/Qwen2.5-14B-Instruct` (excellent reasoning)
- `mistralai/Mistral-7B-Instruct-v0.3` (fast, good quality)

---

### Option 2: Ollama

**Why Ollama?**
- Easiest to set up
- Good for development/testing
- Automatic model downloads
- Simple CLI

**Installation:**
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com
```

**Start server:**
```bash
# Pull model
ollama pull llama3.1:8b

# Server starts automatically
# OpenAI-compatible API at http://localhost:11434/v1
```

**Configuration:**
```yaml
llm:
  backend: "local"
  local_api_base: "http://localhost:11434/v1"
  local_model: "llama3.1:8b"
  local_api_key: null  # Ollama doesn't need API keys
```

**Recommended models:**
- `llama3.1:8b` - Good general purpose
- `qwen2.5:14b` - Excellent reasoning
- `gemma2:9b` - Fast and capable
- `mixtral:8x7b` - Mixture of experts, good quality

---

### Option 3: llama.cpp Server

**Why llama.cpp?**
- Works on CPU (no GPU required)
- Efficient quantization (GGUF models)
- Cross-platform

**Installation:**
```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Or use pre-built releases
```

**Start server:**
```bash
# Download a GGUF model from HuggingFace
# Example: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

./server -m models/llama-2-7b-chat.Q4_K_M.gguf \
  --port 8000 \
  --host 0.0.0.0 \
  -c 4096 \  # Context length
  -ngl 35    # GPU layers (0 for CPU-only)
```

**Configuration:**
```yaml
llm:
  backend: "local"
  local_api_base: "http://localhost:8000/v1"
  local_model: "local-model"  # llama.cpp doesn't use model names
  local_api_key: null
```

---

### Option 4: Text Generation WebUI

**Why text-generation-webui?**
- User-friendly web interface
- Supports many model formats
- Easy model switching

**Installation:**
```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
./start_linux.sh  # or start_windows.bat, start_macos.sh
```

**Start OpenAI API:**
1. Open web UI at http://localhost:7860
2. Load a model
3. Go to Session tab → Enable "OpenAI" extension
4. OpenAI API will be at http://localhost:5001/v1

**Configuration:**
```yaml
llm:
  backend: "local"
  local_api_base: "http://localhost:5001/v1"
  local_model: "your-loaded-model"
```

---

### Option 5: LocalAI

**Why LocalAI?**
- OpenAI drop-in replacement
- Supports images, audio, embeddings
- Docker-friendly

**Installation:**
```bash
# Using Docker
docker run -p 8080:8080 \
  -v $PWD/models:/models \
  localai/localai:latest \
  --models-path /models \
  --context-size 4096
```

**Configuration:**
```yaml
llm:
  backend: "local"
  local_api_base: "http://localhost:8080/v1"
  local_model: "your-model-name"
```

---

## Configuration Reference

### Complete Configuration Example

```yaml
# In config/config.yaml

llm:
  # Backend: "local", "gemini", or "openai"
  backend: "local"

  # === Local LLM Settings ===
  local_api_base: "http://localhost:8000/v1"
  local_model: "meta-llama/Llama-3.1-8B-Instruct"
  local_api_key: null  # Set if your server requires it

  # === Gemini Settings ===
  gemini_api_key: null  # Or set GEMINI_API_KEY env var
  gemini_model: "gemini-2.0-flash-exp"

  # === OpenAI Settings ===
  openai_api_key: null  # Or set OPENAI_API_KEY env var
  openai_api_base: null  # null = default OpenAI endpoint
  openai_model: "gpt-4"

  # === Generation Parameters ===
  temperature: 0.7      # Randomness (0=deterministic, 1=creative)
  max_tokens: 4096      # Maximum output length
  top_p: 0.9            # Nucleus sampling
  timeout_seconds: 120  # Request timeout
```

### Environment Variables

You can also configure via environment variables:

```bash
# Backend selection
export LLM_BACKEND="local"

# Local LLM
export LOCAL_LLM_API_BASE="http://localhost:8000/v1"
export LOCAL_LLM_MODEL="llama3.1:8b"
export LOCAL_LLM_API_KEY="optional-key"

# Gemini
export GEMINI_API_KEY="your-gemini-key"
export GEMINI_MODEL="gemini-2.0-flash-exp"

# OpenAI
export OPENAI_API_KEY="your-openai-key"
```

---

## Model Recommendations

### For Dependency Analysis (Most Important)

Dependency analysis requires strong reasoning. Recommended models:

**Best Quality:**
- `Qwen/Qwen2.5-72B-Instruct` (needs 48GB+ VRAM)
- `meta-llama/Llama-3.1-70B-Instruct` (needs 48GB+ VRAM)

**Good Balance (8-14B):**
- `Qwen/Qwen2.5-14B-Instruct` ⭐ (Recommended)
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

**Fast/Development (≤7B):**
- `microsoft/Phi-3.5-mini-instruct` (3.8B, surprisingly capable)
- `google/gemma-2-9b-it`

### For Step Execution

Step execution is less critical (mostly text generation):

**Any of the above models work well**

For maximum speed, you can even use smaller models for execution:
```yaml
# Use separate configs for analyzer vs executor
# (Advanced: requires code modification)
```

### Hardware Requirements

| Model Size | VRAM Needed | Speed (tokens/s) | Quality |
|------------|-------------|------------------|---------|
| 3-4B       | 4GB         | ~80              | Basic   |
| 7-8B       | 8GB         | ~50              | Good    |
| 13-14B     | 16GB        | ~30              | Better  |
| 70B        | 48GB+       | ~10              | Best    |

*Speed estimates for 4-bit quantization on A100/H100*

---

## Testing Your Setup

### 1. Test LLM Connection

```python
from udf_optimizer.core.llm_client import create_llm_client, LLMConfig

# Create client
config = LLMConfig(
    backend="local",
    local_api_base="http://localhost:8000/v1",
    local_model="llama3.1:8b"
)
client = create_llm_client(config)

# Test generation
result = client.generate_sync(
    prompt="What is 2+2?",
    system_prompt="You are a helpful assistant."
)
print(result)
```

### 2. Test Dependency Analysis

```python
from udf_optimizer.core.executor import DependencyAnalyzer
from pathlib import Path

# Create analyzer (loads from config.yaml)
analyzer = DependencyAnalyzer()

# Load test plan
from udf_optimizer.core.executor import load_plan_from_json
plan = load_plan_from_json(Path("examples/example_plan.json"))

# Analyze
batches = analyzer.analyze_plan(plan)

print(f"Created {len(batches)} batches:")
for batch in batches:
    print(f"  {batch}")
```

### 3. Run Full Workflow

```bash
cd examples
python demo_workflow.py
```

---

## Troubleshooting

### "Connection refused" error

**Problem:** Can't connect to local LLM server

**Solutions:**
1. Make sure server is running: `curl http://localhost:8000/v1/models`
2. Check port number in config matches server
3. Check firewall settings
4. Try `0.0.0.0` instead of `localhost` in server startup

### "Model not found" error

**Problem:** Server doesn't recognize model name

**Solutions:**
1. For vLLM: Use exact model path/name from startup
2. For Ollama: List models with `ollama list`
3. For llama.cpp: Use `"local-model"` (it doesn't use names)
4. Check server logs for available models

### "JSON parsing failed" error

**Problem:** LLM didn't return valid JSON

**Solutions:**
1. Increase `max_tokens` in config (may have been cut off)
2. Try a more capable model (7B+ recommended)
3. Check `temperature` (lower = more reliable, try 0.3)
4. Enable JSON mode if your server supports it

### Slow performance

**Problem:** Generation is very slow

**Solutions:**
1. Check GPU utilization: `nvidia-smi`
2. For vLLM: Increase `--max-num-seqs` and `--max-model-len`
3. For CPU: Use quantized models (Q4_K_M, Q5_K_M)
4. Reduce `max_tokens` in config
5. Use smaller model (8B vs 70B)

### Out of memory errors

**Problem:** Server runs out of VRAM

**Solutions:**
1. Use smaller model
2. For vLLM: Reduce `--max-num-seqs`
3. Use quantization (4-bit, 8-bit)
4. For llama.cpp: Reduce `-ngl` (GPU layers)
5. Close other GPU applications

---

## Advanced: Using Multiple Backends

You can use different backends for different purposes:

```python
from udf_optimizer.core.llm_client import LLMConfig, create_llm_client
from udf_optimizer.core.executor import DependencyAnalyzer, StepExecutor

# Use powerful local model for dependency analysis
analyzer_config = LLMConfig(
    backend="local",
    local_model="Qwen/Qwen2.5-14B-Instruct",
    temperature=0.3  # Lower for reliability
)
analyzer = DependencyAnalyzer(llm_client=create_llm_client(analyzer_config))

# Use faster model for step execution
executor_config = LLMConfig(
    backend="local",
    local_model="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.7  # Higher for creativity
)
executor = StepExecutor(llm_client=create_llm_client(executor_config))
```

---

## Performance Tuning

### For Maximum Throughput

```yaml
llm:
  backend: "local"
  local_api_base: "http://localhost:8000/v1"
  local_model: "meta-llama/Llama-3.1-8B-Instruct"
  temperature: 0.5    # Lower = faster
  max_tokens: 2048    # Reduce if possible
  timeout_seconds: 60  # Shorter timeout
```

**vLLM server:**
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 128 \      # More parallel requests
  --max-model-len 8192 \     # Longer context
  --tensor-parallel-size 2   # Use 2 GPUs
```

### For Best Quality

```yaml
llm:
  backend: "local"
  local_model: "Qwen/Qwen2.5-14B-Instruct"
  temperature: 0.3    # More deterministic
  max_tokens: 4096    # Longer responses
  top_p: 0.95
```

### For Development

```yaml
llm:
  backend: "local"
  local_model: "microsoft/Phi-3.5-mini-instruct"  # Small & fast
  temperature: 0.7
  max_tokens: 1024
  timeout_seconds: 30
```

---

## Migration from Gemini

If you were using Gemini before, here's how to migrate:

### Old Code (Gemini-only)
```python
from udf_optimizer.core.gemini_executor import DependencyAnalyzer

# Required GEMINI_API_KEY environment variable
analyzer = DependencyAnalyzer()
```

### New Code (Multi-backend)
```python
from udf_optimizer.core.executor import DependencyAnalyzer

# Now uses config.yaml (default: local LLM)
analyzer = DependencyAnalyzer()
```

**That's it!** The API is identical. Just update your config.yaml.

### Backward Compatibility

Old imports still work (with deprecation warning):
```python
# This still works
from udf_optimizer.core.gemini_executor import DependencyAnalyzer, GeminiStepExecutor
```

But new code should use:
```python
from udf_optimizer.core.executor import DependencyAnalyzer, StepExecutor
```

---

## FAQ

**Q: Can I use OpenAI's cloud API?**

A: Yes! Set `backend: "openai"` and `openai_api_key` in config.

**Q: Can I use multiple GPUs?**

A: Yes! Use vLLM with `--tensor-parallel-size N` for N GPUs.

**Q: Can I run without GPU?**

A: Yes! Use llama.cpp with CPU-only mode, or Ollama. Performance will be slower.

**Q: Which model should I use?**

A: For dependency analysis: Qwen2.5-14B. For step execution: Llama-3.1-8B.

**Q: How much does it cost?**

A: Local inference is free (just hardware costs). Cloud APIs charge per token.

**Q: Is local inference faster?**

A: Depends on your hardware. A good GPU can be 5-10x faster than Gemini API calls.

**Q: Can I mix local and cloud?**

A: Yes! Use different configs for analyzer and executor (requires code modification).

---

## Next Steps

1. **Choose your LLM server** (vLLM recommended for production)
2. **Install and start the server**
3. **Update config.yaml** with your server details
4. **Test with example workflow**
5. **Tune parameters** for your use case

For more details, see:
- `core/llm_client.py` - Client implementation
- `core/executor.py` - Executor implementation
- `config/config.yaml` - Configuration options
- `examples/demo_workflow.py` - Example usage

---

**Questions?** Check the troubleshooting section or open an issue on GitHub.
