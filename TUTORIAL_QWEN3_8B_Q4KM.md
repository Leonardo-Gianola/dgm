# Tutorial: Running DGM with Qwen3 8B Thinking (Q4_K_M)

This tutorial explains how to run the Darwin Gödel Machine (DGM) using a locally-hosted **Qwen3 8B Thinking** model with **Q4_K_M quantization**.

## Overview

Qwen3 8B Thinking is a reasoning-capable language model. With Q4_K_M quantization:

- **Model size**: ~5.2 GB (down from ~16 GB)
- **RAM/VRAM needed**: ~8 GB minimum (16 GB recommended)
- **Quality**: ~95% of full model performance

## Changes Made to the Code

### 1. `llm.py` - Added Local Model Support

**Added new model identifier:**

```python
"local-qwen3-8b-q4km",  # GGUF Q4_K_M quantized
```

**Added model mapping:**

```python
model_mapping = {
    "qwen3-8b": "Qwen3-8B",
    "qwen3-8b-q4km": "qwen3-8b-Q4_K_M",  # Maps to your GGUF filename
    "qwen3-235b": "Qwen3-235B",
}
```

**Added thinking mode support:**

```python
# Enable thinking mode for Qwen3 reasoning models
extra_body = {}
if os.getenv("LOCAL_LLM_ENABLE_THINKING", "true").lower() == "true":
    extra_body["enable_thinking"] = True
if os.getenv("LOCAL_LLM_THINKING_BUDGET"):
    extra_body["reasoning_budget"] = int(os.getenv("LOCAL_LLM_THINKING_BUDGET"))
```

### 2. `llm_withtools.py` - Updated Default Models

```python
# Local Qwen3 8B Thinking Q4_K_M quantized model:
CLAUDE_MODEL = 'local-qwen3-8b-q4km'
OPENAI_MODEL = 'local-qwen3-8b-q4km'
```

### 3. `llm_withtools.py` - Added Local Model Handling

```python
elif model.startswith('local-'):
    # Local models use manual tool calling
    new_msg_history = chat_with_agent_manualtools(msg, model=model, ...)
```

### 4. `self_improve_step.py` - Updated Diagnose Model

```python
# Local Qwen3 8B Thinking Q4_K_M:
diagnose_model = 'local-qwen3-8b-q4km'
```

## Prerequisites

### Hardware Requirements

| Component | Minimum    | Recommended |
| --------- | ---------- | ----------- |
| RAM       | 16 GB      | 32 GB       |
| VRAM      | 8 GB       | 12+ GB      |
| Storage   | 10 GB free | 20 GB free  |
| CPU       | 4 cores    | 8+ cores    |

### Software Requirements

- Python 3.10+
- llama.cpp (built with server support) OR Ollama
- This DGM repository

## Setup Instructions

### Step 1: Download the GGUF Model

Download the Q4_K_M quantized Qwen3 8B model:

```bash
# Create models directory
mkdir -p models

# Download from HuggingFace (using wget)
wget https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-Q4_K_M.gguf -P models/

# Or download with curl
curl -L -o models/qwen3-8b-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-Q4_K_M.gguf
```

Alternative download methods:

- **HuggingFace Hub CLI**: `hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models`
- **Browser**: https://huggingface.co/Qwen/Qwen3-8B-GGUF

### Step 2: Set Up the Model Server

Choose one of these options:

#### Option A: llama.cpp Server (Recommended)

```bash
# Clone and build llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

# Build with CUDA support (if you have NVIDIA GPU)
make clean && make -j LLAMA_CUDA=1

# Or build for CPU only
make clean && make -j

# Start the server with Qwen3 8B Q4_K_M
./llama-server \
  -m ../models/qwen3-8b-Q4_K_M.gguf \
  -c 32768 \
  -n 4096 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key not-needed
```

Server parameters explained:

- `-m`: Path to the GGUF model file
- `-c`: Context size (32K tokens)
- `-n`: Maximum tokens to generate
- `--host/--port`: API server address

### Step 3: Install Python Dependencies

```bash
# Navigate to the DGM directory
cd /path/to/dgm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 4: Set Environment Variables

```bash
# Windows PowerShell
$env:LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
$env:LOCAL_LLM_API_KEY="not-needed"
$env:LOCAL_LLM_ENABLE_THINKING="true"
$env:LOCAL_LLM_THINKING_BUDGET="8192"  # Optional: limit thinking tokens

# Windows CMD
set LOCAL_LLM_BASE_URL=http://localhost:8000/v1
set LOCAL_LLM_API_KEY=not-needed
set LOCAL_LLM_ENABLE_THINKING=true
set LOCAL_LLM_THINKING_BUDGET=8192

# Linux/macOS
export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
export LOCAL_LLM_API_KEY="not-needed"
export LOCAL_LLM_ENABLE_THINKING="true"
export LOCAL_LLM_THINKING_BUDGET="8192"
```

**Environment Variables Explained:**

| Variable                    | Description                  | Default                    |
| --------------------------- | ---------------------------- | -------------------------- |
| `LOCAL_LLM_BASE_URL`        | URL of your local API server | `http://localhost:8000/v1` |
| `LOCAL_LLM_API_KEY`         | API key (if required)        | `not-needed`               |
| `LOCAL_LLM_ENABLE_THINKING` | Enable Qwen3 thinking mode   | `true`                     |
| `LOCAL_LLM_THINKING_BUDGET` | Max tokens for reasoning     | None (unlimited)           |

### Step 5: Test the Setup

Before running DGM, test that the model is accessible:

```bash
# Test via curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-Q4_K_M",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello! Can you solve coding problems?"}
    ]
  }'
```

Or run the Python test:

```bash
python -c "
from llm import create_client, get_response_from_llm
client, model = create_client('local-qwen3-8b-q4km')
response, _ = get_response_from_llm(
    msg='Hello! What is 2+2?',
    client=client,
    model=model,
    system_message='You are a helpful assistant.'
)
print('Response:', response)
"
```

### Step 6: Run DGM

#### Full DGM Loop

```bash
python DGM_outer.py \
  --max_generation 10 \
  --selfimprove_size 2 \
  --selfimprove_workers 1
```

#### Single Self-Improvement Step (for testing)

```bash
python self_improve_step.py \
  --parent_commit initial \
  --entry django__django-10999 \
  --output_dir ./output
```

#### With Shallow Evaluation (faster)

```bash
python DGM_outer.py \
  --max_generation 10 \
  --selfimprove_size 2 \
  --shallow_eval \
  --selfimprove_workers 1
```

## Configuration Options

### Switching Back to Cloud Models

Edit `llm_withtools.py`:

```python
# Cloud models (default):
CLAUDE_MODEL = 'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0'
OPENAI_MODEL = 'o3-mini-2025-01-31'
```

Edit `self_improve_step.py`:

```python
# Cloud diagnose model:
diagnose_model = 'o1-2024-12-17'
```

### Adjusting for Different Hardware

**Low RAM (8-16 GB):**

```bash
# Reduce context size in llama.cpp server
./llama-server -m models/qwen3-8b-Q4_K_M.gguf -c 16384 ...  # Instead of 32768
```

**CPU Only (slower):**

```bash
# Build llama.cpp without CUDA
make clean && make -j

# Increase threads
./llama-server -m models/qwen3-8b-Q4_K_M.gguf -t 8 ...
```

## Troubleshooting

### Issue: "Connection refused" errors

**Cause**: The model server isn't running or is on a different port.

**Solution**:

```bash
# Check if server is running
curl http://localhost:8000/health

# Verify the port matches your environment variable
echo $env:LOCAL_LLM_BASE_URL  # Should match server port
```

### Issue: "Model not found" errors

**Cause**: The model name in the API call doesn't match what the server expects.

**Solution**: Check the exact model name your server expects:

```bash
# For llama.cpp, it uses the filename by default
curl http://localhost:8000/v1/models

# Update model mapping in llm.py if needed:
"qwen3-8b-q4km": "your-actual-gguf-filename",
```

### Issue: Out of memory errors

**Cause**: Context size too large for your hardware.

**Solution**:

```bash
# Reduce context window
./llama-server -c 8192 ...  # Instead of 32768

# Or use smaller quantization (Q3_K_M instead of Q4_K_M)
```

### Issue: Very slow responses

**Cause**: Running on CPU or insufficient threads.

**Solution**:

```bash
# Increase threads for llama.cpp
./llama-server -t 8 ...

# Enable GPU layers (if you have GPU)
./llama-server -ngl 35 ...  # Offload 35 layers to GPU
```

### Issue: Tool calling not working

**Cause**: Local models use manual tool parsing which may be less reliable.

**Solution**: The system already falls back to manual tool parsing for local models. You can improve it by:

1. Using a more capable base model (Qwen3 14B instead of 8B)
2. Adjusting temperature in `llm.py` (lower for more deterministic tool use)

## Performance Expectations

| Metric                        | Value     |
| ----------------------------- | --------- |
| Tokens/second (CPU, 8 cores)  | 10-20 t/s |
| Tokens/second (GPU, RTX 3060) | 40-60 t/s |
| Tokens/second (GPU, RTX 4090) | 100+ t/s  |
| Memory usage                  | 6-8 GB    |
| Time per DGM generation       | 1-4 hours |

## Tips for Best Results

1. **Use GPU if possible**: Even an older GPU (GTX 1060 6GB) is much faster than CPU
2. **Start with shallow evaluation**: Use `--shallow_eval` flag first to test
3. **Monitor temperatures**: Local inference can heat up hardware
4. **Use single worker**: `--selfimprove_workers 1` to avoid overloading your system
5. **Check logs**: Monitor `output_dgm/[run_id]/dgm_outer.log` for issues

## Next Steps

- Try Qwen3 14B Q4_K_M for better reasoning
- Experiment with different context sizes
- Fine-tune the thinking budget for your use case
- Compare results with cloud models

## References

- [Qwen3 on HuggingFace](https://huggingface.co/Qwen)
- [llama.cpp repository](https://github.com/ggerganov/llama.cpp)
- [Ollama documentation](https://github.com/ollama/ollama)
- [GGUF quantization formats](https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md)
