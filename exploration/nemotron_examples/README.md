# Nemotron 3 Examples

## Overview

NVIDIA Nemotron 3 is a family of open LLMs optimized for agentic AI applications. Released in December 2024, the family includes:

| Model | Parameters | Active Params | Context Window | Use Case |
|-------|-----------|---------------|----------------|----------|
| **Nano** | 30B | 3.5B/token | 1M tokens | High-throughput, efficient |
| **Super** | ~100B | - | - | Multi-agent collaboration |
| **Ultra** | ~500B | - | - | Complex reasoning |

## Key Features

- **Mixture-of-Experts (MoE)**: Only 3.5B parameters active per token despite 30B total
- **1 Million Token Context**: Handle very long documents
- **4x Higher Throughput**: Compared to previous models
- **Open Weights**: Available on Hugging Face

## Access Methods

### 1. NVIDIA NIM API (Recommended for Quick Start)

NVIDIA provides hosted inference via their API at `build.nvidia.com`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="your-nvidia-api-key"  # Get from build.nvidia.com
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-nano-30b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 2. Hugging Face (Local Inference)

For running the model locally (requires significant GPU memory):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 3. vLLM (Production Deployment)

For high-throughput production deployments:

```bash
pip install vllm

# Run vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Nemotron-3-Nano-30B-A3B-BF16
```

## Examples in This Directory

### `api_inference.py`
- Connect to NVIDIA NIM API
- Basic chat completion
- Streaming responses
- Handling errors and retries

### `local_inference.py`
- Load model locally with transformers
- Memory optimization techniques
- Batch processing

### `agent_example.py`
- Build a simple agent with Nemotron
- Tool calling / function calling
- Conversation memory

## Getting Started

1. **Get API Key**: Visit [build.nvidia.com](https://build.nvidia.com) and sign up for an API key

2. **Set Environment Variable**:
   ```bash
   export NVIDIA_API_KEY="nvapi-xxxx"
   ```

3. **Run Examples**:
   ```bash
   python api_inference.py
   ```

## Resources

- [Nemotron 3 White Paper](https://research.nvidia.com/labs/nemotron)
- [Hugging Face Models](https://huggingface.co/nvidia)
- [NVIDIA NIM Documentation](https://build.nvidia.com)


