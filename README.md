# Nemotron AI Assistant

A local AI assistant powered by NVIDIA's Nemotron-3-Nano model with GPU optimization examples.

## Features

- **Local LLM Inference** - Run Nemotron-3-Nano (30B) locally with 4-bit quantization
- **GPU Kernel Optimization** - Learn Triton kernels for Flash Attention and matrix operations
- **Modular Architecture** - Extensible agent-based design

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 4090)
- CUDA 12.x

### Installation

```bash
# Create conda environment
conda create -n nemotron3 python=3.11 -y
conda activate nemotron3

# Install CUDA toolkit
conda install cuda-toolkit=12.8 -c nvidia -y

# Install dependencies
pip install torch transformers accelerate bitsandbytes
pip install mamba-ssm triton

# Run interactive chat
python chat_local.py
```

### Running the Chat

```bash
conda activate nemotron3
python chat_local.py
```

The model uses chain-of-thought reasoning and extracts clean answers automatically.

## Project Structure

```
├── chat_local.py              # Interactive chat interface
├── exploration/
│   ├── kernel_optimization/   # Triton kernel examples
│   │   ├── triton_matmul.py   # Optimized matrix multiplication
│   │   ├── triton_attention.py # Flash Attention implementation
│   │   └── vllm_inference.py  # vLLM integration example
│   ├── nemotron_examples/     # LLM usage examples
│   └── cutile_examples/       # GPU programming basics
├── src/
│   ├── agents/                # Agent implementations
│   ├── llm/                   # LLM client wrappers
│   └── utils/                 # Utility functions
└── interfaces/                # CLI and web interfaces
```

## GPU Optimization Examples

### Flash Attention (Triton)

```bash
cd exploration/kernel_optimization
python triton_attention.py
```

Results on RTX 4090:
- 2K context: **11x speedup**, 10x less memory
- 1K context: **6x speedup**, 5x less memory

### Matrix Multiplication

```bash
python triton_matmul.py
```

## Model Details

| Model | Parameters | Active | VRAM (4-bit) |
|-------|------------|--------|--------------|
| Nemotron-3-Nano-30B | 30B | 3.5B | ~18GB |

## Technologies

- **LLM**: NVIDIA Nemotron-3-Nano (Mamba-2 hybrid architecture)
- **Quantization**: bitsandbytes 4-bit NF4
- **GPU Kernels**: Triton, Flash Attention
- **Inference**: HuggingFace Transformers

## License

MIT License
