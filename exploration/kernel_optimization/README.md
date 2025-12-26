# GPU Kernel Optimization for LLM Inference

This directory contains examples of optimized GPU kernels for faster LLM inference.

## Optimization Approaches (Easiest to Hardest)

| Approach | Speedup | Difficulty | Best For |
|----------|---------|------------|----------|
| **vLLM** | 3-5x | Easy | Production deployment |
| **Flash Attention** | 2-3x | Easy | Memory efficiency |
| **Triton Kernels** | Variable | Medium | Custom optimizations |
| **TensorRT-LLM** | 5-10x | Medium | Maximum performance |
| **Raw CUDA** | Maximum | Hard | Research/learning |

## Quick Start

```bash
# Install Triton for custom kernels
pip install triton

# Install vLLM for production inference
pip install vllm

# Install Flash Attention
pip install flash-attn --no-build-isolation
```

## Files

- `triton_matmul.py` - Optimized matrix multiplication kernel
- `triton_attention.py` - Custom attention kernel  
- `vllm_inference.py` - Production-ready fast inference
- `benchmarks.py` - Performance comparison

## Key Optimization Techniques

1. **Memory Coalescing** - Aligned memory access patterns
2. **Shared Memory Tiling** - Reduce global memory bandwidth
3. **Kernel Fusion** - Combine operations to reduce memory I/O
4. **Quantization-Aware Kernels** - Optimize for 4-bit/8-bit ops
5. **FlashAttention** - Memory-efficient attention computation

