# cuTile Python Examples

## What is cuTile Python?

cuTile Python is NVIDIA's domain-specific language for writing parallel kernels that run on NVIDIA GPUs. It provides a high-level abstraction that:

- Simplifies GPU programming compared to raw CUDA
- Leverages tensor cores automatically
- Is portable across different NVIDIA GPU architectures
- Uses a tile-based programming model

## Requirements

- Python 3.10+
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA Toolkit 13.1+
- cuTile Python package: `pip install cuda-tile`

## Installation

```bash
# Check CUDA version
nvcc --version

# Install cuTile Python
pip install cuda-tile

# Install CuPy (required for array operations)
pip install cupy-cuda12x  # Replace with your CUDA version
```

## Examples in This Directory

### 1. `vector_add.py` - Basic Vector Addition
The "Hello World" of GPU programming. Shows how to:
- Define a cuTile kernel
- Load data into tiles
- Perform parallel operations
- Store results back

### 2. `matrix_ops.py` - Matrix Operations
More advanced operations including:
- Matrix addition
- Matrix multiplication using tensor cores
- Batch processing

### 3. `reduction.py` - Parallel Reduction
Shows how to:
- Aggregate data across threads
- Implement sum, max, min operations
- Handle shared memory

## Key Concepts

### Tiles
Tiles are the fundamental unit of data in cuTile. A tile represents a contiguous block of memory that can be loaded, transformed, and stored efficiently.

```python
# Load a 16x16 tile from an array
tile = ct.load(array, index=(block_x, block_y), shape=(16, 16))
```

### Block and Thread IDs
```python
block_id = ct.bid(0)   # Block ID in dimension 0
block_x = ct.bid(0)
block_y = ct.bid(1)
```

### Grid Configuration
```python
grid = (num_blocks_x, num_blocks_y, 1)
ct.launch(stream, grid, kernel, args)
```

## When to Use cuTile

Good use cases:
- Custom data preprocessing pipelines
- Specialized embedding computations
- Performance-critical inference optimizations

For most LLM inference, the models are already optimized, but cuTile can help with:
- Custom tokenization/detokenization
- Embedding similarity searches
- Batch processing optimizations

## Resources

- [Official Documentation](https://docs.nvidia.com/cuda/cutile-python)
- [NVIDIA GitHub](https://github.com/NVIDIA/cutile-python)


