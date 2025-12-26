#!/usr/bin/env python3
"""
Optimized Matrix Multiplication using Triton.

This shows how to write GPU kernels in Python that are faster than PyTorch defaults.
Matrix multiplication is the core operation in LLMs (attention, FFN layers).

Key optimizations:
1. Tiled computation - process data in blocks that fit in shared memory
2. Memory coalescing - threads access consecutive memory addresses
3. Software pipelining - overlap memory loads with computation
"""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for accessing elements
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (tuned for RTX 4090)
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)
    
    Each thread block computes a BLOCK_M x BLOCK_N tile of C.
    """
    # Program ID - which tile of C this block computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets within our tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to first block of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator for output tile - initialized to zero
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load tiles from A and B with bounds checking
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        
        # Compute partial dot product and accumulate
        acc += tl.dot(a, b)
        
        # Advance pointers to next K-block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Convert to output dtype
    c = acc.to(tl.float16)
    
    # Compute output pointers and store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using Triton.
    
    Args:
        a: (M, K) tensor
        b: (K, N) tensor
    
    Returns:
        c: (M, N) tensor = a @ b
    """
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Block sizes optimized for RTX 4090 (128 SMs, 48KB shared memory)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Grid of thread blocks
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return c


def benchmark():
    """Compare Triton matmul vs PyTorch."""
    print("=" * 60)
    print("Matrix Multiplication Benchmark: Triton vs PyTorch")
    print("=" * 60)
    
    # Typical LLM dimensions
    sizes = [
        (1, 4096, 4096),      # Single token, hidden dim
        (32, 4096, 4096),     # Batch of 32 tokens
        (128, 4096, 11008),   # FFN up-projection
        (512, 4096, 4096),    # Larger batch
    ]
    
    for M, K, N in sizes:
        print(f"\nMatrix size: ({M}, {K}) x ({K}, {N})")
        
        # Create random matrices
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(3):
            _ = torch.matmul(a, b)
            _ = triton_matmul(a, b)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(100):
            c_torch = torch.matmul(a, b)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000
        
        # Benchmark Triton
        start = time.perf_counter()
        for _ in range(100):
            c_triton = triton_matmul(a, b)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / 100 * 1000
        
        # Verify correctness
        max_diff = (c_torch - c_triton).abs().max().item()
        
        print(f"  PyTorch: {pytorch_time:.3f} ms")
        print(f"  Triton:  {triton_time:.3f} ms")
        print(f"  Speedup: {pytorch_time/triton_time:.2f}x")
        print(f"  Max diff: {max_diff:.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    benchmark()

