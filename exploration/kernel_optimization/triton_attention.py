#!/usr/bin/env python3
"""
Flash Attention-style kernel using Triton.

Attention is the most compute-intensive operation in transformers.
Standard attention: O(n²) memory for attention matrix
Flash Attention: O(n) memory using tiling + online softmax

This implements a simplified Flash Attention kernel showing key concepts.
"""
import torch
import triton
import triton.language as tl
import time
import math


@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    N_CTX,
    scale,  # Pre-computed 1/sqrt(head_dim)
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention forward pass.
    
    Key insight: Instead of computing full NxN attention matrix,
    compute it in tiles and accumulate results using online softmax.
    
    This reduces memory from O(N²) to O(N) while maintaining numerical stability.
    """
    # Get batch and head indices
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_m = tl.program_id(2)
    
    # Initialize pointers for this (batch, head, row_block)
    offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointer to Q block for this row
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_d[None, :]
    
    # Load Q block (BLOCK_M x BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Initialize output accumulator and normalizer (for online softmax)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # max so far
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # sum of exp so far
    
    # scale is passed as parameter (pre-computed as 1/sqrt(head_dim))
    
    # Iterate over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Pointer to K, V blocks
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + \
                 (start_n + offs_n[None, :]) * stride_kn + offs_d[:, None]
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + \
                 (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :]
        
        # Load K, V blocks
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        
        # Compute attention scores: QK^T / sqrt(d)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) * scale
        
        # Apply causal mask (for decoder attention)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float('-inf'))
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # row max
        m_new = tl.maximum(m_i, m_ij)
        
        # Correction factors for previous accumulator
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_ij[:, None]) * beta[:, None], axis=1)
        
        # Compute attention weights for this block
        p = tl.exp(qk - m_new[:, None])
        
        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        # Update max
        m_i = m_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + \
               offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def triton_flash_attention(q, k, v):
    """
    Flash Attention using Triton.
    
    Args:
        q, k, v: (batch, heads, seq_len, head_dim) tensors
    
    Returns:
        out: (batch, heads, seq_len, head_dim) tensor
    """
    B, H, N, D = q.shape
    
    assert q.dtype == torch.float16
    assert D in [32, 64, 128]  # Common head dimensions
    
    out = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D
    
    # Pre-compute scale factor
    scale = 1.0 / math.sqrt(D)
    
    grid = (B, H, triton.cdiv(N, BLOCK_M))
    
    flash_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N,
        scale,
        BLOCK_M, BLOCK_N, BLOCK_D,
    )
    
    return out


def naive_attention(q, k, v):
    """Standard attention for comparison (O(n²) memory)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Causal mask
    N = q.shape[2]
    mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def benchmark():
    """Benchmark Triton Flash Attention vs Naive."""
    print("=" * 60)
    print("Flash Attention Benchmark: Triton vs Naive")
    print("=" * 60)
    
    configs = [
        (1, 32, 512, 64),    # batch=1, 32 heads, 512 tokens, 64 dim
        (1, 32, 1024, 64),   # 1K context
        (1, 32, 2048, 64),   # 2K context
        (4, 32, 512, 64),    # Batch of 4
    ]
    
    for B, H, N, D in configs:
        print(f"\nShape: batch={B}, heads={H}, seq_len={N}, head_dim={D}")
        
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(3):
            _ = naive_attention(q, k, v)
            _ = triton_flash_attention(q, k, v)
        torch.cuda.synchronize()
        
        # Benchmark naive
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(20):
            out_naive = naive_attention(q, k, v)
        torch.cuda.synchronize()
        naive_time = (time.perf_counter() - start) / 20 * 1000
        naive_mem = torch.cuda.max_memory_allocated() / 1e6
        
        # Benchmark Triton Flash
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(20):
            out_flash = triton_flash_attention(q, k, v)
        torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / 20 * 1000
        flash_mem = torch.cuda.max_memory_allocated() / 1e6
        
        # Verify correctness (approximately)
        diff = (out_naive - out_flash).abs().max().item()
        
        print(f"  Naive:  {naive_time:.3f} ms, {naive_mem:.1f} MB peak")
        print(f"  Flash:  {flash_time:.3f} ms, {flash_mem:.1f} MB peak")
        print(f"  Speedup: {naive_time/flash_time:.2f}x, Memory: {naive_mem/flash_mem:.2f}x less")
        print(f"  Max diff: {diff:.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    benchmark()

