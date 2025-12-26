"""
cuTile Python Example: Matrix Operations

This example demonstrates more advanced cuTile operations:
- 2D tile loading
- Matrix multiplication leveraging tensor cores
- Working with multi-dimensional grids

Requirements:
    - CUDA Toolkit 13.1+
    - pip install cuda-tile cupy-cuda12x
"""

import sys

try:
    import cuda.tile as ct
    import cupy
    CUTILE_AVAILABLE = True
except ImportError:
    CUTILE_AVAILABLE = False
    print("⚠️  cuTile Python not installed. This is a reference implementation.")
    import numpy as np


# Tile dimensions for matrix operations
TILE_M = 16  # Rows per tile
TILE_N = 16  # Columns per tile
TILE_K = 16  # Reduction dimension


if CUTILE_AVAILABLE:
    @ct.kernel
    def matrix_add_kernel(a, b, result, M, N):
        """
        GPU kernel for element-wise matrix addition.
        
        Each block handles a TILE_M x TILE_N tile of the matrix.
        """
        # Get 2D block indices
        block_row = ct.bid(0)
        block_col = ct.bid(1)
        
        # Load tiles from input matrices
        a_tile = ct.load(
            a, 
            index=(block_row, block_col), 
            shape=(TILE_M, TILE_N)
        )
        b_tile = ct.load(
            b, 
            index=(block_row, block_col), 
            shape=(TILE_M, TILE_N)
        )
        
        # Element-wise addition
        result_tile = a_tile + b_tile
        
        # Store result
        ct.store(result, index=(block_row, block_col), tile=result_tile)


    @ct.kernel
    def matrix_multiply_kernel(a, b, result, M, N, K):
        """
        GPU kernel for matrix multiplication C = A @ B.
        
        Uses tile-based multiplication with accumulation.
        This pattern naturally maps to tensor cores on supported hardware.
        
        A: (M, K)
        B: (K, N)
        C: (M, N)
        """
        block_row = ct.bid(0)
        block_col = ct.bid(1)
        
        # Initialize accumulator tile
        accumulator = ct.zeros(shape=(TILE_M, TILE_N), dtype=ct.float32)
        
        # Number of tiles in K dimension
        num_k_tiles = ct.cdiv(K, TILE_K)
        
        # Iterate over tiles in the K dimension
        for k_tile_idx in range(num_k_tiles):
            # Load tile from A (M x K)
            a_tile = ct.load(
                a,
                index=(block_row, k_tile_idx),
                shape=(TILE_M, TILE_K)
            )
            
            # Load tile from B (K x N)
            b_tile = ct.load(
                b,
                index=(k_tile_idx, block_col),
                shape=(TILE_K, TILE_N)
            )
            
            # Multiply and accumulate
            # This can leverage tensor cores on compatible hardware
            accumulator = accumulator + ct.matmul(a_tile, b_tile)
        
        # Store final result
        ct.store(result, index=(block_row, block_col), tile=accumulator)


    def matrix_add(a: cupy.ndarray, b: cupy.ndarray) -> cupy.ndarray:
        """Add two matrices using GPU acceleration."""
        assert a.shape == b.shape, "Matrices must have the same shape"
        M, N = a.shape
        
        result = cupy.empty_like(a)
        
        # Calculate grid dimensions
        grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), 1)
        
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, matrix_add_kernel, (a, b, result, M, N))
        
        return result


    def matrix_multiply(a: cupy.ndarray, b: cupy.ndarray) -> cupy.ndarray:
        """Multiply two matrices using GPU acceleration with tensor cores."""
        M, K1 = a.shape
        K2, N = b.shape
        assert K1 == K2, f"Inner dimensions must match: {K1} != {K2}"
        
        result = cupy.zeros((M, N), dtype=cupy.float32)
        
        # Calculate grid dimensions
        grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), 1)
        
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, matrix_multiply_kernel, (a, b, result, M, N, K1))
        
        return result


    def demo():
        """Demonstrate matrix operations with cuTile."""
        print("🚀 cuTile Python Matrix Operations Demo")
        print("=" * 50)
        
        # Matrix dimensions (multiples of tile size for simplicity)
        M, N, K = 256, 256, 256
        
        print(f"Matrix dimensions: M={M}, N={N}, K={K}")
        print(f"Tile dimensions: {TILE_M}x{TILE_N}")
        
        # Create test matrices
        A = cupy.random.randn(M, K, dtype=cupy.float32)
        B = cupy.random.randn(K, N, dtype=cupy.float32)
        C = cupy.random.randn(M, N, dtype=cupy.float32)
        
        # Matrix addition
        print("\n--- Matrix Addition ---")
        result_add = matrix_add(A[:M, :N], C)
        expected_add = A[:M, :N] + C
        is_correct_add = cupy.allclose(result_add, expected_add)
        print(f"✅ Addition correct: {is_correct_add}")
        
        # Matrix multiplication
        print("\n--- Matrix Multiplication ---")
        result_mul = matrix_multiply(A, B)
        expected_mul = cupy.matmul(A, B)
        is_correct_mul = cupy.allclose(result_mul, expected_mul, rtol=1e-3)
        print(f"✅ Multiplication correct: {is_correct_mul}")
        
        # Timing
        import time
        stream = cupy.cuda.get_current_stream()
        
        # cuTile matmul timing
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            result = matrix_multiply(A, B)
        stream.synchronize()
        cutile_time = (time.perf_counter() - start) / 100
        
        # CuPy native timing
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            result = cupy.matmul(A, B)
        stream.synchronize()
        cupy_time = (time.perf_counter() - start) / 100
        
        print(f"\n⏱️  Matrix Multiply Timing ({M}x{K} @ {K}x{N}):")
        print(f"  cuTile: {cutile_time*1000:.4f} ms")
        print(f"  CuPy native: {cupy_time*1000:.4f} ms")

else:
    def demo():
        """Reference implementation using NumPy."""
        print("📝 cuTile Matrix Operations - Reference Implementation")
        print("=" * 50)
        print("(Using NumPy since cuTile is not available)")
        
        M, N, K = 256, 256, 256
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        print(f"\nMatrix dimensions: A({M}x{K}) @ B({K}x{N})")
        
        # Matrix multiplication
        result = np.matmul(A, B)
        print(f"Result shape: {result.shape}")
        
        print("\n💡 cuTile Matrix Multiply Structure:")
        print("""
    @ct.kernel
    def matrix_multiply_kernel(a, b, result, M, N, K):
        block_row = ct.bid(0)
        block_col = ct.bid(1)
        
        accumulator = ct.zeros(shape=(TILE_M, TILE_N), dtype=ct.float32)
        
        for k_tile_idx in range(num_k_tiles):
            a_tile = ct.load(a, index=(block_row, k_tile_idx), shape=(TILE_M, TILE_K))
            b_tile = ct.load(b, index=(k_tile_idx, block_col), shape=(TILE_K, TILE_N))
            accumulator = accumulator + ct.matmul(a_tile, b_tile)
        
        ct.store(result, index=(block_row, block_col), tile=accumulator)
        """)


if __name__ == "__main__":
    demo()

