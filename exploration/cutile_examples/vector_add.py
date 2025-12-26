"""
cuTile Python Example: Vector Addition

This is the "Hello World" of GPU programming with cuTile.
It demonstrates the basic structure of a cuTile kernel.

Requirements:
    - CUDA Toolkit 13.1+
    - pip install cuda-tile cupy-cuda12x
"""

import sys

# Check if cuTile is available
try:
    import cuda.tile as ct
    import cupy
    CUTILE_AVAILABLE = True
except ImportError:
    CUTILE_AVAILABLE = False
    print("⚠️  cuTile Python not installed. This is a reference implementation.")
    print("   Install with: pip install cuda-tile")
    print("   Requires CUDA Toolkit 13.1+")


# Configuration
TILE_SIZE = 16  # Process 16 elements per block


if CUTILE_AVAILABLE:
    @ct.kernel
    def vector_add_kernel(a, b, result):
        """
        GPU kernel for element-wise vector addition.
        
        Each block handles TILE_SIZE elements.
        The kernel loads tiles from arrays a and b,
        adds them together, and stores the result.
        """
        # Get the block ID (which segment of the array we're processing)
        block_id = ct.bid(0)
        
        # Load tiles from input arrays
        # index=(block_id,) means we're loading the block_id-th tile
        # shape=(TILE_SIZE,) means each tile has TILE_SIZE elements
        a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
        b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
        
        # Perform element-wise addition
        result_tile = a_tile + b_tile
        
        # Store the result back to memory
        ct.store(result, index=(block_id,), tile=result_tile)


    def vector_add(a: cupy.ndarray, b: cupy.ndarray) -> cupy.ndarray:
        """
        Add two vectors using GPU acceleration.
        
        Args:
            a: First input vector (CuPy array)
            b: Second input vector (CuPy array)
            
        Returns:
            Result of a + b (CuPy array)
        """
        assert a.shape == b.shape, "Arrays must have the same shape"
        assert len(a.shape) == 1, "Arrays must be 1D"
        
        # Create output array
        result = cupy.empty_like(a)
        
        # Calculate grid size (number of blocks needed)
        num_blocks = ct.cdiv(a.shape[0], TILE_SIZE)  # Ceiling division
        grid = (num_blocks, 1, 1)
        
        # Launch the kernel
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, vector_add_kernel, (a, b, result))
        
        return result


    def demo():
        """Demonstrate vector addition with cuTile."""
        print("🚀 cuTile Python Vector Addition Demo")
        print("=" * 50)
        
        # Create test vectors
        n = 1024
        a = cupy.arange(n, dtype=cupy.float32)
        b = cupy.arange(n, dtype=cupy.float32) * 2
        
        print(f"Vector size: {n}")
        print(f"Tile size: {TILE_SIZE}")
        print(f"Number of blocks: {ct.cdiv(n, TILE_SIZE)}")
        
        # Perform addition
        result = vector_add(a, b)
        
        # Verify result
        expected = a + b
        is_correct = cupy.allclose(result, expected)
        
        print(f"\n✅ Result correct: {is_correct}")
        print(f"\nSample results (first 8 elements):")
        print(f"  a:        {a[:8].get()}")
        print(f"  b:        {b[:8].get()}")
        print(f"  a + b:    {result[:8].get()}")
        
        # Timing comparison
        import time
        
        # cuTile timing
        stream = cupy.cuda.get_current_stream()
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            result = vector_add(a, b)
        stream.synchronize()
        cutile_time = (time.perf_counter() - start) / 100
        
        # CuPy native timing
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            result_cupy = a + b
        stream.synchronize()
        cupy_time = (time.perf_counter() - start) / 100
        
        print(f"\n⏱️  Timing (average of 100 runs):")
        print(f"  cuTile: {cutile_time*1000:.4f} ms")
        print(f"  CuPy native: {cupy_time*1000:.4f} ms")

else:
    # Reference implementation without cuTile
    import numpy as np
    
    def demo():
        """Demonstrate the concept using NumPy (CPU fallback)."""
        print("📝 cuTile Python Vector Addition - Reference Implementation")
        print("=" * 50)
        print("(Using NumPy since cuTile is not available)")
        
        # Create test vectors
        n = 1024
        a = np.arange(n, dtype=np.float32)
        b = np.arange(n, dtype=np.float32) * 2
        
        print(f"\nVector size: {n}")
        
        # Perform addition
        result = a + b
        
        print(f"\nSample results (first 8 elements):")
        print(f"  a:        {a[:8]}")
        print(f"  b:        {b[:8]}")
        print(f"  a + b:    {result[:8]}")
        
        print("\n💡 cuTile Kernel Structure:")
        print("""
    @ct.kernel
    def vector_add_kernel(a, b, result):
        block_id = ct.bid(0)
        a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
        b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
        result_tile = a_tile + b_tile
        ct.store(result, index=(block_id,), tile=result_tile)
        """)


if __name__ == "__main__":
    demo()


