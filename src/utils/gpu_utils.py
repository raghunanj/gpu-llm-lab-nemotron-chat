"""
GPU Utilities

Helper functions for working with NVIDIA GPUs and cuTile.
"""

import subprocess
from typing import Optional
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about an NVIDIA GPU."""
    name: str
    memory_total: int  # in MB
    memory_free: int  # in MB
    memory_used: int  # in MB
    compute_capability: str
    driver_version: str
    cuda_version: str


def check_gpu_available() -> bool:
    """Check if an NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_gpu_info() -> Optional[GPUInfo]:
    """
    Get information about the primary NVIDIA GPU.
    
    Returns:
        GPUInfo object if GPU is available, None otherwise
    """
    if not check_gpu_available():
        return None
    
    try:
        # Get GPU info using nvidia-smi
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        # Parse the output
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        
        if len(parts) < 5:
            return None
        
        # Get CUDA version
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        cuda_version = "Unknown"
        
        # Try to get compute capability
        compute_cap = "Unknown"
        try:
            import torch
            if torch.cuda.is_available():
                compute_cap = f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
                cuda_version = torch.version.cuda or "Unknown"
        except ImportError:
            pass
        
        return GPUInfo(
            name=parts[0],
            memory_total=int(float(parts[1])),
            memory_free=int(float(parts[2])),
            memory_used=int(float(parts[3])),
            compute_capability=compute_cap,
            driver_version=parts[4],
            cuda_version=cuda_version
        )
        
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None


def check_cutile_available() -> bool:
    """Check if cuTile Python is available."""
    try:
        import cuda.tile as ct
        return True
    except ImportError:
        return False


def check_cupy_available() -> bool:
    """Check if CuPy is available."""
    try:
        import cupy
        return True
    except ImportError:
        return False


def get_environment_status() -> dict:
    """Get the status of the GPU environment."""
    gpu_available = check_gpu_available()
    gpu_info = get_gpu_info()
    
    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_info.name if gpu_info else None,
        "gpu_memory_total_mb": gpu_info.memory_total if gpu_info else None,
        "gpu_memory_free_mb": gpu_info.memory_free if gpu_info else None,
        "cuda_version": gpu_info.cuda_version if gpu_info else None,
        "compute_capability": gpu_info.compute_capability if gpu_info else None,
        "cutile_available": check_cutile_available(),
        "cupy_available": check_cupy_available(),
    }


def print_environment_status():
    """Print the GPU environment status."""
    status = get_environment_status()
    
    print("=" * 50)
    print("GPU Environment Status")
    print("=" * 50)
    
    if status["gpu_available"]:
        print(f"✅ GPU Available: {status['gpu_name']}")
        print(f"   Memory: {status['gpu_memory_free_mb']}MB free / {status['gpu_memory_total_mb']}MB total")
        print(f"   CUDA Version: {status['cuda_version']}")
        print(f"   Compute Capability: {status['compute_capability']}")
    else:
        print("❌ No NVIDIA GPU detected")
    
    print()
    print(f"{'✅' if status['cutile_available'] else '❌'} cuTile Python: {'Available' if status['cutile_available'] else 'Not installed'}")
    print(f"{'✅' if status['cupy_available'] else '❌'} CuPy: {'Available' if status['cupy_available'] else 'Not installed'}")
    print("=" * 50)


if __name__ == "__main__":
    print_environment_status()


