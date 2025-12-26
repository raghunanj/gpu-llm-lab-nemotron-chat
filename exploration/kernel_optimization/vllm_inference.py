#!/usr/bin/env python3
"""
Fast LLM Inference with vLLM.

vLLM provides 3-5x speedup over HuggingFace through:
1. PagedAttention - Efficient KV-cache memory management
2. Continuous Batching - Better GPU utilization
3. Optimized CUDA kernels - Flash Attention, fused operations
4. Tensor Parallelism - Multi-GPU support

This is the EASIEST way to get production-level inference speed.
"""
import time
from typing import Optional


def run_vllm_inference():
    """
    Run optimized inference with vLLM.
    
    Install first: pip install vllm
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("❌ vLLM not installed. Install with: pip install vllm")
        print("   Note: vLLM requires CUDA 11.8+ and Linux")
        return
    
    print("🚀 Loading model with vLLM (optimized inference)...")
    
    # vLLM supports many models - use a smaller one for demo
    # For Nemotron, you'd use the model ID directly
    model_id = "microsoft/phi-2"  # 2.7B model, fast to load
    
    # Initialize vLLM engine
    # - tensor_parallel_size: Number of GPUs (1 for single GPU)
    # - gpu_memory_utilization: How much VRAM to use (0.9 = 90%)
    # - dtype: Use float16 for speed
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype="float16",
        trust_remote_code=True,
    )
    
    print("✅ Model loaded!")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )
    
    # Test prompts
    prompts = [
        "What is machine learning?",
        "Explain machine learning briefly.",
        "Write a Python function to calculate factorial.",
    ]
    
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)
    
    # Batch inference - process all prompts together
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    
    total_tokens = 0
    for i, output in enumerate(outputs):
        prompt = output.prompt
        response = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        total_tokens += tokens
        
        print(f"\n📝 Prompt {i+1}: {prompt[:50]}...")
        print(f"💬 Response: {response[:200]}...")
        print(f"   Tokens: {tokens}")
    
    print("\n" + "=" * 60)
    print(f"⚡ Total: {total_tokens} tokens in {elapsed:.2f}s")
    print(f"   Throughput: {total_tokens/elapsed:.1f} tokens/sec")
    print("=" * 60)


def compare_vllm_vs_hf():
    """
    Compare vLLM speed vs standard HuggingFace.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("Speed Comparison: vLLM vs HuggingFace Transformers")
    print("=" * 60)
    
    model_id = "microsoft/phi-2"
    prompt = "Explain the concept of recursion in programming with an example."
    max_tokens = 200
    
    # --- HuggingFace Baseline ---
    print("\n📦 Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20)
    torch.cuda.synchronize()
    
    # Benchmark HF
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
    torch.cuda.synchronize()
    hf_time = time.perf_counter() - start
    hf_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    print(f"   HuggingFace: {hf_tokens} tokens in {hf_time:.2f}s ({hf_tokens/hf_time:.1f} tok/s)")
    
    # Clear memory
    del model
    torch.cuda.empty_cache()
    
    # --- vLLM ---
    try:
        from vllm import LLM, SamplingParams
        
        print("\n🚀 Loading vLLM model...")
        llm = LLM(model=model_id, dtype="float16", trust_remote_code=True)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
        )
        
        # Warmup
        _ = llm.generate([prompt], SamplingParams(max_tokens=20))
        
        # Benchmark vLLM
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        vllm_time = time.perf_counter() - start
        vllm_tokens = len(outputs[0].outputs[0].token_ids)
        
        print(f"   vLLM:        {vllm_tokens} tokens in {vllm_time:.2f}s ({vllm_tokens/vllm_time:.1f} tok/s)")
        
        # Comparison
        speedup = (hf_tokens/hf_time) / (vllm_tokens/vllm_time) if vllm_time > 0 else 0
        speedup = (vllm_tokens/vllm_time) / (hf_tokens/hf_time)
        
        print(f"\n✨ vLLM Speedup: {speedup:.2f}x faster!")
        
    except ImportError:
        print("   ❌ vLLM not installed")


def show_vllm_features():
    """Show key vLLM optimization features."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║               vLLM Key Optimizations                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 1. PagedAttention                                            ║
║    - Manages KV-cache like virtual memory pages              ║
║    - Near-zero memory waste (vs 60-80% in naive impl)        ║
║    - Enables larger batch sizes                              ║
║                                                              ║
║ 2. Continuous Batching                                       ║
║    - New requests join batch immediately                     ║
║    - No waiting for longest sequence                         ║
║    - Better GPU utilization                                  ║
║                                                              ║
║ 3. Optimized CUDA Kernels                                    ║
║    - FlashAttention for memory-efficient attention           ║
║    - Fused operations (less memory I/O)                      ║
║    - Custom kernels for common operations                    ║
║                                                              ║
║ 4. Tensor Parallelism                                        ║
║    - Split model across multiple GPUs                        ║
║    - Scale to models larger than single GPU memory           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1)
    
    outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=100))
""")


if __name__ == "__main__":
    show_vllm_features()
    print("\n" + "=" * 60)
    print("Choose an option:")
    print("  1. Run vLLM inference demo")
    print("  2. Compare vLLM vs HuggingFace speed")
    print("=" * 60)
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_vllm_inference()
    elif choice == "2":
        compare_vllm_vs_hf()
    else:
        print("Invalid choice")

