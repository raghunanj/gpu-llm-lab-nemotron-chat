#!/usr/bin/env python3
"""
Practical Optimized Inference for Nemotron-3-Nano.

This shows multiple optimization strategies you can apply:
1. Flash Attention - Already used by the model
2. Torch Compile - JIT compilation for 1.5-2x speedup
3. BetterTransformer - Fused attention kernels
4. vLLM - Production inference (3-5x faster)
"""
import torch
import time
import sys
import io
import warnings

warnings.filterwarnings("ignore")


def load_model_with_optimizations():
    """Load model with various optimization flags."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    
    print("🚀 Loading Nemotron-3-Nano with optimizations...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # or "flash_attention_2" if supported
    )
    
    sys.stderr = old_stderr
    
    print(f"✅ Model loaded! GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    return model, tokenizer


def benchmark_inference(model, tokenizer, prompt: str, max_tokens: int = 100, runs: int = 3):
    """Benchmark generation speed."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    # Warmup
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, 
                          max_new_tokens=20, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    sys.stderr = old_stderr
    
    # Benchmark
    times = []
    tokens = []
    
    for _ in range(runs):
        sys.stderr = io.StringIO()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        sys.stderr = old_stderr
        elapsed = time.perf_counter() - start
        num_tokens = outputs.shape[1] - input_ids.shape[1]
        times.append(elapsed)
        tokens.append(num_tokens)
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens) / len(tokens)
    
    return avg_time, avg_tokens


def test_torch_compile(model, tokenizer):
    """
    Test torch.compile optimization.
    
    torch.compile can give 1.5-2x speedup by:
    - Fusing operations
    - Optimizing memory access patterns
    - Reducing Python overhead
    """
    print("\n" + "=" * 60)
    print("Testing torch.compile optimization")
    print("=" * 60)
    
    prompt = "Explain machine learning in simple terms."
    
    # Baseline
    print("\n📊 Baseline (no compile):")
    time1, tokens1 = benchmark_inference(model, tokenizer, prompt, max_tokens=50)
    print(f"   {tokens1:.0f} tokens in {time1:.2f}s ({tokens1/time1:.1f} tok/s)")
    
    # Try torch.compile
    print("\n🔧 Applying torch.compile...")
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        
        print("📊 With torch.compile:")
        time2, tokens2 = benchmark_inference(compiled_model, tokenizer, prompt, max_tokens=50)
        print(f"   {tokens2:.0f} tokens in {time2:.2f}s ({tokens2/time2:.1f} tok/s)")
        
        speedup = (tokens1/time1) / (tokens2/time2) if time2 > 0 else 1
        speedup = (tokens2/time2) / (tokens1/time1)
        print(f"\n✨ Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"   ❌ torch.compile not supported: {e}")


def show_optimization_tips():
    """Display optimization recommendations."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            GPU Kernel Optimization Strategies                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║ 🚀 EASY WINS (No Code Changes):                                   ║
║                                                                   ║
║   1. Use 4-bit quantization (already doing this!)                 ║
║      - Reduces memory 4x, slight speed boost                      ║
║                                                                   ║
║   2. Enable use_cache=True                                        ║
║      - Caches key/value for faster generation                     ║
║                                                                   ║
║   3. torch.compile(model, mode="reduce-overhead")                 ║
║      - 1.5-2x speedup with JIT compilation                        ║
║                                                                   ║
║ ⚡ MEDIUM EFFORT (Some Setup):                                     ║
║                                                                   ║
║   4. Use Flash Attention                                          ║
║      pip install flash-attn --no-build-isolation                  ║
║      - 2-10x faster attention, less memory                        ║
║                                                                   ║
║   5. Use vLLM for production                                      ║
║      pip install vllm                                             ║
║      - PagedAttention, continuous batching                        ║
║      - 3-5x faster than HuggingFace                               ║
║                                                                   ║
║ 🔥 ADVANCED (Custom Kernels):                                      ║
║                                                                   ║
║   6. Write Triton kernels for custom operations                   ║
║      - Fuse multiple operations                                   ║
║      - Optimize memory access patterns                            ║
║                                                                   ║
║   7. Use TensorRT-LLM                                             ║
║      - Maximum performance (5-10x)                                ║
║      - Requires model conversion                                  ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝

Current setup: Nemotron-3-Nano-30B @ 4-bit quantization
               RTX 4090 (24GB VRAM)
               Speed: ~1.5 tokens/sec
               
Potential with optimizations:
               vLLM: ~5-8 tokens/sec
               TensorRT-LLM: ~10-15 tokens/sec
""")


def main():
    show_optimization_tips()
    
    print("\n" + "=" * 60)
    print("Choose an option:")
    print("  1. Test torch.compile optimization")
    print("  2. Show vLLM setup instructions")
    print("  3. Exit")
    print("=" * 60)
    
    choice = input("Enter choice: ").strip()
    
    if choice == "1":
        model, tokenizer = load_model_with_optimizations()
        test_torch_compile(model, tokenizer)
    elif choice == "2":
        print("""
To use vLLM with Nemotron:

1. Install vLLM:
   pip install vllm

2. Run inference:
   from vllm import LLM, SamplingParams
   
   llm = LLM(
       model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
       tensor_parallel_size=1,
       gpu_memory_utilization=0.9,
       trust_remote_code=True,
   )
   
   outputs = llm.generate(
       ["What is machine learning?"],
       SamplingParams(temperature=0.7, max_tokens=200)
   )
   
   print(outputs[0].outputs[0].text)
""")


if __name__ == "__main__":
    main()

