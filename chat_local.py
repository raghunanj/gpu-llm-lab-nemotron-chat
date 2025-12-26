#!/usr/bin/env python3
"""
Interactive chat with Nemotron-3-Nano-30B running locally on GPU.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
import sys
import io
import time

# Suppress all model warnings
warnings.filterwarnings("ignore")


def load_model():
    """Load the model with 4-bit quantization."""
    model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    
    print("🚀 Loading Nemotron-3-Nano-30B...")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Set pad_token to avoid warnings
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Suppress stderr during model load
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    finally:
        sys.stderr = old_stderr
    
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"✅ Model loaded! GPU memory: {allocated:.1f} GB")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> tuple:
    """Generate a response. Returns (response, generation_time, tokens_generated)."""
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(model.device)
    
    attention_mask = torch.ones_like(input_ids)
    
    # Suppress stderr during generation
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                top_p=0.9,
                use_cache=True,  # Model-managed caching for speed
            )
    finally:
        sys.stderr = old_stderr
    
    generation_time = time.time() - start_time
    tokens_generated = outputs.shape[1] - input_ids.shape[1]
    
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response, generation_time, tokens_generated


def extract_answer(response: str) -> str:
    """Extract the final answer from chain-of-thought response."""
    # Look for </think> tag which marks end of thinking
    if "</think>" in response:
        parts = response.split("</think>", 1)
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip()
    
    # Also check for <think> opening tag and remove everything inside
    if "<think>" in response and "</think>" in response:
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if cleaned.strip():
            return cleaned.strip()
    
    # Fallback: return full response if no think tags found
    return response.strip()


def main():
    print("=" * 60)
    print("  🤖 Nemotron-3-Nano-30B Local Chat")
    print("  Running on your RTX 4090 with 4-bit quantization")
    print("=" * 60)
    print()
    
    model, tokenizer = load_model()
    
    print()
    print("💡 Commands:")
    print("   quit/exit  - End the chat")
    print("   raw <msg>  - Show full response with thinking")
    print()
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            show_raw = user_input.lower().startswith('raw ')
            if show_raw:
                user_input = user_input[4:].strip()
            
            print("\n🤖 Nemotron: ", end="", flush=True)
            
            response, gen_time, num_tokens = generate_response(model, tokenizer, user_input)
            
            if show_raw:
                print(response)
            else:
                answer = extract_answer(response)
                print(answer)
            
            tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0
            print(f"\n   ⚡ {num_tokens} tokens in {gen_time:.1f}s ({tokens_per_sec:.1f} tok/s)")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
