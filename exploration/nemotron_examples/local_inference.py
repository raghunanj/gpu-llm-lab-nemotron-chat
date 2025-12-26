"""
Nemotron Local Inference

Run Nemotron and NVIDIA models locally on your GPU.

Your RTX 4090 (24GB VRAM) Options:
1. Nemotron 3 Nano 30B with 4-bit quantization (~15-18GB VRAM)
2. Mistral-NeMo-Minitron-8B (~16GB in BF16, ~8GB in 4-bit)
3. Llama-3.1-Nemotron-70B-Instruct with heavy quantization
4. Nemotron-Mini-4B-Instruct (~8GB in BF16, ~4GB in 4-bit) - BEST for learning

Requirements:
    pip install torch transformers accelerate bitsandbytes
    
For vLLM (recommended for production):
    pip install vllm

For Ollama (easiest):
    curl -fsSL https://ollama.com/install.sh | sh
"""

import os
import sys
from typing import Optional, Generator
from dataclasses import dataclass

# Check available packages
PACKAGES_STATUS = {}

try:
    import torch
    PACKAGES_STATUS["torch"] = torch.__version__
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        GPU_NAME = "N/A"
        GPU_MEMORY = 0
except ImportError:
    PACKAGES_STATUS["torch"] = None
    CUDA_AVAILABLE = False
    GPU_NAME = "N/A"
    GPU_MEMORY = 0

try:
    import transformers
    PACKAGES_STATUS["transformers"] = transformers.__version__
except ImportError:
    PACKAGES_STATUS["transformers"] = None

try:
    import bitsandbytes
    PACKAGES_STATUS["bitsandbytes"] = "installed"
except ImportError:
    PACKAGES_STATUS["bitsandbytes"] = None


# Available NVIDIA models for local inference
NVIDIA_MODELS = {
    "nemotron-mini-4b": {
        "name": "Nemotron-Mini-4B-Instruct",
        "hf_id": "nvidia/Nemotron-Mini-4B-Instruct",
        "size_gb": 8,
        "quantized_gb": 3,
        "description": "4B parameter model, great for learning and light tasks",
        "recommended_for_24gb": True,
    },
    "minitron-8b": {
        "name": "Mistral-NeMo-Minitron-8B-Instruct", 
        "hf_id": "nvidia/Mistral-NeMo-Minitron-8B-Instruct",
        "size_gb": 16,
        "quantized_gb": 6,
        "description": "8B parameter model based on Mistral-NeMo, good balance",
        "recommended_for_24gb": True,
    },
    "nemotron-nano-30b": {
        "name": "Nemotron-3-Nano-30B",
        "hf_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",  # Correct HF model ID
        "size_gb": 60,
        "quantized_gb": 18,
        "description": "Latest Nemotron 3 Nano, needs 4-bit quantization for 24GB",
        "recommended_for_24gb": True,  # With 4-bit quantization
        "note": "Use 4-bit quantization (load_in_4bit=True)"
    },
    "nemotron-nano-9b": {
        "name": "Nemotron-Nano-9B-v2",
        "hf_id": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "size_gb": 18,
        "quantized_gb": 6,
        "description": "Smaller Nemotron Nano, easier to run on 24GB",
        "recommended_for_24gb": True,
    },
    "llama-nemotron-70b": {
        "name": "Llama-3.1-Nemotron-70B-Instruct",
        "hf_id": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        "size_gb": 140,
        "quantized_gb": 40,
        "description": "Largest model, needs heavy quantization or multiple GPUs",
        "recommended_for_24gb": False,
    },
}


@dataclass
class LocalModel:
    """Wrapper for a locally loaded model."""
    model: any
    tokenizer: any
    model_name: str
    device: str = "cuda"


def print_environment():
    """Print the current environment status."""
    print("=" * 60)
    print("🖥️  Local Environment Status")
    print("=" * 60)
    
    print(f"\n📦 Packages:")
    for pkg, version in PACKAGES_STATUS.items():
        status = f"✅ {version}" if version else "❌ Not installed"
        print(f"   {pkg}: {status}")
    
    print(f"\n🎮 GPU:")
    if CUDA_AVAILABLE:
        print(f"   ✅ CUDA Available")
        print(f"   GPU: {GPU_NAME}")
        print(f"   VRAM: {GPU_MEMORY:.1f} GB")
    else:
        print(f"   ❌ CUDA Not Available")
    
    print(f"\n📊 Recommended Models for your GPU:")
    for key, model in NVIDIA_MODELS.items():
        if model["recommended_for_24gb"]:
            print(f"   • {model['name']}")
            print(f"     VRAM: ~{model['quantized_gb']}GB (quantized) / ~{model['size_gb']}GB (full)")
            if model.get("note"):
                print(f"     Note: {model['note']}")
    
    print("=" * 60)


def load_model_quantized(
    model_id: str,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto"
) -> LocalModel:
    """
    Load a model with quantization for reduced VRAM usage.
    
    Args:
        model_id: Hugging Face model ID
        load_in_4bit: Use 4-bit quantization (most memory efficient)
        load_in_8bit: Use 8-bit quantization (better quality, more memory)
        device_map: Device mapping strategy
        
    Returns:
        LocalModel instance
    """
    if PACKAGES_STATUS["transformers"] is None:
        raise ImportError("transformers not installed. Run: pip install transformers")
    
    if (load_in_4bit or load_in_8bit) and PACKAGES_STATUS["bitsandbytes"] is None:
        raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"🔄 Loading {model_id}...")
    print(f"   Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'None'}")
    
    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None
    )
    
    print(f"✅ Model loaded successfully!")
    
    # Print memory usage
    if CUDA_AVAILABLE:
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return LocalModel(
        model=model,
        tokenizer=tokenizer,
        model_name=model_id,
        device="cuda" if CUDA_AVAILABLE else "cpu"
    )


def generate_response(
    local_model: LocalModel,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    stream: bool = False
) -> str | Generator[str, None, None]:
    """
    Generate a response from the loaded model.
    
    Args:
        local_model: Loaded LocalModel instance
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        stream: Whether to stream the output
        
    Returns:
        Generated text or generator for streaming
    """
    model = local_model.model
    tokenizer = local_model.tokenizer
    
    # Prepare chat format if the model supports it
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(local_model.device)
    
    if stream:
        # Streaming generation
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        return response


class LocalNemotronClient:
    """
    Client for local Nemotron/NVIDIA model inference.
    Drop-in replacement for the API-based NemotronClient.
    """
    
    def __init__(
        self,
        model_key: str = "nemotron-mini-4b",
        load_in_4bit: bool = True
    ):
        """
        Initialize local model.
        
        Args:
            model_key: Key from NVIDIA_MODELS dict
            load_in_4bit: Whether to use 4-bit quantization
        """
        if model_key not in NVIDIA_MODELS:
            available = ", ".join(NVIDIA_MODELS.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")
        
        model_info = NVIDIA_MODELS[model_key]
        self.model_info = model_info
        self.local_model = load_model_quantized(
            model_info["hf_id"],
            load_in_4bit=load_in_4bit
        )
    
    @property
    def is_available(self) -> bool:
        return self.local_model is not None
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ):
        """
        Chat with the local model.
        Compatible with the API client interface.
        """
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        return generate_response(
            self.local_model,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
    
    def simple_chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Simple single-turn chat."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        return self.chat(messages)


def demo():
    """Demonstrate local model inference."""
    print("\n🚀 Local NVIDIA Model Inference Demo")
    print_environment()
    
    # Check if we can proceed
    if not CUDA_AVAILABLE:
        print("\n⚠️  CUDA not available. Local inference requires a GPU.")
        print("   You can still use the NVIDIA NIM API (cloud) instead.")
        return
    
    if PACKAGES_STATUS["transformers"] is None:
        print("\n❌ Required packages not installed.")
        print("   Run: pip install torch transformers accelerate bitsandbytes")
        return
    
    print("\n" + "=" * 60)
    print("🎯 Available Models for Your RTX 4090 (24GB)")
    print("=" * 60)
    
    for i, (key, model) in enumerate(NVIDIA_MODELS.items(), 1):
        if model["recommended_for_24gb"]:
            print(f"\n{i}. {model['name']}")
            print(f"   Model ID: {model['hf_id']}")
            print(f"   {model['description']}")
            print(f"   VRAM: ~{model['quantized_gb']}GB (4-bit) / ~{model['size_gb']}GB (full)")
    
    print("\n" + "-" * 60)
    print("To load a model, use:")
    print("""
    from local_inference import LocalNemotronClient
    
    # For smaller/faster model (recommended to start):
    client = LocalNemotronClient("nemotron-mini-4b", load_in_4bit=True)
    
    # For larger model:
    client = LocalNemotronClient("minitron-8b", load_in_4bit=True)
    
    # For latest Nemotron 3 Nano (needs 4-bit):
    client = LocalNemotronClient("nemotron-nano-30b", load_in_4bit=True)
    
    # Generate response:
    response = client.simple_chat("What is machine learning?")
    print(response)
    """)


def interactive_demo():
    """Run an interactive demo with a local model."""
    print("\n🎮 Interactive Local Model Demo")
    print_environment()
    
    if not CUDA_AVAILABLE:
        print("❌ CUDA required for local inference.")
        return
    
    print("\nSelect a model:")
    recommended = [(k, v) for k, v in NVIDIA_MODELS.items() if v["recommended_for_24gb"]]
    for i, (key, model) in enumerate(recommended, 1):
        print(f"  {i}. {model['name']} (~{model['quantized_gb']}GB with 4-bit)")
    
    try:
        choice = input("\nChoice (1-3, or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return
        
        idx = int(choice) - 1
        if 0 <= idx < len(recommended):
            model_key = recommended[idx][0]
            print(f"\n📥 Loading {NVIDIA_MODELS[model_key]['name']}...")
            print("   (This may take a few minutes on first run to download)")
            
            client = LocalNemotronClient(model_key, load_in_4bit=True)
            
            print("\n✅ Model loaded! Start chatting (type 'quit' to exit)")
            print("-" * 60)
            
            while True:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue
                
                print("🤖 Assistant: ", end="", flush=True)
                response = client.simple_chat(user_input)
                print(response)
        else:
            print("Invalid choice.")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        demo()

