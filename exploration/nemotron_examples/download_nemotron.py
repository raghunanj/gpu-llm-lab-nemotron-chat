"""
Download Nemotron-3-Nano Model

This script helps you download the Nemotron-3-Nano model from Hugging Face.

SETUP:
1. Go to https://huggingface.co/settings/tokens and create a token (Read access)
2. Go to https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct
   or the Nemotron-3 model page and accept the license
3. Run this script with your token

Usage:
    python download_nemotron.py --token YOUR_HF_TOKEN
    
Or set environment variable:
    export HF_TOKEN=your_token_here
    python download_nemotron.py
"""

import os
import sys
import argparse


def list_nvidia_models(token=None):
    """List available NVIDIA models on Hugging Face."""
    from huggingface_hub import list_models
    
    print("🔍 Searching for NVIDIA Nemotron models...")
    print()
    
    models = list(list_models(author='nvidia', search='Nemotron', limit=30, token=token))
    
    print("Available NVIDIA Nemotron models:")
    print("=" * 70)
    
    for m in models:
        print(f"\n📦 {m.modelId}")
        if hasattr(m, 'downloads'):
            print(f"   Downloads: {m.downloads:,}")
        if hasattr(m, 'likes'):
            print(f"   Likes: {m.likes}")
    
    print()
    print("=" * 70)
    return models


def download_model(model_id: str, token: str = None):
    """Download a model from Hugging Face."""
    from huggingface_hub import snapshot_download
    
    print(f"📥 Downloading {model_id}...")
    print("   This may take a while for large models.")
    print()
    
    try:
        path = snapshot_download(
            repo_id=model_id,
            token=token,
            local_dir=None,  # Uses HF cache
            resume_download=True
        )
        print(f"✅ Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error: {e}")
        
        if "401" in str(e) or "gated" in str(e).lower():
            print()
            print("This model requires license acceptance:")
            print(f"  1. Go to: https://huggingface.co/{model_id}")
            print("  2. Log in and accept the license agreement")
            print("  3. Run this script again")
        return None


def test_model(model_id: str, token: str = None):
    """Test loading a model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"🧪 Testing {model_id}...")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    
    print("   Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        token=token,
        trust_remote_code=True
    )
    
    # Print memory usage
    allocated = torch.cuda.memory_allocated() / (1024**3)
    print(f"   GPU Memory used: {allocated:.2f} GB")
    
    # Test generation
    print("   Testing generation...")
    prompt = "What is machine learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print()
    print("📝 Test response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Download Nemotron models from Hugging Face")
    parser.add_argument("--token", type=str, help="Hugging Face token", 
                        default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download", type=str, help="Model ID to download")
    parser.add_argument("--test", type=str, help="Model ID to test")
    
    args = parser.parse_args()
    
    if not args.token:
        print("⚠️  No Hugging Face token provided.")
        print()
        print("To get a token:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Create a new token with 'Read' access")
        print("  3. Run: python download_nemotron.py --token YOUR_TOKEN")
        print()
        print("Or set environment variable:")
        print("  export HF_TOKEN=your_token_here")
        print()
        
        # Still try to list public models
        if args.list:
            list_nvidia_models(None)
        return
    
    if args.list:
        list_nvidia_models(args.token)
    
    if args.download:
        download_model(args.download, args.token)
    
    if args.test:
        test_model(args.test, args.token)
    
    if not (args.list or args.download or args.test):
        print("Nemotron - Nemotron Model Downloader")
        print("=" * 50)
        print()
        print("Options:")
        print("  --list              List available NVIDIA Nemotron models")
        print("  --download MODEL_ID Download a specific model")
        print("  --test MODEL_ID     Download and test a model")
        print()
        print("Recommended models for RTX 4090 (24GB):")
        print()
        print("  1. nvidia/Nemotron-Mini-4B-Instruct")
        print("     Small, fast, good for learning (~3GB with 4-bit)")
        print()
        print("  2. nvidia/Mistral-NeMo-Minitron-8B-Instruct")
        print("     Balanced quality and speed (~6GB with 4-bit)")
        print()
        print("  3. nvidia/Llama-3.1-Nemotron-70B-Instruct")
        print("     High quality but needs heavy quantization")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} --test nvidia/Nemotron-Mini-4B-Instruct")


if __name__ == "__main__":
    main()


