import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from huggingface_hub import login
import config

def authenticate_huggingface():
    """Authenticates using the HF_TOKEN environment variable."""
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("✅ Authenticated to HuggingFace.")
    else:
        print("⚠️ Warning: HF_TOKEN not found in environment variables.")

def load_transformer_model(model_id: str):
    """
    Scalable model loader. Adapts to local 8GB RAM (CPU) or Server (GPU).
    """
    print(f"--- Environment Setup ---")
    print(f"Device detected: {config.DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # CASE 1: Server with GPU (Full or Half Precision)
    if config.HAS_CUDA:
        print("🚀 GPU detected. Applying optimized loading...")

        if config.USE_4BIT:
            print("Applying 4-bit quantization (bitsandbytes)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # FIX: Used torch_dtype and strictly enforced bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

    # CASE 2: Local PC with 8GB RAM (CPU only)
    else:
        print("⚠️ No GPU. Attempting CPU load with memory optimizations...")
        # FIX: float16 causes NaN logits in Gemma (infinite pad tokens).
        # bfloat16 is mandatory for CPU inference with this model.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cpu")

    print(f"✅ Model loaded on: {model.device}")
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    return model, tokenizer