"""
shared_models.py
Loads Phi-2 and the embedding model ONCE.
All 4 agents import from here instead of each loading their own copy.
Phi-2 loaded in 4-bit quantization to fit in 4GB VRAM (RTX 3050 Laptop)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("Loading shared Phi-2 (once for all agents)...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# -----------------------------
# Phi-2 — shared across all agents
# 4-bit quantization on GPU (~1.5GB VRAM)
# Full precision on CPU fallback
# -----------------------------
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

if DEVICE == "cuda":

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    phi_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Shared Phi-2 loaded in 4-bit on GPU")

else:

    phi_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype="auto"
    ).to(DEVICE)

    print("Shared Phi-2 loaded on CPU")

# -----------------------------
# Embedding model — shared
# -----------------------------
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Shared embedding model loaded")
print("Shared models ready\n")