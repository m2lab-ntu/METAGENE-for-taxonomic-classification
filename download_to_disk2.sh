#!/bin/bash
# Download METAGENE-1 model to /media/user/disk2 (has plenty of space)

set -e

echo "================================================================================"
echo "Downloading METAGENE-1 Model to /media/user/disk2"
echo "================================================================================"
echo ""

# Set custom cache directory
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export HF_DATASETS_CACHE=/media/user/disk2/.cache/huggingface

echo "✓ Using custom cache directory: $HF_HOME"
echo ""

# Check available space
echo "Available space on /media/user/disk2:"
df -h /media/user/disk2 | grep -v Filesystem
echo ""

# Create cache directory
mkdir -p $HF_HOME
echo "✓ Cache directory created"
echo ""

# Activate conda environment
source /home/user/anaconda3/bin/activate METAGENE

# Optional: Use mirror for faster download (uncomment if needed)
# export HF_ENDPOINT=https://hf-mirror.com

echo "================================================================================"
echo "Starting Download (this will take some time)"
echo "================================================================================"
echo ""

# Download using Python
python << 'PYEOF'
import os
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "metagene-ai/METAGENE-1"

print("[1/2] Downloading tokenizer...")
print("-" * 80)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    resume_download=True
)
print(f"✓ Tokenizer downloaded: vocab_size={len(tokenizer)}")
print()

print("[2/2] Downloading model (~16GB)...")
print("-" * 80)
print("This may take 30-60 minutes...")
print()

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    resume_download=True,
    device_map="auto",
    low_cpu_mem_usage=True
)
print()
print(f"✓ Model downloaded successfully!")
print(f"  Hidden size: {model.config.hidden_size}")
print()

# Test
print("[3/3] Testing model...")
test_seq = "ACGTACGTACGT"
inputs = tokenizer(test_seq, return_tensors="pt")
# Move inputs to the same device as model
if hasattr(model, 'device'):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(f"✓ Model works! Output shape: {outputs.last_hidden_state.shape}")

print()
print("=" * 80)
print("✓ SUCCESS!")
print("=" * 80)
print()
print(f"Model cached at: {os.environ.get('HF_HOME')}")
print()
PYEOF

echo ""
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo "To use this model cache in your training, run:"
echo ""
echo "  export HF_HOME=/media/user/disk2/.cache/huggingface"
echo "  export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface"
echo ""
echo "Or add these lines to your ~/.bashrc to make it permanent:"
echo ""
echo "  echo 'export HF_HOME=/media/user/disk2/.cache/huggingface' >> ~/.bashrc"
echo "  echo 'export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface' >> ~/.bashrc"
echo ""
echo "================================================================================"


