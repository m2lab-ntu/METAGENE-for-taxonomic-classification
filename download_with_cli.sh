#!/bin/bash
# Download METAGENE-1 model using huggingface-cli

set -e

echo "================================================================================"
echo "Downloading METAGENE-1 Model using huggingface-cli"
echo "================================================================================"
echo ""

# Activate conda environment
source /home/user/anaconda3/bin/activate METAGENE

# Install huggingface-cli if not present
echo "[1/3] Checking huggingface-cli..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install -U huggingface-hub[cli]
fi

echo "✓ huggingface-cli ready"
echo ""

# Optional: Set mirror endpoint for faster download (uncomment if needed)
# export HF_ENDPOINT=https://hf-mirror.com

echo "[2/3] Downloading model files..."
echo "This will download ~16GB of data. The download will:"
echo "  - Resume automatically if interrupted"
echo "  - Show progress bar"
echo "  - Be more stable than Python API"
echo ""
echo "Press Ctrl+C to cancel, or press Enter to start..."
read

# Download using CLI
huggingface-cli download \
    metagene-ai/METAGENE-1 \
    --resume-download \
    --local-dir-use-symlinks False \
    --repo-type model

echo ""
echo "[3/3] Verifying download..."
python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('metagene-ai/METAGENE-1', trust_remote_code=True)
print(f'✓ Tokenizer OK: vocab_size={len(tokenizer)}')

print('Loading model...')
model = AutoModel.from_pretrained('metagene-ai/METAGENE-1', torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
print(f'✓ Model OK: hidden_size={model.config.hidden_size}')

print('')
print('=' * 80)
print('✓ MODEL READY FOR USE!')
print('=' * 80)
"

echo ""
echo "Model cached at: ~/.cache/huggingface/hub/"
echo "You can now run training!"


