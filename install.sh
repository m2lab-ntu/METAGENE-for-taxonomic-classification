#!/bin/bash
set -e

# Check conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda."
    exit 1
fi

# Check active env is METAGENE
if [ "$CONDA_DEFAULT_ENV" != "METAGENE" ]; then
    echo "ERROR: Active conda env is '$CONDA_DEFAULT_ENV', but must be 'METAGENE'."
    echo "Activate with: conda activate METAGENE"
    exit 1
fi

echo "Installing into conda env: METAGENE"

# Detect CUDA version for PyTorch
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
if [[ -z "$CUDA_VERSION" ]]; then
    echo "WARNING: Could not detect CUDA version. Installing CPU-only PyTorch."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
else
    echo "Detected CUDA $CUDA_VERSION. Installing PyTorch with CUDA 12.1 support."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

# Install core dependencies
pip install transformers>=4.40.0 datasets peft accelerate
pip install pandas numpy scikit-learn matplotlib seaborn tqdm pyyaml rich biopython
pip install safetensors lightning

# Optional: WandB (commented by default)
# pip install wandb

# Add minbpe to Python path (no setup.py available)
echo "Adding minbpe to Python path..."
echo "export PYTHONPATH=\"/media/user/disk2/METAGENE/metagene-pretrain/train:\$PYTHONPATH\"" >> ~/.bashrc

echo "✓ Installation complete in METAGENE env."
