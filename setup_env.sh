#!/bin/bash
# Quick setup script for METAGENE classification environment

echo "================================================================================"
echo "METAGENE Classification Environment Setup"
echo "================================================================================"
echo ""

# 設置 HuggingFace cache 到有足夠空間的 disk2
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface

# 設置 CUDA 記憶體分配優化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "✓ Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# 激活 conda 環境
source /home/user/anaconda3/bin/activate METAGENE

echo "✓ Conda environment activated: METAGENE"
echo ""

# 檢查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    echo ""
fi

echo "================================================================================"
echo "Environment Ready!"
echo "================================================================================"
echo ""
echo "Available commands:"
echo "  - Test data loading:  python test_dataloader_only.py"
echo "  - Train (needs 40GB+): python train.py --config configs/default.yaml ..."
echo "  - Inference:          python predict.py --input data.fa --ckpt ..."
echo ""
echo "Note: Training METAGENE-1 (7B) requires 40GB+ GPU"
echo "      Current RTX 4090 (24GB) can only do inference"
echo "================================================================================"

