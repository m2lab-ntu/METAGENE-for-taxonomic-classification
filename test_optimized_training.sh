#!/bin/bash
# Test optimized training on RTX 4090

echo "================================================================================"
echo "Testing Optimized METAGENE Classification Training on RTX 4090"
echo "================================================================================"
echo ""

# Setup environment
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "✓ Environment variables set"
echo ""

# Activate conda
source /home/user/anaconda3/bin/activate METAGENE

echo "✓ METAGENE environment activated"
echo ""

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ GPU cache cleared')"
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""

echo "================================================================================"
echo "Starting Training with Optimized Config"
echo "================================================================================"
echo ""
echo "Optimizations enabled:"
echo "  - Gradient checkpointing: ON (saves ~50% activation memory)"
echo "  - Sequence length: 128 (reduced from 512)"
echo "  - LoRA rank: 4 (reduced from 8)"
echo "  - Target modules: q_proj, v_proj only"
echo "  - Batch size: 1"
echo "  - Gradient accumulation: 8 steps"
echo "  - Memory clearing: Every 10 steps"
echo ""
echo "Expected memory usage: ~18-20GB (should fit in 24GB)"
echo ""

# Run training
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta examples/example_train.fa \
  --val_fasta examples/example_val.fa \
  --mapping_tsv examples/labels.tsv \
  --output_dir outputs/optimized_test \
  --batch_size 1 \
  --max_epochs 1

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS! Training completed without OOM!"
    echo "================================================================================"
    echo ""
    echo "You can now train on your full dataset with:"
    echo ""
    echo "  python train.py \\"
    echo "    --config configs/rtx4090_optimized.yaml \\"
    echo "    --train_fasta YOUR_TRAIN.fa \\"
    echo "    --val_fasta YOUR_VAL.fa \\"
    echo "    --mapping_tsv YOUR_MAPPING.tsv \\"
    echo "    --output_dir outputs/full_training \\"
    echo "    --max_epochs 10"
    echo ""
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "================================================================================"
    echo ""
    echo "If still OOM, try:"
    echo "  1. Reduce max_length further (64 instead of 128)"
    echo "  2. Set precision to 'fp16-mixed' instead of 'bf16-mixed'"
    echo "  3. Use CPU offloading (slower but works)"
    echo ""
fi
echo "================================================================================"

