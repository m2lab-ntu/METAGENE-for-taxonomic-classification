#!/bin/bash
# Training script for full species classification dataset
# Dataset: 100GB train + 25GB validation, 3507 classes

set -e

echo "================================================================================"
echo "METAGENE Species Classification Training"
echo "================================================================================"
echo ""
echo "Dataset Information:"
echo "  Training:   /media/user/disk2/full_labeled_species_train_reads_shuffled/"
echo "  Validation: /media/user/disk2/full_labeled_species_val_reads_shuffled/"
echo "  Classes:    3507 species"
echo "  Mapping:    species_mapping_converted.tsv"
echo ""
echo "GPU: RTX 4090 (24GB) - Optimized Configuration"
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
python -c "import torch; torch.cuda.empty_cache(); print('✓ GPU cache cleared')"
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""

# Create output directory
OUTPUT_DIR="outputs/species_classification_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

echo "================================================================================"
echo "Starting Training"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Config: configs/rtx4090_optimized.yaml"
echo "  - Batch size: 1 (gradient accumulation: 8)"
echo "  - Max length: 128 tokens"
echo "  - LoRA rank: 4"
echo "  - Gradient checkpointing: ON"
echo "  - Expected memory: ~13-15GB / 24GB"
echo ""
echo "Estimated training time:"
echo "  - This is a LARGE dataset (100GB)"
echo "  - Expect several days for full training"
echo "  - Consider using --max_epochs 1 for testing first"
echo ""

# Auto-start training (no user confirmation needed for background runs)
echo "Starting training automatically..."
echo ""

# Run training
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
  --val_fasta /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir $OUTPUT_DIR \
  --batch_size 1 \
  --max_epochs 10 \
  2>&1 | tee $OUTPUT_DIR/training.log

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo "================================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "To evaluate:"
    echo "  python evaluate.py \\"
    echo "    --ckpt $OUTPUT_DIR/checkpoints/best.pt \\"
    echo "    --split val \\"
    echo "    --output_dir $OUTPUT_DIR/eval_results"
    echo ""
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "================================================================================"
    echo ""
    echo "Check logs at: $OUTPUT_DIR/training.log"
    echo ""
    if grep -q "CUDA out of memory" $OUTPUT_DIR/training.log; then
        echo "Detected OOM error. Try:"
        echo "  1. Reduce max_length to 64 in config"
        echo "  2. Use fp16 instead of bf16"
        echo "  3. Increase grad_accum_steps to 16"
    fi
fi
echo "================================================================================"

