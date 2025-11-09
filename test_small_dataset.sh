#!/bin/bash
# Quick training test with small dataset

set -e

echo "================================================================================"
echo "METAGENE Quick Training Test (Small Dataset)"
echo "================================================================================"
echo ""
echo "Dataset: 10,000 train + 2,500 val sequences"
echo "Expected time: 1-2 hours"
echo ""

# Setup environment
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Activate conda
source /home/user/anaconda3/bin/activate METAGENE

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()"

# Create output dir
OUTPUT_DIR="outputs/small_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting training..."
echo ""

# Run training
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta train_small.fa \
  --val_fasta val_small.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir $OUTPUT_DIR \
  --batch_size 1 \
  --max_epochs 2 \
  2>&1 | tee $OUTPUT_DIR/training.log

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training test completed!"
    echo ""
    echo "Results: $OUTPUT_DIR"
    echo ""
    echo "Next: Run full training with 75M sequences"
    echo "  bash train_species_classification.sh"
else
    echo "❌ Training failed. Check: $OUTPUT_DIR/training.log"
fi
echo "================================================================================"

