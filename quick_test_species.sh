#!/bin/bash
# Quick test training for species classification
# Runs for 1 epoch only to verify everything works

set -e

echo "================================================================================"
echo "METAGENE Species Classification - QUICK TEST"
echo "================================================================================"
echo ""
echo "This is a QUICK TEST run (1 epoch only)"
echo "To verify the full pipeline works with your dataset"
echo ""

# Setup environment
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Activate conda
source /home/user/anaconda3/bin/activate METAGENE

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Create output directory
OUTPUT_DIR="outputs/species_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

echo "Starting quick test training (1 epoch)..."
echo ""

# Run training with minimal epochs
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
  --val_fasta /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir $OUTPUT_DIR \
  --batch_size 1 \
  --max_epochs 1 \
  2>&1 | tee $OUTPUT_DIR/training.log

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quick test completed successfully!"
    echo ""
    echo "Your dataset is compatible with the pipeline!"
    echo ""
    echo "Next steps:"
    echo "  1. Run full training: bash train_species_classification.sh"
    echo "  2. Or adjust configuration as needed"
else
    echo "❌ Test failed. Please check logs:"
    echo "   $OUTPUT_DIR/training.log"
fi
echo "================================================================================"

