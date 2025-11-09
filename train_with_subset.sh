#!/bin/bash
# Train with manageable dataset subset

set -e

echo "================================================================================"
echo "METAGENE Species Classification - Training with Subset"
echo "================================================================================"
echo ""

# Setup environment
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Activate conda
source /home/user/anaconda3/bin/activate METAGENE

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()"

# Dataset paths
TRAIN_FASTA="/media/user/disk2/METAGENE/classification/data_subset/train_subset.fa"
VAL_FASTA="/media/user/disk2/METAGENE/classification/data_subset/val_subset.fa"
MAPPING_TSV="/media/user/disk2/METAGENE/classification/species_mapping_converted.tsv"

# Check if subset exists
if [ ! -f "$TRAIN_FASTA" ]; then
    echo "❌ Training subset not found!"
    echo "Please run: bash create_manageable_dataset.sh"
    exit 1
fi

echo "Dataset info:"
echo "  Train sequences: $(grep -c "^>" $TRAIN_FASTA)"
echo "  Val sequences: $(grep -c "^>" $VAL_FASTA)"
echo ""

# Create output directory
OUTPUT_DIR="outputs/subset_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting training..."
echo "Output: $OUTPUT_DIR"
echo ""

# Run training
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta $TRAIN_FASTA \
  --val_fasta $VAL_FASTA \
  --mapping_tsv $MAPPING_TSV \
  --output_dir $OUTPUT_DIR \
  --batch_size 1 \
  --max_epochs 10 \
  2>&1 | tee $OUTPUT_DIR/training.log

echo ""
echo "================================================================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================================================================"

