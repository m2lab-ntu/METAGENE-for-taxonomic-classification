#!/bin/bash
# Create a manageable subset of the full dataset

set -e

echo "================================================================================"
echo "Creating manageable training dataset"
echo "================================================================================"
echo ""

# Determine subset size
TRAIN_SUBSET_SIZE=${1:-1000000}  # Default 1M sequences
VAL_SUBSET_SIZE=${2:-250000}     # Default 250K sequences

echo "Creating subset with:"
echo "  Train: $TRAIN_SUBSET_SIZE sequences (~1-5GB)"
echo "  Val: $VAL_SUBSET_SIZE sequences (~0.25-1.25GB)"
echo ""

OUTPUT_DIR="/media/user/disk2/METAGENE/classification/data_subset"
mkdir -p $OUTPUT_DIR

echo "Extracting training subset..."
head -n $(($TRAIN_SUBSET_SIZE * 2)) /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
    > $OUTPUT_DIR/train_subset.fa

echo "Extracting validation subset..."
head -n $(($VAL_SUBSET_SIZE * 2)) /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
    > $OUTPUT_DIR/val_subset.fa

echo ""
echo "✓ Subset created:"
echo "  Train: $(grep -c "^>" $OUTPUT_DIR/train_subset.fa) sequences"
echo "  Val: $(grep -c "^>" $OUTPUT_DIR/val_subset.fa) sequences"
echo "  Location: $OUTPUT_DIR/"
echo ""
echo "Estimated size:"
du -h $OUTPUT_DIR/train_subset.fa
du -h $OUTPUT_DIR/val_subset.fa
echo ""
echo "Ready for training!"

