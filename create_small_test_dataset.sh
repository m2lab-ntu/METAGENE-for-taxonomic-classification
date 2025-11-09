#!/bin/bash
# Create a small test dataset for quick validation

set -e

echo "Creating small test datasets (10,000 training + 2,500 validation sequences)..."

# Create small train set
head -n 20000 /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
    > /media/user/disk2/METAGENE/classification/train_small.fa

# Create small val set  
head -n 5000 /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
    > /media/user/disk2/METAGENE/classification/val_small.fa

echo "✓ Small datasets created:"
echo "  Train: $(grep -c "^>" /media/user/disk2/METAGENE/classification/train_small.fa) sequences"
echo "  Val: $(grep -c "^>" /media/user/disk2/METAGENE/classification/val_small.fa) sequences"
echo ""
echo "Ready to test training!"

