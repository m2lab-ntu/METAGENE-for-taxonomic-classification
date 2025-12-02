#!/bin/bash

# 50-Species Diagnosis Experiment Runner

# Paths
FULL_MAPPING="/media/user/disk2/METAGENE/classification/species_mapping_converted.tsv"
FULL_TRAIN="/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa"
FULL_VAL="/media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa" 

DATA_DIR="data_diagnosis_50"
OUTPUT_DIR="outputs/diagnosis_50_species"
CONFIG="metaclassifier/configs/rtx4090_optimized.yaml"

# 1. Prepare Data
echo "Step 1: Preparing 50-species subset..."
python prepare_subset_data.py \
    --mapping "$FULL_MAPPING" \
    --train_fasta "$FULL_TRAIN" \
    --val_fasta "$FULL_VAL" \
    --output_dir "$DATA_DIR" \
    --num_species 50 \
    --max_reads_train 1000 \
    --max_reads_val 200

if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed."
    exit 1
fi

# 2. Run Training
echo "Step 2: Starting training..."
/home/user/anaconda3/envs/METAGENE/bin/python metaclassifier/train.py \
    --config "$CONFIG" \
    --train_fasta "$DATA_DIR/subset_train.fa" \
    --val_fasta "$DATA_DIR/subset_val.fa" \
    --mapping_tsv "$DATA_DIR/subset_mapping.tsv" \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Training failed."
    exit 1
fi

echo "Diagnosis experiment complete! Results in $OUTPUT_DIR"
