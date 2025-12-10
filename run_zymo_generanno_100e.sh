#!/bin/bash

# Run GENERanno for 100 Epochs on Zymo Dataset

# Paths
TRAIN_FASTA="zymo_train.fa"
VAL_FASTA="zymo_val.fa"
MAPPING_TSV="zymo_mapping.tsv"
OUTPUT_DIR="outputs/zymo_generanno_100e"
CONFIG="metaclassifier/configs/generanno_transformer_100e.yaml"

echo "=================================================="
echo "Starting GENERanno Training (100 Epochs)..."
echo "=================================================="

# Ensure environment variables if needed (though GENERanno worked fine before)
# export PYTHONPATH if needed, but it seems handled in script or env

/home/user/anaconda3/envs/METAGENE/bin/python metaclassifier/train.py \
    --config "$CONFIG" \
    --train_fasta "$TRAIN_FASTA" \
    --val_fasta "$VAL_FASTA" \
    --mapping_tsv "$MAPPING_TSV" \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Training failed."
    exit 1
fi

echo "=================================================="
echo "Training Complete! Results in $OUTPUT_DIR"
echo "=================================================="
