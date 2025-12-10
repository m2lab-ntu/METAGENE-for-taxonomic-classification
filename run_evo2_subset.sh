#!/bin/bash

# Evo2 Experiment Runner (50-Species Subset)

# Paths
DATA_DIR="data_diagnosis_50"
OUTPUT_DIR="outputs/evo2_subset_50"
CONFIG="metaclassifier/configs/evo2_transformer.yaml"

# Ensure data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist. Run run_diagnosis_experiment.sh first."
    exit 1
fi

# Run Training
echo "Step 1: Starting training with Evo2 (Local)..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/user/Metagenomics/evo2:/home/user/Metagenomics/evo2/vortex:$PYTHONPATH"
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

echo "Evo2 experiment complete! Results in $OUTPUT_DIR"
