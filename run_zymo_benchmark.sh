#!/bin/bash

# Benchmark GENERanno and Evo2 on Zymo Dataset
# This script runs training for both models on the Zymo mock community data.

# Paths
TRAIN_FASTA="zymo_train.fa"
VAL_FASTA="zymo_val.fa"
MAPPING_TSV="zymo_mapping.tsv"

# 1. GENERanno Benchmark
echo "=================================================="
echo "Starting GENERanno Benchmark on Zymo..."
echo "=================================================="
OUTPUT_DIR_GEN="outputs/zymo_generanno"
CONFIG_GEN="metaclassifier/configs/generanno_transformer.yaml"

/home/user/anaconda3/envs/METAGENE/bin/python metaclassifier/train.py \
    --config "$CONFIG_GEN" \
    --train_fasta "$TRAIN_FASTA" \
    --val_fasta "$VAL_FASTA" \
    --mapping_tsv "$MAPPING_TSV" \
    --output_dir "$OUTPUT_DIR_GEN"

if [ $? -ne 0 ]; then
    echo "Error: GENERanno training failed."
    # Continue to Evo2? Maybe not.
    exit 1
fi

# 2. Evo2 Benchmark
echo "=================================================="
echo "Starting Evo2 Benchmark on Zymo..."
echo "=================================================="
OUTPUT_DIR_EVO="outputs/zymo_evo2"
CONFIG_EVO="metaclassifier/configs/evo2_transformer.yaml"

# Evo2 needs special environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/user/Metagenomics/evo2:/home/user/Metagenomics/evo2/vortex:$PYTHONPATH"

/home/user/anaconda3/envs/METAGENE/bin/python metaclassifier/train.py \
    --config "$CONFIG_EVO" \
    --train_fasta "$TRAIN_FASTA" \
    --val_fasta "$VAL_FASTA" \
    --mapping_tsv "$MAPPING_TSV" \
    --output_dir "$OUTPUT_DIR_EVO"

if [ $? -ne 0 ]; then
    echo "Error: Evo2 training failed."
    exit 1
fi

echo "=================================================="
echo "Benchmark Complete!"
echo "GENERanno Results: $OUTPUT_DIR_GEN"
echo "Evo2 Results: $OUTPUT_DIR_EVO"
echo "=================================================="
