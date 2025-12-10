#!/bin/bash

# Benchmark Evo2 on Zymo Dataset (Small Subset)
# Uses 200 reads/species to ensure reasonable training time.

# Paths
TRAIN_FASTA="zymo_train_small.fa"
VAL_FASTA="zymo_val_small.fa"
MAPPING_TSV="zymo_mapping_small.tsv" # Note: Mapping must match the small dataset if generated separately? 
# Wait, generate_zymo_reads.py generates mapping alongside fasta.
# I generated zymo_mapping_small.tsv with zymo_train_small.fa.
# But zymo_val_small.fa has zymo_mapping_val_small.tsv.
# train.py takes one mapping file.
# Since class IDs are deterministic (sorted), zymo_mapping_small.tsv should work for both.

echo "=================================================="
echo "Starting Evo2 Benchmark on Zymo (Small)..."
echo "=================================================="
OUTPUT_DIR="outputs/zymo_evo2_small"
CONFIG="metaclassifier/configs/evo2_transformer.yaml"

# Evo2 needs special environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/user/Metagenomics/evo2:/home/user/Metagenomics/evo2/vortex:$PYTHONPATH"

/home/user/anaconda3/envs/METAGENE/bin/python metaclassifier/train.py \
    --config "$CONFIG" \
    --train_fasta "$TRAIN_FASTA" \
    --val_fasta "$VAL_FASTA" \
    --mapping_tsv "$MAPPING_TSV" \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Evo2 training failed."
    exit 1
fi

echo "Evo2 Benchmark Complete! Results in $OUTPUT_DIR"
