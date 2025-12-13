#!/bin/bash
# Run Remote Training (Full Dataset) on A100/H100
# Usage: ./run_remote_full.sh [PATH_TO_DATASET_ROOT]

# Default dataset path (Change this or pass as argument)
DATASET_ROOT="${1:-/data/MetaTransformer_original_dataset}"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "Error: Dataset directory '$DATASET_ROOT' not found!"
    echo "Usage: ./run_remote_full.sh [PATH_TO_DATASET_ROOT]"
    exit 1
fi

echo "=================================================="
echo "Starting Remote Training (Full Dataset)"
echo "Hardware Target: A100/H100 (80GB)"
echo "Dataset Root: $DATASET_ROOT"
echo "Config: remote_a100_full.yaml"
echo "=================================================="

# Create output directory
mkdir -p outputs/remote_full_run

# Run training
# Note: Adapting paths based on assumed dataset structure
# If 'training_reads' is a directory of files:
#   --train_fasta $DATASET_ROOT/training_reads
# If 'training_reads' is a monolithic file (as discovered locally):
#   --train_fasta $DATASET_ROOT/training_reads/training_reads.fa

# Auto-detect structure
if [ -f "$DATASET_ROOT/training_reads/training_reads.fa" ]; then
    TRAIN_FASTA="$DATASET_ROOT/training_reads/training_reads.fa"
    echo "Detected monolithic training file: $TRAIN_FASTA"
else
    TRAIN_FASTA="$DATASET_ROOT/training_reads"
    echo "Assuming directory structure for training: $TRAIN_FASTA"
fi

if [ -f "$DATASET_ROOT/val_reads/val_reads.fa" ]; then
    VAL_FASTA="$DATASET_ROOT/val_reads/val_reads.fa"
else
    VAL_FASTA="$DATASET_ROOT/val_reads"
fi

echo "Train Data: $TRAIN_FASTA"
echo "Val Data: $VAL_FASTA"

python metaclassifier/train.py \
    --config metaclassifier/configs/remote_a100_full.yaml \
    --train_fasta "$TRAIN_FASTA" \
    --val_fasta "$VAL_FASTA" \
    --mapping_tsv "$DATASET_ROOT/species_mapping.tab" \
    --output_dir outputs/remote_full_run

echo "Training complete. Check outputs/remote_full_run"
