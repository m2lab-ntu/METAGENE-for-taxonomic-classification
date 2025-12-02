#!/bin/bash
set -e

# Setup paths
BASE_DIR="/media/user/disk2/METAGENE/classification"
DATA_DIR="/media/user/disk2/full_labeled_species_train_reads_shuffled"
TRAIN_FA="$DATA_DIR/train_reads_shuffled_fixed.fa"
MAPPING_TSV="$BASE_DIR/species_mapping_converted.tsv"
TEST_DIR="$BASE_DIR/test_pipeline_output"

mkdir -p $TEST_DIR

echo "Creating small test datasets..."
head -n 200 $TRAIN_FA > $TEST_DIR/train_small.fa
head -n 200 $TRAIN_FA > $TEST_DIR/val_small.fa

echo "Running training test..."
python $BASE_DIR/metaclassifier/train.py \
    --config $BASE_DIR/metaclassifier/configs/metagene_bpe.yaml \
    --train_fasta $TEST_DIR/train_small.fa \
    --val_fasta $TEST_DIR/val_small.fa \
    --mapping_tsv $MAPPING_TSV \
    --output_dir $TEST_DIR/model_output

echo "Running inference test..."
python $BASE_DIR/metaclassifier/inference_abundance.py \
    --config $BASE_DIR/metaclassifier/configs/metagene_bpe.yaml \
    --checkpoint $TEST_DIR/model_output/best.pt \
    --input_fasta $TEST_DIR/val_small.fa \
    --output_csv $TEST_DIR/abundance.csv \
    --mapping_tsv $MAPPING_TSV

echo "Checking output..."
head -n 5 $TEST_DIR/abundance.csv

echo "✓ Test pipeline completed successfully!"
