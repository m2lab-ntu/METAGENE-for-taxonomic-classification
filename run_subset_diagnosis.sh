#!/bin/bash
# Run Local Subset Diagnosis (100 Species)
# Uses the stable config (LR 2e-5) from the 100-epoch run but on real data.

# Source environment
source setup_env.sh

echo "=================================================="
echo "Starting Subset Diagnosis (100 Species)..."
echo "Datset: data_subset_100"
echo "Config: rtx4090_optimized_100e.yaml (LR=2e-5)"
echo "Epochs: 20"
echo "=================================================="

python metaclassifier/train.py \
    --config metaclassifier/configs/rtx4090_subset_diagnosis.yaml \
    --train_fasta data_subset_100/subset_train.fa \
    --val_fasta data_subset_100/subset_val.fa \
    --mapping_tsv data_subset_100/subset_mapping.tsv \
    --output_dir outputs/subset_diagnosis_100
