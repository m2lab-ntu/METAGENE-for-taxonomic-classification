#!/bin/bash
# Fast Training Script - Balanced speed and accuracy
# Expected time: 1-1.5 days with 500K samples

set -e

echo "================================================================================"
echo "METAGENE Fast Training"
echo "================================================================================"
echo ""
echo "Configuration: fast_training.yaml"
echo "  - Epochs: 3 (vs 10)"
echo "  - Max Length: 64 (vs 128)"
echo "  - Grad Accum: 4 (vs 8)"
echo "  - Early Stopping Patience: 2 (vs 3)"
echo ""
echo "Recommended: Use 500K training samples for best balance"
echo "  Run: ./create_manageable_dataset.sh 500000 125000"
echo ""
read -p "Press Enter to continue with current dataset..."
echo ""

# Setup environment
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Activate conda
source /home/user/anaconda3/bin/activate METAGENE

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()"

# Create output dir with timestamp
OUTPUT_DIR="outputs/fast_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Check if manageable dataset exists, otherwise use subset
if [ -f "train_manageable.fa" ]; then
    TRAIN_FASTA="train_manageable.fa"
    VAL_FASTA="val_manageable.fa"
    echo "✓ Using manageable dataset"
else
    TRAIN_FASTA="/media/user/disk2/METAGENE/classification/data_subset/train_subset.fa"
    VAL_FASTA="/media/user/disk2/METAGENE/classification/data_subset/val_subset.fa"
    echo "✓ Using existing subset dataset"
fi

echo "Starting fast training..."
echo "Output directory: $OUTPUT_DIR"
echo ""

python train.py \
  --config configs/fast_training.yaml \
  --train_fasta $TRAIN_FASTA \
  --val_fasta $VAL_FASTA \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir $OUTPUT_DIR \
  --batch_size 1 \
  2>&1 | tee $OUTPUT_DIR/training.log

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check metrics: cat $OUTPUT_DIR/final_metrics.json"
echo "  2. View plots: ls $OUTPUT_DIR/plots/"
echo "  3. Evaluate: python evaluate.py --checkpoint $OUTPUT_DIR/checkpoints/best.pt"

