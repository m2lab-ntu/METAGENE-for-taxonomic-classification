#!/bin/bash
# Ultra-Fast Training Script - Maximum speed
# Expected time: 4-6 hours with 100K samples

set -e

echo "================================================================================"
echo "METAGENE Ultra-Fast Training"
echo "================================================================================"
echo ""
echo "⚡ SPEED MODE - Optimized for rapid iteration"
echo ""
echo "Configuration: fast_training.yaml"
echo "Recommended dataset: 100K training + 25K validation samples"
echo ""
echo "Creating ultra-fast dataset..."

# Create ultra-fast dataset if not exists
if [ ! -f "train_ultrafast.fa" ]; then
    echo "Creating 100K train + 25K val dataset..."
    head -n 200000 /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
        > train_ultrafast.fa
    head -n 50000 /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
        > val_ultrafast.fa
    
    TRAIN_COUNT=$(grep -c "^>" train_ultrafast.fa || echo "0")
    VAL_COUNT=$(grep -c "^>" val_ultrafast.fa || echo "0")
    
    echo "✓ Created ultra-fast dataset:"
    echo "  Train: $TRAIN_COUNT sequences"
    echo "  Val: $VAL_COUNT sequences"
else
    echo "✓ Ultra-fast dataset already exists"
fi

echo ""
read -p "Press Enter to start ultra-fast training..."
echo ""

# Setup environment
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Activate conda
source /home/user/anaconda3/bin/activate METAGENE

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()"

# Create output dir
OUTPUT_DIR="outputs/ultrafast_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "⚡ Starting ultra-fast training..."
echo "Output: $OUTPUT_DIR"
echo ""

python train.py \
  --config configs/fast_training.yaml \
  --train_fasta train_ultrafast.fa \
  --val_fasta val_ultrafast.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir $OUTPUT_DIR \
  --batch_size 1 \
  2>&1 | tee $OUTPUT_DIR/training.log

echo ""
echo "================================================================================"
echo "⚡ Ultra-Fast Training Complete!"
echo "================================================================================"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Quick evaluation:"
python -c "
import json
try:
    with open('$OUTPUT_DIR/final_metrics.json') as f:
        metrics = json.load(f)
        print(f\"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\")
        print(f\"  Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}\")
except:
    print('  Metrics file not found')
"

