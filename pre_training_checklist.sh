#!/bin/bash
# Pre-training checklist - verify everything is ready

set -e

echo "================================================================================"
echo "METAGENE Species Classification - Pre-Training Checklist"
echo "================================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

ISSUES=0

# 1. Check data files
echo "1. Checking data files..."
if [ -f "/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa" ]; then
    SIZE=$(du -h /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa | cut -f1)
    check_pass "Training data found (${SIZE})"
else
    check_fail "Training data not found"
    ISSUES=$((ISSUES+1))
fi

if [ -f "/media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa" ]; then
    SIZE=$(du -h /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa | cut -f1)
    check_pass "Validation data found (${SIZE})"
else
    check_fail "Validation data not found"
    ISSUES=$((ISSUES+1))
fi
echo ""

# 2. Check mapping file
echo "2. Checking mapping file..."
if [ -f "species_mapping_converted.tsv" ]; then
    LINES=$(wc -l < species_mapping_converted.tsv)
    check_pass "Mapping file found (${LINES} lines = $((LINES-1)) classes + header)"
    
    # Show first few entries
    echo "   Sample entries:"
    head -5 species_mapping_converted.tsv | while read line; do
        echo "     $line"
    done
else
    check_fail "Mapping file not found - run: python prepare_species_mapping.py"
    ISSUES=$((ISSUES+1))
fi
echo ""

# 3. Check model cache
echo "3. Checking METAGENE-1 model..."
if [ -d "/media/user/disk2/.cache/huggingface/hub/models--metagene-ai--METAGENE-1" ]; then
    SIZE=$(du -sh /media/user/disk2/.cache/huggingface/hub/models--metagene-ai--METAGENE-1 | cut -f1)
    check_pass "METAGENE-1 model cached (${SIZE})"
else
    check_warn "METAGENE-1 model not cached - will download on first run (~14GB)"
    echo "   Run: bash download_to_disk2.sh"
fi
echo ""

# 4. Check GPU
echo "4. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    check_pass "GPU detected: ${GPU_INFO}"
    
    # Check available memory
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$FREE_MEM" -gt 20000 ]; then
        check_pass "GPU memory available: ${FREE_MEM} MB"
    else
        check_warn "GPU memory may be insufficient: ${FREE_MEM} MB free"
        echo "   Recommended: >20GB free for initial model loading"
    fi
else
    check_fail "nvidia-smi not found"
    ISSUES=$((ISSUES+1))
fi
echo ""

# 5. Check disk space
echo "5. Checking disk space..."
DISK_FREE=$(df -h /media/user/disk2 | tail -1 | awk '{print $4}')
check_pass "Disk space available: ${DISK_FREE}"

# Check if we have enough space for outputs
DISK_FREE_GB=$(df -BG /media/user/disk2 | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_FREE_GB" -lt 50 ]; then
    check_warn "Disk space may be tight (<50GB free)"
    echo "   Training outputs can be large"
fi
echo ""

# 6. Check Python environment
echo "6. Checking Python environment..."
source /home/user/anaconda3/bin/activate METAGENE 2>/dev/null || true
if python -c "import torch; import transformers; import peft" 2>/dev/null; then
    check_pass "Python packages available"
    
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "   PyTorch: ${TORCH_VERSION}"
    
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        check_pass "CUDA available"
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        echo "   CUDA: ${CUDA_VERSION}"
    else
        check_fail "CUDA not available in PyTorch"
        ISSUES=$((ISSUES+1))
    fi
else
    check_fail "Required packages not found"
    echo "   Run: conda activate METAGENE"
    ISSUES=$((ISSUES+1))
fi
echo ""

# 7. Check configuration files
echo "7. Checking configuration files..."
if [ -f "configs/rtx4090_optimized.yaml" ]; then
    check_pass "RTX 4090 config found"
else
    check_fail "RTX 4090 config not found"
    ISSUES=$((ISSUES+1))
fi

if [ -f "train.py" ]; then
    check_pass "Training script found"
else
    check_fail "Training script not found"
    ISSUES=$((ISSUES+1))
fi
echo ""

# Summary
echo "================================================================================"
if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Ready to start training.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Quick test (1 epoch):  bash quick_test_species.sh"
    echo "  2. Full training:         bash train_species_classification.sh"
else
    echo -e "${RED}✗ Found ${ISSUES} issue(s). Please fix before training.${NC}"
fi
echo "================================================================================"

