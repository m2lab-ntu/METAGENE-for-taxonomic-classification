#!/bin/bash
# Diagnose and fix HuggingFace download issues

echo "================================================================================"
echo "Diagnosing HuggingFace Download Issues"
echo "================================================================================"
echo ""

# 1. Check internet connection
echo "[1/5] Checking internet connection..."
if ping -c 3 huggingface.co &> /dev/null; then
    echo "✓ Can reach huggingface.co"
else
    echo "❌ Cannot reach huggingface.co"
    echo "   Trying alternative endpoints..."
    
    if ping -c 3 hf-mirror.com &> /dev/null; then
        echo "✓ Can reach hf-mirror.com (mirror)"
        echo "   → Recommend using mirror download method"
    else
        echo "❌ Network connectivity issues detected"
    fi
fi
echo ""

# 2. Check disk space
echo "[2/5] Checking disk space..."
df -h ~/.cache/huggingface/ 2>/dev/null || df -h ~
echo ""

# 3. Check existing cache
echo "[3/5] Checking HuggingFace cache..."
CACHE_DIR=~/.cache/huggingface/hub/models--metagene-ai--METAGENE-1

if [ -d "$CACHE_DIR" ]; then
    echo "✓ Model cache exists at: $CACHE_DIR"
    
    # Check for incomplete files
    INCOMPLETE_COUNT=$(find "$CACHE_DIR/blobs" -name "*.incomplete" 2>/dev/null | wc -l)
    COMPLETE_COUNT=$(find "$CACHE_DIR/blobs" -type f ! -name "*.incomplete" 2>/dev/null | wc -l)
    
    echo "  Complete files: $COMPLETE_COUNT"
    echo "  Incomplete files: $INCOMPLETE_COUNT"
    
    if [ $INCOMPLETE_COUNT -gt 0 ]; then
        echo ""
        echo "  ⚠️  Found incomplete downloads. Options:"
        echo "     Option A: Remove incomplete files to restart clean"
        echo "     Option B: Try resume download (may still get stuck)"
        echo ""
        read -p "  Remove incomplete files? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "  Removing incomplete files..."
            find "$CACHE_DIR/blobs" -name "*.incomplete" -delete
            echo "  ✓ Cleaned up incomplete downloads"
        fi
    fi
else
    echo "  No cache found - will download from scratch"
fi
echo ""

# 4. Check Python packages
echo "[4/5] Checking Python packages..."
source /home/user/anaconda3/bin/activate METAGENE

python -c "
import sys
try:
    import transformers
    print(f'✓ transformers: {transformers.__version__}')
except:
    print('❌ transformers not installed')
    sys.exit(1)

try:
    import huggingface_hub
    print(f'✓ huggingface-hub: {huggingface_hub.__version__}')
except:
    print('⚠️  huggingface-hub not installed (optional but recommended)')
" || {
    echo ""
    echo "  Installing/updating required packages..."
    pip install -U transformers huggingface-hub[cli]
}
echo ""

# 5. Test download with timeout
echo "[5/5] Testing download capability..."
echo "Running quick test (will timeout after 30 seconds)..."
echo ""

timeout 30 python -c "
from transformers import AutoTokenizer
print('Attempting to download tokenizer (small file)...')
tokenizer = AutoTokenizer.from_pretrained('metagene-ai/METAGENE-1', trust_remote_code=True)
print('✓ Tokenizer download successful!')
" && {
    echo ""
    echo "✓ Download capability confirmed!"
    echo ""
    echo "================================================================================"
    echo "Recommended Next Steps:"
    echo "================================================================================"
    echo ""
    echo "Option 1: Try mirror download (recommended if in China/Asia)"
    echo "  bash download_with_mirror.sh"
    echo ""
    echo "Option 2: Try CLI download (most stable)"
    echo "  bash download_with_cli.sh"
    echo ""
    echo "Option 3: Manual download from browser"
    echo "  Visit: https://huggingface.co/metagene-ai/METAGENE-1/tree/main"
    echo "  Download all files to: ~/.cache/huggingface/hub/"
    echo ""
} || {
    echo ""
    echo "❌ Download test failed or timed out"
    echo ""
    echo "================================================================================"
    echo "Troubleshooting Recommendations:"
    echo "================================================================================"
    echo ""
    echo "1. Check firewall/proxy settings"
    echo "2. Try using VPN or different network"
    echo "3. Use mirror endpoint:"
    echo "   export HF_ENDPOINT=https://hf-mirror.com"
    echo "4. Use CLI download (more robust):"
    echo "   bash download_with_cli.sh"
    echo "5. Manual download from HuggingFace website"
    echo ""
}


