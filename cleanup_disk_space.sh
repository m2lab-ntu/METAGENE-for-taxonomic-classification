#!/bin/bash
# Clean up disk space safely

echo "================================================================================"
echo "Disk Space Cleanup Tool"
echo "================================================================================"
echo ""
echo "Current disk usage:"
df -h / | grep -v Filesystem
echo ""

cleanup_conda_cache() {
    echo "Cleaning conda package cache..."
    conda clean --all --yes
    echo "✓ Conda cache cleaned"
}

cleanup_pip_cache() {
    echo "Cleaning pip cache..."
    pip cache purge
    echo "✓ Pip cache cleaned"
}

cleanup_hf_incomplete() {
    echo "Removing incomplete HuggingFace downloads..."
    find ~/.cache/huggingface/hub -name "*.incomplete" -delete 2>/dev/null
    echo "✓ Incomplete downloads removed"
}

echo "================================================================================"
echo "Safe Cleanup Options:"
echo "================================================================================"
echo ""
echo "1. Clean conda package cache (~96GB)"
echo "   - Safe: Only removes cached packages, not installed ones"
echo ""
echo "2. Clean pip cache (~32GB)"
echo "   - Safe: Only removes pip download cache"
echo ""
echo "3. Remove incomplete HuggingFace downloads"
echo "   - Safe: Only removes failed downloads"
echo ""
echo "4. ALL OF THE ABOVE (recommended)"
echo ""
echo "5. Skip cleanup"
echo ""

read -p "Select option (1-5): " choice

case $choice in
    1)
        cleanup_conda_cache
        ;;
    2)
        cleanup_pip_cache
        ;;
    3)
        cleanup_hf_incomplete
        ;;
    4)
        cleanup_conda_cache
        cleanup_pip_cache
        cleanup_hf_incomplete
        ;;
    5)
        echo "Skipping cleanup"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "Cleanup Complete!"
echo "================================================================================"
echo ""
echo "New disk usage:"
df -h / | grep -v Filesystem
echo ""
echo "Free space gained!"


