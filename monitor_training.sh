#!/bin/bash
# Monitor training progress

# Find the latest output directory
LATEST_DIR=$(ls -dt /media/user/disk2/METAGENE/classification/outputs/species_*/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No training directory found"
    exit 1
fi

echo "================================================================================"
echo "METAGENE Training Monitor"
echo "================================================================================"
echo "Training directory: $LATEST_DIR"
echo ""

# Check if log exists
LOG_FILE="${LATEST_DIR}/training.log"
if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not yet created"
    exit 0
fi

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo ""

# Check training status
echo "Training Status:"
echo "----------------"

# Check if still loading data
if grep -q "Loading sequences:" "$LOG_FILE" | tail -1; then
    LAST_LOAD=$(grep "Loading sequences:" "$LOG_FILE" | tail -1)
    echo "📦 Data Loading: $LAST_LOAD"
elif grep -q "Loaded train dataset" "$LOG_FILE"; then
    TRAIN_SIZE=$(grep "Loaded train dataset" "$LOG_FILE" | tail -1)
    echo "✓ $TRAIN_SIZE"
    
    if grep -q "Loaded val dataset" "$LOG_FILE"; then
        VAL_SIZE=$(grep "Loaded val dataset" "$LOG_FILE" | tail -1)
        echo "✓ $VAL_SIZE"
    fi
    
    # Check training progress
    if grep -q "Epoch" "$LOG_FILE"; then
        echo ""
        echo "Training Progress:"
        echo "------------------"
        grep -E "Epoch \[|Step \[|Loss:" "$LOG_FILE" | tail -20
    fi
else
    echo "⏳ Initializing..."
fi

echo ""
echo "================================================================================"
echo "Recent Log (last 30 lines):"
echo "================================================================================"
tail -30 "$LOG_FILE" | grep -v "Loading sequences:" || tail -30 "$LOG_FILE"

echo ""
echo "================================================================================"
echo "Commands:"
echo "  Full log:     tail -f $LOG_FILE"
echo "  GPU monitor:  watch -n 1 nvidia-smi"
echo "  This monitor: bash monitor_training.sh"
echo "================================================================================"

