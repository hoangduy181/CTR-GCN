#!/bin/bash
# Quick inference script for UCLA dataset
# Place this in the CTR-GCN root directory

echo "=========================================="
echo "CTR-GCN UCLA Inference"
echo "=========================================="

# Check if weights are provided as argument
if [ -z "$1" ]; then
    echo "Usage: ./run_inference_ucla.sh <weights_path> [device_id]"
    echo ""
    echo "Example:"
    echo "  ./run_inference_ucla.sh work_dir/ucla/ctrgcn_joint/runs-65-12345.pt 0"
    echo ""
    echo "Or use the Python script for auto-detection:"
    echo "  python scripts/inference_ucla.py"
    exit 1
fi

WEIGHTS=$1
DEVICE=${2:-0}

# Check if weights file exists
if [ ! -f "$WEIGHTS" ]; then
    echo "Error: Weights file not found: $WEIGHTS"
    exit 1
fi

# Run inference
python main.py \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/inference \
    --phase test \
    --save-score True \
    --weights "$WEIGHTS" \
    --device $DEVICE

echo ""
echo "Inference completed! Check work_dir/ucla/inference/ for results."
