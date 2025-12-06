#!/bin/bash
# Inference script for UCLA dataset
# Usage: ./scripts/inference_ucla.sh [weights_path] [device]

# Default values
WEIGHTS_PATH=${1:-"work_dir/ucla/ctrgcn_joint/runs-65-*.pt"}
DEVICE=${2:-0}
CONFIG="config/ucla/default.yaml"
WORK_DIR="work_dir/ucla/inference"

# Check if weights file exists
if [ ! -f "$WEIGHTS_PATH" ] && [ ! -z "$(ls -A $WEIGHTS_PATH 2>/dev/null)" ]; then
    echo "Error: Weights file not found: $WEIGHTS_PATH"
    echo "Please provide a valid weights path or train the model first."
    exit 1
fi

# Create work directory if it doesn't exist
mkdir -p $WORK_DIR

echo "=========================================="
echo "Running Inference on UCLA Dataset"
echo "=========================================="
echo "Config: $CONFIG"
echo "Weights: $WEIGHTS_PATH"
echo "Device: $DEVICE"
echo "Work Dir: $WORK_DIR"
echo "=========================================="

# Run inference
python main.py \
    --config $CONFIG \
    --work-dir $WORK_DIR \
    --phase test \
    --save-score True \
    --weights "$WEIGHTS_PATH" \
    --device $DEVICE

echo "=========================================="
echo "Inference completed!"
echo "Results saved in: $WORK_DIR"
echo "=========================================="
