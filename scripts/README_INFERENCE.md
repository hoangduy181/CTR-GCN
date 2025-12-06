# Inference Scripts for UCLA Dataset

This directory contains scripts to run inference on the UCLA dataset.

## Prerequisites

1. **Data**: UCLA data should be downloaded and extracted to `data/NW-UCLA/all_sqe/`
2. **Model Weights**: You need either:
   - Pretrained weights (download from [Google Drive](https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing))
   - Or train your own model first

## Quick Start

### Option 1: Using Python Script (Recommended)

```bash
# With explicit weights path
python scripts/inference_ucla.py --weights path/to/weights.pt --device 0

# Auto-detect weights from training directory
python scripts/inference_ucla.py --device 0

# Custom config and work directory
python scripts/inference_ucla.py \
    --weights work_dir/ucla/ctrgcn_joint/runs-65-12345.pt \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/inference \
    --device 0
```

### Option 2: Using Bash Script

```bash
# Make script executable
chmod +x scripts/inference_ucla.sh

# Run inference
./scripts/inference_ucla.sh path/to/weights.pt 0

# Or with default values (searches for weights automatically)
./scripts/inference_ucla.sh
```

### Option 3: Direct Command

```bash
python main.py \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/inference \
    --phase test \
    --save-score True \
    --weights work_dir/ucla/ctrgcn_joint/runs-65-12345.pt \
    --device 0
```

## Training First (If No Pretrained Weights)

If you don't have pretrained weights, train the model first:

```bash
python main.py \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/ctrgcn_joint \
    --device 0
```

After training, the weights will be saved in `work_dir/ucla/ctrgcn_joint/runs-{epoch}-{step}.pt`

## Output Files

After inference, you'll find:

- `work_dir/ucla/inference/log.txt` - Inference log
- `work_dir/ucla/inference/config.yaml` - Configuration used
- `work_dir/ucla/inference/epoch1_test_score.pkl` - Prediction scores (if --save-score True)
- `work_dir/ucla/inference/epoch1_test_each_class_acc.csv` - Per-class accuracy
- `work_dir/ucla/inference/{weights_name}_wrong.txt` - Wrong predictions
- `work_dir/ucla/inference/{weights_name}_right.txt` - Correct predictions

## Different Modalities

To test with different modalities (bone, motion), modify the config file or use command-line arguments:

```bash
python main.py \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/inference_bone \
    --phase test \
    --save-score True \
    --weights path/to/weights.pt \
    --device 0 \
    --test-feeder-args data_path="bone"
```

## Troubleshooting

1. **No weights found**: Train the model first or download pretrained weights
2. **CUDA out of memory**: Reduce `test_batch_size` in config file
3. **Data not found**: Ensure UCLA data is in `data/NW-UCLA/all_sqe/`
