# GPU Environment Setup Guide

This guide helps you set up CTR-GCN in a GPU-enabled environment for training.

## Environment Options

### Option 1: Google Colab (Recommended for Quick Start)

**Pros:**
- Free GPU access (T4, V100, A100)
- No local setup required
- Easy to share and collaborate

**Cons:**
- Session timeout (12 hours)
- Limited storage
- Need to re-upload data each session

### Option 2: Local GPU Machine

**Pros:**
- Full control
- Persistent storage
- No time limits

**Cons:**
- Requires GPU hardware
- Setup complexity

### Option 3: Cloud Services (AWS, GCP, Azure)

**Pros:**
- Scalable
- Multiple GPU options
- Pay-as-you-go

**Cons:**
- Costs money
- Setup complexity

## Google Colab Setup

### Step 1: Upload Project to Colab

1. Upload the CTR-GCN folder to Google Drive
2. Open a new Colab notebook
3. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Install Dependencies

```python
# Navigate to project directory
import os
os.chdir('/content/drive/MyDrive/CTR-GCN')  # Adjust path as needed

# Install dependencies
!pip install -r requirements_updated.txt
!pip install -e torchlight

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 3: Prepare Data

```python
# If data is in Google Drive, create symlink or copy
# Or download directly in Colab
```

### Step 4: Train Model

```python
!python main.py \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/ctrgcn_joint \
    --device 0
```

## Local GPU Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- cuDNN installed

### Step 1: Check GPU

```bash
nvidia-smi
```

### Step 2: Create Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml
conda activate ctrgcn

# Or create manually
conda create -n ctrgcn python=3.9 -y
conda activate ctrgcn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements_updated.txt
pip install -e torchlight
```

### Step 3: Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Step 4: Train Model

```bash
python main.py \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/ctrgcn_joint \
    --device 0
```

## Training Tips

### Monitor Training

```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Or use TensorBoard
tensorboard --logdir work_dir/ucla/ctrgcn_joint/runs
```

### Adjust Batch Size

If you get CUDA out of memory errors, reduce batch size in config:

```yaml
batch_size: 16  # Reduce from default
test_batch_size: 32  # Reduce from default
```

### Save Checkpoints

The model automatically saves checkpoints. To resume training:

```bash
python main.py \
    --config config/ucla/default.yaml \
    --work-dir work_dir/ucla/ctrgcn_joint \
    --device 0 \
    --start-epoch 30 \
    --weights work_dir/ucla/ctrgcn_joint/runs-30-12345.pt
```

## Expected Training Time

- **UCLA Dataset**: ~2-4 hours on T4 GPU
- **NTU RGB+D 60**: ~12-24 hours on T4 GPU
- **NTU RGB+D 120**: ~24-48 hours on T4 GPU

## Troubleshooting

### CUDA Out of Memory

1. Reduce batch size in config
2. Reduce number of workers: `--num-worker 8`
3. Use gradient accumulation (modify code)

### GPU Not Detected

1. Check CUDA installation: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with correct CUDA version

### Slow Training

1. Increase batch size (if memory allows)
2. Use mixed precision training (modify code)
3. Use DataParallel for multiple GPUs
