# CTR-GCN Setup Guide

This guide will help you set up the CTR-GCN environment using conda.

## Prerequisites

- Anaconda or Miniconda installed
- CUDA-capable GPU (recommended for training)
- Linux, macOS, or Windows with WSL2

## Step 1: Create Conda Environment

Create a new conda environment with Python 3.9:

```bash
conda create -n ctrgcn python=3.9 -y
conda activate ctrgcn
```

## Step 2: Install PyTorch

Install PyTorch with CUDA support (for GPU acceleration):

```bash
# For CUDA 11.8 (most common)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CPU only (not recommended for training)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

## Step 3: Install Dependencies

Install the updated requirements:

```bash
cd /path/to/CTR-GCN
pip install -r requirements_updated.txt
```

## Step 4: Install torchlight

Install the custom torchlight package:

```bash
pip install -e torchlight
```

## Step 5: Verify Installation

Test that everything is installed correctly:

```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; import yaml; import tqdm; print('Core dependencies OK')"
```

## Step 6: Test the Code

Run a quick import test:

```bash
python -c "from model.ctrgcn import Model; print('Model import successful')"
```

## Alternative: Using environment.yml

You can also create the environment directly from the environment.yml file:

```bash
conda env create -f environment.yml
conda activate ctrgcn
pip install -e torchlight
```

## Troubleshooting

If you encounter issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common problems and solutions.

## Next Steps

- Prepare your dataset following the instructions in the main README.md
- Configure your training parameters in the config files
- Start training with: `python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ntu60/xsub/ctrgcn --device 0`
