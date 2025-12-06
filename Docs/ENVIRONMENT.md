# Environment Details

This document describes the conda environment setup and rationale for package choices.

## Environment Specification

### Base Environment
- **Name**: `ctrgcn`
- **Python**: 3.9 (recommended) or 3.8
- **Package Manager**: Conda + pip

### Why Python 3.9?

- **Stability**: Python 3.9 is stable and well-tested
- **Colab Compatibility**: Google Colab supports Python 3.9
- **Package Support**: All required packages have good support for Python 3.9
- **Performance**: Better than Python 3.6, not too new to cause compatibility issues
- **EOL Status**: Python 3.9 is supported until October 2025

### Why Not Python 3.10+?

- Some packages may have compatibility issues
- Colab may default to Python 3.9
- More conservative approach for stability

## Core Dependencies

### PyTorch Ecosystem

**PyTorch**: >=1.8.0
- Modern CUDA support (11.8, 12.1)
- Better performance optimizations
- Colab-compatible
- Required for modern GPU acceleration

**torchvision**: >=0.9.0
- Compatible with PyTorch >=1.8.0
- Required for some utilities

**torchaudio**: Optional
- Included for completeness
- Not strictly required for CTR-GCN

### Scientific Computing

**NumPy**: >=1.21.0
- Required by PyTorch
- Security fixes from older versions
- Better performance

**SciPy**: >=1.7.0
- Used by some data processing utilities
- Compatible with NumPy >=1.21.0

**scikit-learn**: >=0.24.0
- Used for metrics and utilities
- Compatible with newer NumPy

### Configuration & Utilities

**PyYAML**: >=5.4.1
- Configuration file parsing
- Updated to use `safe_load()` for security

**tqdm**: >=4.60.0
- Progress bars
- Minor update for compatibility

**tensorboard**: >=2.8.0
- Replaces deprecated tensorboardX
- Official TensorBoard package
- Better maintenance

### Custom Packages

**torchlight**: Local package
- Custom framework utilities
- Installed via `pip install -e torchlight`

## Package Installation Strategy

### Conda vs Pip

**Conda** is used for:
- Python itself
- PyTorch (with CUDA support)
- Large scientific packages (if needed)

**Pip** is used for:
- Most Python packages
- Custom/local packages (torchlight)
- Packages not available in conda

### Why This Approach?

1. **CUDA Support**: Conda provides better CUDA-enabled PyTorch installation
2. **Flexibility**: Pip allows more package versions
3. **Compatibility**: Mix of both works well in practice
4. **Colab**: Colab uses pip primarily, so pip-first approach is more compatible

## Environment Files

### environment.yml

Conda environment file that:
- Specifies Python version
- Includes PyTorch with CUDA
- Lists essential conda packages
- Can be used to recreate environment

### requirements_updated.txt

Pip requirements file with:
- Only essential packages
- Modern, compatible versions
- Removed unnecessary packages
- Colab-compatible versions

## CUDA Considerations

### CUDA Versions

The environment supports multiple CUDA versions:
- **CUDA 11.8**: Most common, well-supported
- **CUDA 12.1**: Newer, better performance
- **CPU**: Fallback option (not recommended for training)

### Installation Commands

```bash
# CUDA 11.8 (recommended)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Colab Compatibility

### Colab Environment

Google Colab provides:
- Python 3.9 (default)
- CUDA 11.8 or 12.1 (depending on runtime)
- Pre-installed packages (some may conflict)

### Colab Setup

1. Install PyTorch with matching CUDA version
2. Install requirements_updated.txt
3. Install torchlight
4. Restart runtime if needed

### Colab-Specific Notes

- Colab may have pre-installed packages that conflict
- Restart runtime after installing packages
- Use `!pip install` in Colab cells
- Mount Google Drive for data storage

## Version Compatibility Matrix

| Package | Min Version | Max Version | Tested Version |
|---------|-------------|-------------|----------------|
| Python | 3.8 | 3.10 | 3.9 |
| PyTorch | 1.8.0 | Latest | 2.0.0+ |
| NumPy | 1.21.0 | Latest | 1.24.0 |
| PyYAML | 5.4.1 | Latest | 6.0 |
| tensorboard | 2.8.0 | Latest | 2.13.0 |

## Environment Recreation

To recreate the environment:

```bash
# From environment.yml
conda env create -f environment.yml
conda activate ctrgcn
pip install -e torchlight

# Or manually
conda create -n ctrgcn python=3.9 -y
conda activate ctrgcn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements_updated.txt
pip install -e torchlight
```

## Verification

After setup, verify:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import numpy as np
print(f"NumPy: {np.__version__}")

import yaml
print(f"PyYAML: {yaml.__version__}")
```

## Maintenance

### Updating Packages

- **PyTorch**: Update carefully, test CUDA compatibility
- **Other packages**: Generally safe to update to latest
- **Python**: Stick to 3.8 or 3.9 for now

### Security

- Regularly update packages for security patches
- Use `pip list --outdated` to check for updates
- Review CHANGELOG.md before major updates
