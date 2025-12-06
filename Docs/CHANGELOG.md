# Changelog - Package Compatibility Updates

This document tracks all changes made to fix package compatibility issues for CTR-GCN.

## 2024 - Package Compatibility Update

### Major Changes

#### Python Version
- **Old**: Python 3.6 (EOL, unsupported)
- **New**: Python 3.8/3.9 (stable, Colab-compatible)
- **Reason**: Python 3.6 reached end-of-life in December 2021

#### PyTorch
- **Old**: torch==1.1.0, torchvision==0.2.1
- **New**: torch>=1.8.0, torchvision>=0.9.0
- **Reason**: 
  - PyTorch 1.1.0 is incompatible with modern CUDA versions
  - Required for Colab compatibility
  - Better performance and bug fixes

#### NumPy
- **Old**: numpy==1.19.4
- **New**: numpy>=1.21.0
- **Reason**: 
  - Security vulnerabilities in older versions
  - Better compatibility with Python 3.8+
  - Required by newer PyTorch versions

#### TensorBoard
- **Old**: tensorboardX==2.1, tensorboard==2.4.0
- **New**: tensorboard>=2.8.0
- **Reason**: 
  - tensorboardX is deprecated
  - tensorboard is the official package
  - Better maintenance and features

#### YAML Loading
- **Old**: `yaml.load(f)` in main.py
- **New**: `yaml.safe_load(f)`
- **Reason**: 
  - `yaml.load()` is unsafe and deprecated
  - `yaml.safe_load()` prevents arbitrary code execution

#### PyTorch Variable
- **Old**: `from torch.autograd import Variable`
- **New**: Removed (tensors are Variables by default in modern PyTorch)
- **Reason**: 
  - Variable is deprecated since PyTorch 1.0
  - All tensors are Variables by default
  - Code updated to use regular tensors

### Package Version Updates

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| Python | 3.6 | 3.8/3.9 | EOL → Stable |
| torch | 1.1.0 | >=1.8.0 | Modern CUDA support |
| torchvision | 0.2.1 | >=0.9.0 | Compatible with new torch |
| numpy | 1.19.4 | >=1.21.0 | Security & compatibility |
| tensorboardX | 2.1 | Removed | Deprecated |
| tensorboard | 2.4.0 | >=2.8.0 | Official package |
| PyYAML | 5.4.1 | >=5.4.1 | Kept same, but usage updated |
| tqdm | 4.53.0 | >=4.60.0 | Minor update |
| scikit-learn | 0.23.2 | >=0.24.0 | Compatibility |
| scipy | 1.5.4 | >=1.7.0 | Compatibility |

### Removed Packages

The following packages were removed from requirements as they are not essential for the core functionality:
- jupyter, ipython, notebook (use Colab's built-in)
- awscli, boto, botocore (not needed for local/Colab)
- Various other packages that were environment-specific

### Code Changes

1. **main.py** (line 564):
   - Changed `yaml.load(f)` to `yaml.safe_load(f)`

2. **model/ctrgcn.py** (line 222):
   - Changed `Variable(torch.from_numpy(...))` to `torch.from_numpy(...)`
   - Removed unused `Variable` import

3. **model/baseline.py** (line 65):
   - Changed `Variable(torch.from_numpy(...))` to `torch.from_numpy(...)`
   - Removed unused `Variable` import

### Testing Status

- [x] Environment creation
- [x] Package installation
- [x] Basic imports
- [ ] Full training run (to be tested)
- [ ] Colab compatibility (to be tested)

### Notes

- The original `requirements.txt` is kept for reference
- New `requirements_updated.txt` contains only essential packages
- `environment.yml` provides conda-based installation option
- All changes are backward-compatible with the model architecture
