# Troubleshooting Guide

Common issues and solutions when setting up and running CTR-GCN.

## Installation Issues

### Issue: PyTorch CUDA version mismatch

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution:**
1. Check your CUDA version: `nvidia-smi`
2. Install matching PyTorch version:
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

### Issue: Package conflicts during installation

**Symptoms:**
```
ERROR: Cannot install package X because it conflicts with package Y
```

**Solution:**
1. Create a fresh conda environment:
   ```bash
   conda create -n ctrgcn_new python=3.9 -y
   conda activate ctrgcn_new
   ```
2. Install PyTorch first, then other packages
3. Use `pip install --no-deps` for problematic packages if needed

### Issue: NumPy version conflicts

**Symptoms:**
```
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
pip install --upgrade numpy
# Or reinstall
pip uninstall numpy
pip install numpy>=1.21.0
```

## Runtime Issues

### Issue: YAML loading error

**Symptoms:**
```
yaml.constructor.ConstructorError: could not determine a constructor
```

**Solution:**
This should be fixed in the updated code. If you still see this:
1. Ensure you're using the updated `main.py` with `yaml.safe_load()`
2. Check your PyYAML version: `pip install --upgrade PyYAML`

### Issue: Variable is deprecated

**Symptoms:**
```
UserWarning: Variable is deprecated
```

**Solution:**
This is fixed in the updated code. If you see this:
1. Ensure you're using updated `model/ctrgcn.py` and `model/baseline.py`
2. The Variable import has been removed and replaced with regular tensors

### Issue: CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size in config file or command line:
   ```bash
   python main.py --config config/.../default.yaml --batch-size 32
   ```
2. Use CPU (slower):
   ```bash
   # Set device to CPU in config or use --device -1
   ```
3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue: Import errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'torchlight'
```

**Solution:**
```bash
cd /path/to/CTR-GCN
pip install -e torchlight
```

**Symptoms:**
```
ModuleNotFoundError: No module named 'feeders'
```

**Solution:**
Ensure you're running from the CTR-GCN root directory:
```bash
cd /path/to/CTR-GCN
python main.py ...
```

## Colab-Specific Issues

### Issue: Package installation fails in Colab

**Solution:**
1. Restart runtime after installing packages
2. Install packages in this order:
   ```python
   !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   !pip install -r requirements_updated.txt
   !pip install -e torchlight
   ```

### Issue: CUDA not available in Colab

**Solution:**
1. Go to Runtime → Change runtime type
2. Select GPU as hardware accelerator
3. Restart runtime

### Issue: File path issues in Colab

**Solution:**
Use absolute paths or mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
# Then use /content/drive/MyDrive/... paths
```

## Data Issues

### Issue: Dataset not found

**Symptoms:**
```
FileNotFoundError: data/ntu/NTU60_CS.npz
```

**Solution:**
1. Follow data preparation steps in README.md
2. Ensure data files are in correct directories
3. Check config file paths match your data location

### Issue: Data loading errors

**Symptoms:**
```
KeyError: 'x_train'
```

**Solution:**
1. Regenerate data files using the data processing scripts
2. Check data file format matches expected structure

## Performance Issues

### Issue: Training is very slow

**Solutions:**
1. Ensure CUDA is being used: `torch.cuda.is_available()` should return `True`
2. Increase `num_worker` in config (if you have CPU cores available)
3. Use mixed precision training (if supported by your PyTorch version)
4. Reduce data augmentation if not needed

### Issue: Memory usage is high

**Solutions:**
1. Reduce batch size
2. Reduce `window_size` in config
3. Use gradient checkpointing (if implemented)
4. Process data in smaller chunks

## Getting Help

If you encounter issues not listed here:

1. Check the [CHANGELOG.md](CHANGELOG.md) for recent changes
2. Review the [ENVIRONMENT.md](ENVIRONMENT.md) for environment details
3. Check GitHub issues (if available)
4. Verify you're using the updated code and packages

## Verification Checklist

Before reporting issues, verify:

- [ ] Conda environment is activated
- [ ] Python version is 3.8 or 3.9
- [ ] PyTorch is installed and CUDA is available
- [ ] All packages from requirements_updated.txt are installed
- [ ] torchlight is installed (`pip install -e torchlight`)
- [ ] Running from CTR-GCN root directory
- [ ] Data files are prepared and in correct locations
- [ ] Config file paths are correct
