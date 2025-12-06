#!/usr/bin/env python
"""
Check GPU availability and setup for CTR-GCN
Usage: python scripts/check_gpu.py
"""

import sys

def check_gpu():
    """Check if GPU is available and print system information"""
    print("=" * 60)
    print("CTR-GCN GPU Environment Check")
    print("=" * 60)
    
    # Check PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("\n✗ PyTorch not installed!")
        print("  Install with: pip install torch torchvision")
        return False
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("\n✗ No GPU detected!")
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU installed")
        print("  2. CUDA not installed")
        print("  3. PyTorch installed without CUDA support")
        print("\nTo install PyTorch with CUDA:")
        print("  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("\nOr use CPU (slower):")
        print("  Training will use CPU if GPU not available")
    
    # Check other dependencies
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    dependencies = {
        'numpy': 'numpy',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn',
    }
    
    all_ok = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            all_ok = False
    
    # Check tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✓ TensorBoard (via PyTorch) available")
    except ImportError:
        try:
            import tensorboard
            print("✓ TensorBoard installed")
        except ImportError:
            print("✗ TensorBoard not installed")
            print("  Install with: pip install tensorboard")
            all_ok = False
    
    # Check torchlight
    try:
        from torchlight import DictAction
        print("✓ torchlight package installed")
    except ImportError:
        print("✗ torchlight package not installed")
        print("  Install with: pip install -e torchlight")
        all_ok = False
    
    print("\n" + "=" * 60)
    if cuda_available and all_ok:
        print("✓ System ready for GPU training!")
        print("\nYou can start training with:")
        print("  python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/ctrgcn_joint --device 0")
    elif all_ok:
        print("⚠ System ready but will use CPU (slower)")
        print("\nYou can start training with:")
        print("  python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/ctrgcn_joint --device 0")
    else:
        print("✗ Some dependencies missing. Please install them first.")
    print("=" * 60)
    
    return cuda_available and all_ok

if __name__ == '__main__':
    success = check_gpu()
    sys.exit(0 if success else 1)
