# NTU-RGB+D Data Preparation Guide for Google Colab

This guide walks you through preparing NTU-RGB+D data for training in Google Colab when your data is stored in Google Drive.

## Prerequisites

- Google Colab notebook
- NTU-RGB+D dataset zip file in Google Drive
- Access to the CTR-GCN repository

## Your Setup

- **Google Drive Path**: `drive/MyDrive/"Colab Notebooks"/thesis_outline_colabs/CTR-GCN/data`
- **Zip File Location**: `NTU_RAW_DIR = 'drive/MyDrive/"Colab Notebooks"/thesis_outline_colabs/CTR-GCN/data'`

## Step-by-Step Instructions

### Step 1: Mount Google Drive and Set Up Paths

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set up your specific paths
DRIVE_ROOT = '/content/drive/MyDrive'
NTU_RAW_DIR = os.path.join(DRIVE_ROOT, 'Colab Notebooks', 'thesis_outline_colabs', 'CTR-GCN', 'data')
PROJECT_ROOT = os.path.join(DRIVE_ROOT, 'Colab Notebooks', 'thesis_outline_colabs', 'CTR-GCN')

# Create necessary directories
os.makedirs(NTU_RAW_DIR, exist_ok=True)

print(f"NTU Raw Data Directory: {NTU_RAW_DIR}")
print(f"Project Root: {PROJECT_ROOT}")
```

### Step 2: Navigate to Project Directory

```python
# Change to project root
os.chdir(PROJECT_ROOT)
print(f"Current directory: {os.getcwd()}")

# Verify CTR-GCN repository structure
if not os.path.exists('data/ntu'):
    print("⚠️  data/ntu directory not found. Creating...")
    os.makedirs('data/ntu', exist_ok=True)
```

### Step 3: Locate and Extract Zip File

```python
import zipfile
import glob

# Find zip file in your data directory
zip_files = glob.glob(os.path.join(NTU_RAW_DIR, '*.zip'))
print(f"Found zip files: {zip_files}")

# Expected zip file name for NTU60
zip_file = None
for zf in zip_files:
    if 'nturgbd_skeletons_s001_to_s017' in zf or 's001_to_s017' in zf:
        zip_file = zf
        break

if zip_file:
    print(f"✅ Found NTU60 zip file: {zip_file}")
    
    # Extract to nturgbd_raw directory
    extract_dir = os.path.join(NTU_RAW_DIR, 'nturgbd_raw')
    os.makedirs(extract_dir, exist_ok=True)
    
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✅ Extraction complete!")
    
    # Verify extraction
    skeleton_dir = os.path.join(extract_dir, 'nturgb+d_skeletons')
    if os.path.exists(skeleton_dir):
        skeleton_files = glob.glob(os.path.join(skeleton_dir, '*.skeleton'))
        print(f"✅ Found {len(skeleton_files)} skeleton files")
    else:
        print("⚠️  Expected directory 'nturgb+d_skeletons' not found")
        print("   Listing extracted directories:")
        for item in os.listdir(extract_dir):
            print(f"     - {item}")
else:
    print("❌ NTU60 zip file not found!")
    print(f"   Please ensure the zip file is in: {NTU_RAW_DIR}")
    print("   Expected filename contains: 'nturgbd_skeletons_s001_to_s017'")
```

### Step 4: Install Required Dependencies

```python
# Install required packages
!pip install numpy scipy scikit-learn pyyaml tqdm

# Verify installation
import numpy as np
import scipy
import sklearn
print(f"✅ NumPy: {np.__version__}")
print(f"✅ SciPy: {scipy.__version__}")
print(f"✅ scikit-learn: {sklearn.__version__}")
```

### Step 5: Prepare Processing Scripts Directory

```python
# Navigate to data/ntu directory
processing_dir = os.path.join(PROJECT_ROOT, 'data', 'ntu')
os.chdir(processing_dir)
print(f"Current directory: {os.getcwd()}")

# Verify processing scripts exist
required_scripts = [
    'get_raw_skes_data.py',
    'get_raw_denoised_data.py',
    'seq_transformation.py'
]

missing_scripts = []
for script in required_scripts:
    if os.path.exists(script):
        print(f"✅ Found: {script}")
    else:
        print(f"❌ Missing: {script}")
        missing_scripts.append(script)

if missing_scripts:
    print("\n⚠️  Some scripts are missing. Please ensure CTR-GCN repository is complete.")
    print("   You may need to clone the repository:")
    print(f"   !git clone https://github.com/Uason-Chen/CTR-GCN.git {PROJECT_ROOT}")
```

### Step 6: Update Paths in Processing Scripts

The processing scripts need to know where your raw data is located. You may need to modify the paths in the scripts.

**Option A: Modify scripts directly**

```python
# Check current paths in get_raw_skes_data.py
import re

script_path = 'get_raw_skes_data.py'
with open(script_path, 'r') as f:
    content = f.read()
    
# Find the skes_path definition
match = re.search(r"skes_path\s*=\s*['\"]([^'\"]+)['\"]", content)
if match:
    current_path = match.group(1)
    print(f"Current skes_path in script: {current_path}")
    
    # Update to your path
    new_path = os.path.join(NTU_RAW_DIR, 'nturgbd_raw', 'nturgb+d_skeletons')
    if os.path.exists(new_path):
        print(f"✅ Raw skeleton directory exists: {new_path}")
    else:
        print(f"⚠️  Raw skeleton directory not found: {new_path}")
        print("   Please check the extraction step.")
```

**Option B: Set environment variables or modify paths programmatically**

```python
# Update paths in get_raw_skes_data.py
script_path = 'get_raw_skes_data.py'
with open(script_path, 'r') as f:
    content = f.read()

# Replace the path
old_path = "../nturgbd_raw/nturgb+d_skeletons/"
new_path = os.path.join(NTU_RAW_DIR, 'nturgbd_raw', 'nturgb+d_skeletons') + '/'

# Use relative path from data/ntu to the raw data
relative_path = os.path.relpath(
    os.path.join(NTU_RAW_DIR, 'nturgbd_raw', 'nturgb+d_skeletons'),
    processing_dir
) + '/'

content = content.replace(
    "skes_path = '../nturgbd_raw/nturgb+d_skeletons/'",
    f"skes_path = '{relative_path}'"
)

with open(script_path, 'w') as f:
    f.write(content)
    
print(f"✅ Updated skes_path to: {relative_path}")
```

### Step 7: Run Processing Scripts

**Step 7.1: Get Raw Skeleton Data**

```python
print("=" * 70)
print("STEP 1/3: Getting raw skeleton data from .skeleton files")
print("=" * 70)
print("This step reads all .skeleton files and extracts joint positions.")
print("This may take 10-20 minutes...\n")

!python get_raw_skes_data.py

print("\n✅ Step 1 complete!")
print("   Output: raw_data/raw_skes_data.pkl")
```

**Step 7.2: Remove Bad Skeletons (Denoising)**

```python
print("=" * 70)
print("STEP 2/3: Removing bad skeletons (denoising)")
print("=" * 70)
print("This step filters out noisy or invalid skeleton sequences.")
print("This may take 10-15 minutes...\n")

!python get_raw_denoised_data.py

print("\n✅ Step 2 complete!")
print("   Output: denoised_data/raw_denoised_joints.pkl")
```

**Step 7.3: Transform Sequences**

```python
print("=" * 70)
print("STEP 3/3: Transforming sequences")
print("=" * 70)
print("This step:")
print("  - Centers skeletons to first frame")
print("  - Aligns all sequences to same length")
print("  - Splits into train/test sets")
print("  - Creates final .npz files")
print("This may take 10-15 minutes...\n")

!python seq_transformation.py

print("\n✅ Step 3 complete!")
print("   Output: NTU60_CS.npz and NTU60_CV.npz")
```

### Step 8: Verify Processed Files

```python
import numpy as np
import glob

# Check for processed .npz files
npz_files = glob.glob('*.npz')
print(f"Found {len(npz_files)} .npz files:")

for npz_file in npz_files:
    size_mb = os.path.getsize(npz_file) / (1024 * 1024)
    print(f"  📁 {npz_file} ({size_mb:.2f} MB)")
    
    # Load and inspect
    data = np.load(npz_file, allow_pickle=True)
    print(f"     Contents:")
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"       - {key}: shape={arr.shape}, dtype={arr.dtype}")
    print()

# Expected files
expected = ['NTU60_CS.npz', 'NTU60_CV.npz']
missing = [f for f in expected if f not in npz_files]

if missing:
    print(f"⚠️  Missing expected files: {missing}")
else:
    print("✅ All expected files created successfully!")
```

### Step 9: Verify Data Structure

```python
# Load and inspect one file in detail
if 'NTU60_CS.npz' in npz_files:
    data = np.load('NTU60_CS.npz', allow_pickle=True)
    
    print("NTU60_CS.npz Structure:")
    print("=" * 50)
    
    for key in ['x_train', 'y_train', 'x_test', 'y_test']:
        if key in data:
            arr = data[key]
            print(f"\n{key}:")
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            if key.startswith('x_'):
                print(f"  Format: (samples, time, persons, joints, coords)")
                print(f"  Sample shape: {arr[0].shape}")
            else:
                print(f"  Format: (samples, num_classes) - one-hot encoded")
                print(f"  Number of classes: {arr.shape[1]}")
                # Show label distribution
                labels = np.where(arr > 0)[1]
                unique, counts = np.unique(labels, return_counts=True)
                print(f"  Label range: {labels.min()} - {labels.max()}")
                print(f"  Total samples: {len(labels)}")
    
    print("\n✅ Data structure verified!")
else:
    print("⚠️  NTU60_CS.npz not found for verification")
```

### Step 10: Update Config File Paths

```python
import yaml

# Update config file to point to processed data
config_path = os.path.join(PROJECT_ROOT, 'config', 'nturgbd-cross-subject', 'default.yaml')

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get absolute path to processed data
    processed_data_path = os.path.join(processing_dir, 'NTU60_CS.npz')
    
    print("Current config paths:")
    print(f"  train_feeder_args.data_path: {config.get('train_feeder_args', {}).get('data_path', 'N/A')}")
    print(f"  test_feeder_args.data_path: {config.get('test_feeder_args', {}).get('data_path', 'N/A')}")
    
    # Update paths (use relative path from project root)
    relative_data_path = os.path.relpath(processed_data_path, PROJECT_ROOT)
    
    if 'train_feeder_args' not in config:
        config['train_feeder_args'] = {}
    if 'test_feeder_args' not in config:
        config['test_feeder_args'] = {}
    
    config['train_feeder_args']['data_path'] = relative_data_path
    config['test_feeder_args']['data_path'] = relative_data_path
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Config updated!")
    print(f"   New data_path: {relative_data_path}")
else:
    print(f"⚠️  Config file not found: {config_path}")
```

### Step 11: Test Data Loading

```python
# Test loading data with the feeder
import sys
sys.path.insert(0, PROJECT_ROOT)

from feeders.feeder_ntu import Feeder

# Test train feeder
print("Testing train feeder...")
train_feeder = Feeder(
    data_path=os.path.join(processing_dir, 'NTU60_CS.npz'),
    split='train',
    window_size=64,
    p_interval=[0.95],
    random_rot=False,
    bone=False,
    vel=False
)

print(f"✅ Train feeder created!")
print(f"   Dataset size: {len(train_feeder)} samples")

# Test loading a sample
data, label, index = train_feeder[0]
print(f"\nSample 0:")
print(f"   Data shape: {data.shape}")
print(f"   Label: {label}")
print(f"   Data dtype: {data.dtype}")
print(f"   Data range: [{data.min():.2f}, {data.max():.2f}]")

# Test test feeder
print("\nTesting test feeder...")
test_feeder = Feeder(
    data_path=os.path.join(processing_dir, 'NTU60_CS.npz'),
    split='test',
    window_size=64,
    p_interval=[0.95],
    random_rot=False,
    bone=False,
    vel=False
)

print(f"✅ Test feeder created!")
print(f"   Dataset size: {len(test_feeder)} samples")

print("\n✅ Data loading test successful!")
```

## Complete Colab Notebook Cell Sequence

Here's the complete sequence you can copy into Colab:

```python
# ============================================================================
# COMPLETE NTU-RGB+D DATA PREPARATION FOR GOOGLE COLAB
# ============================================================================

from google.colab import drive
import os
import zipfile
import glob
import numpy as np
import yaml
import sys

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Set up paths
DRIVE_ROOT = '/content/drive/MyDrive'
NTU_RAW_DIR = os.path.join(DRIVE_ROOT, 'Colab Notebooks', 'thesis_outline_colabs', 'CTR-GCN', 'data')
PROJECT_ROOT = os.path.join(DRIVE_ROOT, 'Colab Notebooks', 'thesis_outline_colabs', 'CTR-GCN')

os.makedirs(NTU_RAW_DIR, exist_ok=True)
os.chdir(PROJECT_ROOT)

print(f"Project Root: {PROJECT_ROOT}")
print(f"NTU Raw Dir: {NTU_RAW_DIR}")

# Step 3: Find and extract zip file
zip_files = glob.glob(os.path.join(NTU_RAW_DIR, '*.zip'))
zip_file = None
for zf in zip_files:
    if 's001_to_s017' in zf or 'nturgbd_skeletons' in zf.lower():
        zip_file = zf
        break

if zip_file:
    extract_dir = os.path.join(NTU_RAW_DIR, 'nturgbd_raw')
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✅ Extraction complete!")

# Step 4: Install dependencies
!pip install numpy scipy scikit-learn pyyaml tqdm

# Step 5: Navigate to processing directory
processing_dir = os.path.join(PROJECT_ROOT, 'data', 'ntu')
os.makedirs(processing_dir, exist_ok=True)
os.chdir(processing_dir)

# Step 6: Update paths in get_raw_skes_data.py
script_path = 'get_raw_skes_data.py'
if os.path.exists(script_path):
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Calculate relative path
    raw_skeleton_dir = os.path.join(NTU_RAW_DIR, 'nturgbd_raw', 'nturgb+d_skeletons')
    relative_path = os.path.relpath(raw_skeleton_dir, processing_dir) + '/'
    
    # Update path
    content = content.replace(
        "skes_path = '../nturgbd_raw/nturgb+d_skeletons/'",
        f"skes_path = '{relative_path}'"
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    print(f"✅ Updated skes_path to: {relative_path}")

# Step 7: Run processing scripts
print("\n" + "="*70)
print("Running data processing scripts...")
print("="*70)

print("\n[1/3] Getting raw skeleton data...")
!python get_raw_skes_data.py

print("\n[2/3] Removing bad skeletons...")
!python get_raw_denoised_data.py

print("\n[3/3] Transforming sequences...")
!python seq_transformation.py

# Step 8: Verify output
npz_files = glob.glob('*.npz')
print(f"\n✅ Created {len(npz_files)} .npz files:")
for f in npz_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"   - {f} ({size_mb:.2f} MB)")

print("\n✅ Data preparation complete!")
print(f"   Processed files location: {processing_dir}")
```

## Troubleshooting

### Issue: "Skeleton file not found"
- **Solution**: Check that the zip file was extracted correctly
- Verify path: `{NTU_RAW_DIR}/nturgbd_raw/nturgb+d_skeletons/`
- Ensure `.skeleton` files exist in that directory

### Issue: "statistics directory not found"
- **Solution**: The processing scripts need a `statistics` directory with metadata files
- These should be included in the CTR-GCN repository
- Check: `data/ntu/statistics/` directory exists

### Issue: "Memory error during processing"
- **Solution**: 
  - Process in smaller batches (modify scripts)
  - Use Colab Pro for more RAM
  - Process locally and upload final `.npz` files

### Issue: "Config file path errors"
- **Solution**: Use relative paths from project root
- Or update config files to use absolute paths

## Expected Output Files

After successful processing, you should have:

1. **`NTU60_CS.npz`** - Cross-Subject split
   - `x_train`: Training data (N, T, 2, 25, 3)
   - `y_train`: Training labels (one-hot encoded)
   - `x_test`: Test data
   - `y_test`: Test labels

2. **`NTU60_CV.npz`** - Cross-View split
   - Same structure as CS split

3. **Intermediate files** (can be deleted to save space):
   - `raw_data/raw_skes_data.pkl`
   - `denoised_data/raw_denoised_joints.pkl`
   - Various log files

## File Sizes (Approximate)

- Raw zip file: ~5-7 GB
- Extracted raw data: ~5-7 GB
- Processed `.npz` files: ~500 MB - 2 GB each
- **Total space needed**: ~7-10 GB

## Next Steps

Once data preparation is complete:

1. **Verify data loading**:
   ```python
   from feeders.feeder_ntu import Feeder
   feeder = Feeder('data/ntu/NTU60_CS.npz', split='train')
   ```

2. **Start training**:
   ```bash
   python main.py --config config/nturgbd-cross-subject/default.yaml --device 0
   ```

3. **Monitor progress**: Check logs and tensorboard outputs

## Notes

- Processing time: ~30-60 minutes total
- The processed `.npz` files are much smaller and faster to load than raw data
- You can delete intermediate files (`raw_data/`, `denoised_data/`) after processing to save space
- Keep the `.npz` files - these are what you need for training
