# UCLA Feeder Exploration

This document explores the `feeders.feeder_ucla.Feeder` class, which handles data loading and preprocessing for the NW-UCLA dataset.

## Overview

The UCLA Feeder is a PyTorch `Dataset` class that:
- Loads skeleton data from JSON files in the NW-UCLA dataset
- Applies data augmentation for training (viewpoint transformation, normalization)
- Supports multiple modalities: joint, bone, and motion
- Handles train/validation splits
- Returns data in the format expected by CTR-GCN models

## Dataset Information

### NW-UCLA Dataset
- **Number of Actions**: 10 action classes
- **Number of Joints**: 20 joints per skeleton
- **Number of Persons**: 1 person per sample
- **Data Format**: JSON files containing skeleton sequences
- **Data Location**: `data/NW-UCLA/all_sqe/`

### File Naming Convention
Files follow the pattern: `a{action}_s{subject}_e{environment}_v{view}.json`
- `a{action}`: Action class (1-10)
- `s{subject}`: Subject ID
- `e{environment}`: Environment/execution ID
- `v{view}`: Camera view (01, 02, 03)

## Class Structure

### Initialization (`__init__`)

**Parameters:**
- `data_path` (str): Modality type - `'joint'`, `'bone'`, `'motion'`, or combinations
- `label_path` (str): Split indicator - `'train'` or `'val'` (validation)
- `repeat` (int): Number of times to repeat the dataset (default: 1)
- `random_choose` (bool): Whether to randomly choose frames (not used in current implementation)
- `random_shift` (bool): Whether to apply random temporal shift (not used)
- `random_move` (bool): Whether to apply random spatial movement (not used)
- `window_size` (int): Number of frames per sample (default: -1, uses `time_steps=52`)
- `normalization` (bool): Whether to normalize data (not fully implemented)
- `debug` (bool): Debug mode flag
- `use_mmap` (bool): Whether to use memory mapping (not used)

**Key Initialization Steps:**

1. **Split Detection**: Determines train/val split based on `label_path`
   - If `'val'` in `label_path` → validation set
   - Otherwise → training set

2. **Data Dictionary**: Hardcoded list of samples with:
   - `file_name`: JSON filename (without extension)
   - `length`: Number of frames in the sequence
   - `label`: Action class label (1-10)

3. **Bone Structure**: Defines skeleton bone connections as (parent, child) pairs
   - 20 joints with parent-child relationships
   - Used for bone modality computation

4. **Label Extraction**: Converts labels from 1-indexed to 0-indexed

5. **Data Loading**: Calls `load_data()` to load all JSON files

## Data Loading (`load_data`)

**Process:**
1. Iterates through `data_dict` entries
2. For each entry, loads corresponding JSON file from `data/NW-UCLA/all_sqe/`
3. Extracts `skeletons` array from JSON
4. Converts to NumPy array and stores in `self.data`

**Data Shape**: `(N, T, V, C)` where:
- `N`: Number of samples
- `T`: Temporal frames (varies per sample)
- `V`: Vertices/joints (20)
- `C`: Coordinates (3D: x, y, z)

## Data Processing (`__getitem__`)

This is the core method that processes a single sample when accessed by DataLoader.

### Training Mode (`train_val == 'train'`)

**Step 1: Random View Transformation**
- Generates random rotation angles: `agx`, `agy` ∈ [-60°, 60°]
- Generates random scale: `s` ∈ [0.5, 1.5]
- Applies rotation and scaling using `rand_view_transform()`

**Step 2: Centering**
- Centers skeleton around joint 1 (index 1) of first frame
- Subtracts center coordinates from all joints

**Step 3: Normalization**
- Reshapes to `(-1, 3)` (all joints, all frames, 3D coords)
- Min-max normalization per coordinate dimension
- Scales to [-1, 1] range: `(value - min) / (max - min) * 2 - 1`
- Reshapes back to `(T, V, 3)`

**Step 4: Temporal Sampling**
- Creates fixed-size array: `(time_steps=52, 20, 3)`
- Randomly samples frames with replacement (repeats sequence 100x)
- Sorts sampled indices to maintain temporal order

### Validation Mode (`train_val == 'val'`)

**Similar to training but:**
- No random transformations (`agx=0`, `agy=0`, `s=1.0`)
- Uniform temporal sampling using `np.linspace()` instead of random sampling
- Ensures consistent, reproducible evaluation

### Modality Processing

**Bone Modality** (if `'bone'` in `data_path`):
- Computes bone vectors: `bone_vector = joint - parent_joint`
- Uses predefined bone structure from `self.bone`
- Replaces joint coordinates with bone vectors

**Motion Modality** (if `'motion'` in `data_path`):
- Computes velocity: `motion[t] = joint[t+1] - joint[t]`
- Represents temporal dynamics
- Last frame is zero-padded

**Combined Modalities**: Can use `'bone motion'` for both

### Final Formatting

1. **Transpose**: `(T, V, C)` → `(C, T, V)`
   - Channels (x, y, z) become first dimension
   - Expected by CTR-GCN model

2. **Add Person Dimension**: `(C, T, V)` → `(C, T, V, 1)`
   - Adds singleton dimension for person (M=1 for UCLA)
   - Final shape: `(3, 52, 20, 1)`

**Returns**: `(data, label, index)`
- `data`: Processed skeleton data `(C, T, V, M)`
- `label`: Action class label (0-indexed)
- `index`: Sample index

## Helper Methods

### `rand_view_transform(X, agx, agy, s)`
Applies 3D rotation and scaling transformation:
- **Rotation**: Around X and Y axes using rotation matrices
- **Scale**: Uniform scaling factor
- **Purpose**: Data augmentation for viewpoint invariance

### `get_mean_map()` (if normalization=True)
Computes mean and standard deviation for normalization:
- Mean: Average across time, persons, and samples
- Std: Standard deviation per channel and joint
- **Note**: Currently not fully utilized in `__getitem__`

### `top_k(score, top_k)`
Computes Top-K accuracy:
- `score`: Model predictions `(N, num_classes)`
- Returns accuracy where true label is in top-K predictions

## Data Flow Summary

```
JSON File → Load Skeletons → View Transform → Center → Normalize → Sample Frames
                                                                    ↓
                                                              Modality (bone/motion)
                                                                    ↓
                                                              Transpose → Add Person Dim
                                                                    ↓
                                                              (C, T, V, M)
```

## Usage Example

```python
from feeders.feeder_ucla import Feeder

# Training feeder
train_feeder = Feeder(
    data_path='joint',
    label_path='train',
    repeat=5,
    window_size=52
)

# Validation feeder
val_feeder = Feeder(
    data_path='joint',
    label_path='val',
    repeat=1,
    window_size=52
)

# Get a sample
data, label, index = train_feeder[0]
print(f"Data shape: {data.shape}")  # (3, 52, 20, 1)
print(f"Label: {label}")  # 0-9
```

## Key Differences from NTU Feeder

1. **Data Format**: JSON files vs. NPZ files
2. **Fixed Split**: Hardcoded train/val split vs. configurable
3. **View Transformation**: Built-in random view augmentation
4. **Temporal Sampling**: Random with replacement vs. uniform sampling
5. **Modality Handling**: String-based (`'bone'`, `'motion'`) vs. boolean flags
6. **Dataset Size**: Smaller (10 classes, ~200 samples) vs. NTU (60/120 classes, thousands)

## Configuration

From `config/ucla/default.yaml`:
```yaml
train_feeder_args:
  data_path: joint          # Modality: joint, bone, motion, or combinations
  label_path: train         # Split: train or val
  repeat: 5                 # Repeat dataset 5x for training
  window_size: 52           # Fixed temporal length
  normalization: False      # Not fully implemented

test_feeder_args:
  data_path: joint
  label_path: val
```

## Notes

- The hardcoded `data_dict` contains all samples for train/val splits
- `repeat` parameter allows dataset augmentation by repeating samples
- View transformation helps with viewpoint invariance
- Bone and motion modalities provide additional features beyond joint positions
- The feeder is designed specifically for NW-UCLA dataset structure
