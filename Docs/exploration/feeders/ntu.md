# NTU-RGB+D Feeder Exploration

This document explores the `feeders.feeder_ntu.Feeder` class, which handles data loading and preprocessing for the NTU-RGB+D dataset.

## Overview

The NTU-RGB+D feeder is a PyTorch Dataset class designed to load and preprocess skeleton action recognition data from the NTU-RGB+D dataset. It supports multiple data modalities (joint, bone, motion), various data augmentation techniques, and flexible data splitting strategies.

## Dataset Information

### NTU-RGB+D Dataset
- **Total Actions**: 60 action classes (NTU60) or 120 action classes (NTU120)
- **Skeleton Structure**: 25 joints per skeleton
- **Persons**: Up to 2 persons per sample
- **Coordinates**: 3D (X, Y, Z)
- **Data Format**: Preprocessed `.npz` files containing numpy arrays

### Data Structure
- **Shape**: `(N, T, 2, 25, 3)` → reshaped to `(N, C, T, V, M)`
  - `N`: Number of samples
  - `C`: Channels/coordinates (3: X, Y, Z)
  - `T`: Time steps/frames
  - `V`: Vertices/joints (25)
  - `M`: Persons (2)

## Class Definition

```python
class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False, 
                 random_rot=False, window_size=-1, normalization=False, 
                 debug=False, use_mmap=False, bone=False, vel=False)
```

## Parameters

### Required Parameters
- **`data_path`** (str): Path to the `.npz` file containing preprocessed NTU-RGB+D data
  - Example: `'data/ntu/NTU60_CS.npz'`
  - The `.npz` file should contain:
    - `x_train`: Training data
    - `y_train`: Training labels (one-hot encoded)
    - `x_test`: Test data
    - `y_test`: Test labels (one-hot encoded)

### Optional Parameters

#### Data Splitting
- **`split`** (str): Dataset split to use
  - Options: `'train'` or `'test'`
  - Default: `'train'`

- **`label_path`** (str): Path to label file (currently not used, can be `None`)
  - Default: `None`

#### Temporal Processing
- **`window_size`** (int): Target number of frames per sample
  - Default: `-1` (uses original length)
  - Common values: `64`, `128`, `300`
  - Used in `valid_crop_resize()` to resize sequences

- **`p_interval`** (float or list): Proportion of valid frames to use
  - Single value (e.g., `0.95`): Center crop with that proportion
  - List (e.g., `[0.5, 1]`): Random crop with proportion in that range
  - Default: `1` (use all valid frames)
  - Used for temporal cropping augmentation

#### Data Augmentation
- **`random_choose`** (bool): Randomly select a portion of the sequence
  - Default: `False`
  - Not currently implemented in `__getitem__`

- **`random_shift`** (bool): Randomly pad zeros at beginning or end
  - Default: `False`
  - Not currently implemented in `__getitem__`

- **`random_move`** (bool): Apply random spatial transformations
  - Default: `False`
  - Not currently implemented in `__getitem__`

- **`random_rot`** (bool): Apply random 3D rotation around XYZ axes
  - Default: `False`
  - When `True`, applies rotation using `tools.random_rot()`
  - Rotation angles: Uniform random in `[-theta, theta]` where `theta=0.3` radians

#### Modalities
- **`bone`** (bool): Convert to bone representation
  - Default: `False`
  - When `True`, computes bone vectors (child joint - parent joint)
  - Uses bone pairs from `bone_pairs.ntu_pairs`

- **`vel`** (bool): Convert to motion/velocity representation
  - Default: `False`
  - When `True`, computes temporal differences between consecutive frames
  - Last frame is set to zero

#### Normalization
- **`normalization`** (bool): Enable data normalization
  - Default: `False`
  - When `True`, computes mean and std maps using `get_mean_map()`
  - Normalization statistics are computed per channel and per joint

#### Other Options
- **`debug`** (bool): Debug mode (use only first 100 samples)
  - Default: `False`

- **`use_mmap`** (bool): Use memory mapping for data loading
  - Default: `False`
  - Can save memory for large datasets

## Key Methods

### `load_data()`
Loads skeleton data from the `.npz` file.

**Process:**
1. Loads `.npz` file using `np.load()`
2. Extracts data based on `split` parameter:
   - `'train'`: Uses `x_train` and `y_train`
   - `'test'`: Uses `x_test` and `y_test`
3. Converts one-hot encoded labels to class indices using `np.where()`
4. Reshapes data from `(N, T, 2, 25, 3)` to `(N, C, T, V, M)`:
   - Original: `(N, T, 2, 25, 3)` - (samples, time, persons, joints, coords)
   - Reshaped: `(N, T, 2, 25, 3)` → `(N, 3, T, 25, 2)` - (samples, coords, time, joints, persons)

**Data Format:**
- Input `.npz` contains flattened skeleton sequences
- Output shape: `(N, C, T, V, M)` where:
  - `C=3` (X, Y, Z coordinates)
  - `V=25` (joints)
  - `M=2` (persons)

### `get_mean_map()`
Computes mean and standard deviation maps for normalization.

**Process:**
1. Computes mean across time and person dimensions
2. Computes std across all samples, time steps, and persons
3. Stores results in `self.mean_map` and `self.std_map`

**Output Shapes:**
- `mean_map`: `(C, 1, V, 1)` - Mean per channel and joint
- `std_map`: `(C, 1, V, 1)` - Std per channel and joint

### `__getitem__(index)`
Retrieves and preprocesses a single data sample.

**Processing Pipeline:**

1. **Load Sample**
   ```python
   data_numpy = self.data[index]  # Shape: (C, T, V, M)
   label = self.label[index]
   ```

2. **Compute Valid Frames**
   - Counts frames with non-zero data
   - Used to determine actual sequence length

3. **Temporal Cropping and Resizing** (`tools.valid_crop_resize()`)
   - Crops valid portion of sequence based on `p_interval`
   - Resizes to `window_size` using bilinear interpolation
   - Handles both upsampling and downsampling

4. **Random Rotation** (if `random_rot=True`)
   - Applies random 3D rotation around XYZ axes
   - Rotation angles: Uniform `[-0.3, 0.3]` radians per axis

5. **Bone Conversion** (if `bone=True`)
   - Computes bone vectors: `bone[joint] = joint - parent_joint`
   - Uses bone pairs from `bone_pairs.ntu_pairs`
   - 25 bone pairs defined for NTU skeleton structure

6. **Motion/Velocity Conversion** (if `vel=True`)
   - Computes temporal differences: `motion[t] = data[t+1] - data[t]`
   - Last frame set to zero

**Returns:**
- `data_numpy`: Preprocessed skeleton data `(C, T, V, M)`
- `label`: Action class label (0-indexed)
- `index`: Original sample index

### `top_k(score, top_k)`
Calculates top-k accuracy for evaluation.

**Parameters:**
- `score`: Prediction scores `(N, num_classes)`
- `top_k`: Number of top predictions to consider

**Returns:**
- Top-k accuracy (fraction of samples with correct label in top-k predictions)

## Helper Functions (`feeders.tools`)

### `valid_crop_resize(data_numpy, valid_frame_num, p_interval, window)`
Performs temporal cropping and resizing.

**Process:**
1. **Cropping:**
   - If `p_interval` is single value: Center crop with that proportion
   - If `p_interval` is list: Random crop with proportion in range
   - Minimum cropped length: 64 frames

2. **Resizing:**
   - Uses PyTorch's bilinear interpolation
   - Reshapes to `(C, window, V, M)`
   - Handles both shorter and longer sequences

### `random_rot(data_numpy, theta=0.3)`
Applies random 3D rotation to skeleton data.

**Process:**
1. Generates random rotation angles for X, Y, Z axes
2. Constructs rotation matrices using Euler angles
3. Applies rotation: `R = Rz @ Ry @ Rx`
4. Rotates all joint coordinates

**Parameters:**
- `theta`: Maximum rotation angle in radians (default: 0.3)

## Bone Structure

The NTU-RGB+D dataset uses a 25-joint skeleton model. Bone pairs are defined in `bone_pairs.ntu_pairs`:

```python
ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)
```

- Format: `(child_joint, parent_joint)`
- Joint indices are 1-indexed (converted to 0-indexed when used)
- Joint 21 is the root/spine joint (appears as parent for many joints)

## Data Modalities

### 1. Joint (Default)
- Raw 3D joint coordinates
- Shape: `(C, T, V, M)` where `C=3` (X, Y, Z)

### 2. Bone
- Vector from parent joint to child joint
- Computed as: `bone[joint] = joint_coords - parent_coords`
- Same shape as joint data
- Provides relative spatial relationships

### 3. Motion/Velocity
- Temporal difference between consecutive frames
- Computed as: `motion[t] = joint[t+1] - joint[t]`
- Last frame set to zero
- Captures movement dynamics

## Usage Example

```python
from feeders.feeder_ntu import Feeder

# Training feeder
train_feeder = Feeder(
    data_path='data/ntu/NTU60_CS.npz',
    split='train',
    window_size=64,
    p_interval=[0.5, 1],  # Random crop 50-100% of valid frames
    random_rot=True,       # Apply random rotation
    bone=False,            # Use joint modality
    vel=False,             # Don't use motion modality
    normalization=False
)

# Test feeder
test_feeder = Feeder(
    data_path='data/ntu/NTU60_CS.npz',
    split='test',
    window_size=64,
    p_interval=[0.95],    # Center crop 95% of valid frames
    random_rot=False,      # No augmentation for test
    bone=False,
    vel=False,
    normalization=False
)

# Get a sample
data, label, index = train_feeder[0]
print(f"Data shape: {data.shape}")  # (3, 64, 25, 2)
print(f"Label: {label}")            # Action class (0-59)
```

## Configuration Example

From `config/nturgbd-cross-subject/default.yaml`:

```yaml
feeder: feeders.feeder_ntu.Feeder
train_feeder_args:
  data_path: data/ntu/NTU60_CS.npz
  split: train
  window_size: 64
  p_interval: [0.5, 1]  # Random crop
  random_rot: True      # Rotation augmentation
  bone: False
  vel: False

test_feeder_args:
  data_path: data/ntu/NTU60_CS.npz
  split: test
  window_size: 64
  p_interval: [0.95]    # Center crop
  random_rot: False     # No augmentation
  bone: False
  vel: False
```

## Differences from UCLA Feeder

1. **Data Format:**
   - NTU: Preprocessed `.npz` files
   - UCLA: Raw JSON files

2. **Skeleton Structure:**
   - NTU: 25 joints, 2 persons
   - UCLA: 20 joints, 1 person

3. **Temporal Processing:**
   - NTU: Uses `valid_crop_resize()` with bilinear interpolation
   - UCLA: Uses random sampling or uniform sampling

4. **Augmentation:**
   - NTU: Random rotation (`random_rot`)
   - UCLA: Random viewpoint transformation (rotation + scaling)

5. **Data Loading:**
   - NTU: Loads entire dataset into memory from `.npz`
   - UCLA: Loads JSON files individually

## Notes

- The feeder expects preprocessed `.npz` files, not raw NTU-RGB+D data
- Data should be preprocessed using the official NTU-RGB+D preprocessing scripts
- The `random_choose`, `random_shift`, and `random_move` parameters are defined but not currently used in `__getitem__`
- Bone and velocity modalities can be combined (computed sequentially)
- The `p_interval` parameter is crucial for temporal augmentation during training
