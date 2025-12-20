# NTU60 2D Data Analysis - Complete Findings

## ✅ Data Successfully Loaded!

The pickle file `data/ntu2d/ntu60_2d.pkl` is **NOT corrupted** - it loads successfully with standard pickle.

## Data Structure

### Top-Level Structure
```python
{
    'split': {
        'xsub_train': [list of sample IDs],
        'xsub_val': [list of sample IDs],
        'xview_train': [list of sample IDs],
        'xview_val': [list of sample IDs]
    },
    'annotations': [list of annotation dicts]
}
```

### Sample Statistics
- **Total samples**: 56,578
- **Cross-Subject Split**:
  - Train: 40,091 samples
  - Val: 16,487 samples
  - Ratio: ~70.9% train / 29.1% val
- **Cross-View Split**: Also available (xview_train, xview_val)

### Annotation Structure
Each annotation is a dictionary with:
```python
{
    'frame_dir': str,           # Sample ID (e.g., 'S001C001P001R001A001')
    'label': int,                # Action class (0-indexed, 0-59)
    'img_shape': tuple,          # Image dimensions (1080, 1920)
    'original_shape': tuple,     # Original image dimensions
    'total_frames': int,         # Number of frames (variable)
    'keypoint': np.array,        # Shape: (1, T, 17, 2) - SKELETON DATA
    'keypoint_score': np.array   # Shape: (1, T, 17) - Confidence scores
}
```

## ✅ Key Confirmations

### 1. **17 Joints Confirmed** ✅
- `keypoint` shape: `(1, T, 17, 2)`
- **V = 17 joints** per skeleton

### 2. **2D Coordinates Confirmed** ✅
- Last dimension is 2 (X, Y coordinates)
- Values are pixel coordinates (e.g., X: 1023-1066, Y: 313-707)
- **C = 2** (not 3D)

### 3. **Single Person Per Sample** ✅
- First dimension is always 1
- **M = 1** person per sample

### 4. **Variable Frame Lengths** ✅
- Frame count varies per sample
- Example: 103 frames in first sample
- Need temporal cropping/resizing in feeder

### 5. **60 Action Classes** ✅
- Labels are 0-indexed (0-59)
- Total of 60 action classes (NTU60)

## Data Format Conversion

### Current Format
```
keypoint: (M=1, T, V=17, C=2)
```

### CTR-GCN Expected Format
```
(C=2, T, V=17, M=1)
```

### Conversion Steps
1. Remove M dimension: `(1, T, 17, 2)` → `(T, 17, 2)`
2. Transpose: `(T, 17, 2)` → `(2, T, 17)`
3. Add M dimension: `(2, T, 17)` → `(2, T, 17, 1)`

**Result**: `(C=2, T, V=17, M=1)` ✅

## Compatibility Assessment

### ✅ Fully Compatible Components

1. **Model Architecture** (`model/ctrgcn.py`)
   - Uses `num_point` parameter → can handle 17 joints
   - Uses `in_channels` parameter → can handle 2 channels
   - BatchNorm adapts: `num_person * in_channels * num_point` = `1 * 2 * 17`

2. **Main Script** (`main.py`)
   - Fully configurable via YAML
   - No hardcoded values

3. **Graph Tools** (`graph/tools.py`)
   - Generic graph construction
   - Works with any number of nodes

### ⚠️ Components That Need Implementation

1. **Graph Definition** (`graph/joint17.py`)
   - Need to define 17-joint skeleton structure
   - Determine bone connectivity
   - Create adjacency matrix

2. **Feeder** (`feeders/feeder_ntu_2d.py`)
   - Load pickle file
   - Map split samples to annotations
   - Convert format: `(1, T, 17, 2)` → `(2, T, 17, 1)`
   - Handle variable frame lengths with `valid_crop_resize`
   - Return: `(C=2, T, V=17, M=1)`

3. **Configuration** (`config/ntu2d/default.yaml`)
   - Set `num_point: 17`
   - Set `num_person: 1`
   - Set `in_channels: 2` (or modify model to accept this)
   - Set `graph: graph.joint17.Graph`
   - Set `feeder: feeders.feeder_ntu_2d.Feeder`

## Implementation Plan

### Phase 1: Graph Structure (17 Joints)

**Question**: What is the 17-joint skeleton structure?

Common 17-joint formats:
1. **COCO format** (17 joints):
   - 0: Nose, 1-2: Eyes, 3-4: Ears
   - 5-6: Shoulders, 7-8: Elbows, 9-10: Wrists
   - 11-12: Hips, 13-14: Knees, 15-16: Ankles

2. **MPII format** (16 joints, but might be extended to 17)

3. **Custom format** specific to NTU60 2D

**Action**: Need to determine which 17 joints from NTU60's 25 joints are kept, or if it's a different format entirely.

### Phase 2: Feeder Implementation

Create `feeders/feeder_ntu_2d.py`:

```python
class Feeder(Dataset):
    def __init__(self, data_path, split='train', window_size=64, ...):
        # Load pickle
        # Map split samples to annotations
        # Store annotations
        
    def __getitem__(self, index):
        # Get annotation
        # Extract keypoint: (1, T, 17, 2)
        # Convert to (2, T, 17, 1)
        # Handle temporal cropping/resizing
        # Return (C, T, V, M) and label
```

### Phase 3: Configuration

Create `config/ntu2d/default.yaml`:

```yaml
model_args:
  num_class: 60
  num_point: 17
  num_person: 1
  in_channels: 2  # For 2D coordinates
  graph: graph.joint17.Graph
```

### Phase 4: Model Modification (if needed)

Check if model needs modification for `in_channels=2`:
- Current default: `in_channels=3` (for 3D)
- Need to verify if model accepts `in_channels` parameter

## Next Steps

1. **Determine 17-joint skeleton structure**
   - Check if it's COCO format or custom
   - Map to bone connectivity
   - Create `graph/joint17.py`

2. **Implement feeder**
   - Create `feeders/feeder_ntu_2d.py`
   - Handle data loading and format conversion
   - Test with sample data

3. **Create configuration**
   - Create `config/ntu2d/default.yaml`
   - Set all required parameters

4. **Test end-to-end**
   - Load data through feeder
   - Initialize model
   - Run forward pass
   - Verify shapes match

## Summary

✅ **Data is ready and compatible!**
- 17 joints confirmed
- 2D coordinates confirmed
- Format can be converted to CTR-GCN format
- System architecture supports it

⚠️ **Need to implement**:
- Graph structure for 17 joints
- Feeder for 2D data
- Configuration file

The data structure is well-organized and can definitely fit into the CTR-GCN system!
