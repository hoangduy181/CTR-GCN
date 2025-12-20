# NTU60 2D Data Exploration Results

## File Information
- **Location**: `data/ntu2d/ntu60_2d.pkl`
- **Size**: 582 MB
- **Status**: ⚠️ **File appears to be corrupted/truncated**

## Attempted Loading Methods
All methods failed with `UnpicklingError: pickle data was truncated`:
1. Standard `pickle.load()`
2. `pickle.load()` with `encoding='latin1'`
3. `pickle.load()` with `encoding='bytes'`
4. `joblib.load()` (if available)

## File Structure Clues (from hex dump)
From the first 100 bytes, we can see:
- Pickle protocol 2 (`\x80\x02`)
- Dictionary structure (`}q`)
- Keys visible in hex dump:
  - `split` - likely a dictionary key
  - `xsub_train` - likely cross-subject train data
  - `S0` - possibly sample identifiers

This suggests the structure might be:
```python
{
    'split': {
        'xsub_train': [...],
        'xsub_test': [...],
        # possibly other splits
    },
    # possibly other top-level keys
}
```

## Expected Structure (if 17-joint 2D data)

If this is 17-joint 2D skeleton data, we would expect:

### Data Format
- **Joints**: 17 (instead of 25 for NTU60)
- **Coordinates**: 2D (X, Y) instead of 3D (X, Y, Z)
- **Shape**: Possibly:
  - `(N, T, M, 17, 2)` - samples, time, persons, joints, coords
  - `(N, T, 17, 2)` - if single person
  - Or flattened: `(N, T, 17*2)` = `(N, T, 34)`

### Comparison with NTU60 3D
- **NTU60 3D**: `(N, T, 2, 25, 3)` → reshaped to `(N, 3, T, 25, 2)`
- **NTU60 2D (expected)**: `(N, T, 2, 17, 2)` → reshaped to `(N, 2, T, 17, 2)`

## Compatibility with CTR-GCN System

### ✅ What Will Work
1. **Model Architecture**: The model uses `num_point` parameter, so it can handle 17 joints
2. **Main Script**: Fully configurable via YAML, no hardcoded values
3. **Graph Structure**: Can create new graph file for 17 joints

### ⚠️ What Needs to be Done
1. **Fix/Re-download the pickle file** - File is corrupted
2. **Create Graph File**: `graph/joint17.py` with 17-joint skeleton structure
3. **Create Feeder**: `feeders/feeder_ntu_2d.py` (currently empty) to handle:
   - 2D coordinates (2 channels instead of 3)
   - 17 joints instead of 25
   - Data loading from pickle file
4. **Create Config**: `config/ntu2d/default.yaml` with:
   - `num_point: 17`
   - `graph: graph.joint17.Graph`
   - `feeder: feeders.feeder_ntu_2d.Feeder`
   - Adjust `in_channels: 2` (for 2D coordinates)

## Next Steps

### Immediate Actions
1. **Fix the pickle file**:
   - Check if download was complete
   - Re-download if necessary
   - Or re-run data preparation script if available

2. **Once file is fixed, verify**:
   ```python
   import pickle
   data = pickle.load(open('data/ntu2d/ntu60_2d.pkl', 'rb'))
   
   # Check structure
   print(type(data))
   if isinstance(data, dict):
       print("Keys:", list(data.keys()))
       # Check data shapes
       for key in data.keys():
           if hasattr(data[key], 'shape'):
               print(f"{key} shape: {data[key].shape}")
   ```

3. **Determine joint count**:
   - Load a sample and check the joint dimension
   - Verify if it's 17 joints

4. **Determine coordinate dimension**:
   - Check if coordinates are 2D (shape includes 2) or 3D (shape includes 3)

### Implementation Steps (after file is fixed)
1. Create `graph/joint17.py` with 17-joint skeleton structure
2. Create `feeders/feeder_ntu_2d.py` to load 2D data
3. Create `config/ntu2d/default.yaml` configuration
4. Test data loading and model initialization

## Questions to Answer

1. **Is the file actually corrupted?**
   - Can it be re-downloaded?
   - Is there a data preparation script that creates it?

2. **What is the exact data structure?**
   - Dictionary format?
   - Array format?
   - How are train/test splits organized?

3. **How many joints?**
   - Is it actually 17 joints?
   - Or a different number?

4. **What are the coordinate dimensions?**
   - 2D (X, Y)?
   - Or still 3D but with Z=0?

5. **What is the data format?**
   - Similar to NTU60 NPZ structure?
   - Or different format?

## Files Created for Exploration

1. `explore_ntu2d.py` - Python script to explore the file
2. `explore_ntu2d_data.ipynb` - Jupyter notebook for interactive exploration

## Recommendations

1. **First Priority**: Fix the corrupted pickle file
2. **Second Priority**: Once file is accessible, run exploration to understand structure
3. **Third Priority**: Implement graph, feeder, and config files based on findings
