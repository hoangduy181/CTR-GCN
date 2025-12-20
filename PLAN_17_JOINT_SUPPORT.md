# Plan: Supporting 17-Joint Skeleton Data in CTR-GCN

## Overview
This document outlines the plan to enable CTR-GCN to work with 17-joint skeleton data. Currently, the codebase supports:
- **NTU60**: 25 joints (`graph/ntu_rgb_d.py`)
- **NW-UCLA**: 20 joints (`graph/ucla.py`)

## Analysis of Current Codebase

### 1. Key Components That Need Modification

#### A. Graph Definition (`graph/` directory)
- **Current**: Each dataset has its own graph file defining skeleton structure
- **Files**: 
  - `graph/ntu_rgb_d.py` (25 joints)
  - `graph/ucla.py` (20 joints)
- **What it defines**:
  - `num_node`: Number of joints
  - `self_link`: Self-connections for each joint
  - `inward_ori_index`: Bone connections (child, parent) in 1-indexed format
  - `inward`: Converted to 0-indexed
  - `outward`: Reverse connections
  - `neighbor`: Combined inward + outward
  - `Graph` class that builds adjacency matrix

#### B. Model Configuration (`model/ctrgcn.py`)
- **Current**: Uses `num_point` parameter (passed via config)
- **Status**: ✅ **No changes needed** - Model is flexible and uses `num_point` parameter
- **Key line**: `self.num_point = num_point` (line 286)
- **Data BN**: `nn.BatchNorm1d(num_person * in_channels * num_point)` - automatically adapts

#### C. Configuration Files (`config/` directory)
- **Current**: YAML files specify `model_args.num_point`
- **Examples**:
  - `config/nturgbd-cross-subject/default.yaml`: `num_point: 25`
  - `config/ucla/default.yaml`: `num_point: 20`
- **Status**: ✅ **Need to create new config** for 17-joint dataset

#### D. Feeder (`feeders/` directory)
- **Current**: Dataset-specific feeders handle data loading
- **Files**: 
  - `feeders/feeder_ntu.py` (for NTU60)
  - `feeders/feeder_ucla.py` (for NW-UCLA)
- **Status**: ⚠️ **May need modification** depending on data format
  - If 17-joint data uses same format as existing datasets → reuse feeder
  - If different format → create new feeder or modify existing

#### E. Main Script (`main.py`)
- **Current**: No hardcoded joint numbers
- **Status**: ✅ **No changes needed** - Fully configurable via YAML

### 2. Data Flow Analysis

```
Config YAML
  ↓
  model_args.num_point = 17
  ↓
Model.__init__(num_point=17)
  ↓
Graph.__init__() → builds adjacency matrix (17x17)
  ↓
Feeder loads data → shape (N, C, T, V=17, M)
  ↓
Model processes → data_bn adapts to 17 joints
```

## Implementation Plan

### Phase 1: Define 17-Joint Skeleton Structure

**Task 1.1**: Create `graph/joint17.py`
- Define skeleton structure for 17 joints
- Need to determine:
  - Which joints are included (subset of NTU25 or UCLA20?)
  - Bone connectivity structure
  - Joint indexing scheme

**Questions to Answer**:
- What is the source of 17-joint data? (Which dataset?)
- What is the joint mapping? (Which joints from NTU25/UCLA20 are kept?)
- What is the bone connectivity structure?

**Example Structure** (to be confirmed):
```python
# graph/joint17.py
num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    # Define 17-joint skeleton structure
    # Format: (child_joint, parent_joint) in 1-indexed
    # Example (needs verification):
    (1, 2), (2, 3), (3, 3),  # Head/neck
    (4, 3), (5, 3),          # Shoulders
    (6, 5), (7, 6),          # Right arm
    (8, 4), (9, 8),          # Left arm
    (10, 3), (11, 10),       # Spine
    (12, 11), (13, 12),      # Right leg
    (14, 11), (15, 14),      # Left leg
    (16, 13), (17, 15)       # Feet
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
    
    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
```

### Phase 2: Create Configuration File

**Task 2.1**: Create `config/joint17/default.yaml`
- Copy from existing config (e.g., `config/ucla/default.yaml`)
- Modify:
  - `model_args.num_point: 17`
  - `model_args.graph: graph.joint17.Graph`
  - Update feeder args if needed
  - Update `num_class` based on dataset

**Example**:
```yaml
work_dir: ./work_dir/joint17/ctrgcn_joint

# feeder
feeder: feeders.feeder_ucla.Feeder  # or new feeder
train_feeder_args:
  data_path: joint  # or path to 17-joint data
  label_path: train
  debug: False
  window_size: 52
  normalization: False

test_feeder_args:
  data_path: joint
  label_path: val
  debug: False

# model
model: model.ctrgcn.Model
model_args:
  num_class: 10  # Adjust based on dataset
  num_point: 17  # ← KEY CHANGE
  num_person: 1  # Adjust if needed
  graph: graph.joint17.Graph  # ← KEY CHANGE
  graph_args:
    labeling_mode: 'spatial'

# optim
weight_decay: 0.0001
base_lr: 0.1
lr_decay_rate: 0.1
step: [50]

# training
device: 0
batch_size: 16
test_batch_size: 64
num_epoch: 65
nesterov: True
warm_up_epoch: 5
```

### Phase 3: Data Feeder (If Needed)

**Task 3.1**: Determine if new feeder is needed
- **Option A**: Reuse existing feeder if data format matches
  - If 17-joint data uses same JSON/NPZ format → reuse `feeder_ucla.py` or `feeder_ntu.py`
  - Just ensure data has 17 joints per skeleton
  
- **Option B**: Create new feeder if format differs
  - Create `feeders/feeder_joint17.py`
  - Handle data loading, preprocessing, augmentation
  - Ensure output shape: `(C, T, V=17, M)`

**Checklist**:
- [ ] Verify data format (JSON, NPZ, other?)
- [ ] Check data shape: `(N, T, V, C)` where V=17?
- [ ] Verify label format
- [ ] Check if preprocessing needed (centering, normalization)

### Phase 4: Testing & Validation

**Task 4.1**: Create test script
- Verify graph structure loads correctly
- Verify model initializes with `num_point=17`
- Verify data loader produces correct shapes
- Test forward pass

**Task 4.2**: Validation checks
- [ ] Graph adjacency matrix shape: (3, 17, 17)
- [ ] Model `data_bn` input size: `num_person * 3 * 17`
- [ ] Data shape from feeder: `(N, C=3, T, V=17, M)`
- [ ] Model forward pass works without errors

## Implementation Steps (Summary)

1. **Research**: Determine 17-joint skeleton structure
   - Which joints? (mapping from NTU25 or UCLA20)
   - Bone connectivity?
   - Data format?

2. **Create Graph File**: `graph/joint17.py`
   - Define `num_node = 17`
   - Define bone connections
   - Implement `Graph` class

3. **Create Config File**: `config/joint17/default.yaml`
   - Set `num_point: 17`
   - Set `graph: graph.joint17.Graph`
   - Configure feeder args

4. **Handle Feeder**:
   - If data format matches existing → reuse feeder
   - If different → create/modify feeder

5. **Test**:
   - Load config
   - Initialize model
   - Load data
   - Run forward pass

## Files to Create/Modify

### New Files:
1. `graph/joint17.py` - Graph definition for 17 joints
2. `config/joint17/default.yaml` - Configuration file
3. `feeders/feeder_joint17.py` (if needed) - Data loader
4. `main_17joint.py` (optional) - Modified main script if needed

### Files to Verify (No Changes Expected):
- ✅ `main.py` - Already flexible
- ✅ `model/ctrgcn.py` - Uses `num_point` parameter
- ✅ `graph/tools.py` - Generic graph tools

## Open Questions

1. **What is the source of 17-joint data?**
   - Which dataset uses 17 joints?
   - Is it a subset of NTU25 or UCLA20?
   - Or a different dataset entirely?

2. **What is the joint mapping?**
   - Which 17 joints from NTU25 (or 20 from UCLA)?
   - What are the joint indices?

3. **What is the bone connectivity?**
   - How are the 17 joints connected?
   - What is the parent-child relationship?

4. **What is the data format?**
   - JSON (like UCLA)?
   - NPZ (like NTU)?
   - Other format?

5. **What is the dataset structure?**
   - Number of action classes?
   - Number of persons per sample?
   - Train/test split?

## Next Steps

1. **Gather Information**:
   - Identify 17-joint dataset
   - Obtain joint mapping and bone structure
   - Verify data format

2. **Create Graph File**:
   - Implement `graph/joint17.py` with correct structure

3. **Create Config File**:
   - Set up `config/joint17/default.yaml`

4. **Test Implementation**:
   - Run with sample data
   - Verify all components work together

5. **Documentation**:
   - Update README if needed
   - Document joint mapping

## Notes

- The model architecture (`model/ctrgcn.py`) is **already flexible** and will work with any number of joints as long as `num_point` is set correctly
- The main script (`main.py`) is **configuration-driven** and doesn't need modification
- The key work is defining the **graph structure** and **configuration file**
- If data format matches existing feeders, **no feeder changes needed**

