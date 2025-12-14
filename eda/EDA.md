# Exploratory Data Analysis (EDA) Plan
## NW-UCLA and NTU60 Skeleton Action Recognition Datasets

---

## 1. Dataset Overview

### 1.1 Basic Statistics
- **NW-UCLA:**
  - Total number of samples (train + val)
  - Number of action classes (10)
  - Number of subjects
  - Number of camera views (3)
  - Number of environments/executions per action
  - Train/validation split ratio and counts
  
- **NTU60:**
  - Total number of samples (train + test)
  - Number of action classes (60)
  - Number of performers/subjects
  - Number of camera setups
  - Cross-subject (CS) vs Cross-view (CV) split counts
  - Train/test split ratio

### 1.2 Dataset Structure
- **NW-UCLA:**
  - File format: JSON files
  - Directory structure: `data/NW-UCLA/all_sqe/`
  - File naming convention: `a{action}_s{subject}_e{environment}_v{view}.json`
  - Total files count: ~1484 JSON files
  
- **NTU60:**
  - File format: Preprocessed `.npz` files
  - Files: `NTU60_CS.npz` (Cross-Subject), `NTU60_CV.npz` (Cross-View)
  - Raw data location: `data/ntu/nturgb+d_skeletons/`
  - Statistics files available in `data/ntu/statistics/`

### 1.3 Action Class Distribution
- **NW-UCLA:**
  - Count samples per action class (1-10)
  - Visualize class distribution (bar plot)
  - Check for class imbalance
  
- **NTU60:**
  - Count samples per action class (1-60)
  - Visualize class distribution (bar plot)
  - Analyze class imbalance
  - Group actions by categories if available

---

## 2. File Format & Data Schema

### 2.1 NW-UCLA JSON Structure
- Load and inspect sample JSON file
- Document schema:
  - `file_name`: string identifier
  - `skeletons`: list of frames, each frame contains 20 joints
  - `label`: action class (1-indexed)
- Verify joint coordinate format (X, Y, Z)
- Check data types and value ranges
- Identify any metadata fields

### 2.2 NTU60 NPZ Structure
- Load and inspect `.npz` file structure
- Document arrays:
  - `x_train`, `x_test`: shape `(N, T, 2, 25, 3)`
  - `y_train`, `y_test`: one-hot encoded labels
- Verify data dimensions:
  - N: number of samples
  - T: temporal frames (variable)
  - 2: maximum persons
  - 25: joints per skeleton
  - 3: coordinates (X, Y, Z)
- Check data types and value ranges

### 2.3 Data Shape Analysis
- **NW-UCLA:**
  - Frame count distribution per sample
  - Joint count verification (should be 20)
  - Coordinate dimensions (should be 3D)
  
- **NTU60:**
  - Frame count distribution per sample
  - Joint count verification (should be 25)
  - Person count distribution (1 vs 2 persons)
  - Coordinate dimensions (should be 3D)

---

## 3. Temporal Characteristics

### 3.1 Sequence Length Analysis
- **NW-UCLA:**
  - Distribution of sequence lengths (frames per sample)
  - Min, max, mean, median sequence lengths
  - Visualize histogram of sequence lengths
  - Identify outliers (very short/long sequences)
  - Sequence length by action class
  
- **NTU60:**
  - Distribution of sequence lengths (frames per sample)
  - Min, max, mean, median sequence lengths
  - Visualize histogram of sequence lengths
  - Sequence length by action class
  - Sequence length by number of persons

### 3.2 Temporal Sampling
- **NW-UCLA:**
  - Check if sequences are uniformly sampled
  - Analyze frame rate consistency
  - Identify any temporal gaps or irregularities
  
- **NTU60:**
  - Check temporal sampling patterns
  - Analyze valid frame counts vs total frames
  - Identify padding patterns

### 3.3 Temporal Dynamics
- Compute frame-to-frame differences
- Analyze motion intensity over time
- Identify action phases (start, middle, end)
- Compare temporal patterns across action classes

---

## 4. Spatial (Skeleton) Characteristics

### 4.1 Skeleton Structure
- **NW-UCLA:**
  - Joint count: 20 joints
  - Bone connectivity structure (parent-child relationships)
  - Visualize skeleton topology/graph
  - Joint naming/indexing convention
  
- **NTU60:**
  - Joint count: 25 joints
  - Bone connectivity structure (25 bone pairs)
  - Visualize skeleton topology/graph
  - Joint naming/indexing convention
  - Compare with NW-UCLA skeleton structure

### 4.2 Joint Coordinate Analysis
- **Coordinate Ranges:**
  - Min/max X, Y, Z coordinates per dataset
  - Mean and std of coordinates per joint
  - Identify coordinate system (world space, relative, normalized)
  
- **Joint Position Statistics:**
  - Mean position of each joint across all samples
  - Joint position variance
  - Most/least variable joints
  - Joint position distributions (histograms)

### 4.3 Spatial Relationships
- **Bone Length Analysis:**
  - Compute bone vectors (child - parent)
  - Bone length distributions
  - Average bone lengths
  - Most/least variable bones
  
- **Skeleton Scale:**
  - Overall skeleton size (bounding box dimensions)
  - Scale variation across samples
  - Scale by action class

### 4.4 Person Detection (NTU60 only)
- Distribution of single vs dual person samples
- Analyze person detection quality
- Check for missing person data (zeros)
- Person position relationships in dual-person samples

---

## 5. Motion Statistics

### 5.1 Velocity Analysis
- Compute per-joint velocities (frame-to-frame differences)
- Velocity magnitude distributions
- Identify fastest/slowest moving joints
- Velocity patterns by action class
  
### 5.2 Acceleration Analysis
- Compute per-joint accelerations (second-order differences)
- Acceleration magnitude distributions
- Identify joints with highest acceleration
- Acceleration patterns by action class

### 5.3 Motion Patterns
- **Temporal Motion Profiles:**
  - Plot velocity over time for sample actions
  - Identify motion peaks and valleys
  - Compare motion intensity across actions
  
- **Spatial Motion Patterns:**
  - Which body parts move most/least
  - Motion correlation between joints
  - Symmetry in motion (left vs right limbs)

### 5.4 Action-Specific Motion Characteristics
- Average motion intensity per action class
- Motion duration patterns
- Distinguishability of actions based on motion features

---

## 6. Missing / Noise Analysis

### 6.1 Missing Data Detection
- **NW-UCLA:**
  - Check for missing joints (zero coordinates)
  - Check for missing frames
  - Identify incomplete samples
  
- **NTU60:**
  - Check for missing joints (zero coordinates)
  - Check for missing persons (all zeros)
  - Analyze `valid_frame_num` patterns
  - Identify samples with missing data

### 6.2 Data Quality Assessment
- **Noise Detection:**
  - Identify outliers in joint positions
  - Detect unrealistic joint movements (jumps)
  - Check for coordinate system inconsistencies
  
- **Data Completeness:**
  - Percentage of complete samples
  - Percentage of complete frames
  - Missing data patterns by action class

### 6.3 Data Cleaning Insights
- Document data quality issues
- Recommend preprocessing steps
- Identify samples that may need filtering

---

## 7. Viewpoint & Subject Variations

### 7.1 Viewpoint Analysis (NW-UCLA)
- **Camera Views:**
  - Distribution across views (v01, v02, v03)
  - View-specific characteristics
  - Viewpoint variation impact on skeleton appearance
  - Cross-view consistency analysis
  
- **Viewpoint Transformation:**
  - Analyze viewpoint augmentation effects
  - Check for view-specific biases

### 7.2 Subject/Performer Analysis
- **NW-UCLA:**
  - Distribution across subjects (s01-s10)
  - Subject-specific characteristics
  - Inter-subject variation
  - Subject bias in train/val split
  
- **NTU60:**
  - Distribution across performers
  - Performer-specific characteristics
  - Cross-subject split analysis
  - Performer bias in train/test split

### 7.3 Environment/Execution Variations
- **NW-UCLA:**
  - Distribution across environments/executions
  - Execution variability within same action
  - Impact on action recognition difficulty

### 7.4 Intra-Class Variability
- Analyze variation within same action class
- Compare inter-class vs intra-class distances
- Identify most/least variable action classes

---

## 8. Cross-Dataset Comparison

### 8.1 Dataset Scale Comparison
- Sample count comparison
- Action class count comparison
- Temporal length comparison
- Spatial complexity comparison (joints, persons)

### 8.2 Skeleton Structure Comparison
- Joint count: 20 (NW-UCLA) vs 25 (NTU60)
- Bone structure differences
- Skeleton topology comparison
- Coordinate system differences

### 8.3 Data Format Comparison
- JSON (NW-UCLA) vs NPZ (NTU60)
- Loading efficiency comparison
- Preprocessing requirements
- Storage size comparison

### 8.4 Action Complexity Comparison
- Average sequence length comparison
- Motion complexity comparison
- Action diversity comparison
- Difficulty assessment

### 8.5 Split Strategy Comparison
- Train/val (NW-UCLA) vs Train/test (NTU60)
- Cross-subject vs cross-view evaluation
- Split size comparison
- Split bias analysis

---

## 9. Key Observations & Modeling Implications

### 9.1 Data Characteristics Summary
- Key statistics summary table
- Notable patterns and anomalies
- Dataset strengths and limitations

### 9.2 Preprocessing Recommendations
- Normalization requirements
- Temporal alignment strategies
- Data augmentation opportunities
- Missing data handling

### 9.3 Model Design Implications
- Input size considerations
- Architecture choices (joint count, temporal modeling)
- Multi-person handling (NTU60)
- Modality selection (joint, bone, motion)

### 9.4 Training Strategy Recommendations
- Class imbalance handling
- Data augmentation strategies
- Validation strategy
- Hyperparameter considerations

### 9.5 Potential Challenges
- Identified difficulties
- Edge cases to handle
- Dataset-specific considerations
- Transfer learning opportunities

---

## Implementation Notes

### Tools & Libraries
- Python: numpy, pandas, matplotlib, seaborn
- Visualization: plotly, matplotlib
- Data loading: json, pickle, numpy
- Statistics: scipy

### Notebook Structure
- `nw_ucla.ipynb`: NW-UCLA specific analysis
- `ntu60.ipynb`: NTU60 specific analysis
- Shared analysis functions in separate cells
- Visualization functions reusable across datasets

### Deliverables
- Comprehensive analysis notebooks
- Summary statistics tables
- Visualization plots and figures
- Key findings document
- Preprocessing recommendations
