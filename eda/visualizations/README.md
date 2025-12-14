# Skeleton Visualization Scripts

## visualize_skeletons.py

This script visualizes all frames of sample skeletons for each action class in the NW-UCLA dataset.

### Usage

**Option 1: Run from Jupyter Notebook**

Add this cell to your `nw_ucla.ipynb` notebook:

```python
import sys
sys.path.append('visualizations')

from visualize_skeletons import *

# Update paths to match your notebook location
DATA_ROOT = Path('../data/NW-UCLA/all_sqe')
OUTPUT_ROOT = Path('visualizations')

# Run visualization
main()
```

**Option 2: Run from command line**

Make sure you're in the correct conda environment (e.g., `jupyter_notebooks`):

```bash
cd /home/duyth/ai_coding/CTR-GCN/eda/visualizations
conda activate jupyter_notebooks  # or your environment name
python visualize_skeletons.py
```

### Output Structure

The script creates folders for each action class:
```
visualizations/
├── action_01_a01_s01_e00_v01/
│   ├── frame_000.png
│   ├── frame_001.png
│   └── ...
├── action_02_a02_s01_e00_v01/
│   ├── frame_000.png
│   └── ...
└── ...
```

Each folder contains all frames of one sample skeleton for that action class.
