# Notebooks

This folder contains Jupyter notebooks for exploring and working with the CTR-GCN codebase.

## Available Notebooks

### `ntu60_google_drive_setup.ipynb`
**Purpose**: Download, process, and set up NTU RGB+D 60 dataset in Google Drive

**Use Cases**:
- Setting up NTU60 dataset for use in Google Colab
- Processing raw skeleton data into `.npz` format
- Organizing data in Google Drive for easy access

**Steps Covered**:
1. Mount Google Drive
2. Download/upload NTU60 raw data
3. Process raw data into `.npz` format
4. Verify processed files
5. Set up for training

**Estimated Time**: 30-60 minutes (depending on processing speed)

**Storage Requirements**: ~5-10 GB in Google Drive

---

## Usage

### In Google Colab:
1. Upload the notebook to Google Colab
2. Run cells sequentially
3. Follow the instructions in each cell

### Locally:
1. Install Jupyter: `pip install jupyter`
2. Start Jupyter: `jupyter notebook`
3. Open the notebook from this directory

---

## Notes

- These notebooks are designed to be run in Google Colab for easy Google Drive integration
- All paths assume the CTR-GCN project is located at `/content/drive/MyDrive/CTR-GCN`
- Adjust paths as needed for your setup

---

## Future Notebooks

Additional notebooks may be added for:
- Data exploration and visualization
- Model training workflows
- Inference and evaluation
- Parameter tuning experiments
