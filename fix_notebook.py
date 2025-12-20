import json
import re

# Read notebook
with open('eda/nw_ucla.ipynb', 'r') as f:
    nb = json.load(f)

# Process each cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Remove validation split loading
        if 'VAL_LABEL_PATH' in source:
            new_source = []
            for line in cell['source']:
                if 'VAL_LABEL_PATH' not in line and 'val_data_dict' not in line and 'Load validation split' not in line:
                    if 'print(f"Validation samples:' not in line and 'print(f"Train samples:' not in line:
                        new_source.append(line)
            cell['source'] = new_source
        
        # Remove split column creation
        if "df_files['split']" in source and 'Create train/val split' in source:
            new_source = []
            skip = False
            for line in cell['source']:
                if 'Create train/val split' in line:
                    skip = True
                    continue
                if skip and ('val_file_names' in line or "df_files['split']" in line):
                    continue
                if skip and line.strip() == '':
                    skip = False
                if not skip:
                    new_source.append(line)
            cell['source'] = new_source
        
        # Remove train/val statistics
        if 'Train samples:' in source or 'Val samples:' in source:
            cell['source'] = [line for line in cell['source'] 
                             if 'Train samples:' not in line and 'Val samples:' not in line]
        
        # Update action class distribution
        if 'action_counts_train' in source:
            # Replace the entire action distribution cell
            new_source = []
            in_action_cell = False
            for line in cell['source']:
                if '# Action class distribution' in line:
                    in_action_cell = True
                    new_source.append(line)
                elif in_action_cell:
                    if 'action_counts_train' in line or 'action_counts_val' in line:
                        continue
                    elif 'action_summary' in line and 'Train' in line:
                        continue
                    elif 'action_summary[' in line and 'Train' in line or 'Val' in line:
                        continue
                    elif 'Train vs Val' in line:
                        new_source.append(line.replace('Train vs Val', '').replace(':', ''))
                    elif "['Train', 'Val']" in line:
                        new_source.append(line.replace("action_summary[['Train', 'Val']]", "action_counts").replace(", color=['skyblue', 'salmon']", ", color='skyblue'"))
                    elif 'Action Class Distribution: Train vs Val' in line:
                        new_source.append(line.replace('Action Class Distribution: Train vs Val', 'Action Class Distribution'))
                    elif 'Overall Action Class Distribution' in line:
                        new_source.append(line.replace('Overall Action Class Distribution', 'Action Class Distribution'))
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
        
        # Remove split from sequence length analysis
        if 'df_seq.merge' in source and 'split' in source:
            cell['source'] = [line.replace("df_files[['file_name', 'action', 'split']]", "df_files[['file_name', 'action']]") 
                             for line in cell['source']]
        
        if 'by=\'split\'' in source:
            cell['source'] = [line for line in cell['source'] if 'by=\'split\'' not in line]
            # Need to adjust subplot layout
            for j, line in enumerate(cell['source']):
                if 'subplots(' in line and '2, 2' in line:
                    cell['source'][j] = line.replace('2, 2', '1, 2')
                elif 'axes[1, 1]' in line:
                    cell['source'][j] = line.replace('axes[1, 1]', 'axes[1]')
                elif 'Sequence Length: Train vs Val' in line:
                    cell['source'][j] = line.replace('Sequence Length: Train vs Val', 'Sequence Length Distribution')
        
        # Update viewpoint distribution
        if 'view_by_split' in source:
            cell['source'] = [line for line in cell['source'] 
                             if 'view_by_split' not in line and 'Viewpoint Distribution by Split' not in line
                             and 'Train vs Val' not in line]
        
        # Update subject distribution  
        if 'subject_by_split' in source:
            cell['source'] = [line for line in cell['source']
                             if 'subject_by_split' not in line and 'Subject Distribution by Split' not in line
                             and 'Subject Bias Analysis' not in line and 'train_count' not in line
                             and 'val_count' not in line and 'Train=' not in line and 'Val=' not in line
                             and 'Train vs Val' not in line]
        
        # Update summary statistics
        if 'Train/Val split:' in source:
            cell['source'] = [line for line in cell['source'] 
                             if 'Train/Val split:' not in line]

# Write back
with open('eda/nw_ucla.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated!")
