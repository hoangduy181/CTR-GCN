import json
import re

# Read notebook
with open('eda/nw_ucla.ipynb', 'r') as f:
    nb = json.load(f)

def fix_cell_source(source_lines):
    """Fix a cell's source code"""
    new_lines = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        
        # Skip validation split loading lines
        if any(x in line for x in ['VAL_LABEL_PATH', 'val_data_dict', 'Load validation split', 
                                   'Validation samples:', 'Train samples:']):
            i += 1
            continue
        
        # Remove split column creation
        if 'Create train/val split indicator' in line:
            # Skip until blank line after split assignment
            i += 1
            while i < len(source_lines) and ('val_file_names' in source_lines[i] or 
                                           "df_files['split']" in source_lines[i] or
                                           source_lines[i].strip() == ''):
                i += 1
            continue
        
        # Fix action class distribution
        if 'action_counts_train' in line or 'action_counts_val' in line:
            i += 1
            continue
        
        if 'action_summary = pd.DataFrame' in line:
            # Replace with simple action_counts
            new_lines.append("action_counts = df_files['action'].value_counts().sort_index()\n")
            i += 1
            # Skip the DataFrame creation lines
            while i < len(source_lines) and ('Total' in source_lines[i] or 
                                            'Train' in source_lines[i] or 
                                            'Val' in source_lines[i] or
                                            'fillna' in source_lines[i] or
                                            'astype' in source_lines[i] or
                                            "action_summary['" in source_lines[i]):
                i += 1
            continue
        
        if 'print(action_summary)' in line:
            new_lines.append("print(action_counts)\n")
            i += 1
            continue
        
        if "action_summary[['Train', 'Val']]" in line:
            new_lines.append("action_counts.plot(kind='bar', ax=axes[0], color='skyblue')\n")
            i += 1
            continue
        
        if 'Action Class Distribution: Train vs Val' in line:
            new_lines.append(line.replace('Action Class Distribution: Train vs Val', 'Action Class Distribution'))
            i += 1
            continue
        
        if 'axes[0].set_title' in line and 'Train vs Val' in line:
            new_lines.append("axes[0].set_title('Action Class Distribution', fontsize=14, fontweight='bold')\n")
            i += 1
            continue
        
        if 'action_summary.index' in line:
            new_lines.append(line.replace('action_summary.index', 'action_counts.index'))
            i += 1
            continue
        
        if 'Overall Action Class Distribution' in line:
            new_lines.append(line.replace('Overall Action Class Distribution', 'Action Class Distribution'))
            i += 1
            continue
        
        # Fix sequence length merge
        if "df_files[['file_name', 'action', 'split']]" in line:
            new_lines.append(line.replace("df_files[['file_name', 'action', 'split']]", "df_files[['file_name', 'action']]"))
            i += 1
            continue
        
        # Remove train/val boxplot
        if "by='split'" in line or 'Train vs Val' in line and 'Sequence Length' in ''.join(source_lines[max(0,i-5):i+5]):
            i += 1
            continue
        
        # Fix subplot layout if needed
        if 'subplots(2, 2' in line:
            new_lines.append(line.replace('subplots(2, 2', 'subplots(1, 2'))
            i += 1
            continue
        
        if 'axes[1, 1]' in line:
            new_lines.append(line.replace('axes[1, 1]', 'axes[1]'))
            i += 1
            continue
        
        # Remove viewpoint by split
        if 'view_by_split' in line or 'Viewpoint Distribution by Split' in line:
            i += 1
            continue
        
        if "view_by_split.plot" in line:
            i += 1
            continue
        
        # Remove subject by split
        if 'subject_by_split' in line or 'Subject Distribution by Split' in line:
            i += 1
            continue
        
        if 'Subject Bias Analysis' in line:
            # Skip until next section
            i += 1
            while i < len(source_lines) and ('train_count' in source_lines[i] or 
                                           'val_count' in source_lines[i] or
                                           'Train=' in source_lines[i] or
                                           'Val=' in source_lines[i] or
                                           'Subject' in source_lines[i] and ':' in source_lines[i]):
                i += 1
            continue
        
        if "subject_by_split.plot" in line:
            i += 1
            continue
        
        # Remove train/val from summary
        if 'Train/Val split:' in line or 'Train samples:' in line or 'Val samples:' in line:
            i += 1
            continue
        
        # Keep the line
        new_lines.append(line)
        i += 1
    
    return new_lines

# Process each code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = fix_cell_source(cell['source'])

# Write back
with open('eda/nw_ucla.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook comprehensively updated!")
