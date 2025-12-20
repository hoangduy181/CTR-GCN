import json

# Read notebook
with open('eda/nw_ucla.ipynb', 'r') as f:
    nb = json.load(f)

# Process each code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        
        for i, line in enumerate(source):
            # Remove duplicate action_counts line
            if i > 0 and 'action_counts = df_files' in line and 'action_counts = df_files' in source[i-1]:
                continue
            
            # Remove "Create summary DataFrame" comment if followed by duplicate action_counts
            if '# Create summary DataFrame' in line:
                if i+1 < len(source) and 'action_counts = df_files' in source[i+1]:
                    continue
            
            # Fix missing plot call
            if 'axes[0].set_xlabel' in line and i > 0 and 'axes[0].set_xlabel' not in source[i-1]:
                # Check if plot call is missing
                prev_line = source[i-1] if i > 0 else ''
                if 'action_counts.plot' not in prev_line and 'Bar plot' not in prev_line:
                    new_source.append("action_counts.plot(kind='bar', ax=axes[0], color='skyblue')\n")
                    new_source.append("axes[0].set_title('Action Class Distribution', fontsize=14, fontweight='bold')\n")
            
            # Remove train/val references in summary
            if "len(df_files[df_files['split']" in line:
                continue
            
            if 'Train/Val split:' in line:
                continue
            
            # Keep other lines
            new_source.append(line)
        
        cell['source'] = new_source

# Write back
with open('eda/nw_ucla.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Final cleanup complete!")
