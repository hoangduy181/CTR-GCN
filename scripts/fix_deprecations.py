#!/usr/bin/env python
"""
Script to fix deprecated PyTorch and NumPy functions in CTR-GCN codebase
Run this script to automatically fix deprecated code patterns.
"""

import os
import re
import sys

def fix_variable_imports(file_path):
    """Remove Variable imports and usage"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Remove Variable import
    content = re.sub(
        r'from torch\.autograd import Variable\n',
        '',
        content
    )
    
    # Replace Variable(...) with just the tensor creation
    # Pattern: Variable(torch.from_numpy(...), requires_grad=False)
    content = re.sub(
        r'Variable\(torch\.from_numpy\(([^)]+)\)\s*,\s*requires_grad=False\)',
        r'torch.from_numpy(\1)',
        content
    )
    
    # Pattern: Variable(torch.from_numpy(...), requires_grad=True)
    content = re.sub(
        r'Variable\(torch\.from_numpy\(([^)]+)\)\s*,\s*requires_grad=True\)',
        r'torch.from_numpy(\1).requires_grad_(True)',
        content
    )
    
    # Generic Variable removal (if no requires_grad specified)
    content = re.sub(
        r'Variable\(([^)]+)\)',
        r'\1',
        content
    )
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def fix_get_device(file_path):
    """Replace .get_device() with .device"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace .get_device() with .device
    # Pattern: .cuda(x.get_device())
    content = re.sub(
        r'\.cuda\(([^)]+)\.get_device\(\)\)',
        r'.to(\1.device)',
        content
    )
    
    # Pattern: x.get_device() standalone
    content = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)\.get_device\(\)',
        r'\1.device',
        content
    )
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all deprecated code patterns"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    files_to_fix = [
        'model/ctrgcn.py',
        'model/baseline.py',
        'torchlight/torchlight/util.py',
    ]
    
    print("=" * 60)
    print("Fixing Deprecated Code Patterns")
    print("=" * 60)
    
    for file_rel_path in files_to_fix:
        file_path = os.path.join(base_dir, file_rel_path)
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_rel_path}")
            continue
        
        print(f"\n📝 Processing: {file_rel_path}")
        
        fixed_variable = fix_variable_imports(file_path)
        fixed_device = fix_get_device(file_path)
        
        if fixed_variable or fixed_device:
            print(f"   ✅ Fixed deprecated patterns")
            if fixed_variable:
                print(f"      - Removed Variable imports/usage")
            if fixed_device:
                print(f"      - Replaced .get_device() with .device")
        else:
            print(f"   ℹ️  No changes needed")
    
    print("\n" + "=" * 60)
    print("Done! Please review the changes before committing.")
    print("=" * 60)

if __name__ == '__main__':
    main()
