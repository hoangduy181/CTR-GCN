#!/usr/bin/env python
"""Explore NTU60 2D pickle file structure"""
import pickle
import numpy as np
import sys

def explore_pickle(filepath):
    """Try multiple methods to load and explore pickle file"""
    print(f"Exploring: {filepath}")
    print("=" * 60)
    
    # Method 1: Standard pickle load
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print("✓ Successfully loaded with standard pickle.load()")
        analyze_data(data)
        return data
    except Exception as e:
        print(f"✗ Standard load failed: {type(e).__name__}: {e}")
    
    # Method 2: Try with encoding
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print("✓ Successfully loaded with encoding='latin1'")
        analyze_data(data)
        return data
    except Exception as e:
        print(f"✗ Encoding load failed: {type(e).__name__}: {e}")
    
    # Method 3: Try reading bytes and checking structure
    try:
        with open(filepath, 'rb') as f:
            first_bytes = f.read(100)
            print(f"\nFirst 100 bytes (hex): {first_bytes[:50].hex()}")
            print(f"First 100 bytes (repr): {repr(first_bytes[:50])}")
    except Exception as e:
        print(f"✗ Byte reading failed: {e}")
    
    # Method 4: Try joblib
    try:
        import joblib
        data = joblib.load(filepath)
        print("✓ Successfully loaded with joblib")
        analyze_data(data)
        return data
    except ImportError:
        print("✗ joblib not available")
    except Exception as e:
        print(f"✗ joblib load failed: {type(e).__name__}: {e}")
    
    return None

def analyze_data(data):
    """Analyze the loaded data structure"""
    print("\n" + "=" * 60)
    print("DATA STRUCTURE ANALYSIS")
    print("=" * 60)
    
    print(f"\nType: {type(data)}")
    
    if isinstance(data, dict):
        print(f"\nDictionary with {len(data)} keys")
        print(f"First 10 keys: {list(data.keys())[:10]}")
        
        # Check for common keys
        common_keys = ['x_train', 'x_test', 'y_train', 'y_test', 'train', 'test', 'data', 'label']
        found_keys = [k for k in common_keys if k in data]
        if found_keys:
            print(f"\nFound common keys: {found_keys}")
            for key in found_keys[:5]:  # Analyze first 5
                print(f"\n  {key}:")
                analyze_item(data[key], indent=4)
    
    elif isinstance(data, (list, tuple)):
        print(f"\nSequence with {len(data)} elements")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            analyze_item(data[0], indent=2)
    
    elif hasattr(data, 'shape'):
        print(f"\nNumPy array")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        if data.size > 0:
            print(f"Min: {data.min()}, Max: {data.max()}")
            print(f"Sample values:\n{data.flat[:10]}")
    
    else:
        print(f"\nOther type: {type(data)}")
        print(f"Attributes: {dir(data)[:10]}")

def analyze_item(item, indent=0):
    """Recursively analyze an item"""
    prefix = " " * indent
    if isinstance(item, np.ndarray):
        print(f"{prefix}NumPy array: shape={item.shape}, dtype={item.dtype}")
        if len(item.shape) >= 2:
            print(f"{prefix}  Sample shape interpretation:")
            if len(item.shape) == 4:
                print(f"{prefix}    (N, T, V, C) or (N, T, M, V) or (N, C, T, V)")
            elif len(item.shape) == 5:
                print(f"{prefix}    (N, T, M, V, C) or (N, C, T, V, M)")
    elif isinstance(item, (list, tuple)):
        print(f"{prefix}Sequence: length={len(item)}")
        if len(item) > 0:
            print(f"{prefix}  First element type: {type(item[0])}")
            if isinstance(item[0], np.ndarray):
                analyze_item(item[0], indent=indent+2)
    elif isinstance(item, dict):
        print(f"{prefix}Dictionary: {len(item)} keys")
        print(f"{prefix}  Keys: {list(item.keys())[:5]}")
    else:
        print(f"{prefix}Type: {type(item)}")

if __name__ == '__main__':
    filepath = 'data/ntu2d/ntu60_2d.pkl'
    data = explore_pickle(filepath)
    
    if data is not None:
        print("\n" + "=" * 60)
        print("JOINT COUNT ANALYSIS")
        print("=" * 60)
        
        # Try to find joint dimension
        if isinstance(data, dict):
            for key in ['x_train', 'x_test', 'train', 'test', 'data']:
                if key in data:
                    arr = data[key]
                    if isinstance(arr, np.ndarray):
                        print(f"\n{key} shape: {arr.shape}")
                        if len(arr.shape) >= 3:
                            # Assuming format like (N, T, V, C) or (N, T, M, V, C)
                            print(f"  Possible joint dimension (V): {arr.shape[-2] if len(arr.shape) >= 2 else 'unknown'}")
                    elif isinstance(arr, list) and len(arr) > 0:
                        sample = arr[0]
                        if isinstance(sample, np.ndarray):
                            print(f"\n{key}[0] shape: {sample.shape}")
                            if len(sample.shape) >= 2:
                                print(f"  Possible joint dimension: {sample.shape[-2] if len(sample.shape) >= 2 else sample.shape[-1]}")
