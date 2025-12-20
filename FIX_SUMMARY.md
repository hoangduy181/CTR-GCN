# Fix Summary: 17-Joint 2D Data Support

## Issue Fixed

### Problem
When using `in_channels=2` (for 2D coordinates), the CTRGC convolution layer was calculating:
- `rel_channels = in_channels // rel_reduction = 2 // 8 = 0`

This caused a `RuntimeError: Given groups=1, expected weight to be at least 1 at dimension 0, but got weight of size [0, 2, 1, 1]`

### Solution
Modified `model/ctrgcn.py` line 158-159 to ensure minimum channel count:

**Before:**
```python
self.rel_channels = in_channels // rel_reduction
self.mid_channels = in_channels // mid_reduction
```

**After:**
```python
self.rel_channels = max(1, in_channels // rel_reduction)  # Ensure at least 1 channel
self.mid_channels = max(1, in_channels // mid_reduction)  # Ensure at least 1 channel
```

### Result
- With `in_channels=2`: `rel_channels = max(1, 2 // 8) = 1` ✅
- With `in_channels=3`: `rel_channels = max(1, 3 // 8) = 1` ✅ (unchanged behavior)
- With `in_channels=64`: `rel_channels = max(1, 64 // 8) = 8` ✅ (unchanged behavior)

## Files Modified

1. ✅ `model/ctrgcn.py` - Fixed CTRGC to handle 2 input channels
2. ✅ `feeders/feeder_ntu_2d.py` - Fixed p_interval to always be a list

## Status

✅ **Fixed and ready to use!**

The model can now handle 2D input data (17 joints, 2 coordinates) without errors.
