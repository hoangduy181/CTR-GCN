# Parameters Exploration

This folder contains documentation about all parameters used in the CTR-GCN codebase.

## Files

- **PARAMETERS.md** - Comprehensive documentation of all parameters, organized by category

## Overview

Parameters in CTR-GCN are managed through:
1. **Command-line arguments** (highest priority)
2. **YAML configuration files** (medium priority)
3. **Default values** (lowest priority)

All parameters are defined in `main.py` using `argparse.ArgumentParser` and can be explored in detail in the PARAMETERS.md file.

## Quick Reference

### Parameter Categories
1. **General Configuration** - Work directory, config file, model names
2. **Processor Parameters** - Phase (train/test), score saving
3. **Visualization & Debugging** - Logging, evaluation intervals, random seed
4. **Data Feeder Parameters** - Data loading, preprocessing, augmentation
5. **Model Parameters** - Model architecture, weights, graph structure
6. **Optimizer Parameters** - Learning rate, batch size, epochs, optimizer settings

## Related Files

- Parameter definitions: `main.py` → `get_parser()` function
- Parameter usage: `main.py` → `Processor` class
- Example configs: `config/` directory
