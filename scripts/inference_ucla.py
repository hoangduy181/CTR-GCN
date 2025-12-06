#!/usr/bin/env python
"""
Inference script for UCLA dataset
Usage: python scripts/inference_ucla.py [--weights WEIGHTS_PATH] [--device DEVICE]
"""

import argparse
import os
import sys
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_weights(work_dir):
    """Find the latest weights file in work directory"""
    pattern = os.path.join(work_dir, "runs-*.pt")
    weights_files = glob.glob(pattern)
    if weights_files:
        # Sort by modification time, get the latest
        weights_files.sort(key=os.path.getmtime, reverse=True)
        return weights_files[0]
    return None

def main():
    parser = argparse.ArgumentParser(description='Run inference on UCLA dataset')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights file (.pt). If not provided, will search in work_dir')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--config', type=str, default='config/ucla/default.yaml',
                        help='Path to config file')
    parser.add_argument('--work-dir', type=str, default='work_dir/ucla/inference',
                        help='Working directory for inference results')
    parser.add_argument('--save-score', action='store_true', default=True,
                        help='Save prediction scores')
    
    args = parser.parse_args()
    
    # If weights not provided, try to find them
    if args.weights is None:
        # Check common training directories
        possible_dirs = [
            'work_dir/ucla/ctrgcn_joint',
            'work_dir/ucla/ctrgcn',
            'work_dir/ucla'
        ]
        
        for work_dir in possible_dirs:
            weights = find_weights(work_dir)
            if weights:
                args.weights = weights
                print(f"Found weights: {weights}")
                break
        
        if args.weights is None:
            print("Error: No weights file found!")
            print("Please provide --weights argument or train the model first.")
            print("\nTo train:")
            print("  python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/ctrgcn_joint --device 0")
            return 1
    
    # Check if weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return 1
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Import and run main
    from main import get_parser, Processor, init_seed
    
    print("=" * 50)
    print("Running Inference on UCLA Dataset")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Weights: {args.weights}")
    print(f"Device: {args.device}")
    print(f"Work Dir: {args.work_dir}")
    print("=" * 50)
    
    # Parse arguments
    main_parser = get_parser()
    main_args = main_parser.parse_args([
        '--config', args.config,
        '--work-dir', args.work_dir,
        '--phase', 'test',
        '--save-score', 'True' if args.save_score else 'False',
        '--weights', args.weights,
        '--device', str(args.device)
    ])
    
    # Initialize seed
    init_seed(main_args.seed)
    
    # Create processor and run
    processor = Processor(main_args)
    processor.start()
    
    print("=" * 50)
    print("Inference completed!")
    print(f"Results saved in: {args.work_dir}")
    print("=" * 50)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
