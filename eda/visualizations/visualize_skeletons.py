"""
Visualize all frames of sample skeletons for each action class in NW-UCLA dataset.

For each action class, this script:
1. Selects one sample skeleton
2. Visualizes all frames of that skeleton
3. Saves each frame as an image in a folder named after the skeleton file
"""

import numpy as np
import json
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# Define paths
DATA_ROOT = Path('../../data/NW-UCLA/all_sqe')
OUTPUT_ROOT = Path('.')  # Current directory (visualizations folder)

# NW-UCLA bone pairs (child, parent) - 1-indexed
UCLA_BONE_PAIRS = [
    (1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), 
    (9, 3), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), 
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)
]


# def plot_skeleton_3d(skeleton, bone_pairs, ax=None, title="Skeleton 3D"):
#     """
#     Plot skeleton in 3D space
    
#     Args:
#         skeleton: (V, 3) array of joint coordinates
#         bone_pairs: list of (child, parent) tuples
#         ax: matplotlib 3D axis (optional)
#         title: plot title
#     """
#     if ax is None:
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
    
#     # Plot joints
#     ax.scatter(
#         skeleton[:, 2], 
#         skeleton[:, 1], 
#         skeleton[:, 0], 
#         s=100, c='red', marker='o', label='Joints'
#     )
    
#     # Plot bones
#     for child, parent in bone_pairs:
#         if child != parent:
#             child_idx = child - 1  # Convert to 0-indexed
#             parent_idx = parent - 1
#             if child_idx < len(skeleton) and parent_idx < len(skeleton):
#                 ax.plot(
#                     [skeleton[parent_idx, 2], skeleton[child_idx, 2]],
#                     [skeleton[parent_idx, 1], skeleton[child_idx, 1]],
#                     [skeleton[parent_idx, 0], skeleton[child_idx, 0]],
#                     'b-', linewidth=2)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(title)
#     ax.legend()
#     return ax

def plot_skeleton_3d(skeleton, bone_pairs, ax=None, title="Skeleton 3D", axis_limits=None):
    """
    Plot skeleton in 3D space with fixed axis limits
    
    Args:
        skeleton: (V, 3) array of joint coordinates [x, y, z]
        bone_pairs: list of (child, parent) tuples
        ax: matplotlib 3D axis (optional)
        title: plot title
        axis_limits: dict with 'xlim', 'ylim', 'zlim' tuples (optional)
                    Note: plotted as (z, y, x) so limits match that mapping
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints (note: plotting skeleton[:, 2] as X, skeleton[:, 1] as Y, skeleton[:, 0] as Z)
    ax.scatter(
        skeleton[:, 2], 
        skeleton[:, 1], 
        skeleton[:, 0], 
        s=100, c='red', marker='o', label='Joints'
    )
    
    # Plot bones
    for child, parent in bone_pairs:
        if child != parent:
            child_idx = child - 1  # Convert to 0-indexed
            parent_idx = parent - 1
            if child_idx < len(skeleton) and parent_idx < len(skeleton):
                ax.plot(
                    [skeleton[parent_idx, 2], skeleton[child_idx, 2]],
                    [skeleton[parent_idx, 1], skeleton[child_idx, 1]],
                    [skeleton[parent_idx, 0], skeleton[child_idx, 0]],
                    'b-', linewidth=2)
    
    # Set fixed axis limits if provided
    if axis_limits is not None:
        ax.set_xlim(axis_limits['xlim'])
        ax.set_ylim(axis_limits['ylim'])
        ax.set_zlim(axis_limits['zlim'])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    return ax


def parse_filename(filename):
    """Parse NW-UCLA filename: a{action}_s{subject}_e{environment}_v{view}.json"""
    basename = Path(filename).stem
    parts = basename.split('_')
    return {
        'action': int(parts[0][1:]),
        'subject': int(parts[1][1:]),
        'environment': int(parts[2][1:]),
        'view': int(parts[3][1:]),
        'file_name': basename
    }


def visualize_skeleton_frames(json_file, output_dir):
    """
    Visualize all frames of a skeleton with fixed axis limits and consistent image sizes
    
    Args:
        json_file: path to JSON file
        output_dir: directory to save frame images
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load skeleton data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    skeletons = np.array(data['skeletons'])  # Shape: (T, V, 3) where 3 is [x, y, z]
    file_name = Path(json_file).stem
    
    print(f"  Processing {file_name}: {skeletons.shape[0]} frames")
    
    # Calculate axis limits from all frames (with padding)
    # Note: We plot skeleton[:, 2] as X, skeleton[:, 1] as Y, skeleton[:, 0] as Z
    all_x = skeletons[:, :, 2].flatten()  # All X coordinates (plotted as X axis)
    all_y = skeletons[:, :, 1].flatten()  # All Y coordinates (plotted as Y axis)
    all_z = skeletons[:, :, 0].flatten()  # All Z coordinates (plotted as Z axis)
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()
    
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    z_range = z_max - z_min if z_max != z_min else 1.0
    
    padding = 0.1  # 10% padding
    
    axis_limits = {
        'xlim': (x_min - padding * x_range, x_max + padding * x_range),
        'ylim': (y_min - padding * y_range, y_max + padding * y_range),
        'zlim': (z_min - padding * z_range, z_max + padding * z_range)
    }
    
    print(f"  Fixed axis limits: X=[{axis_limits['xlim'][0]:.2f}, {axis_limits['xlim'][1]:.2f}], "
          f"Y=[{axis_limits['ylim'][0]:.2f}, {axis_limits['ylim'][1]:.2f}], "
          f"Z=[{axis_limits['zlim'][0]:.2f}, {axis_limits['zlim'][1]:.2f}]")
    
    # Visualize each frame with fixed limits and consistent size
    for frame_idx in range(skeletons.shape[0]):
        skeleton_frame = skeletons[frame_idx]  # (V, 3)
        
        # Create figure with fixed size
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot skeleton with fixed axis limits
        plot_skeleton_3d(
            skeleton_frame, 
            UCLA_BONE_PAIRS, 
            ax=ax, 
            title=f"{file_name} - Frame {frame_idx:03d}",
            axis_limits=axis_limits
        )
        
        # Set fixed subplot parameters to ensure consistent image size
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Save figure with fixed size (no tight bbox to ensure consistent dimensions)
        output_path = output_dir / f"frame_{frame_idx:03d}.png"
        plt.savefig(output_path, dpi=100, bbox_inches=None, facecolor='white')
        plt.close(fig)
    
    print(f"  Saved {skeletons.shape[0]} frames to {output_dir}")


def main():
    """Main function to visualize sample skeletons for each action"""
    print("Loading NW-UCLA dataset...")
    
    # Get all JSON files
    all_json_files = sorted(glob.glob(str(DATA_ROOT / '*.json')))
    print(f"Found {len(all_json_files)} JSON files")
    
    # Group files by action
    files_by_action = defaultdict(list)
    for json_file in all_json_files:
        file_info = parse_filename(json_file)
        action = file_info['action']
        files_by_action[action].append(json_file)
    
    print(f"\nFound {len(files_by_action)} action classes")
    
    # For each action, select one sample and visualize all frames
    for action in sorted(files_by_action.keys()):
        action_files = files_by_action[action]
        # Select the first file as sample (you can modify this to select differently)
        sample_file = action_files[0]
        file_name = Path(sample_file).stem
        
        print(f"\nAction {action:02d}: Visualizing skeleton '{file_name}'")
        
        # Create output directory for this skeleton
        output_dir = OUTPUT_ROOT / f"action_{action:02d}_{file_name}"
        
        # Visualize all frames
        visualize_skeleton_frames(sample_file, output_dir)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Output saved to: {OUTPUT_ROOT.absolute()}")


if __name__ == "__main__":
    main()
