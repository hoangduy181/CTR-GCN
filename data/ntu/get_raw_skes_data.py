# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Parse a single .skeleton file and extract raw body data for all actors in the sequence.
    
    This function reads the NTU RGB+D skeleton file format which has the following structure:
    - Line 0: Number of frames
    - For each frame:
      - Line: Number of bodies detected
      - For each body:
        - Line: bodyID (e.g., "0" or "1")
        - Line: Number of joints (always 25)
        - For each joint (25 lines):
          - Format: x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ
          - We extract: x y z (joints) and colorX colorY (colors)

    Args:
        skes_path: Directory path containing the .skeleton files
        ske_name: Name of the skeleton file (without .skeleton extension)
        frames_drop_skes: Dictionary to track which frames were dropped for each skeleton
        frames_drop_logger: Logger object to record dropped frames
    
    Returns:
        Dictionary containing:
            - name: skeleton filename
            - data: dictionary mapping bodyID to body data (joints, colors, interval, motion)
            - num_frames: number of valid frames (excluding dropped frames)
    
    Each body's data contains:
      - joints: 3D joint positions stacked across frames. Shape: (num_frames x 25, 3)
      - colors: 2D color locations stacked across frames. Shape: (num_frames, 25, 2)
      - interval: list of frame indices where this body appears
      - motion: motion variance (only calculated if 2+ bodies exist)
    """
    # Construct full path to skeleton file
    ske_file = osp.join(skes_path, ske_name + '.skeleton')
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    
    # Read all lines from the skeleton file
    print('Reading data from %s' % ske_file[-29:])
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    # Parse header: first line contains total number of frames
    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []  # Track frames with no body data
    bodies_data = dict()  # Store data for each bodyID
    valid_frames = -1  # Counter for valid frames (0-based index)
    current_line = 1  # Start reading from line 1 (after header)

    # Process each frame
    for f in range(num_frames):
        # Read number of bodies detected in this frame
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        # Skip frames with no body data (missing/dropped frames)
        if num_bodies == 0:
            frames_drop.append(f)  # Record dropped frame index
            continue

        # Increment valid frame counter
        valid_frames += 1
        # Initialize arrays for joints (3D positions) and colors (2D locations) for all bodies in this frame
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        # Process each body detected in this frame
        for b in range(num_bodies):
            # Read bodyID (e.g., "0" or "1")
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            # Read number of joints (should always be 25 for NTU dataset)
            num_joints = int(str_data[current_line].strip('\r\n'))
            current_line += 1

            # Read joint data: each joint has 11 values, we extract positions and colors
            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                # Extract 3D joint position (x, y, z) - first 3 values
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                # Extract 2D color location (colorX, colorY) - values at indices 5 and 6
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            # Store or update body data
            if bodyID not in bodies_data:
                # First time seeing this bodyID - create new entry
                body_data = dict()
                body_data['joints'] = joints[b]  # Shape: (25, 3) for first frame
                body_data['colors'] = colors[b, np.newaxis]  # Shape: (1, 25, 2) for first frame
                body_data['interval'] = [valid_frames]  # Record first frame index
            else:
                # BodyID already exists - append data from this frame
                body_data = bodies_data[bodyID]
                # Stack joints vertically: (num_frames x 25, 3)
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                # Stack colors vertically: (num_frames, 25, 2)
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                # Calculate next frame index in sequence
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)

            bodies_data[bodyID] = body_data  # Update the dictionary

    # Validate that not all frames were dropped
    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    
    # Log dropped frames if any
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=int)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # Calculate motion variance for each body (only if multiple bodies exist)
    # Motion is the sum of variances across all joints, used to identify the main actor
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            # Sum of variances across all 3 dimensions (x, y, z) for all joints
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data():
    """
    Batch process all skeleton files and convert them to a structured pickle format.
    
    This function:
    1. Loads the list of available skeleton filenames
    2. Processes each skeleton file to extract raw body data
    3. Saves the processed data to pickle files for later use
    
    Note: This function uses global variables defined in the __main__ block:
        - skes_name_file: Path to file containing list of skeleton filenames
        - skes_path: Directory containing .skeleton files
        - save_data_pkl: Output path for raw skeleton data pickle file
        - frames_drop_pkl: Output path for dropped frames pickle file
        - frames_drop_logger: Logger for recording dropped frames
        - frames_drop_skes: Dictionary to track dropped frames
    """
    # Load list of skeleton filenames (one per line, without .skeleton extension)
    skes_name = np.loadtxt(skes_name_file, dtype=str)

    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    # Initialize storage for processed data
    raw_skes_data = []  # List to store processed skeleton data
    frames_cnt = np.zeros(num_files, dtype=int)  # Track frame count for each skeleton

    # Process each skeleton file
    for (idx, ske_name) in enumerate(skes_name):
        # Parse skeleton file and extract body data
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames']
        
        # Progress update every 1000 files
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    # Save processed raw skeleton data to pickle file
    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, 4)
    
    # Save frame counts to text file (one count per line)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    # Save dropped frames information to pickle file
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, 4)

if __name__ == '__main__':
    """
    Main execution block: Set up paths and run the skeleton data extraction pipeline.
    
    This script processes raw NTU RGB+D skeleton files (.skeleton format) and converts
    them into a structured pickle format that can be used for training and analysis.
    
    Output files:
        - raw_data/raw_skes_data.pkl: All processed skeleton data
        - raw_data/frames_cnt.txt: Frame count for each skeleton file
        - raw_data/frames_drop_skes.pkl: Dictionary of dropped frames per skeleton
        - raw_data/frames_drop.log: Log file recording dropped frames
    """
    save_path = './'  # Base directory for output files

    # Path to directory containing raw .skeleton files
    skes_path = './nturgb+d_skeletons/'
    stat_path = osp.join(save_path, 'statistics')
    
    # Create output directory if it doesn't exist
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    # Define input/output file paths
    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')  # List of skeleton filenames
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')  # Output: processed data
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')  # Output: dropped frames

    # Set up logger to record frames that were dropped (missing body data)
    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log')))
    frames_drop_skes = dict()  # Dictionary to track dropped frames: {skeleton_name: [frame_indices]}

    # Run the main processing function
    get_raw_skes_data()

    # Save dropped frames information (redundant but ensures it's saved)
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, 4)
        
