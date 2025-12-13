# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging

# ============================================================================
# Configuration: Paths and Denoising Thresholds
# ============================================================================
# Paths for input/output files
root_path = './'
raw_data_file = osp.join(root_path, 'raw_data', 'raw_skes_data.pkl')  # Input: raw skeleton data
save_path = osp.join(root_path, 'denoised_data')  # Output directory

# Create output directories if they don't exist
if not osp.exists(save_path):
    os.mkdir(save_path)

rgb_ske_path = osp.join(save_path, 'rgb+ske')  # For RGB+skeleton visualization (if needed)
if not osp.exists(rgb_ske_path):
    os.mkdir(rgb_ske_path)

actors_info_dir = osp.join(save_path, 'actors_info')  # Store actor information logs
if not osp.exists(actors_info_dir):
    os.mkdir(actors_info_dir)

# ============================================================================
# Denoising Thresholds
# ============================================================================
missing_count = 0  # Global counter for sequences with missing frames

# Length threshold: Bodies appearing in <= 11 frames are considered noise
noise_len_thres = 11

# Spread threshold 1: For detecting noisy frames (X spread <= 0.8 * Y spread = valid)
noise_spr_thres1 = 0.8

# Spread threshold 2: Bodies with >= 69.754% noisy frames are filtered out
noise_spr_thres2 = 0.69754

# Motion thresholds: Bodies with motion outside [0.089925, 2.0] are filtered
# (Currently not used in main pipeline, but available for future use)
noise_mot_thres_lo = 0.089925  # Lower bound: very low motion = static noise
noise_mot_thres_hi = 2  # Upper bound: very high motion = tracking error

noise_len_logger = logging.getLogger('noise_length')
noise_len_logger.setLevel(logging.INFO)
noise_len_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_length.log')))
noise_len_logger.info('{:^20}\t{:^17}\t{:^8}\t{}'.format('Skeleton', 'bodyID', 'Motion', 'Length'))

noise_spr_logger = logging.getLogger('noise_spread')
noise_spr_logger.setLevel(logging.INFO)
noise_spr_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_spread.log')))
noise_spr_logger.info('{:^20}\t{:^17}\t{:^8}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion', 'Rate'))

noise_mot_logger = logging.getLogger('noise_motion')
noise_mot_logger.setLevel(logging.INFO)
noise_mot_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_motion.log')))
noise_mot_logger.info('{:^20}\t{:^17}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion'))

fail_logger_1 = logging.getLogger('noise_outliers_1')
fail_logger_1.setLevel(logging.INFO)
fail_logger_1.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_1.log')))

fail_logger_2 = logging.getLogger('noise_outliers_2')
fail_logger_2.setLevel(logging.INFO)
fail_logger_2.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_2.log')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes.log')))
missing_skes_logger.info('{:^20}\t{}\t{}'.format('Skeleton', 'num_frames', 'num_missing'))

missing_skes_logger1 = logging.getLogger('missing_frames_1')
missing_skes_logger1.setLevel(logging.INFO)
missing_skes_logger1.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_1.log')))
missing_skes_logger1.info('{:^20}\t{}\t{}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1',
                                                              'Actor2', 'Start', 'End'))

missing_skes_logger2 = logging.getLogger('missing_frames_2')
missing_skes_logger2.setLevel(logging.INFO)
missing_skes_logger2.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_2.log')))
missing_skes_logger2.info('{:^20}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1', 'Actor2'))


def denoising_by_length(ske_name, bodies_data):
    """
    Remove bodies (actors) that appear in too few frames - likely noise or tracking errors.
    
    Bodies that appear in <= 11 frames are considered noise and filtered out.
    This helps remove spurious detections that don't represent actual actors.
    
    Args:
        ske_name: Name of the skeleton sequence
        bodies_data: Dictionary mapping bodyID to body data (will be modified in-place)
    
    Returns:
        bodies_data: Modified dictionary with short-duration bodies removed
        noise_info: String describing which bodies were filtered out
    """
    noise_info = str()
    new_bodies_data = bodies_data.copy()  # Copy to avoid modification during iteration
    
    # Check each body's duration (number of frames it appears in)
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])  # Number of frames this body appears in
        
        # Filter out bodies that appear in too few frames (threshold: 11 frames)
        if length <= noise_len_thres:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            # Log the filtered body for analysis
            noise_len_logger.info('{}\t{}\t{:.6f}\t{:^6d}'.format(ske_name, bodyID,
                                                                  body_data['motion'], length))
            del bodies_data[bodyID]  # Remove from original dictionary
    
    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info


def get_valid_frames_by_spread(points):
    """
    Identify frames where the body pose is valid based on spatial spread analysis.
    
    This function detects frames where joints are spread abnormally (likely noise).
    A valid frame should have reasonable X-Y spread ratio. If X spread is too large
    compared to Y spread, it indicates the skeleton might be corrupted or misaligned.
    
    Args:
        points: Joint positions or color locations, shape (num_frames, 25, 2 or 3)
                Expected to have at least 2D coordinates (x, y)
    
    Returns:
        valid_frames: List of frame indices that pass the spread validation
    """
    num_frames = points.shape[0]
    valid_frames = []
    
    # Check each frame's joint spread
    for i in range(num_frames):
        x = points[i, :, 0]  # X coordinates of all joints in this frame
        y = points[i, :, 1]  # Y coordinates of all joints in this frame
        
        # Frame is valid if X spread <= 0.8 * Y spread
        # This filters out frames where joints are spread too wide horizontally (noise)
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    
    return valid_frames


def denoising_by_spread(ske_name, bodies_data):
    """
    Remove bodies with too many noisy frames based on spatial spread analysis.
    
    For each body, this function:
    1. Identifies frames with abnormal joint spread (using get_valid_frames_by_spread)
    2. Calculates the ratio of noisy frames
    3. If ratio >= 0.69754, removes the entire body (too noisy)
    4. Otherwise, recalculates motion using only valid frames
    
    This helps filter out bodies that are mostly corrupted or misdetected.
    
    Args:
        ske_name: Name of the skeleton sequence
        bodies_data: Dictionary mapping bodyID to body data (will be modified in-place)
                     Must contain at least 2 bodyIDs
    
    Returns:
        bodies_data: Modified dictionary with noisy bodies removed
        noise_info: String describing filtering actions taken
        denoised_by_spr: Boolean indicating if any body was filtered out
    """
    noise_info = str()
    denoised_by_spr = False  # Track if any body was filtered by this method

    new_bodies_data = bodies_data.copy()  # Copy to avoid modification during iteration
    
    # Process each body
    for (bodyID, body_data) in new_bodies_data.items():
        # Stop if only one body remains (need at least 2 for comparison)
        if len(bodies_data) == 1:
            break
        
        # Reshape joints to (num_frames, 25, 3) and find valid frames
        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)  # Number of noisy frames
        
        # Skip if no noisy frames found
        if num_noise == 0:
            continue

        # Calculate ratio of noisy frames
        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        
        # If too many frames are noisy, remove the entire body
        if ratio >= noise_spr_thres2:  # 0.69754 (69.754% threshold)
            del bodies_data[bodyID]
            denoised_by_spr = True
            noise_info += 'Filter out: %s (spread rate >= %.2f).\n' % (bodyID, noise_spr_thres2)
            noise_spr_logger.info('%s\t%s\t%.6f\t%.6f' % (ske_name, bodyID, motion, ratio))
        else:
            # Recalculate motion using only valid frames (motion might decrease)
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            noise_info += '%s: motion %.6f -> %.6f\n' % (bodyID, motion, body_data['motion'])
            # TODO: Consider removing noisy frames for each bodyID

    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info, denoised_by_spr


def denoising_by_motion(ske_name, bodies_data, bodies_motion):
    """
    Filter out bodies with motion values outside acceptable range.
    
    Bodies with very low motion (< 0.089925) are likely static noise.
    Bodies with very high motion (> 2.0) are likely tracking errors or outliers.
    The body with the largest motion is always kept (main actor).
    
    Note: This function is currently not used in the main pipeline (commented out).
    
    Args:
        ske_name: Name of the skeleton sequence
        bodies_data: Dictionary mapping bodyID to body data
        bodies_motion: Dictionary mapping bodyID to motion value
    
    Returns:
        denoised_bodies_data: List of tuples (bodyID, body_data) for valid bodies
        noise_info: String describing which bodies were filtered out
    """
    # Sort bodies by motion in descending order (highest motion first)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)

    # Always keep the body with the largest motion (main actor)
    denoised_bodies_data = [(bodies_motion[0][0], bodies_data[bodies_motion[0][0]])]
    noise_info = str()

    # Check remaining bodies against motion thresholds
    for (bodyID, motion) in bodies_motion[1:]:
        # Filter out bodies with motion outside acceptable range
        if (motion < noise_mot_thres_lo) or (motion > noise_mot_thres_hi):
            noise_info += 'Filter out: %s, %.6f (motion).\n' % (bodyID, motion)
            noise_mot_logger.info('{}\t{}\t{:.6f}'.format(ske_name, bodyID, motion))
        else:
            # Keep body if motion is within acceptable range
            denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
    
    if noise_info != '':
        noise_info += '\n'

    return denoised_bodies_data, noise_info


def denoising_bodies_data(bodies_data):
    """
    Apply a multi-stage denoising pipeline to filter out noisy or invalid body detections.
    
    The denoising process consists of:
    1. Filter by length: Remove bodies that appear in too few frames
    2. Filter by spread: Remove bodies with too many frames having abnormal joint spread
    3. Sort by motion: Order remaining bodies by motion amount (highest first)
    
    This helps identify the main actor(s) and remove tracking errors, static noise, etc.
    
    Args:
        bodies_data: Dictionary with keys 'name', 'data', 'num_frames'
                    'data' maps bodyID to body data dictionaries
    
    Returns:
        denoised_bodies_data: List of tuples (bodyID, body_data) sorted by motion (descending)
        noise_info: Combined string describing all filtering actions taken
    """
    ske_name = bodies_data['name']
    bodies_data = bodies_data['data']  # Extract the actual bodies dictionary

    # Step 1: Remove bodies that appear in too few frames (likely noise)
    bodies_data, noise_info_len = denoising_by_length(ske_name, bodies_data)

    # If only one body remains, return early (no need for further denoising)
    if len(bodies_data) == 1:
        return bodies_data.items(), noise_info_len

    # Step 2: Remove bodies with too many frames having abnormal spatial spread
    bodies_data, noise_info_spr, denoised_by_spr = denoising_by_spread(ske_name, bodies_data)

    # If only one body remains after spread filtering, return
    if len(bodies_data) == 1:
        return bodies_data.items(), noise_info_len + noise_info_spr

    # Step 3: Sort remaining bodies by motion amount (highest motion = main actor)
    bodies_motion = dict()
    for (bodyID, body_data) in bodies_data.items():
        bodies_motion[bodyID] = body_data['motion']
    
    # Sort by motion in descending order
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    
    # Build list of (bodyID, body_data) tuples sorted by motion
    denoised_bodies_data = list()
    for (bodyID, _) in bodies_motion:
        denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

    return denoised_bodies_data, noise_info_len + noise_info_spr

    # TODO: Consider denoising further by integrating motion method
    # Note: Motion-based filtering is currently disabled but could be added here


def get_one_actor_points(body_data, num_frames):
    """
    Convert single actor's body data into frame-aligned arrays.
    
    This function takes body data (which may only span a subset of frames) and creates
    arrays aligned to the full sequence length, filling missing frames with zeros/NaN.
    
    Args:
        body_data: Dictionary containing 'joints', 'colors', and 'interval' for one actor
        num_frames: Total number of frames in the sequence
    
    Returns:
        joints: Array of shape (num_frames, 75) - flattened 3D positions (25 joints × 3 coords)
                Missing frames are filled with zeros
        colors: Array of shape (num_frames, 1, 25, 2) - 2D color locations
                Missing frames are filled with NaN
    """
    # Initialize arrays: joints filled with zeros, colors filled with NaN
    joints = np.zeros((num_frames, 75), dtype=np.float32)  # 25 joints × 3 coords = 75
    colors = np.ones((num_frames, 1, 25, 2), dtype=np.float32) * np.nan
    
    # Get the frame range where this actor appears
    start, end = body_data['interval'][0], body_data['interval'][-1]
    
    # Fill in the frames where actor data exists
    # Reshape joints from (num_frames × 25, 3) to (num_frames, 75)
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
    colors[start:end + 1, 0] = body_data['colors']  # Shape: (num_frames, 25, 2)

    return joints, colors


def remove_missing_frames(ske_name, joints, colors):
    """
    Remove frames where all joint positions are zero (missing data).
    
    For sequences with 2 actors, this function also logs detailed statistics about
    which actor has missing frames (for debugging and analysis).
    
    Args:
        ske_name: Name of the skeleton sequence
        joints: Array of shape (num_frames, 75 or 150) - joint positions
        colors: Array of shape (num_frames, num_bodies, 25, 2) - color locations
    
    Returns:
        joints: Filtered joints array with missing frames removed
        colors: Colors array with missing frames marked as NaN (not removed to preserve shape)
    """
    num_frames = joints.shape[0]
    num_bodies = colors.shape[1]  # 1 or 2

    # For 2-actor sequences, log detailed missing frame statistics
    if num_bodies == 2:
        # Find frames where actor1's joints are all zeros (missing)
        missing_indices_1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
        # Find frames where actor2's joints are all zeros (missing)
        missing_indices_2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_indices_1)
        cnt2 = len(missing_indices_2)

        # Check if missing frames are at start/end (for logging)
        start = 1 if 0 in missing_indices_1 else 0
        end = 1 if num_frames - 1 in missing_indices_1 else 0
        
        # Log detailed statistics if any missing frames found
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                # Actor1 has more missing frames - log with start/end info
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(ske_name, num_frames,
                                                                            cnt1, cnt2, start, end)
                missing_skes_logger1.info(info)
            else:
                # Actor2 has more missing frames - log basic info
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2)
                missing_skes_logger2.info(info)

    # Find frames where ALL joints are zero (both actors missing in 2-actor case)
    # These are frames that should be completely removed
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # Frames with at least some data
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]  # Frames with all zeros
    num_missing = len(missing_indices)

    # Remove missing frames from joints, mark as NaN in colors
    if num_missing > 0:
        joints = joints[valid_indices]  # Remove missing frames
        colors[missing_indices] = np.nan  # Mark missing frames in colors (preserve shape)
        global missing_count
        missing_count += 1  # Increment global counter
        missing_skes_logger.info('{}\t{:^10d}\t{:^11d}'.format(ske_name, num_frames, num_missing))

    return joints, colors


def get_bodies_info(bodies_data):
    """
    Generate a formatted string with information about all bodies in a sequence.
    
    This is used for logging and debugging purposes to track which bodies were detected
    and their properties (frame intervals and motion amounts).
    
    Args:
        bodies_data: Dictionary mapping bodyID to body data
    
    Returns:
        bodies_info: Formatted string with body information (for logging)
    """
    bodies_info = '{:^17}\t{}\t{:^8}\n'.format('bodyID', 'Interval', 'Motion')
    for (bodyID, body_data) in bodies_data.items():
        start, end = body_data['interval'][0], body_data['interval'][-1]
        bodies_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), body_data['motion'])

    return bodies_info + '\n'


def get_two_actors_points(bodies_data):
    """
    Extract and organize joints/colors for up to 2 main actors from a skeleton sequence.
    
    This function:
    1. Applies denoising to filter out noise bodies
    2. Selects the main actor(s) based on motion (highest motion = actor1)
    3. Assigns remaining bodies to actor1 or actor2 slots based on temporal overlap
    4. Handles edge cases (e.g., only 1 actor after denoising, wrong number of actors)
    
    The NTU dataset has 60 action classes:
    - Classes 1-49: Single-person actions
    - Classes 50-60: Two-person interactions
    
    Args:
        bodies_data: Dictionary with keys 'name', 'data', 'num_frames'
                    'data' maps bodyID to body data dictionaries
    
    Returns:
        joints: Array of shape (num_frames, 150) - flattened positions for 2 actors
                First 75 values = actor1, last 75 values = actor2
        colors: Array of shape (num_frames, 2, 25, 2) - color locations for 2 actors
    """
    ske_name = bodies_data['name']
    label = int(ske_name[-2:])  # Extract action class label (last 2 digits)
    num_frames = bodies_data['num_frames']
    bodies_info = get_bodies_info(bodies_data['data'])  # Get initial body info for logging

    # Apply denoising pipeline to filter out noise bodies
    bodies_data, noise_info = denoising_bodies_data(bodies_data)
    bodies_info += noise_info  # Append denoising info to log

    bodies_data = list(bodies_data)  # Convert to list for easier manipulation
    
    if len(bodies_data) == 1:
        # Only one actor remains after denoising
        # This is expected for single-person actions, but unexpected for two-person actions
        if label >= 50:  # Two-person action class but only 1 actor detected
            fail_logger_2.info(ske_name)  # Log as denoising failure

        # Extract single actor data
        bodyID, body_data = bodies_data[0]
        joints, colors = get_one_actor_points(body_data, num_frames)
        bodies_info += 'Main actor: %s' % bodyID
    else:
        # Multiple actors detected - need to assign to actor1 and actor2 slots
        # This is expected for two-person actions, but unexpected for single-person actions
        if label < 50:  # Single-person action class but multiple actors detected
            fail_logger_1.info(ske_name)  # Log as denoising failure

        # Initialize arrays for 2 actors: 150 dims total (75 per actor)
        joints = np.zeros((num_frames, 150), dtype=np.float32)
        colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan

        # Get actor1 (body with highest motion - main actor)
        bodyID, actor1 = bodies_data[0]
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)  # First 75 dims
        colors[start1:end1 + 1, 0] = actor1['colors']
        actor1_info = '{:^17}\t{}\t{:^8}\n'.format('Actor1', 'Interval', 'Motion') + \
                      '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start1, end1]), actor1['motion'])
        del bodies_data[0]  # Remove from list

        # Initialize actor2 info
        actor2_info = '{:^17}\t{}\t{:^8}\n'.format('Actor2', 'Interval', 'Motion')
        start2, end2 = [0, 0]  # Virtual initial interval for actor2

        # Assign remaining bodies to actor1 or actor2 based on temporal overlap
        while len(bodies_data) > 0:
            bodyID, actor = bodies_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            
            # Check if this body overlaps with actor1
            if min(end1, end) - max(start1, start) <= 0:  # No temporal overlap with actor1
                # Assign to actor1 slot (they don't overlap, so can share the slot)
                joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 0] = actor['colors']
                actor1_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Expand actor1's interval to cover this body
                start1 = min(start, start1)
                end1 = max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:  # No overlap with actor2
                # Assign to actor2 slot (last 75 dims)
                joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 1] = actor['colors']
                actor2_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Expand actor2's interval
                start2 = min(start, start2)
                end2 = max(end, end2)
            # If body overlaps with both, it's skipped (shouldn't happen after denoising)
            del bodies_data[0]

        bodies_info += ('\n' + actor1_info + '\n' + actor2_info)

    # Save actor information to file for debugging/analysis
    with open(osp.join(actors_info_dir, ske_name + '.txt'), 'w') as fw:
        fw.write(bodies_info + '\n')

    return joints, colors


def get_raw_denoised_data(batch_size=1000):
    """
    Main function: Process raw skeleton data through denoising pipeline and save results.
    
    This function processes data in batches to reduce memory usage:
    1. Loads raw skeleton data from pickle file (created by get_raw_skes_data.py)
    2. Processes sequences in batches:
       - Applies denoising to filter out noise bodies
       - Extracts joints and colors for 1-2 main actors
       - Removes frames with missing data
    3. Saves denoised data incrementally to pickle files
    
    Args:
        batch_size: Number of sequences to process before saving intermediate results (default: 1000)
    
    Data Format:
    - Joints: Each frame is a 150-dim vector (75 dims per actor)
              - Single actor: first 75 dims contain data, last 75 are zeros
              - Two actors: first 75 = actor1, last 75 = actor2
              - Each 75-dim vector = 25 joints × 3 coords (x, y, z) flattened
    - Colors: Each frame has shape (num_bodies, 25, 2) - 2D color locations
    
    Output Files:
    - raw_denoised_joints.pkl: List of joint arrays (one per skeleton)
    - raw_denoised_colors.pkl: List of color arrays (one per skeleton)
    - frames_cnt.txt: Frame count for each skeleton
    - actors_info/*.txt: Detailed actor information for each skeleton
    - Various log files: noise detection, missing frames, denoising failures
    """

    # Load raw skeleton data created by get_raw_skes_data.py
    print('Loading raw skeleton data...')
    with open(raw_data_file, 'rb') as fr:
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print('Found %d available skeleton sequences.' % num_skes)
    print(f'Processing in batches of {batch_size} sequences to reduce memory usage.')

    # Define output file paths
    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    raw_skes_colors_pkl = osp.join(save_path, 'raw_denoised_colors.pkl')
    frames_cnt_file = osp.join(save_path, 'frames_cnt.txt')

    # Initialize storage for processed data (will be cleared after each batch)
    raw_denoised_joints = []
    raw_denoised_colors = []
    frames_cnt = []

    # Process sequences in batches
    for batch_start in range(0, num_skes, batch_size):
        batch_end = min(batch_start + batch_size, num_skes)
        batch_num = batch_start // batch_size + 1
        total_batches = (num_skes + batch_size - 1) // batch_size
        
        print(f'\n=== Processing Batch {batch_num}/{total_batches}: sequences {batch_start} to {batch_end-1} ===')
        
        # Process each skeleton sequence in this batch
        for idx in range(batch_start, batch_end):
            bodies_data = raw_skes_data[idx]
            ske_name = bodies_data['name']
            
            if (idx + 1) % 500 == 0 or idx == batch_start:
                print(f'  Processing {idx+1}/{num_skes}: {ske_name}')
            
            num_bodies = len(bodies_data['data'])

            if num_bodies == 1:
                # Simple case: only one actor detected (no denoising needed)
                num_frames = bodies_data['num_frames']
                body_data = list(bodies_data['data'].values())[0]
                joints, colors = get_one_actor_points(body_data, num_frames)
            else:
                # Multiple actors: apply denoising and select main actors
                joints, colors = get_two_actors_points(bodies_data)
                # Remove frames where all joints are zero (missing data)
                joints, colors = remove_missing_frames(ske_name, joints, colors)
                num_frames = joints.shape[0]  # Update frame count after removal

            # Store processed data for this batch
            raw_denoised_joints.append(joints)
            raw_denoised_colors.append(colors)
            frames_cnt.append(num_frames)

        # Save batch results incrementally
        print(f'  Saving batch {batch_num} results...')
        
        # Load existing data if file exists, otherwise start fresh
        if batch_start == 0:
            # First batch: create new files
            all_joints = raw_denoised_joints.copy()
            all_colors = raw_denoised_colors.copy()
            all_frames_cnt = frames_cnt.copy()
        else:
            # Subsequent batches: load existing and append
            try:
                with open(raw_skes_joints_pkl, 'rb') as f:
                    all_joints = pickle.load(f)
                with open(raw_skes_colors_pkl, 'rb') as f:
                    all_colors = pickle.load(f)
                all_frames_cnt = np.loadtxt(frames_cnt_file, dtype=int).tolist()
                
                # Append new batch data
                all_joints.extend(raw_denoised_joints)
                all_colors.extend(raw_denoised_colors)
                all_frames_cnt.extend(frames_cnt)
            except FileNotFoundError:
                # If files don't exist, start fresh
                all_joints = raw_denoised_joints.copy()
                all_colors = raw_denoised_colors.copy()
                all_frames_cnt = frames_cnt.copy()
        
        # Save accumulated data
        try:
            with open(raw_skes_joints_pkl, 'wb') as f:
                pickle.dump(all_joints, f, protocol=4)
            with open(raw_skes_colors_pkl, 'wb') as f:
                pickle.dump(all_colors, f, protocol=4)
            np.savetxt(frames_cnt_file, np.array(all_frames_cnt, dtype=int), fmt='%d')
        except Exception as e:
            print(f"Error saving batch {batch_num}: {e}")
            raise
        
        # Clear batch data from memory
        del raw_denoised_joints, raw_denoised_colors, frames_cnt
        del all_joints, all_colors, all_frames_cnt
        
        # Reinitialize for next batch
        raw_denoised_joints = []
        raw_denoised_colors = []
        frames_cnt = []
        
        print(f'  Batch {batch_num} completed. Progress: {100.0 * batch_end / num_skes:.2f}%')
        print(f'  Missing count so far: {missing_count}')

    # Final summary
    print('\n=== Processing Complete ===')
    final_frames_cnt = np.loadtxt(frames_cnt_file, dtype=int)
    print('Saved raw denoised positions of {} frames into {}'.format(
        np.sum(final_frames_cnt), raw_skes_joints_pkl))
    print('Found %d files that have missing data' % missing_count)

if __name__ == '__main__':
    # Process in batches of 1000 sequences to reduce memory usage
    # Adjust batch_size based on available RAM (smaller = less memory, larger = faster)
    get_raw_denoised_data(batch_size=1000)
