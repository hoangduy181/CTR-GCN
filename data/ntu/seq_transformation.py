# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging
import h5py
from sklearn.model_selection import train_test_split

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'


if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]

def seq_translation(skes_joints, batch_start=0, batch_end=None):
    """
    Translate sequences to origin (joint-2) for each skeleton.
    
    Args:
        skes_joints: List of joint arrays to process
        batch_start: Starting index for batch processing (default: 0)
        batch_end: Ending index for batch processing (default: None = all)
    
    Returns:
        Modified skes_joints list
    """
    if batch_end is None:
        batch_end = len(skes_joints)
    
    for idx in range(batch_start, batch_end):
        ske_joints = skes_joints[idx]
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def frame_translation(skes_joints, skes_name, frames_cnt, batch_start=0, batch_end=None):
    """
    Translate frames to origin and normalize by spine distance.
    
    Args:
        skes_joints: List of joint arrays to process
        skes_name: List of skeleton names
        frames_cnt: Array of frame counts (will be modified)
        batch_start: Starting index for batch processing (default: 0)
        batch_end: Ending index for batch processing (default: None = all)
    
    Returns:
        Modified skes_joints list and frames_cnt array
    """
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    if batch_start == 0:
        nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    if batch_end is None:
        batch_end = len(skes_joints)

    for idx in range(batch_start, batch_end):
        ske_joints = skes_joints[idx]
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                      dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                 dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # 300
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 60))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints, label, performer, camera, evaluation, save_path):
    try:
        train_indices, test_indices = get_indices(performer, camera, evaluation)
        m = 'sklearn'  # 'sklearn' or 'numpy'
        # Select validation set from training set
        # train_indices, val_indices = split_train_val(train_indices, m)

        # Save labels and num_frames for each sequence of each data set
        train_labels = label[train_indices]
        test_labels = label[test_indices]

        train_x = skes_joints[train_indices]
        train_y = one_hot_vector(train_labels)
        test_x = skes_joints[test_indices]
        test_y = one_hot_vector(test_labels)

        save_name = 'NTU60_%s.npz' % evaluation
        np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
    except Exception as e:
        print(f"Error saving dataset: {e}")
        raise
    # Save data into a .h5 file
    # h5file = h5py.File(osp.join(save_path, 'NTU_%s.h5' % (evaluation)), 'w')
    # Training set
    # h5file.create_dataset('x', data=skes_joints[train_indices])
    # train_one_hot_labels = one_hot_vector(train_labels)
    # h5file.create_dataset('y', data=train_one_hot_labels)
    # Validation set
    # h5file.create_dataset('valid_x', data=skes_joints[val_indices])
    # val_one_hot_labels = one_hot_vector(val_labels)
    # h5file.create_dataset('valid_y', data=val_one_hot_labels)
    # Test set
    # h5file.create_dataset('test_x', data=skes_joints[test_indices])
    # test_one_hot_labels = one_hot_vector(test_labels)
    # h5file.create_dataset('test_y', data=test_one_hot_labels)

    # h5file.close()

# def split_dataset(skes_joints, label, performer, camera, evaluation, save_path, batch_size=500):
#     """
#     Split dataset into train/test sets and save incrementally to avoid memory issues.
    
#     Args:
#         skes_joints: Array of skeleton joints (num_skes, max_frames, 150)
#         label: Array of labels
#         performer: Array of performer IDs
#         camera: Array of camera IDs
#         evaluation: 'CS' or 'CV'
#         save_path: Path to save output
#         batch_size: Number of sequences to process at once (default: 500)
#     """
#     try:
#         train_indices, test_indices = get_indices(performer, camera, evaluation)
#         print(f"    Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
        
#         # Get labels (small arrays, safe to load all at once)
#         train_labels = label[train_indices]
#         test_labels = label[test_indices]
#         train_y = one_hot_vector(train_labels)
#         test_y = one_hot_vector(test_labels)

#         save_name = 'NTU60_%s.npz' % evaluation
        
#         # Process train data in batches
#         print(f"    Processing train data in batches of {batch_size}...")
#         train_batches = []
#         for batch_start in range(0, len(train_indices), batch_size):
#             batch_end = min(batch_start + batch_size, len(train_indices))
#             batch_indices = train_indices[batch_start:batch_end]
#             batch_x = skes_joints[batch_indices]
#             train_batches.append(batch_x)
#             if (batch_start // batch_size + 1) % 10 == 0:
#                 print(f"      Processed {batch_end}/{len(train_indices)} train samples")
        
#         # Concatenate train batches
#         print(f"    Concatenating train batches...")
#         train_x = np.concatenate(train_batches, axis=0)
#         del train_batches
        
#         # Process test data in batches
#         print(f"    Processing test data in batches of {batch_size}...")
#         test_batches = []
#         for batch_start in range(0, len(test_indices), batch_size):
#             batch_end = min(batch_start + batch_size, len(test_indices))
#             batch_indices = test_indices[batch_start:batch_end]
#             batch_x = skes_joints[batch_indices]
#             test_batches.append(batch_x)
#             if (batch_start // batch_size + 1) % 10 == 0:
#                 print(f"      Processed {batch_end}/{len(test_indices)} test samples")
        
#         # Concatenate test batches
#         print(f"    Concatenating test batches...")
#         test_x = np.concatenate(test_batches, axis=0)
#         del test_batches
        
#         # Save with compression to reduce file size and memory usage
#         print(f"    Saving to {save_name}...")
#         np.savez_compressed(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
        
#         # Clean up
#         del train_x, test_x
        
#     except Exception as e:
#         print(f"Error saving dataset: {e}")
#         raise

def get_indices(performer, camera, evaluation='CS'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                    24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    else:  # Cross View (Camera IDs)
        train_ids = [2, 3]
        test_ids = 1
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)

    return train_indices, test_indices


if __name__ == '__main__':
    """
    Main execution: Process skeleton joints in batches to reduce memory usage.
    
    This script:
    1. Loads skeleton joints from pickle file
    2. Processes sequences in batches (translation, alignment)
    3. Splits dataset into train/test sets for different evaluations
    """
    # Configuration
    batch_size = 2000  # Process sequences in batches to reduce memory usage
    
    # Load metadata files
    camera = np.loadtxt(camera_file, dtype=int)  # camera id: 1, 2, 3
    performer = np.loadtxt(performer_file, dtype=int)  # subject id: 1~40
    label = np.loadtxt(label_file, dtype=int) - 1  # action label: 0~59
    frames_cnt = np.loadtxt(frames_file, dtype=int)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)
    
    print("=== Loading skeleton joints from:", raw_skes_joints_pkl)
    try:
        with open(raw_skes_joints_pkl, 'rb') as fr:
            skes_joints = pickle.load(fr)  # a list
    except (pickle.UnpicklingError, EOFError, ImportError) as e:
        print(f"Error loading pickle file: {e}")
        raise
    
    num_skes = len(skes_joints)
    print(f"Loaded {num_skes} skeleton sequences")
    print(f"Processing in batches of {batch_size} sequences to reduce memory usage")
    
    # Process sequences in batches
    total_batches = (num_skes + batch_size - 1) // batch_size
    
    # Step 1: Sequence translation (translate to origin)
    print("\n=== Step 1: Sequence Translation ===")
    for batch_start in range(0, num_skes, batch_size):
        batch_end = min(batch_start + batch_size, num_skes)
        batch_num = batch_start // batch_size + 1
        print(f"  Processing batch {batch_num}/{total_batches}: sequences {batch_start} to {batch_end-1}")
        skes_joints = seq_translation(skes_joints, batch_start=batch_start, batch_end=batch_end)
    
    # Step 2: Align frames to same length
    print("\n=== Step 2: Frame Alignment ===")
    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length
    print(f"Aligned all sequences to max frame length: {frames_cnt.max()}")
    
    # Step 3: Split dataset for different evaluations
    print("\n=== Step 3: Dataset Splitting ===")
    evaluations = ['CS', 'CV']
    for evaluation in evaluations:
        print(f"  Splitting dataset for {evaluation} evaluation...")
        split_dataset(skes_joints, label, performer, camera, evaluation, save_path)
        print(f"  Saved NTU60_{evaluation}.npz")
    
    print("\n=== Processing Complete ===")
