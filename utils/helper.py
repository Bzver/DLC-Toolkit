import numpy as np
from typing import List, Tuple

def get_instances_on_current_frame(pred_data_array:np.ndarray, current_frame_idx:int) -> List[int]:
    """
    Identifies which instances are present in a given frame based on non-NaN keypoint data.

    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence) for each keypoint.
        current_frame_idx (int): Index of the frame to check.

    Returns:
        List[int]: List of instance indices that have at least one valid keypoint 
                   in the specified frame.
    """
    instance_count = pred_data_array.shape[1]
    current_frame_inst = []
    for inst_idx in range(instance_count):
        if np.any(~np.isnan(pred_data_array[current_frame_idx, inst_idx, :])):
            current_frame_inst.append(inst_idx)
    return current_frame_inst

def get_instance_count_per_frame(pred_data_array:np.ndarray) -> np.ndarray:
    """
    Count the number of non-empty instances per frame.
    
    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence) for each keypoint.
    
    Returns:
        Array of shape (n_frames,) with count of valid instances per frame.
    """
    non_empty_instance_numerical = (np.any(~np.isnan(pred_data_array), axis=2)) * 1
    instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
    return instance_count_per_frame

#########################################################################################################################################################1

def infer_head_tail_indices(keypoint_names:List[str]) -> Tuple[int,int]:
    """
    Infer head and tail keypoint indices from keypoint names with robust handling
    of capitalization, underscores, and common anatomical naming patterns.
    
    Args:
        keypoint_names: list of all the keypoint names

    Returns:
        idx of supposed head and tail keypoint
    """
    # Define priority-ordered keywords (lowercase, without underscores)
    head_keywords_priority = [
        'nose',
        'head',
        'forehead',
        'front',
        'snout',
        'face',
        'mouth',
        'muzzle',
        'spinF',
        'neck',
        'eye',
        'ear',
        'cheek',
        'chin',
        'anterior',
    ]
    tail_keywords_priority = [
        'tailbase',
        'base_tail',
        'tail_base',
        'butt',
        'hip',
        'rump',
        'thorax',
        'ass',
        'pelvis',
        'tail',
        'spineM',
        'cent',
        'posterior',
        'back',
    ]

    def normalize(name): # Normalize keypoint names: lowercase, remove non-alphanumeric, collapse underscores
        return ''.join(c.lower() for c in name if c.isalnum())

    normalized_names = [normalize(name) for name in keypoint_names]

    # Search with priority: return first match in priority list
    head_idx = None
    for kw in head_keywords_priority:
        normalized_kw = normalize(kw)
        for idx, norm_name in enumerate(normalized_names):
            if normalized_kw in norm_name:
                head_idx = idx
                break
        if head_idx is not None:
            break

    tail_idx = None
    for kw in tail_keywords_priority:
        normalized_kw = normalize(kw)
        for idx, norm_name in enumerate(normalized_names):
            if normalized_kw in norm_name:
                tail_idx = idx
                break
        if tail_idx is not None:
            break

    if head_idx is None:
        print("Warning: Could not infer head keypoint from keypoint names.")
    if tail_idx is None:
        print("Warning: Could not infer tail keypoint from keypoint names.")

    return head_idx, tail_idx

def build_angle_map(canon_pose:np.ndarray, all_frame_poses:np.ndarray , head_idx:int, tail_idx:int) -> dict:
    canonical_vec = canon_pose[head_idx] - canon_pose[tail_idx]
    num_keypoint = canon_pose.shape[0]
    if np.linalg.norm(canonical_vec) < 1e-6:
        canonical_body_angle = 0.0
    else:
        canonical_body_angle = np.arctan2(canonical_vec[1], canonical_vec[0])

    # Build angle map for every possible connection
    angle_map = []  # (i, j, expected_offset, weight)
    all_angles = np.arctan2(all_frame_poses[:, 1::2], all_frame_poses[:, 0::2])  # (N, K)

    for i in range(num_keypoint):
        for j in range(num_keypoint):
            if i == j:
                continue

            # Vector from i to j in canon pose
            vec = canon_pose[j] - canon_pose[i]
            if np.linalg.norm(vec) < 1e-6:
                continue

            # Expected angle of this vector
            raw_angle = np.arctan2(vec[1], vec[0])

            # Offset relative to canonical body angle
            offset = np.arctan2(
                np.sin(raw_angle - canonical_body_angle),
                np.cos(raw_angle - canonical_body_angle)
            )  # Wrap to [-π, π]

            # Measure angular variation (in radians)
            ij_angles = all_angles[:, j] - all_angles[:, i]  # (N,)
            ij_angles = np.arctan2(np.sin(ij_angles), np.cos(ij_angles))  # Unwrap
            var = np.nanvar(ij_angles)

            # Weight: high if stable and aligned with body
            length = np.linalg.norm(vec)
            stability = 1.0 / (1.0 + var) if var > 0 else 1.0
            alignment = abs(np.dot(vec / np.linalg.norm(vec), canonical_vec / np.linalg.norm(canonical_vec)))

            weight = length * stability * alignment

            angle_map.append({"i": i, "j": j,"offset": offset,"weight": weight})

    # Sort by weight (most reliable first)
    angle_map.sort(key=lambda x: x["weight"], reverse=True)
    
    angle_map_data = {"head_idx": head_idx, "tail_idx": tail_idx, "angle_map": angle_map}

    return angle_map_data

#########################################################################################################################################################1

def log_print(*args, **kwargs):
    try:
        log_file = "D:/Project/debug_log.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs)
    except:
        pass

#########################################################################################################################################################1

def clean_inconsistent_nans(pred_data_array:np.ndarray):
    print("Cleaning up NaN keypoints that somehow has confidence value...")
    nan_mask = np.isnan(pred_data_array)
    x_is_nan = nan_mask[:, :, 0::3]
    y_is_nan = nan_mask[:, :, 1::3]
    keypoints_to_fully_nan = x_is_nan | y_is_nan
    full_nan_sweep_mask = np.repeat(keypoints_to_fully_nan, 3, axis=-1)
    pred_data_array[full_nan_sweep_mask] = np.nan
    print("NaN keypoint confidence cleaned.")
    return pred_data_array