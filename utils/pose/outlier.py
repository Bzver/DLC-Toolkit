import numpy as np
from itertools import combinations
from typing import Tuple, Dict

from .pose_analysis import calculate_pose_centroids, calculate_pose_bbox, calculate_pose_rotations
from .pose_worker import pose_alignment_worker
from utils.helper import get_instances_on_current_frame, bye_bye_runtime_warning


def outlier_removal(pred_data_array:np.ndarray, outlier_mask:np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Removes outliers from the prediction data by setting flagged values to NaN.

    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence).
        outlier_mask (np.ndarray): Boolean mask of shape (num_frames, num_instances) indicating 
            which instance-frame pairs are outliers.

    Returns:
        Tuple[np.ndarray, int, int]:
            - Modified prediction array with outliers set to NaN.
            - Number of frames that contained at least one outlier.
            - Total number of instance outliers removed.
    """
    removed_frames_count = np.sum(np.any(outlier_mask, axis=1))
    removed_instances_count = np.sum(outlier_mask)
    pred_data_array[outlier_mask] = np.nan
    return pred_data_array, removed_frames_count, removed_instances_count

########################################################################################################

def outlier_confidence(pred_data_array:np.ndarray, threshold:float=0.5) -> np.ndarray:
    """
    Flags instances with average keypoint confidence below a given threshold.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        threshold (float): Minimum average confidence for an instance to be considered valid.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates low confidence.
    """
    confidence_scores = pred_data_array[:, :, 2::3]
    with bye_bye_runtime_warning():
        inst_conf = np.nanmean(confidence_scores, axis=2)
    low_conf_mask = inst_conf < threshold
    return low_conf_mask

def outlier_bodypart(pred_data_array:np.ndarray, threshold:int=2) -> np.ndarray:
    """
    Flags instances that have fewer detected keypoints than the specified threshold.

    Only considers instances with at least one valid keypoint (excludes completely missing ones).

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        threshold (int): Minimum number of visible keypoints required.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates too few body parts.
    """
    x_coords_mask = ~np.isnan(pred_data_array[:, :, 0::3])
    y_coords_mask = ~np.isnan(pred_data_array[:, :, 1::3])

    discovered_bodyparts = np.sum(np.logical_and(x_coords_mask, y_coords_mask), axis=2)
    low_bp_mask = np.logical_and(discovered_bodyparts < threshold,
        discovered_bodyparts > 0) # No need to include empty instances
    
    return low_bp_mask

def outlier_duplicate(
        pred_data_array:np.ndarray,
        bp_threshold:float=0.5,
        dist_threshold:float=10.0
        ) -> np.ndarray:
    """
    Detects duplicate detections by identifying instances whose keypoints are abnormally close.

    Compares all pairs of instances. An instance is flagged if a large proportion of its keypoints 
    are within a small distance of another instance's keypoints. The lower-confidence or smaller 
    instance is marked as the duplicate.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        bp_threshold (float): Proportion of overlapping keypoints required to flag duplicates.
        dist_threshold (float): Maximum distance (in pixels) for two keypoints to be considered overlapping.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates a duplicate instance.
    """
    total_frames, instance_count, _ = pred_data_array.shape
    instances = list(range(instance_count))
    duplicate_mask = np.zeros((total_frames, instance_count), dtype=bool)

    for inst_1_idx, inst_2_idx in combinations(instances, 2):
        inst_1_x = pred_data_array[:, inst_1_idx, 0::3]
        inst_1_y = pred_data_array[:, inst_1_idx, 1::3]
        inst_2_x = pred_data_array[:, inst_2_idx, 0::3]
        inst_2_y = pred_data_array[:, inst_2_idx, 1::3]

        if np.all(np.isnan(inst_1_x)) or np.all(np.isnan(inst_2_x)):
            continue

        with bye_bye_runtime_warning():
            inst_1_conf = np.nanmean(pred_data_array[:, inst_1_idx, 2::3], axis=1)
            inst_2_conf = np.nanmean(pred_data_array[:, inst_2_idx, 2::3], axis=1)
        inst_1_valid_kp = np.sum(~np.isnan(inst_1_x), axis=1)
        inst_2_valid_kp = np.sum(~np.isnan(inst_2_x), axis=1)

        euclidean_dist = np.sqrt((inst_1_x - inst_2_x)**2 + (inst_1_y - inst_2_y)**2)

        close_kp = np.sum(euclidean_dist <= dist_threshold, axis=1)
        suspect_list = np.where(close_kp > 0)[0].tolist()

        for frame_idx in suspect_list:
            if close_kp[frame_idx] < min(inst_1_valid_kp[frame_idx],inst_2_valid_kp[frame_idx]) * bp_threshold:
                continue
            elif close_kp[frame_idx] > max(inst_1_valid_kp[frame_idx],inst_2_valid_kp[frame_idx]) * bp_threshold:
                inst_to_mark = inst_2_idx if inst_1_conf[frame_idx] >= inst_2_conf[frame_idx] else inst_1_idx
            else:
                inst_to_mark = inst_2_idx if inst_1_valid_kp[frame_idx] > inst_2_valid_kp[frame_idx] else inst_1_idx
            
            duplicate_mask[frame_idx, inst_to_mark] = True
            
    return duplicate_mask

def outlier_size(
        pred_data_array:np.ndarray,
        canon_pose:np.ndarray,
        min_ratio:float=0.3,
        max_ratio:float=2.5
        ) -> np.ndarray:
    """
    Flags instances whose size (mean distance of keypoints from centroid) is significantly different 
    from the canonical pose size.

    Helps detect implausible detections (e.g., extremely small or large animal poses).

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        canon_pose (np.ndarray): Canonical 2D pose of shape (num_keypoints, 2) used as size reference.
        min_ratio (float): Instances smaller than this ratio of the canonical size are flagged.
        max_ratio (float): Instances larger than this ratio of the canonical size are flagged.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates abnormal size.
    """
    total_frames, instance_count, _ = pred_data_array.shape
    canon_radius = np.mean(np.sqrt(canon_pose[:, 0]**2 + canon_pose[:, 1]**2))
    _, local_coords = calculate_pose_centroids(pred_data_array)
    inst_radius = np.full((total_frames, instance_count), np.nan)
    for inst_idx in range(instance_count):
        
        with bye_bye_runtime_warning():
            inst_radius[:, inst_idx] = np.nanmean(
                np.sqrt(local_coords[:, inst_idx, 0::2]**2 + local_coords[:, inst_idx, 1::2]**2), axis=1)
        
    small_mask = inst_radius < canon_radius * min_ratio
    large_mask = inst_radius >= canon_radius * max_ratio

    return np.logical_or(small_mask, large_mask)

def outlier_pose(
    pred_data_array: np.ndarray,
    angle_map_data: Dict[str, any],
    quant_step: float = 1.0,
    min_samples: int = 3
) -> np.ndarray:
    """
    Flags pose outliers using pose quantization and frequency counting across all frames and instances.

    Args:
        pred_data_array (np.ndarray): Shape (F, I, 3*K)
        angle_map_data : Dict with 'head_idx', 'tail_idx', 'angle_map'.
            - angle map: list of (i, j, offset, weight) of each connection
        quant_step (float): Grid spacing for quantization (e.g., 0.25). Smaller = stricter.
        min_samples (int): Minimum number of occurrences for a quantized pose to be considered valid.
            Poses appearing fewer than this threshold are flagged as outliers.

    Returns:
        np.ndarray: Boolean mask (F, I) where True = outlier pose.
    """
    angle_map = angle_map_data["angle_map"]
    num_kp = max(max(entry["i"], entry["j"]) for entry in angle_map) + 1

    scores = np.zeros(num_kp)
    for entry in angle_map:
        i, j, w = entry["i"], entry["j"], entry["weight"]
        scores[i] += w
        scores[j] += w

    top_indices = np.argsort(scores)[::-1][:6]

    _, local_coords = calculate_pose_centroids(pred_data_array) # (F, I, K*2)
    local_coords = np.nan_to_num(local_coords, nan=0.0)

    local_x_all, local_y_all = local_coords[:,:,0::2], local_coords[:,:,1::2]

    local_max = np.max(np.sqrt(local_x_all**2 + local_y_all**2), axis=2) # (F, I)
    outlier_mask = np.zeros_like(local_max, dtype=bool)
    local_max[local_max < 1e-6] = 1.0
    
    local_norm = local_coords / local_max[..., np.newaxis]
    local_norm_flat = local_norm.reshape(-1,  local_norm.shape[-1])  # (F*I, K*2)

    slice_indices = np.sort(np.concatenate([2 * top_indices, 2 * top_indices + 1]))

    local_norm_sliced = local_norm_flat[:, slice_indices]
    valid_mask = ~np.all(local_norm_sliced == 0.0, axis=1)
    valid_indices = np.where(valid_mask)[0]

    local_norm_filtered = local_norm_sliced[valid_mask]
    angles = calculate_pose_rotations(
        local_x=local_norm_filtered[:, 0::2],
        local_y=local_norm_filtered[:, 1::2],
        angle_map_data=angle_map_data)
    local_norm_rotated = pose_alignment_worker(local_norm_filtered, angles=angles)

    if quant_step > 0:
        X_quantized = np.round(local_norm_rotated / quant_step) * quant_step
    else:
        X_quantized = local_norm_rotated.copy()

    _, inverse_indices, counts = np.unique(
        X_quantized, axis=0, return_inverse=True, return_counts=True
    )

    outlier_valid = counts[inverse_indices] < min_samples

    for idx, is_outlier in enumerate(outlier_valid):
        if is_outlier:
            global_idx = valid_indices[idx]
            f = global_idx // pred_data_array.shape[1]
            i = global_idx % pred_data_array.shape[1]
            outlier_mask[f, i] = True

    return outlier_mask

def outlier_flicker(pred_data_array: np.ndarray) -> np.ndarray:
    """
    Flags instances that appear only in the current frame but not in adjacent frames (flickering).
    
    Args:
        pred_data_array (np.ndarray): Shape (num_frames, num_instances, num_keypoints * 3).

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True = flickering.
    """
    all_x, all_y = pred_data_array[:, :, 0::3], pred_data_array[:, :, 1::3]
    presence = np.any(~np.isnan(all_x) & ~np.isnan(all_y), axis=2)

    instance_count = presence.shape[1]

    prev_presence = np.concatenate([np.zeros((1, instance_count), dtype=bool), presence[:-1]], axis=0)
    next_presence = np.concatenate([presence[1:], np.zeros((1, instance_count), dtype=bool)], axis=0)
    flicker_mask = presence & (~prev_presence) & (~next_presence)

    return flicker_mask

def outlier_enveloped(pred_data_array:np.ndarray, threshold:float=0.8) -> np.ndarray:
    """
    Flags smaller instances that are largely contained within the bounding box of a larger, nearby instance.

    Used to detect potential occlusion or false positives where one animal's pose is incorrectly 
    detected inside another's.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        threshold (float): Proportion of a small instance's keypoints that must lie within 
                                     the larger instance's bounding box to be flagged.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates an enveloped instance.
    """
    total_frames, instance_count, _ = pred_data_array.shape

    all_x, all_y = pred_data_array[:, :, 0::3], pred_data_array[:, :, 1::3]
    min_x_array, min_y_array, max_x_array, max_y_array = calculate_pose_bbox(all_x, all_y)
    size_array = (max_y_array - min_y_array) * (max_x_array - min_x_array)
    valid_sizes = size_array[~np.isnan(size_array)]
    bbox_length = np.sqrt(np.nanmax(valid_sizes)) 

    enveloped_mask = np.zeros((total_frames, instance_count), dtype=bool)

    for frame_idx in range(total_frames):
        valid_inst = get_instances_on_current_frame(pred_data_array, frame_idx)
        if len(valid_inst) < 2:
            continue

        close_pairs = []
        centroids, _ = calculate_pose_centroids(pred_data_array, frame_idx)
        for inst_1_idx, inst_2_idx in combinations(valid_inst, 2):
            inst_dist = np.linalg.norm(centroids[inst_2_idx] - centroids[inst_1_idx])
            if inst_dist < bbox_length * 0.5:
                close_pairs.append((inst_1_idx,inst_2_idx))
        
        if not close_pairs:
            continue

        for inst_1_idx, inst_2_idx in close_pairs:
            inst_1_size = size_array[frame_idx, inst_1_idx]
            inst_2_size = size_array[frame_idx, inst_2_idx]
            if inst_1_size > inst_2_size:
                small_inst = inst_2_idx
                large_inst = inst_1_idx
            else:
                small_inst = inst_1_idx
                large_inst = inst_2_idx

            small_x = all_x[frame_idx, small_inst]
            small_y = all_y[frame_idx, small_inst]

            min_x = min_x_array[frame_idx, large_inst]
            max_x = max_x_array[frame_idx, large_inst]
            min_y = min_y_array[frame_idx, large_inst]
            max_y = max_y_array[frame_idx, large_inst]
            width = max_x - min_x
            height = max_y - min_y
            padded_min_x = min_x - width * 0.1
            padded_max_x = max_x + width * 0.1
            padded_min_y = min_y - height * 0.1
            padded_max_y = max_y + height * 0.1

            valid = np.logical_and(~np.isnan(small_x), ~np.isnan(small_y))
            in_bbox_mask = (
                (small_x >= padded_min_x) &
                (small_x <= padded_max_x) &
                (small_y >= padded_min_y) &
                (small_y <= padded_max_y)
            )

            in_ratio = np.sum(in_bbox_mask & valid) / np.sum(valid)
            if in_ratio >= threshold:
                enveloped_mask[frame_idx, small_inst] = True
            
    return enveloped_mask