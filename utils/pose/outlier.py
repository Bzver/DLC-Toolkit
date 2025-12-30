import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from typing import Tuple, Dict

from .pose_analysis import calculate_pose_centroids, calculate_pose_rotations
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
    data_array = pred_data_array.copy()
    removed_frames_count = np.sum(np.any(outlier_mask, axis=1))
    removed_instances_count = np.sum(outlier_mask)
    data_array[outlier_mask] = np.nan
    return data_array, removed_frames_count, removed_instances_count

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
    angle_map_data: Dict[str, int],
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


def outlier_enveloped(
    pred_data_array: np.ndarray,
    threshold: float = 0.75,
    use_convex_hull: bool = True,
    padding: float = 0.1  # padding as fraction of hull size (only for AABB fallback)
) -> np.ndarray:
    """
    Flags smaller instances whose keypoints are largely contained inside the *shape* (e.g., convex hull)
    of a larger, nearby instance — instead of just its AABB.

    Args:
        pred_data_array: (num_frames, num_instances, num_keypoints * 3)
        threshold: proportion of small instance's valid points that must be inside the large instance's hull.
        use_convex_hull: if False, falls back to padded AABB.
        padding: only used if use_convex_hull=False (same as before: 10% bbox padding).

    Returns:
        Boolean mask (num_frames, num_instances): True → flagged as enveloped.
    """
    total_frames, instance_count, _ = pred_data_array.shape

    all_x, all_y = pred_data_array[:, :, 0::3], pred_data_array[:, :, 1::3]

    with bye_bye_runtime_warning():
        min_x_array, max_x_array = np.nanmin(all_x, axis=2), np.nanmax(all_x, axis=2)
        min_y_array, max_y_array = np.nanmin(all_y, axis=2), np.nanmax(all_y, axis=2)

    size_array = (max_y_array - min_y_array) * (max_x_array - min_x_array)
    valid_sizes = size_array[~np.isnan(size_array)]
    bbox_length = np.sqrt(np.nanmax(valid_sizes)) if valid_sizes.size > 0 else 1.0

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

        instance_points = {}
        for inst_idx in valid_inst:
            x = all_x[frame_idx, inst_idx]
            y = all_y[frame_idx, inst_idx]
            mask = ~np.isnan(x) & ~np.isnan(y)
            pts = np.column_stack([x[mask], y[mask]])  # (K_valid, 2)
            if pts.shape[0] >= 3:
                instance_points[inst_idx] = pts

        for inst_1_idx, inst_2_idx in close_pairs:
            if inst_1_idx not in instance_points or inst_2_idx not in instance_points:
                continue

            pts1 = instance_points[inst_1_idx]
            pts2 = instance_points[inst_2_idx]

            size1 = size_array[frame_idx, inst_1_idx]
            size2 = size_array[frame_idx, inst_2_idx]

            if size1 > size2:
                large_idx, small_idx = inst_1_idx, inst_2_idx
                large_pts, small_pts = pts1, pts2
            else:
                large_idx, small_idx = inst_2_idx, inst_1_idx
                large_pts, small_pts = pts2, pts1

            try:
                if use_convex_hull:
                    hull = ConvexHull(large_pts)
                    hull_vertices = large_pts[hull.vertices]  # (M, 2)
                    path = Path(hull_vertices)
                    in_hull = path.contains_points(small_pts)
                else:
                    min_x, max_x = min_x_array[frame_idx, large_idx], max_x_array[frame_idx, large_idx]
                    min_y, max_y = min_y_array[frame_idx, large_idx], max_y_array[frame_idx, large_idx]
                    w, h = max_x - min_x, max_y - min_y
                    padded_min_x = min_x - w * padding
                    padded_max_x = max_x + w * padding
                    padded_min_y = min_y - h * padding
                    padded_max_y = max_y + h * padding
                    in_hull = (
                        (small_pts[:, 0] >= padded_min_x) &
                        (small_pts[:, 0] <= padded_max_x) &
                        (small_pts[:, 1] >= padded_min_y) &
                        (small_pts[:, 1] <= padded_max_y)
                    )
            except Exception as e:
                min_x, max_x = np.nanmin(large_pts[:, 0]), np.nanmax(large_pts[:, 0])
                min_y, max_y = np.nanmin(large_pts[:, 1]), np.nanmax(large_pts[:, 1])
                w, h = (max_x - min_x), (max_y - min_y)
                if w == 0 or h == 0: 
                    continue
                padded_min_x = min_x - w * padding
                padded_max_x = max_x + w * padding
                padded_min_y = min_y - h * padding
                padded_max_y = max_y + h * padding
                in_hull = (
                    (small_pts[:, 0] >= padded_min_x) &
                    (small_pts[:, 0] <= padded_max_x) &
                    (small_pts[:, 1] >= padded_min_y) &
                    (small_pts[:, 1] <= padded_max_y)
                )

            in_ratio = np.mean(in_hull)
            if in_ratio >= threshold:
                enveloped_mask[frame_idx, small_idx] = True

    return enveloped_mask