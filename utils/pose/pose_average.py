import numpy as np

from typing import Optional, Dict

from .pose_analysis import calculate_pose_centroids, calculate_pose_rotations
from .pose_worker import pose_alignment_worker, pose_rotation_worker
from .outlier import outlier_confidence, outlier_bodypart, outlier_removal

def get_average_pose(
        pred_data_array:np.ndarray,
        selected_instance_idx:int,
        frame_idx:int, 
        angle_map_data:Dict[str, int],
        initial_pose_range:int = 30,
        confidence_threshold:float = 0.5,
        bodypart_threshold:int = 3,
        max_attempts:int = 10,
        valid_frames_threshold:int = 30,
        set_centroid:Optional[np.ndarray]=None,
        set_rotation:Optional[float]=None
        ) -> np.ndarray:
    """
    Compute a rotation-normalized average pose for a specific instance around a given frame.

    Args:
        pred_data_array: (F, I, K*3)
        selected_instance_idx: Index of the instance to average
        frame_idx: Center frame of temporal window
        initial_pose_range: Initial half-window size (± frames)
        confidence_threshold: Min confidence to count a keypoint
        bodypart_threshold: Min valid keypoints percentage per frame to keep
        max_attempts: Max times to double window size
        valid_frames_threshold: Min number of valid frames required
        set_centroid: Optional 2D array of shape (2,) to set a specific centroid for alignment
        set_rotation: Optional float to set a specific angle for alignment
    Returns:
        average_pose: (K*3,) array — mean pose in aligned orientation
    """
    pose_range = initial_pose_range
    attempt = 0

    while attempt < max_attempts:
        pose_window = _get_pose_window(frame_idx, len(pred_data_array), pose_range)
        pred_data_sliced = pred_data_array[pose_window].copy()

        outlier_conf_mask = outlier_confidence(pred_data_sliced, confidence_threshold)
        outlier_bp_mask = outlier_bodypart(pred_data_sliced, bodypart_threshold)
        outlier_mask = np.logical_or(outlier_bp_mask, outlier_conf_mask)
        pred_data_filtered, _, _ = outlier_removal(pred_data_sliced, outlier_mask)

        truncate_mask = np.any(~np.isnan(pred_data_filtered[:, selected_instance_idx, :]), axis=1)
        pred_data_truncated = pred_data_filtered[truncate_mask]

        inst_slice = pred_data_truncated[:, selected_instance_idx:selected_instance_idx+1, :]
        valid_frame_count = _get_valid_slice_count(inst_slice)

        if valid_frame_count > valid_frames_threshold:
            break

        pose_range *= 2
        attempt += 1
    else:
        raise ValueError(f"Only {valid_frame_count} valid frames found for instance {selected_instance_idx} "
            f"around frame {frame_idx}, less than required {valid_frames_threshold}, ")

    inst_data = pred_data_truncated[:, selected_instance_idx, :]
    conf_scores = inst_data[:, 2::3]

    centroids, local_coords = calculate_pose_centroids(inst_data[:, np.newaxis, :])

    centroids = np.squeeze(centroids, axis=1)
    local_coords = np.squeeze(local_coords, axis=1)
    rotation_angles = calculate_pose_rotations(local_coords[:, 0::2], local_coords[:, 1::2], angle_map_data=angle_map_data)
    aligned_local = pose_alignment_worker(local_coords, rotation_angles)
    avg_angle = _calculate_circular_mean(rotation_angles)
    if set_centroid is not None :
        centroids = set_centroid
    if set_rotation is not None and ~np.isnan(set_rotation):
        avg_angle = set_rotation
    average_pose = pose_rotation_worker(avg_angle, centroids, aligned_local, conf_scores)

    return average_pose

def _get_pose_window(frame_idx:int, total_frames:int, pose_range:int) -> slice:
    min_frame = frame_idx - pose_range
    max_frame = frame_idx + pose_range + 1

    offset = 0
    if min_frame < 0:
        offset = -min_frame
    elif max_frame > total_frames:
        offset = total_frames - max_frame

    min_frame += offset
    max_frame += offset

    # Ensure bounds
    min_frame = max(0, min_frame)
    max_frame = min(total_frames, max_frame)

    return slice(min_frame, max_frame)

def _get_valid_slice_count(arr_3D:np.ndarray) -> int:
    if arr_3D.ndim != 3:
        raise ValueError("Error: Input array must be a 3D array!")
    not_nan = ~np.isnan(arr_3D)
    has_non_nan = np.any(not_nan, axis=(1,2))
    count = np.sum(has_non_nan)
    return count

def _calculate_circular_mean(angles: np.ndarray) -> float:
    """Compute mean of angles in radians."""
    x = np.nanmean(np.cos(angles))
    y = np.nanmean(np.sin(angles))
    return np.arctan2(y, x)