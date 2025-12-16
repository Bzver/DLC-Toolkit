import numpy as np
from typing import Tuple, Union, Dict

from .pose_worker import pose_alignment_worker
from utils.helper import bye_bye_runtime_warning

def calculate_pose_centroids(
        pred_data_array:np.ndarray,
        frame_slice:Union[slice, int]=slice(None)
        )-> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the centroid (mean x, mean y) of pose keypoints for each instance and frame,
    and compute the relative position of each keypoint with respect to the centroid.

    Args:
        pred_data_array (np.ndarray): 3D array of shape (frames, instances, keypoints*3).
        frame_slice (Union[slice, int]): Slice or index to select specific frames.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pose_centroids: Array of shape (instances, 2) or (N, instances, 2) with mean x and y coordinates.
            - local_coords: Array of shape (instances, keypoints*2) or (N, instances, keypoints*2) with relative positions.
    """
    _, instance_count, xyconf = pred_data_array.shape
    num_keypoint = xyconf // 3

    x_vals = pred_data_array[frame_slice, :, 0::3]
    y_vals = pred_data_array[frame_slice, :, 1::3]

    if isinstance(frame_slice, (int, np.integer)): # Determine output shape
        output_shape_pose = (instance_count, 2)
        output_shape_rltv = (instance_count, num_keypoint * 2)
    else:
        output_shape_pose = (x_vals.shape[0], instance_count, 2)
        output_shape_rltv = (x_vals.shape[0], instance_count, num_keypoint * 2)

    pose_centroids = np.full(output_shape_pose, np.nan, dtype=np.float64)
    local_coords = np.full(output_shape_rltv, np.nan, dtype=np.float64)
    
    valid_mask = np.any(~np.isnan(x_vals), axis=-1)

    pose_centroids[valid_mask, 0] = np.nanmean(x_vals[valid_mask], axis=-1)
    pose_centroids[valid_mask, 1] = np.nanmean(y_vals[valid_mask], axis=-1)

    local_coords[..., 0::2] = x_vals - pose_centroids[..., 0, np.newaxis]
    local_coords[..., 1::2] = y_vals - pose_centroids[..., 1, np.newaxis]
        
    return pose_centroids, local_coords

def calculate_pose_rotations(
        local_x: np.ndarray,
        local_y: np.ndarray,
        angle_map_data:Dict[str, any]
        ) -> Union[np.ndarray, float]:
    """
    Calculate the rotation angles for poses based on relative keypoint positions.
    Uses the vector from the most-central keypoint (lowest spread) to the most-peripheral 
    keypoint (highest spread) as the reference direction to compute orientation.

    Args:
        local_x : Relative x-coordinates of keypoints, shape (K,) or (N, K).
        local_y : Relative y-coordinates of keypoints, shape (K,) or (N, K).
        angle_map_data : Dict with 'head_idx', 'tail_idx', 'angle_map'.
            - angle map: list of (i, j, offset, weight) of each connection

    Returns:
        Union[np.ndarray, float]: Rotation angle(s) in radians.
                                  - If input is (K,), returns float.
                                  - If input is (N, K), returns (N,) array.
    """
    n_dim = local_x.ndim
    if n_dim == 1:
        local_x = local_x[np.newaxis, :]  # (1, K)
        local_y = local_y[np.newaxis, :]  # (1, K)

    N, K = local_x.shape    
    angles = np.full(N, np.nan)

    head_idx = angle_map_data['head_idx']
    tail_idx = angle_map_data['tail_idx']

    # Use head-tail first
    xh, yh = local_x[:, head_idx], local_y[:, head_idx]
    xt, yt = local_x[:, tail_idx], local_y[:, tail_idx]

    valid = np.isfinite(xh) & np.isfinite(yh) & np.isfinite(xt) & np.isfinite(yt)
    if np.any(valid): # Compute angle from tail to head (anterior direction)
        dx = xh[valid] - xt[valid]
        dy = yh[valid] - yt[valid]
        head_tail_angles = np.arctan2(dy, dx)
        angles[valid] = head_tail_angles

    # Fill gaps using angle map with weight voting
    total_weights = np.zeros(N)
    weighted_sum = np.zeros(N)

    for entry in angle_map_data.get('angle_map', []):
        i, j = entry['i'], entry['j']
        offset = entry['offset']
        weight = entry['weight']

        if not (0 <= i < K and 0 <= j < K):
            continue

        xi, yi = local_x[:, i], local_y[:, i]
        xj, yj = local_y[:, j], local_y[:, j]
        connection_valid_mask = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(xj) & np.isfinite(yj)
        if not np.any(connection_valid_mask):
            continue

        # Compute observed vector angle (from i to j) for valid connections
        obs_angle_subset = np.arctan2(yj[connection_valid_mask] - yi[connection_valid_mask],
                                      xj[connection_valid_mask] - xi[connection_valid_mask])
        inferred_angle_subset = obs_angle_subset - offset
        inferred_angle_subset = np.arctan2(np.sin(inferred_angle_subset), np.cos(inferred_angle_subset))

        # Use weighted averaging in complex space to avoid angle wrapping issues
        inferred_angle_complex_subset = np.exp(1j * inferred_angle_subset)
        temp_inferred_complex_full = np.full(N, np.nan + 0j, dtype=complex)
        temp_inferred_complex_full[connection_valid_mask] = inferred_angle_complex_subset

        global_target_mask = connection_valid_mask & np.isnan(angles)

        if np.any(global_target_mask):
            weighted_sum[global_target_mask] += weight * np.angle(temp_inferred_complex_full[global_target_mask])
            total_weights[global_target_mask] += weight

    # Apply weighted average for frames that were filled via consensus
    final_mask = (total_weights > 0) & np.isnan(angles)
    angles[final_mask] = np.arctan2(
        np.sin(weighted_sum[final_mask] / total_weights[final_mask]),
        np.cos(weighted_sum[final_mask] / total_weights[final_mask])
    )

    return angles[0].item() if n_dim == 1 else angles

def calculate_pose_bbox(
        coords_x:np.ndarray,
        coords_y:np.ndarray,
        padding:float=10.0
        ) -> Union[Tuple[float, float, float, float],\
                   Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate bounding box (with padding) around pose coordinates.

    Args:
    - coords_x: Array of x-coordinates. Shape: (keypoints,) or (frames, keypoints) or (..., K)
    - coords_y: Array of y-coordinates. Same shape as coords_x.
    - padding: Padding to add around the bounding box (default: 10.0).

    Returns:
    - If input is 1D: returns (min_x, min_y, max_x, max_y) as scalars.
    - If input is ND (N>=2): returns arrays of shape (...,) for each bound.
    """
    with bye_bye_runtime_warning():
        min_x, min_y = np.nanmin(coords_x, axis=-1) - padding, np.nanmin(coords_y, axis=-1) - padding
        max_x, max_y = np.nanmax(coords_x, axis=-1) + padding, np.nanmax(coords_y, axis=-1) + padding
    return min_x, min_y, max_x, max_y

def calculate_canonical_pose(
        pred_data_array:np.ndarray,
        head_idx:int,
        tail_idx:int
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a canonical (aligned and averaged) pose by aligning individual animal poses to a common orientation
    based on the head-to-tail body axis.

    Args:
        pred_data_array (np.ndarray):, prediction array of shape (F, I, 3*K)
        head_idx / tail_idx (int): Index of the keypoint representing the head /tail, used to define the two ends
        of the body axis.

    Returns:
        canon_pose (np.ndarray): Canonical average pose of shape (K, 2)
        aligned_frame_poses (np.ndarray):  Aligned individual poses of shape (M, 2*K)
        """
    F, I, XYCONF = pred_data_array.shape
    pred_data_combined = pred_data_array.reshape(F*I, 1, XYCONF)
    _, local_coords = calculate_pose_centroids(pred_data_combined)
    all_frame_poses = np.squeeze(local_coords)

    no_head_mask = np.any(np.isnan(all_frame_poses[:, head_idx*2:head_idx*2+2]), axis=1)
    no_tail_mask = np.any(np.isnan(all_frame_poses[:, tail_idx*2:tail_idx*2+2]), axis=1)
    valid_frame_poses = all_frame_poses[~(no_head_mask | no_tail_mask)]

    dy = valid_frame_poses[:, head_idx*2+1] - valid_frame_poses[:, tail_idx*2+1]
    dx = valid_frame_poses[:, head_idx*2] - valid_frame_poses[:, tail_idx*2]
    valid_angles = np.arctan2(dy, dx)

    aligned_frame_poses = pose_alignment_worker(valid_frame_poses, valid_angles)

    average_pose = np.nanmean((aligned_frame_poses), axis=0)
    canon_pose = average_pose.reshape(XYCONF//3, 2) # (kp,2)
    
    return canon_pose, aligned_frame_poses