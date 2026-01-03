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
    angle_map_data: Dict[str, int]
) -> Union[np.ndarray, float]:
    """
    Compute rotation angle (in radians) representing the orientation of a pose.
    
    Args:
        local_x, local_y: (K,) or (N, K)
        angle_map_data: dict with 'head_idx', 'tail_idx', 'center_idx'

    Returns:
        float if input 1D, (N,) array if 2D.
    """
    orig_ndim = local_x.ndim
    if orig_ndim == 1:
        local_x = local_x[None, :]  # (1, K)
        local_y = local_y[None, :]

    N= local_x.shape[0]
    angles = np.full(N, np.nan, dtype=np.float64)

    head_idx = angle_map_data['head_idx']
    tail_idx = angle_map_data['tail_idx']
    center_idx = angle_map_data['center_idx']

    xh, yh = local_x[:, head_idx], local_y[:, head_idx]
    xc, yc = local_x[:, center_idx], local_y[:, center_idx]
    xt, yt = local_x[:, tail_idx], local_y[:, tail_idx]

    def angle_from_to(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return np.arctan2(dy, dx)

    valid_ch = np.isfinite(xc) & np.isfinite(yc) & np.isfinite(xh) & np.isfinite(yh)
    if np.any(valid_ch):
        angles[valid_ch] = angle_from_to(xc[valid_ch], yc[valid_ch], xh[valid_ch], yh[valid_ch])

    still_nan = np.isnan(angles)
    valid_ct = still_nan & np.isfinite(xc) & np.isfinite(yc) & np.isfinite(xt) & np.isfinite(yt)
    if np.any(valid_ct):
        angles[valid_ct] = angle_from_to(xt[valid_ct], yt[valid_ct], xc[valid_ct], yc[valid_ct])

    still_nan = np.isnan(angles)
    valid_ht = still_nan & np.isfinite(xh) & np.isfinite(yh) & np.isfinite(xt) & np.isfinite(yt)
    if np.any(valid_ht):
        angles[valid_ht] = angle_from_to(xt[valid_ht], yt[valid_ht], xh[valid_ht], yh[valid_ht])

    still_nan = np.isnan(angles)
    if np.any(still_nan):
        points = np.stack([local_x, local_y], axis=-1)

        for i in np.where(still_nan)[0]:
            pts = points[i]
            mask = np.isfinite(pts).all(axis=1)
            valid_pts = pts[mask]

            if len(valid_pts) < 2:
                angles[i] = 0.0
                continue

            mean = valid_pts.mean(axis=0)
            centered = valid_pts - mean

            try:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                direction = Vt[0]
                angles[i] = np.arctan2(direction[1], direction[0])
            except np.linalg.LinAlgError:
                angles[i] = 0.0

    angles[np.isnan(angles)] = 0.0

    return float(angles[0]) if orig_ndim == 1 else angles

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

def calculate_aligned_local(
        pred_data_array: np.ndarray,
        angle_map_data: Dict[str, int]
        ) -> np.ndarray:
    """
    Compute centroid-centered and rotation-normalized (canonical) pose coordinates 
    for all frames and instances in a single vectorized pass.
    Args:
        pred_data_array (np.ndarray): (frames, instances, keypoints * 3), 
        angle_map_data (Dict[str, int]): 
            Dictionary containing at least:
              - 'head_idx': int, keypoint index for head
              - 'tail_idx': int, keypoint index for tail
              - 'angle_map': list of connection dicts (optional, for fallback angle estimation)
    Returns:
        np.ndarray: 
            Aligned local coordinates of shape (frames, instances, keypoints * 2), 
    """
    F, I, XYCONF = pred_data_array.shape
    K = XYCONF // 3

    local_poses = np.full((F, I, K*2), np.nan)

    center_idx = angle_map_data["center_idx"]

    pose_centroids = pred_data_array[:, :, center_idx*3:center_idx*3+2]
    math_centroids, _ = calculate_pose_centroids(pred_data_array)

    nan_mask = np.isnan(pose_centroids)
    pose_centroids[nan_mask] = math_centroids[nan_mask]
    
    local_poses[..., 0::2] = pred_data_array[:, :, 0::3] - pose_centroids[..., 0, np.newaxis]
    local_poses[..., 1::2] = pred_data_array[:, :, 1::3] - pose_centroids[..., 1, np.newaxis]

    flat_local = local_poses.reshape(F*I, 2*K)
    flat_x = flat_local[:, 0::2]
    flat_y = flat_local[:, 1::2]

    angles_flat = calculate_pose_rotations(flat_x, flat_y, angle_map_data)
    aligned_flat = pose_alignment_worker(flat_local, angles_flat)

    aligned_local_coords = aligned_flat.reshape(F, I, 2*K)

    return aligned_local_coords