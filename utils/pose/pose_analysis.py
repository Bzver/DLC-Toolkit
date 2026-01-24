import numpy as np
from typing import Tuple, Union, Dict

from .pose_worker import pose_alignment_worker
from utils.helper import bye_bye_runtime_warning


def calculate_pose_centroids(
        pred_data_array:np.ndarray,
        frame_idx:int=-1,
        )-> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the centroid (mean x, mean y) of pose keypoints for each instance and frame,
    and compute the relative position of each keypoint with respect to the centroid.

    Args:
        pred_data_array (np.ndarray): 3D array of shape (frames, instances, keypoints*3).
        frame_idx (int): Use the entire array if not given
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pose_centroids: Array of shape (instances, 2) or (N, instances, 2) with mean x and y coordinates.
            - local_coords: Array of shape (instances, keypoints*2) or (N, instances, keypoints*2) with relative positions.
    """
    _, I, D = pred_data_array.shape
    K = D // 3

    pred_data_array_sliced = pred_data_array[frame_idx:frame_idx+1] if frame_idx > 0 else pred_data_array

    x_vals = pred_data_array_sliced[..., 0::3]
    y_vals = pred_data_array_sliced[..., 1::3]
    F = x_vals.shape[0]

    pose_centroids = np.full((F, I, 2), np.nan, dtype=np.float64)
    local_coords = np.full((F, I, K*2), np.nan, dtype=np.float64)
    
    valid_mask = np.any(~np.isnan(x_vals), axis=-1)

    pose_centroids[valid_mask, 0] = np.nanmean(x_vals[valid_mask], axis=-1)
    pose_centroids[valid_mask, 1] = np.nanmean(y_vals[valid_mask], axis=-1)

    local_coords[..., 0::2] = x_vals - pose_centroids[..., 0, np.newaxis]
    local_coords[..., 1::2] = y_vals - pose_centroids[..., 1, np.newaxis]
    
    if F == 1:
        pose_centroids = np.squeeze(pose_centroids, axis=0)
        local_coords = np.squeeze(local_coords, axis=0)

    return pose_centroids, local_coords

def calculate_anatomical_centers(
    pred_data_array: np.ndarray,
    angle_map_data: Dict[str, int]
) -> np.ndarray:
    """
    Compute pose centers using anatomical center keypoint when available,
    falling back to geometric centroid otherwise.
    
    Args:
        pred_data_array : (F, I, 3*K)
        angle_map_data : dict with 'head_idx', 'tail_idx', 'center_idx'
    
    Returns:
        centers : (F, I, 2) array of [x, y] centers
    """
    center_idx = angle_map_data["center_idx"]

    centroids, _ = calculate_pose_centroids(pred_data_array)
    
    center_x = pred_data_array[:, :, center_idx * 3]
    center_y = pred_data_array[:, :, center_idx * 3 + 1]
    valid = ~np.isnan(center_x) & ~np.isnan(center_y)

    centers = centroids.copy()
    centers[valid, 0] = center_x[valid]
    centers[valid, 1] = center_y[valid]

    return centers

def calculate_pose_rotations(
    local_x: np.ndarray,
    local_y: np.ndarray,
    angle_map_data: Dict[str, int]
) -> Union[np.ndarray, float]:
    """
    Compute rotation angle (in radians) representing the orientation of a pose.
    
    Args:
        local_x, local_y: (K,) or (N, K) or (F, N, K)
        angle_map_data: dict with 'head_idx', 'tail_idx', 'center_idx'

    Returns:
        float if input 1D, (N,) or (F, N,) if otherwise
    """
    assert local_x.shape == local_y.shape, f"Shape mismatch: {local_x.shape} vs {local_y.shape}"

    orig_ndim = local_x.ndim
    if orig_ndim == 1:
        N = 1
        lcx = local_x[None, :]  # (1, K)
        lcy = local_y[None, :]
    elif orig_ndim == 3:
        F, I = local_x.shape[:2]
        N = F * I
        lcx = local_x.reshape((N, -1))
        lcy = local_y.reshape((N, -1))
    else:
        N = local_x.shape[0]
        lcx = local_x
        lcy = local_y

    head_idx = angle_map_data['head_idx']
    tail_idx = angle_map_data['tail_idx']
    center_idx = angle_map_data['center_idx']

    xh, yh = lcx[:, head_idx], lcy[:, head_idx]
    xc, yc = lcx[:, center_idx], lcy[:, center_idx]
    xt, yt = lcx[:, tail_idx], lcy[:, tail_idx]

    def angle_from_to(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return np.arctan2(dy, dx)

    cos_sum = np.zeros(N, dtype=np.float64)
    sin_sum = np.zeros(N, dtype=np.float64)
    count = np.zeros(N, dtype=int)

    valid_ch = np.isfinite(xc) & np.isfinite(yc) & np.isfinite(xh) & np.isfinite(yh)
    if np.any(valid_ch):
        ang = angle_from_to(xc[valid_ch], yc[valid_ch], xh[valid_ch], yh[valid_ch])
        cos_sum[valid_ch] += np.cos(ang)
        sin_sum[valid_ch] += np.sin(ang)
        count[valid_ch] += 1

    valid_th = np.isfinite(xt) & np.isfinite(yt) & np.isfinite(xh) & np.isfinite(yh)
    if np.any(valid_th):
        ang = angle_from_to(xt[valid_th], yt[valid_th], xh[valid_th], yh[valid_th])
        cos_sum[valid_th] += np.cos(ang)
        sin_sum[valid_th] += np.sin(ang)
        count[valid_th] += 1

    valid_tc = np.isfinite(xt) & np.isfinite(yt) & np.isfinite(xc) & np.isfinite(yc)
    if np.any(valid_tc):
        ang = angle_from_to(xt[valid_tc], yt[valid_tc], xc[valid_tc], yc[valid_tc])
        cos_sum[valid_tc] += np.cos(ang)
        sin_sum[valid_tc] += np.sin(ang)
        count[valid_tc] += 1

    angles = np.full(N, np.nan, dtype=np.float64)
    has_data = count > 0
    if np.any(has_data):
        mean_cos = cos_sum[has_data] / count[has_data]
        mean_sin = sin_sum[has_data] / count[has_data]
        angles[has_data] = np.arctan2(mean_sin, mean_cos)

    still_nan = np.isnan(angles)
    if np.any(still_nan):
        points = np.stack([lcx, lcy], axis=-1)

        for i in np.where(still_nan)[0]:
            pts = points[i]
            mask = np.isfinite(pts).all(axis=1)
            valid_pts = pts[mask]

            if len(valid_pts) < 2:
                angles[i] = 0.0
                continue

            mean_pt = valid_pts.mean(axis=0)
            centered = valid_pts - mean_pt

            try:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                d = Vt[0]
            except np.linalg.LinAlgError:
                angles[i] = 0.0
                continue

            if np.isfinite(xh[i]) and np.isfinite(yh[i]):
                p = np.array([xh[i], yh[i]])
                proj = np.dot(p - mean_pt, d)
                if proj < 0:
                    d = -d
            elif np.isfinite(xt[i]) and np.isfinite(yt[i]):
                p = np.array([xt[i], yt[i]])
                proj = np.dot(p - mean_pt, d)
                if proj > 0:
                    d = -d

            angles[i] = np.arctan2(d[1], d[0])

    angles = np.nan_to_num(angles, nan=0.0)

    if orig_ndim == 1:
        return float(angles[0])
    elif orig_ndim == 3:
        return angles.reshape((F, I))
    else:
        return angles

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
    F, I, D = pred_data_array.shape
    K = D // 3
    pred_data_combined = pred_data_array.reshape(F*I, 1, D)
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
    canon_pose = average_pose.reshape(K, 2)
    
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
            Aligned local coordinates of shape (frames, instances, keypoints * 3), 
    """
    F, I, D = pred_data_array.shape
    K = D // 3

    local_poses = np.full((F, I, K*2), np.nan)

    centers = calculate_anatomical_centers(pred_data_array, angle_map_data)
    
    local_poses[..., 0::2] = pred_data_array[:, :, 0::3] - centers[..., 0, np.newaxis]
    local_poses[..., 1::2] = pred_data_array[:, :, 1::3] - centers[..., 1, np.newaxis]
    flat_local = local_poses.reshape(F*I, 2*K)
    flat_x = flat_local[:, 0::2]
    flat_y = flat_local[:, 1::2]

    angles_flat = calculate_pose_rotations(flat_x, flat_y, angle_map_data)
    aligned_flat = pose_alignment_worker(flat_local, angles_flat)

    aligned_local_coords = aligned_flat.reshape(F, I, K*2)

    ouput_array = np.full((F, I, K*3), np.nan)
    ouput_array[..., 0::3] = aligned_local_coords[..., 0::2]
    ouput_array[..., 1::3] = aligned_local_coords[..., 1::2]
    ouput_array[..., 2::3] = pred_data_array[:, :, 2::3]
    
    return ouput_array