import numpy as np
from itertools import combinations
from typing import Dict, Optional

from .pose_analysis import calculate_pose_centroids, calculate_anatomical_centers, calculate_aligned_local
from utils.helper import bye_bye_runtime_warning, clean_inconsistent_nans


def outlier_removal(pred_data_array:np.ndarray, outlier_mask:np.ndarray) -> np.ndarray:
    if outlier_mask.ndim == 2:
        data_array = pred_data_array.copy()
        data_array[outlier_mask] = np.nan
        return data_array
    elif outlier_mask.ndim == 3:
        F, I, D = pred_data_array.shape
        K = D // 3
        data_array = pred_data_array.reshape(F, I, K, 3)
        data_array[outlier_mask] = np.nan
        return data_array.reshape(F, I, D)

########################################################################################################

def outlier_confidence(pred_data_array:np.ndarray, threshold:float=0.5, kp_mode:bool=False) -> np.ndarray:
    pred_data_array = clean_inconsistent_nans(pred_data_array)
    confidence_scores = pred_data_array[:, :, 2::3]

    if kp_mode:
        return confidence_scores < threshold

    with bye_bye_runtime_warning():
        inst_conf = np.nanmean(confidence_scores, axis=2)
        return inst_conf < threshold

def outlier_bodypart(pred_data_array:np.ndarray, threshold:int=2) -> np.ndarray:
    x_coords_mask = ~np.isnan(pred_data_array[:, :, 0::3])
    y_coords_mask = ~np.isnan(pred_data_array[:, :, 1::3])

    discovered_bodyparts = np.sum(np.logical_and(x_coords_mask, y_coords_mask), axis=2)
    low_bp_mask = np.logical_and(discovered_bodyparts < threshold, discovered_bodyparts > 0)
    
    return low_bp_mask

def outlier_size(pred_data_array:np.ndarray, min_ratio:float=0.3, max_ratio:float=2.5) -> np.ndarray:
    total_frames, instance_count, _ = pred_data_array.shape

    _, local_coords = calculate_pose_centroids(pred_data_array)
    inst_radius = np.full((total_frames, instance_count), np.nan)
    for inst_idx in range(instance_count):
        with bye_bye_runtime_warning():
            inst_radius[:, inst_idx] = np.nanmean(
                np.sqrt(local_coords[:, inst_idx, 0::2]**2 + local_coords[:, inst_idx, 1::2]**2), axis=1)

    mean_radius = np.nanmean(inst_radius)

    small_mask = inst_radius < mean_radius * min_ratio
    large_mask = inst_radius >= mean_radius * max_ratio

    return np.logical_or(small_mask, large_mask)

def outlier_pose(
    pred_data_array: np.ndarray,
    canon_pose: np.ndarray,
    threshold_px: float = 15.0,
    kp_mode:bool=False
) -> np.ndarray:
    F, I, _ = pred_data_array.shape
    K = canon_pose.shape[0]
    assert pred_data_array.shape[2] == K * 3, "Keypoint count mismatch"

    data = pred_data_array.reshape(F, I, K, 3)
    xy = data[..., :2]

    fully_observed = np.all(~np.isnan(xy), axis=(2, 3))

    outlier_mask = np.zeros((F, I), dtype=bool)

    if not np.any(fully_observed):
        return outlier_mask

    N = np.sum(fully_observed)
    xy_complete = xy[fully_observed]
    canon_batch = np.tile(canon_pose[None, :, :], (N, 1, 1))

    src_mean = np.mean(canon_batch, axis=1, keepdims=True)
    dst_mean = np.mean(xy_complete, axis=1, keepdims=True)

    src_centered = canon_batch - src_mean
    dst_centered = xy_complete - dst_mean

    H = np.einsum('nki,nkj->nij', src_centered, dst_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.einsum('nij,nkj->nik', Vt, U)  # R = V @ U^T

    detR = np.linalg.det(R)
    reflect_mask = detR < 0
    if np.any(reflect_mask):
        Vt_reflect = Vt.copy()
        Vt_reflect[reflect_mask, 1, :] *= -1
        R[reflect_mask] = np.einsum('nij,nkj->nik', Vt_reflect[reflect_mask], U[reflect_mask])

    t = dst_mean.squeeze(axis=1) - np.einsum('nij,nj->ni', R, src_mean.squeeze(axis=1))
    transformed_canon = np.einsum('nij,nkj->nki', R, canon_batch) + t[:, None, :]

    flat_indices = np.flatnonzero(fully_observed.ravel())

    errors = np.linalg.norm(xy_complete - transformed_canon, axis=2)
    if kp_mode:
        outliers_complete_kp = errors > threshold_px
        outlier_kp_mask_flat = np.zeros((F * I, K), dtype=bool)
        outlier_kp_mask_flat[flat_indices] = outliers_complete_kp
        return outlier_kp_mask_flat.reshape(F, I, K)

    outlier_complete = np.mean(errors, axis=1) > threshold_px
    outlier_mask_flat = np.zeros(F * I, dtype=bool)
    outlier_mask_flat[flat_indices[outlier_complete]] = True
    return outlier_mask_flat.reshape(F, I)

def outlier_rotation(pred_data_array: np.ndarray, angle_map_data: Dict[str, int], threshold_deg: float = 10.0) -> np.ndarray:
    head_idx = angle_map_data['head_idx']
    center_idx = angle_map_data['center_idx']
    tail_idx = angle_map_data['tail_idx']

    hx, hy = pred_data_array[:, :, head_idx * 3], pred_data_array[:, :, head_idx * 3 + 1]
    cx, cy = pred_data_array[:, :, center_idx * 3], pred_data_array[:, :, center_idx * 3 + 1]
    tx, ty = pred_data_array[:, :, tail_idx * 3], pred_data_array[:, :, tail_idx * 3 + 1]

    v1x = hx - cx
    v1y = hy - cy
    v2x = tx - cx
    v2y = ty - cy

    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    angles_rad = np.arctan2(np.abs(cross), dot)
    angles_deg = np.degrees(angles_rad)

    valid_head = ~np.isnan(hx) & ~np.isnan(hy)
    valid_center = ~np.isnan(cx) & ~np.isnan(cy)
    valid_tail = ~np.isnan(tx) & ~np.isnan(ty)
    all_valid = valid_head & valid_center & valid_tail

    low_angle_mask = (angles_deg < threshold_deg) & all_valid

    return low_angle_mask

def outlier_duplicate(
        pred_data_array:np.ndarray,
        bp_threshold:float=0.5,
        dist_threshold:float=5.0
        ) -> np.ndarray:
    F, I, _ = pred_data_array.shape
    instances = list(range(I))
    duplicate_mask = np.zeros((F, I), dtype=bool)

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

def outlier_speed(
    pred_data_array: np.ndarray,
    angle_map_data: Optional[Dict[str, int]]=None,
    max_speed_px: float = 50.0,
    kp_mode: bool = False
) -> np.ndarray:
    F, I, D = pred_data_array.shape
    K = D // 3

    x = pred_data_array[:, :, 0::3]
    y = pred_data_array[:, :, 1::3]

    if kp_mode:
        if angle_map_data is not None:
            aligned_array = calculate_aligned_local(pred_data_array, angle_map_data)
            x = aligned_array[:, :, 0::3]
            y = aligned_array[:, :, 1::3]

        dx = np.diff(x, axis=0)
        dy = np.diff(y, axis=0)
        speed = np.sqrt(dx**2 + dy**2)

        valid_curr = ~np.isnan(x[:-1]) & ~np.isnan(y[:-1])
        valid_next = ~np.isnan(x[1:]) & ~np.isnan(y[1:])
        valid = valid_curr & valid_next

        mask = np.zeros((F, I, K), dtype=bool)
        mask[1:] = (speed > max_speed_px) & valid

        return mask
    else:
        if angle_map_data is not None:
            centers = calculate_anatomical_centers(pred_data_array, angle_map_data)
        else:
            centroids, _ = calculate_pose_centroids(pred_data_array)
            centers = centroids

        dx = np.diff(centers[..., 0], axis=0)
        dy = np.diff(centers[..., 1], axis=0)
        speed = np.sqrt(dx**2 + dy**2)

        valid_curr = np.any(~np.isnan(x[:-1]), axis=2)
        valid_next = np.any(~np.isnan(x[1:]), axis=2)
        valid = valid_curr & valid_next

        mask = np.zeros((F, I), dtype=bool)
        mask[1:] = (speed > max_speed_px) & valid

        return mask