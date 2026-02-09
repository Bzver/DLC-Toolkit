import numpy as np
from itertools import combinations
from typing import Dict, Optional, List

from .pose_analysis import calculate_pose_centroids, calculate_anatomical_centers, calculate_aligned_local, calculate_pose_bbox
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

def outlier_bad_to_the_bone(
    pred_data_array: np.ndarray,
    skele_list: List[List[int]],
    threshold_max: float = 2.0,
    kp_mode: bool = False,
    ignored_bones: List[List[int]] = None
) -> np.ndarray:
    if ignored_bones is None:
        ignored_bones = []

    ignore_set = {tuple(sorted(b)) for b in ignored_bones}

    active_bones = [b for b in skele_list if tuple(sorted(b)) not in ignore_set]

    if len(active_bones) == 0:
        F, I, _ = pred_data_array.shape
        K = pred_data_array.shape[2] // 3
        return np.zeros((F, I, K), dtype=bool) if kp_mode else np.zeros((F, I), dtype=bool)

    bones = np.array(active_bones)
    B = bones.shape[0]

    F, I, D = pred_data_array.shape
    K = D // 3

    data = pred_data_array.reshape(F, I, K, 3)
    xy = data[..., :2]

    p1 = xy[:, :, bones[:, 0], :]
    p2 = xy[:, :, bones[:, 1], :]
    bone_lengths = np.linalg.norm(p2 - p1, axis=-1)

    valid_p1 = ~np.isnan(p1).any(axis=-1)
    valid_p2 = ~np.isnan(p2).any(axis=-1)
    valid_bones = valid_p1 & valid_p2

    bone_lengths_flat = bone_lengths.reshape(-1, B)
    valid_bones_flat = valid_bones.reshape(-1, B)
    reference_lengths = np.full(B, np.nan)
    for b in range(B):
        vals = bone_lengths_flat[valid_bones_flat[:, b], b]
        if len(vals) > 0:
            reference_lengths[b] = np.percentile(vals, 75)
        else:
            reference_lengths[b] = 1.0
    reference_lengths = np.clip(reference_lengths, 1e-8, None)

    rel_lengths = bone_lengths / reference_lengths[None, None, :]
    bad_bones = (rel_lengths > threshold_max) & valid_bones

    if not kp_mode:
        return np.any(bad_bones, axis=2)

    bones_flat = bones.ravel()
    degree = np.bincount(bones_flat, minlength=K)
    is_leaf = (degree == 1)

    bad_kp_count = np.zeros((F, I, K), dtype=int)
    for b, (i1, i2) in enumerate(bones):
        mask = bad_bones[:, :, b]
        bad_kp_count[:, :, i1] += mask
        bad_kp_count[:, :, i2] += mask

    outlier_kp = np.zeros((F, I, K), dtype=bool)
    for k in range(K):
        if is_leaf[k]:
            outlier_kp[:, :, k] = (bad_kp_count[:, :, k] >= 1)
        else:
            outlier_kp[:, :, k] = (bad_kp_count[:, :, k] >= 2)

    return outlier_kp

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

def outlier_envelop(
    pred_data_array: np.ndarray,
    padding: int = 20,
):
    F, I, D = pred_data_array.shape
    K = D // 3

    envelop_mask = np.zeros((F, I), dtype=bool)

    _, local_coords = calculate_pose_centroids(pred_data_array)
    with bye_bye_runtime_warning():
        local_coords_reshaped = local_coords.reshape(F, I, K, 2)
        sizes = np.nanmax(local_coords_reshaped[..., 0]**2 + local_coords_reshaped[..., 1]**2, axis=-1)

    for inst_1_idx, inst_2_idx in combinations(range(I), 2):
        inst_1_larger = sizes[:, inst_1_idx] >= sizes[:, inst_2_idx] # (F,)
        inst_2_larger = sizes[:, inst_1_idx] < sizes[:, inst_2_idx]

        inst_1_x = pred_data_array[:, inst_1_idx, 0::3]
        inst_1_y = pred_data_array[:, inst_1_idx, 1::3]
        inst_2_x = pred_data_array[:, inst_2_idx, 0::3]
        inst_2_y = pred_data_array[:, inst_2_idx, 1::3]

        inst_bbox = np.zeros((F, 4))
        inst_bbox[inst_1_larger] = np.transpose(calculate_pose_bbox(inst_1_x, inst_1_y, padding))[inst_1_larger]
        inst_bbox[inst_2_larger] = np.transpose(calculate_pose_bbox(inst_2_x, inst_2_y, padding))[inst_2_larger]

        inst_x = inst_1_x.copy()
        inst_y = inst_1_y.copy()
        inst_x[inst_1_larger] = inst_2_x[inst_1_larger]
        inst_y[inst_1_larger] = inst_2_y[inst_1_larger]

        enveloped = np.all((
            (inst_x > inst_bbox[:, 0:1]) & (inst_y > inst_bbox[:, 1:2]) & (inst_x < inst_bbox[:, 2:3]) & (inst_y < inst_bbox[:, 3:4]) |
            (np.isnan(inst_x) & np.isnan(inst_y))
            ), axis=-1)

        inst_1_enveloped = inst_2_larger & enveloped
        inst_2_enveloped = inst_1_larger & enveloped

        envelop_mask[inst_1_enveloped, inst_1_idx] = True
        envelop_mask[inst_2_enveloped, inst_2_idx] = True

    return envelop_mask

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

def outlier_flicker(pred_data_array: np.ndarray) -> np.ndarray:
    all_x, all_y = pred_data_array[:, :, 0::3], pred_data_array[:, :, 1::3]
    presence = np.any(~np.isnan(all_x) & ~np.isnan(all_y), axis=2)

    instance_count = presence.shape[1]

    prev_presence = np.concatenate([np.zeros((1, instance_count), dtype=bool), presence[:-1]], axis=0)
    next_presence = np.concatenate([presence[1:], np.zeros((1, instance_count), dtype=bool)], axis=0)
    flicker_mask = presence & (~prev_presence) & (~next_presence)

    return flicker_mask