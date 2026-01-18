import numpy as np

from typing import Dict, List, Optional

from .pose_analysis import calculate_pose_centroids
from .pose_worker import pose_rotation_worker
from .pose_average import get_average_pose


def rotate_selected_inst(
        pred_data_array:np.ndarray,
        frame_idx:int,
        selected_instance_idx:int,
        angle:float
        ) -> np.ndarray:
    
    conf_scores = pred_data_array[frame_idx, selected_instance_idx, 2::3]
    pose_centroids, local_coords = calculate_pose_centroids(pred_data_array, frame_idx)
    pose_centroids = pose_centroids[selected_instance_idx, :]
    local_coords = local_coords[selected_instance_idx, :]
    pose_rotated = pose_rotation_worker(angle, pose_centroids, local_coords, conf_scores)
    pred_data_array[frame_idx, selected_instance_idx, :] = pose_rotated
    return pred_data_array

def generate_missing_inst(
        pred_data_array:np.ndarray,
        current_frame_idx:int,
        missing_instances:List[int],
        angle_map_data:Dict[str, int],
        canon_pose:Optional[np.ndarray]=None
        ) -> np.ndarray:
    
    num_keypoint = pred_data_array.shape[2] // 3
    for instance_idx in missing_instances:
        try:
            average_pose = get_average_pose(
                pred_data_array, instance_idx, angle_map_data=angle_map_data, frame_idx=current_frame_idx)
        except:
            canon_pose = canon_pose.flatten()
            set_rotation = float(instance_idx)
            set_centroid = np.full((1,2),200)
            average_pose = pose_rotation_worker(set_rotation, set_centroid, canon_pose, np.full(num_keypoint, 1.0))

        pred_data_array[current_frame_idx, instance_idx, :] = average_pose

    return pred_data_array

def generate_missing_kp_for_inst(
        pred_data_array:np.ndarray,
        current_frame_idx:int,
        selected_instance_idx:int,
        canon_pose:np.ndarray,
        ) -> np.ndarray:
    num_keypoint = pred_data_array.shape[2] // 3
    current_inst_data = pred_data_array[current_frame_idx, selected_instance_idx, :].reshape([num_keypoint, 3])
    non_nan_mask = np.all(~np.isnan(current_inst_data), axis=1)
    non_nan_indices = np.where(non_nan_mask)[0]
    
    non_nan_kp = current_inst_data[non_nan_indices, :2]
    canon_observed = canon_pose[non_nan_indices]

    R, t = _fit_rigid_transform(canon_observed, non_nan_kp)

    transformed_canon = (R @ canon_pose.T).T + t

    completed_kp = current_inst_data.copy()
    missing_mask = ~non_nan_mask
    completed_kp[missing_mask, :2] = transformed_canon[missing_mask]
    completed_kp[missing_mask, 2] = 1.0

    pred_data_array[current_frame_idx, selected_instance_idx] = completed_kp.flatten()

    return pred_data_array

def _fit_rigid_transform(src: np.ndarray, dst: np.ndarray):
    assert src.shape == dst.shape and src.shape[0] >= 2

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean

    return R, t
