import numpy as np

from typing import Dict, Any, List

from .pose_analysis import calculate_pose_centroids, calculate_pose_rotations
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
        angle_map_data:Dict[str, Any]
        ) -> np.ndarray:
    
    for instance_idx in missing_instances:
        average_pose = get_average_pose(pred_data_array, instance_idx, angle_map_data=angle_map_data, frame_idx=current_frame_idx)
        pred_data_array[current_frame_idx, instance_idx, :] = average_pose

    return pred_data_array

def generate_missing_kp_for_inst(
        pred_data_array:np.ndarray,
        current_frame_idx:int,
        selected_instance_idx:int,
        angle_map_data:Dict[str, any]
        ) -> np.ndarray:
    
    num_keypoint = pred_data_array.shape[2] // 3
    missing_keypoints = []
    for keypoint_idx in range(num_keypoint):
        confidence_idx = keypoint_idx * 3 + 2
        confidence = pred_data_array[current_frame_idx, selected_instance_idx, confidence_idx]
        if np.isnan(confidence):
            missing_keypoints.append(keypoint_idx)

    if not missing_keypoints:
        return pred_data_array

    current_frame_centroids, local_coords = calculate_pose_centroids(pred_data_array, current_frame_idx)
    set_centroid = current_frame_centroids[selected_instance_idx, :]
    local_inst_x = local_coords[selected_instance_idx, 0::2]
    local_inst_y = local_coords[selected_instance_idx, 1::2]
    set_rotation = calculate_pose_rotations(local_inst_x, local_inst_y, angle_map_data=angle_map_data)
    average_pose = get_average_pose(pred_data_array, selected_instance_idx, angle_map_data=angle_map_data, frame_idx=current_frame_idx, 
        initial_pose_range=10, max_attempts=20, valid_frames_threshold=100, set_centroid=set_centroid, set_rotation=set_rotation)
    
    # Interpolate keypoint coordinates
    for keypoint_idx in missing_keypoints:
        try:
            pred_data_array[current_frame_idx, selected_instance_idx, keypoint_idx*3:keypoint_idx*3 +3] = average_pose[keypoint_idx*3:keypoint_idx*3 + 3]
        except ValueError:
            # get_average_pose failed, fallback to centroid values
            pred_data_array[current_frame_idx, selected_instance_idx, keypoint_idx*3] = set_centroid[0]
            pred_data_array[current_frame_idx, selected_instance_idx, keypoint_idx*3+1] = set_centroid[1]
            pred_data_array[current_frame_idx, selected_instance_idx, keypoint_idx*3+2] = 1.0

    return pred_data_array