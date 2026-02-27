import numpy as np
from typing import List


def get_instances_on_current_frame(pred_data_array:np.ndarray, current_frame_idx:int) -> List[int]:
    """
    Identifies which instances are present in a given frame based on non-NaN keypoint data.

    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence) for each keypoint.
        current_frame_idx (int): Index of the frame to check.

    Returns:
        List[int]: List of instance indices that have at least one valid keypoint 
                   in the specified frame.
    """
    instance_count = pred_data_array.shape[1]
    current_frame_inst = []
    for inst_idx in range(instance_count):
        if np.any(~np.isnan(pred_data_array[current_frame_idx, inst_idx, :])):
            current_frame_inst.append(inst_idx)
    return current_frame_inst

def get_instance_count_per_frame(pred_data_array:np.ndarray) -> np.ndarray:
    """
    Count the number of non-empty instances per frame.
    
    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence) for each keypoint.
    
    Returns:
        Array of shape (n_frames,) with count of valid instances per frame.
    """
    non_empty_instance_numerical = (np.any(~np.isnan(pred_data_array), axis=2)) * 1
    instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
    return instance_count_per_frame