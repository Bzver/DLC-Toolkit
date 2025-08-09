import numpy as np

from typing import List, Optional, Tuple
from numpy.typing import NDArray

from . import dtu_helper as duh

def delete_track(pred_data_array:NDArray, current_frame_idx:int, roi_frame_list:List[int], selected_instance_idx:int,
                mode:str="point", deletion_range:Optional[List[int]]=None) -> Optional[NDArray]:
    
    if mode == "point": # Only removing the current frame
        frames_to_delete = current_frame_idx

    elif mode == "range": # Remove the frame of supplied range
        if deletion_range:
            frames_to_delete = deletion_range
        else:
            raise ValueError("Deletion range must be provided for 'range' mode.")

    else: # Remove until the next frame swap
        next_roi_frame_idx = duh.get_next_frame_in_list(roi_frame_list, current_frame_idx)
        if next_roi_frame_idx: 
            frames_to_delete = range(current_frame_idx, next_roi_frame_idx)
        else:
            frames_to_delete = range(current_frame_idx, pred_data_array.shape[0])

    pred_data_array[frames_to_delete, selected_instance_idx, :] = np.nan

    return pred_data_array

def swap_track(pred_data_array:NDArray, current_frame_idx:int, mode:str="point",
               swap_range:Optional[List[int]]=None) -> Optional[NDArray]:
    
    if mode == "point":
        frames_to_swap = current_frame_idx

    elif mode == "range":
        if swap_range:
            frames_to_swap = swap_range
        else:
            raise ValueError("Swap range must be provided for 'range' mode.")

    else:  # Swap until the end of time
        frames_to_swap = range(current_frame_idx, pred_data_array.shape[0])

    pred_data_array[frames_to_swap, 0, :], pred_data_array[frames_to_swap, 1, :] = \
    pred_data_array[frames_to_swap, 1, :].copy(), pred_data_array[frames_to_swap, 0, :].copy()

    return pred_data_array

def interpolate_track(pred_data_array:NDArray, frames_to_interpolate:List[int], selected_instance_idx:int) -> NDArray:
    start_frame_for_interpol = frames_to_interpolate[0] - 1
    end_frame_for_interpol = frames_to_interpolate[-1] + 1

    start_kp_data = pred_data_array[start_frame_for_interpol, selected_instance_idx, :]
    end_kp_data = pred_data_array[end_frame_for_interpol, selected_instance_idx, :]
    interpolated_values = np.linspace(start_kp_data, end_kp_data, num=len(frames_to_interpolate)+2, axis=0)
    pred_data_array[start_frame_for_interpol : end_frame_for_interpol + 1, selected_instance_idx, :] = interpolated_values
    
    return pred_data_array
        
def generate_track(pred_data_array:NDArray, current_frame_idx:int, missing_instances:List[int], num_keypoint:int) -> NDArray:
    for instance_idx in missing_instances:
        avg_pose = get_average_pose(pred_data_array, instance_idx, num_keypoint)
        pred_data_array[current_frame_idx, instance_idx, :] = avg_pose

    return pred_data_array

    ###################################################################################################################################################

def purge_by_conf_and_bp(pred_data_array:NDArray, num_keypoint:int,
                         confidence_threshold:float, bodypart_threshold:int) -> Tuple[NDArray, int, int]:
    # Calculate a mask for low confidence instances
    confidence_scores = pred_data_array[:, :, 2:num_keypoint*3:3]
    inst_conf_all = np.nanmean(confidence_scores, axis=2)
    low_conf_mask = inst_conf_all < confidence_threshold

    # Calculate a mask for instances with too few body parts
    discovered_bodyparts = np.sum(confidence_scores > 0.0, axis=2)
    discovery_perc = discovered_bodyparts / num_keypoint
    low_bodypart_mask = np.logical_and(discovery_perc * 100 < bodypart_threshold, discovery_perc > 0)

    # Combine the two masks with a logical OR
    removal_mask = np.logical_or(low_conf_mask, low_bodypart_mask)
    removed_frames_count = np.sum(np.any(removal_mask, axis=1))
    removed_instances_count = np.sum(removal_mask.flatten())

    # Apply the combined mask to set the data to NaN
    f_idx, i_idx = np.where(removal_mask)
    pred_data_array[f_idx, i_idx, :] = np.nan
    return pred_data_array, removed_frames_count, removed_instances_count

    ###################################################################################################################################################

def get_average_pose(pred_data_array:NDArray, selected_instance_idx:int, num_keypoint:int) -> NDArray:
    inst_array = pred_data_array[:, selected_instance_idx:selected_instance_idx+1, :]

    # Purge the data for frames with low confidence or too few body parts
    # This ensures only valid poses contribute to the average.
    inst_array_filtered, removed_frames_count, removed_instances_count = purge_by_conf_and_bp(inst_array, num_keypoint, 0.6, 80)
    print(f'Ignored {removed_instances_count} instances on {removed_frames_count} frames for pose calculation,')
    inst_array_filtered = np.squeeze(inst_array_filtered)

    # Separate coordinates and confidence scores
    conf_scores = inst_array_filtered[:, 2:num_keypoint*3:3]
    relative_x, relative_y, centroid_x, centroid_y = normalize_poses_by_centroid(inst_array_filtered, num_keypoint)
    rotated_relative_x, rotated_relative_y = align_poses_by_vector(relative_x, relative_y)

    avg_relative_x = np.nanmean(rotated_relative_x, axis=0)
    avg_relative_y = np.nanmean(rotated_relative_y, axis=0)
    avg_conf = np.nanmean(conf_scores, axis=0)

    avg_centroid_x, avg_centroid_y = np.nanmean(centroid_x, axis=0), np.nanmean(centroid_y, axis=0)

    avg_absolute_x = avg_relative_x + avg_centroid_x
    avg_absolute_y = avg_relative_y + avg_centroid_y

    average_pose = np.stack([avg_absolute_x, avg_absolute_y, avg_conf], axis=-1).flatten()
    return average_pose

def normalize_poses_by_centroid(inst_array_filtered: NDArray, num_keypoint: int) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Calculates centroids and normalizes poses relative to them."""
    x_coords = inst_array_filtered[:, 0:num_keypoint*3:3]
    y_coords = inst_array_filtered[:, 1:num_keypoint*3:3]
    
    centroid_x = np.nanmean(x_coords, axis=1)
    centroid_y = np.nanmean(y_coords, axis=1) 
    
    relative_x = x_coords - centroid_x[:, np.newaxis]
    relative_y = y_coords - centroid_y[:, np.newaxis]
    
    return relative_x, relative_y, centroid_x, centroid_y

def align_poses_by_vector(relative_x: NDArray, relative_y: NDArray) -> Tuple[NDArray, NDArray]:
    """Rotates all poses to a common orientation using an anchor-reference vector."""
    rmse_from_centroid = np.nanmean(np.sqrt(relative_x**2 + relative_y**2), axis=0)
    anchor_keypoint_idx = np.argmin(rmse_from_centroid)
    ref_keypoint_idx = np.argmax(rmse_from_centroid)
    
    ref_vec_x = relative_x[:, ref_keypoint_idx] - relative_x[:, anchor_keypoint_idx]
    ref_vec_y = relative_y[:, ref_keypoint_idx] - relative_y[:, anchor_keypoint_idx]
    angles = np.arctan2(ref_vec_y, ref_vec_x)
    
    cos_angles, sin_angles = np.cos(-angles), np.sin(-angles)
    rotated_relative_x = relative_x * cos_angles[:, np.newaxis] - relative_y * sin_angles[:, np.newaxis]
    rotated_relative_y = relative_x * sin_angles[:, np.newaxis] + relative_y * cos_angles[:, np.newaxis]
    
    return rotated_relative_x, rotated_relative_y

    ###################################################################################################################################################

def find_segment_for_autocorrect(instance_count_per_frame:List[int], min_segment_length:int=100) -> List[Tuple[int, int]]:
    segments_to_correct = []
    current_segment_start = -1

    for i in range(len(instance_count_per_frame)):
        if instance_count_per_frame[i] <= 1:
            if current_segment_start == -1:
                current_segment_start = i
        else:
            if current_segment_start != -1:
                segment_length = i - current_segment_start
                if segment_length >= min_segment_length:
                    segments_to_correct.append((current_segment_start, i - 1))
                current_segment_start = -1
    
    # Handle the last segment if it extends to the end of the video
    if current_segment_start != -1:
        segment_length = len(instance_count_per_frame) - current_segment_start
        if segment_length >= min_segment_length:
            segments_to_correct.append((current_segment_start, len(instance_count_per_frame) - 1))

    return segments_to_correct

def apply_segmental_autocorrect(pred_data_array:NDArray, instance_count_per_frame:List[int],
                                segments_to_correct:List[Tuple[int, int]]) -> Tuple[NDArray, int]:
        
        num_corrections_applied = 0

        for start_frame, end_frame in segments_to_correct:

            for frame_idx in range(start_frame, end_frame + 1): # Swap non 'instance 0' with 'instance 0' for all frames in the segment
                if instance_count_per_frame[frame_idx] == 0: # Skip swapping for empty predictions
                    continue
                current_present_at_frame = np.where(~np.all(np.isnan(pred_data_array[frame_idx]), axis=1))[0]
                if current_present_at_frame[0] != 0: # Ensure that at this specific frame, the instance to be swapped is not instance 0
                    swap_track(pred_data_array, frame_idx, mode="point")
                last_present_instance = current_present_at_frame[0]

            # Apply the swap from (end_frame + 1) to the end of the video, IF the last instance detected was not 0
            if last_present_instance is not None and last_present_instance != 0:
                print(f"Applying global swap from frame {end_frame + 1} to end.")
                swap_track(pred_data_array, end_frame + 1, mode="batch")
            
            num_corrections_applied += 1
        
        return pred_data_array, num_corrections_applied

    ###################################################################################################################################################

def track_swap_3D_plotter(pred_data_array:NDArray, frame_idx:int, selected_cam_idx:int) -> NDArray:
    pred_data_array_to_swap = pred_data_array[:, selected_cam_idx, :, :]
    pred_data_array_swapped = swap_track(pred_data_array_to_swap, frame_idx, mode="batch")
    pred_data_array[:, selected_cam_idx, :, :] = pred_data_array_swapped
    return pred_data_array

def clean_inconsistent_nans(pred_data_array:NDArray):
    print("Cleaning up NaN keypoints that somehow has confidence value...")
    nan_mask = np.isnan(pred_data_array)
    x_is_nan = nan_mask[:, :, 0::3]
    y_is_nan = nan_mask[:, :, 1::3]
    keypoints_to_fully_nan = x_is_nan | y_is_nan
    full_nan_sweep_mask = np.repeat(keypoints_to_fully_nan, 3, axis=-1)
    pred_data_array[full_nan_sweep_mask] = np.nan
    print("NaN keypoint confidence cleaned.")
    return pred_data_array