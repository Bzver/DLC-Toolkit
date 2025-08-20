import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

from typing import List, Optional, Tuple

from . import dtu_helper as duh

def delete_track(pred_data_array:np.ndarray, current_frame_idx:int, roi_frame_list:List[int], selected_instance_idx:int,
                mode:str="point", deletion_range:Optional[List[int]]=None) -> Optional[np.ndarray]:
    
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

def swap_track(pred_data_array:np.ndarray, current_frame_idx:int, mode:str="point",
               swap_range:Optional[List[int]]=None) -> Optional[np.ndarray]:
    
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

def interpolate_track(pred_data_array:np.ndarray, frames_to_interpolate:List[int], selected_instance_idx:int) -> np.ndarray:
    start_frame_for_interpol = frames_to_interpolate[0] - 1
    end_frame_for_interpol = frames_to_interpolate[-1] + 1

    start_kp_data = pred_data_array[start_frame_for_interpol, selected_instance_idx, :]
    end_kp_data = pred_data_array[end_frame_for_interpol, selected_instance_idx, :]
    interpolated_values = np.linspace(start_kp_data, end_kp_data, num=len(frames_to_interpolate)+2, axis=0)
    pred_data_array[start_frame_for_interpol : end_frame_for_interpol + 1, selected_instance_idx, :] = interpolated_values
    
    return pred_data_array
        
def generate_track(pred_data_array:np.ndarray, current_frame_idx:int, missing_instances:List[int]) -> np.ndarray:
    for instance_idx in missing_instances:
        avg_pose = get_average_pose(pred_data_array, instance_idx, frame_idx=current_frame_idx)
        pred_data_array[current_frame_idx, instance_idx, :] = avg_pose

    return pred_data_array

def interpolate_missing_keypoints(pred_data_array:np.ndarray, current_frame_idx:int, selected_instance_idx:int) -> np.ndarray:
    num_keypoint = pred_data_array.shape[2] // 3
    missing_keypoints = []
    for keypoint_idx in range(num_keypoint):
        confidence_idx = keypoint_idx * 3 + 2
        confidence = pred_data_array[current_frame_idx, selected_instance_idx, confidence_idx]
        if np.isnan(confidence) or confidence < 0.1:
            missing_keypoints.append(keypoint_idx)

    if not missing_keypoints:
        return pred_data_array

    # Interpolate keypoint coordinates
    for keypoint_idx in missing_keypoints:
        try:
            average_pose = get_average_pose(pred_data_array, selected_instance_idx, frame_idx=current_frame_idx)
            pred_data_array[current_frame_idx, selected_instance_idx, keypoint_idx * 3:keypoint_idx * 3 + 3] = average_pose[keypoint_idx * 3:keypoint_idx * 3 + 3]
        except ValueError:
            # get_average_pose failed, use default values (0, 0, 1)
            pred_data_array[current_frame_idx, selected_instance_idx, keypoint_idx * 3 : keypoint_idx * 3 + 2] = 0.0
            pred_data_array[current_frame_idx, selected_instance_idx, keypoint_idx * 3 + 2] = 1.0

    return pred_data_array

###################################################################################################################################################

def purge_by_conf_and_bp(pred_data_array:np.ndarray, confidence_threshold:float, bodypart_threshold:int
        ) -> Tuple[np.ndarray, int, int]:
    # Calculate a mask for low confidence instances
    num_keypoint = pred_data_array.shape[2] // 3
    confidence_scores = pred_data_array[:, :, 2::3]
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

def interpolate_track_all(pred_data_array:np.ndarray, selected_instance_idx:int, max_gap:int) -> np.ndarray:
    if max_gap == 0: # Interpolate all gaps regardless of length (no gap limit)
        return interpolation_pd_operator(pred_data_array, selected_instance_idx, slice(None))

    # First find all nan gaps
    all_nans = np.isnan(pred_data_array[:, selected_instance_idx, :])
    all_nan_frames = np.all(all_nans, axis=1)
    partial_nan_frames = np.any(all_nans, axis=1) & ~all_nan_frames
    
    # Interpolate the partial nan frames using average pose
    partial_nan_indices = np.where(partial_nan_frames)[0]
    for frame_idx in partial_nan_indices:
        interpolate_missing_keypoints(pred_data_array, frame_idx, selected_instance_idx)
    
    # Identify gap lengths
    padded = np.concatenate(([False], all_nan_frames, [False]))
    deltas = np.diff(padded.astype(np.int8))  # or np.int64
    gap_starts = np.where(deltas == 1)[0]
    gap_ends = np.where(deltas == -1)[0]

    assert len(gap_starts) == len(gap_ends), f"Mismatched starts and ends. gap_starts: {len(gap_starts)}, gap_ends: {len(gap_ends)}"

    # Build gap DataFrame and filter gaps longer than max_gap
    gap_data = []
    for start, end in zip(gap_starts, gap_ends):
        gap_data.append({'Gap_Start': start, 'Gap_End': end, 'Length': end - start})
    gaps_df = pd.DataFrame(gap_data)

    if gaps_df.empty:
        print(f"No gap found for instance {selected_instance_idx}.")
        return pred_data_array
        
    for _, row in gaps_df.iterrows():
        start, end, length = int(row['Gap_Start']), int(row['Gap_End']), row['Length']
        if length > max_gap:
            continue

        total_frames = pred_data_array.shape[0]
        context = 10
        window_start = max(0, start - context)
        window_end   = min(total_frames, end + context)    
        gap_slice = slice(window_start, window_end)
        pred_data_array = interpolation_pd_operator(pred_data_array, selected_instance_idx, gap_slice)
        
    return pred_data_array

def interpolation_pd_operator(pred_data_array:np.ndarray, selected_instance_idx:int, interpolation_range:slice) -> np.ndarray:
    num_keypoint = pred_data_array.shape[2] // 3
    for kp_idx in range(num_keypoint):
        # Extract x, y, confidence for the current keypoint and instance across all frames
        x_coords = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3]
        y_coords = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+1]
        conf_values = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+2]

        # Convert to pandas Series for interpolation
        x_series = pd.Series(x_coords)
        y_series = pd.Series(y_coords)
        conf_series = pd.Series(conf_values)

        # Interpolate NaNs
        x_interpolated = x_series.interpolate(method='linear', limit_direction='both').values
        y_interpolated = y_series.interpolate(method='linear', limit_direction='both').values
        conf_interpolated = conf_series.interpolate(method='linear', limit_direction='both').values

        # Update the pred_data_array
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3] = x_interpolated
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+1] = y_interpolated
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+2] = conf_interpolated
    
    return pred_data_array

###################################################################################################################################################

def get_pose_window(frame_idx:int, total_frames:int, pose_range:int):
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

    return list(range(min_frame, max_frame))

def get_average_pose(pred_data_array:np.ndarray, selected_instance_idx:int, frame_idx:int) -> np.ndarray:
    pose_range = 30
    pose_window = get_pose_window(frame_idx, len(pred_data_array), pose_range)
    inst_array = pred_data_array[pose_window, selected_instance_idx:selected_instance_idx+1, :]

    # Purge the data for frames with low confidence or too few body parts
    inst_array_filtered, _, _ = purge_by_conf_and_bp(inst_array, 0.6, 80)
    non_purged_frame_count = duh.get_non_completely_nan_slice_count(inst_array_filtered)

    max_attempts = 10
    attempt = 0
    while non_purged_frame_count <= 50 and attempt < max_attempts:
        pose_range *= 2
        pose_window = get_pose_window(frame_idx, len(pred_data_array), pose_range)
        inst_array = pred_data_array[pose_window, selected_instance_idx:selected_instance_idx+1, :]
        inst_array_filtered, _, _ = purge_by_conf_and_bp(inst_array, 0.6, 80)
        non_purged_frame_count = duh.get_non_completely_nan_slice_count(inst_array_filtered)
        attempt += 1

    if non_purged_frame_count <= 50:
        raise ValueError("Not enough valid frames to compute average pose, even after expanding window.")

    inst_array_filtered = np.squeeze(inst_array_filtered)

    # Separate coordinates and confidence scores
    conf_scores = inst_array_filtered[:, 2::3]
    relative_x, relative_y, centroid_x, centroid_y = normalize_poses_by_centroid(inst_array_filtered)
    rotated_relative_x, rotated_relative_y, rotation_angles = align_poses_by_vector(relative_x, relative_y)

    avg_relative_x = np.nanmean(rotated_relative_x, axis=0)
    avg_relative_y = np.nanmean(rotated_relative_y, axis=0)
    avg_conf = np.nanmean(conf_scores, axis=0)
    avg_angle = duh.circular_mean(rotation_angles)
    cos_a, sin_a = np.cos(avg_angle), np.sin(avg_angle)

    avg_centroid_x, avg_centroid_y = np.nanmean(centroid_x, axis=0), np.nanmean(centroid_y, axis=0)

    avg_absolute_x = avg_relative_x * cos_a - avg_relative_y * sin_a + avg_centroid_x
    avg_absolute_y = avg_relative_x * sin_a + avg_relative_y * cos_a + avg_centroid_y

    average_pose = np.stack([avg_absolute_x, avg_absolute_y, avg_conf], axis=-1).flatten()
    return average_pose

def normalize_poses_by_centroid(inst_array_filtered: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates centroids and normalizes poses relative to them."""
    x_coords = inst_array_filtered[:, 0::3]
    y_coords = inst_array_filtered[:, 1::3]
    
    centroid_x = np.nanmean(x_coords, axis=1)
    centroid_y = np.nanmean(y_coords, axis=1) 
    
    relative_x = x_coords - centroid_x[:, np.newaxis]
    relative_y = y_coords - centroid_y[:, np.newaxis]
    
    return relative_x, relative_y, centroid_x, centroid_y

def align_poses_by_vector(relative_x: np.ndarray, relative_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    return rotated_relative_x, rotated_relative_y, angles

###################################################################################################################################################

def idt_track_correction(pred_data_array:np.ndarray, idt_traj_array:np.ndarray, progress) -> Tuple[np.ndarray, int]:
    assert pred_data_array.shape[1] == idt_traj_array.shape[1], "Instance count must match between prediction and idTracker"

    total_frames, instance_count, _ = pred_data_array.shape
    pred_positions = np.full((total_frames, instance_count, 2), np.nan)
    x_vals = pred_data_array[:, :, 0::3]
    y_vals = pred_data_array[:, :, 1::3]
    pred_positions[:, :, 0] = np.nanmean(x_vals, axis=2)
    pred_positions[:, :, 1] = np.nanmean(y_vals, axis=2)

    remapped_idt = remap_idt_array(pred_positions, idt_traj_array)
    corrected_pred_data = pred_data_array.copy()
    pred_position_curr = np.full((instance_count, 2), np.nan)

    last_order = list(range(instance_count))
    changes_applied = 0
    for frame_idx in range(total_frames):
        progress.setValue(frame_idx)
        if progress.wasCanceled():
            return pred_data_array, 0
        
        # Trying last order first no matter what
        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, last_order, :]

        x_curr = corrected_pred_data[frame_idx, :, 0::3]
        y_curr = corrected_pred_data[frame_idx, :, 1::3]
        pred_position_curr[:, 0] = np.nanmean(x_curr, axis=1)
        pred_position_curr[:, 1] = np.nanmean(y_curr, axis=1)

        valid_pred_curr = np.all(~np.isnan(pred_position_curr), axis=1)
        valid_idt_curr = np.all(~np.isnan(remapped_idt[frame_idx]), axis=1) 

        n_pred = np.sum(valid_pred_curr)
        n_idt  = np.sum(valid_idt_curr)
        if n_pred != n_idt or n_pred == 0 or n_idt == 0:
            continue

        if n_pred == 1:
            valid_pred_inst = np.where(valid_pred_curr)[0][0]
            valid_idt_inst = np.where(valid_idt_curr)[0][0]
            if valid_pred_inst == valid_idt_inst:
                continue

            new_order = list(range(instance_count))
            new_order[valid_pred_inst] = valid_idt_inst
            new_order[valid_idt_inst] = valid_pred_inst

            corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, new_order, :]
            last_order = new_order
            
            print(f"Swap instance {valid_pred_inst} to instance {valid_idt_inst} from frame {frame_idx}.")

            changes_applied += 1
            continue

        valid_positions_pred = pred_position_curr[valid_pred_curr]
        valid_positions_idt  = remapped_idt[frame_idx][valid_idt_curr]

        distances = np.linalg.norm(valid_positions_pred - valid_positions_idt, axis=1)
        max_dist = 20.0

        if np.all(distances < max_dist):
            continue

        cost_matrix = np.linalg.norm(valid_positions_pred[:, np.newaxis, :] - valid_positions_idt[np.newaxis, :, :], axis=2)
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except:
            continue

        new_order = np.arange(instance_count)
        pred_indices = np.where(valid_pred_curr)[0]
        idt_indices  = np.where(valid_idt_curr)[0]

        for i, j in zip(row_ind, col_ind):
            new_order[idt_indices[j]] = pred_indices[i]

        new_order = new_order.tolist()
        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, new_order, :]
        last_order = new_order
        print(f"Rearrange order of instances from frame {frame_idx} onwards. New order: {new_order}")
        changes_applied += 1
        
    return corrected_pred_data, changes_applied

def velocity_track_correction(pred_data_array:np.ndarray, progress) -> np.ndarray: # Unused
    total_frames, instance_counts, _ = pred_data_array.shape
    avg_velocity_array = np.full((total_frames, instance_counts), np.nan)

    for frame_idx in range(total_frames):
        progress.setValue(frame_idx)
        if progress.wasCanceled():
            return avg_velocity_array
        
        avg_velocity_array[frame_idx] = duh.calculate_temporal_velocity_2d(pred_data_array, frame_idx)

    return avg_velocity_array

def remap_idt_array(pred_positions:np.ndarray, idt_traj_array:np.ndarray) -> np.ndarray:
    total_frames, _, _ = pred_positions.shape

    remapped_idt = idt_traj_array.copy()
    valid_pred = np.all(~np.isnan(pred_positions), axis=2)
    valid_idt = np.all(~np.isnan(idt_traj_array), axis=2) 

    valid_frame_idx = None
    for frame_idx in range(total_frames):
        n_pred = np.sum(valid_pred[frame_idx])
        n_idt  = np.sum(valid_idt[frame_idx])
        if n_pred == n_idt and n_pred > 0:
            valid_frame_idx = frame_idx
            break

    if valid_frame_idx is None:
        raise ValueError("No frame with valid data in both arrays.")
    
    valid_pred_indices = np.where(valid_pred[valid_frame_idx])[0]
    valid_idt_indices = np.where(valid_idt[valid_frame_idx])[0] 

    if np.sum(valid_pred[valid_frame_idx]) == 1:
        valid_pred_inst = valid_pred_indices[0]
        valid_idt_inst = valid_idt_indices[0]
        remapped_idt[:, valid_pred_inst, :], remapped_idt[:, valid_idt_inst, :] = \
            idt_traj_array[:, valid_idt_inst, :], idt_traj_array[:, valid_pred_inst, :]
        
        print(f"Successfully remapped idtracker trajectory using frame {valid_frame_idx}.")
        print(f"Mapping: idTracker instance {valid_idt_inst} → Prediction instance {valid_pred_inst}")

        return remapped_idt
        
    p_pred = pred_positions[valid_frame_idx]
    p_idt = idt_traj_array[valid_frame_idx]

    cost_matrix = np.linalg.norm(p_pred[:, np.newaxis, :] - p_idt[np.newaxis, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping_details = []
    for pred_local_idx, idt_local_idx in zip(row_ind, col_ind):
        pred_global_idx = valid_pred_indices[pred_local_idx]
        idt_global_idx = valid_idt_indices[idt_local_idx]
        remapped_idt[:, pred_global_idx, :] = idt_traj_array[:, idt_global_idx, :]
        mapping_details.append((idt_global_idx, pred_global_idx))

    # Optional: sort by prediction index for cleaner output
    mapping_details.sort(key=lambda x: x[1])

    print(f"Successfully remapped idtracker trajectory using frame {valid_frame_idx}.")
    print(f"Matched using Hungarian algorithm (total cost: {cost_matrix[row_ind, col_ind].sum():.3f}):")
    for idt_inst, pred_inst in mapping_details:
        print(f"  idTracker instance {idt_inst:2d} → Prediction instance {pred_inst:2d}")

    return remapped_idt

###################################################################################################################################################

def track_swap_3D_plotter(pred_data_array:np.ndarray, frame_idx:int, selected_cam_idx:int) -> np.ndarray:
    pred_data_array_to_swap = pred_data_array[:, selected_cam_idx, :, :]
    pred_data_array_swapped = swap_track(pred_data_array_to_swap, frame_idx, mode="batch")
    pred_data_array[:, selected_cam_idx, :, :] = pred_data_array_swapped
    return pred_data_array

def clean_inconsistent_nans(pred_data_array:np.ndarray):
    print("Cleaning up NaN keypoints that somehow has confidence value...")
    nan_mask = np.isnan(pred_data_array)
    x_is_nan = nan_mask[:, :, 0::3]
    y_is_nan = nan_mask[:, :, 1::3]
    keypoints_to_fully_nan = x_is_nan | y_is_nan
    full_nan_sweep_mask = np.repeat(keypoints_to_fully_nan, 3, axis=-1)
    pred_data_array[full_nan_sweep_mask] = np.nan
    print("NaN keypoint confidence cleaned.")
    return pred_data_array