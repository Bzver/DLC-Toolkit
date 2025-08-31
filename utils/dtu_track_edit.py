import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from typing import List, Optional, Tuple, Dict

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
        
def generate_track(pred_data_array:np.ndarray, current_frame_idx:int, missing_instances:List[int],
        angle_map_data:Dict[str, any]) -> np.ndarray:
    for instance_idx in missing_instances:
        average_pose = get_average_pose(pred_data_array, instance_idx, angle_map_data=angle_map_data, frame_idx=current_frame_idx)
        pred_data_array[current_frame_idx, instance_idx, :] = average_pose

    return pred_data_array

def rotate_track(pred_data_array:np.ndarray, frame_idx:int, selected_instance_idx:int, angle:float) -> np.ndarray:
    conf_scores = pred_data_array[frame_idx, selected_instance_idx, 2::3]
    pose_centroids, local_coords = duh.calculate_pose_centroids(pred_data_array, frame_idx)
    pose_centroids = pose_centroids[selected_instance_idx, :]
    local_coords = local_coords[selected_instance_idx, :]
    pose_rotated = duh.track_rotation_worker(angle, pose_centroids, local_coords, conf_scores)
    pred_data_array[frame_idx, selected_instance_idx, :] = pose_rotated
    return pred_data_array

def interpolate_missing_keypoints(pred_data_array:np.ndarray, current_frame_idx:int, selected_instance_idx:int,
        angle_map_data:Dict[str, any]) -> np.ndarray:
    num_keypoint = pred_data_array.shape[2] // 3
    missing_keypoints = []
    for keypoint_idx in range(num_keypoint):
        confidence_idx = keypoint_idx * 3 + 2
        confidence = pred_data_array[current_frame_idx, selected_instance_idx, confidence_idx]
        if np.isnan(confidence):
            missing_keypoints.append(keypoint_idx)

    if not missing_keypoints:
        return pred_data_array

    current_frame_centroids, local_coords = duh.calculate_pose_centroids(pred_data_array, current_frame_idx)
    set_centroid = current_frame_centroids[selected_instance_idx, :]
    local_inst_x = local_coords[selected_instance_idx, 0::2]
    local_inst_y = local_coords[selected_instance_idx, 1::2]
    set_rotation = duh.calculate_pose_rotations(local_inst_x, local_inst_y, angle_map_data=angle_map_data)
    average_pose = get_average_pose(pred_data_array, selected_instance_idx, angle_map_data=angle_map_data, frame_idx=current_frame_idx, 
        initial_pose_range=10, max_attempts=20, valid_frames_threshold=100, confidence_threshold=0.8, bodypart_threshold=90,
        set_centroid=set_centroid, set_rotation=set_rotation)
    
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

###################################################################################################################################################

def get_instance_count_per_frame(pred_data_array:np.ndarray) -> np.ndarray:
    """
    Count the number of non-empty instances per frame.
    
    Args:
        pred_data_array: Shape (n_frames, n_individuals, n_keypoints * 3)
                         Last dim: [x, y, confidence, ...]
    
    Returns:
        Array of shape (n_frames,) with count of valid instances per frame.
    """
    nan_mask = np.isnan(pred_data_array)
    empty_instance = np.all(nan_mask, axis=2)
    non_empty_instance_numerical = (~empty_instance) * 1
    instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
    return instance_count_per_frame

def filter_by_conf_bp_instance(pred_data_array:np.ndarray, confidence_threshold:float, bodypart_threshold:int,
        instance_threshold:int, use_or:Optional[bool]=True, return_frame_list:Optional[bool]=False):
    """
    Filter frames where detection quality is low.
    
    Conditions:
      - Low average confidence
      - Too few body parts detected (below bodypart_threshold %)
      - Too few animal instances (below instance_threshold)

    Logic:
      - use_or=True: mark if ANY condition is True (weaker filter)
      - use_or=False: mark if ALL conditions are True (stricter filter)

    Args:
        pred_data_array: Shape (n_frames, n_individuals, n_keypoints * 3)
        confidence_threshold: e.g., 0.5 → avg confidence < 0.5 → low
        bodypart_threshold: percentage (0–100), below which frame is flagged
        instance_threshold: minimum number of instances expected per frame
        use_or: If True, use OR logic; else use AND
        return_frame_list: If True, return list of frames that fits the criteria,
            else return instance-specific mask (n_frames, n_instances)
    """

    _, I, K = pred_data_array.shape
    n_keypoints = K//3
    n_individuals = I

    # Calculate a mask for low confidence instances
    confidence_scores = pred_data_array[:, :, 2::3]
    inst_conf_all = np.nanmean(confidence_scores, axis=2)
    low_conf_mask = inst_conf_all < confidence_threshold
    
    # Calculate a mask for instances with too few body parts
    discovered_bodyparts = np.sum(confidence_scores > 0.0, axis=2)
    discovery_perc = discovered_bodyparts / n_keypoints
    low_bodypart_mask = np.logical_and(discovery_perc * 100 < bodypart_threshold, discovery_perc > 0)

    # Calculate a frame-level mask for frames with too little instance spotted
    instance_count_per_frame = get_instance_count_per_frame(pred_data_array)
    low_instance_mask_frame = instance_count_per_frame < instance_threshold
    # Broadcast to all instances: if frame is bad, all instances are flagged
    low_instance_mask = np.tile(low_instance_mask_frame[:, np.newaxis], (1, n_individuals)) 
    
    if use_or:
        combined_mask = low_conf_mask | low_bodypart_mask | low_instance_mask
    else:
        combined_mask = low_conf_mask & low_bodypart_mask & low_instance_mask

    if return_frame_list:
        frame_mask = np.any(combined_mask, axis=1)
        return np.where(frame_mask)[0].tolist() 

    return combined_mask

def purge_by_conf_and_bp(pred_data_array:np.ndarray, confidence_threshold:float, bodypart_threshold:int
        ) -> Tuple[np.ndarray, int, int]:
    removal_mask = filter_by_conf_bp_instance(pred_data_array, confidence_threshold, bodypart_threshold, 0)
    removed_frames_count = np.sum(np.any(removal_mask, axis=1))
    removed_instances_count = np.sum(removal_mask.flatten())

    f_idx, i_idx = np.where(removal_mask)  # Apply the combined mask to set the data to NaN
    pred_data_array[f_idx, i_idx, :] = np.nan
    return pred_data_array, removed_frames_count, removed_instances_count

def interpolate_track_all(pred_data_array:np.ndarray, selected_instance_idx:int, max_gap:int) -> np.ndarray:
    if max_gap == 0: # Interpolate all gaps regardless of length (no gap limit)
        return interpolation_pd_operator(pred_data_array, selected_instance_idx, slice(None))

    # First find all nan gaps
    all_nans = np.isnan(pred_data_array[:, selected_instance_idx, :])
    all_nan_frames = np.all(all_nans, axis=1)
    
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

def get_pose_window(frame_idx:int, total_frames:int, pose_range:int) -> slice:
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

    return slice(min_frame, max_frame)

def get_average_pose(
        pred_data_array:np.ndarray, selected_instance_idx:int, frame_idx:int, 
        angle_map_data:Dict[str, any],
        initial_pose_range:int = 30,
        confidence_threshold:float = 0.6,
        bodypart_threshold:int = 80,
        max_attempts:int = 10,
        valid_frames_threshold:int = 30,
        set_centroid:Optional[np.ndarray]=None,
        set_rotation:Optional[float]=None
        ) -> np.ndarray:
    """
    Compute a rotation-normalized average pose for a specific instance around a given frame.

    Args:
        pred_data_array: (F, I, K*3)
        selected_instance_idx: Index of the instance to average
        frame_idx: Center frame of temporal window
        initial_pose_range: Initial half-window size (± frames)
        confidence_threshold: Min confidence to count a keypoint
        bodypart_threshold: Min valid keypoints percentage per frame to keep
        max_attempts: Max times to double window size
        valid_frames_threshold: Min number of valid frames required
        set_centroid: Optional 2D array of shape (2,) to set a specific centroid for alignment
        set_rotation: Optional float to set a specific angle for alignment
    Returns:
        average_pose: (K*3,) array — mean pose in aligned orientation
    """
    pose_range = initial_pose_range
    attempt = 0

    while attempt < max_attempts:
        pose_window = get_pose_window(frame_idx, len(pred_data_array), pose_range)
        pred_data_sliced = pred_data_array[pose_window].copy()  # Shape: (W, N, D)

        # Filter frames based on confidence and body part count
        pred_data_filtered, _, _ = purge_by_conf_and_bp(pred_data_sliced, confidence_threshold, bodypart_threshold)

        # Extract only the selected instance for frame validity check
        inst_slice = pred_data_filtered[:, selected_instance_idx:selected_instance_idx+1, :]
        non_purged_frame_count = duh.get_non_completely_nan_slice_count(inst_slice)

        if non_purged_frame_count > valid_frames_threshold:
            break

        pose_range *= 2
        attempt += 1
    else:
        raise ValueError(f"Only {non_purged_frame_count} valid frames found for instance {selected_instance_idx} "
            f"around frame {frame_idx}, less than required {valid_frames_threshold}, ")

    inst_data = pred_data_filtered[:, selected_instance_idx, :]  # (W', K*3)
    conf_scores = inst_data[:, 2::3]  # (W', K)

    centroids, local_coords = duh.calculate_pose_centroids(inst_data[:, np.newaxis, :]) # inst data is already sliced, no need to slice again

    centroids = np.squeeze(centroids, axis=1)  # (W', 2)
    local_coords = np.squeeze(local_coords, axis=1)  # (W', K*2)
    rotation_angles = duh.calculate_pose_rotations(local_coords[:, 0::2], local_coords[:, 1::2], angle_map_data=angle_map_data)
    aligned_local = duh.align_poses_by_vector(local_coords, rotation_angles)
    avg_angle = duh.circular_mean(rotation_angles)
    if set_centroid is not None : # If a specific centroid is provided, use it for alignment
        centroids = set_centroid
    if set_rotation is not None and ~np.isnan(set_rotation): # If a specific angle is provided, use it for alignment
        avg_angle = set_rotation
    average_pose = duh.track_rotation_worker(avg_angle, centroids, aligned_local, conf_scores)

    return average_pose

###################################################################################################################################################

def ghost_prediction_buster(pred_data_array:np.ndarray, canon_pose:np.ndarray,
        ghost_threshold_bp:float=0.7, ghost_threshold_dist:float=5.0, debug_print:Optional[bool]=False
        ) -> np.ndarray:
    """
    Filter out ghost predictions: i.e. identical predictions in the exact same coordinates.
    And those suspicious bunch which has high possibility of being stemmed from the same instance.
    """
    total_frames, instance_count, xyconf = pred_data_array.shape
    num_keypoints = xyconf // 3
    instances = list(range(instance_count))
    xy_indices = [i for i in range(num_keypoints*3) if i % 3 != 2]

    # Step 1: Trial the most brazenly guilty
    for inst_1_idx, inst_2_idx in combinations(instances, 2):
        inst_1_coords = pred_data_array[:, inst_1_idx, xy_indices]
        inst_2_coords = pred_data_array[:, inst_2_idx, xy_indices]

        if np.all(np.isnan(inst_1_coords)) or np.all(np.isnan(inst_2_coords)):
            continue

        inst_1_conf = np.nanmean(pred_data_array[:, inst_1_idx, 2::3], axis=1)
        inst_2_conf = np.nanmean(pred_data_array[:, inst_2_idx, 2::3], axis=1)

        pose_diff_x = inst_1_coords[:, 0::2] - inst_2_coords[:, 0::2]
        pose_diff_y = inst_1_coords[:, 1::2] - inst_2_coords[:, 1::2]

        euclidean_dist = np.sqrt(pose_diff_x**2 + pose_diff_y**2)

        matching_keypoints = np.sum(euclidean_dist <= ghost_threshold_dist, axis=1)
        ghost_mask = matching_keypoints >= int(num_keypoints * ghost_threshold_bp)
        ghost_list = np.where(ghost_mask)[0].tolist()

        for frame_idx in ghost_list:
            inst_to_delete = inst_2_idx if inst_1_conf[frame_idx] >= inst_2_conf[frame_idx] else inst_1_idx
            pred_data_array[frame_idx, inst_to_delete, :] = np.nan

        if ghost_list: # Get matching stats for the guilty frames
            print(f"Ghost prediction detected: instances {inst_1_idx} and {inst_2_idx} "
                f"likely stem from the same instance in frame {ghost_list}")

    # Step 2: Due process for the remaining suspects, three criteria must be met simultaneously to ensure fair judgment
    detected_instance_count = get_instance_count_per_frame(pred_data_array)
    mean_radius = np.mean(np.sqrt(canon_pose[0]**2 + canon_pose[1]**2))

    for frame_idx in range(total_frames):
        if detected_instance_count[frame_idx] < 2:
            continue

        # Check for small instances
        small_instances = []
        inst_radius = np.full(instance_count, np.nan)
        _, local_coords = duh.calculate_pose_centroids(pred_data_array, frame_idx)
        for inst_idx in instances:
            inst_radius[inst_idx] = np.nanmean(
                np.sqrt(local_coords[inst_idx, 0::2]**2 + local_coords[inst_idx, 1::2]**2))

        if inst_radius[inst_idx] < mean_radius * 0.9:
            small_instances.append(inst_idx)

        if not small_instances:
            continue

        other_instances = [inst_idx for inst_idx in instances \
            if inst_idx not in small_instances and np.any(~np.isnan(pred_data_array[frame_idx, inst_idx, :]))]
        if not other_instances:
            if debug_print:
                duh.log_print(f"Frame {frame_idx}: Case dismissed due to insufficient witnesses ----------------\n")
            continue

        # Check for flickering
        for small_idx in small_instances:
            prev_frame_present, next_frame_present = True, True
            if frame_idx > 0:
                prev_frame_present = not np.all(np.isnan(pred_data_array[frame_idx - 1, small_idx, :]))
            if frame_idx < total_frames - 1:
                next_frame_present = not np.all(np.isnan(pred_data_array[frame_idx + 1, small_idx, :]))
            
            flickering = not(prev_frame_present and next_frame_present)
            if not flickering: # Skip the envoloped check to avoid unnecessary calculation, not flickering == not guilty already
                continue
            
            # Check for envoloped
            current_frame_data = pred_data_array[frame_idx]
            for inst_idx in other_instances:
                inst_kp_x = current_frame_data[inst_idx, 0::3]
                inst_kp_y = current_frame_data[inst_idx, 1::3]

                min_x, min_y, max_x, max_y = duh.calculate_bbox(inst_kp_x, inst_kp_y)

                small_kp_x = current_frame_data[small_idx, 0::3]
                small_kp_y = current_frame_data[small_idx, 1::3]
                valid_mask = ~(np.isnan(small_kp_x) | np.isnan(small_kp_y))

                enveloped_x_mask = np.logical_and(small_kp_x > min_x, small_kp_x < max_x)
                enveloped_y_mask = np.logical_and(small_kp_y > min_y, small_kp_y < max_y)
                enveloped_mask = np.logical_and(enveloped_x_mask, enveloped_y_mask)
                enveloped_mask = np.logical_and(enveloped_mask, valid_mask)  # Only count valid keypoints

                enveloped_kp_num = np.sum(enveloped_mask)
                total_valid_kp = np.sum(valid_mask)

                if total_valid_kp > 0 and (enveloped_kp_num / total_valid_kp) > ghost_threshold_bp:
                    if debug_print:
                        duh.log_print(f"----------------- Ghost Prediction Court: Frame {frame_idx} ----------------\n"
                            f"About to NaNified inst {small_idx}. Verdict GUILTY due to following transgressions:\n"
                            f"1. TOO SMALL: Radius {inst_radius[small_idx]} vs orthodox radius {mean_radius}\n"
                            f"2. FLICKERING: Present at prior frame: {prev_frame_present}; Present at later frame: {next_frame_present}\n"
                            f"3. ENVOLOPED: Total valid keypoints: {total_valid_kp}; Enveloped ones: {enveloped_kp_num}."
                            )

                    pred_data_array[frame_idx, small_idx, :] = np.nan
                    break
    
    return pred_data_array

def applying_last_order(last_order:List[int], corrected_pred_data:np.ndarray, frame_idx:int, changes_applied:int, debug_print:bool):
    if last_order:
        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, last_order, :]
        changes_applied += 1
        if debug_print:
            duh.log_print(f"[TMOD] SWAP, Applying the last order: {last_order}")
    return changes_applied

def track_correction(pred_data_array: np.ndarray, idt_traj_array: Optional[np.ndarray], progress,
    debug_status: bool = False, max_dist: float = 10.0) -> Tuple[np.ndarray, int]:
    """
    Correct instance identities in DLC predictions using idTracker trajectories,
    with fallback to temporal coherence from prior DLC frames when idTracker fails.

    Parameters
    ----------
    pred_data_array : np.ndarray, DLC predictions of shape (T, N, 3*keypoints)
    idt_traj_array : Optional, idTracker trajectories of shape (T, N, 2)
    progress : QProgressBar for GUI progress updates
    debug_status : bool, Whether to log detailed debug info
    max_dist : float, Max per-mouse displacement to skip Hungarian (in pixels)
    lookback_limit : int, Max frames to look back for valid prior (default 5)
        
    Returns
    -------
    corrected_pred_data : np.ndarray, Identity-corrected DLC predictions
    changes_applied : int, Number of frames where identity swap was applied
    """
    total_frames, instance_count, _ = pred_data_array.shape
    pred_positions, _ = duh.calculate_pose_centroids(pred_data_array)

    if idt_traj_array is not None:
        idt_mode = True
        assert pred_data_array.shape[1] == idt_traj_array.shape[1], "Instance count must match between prediction and idTracker"
        remapped_idt = remap_idt_array(pred_positions, idt_traj_array)
    else:
        idt_mode = False

    corrected_pred_data = pred_data_array.copy()

    last_order = None
    changes_applied = 0
    debug_print = debug_status
    if debug_print:
        duh.log_print("----------  Starting IDT Autocorrection  ----------")

    for frame_idx in range(total_frames):
        progress.setValue(frame_idx)
        if progress.wasCanceled():
            return pred_data_array, 0

        pred_centroids, _ = duh.calculate_pose_centroids(corrected_pred_data, frame_idx)
        valid_pred_mask = np.all(~np.isnan(pred_centroids), axis=1)

        if debug_print:
            duh.log_print(f"---------- frame: {frame_idx} ---------- ")
            for i in range(instance_count):
                if valid_pred_mask[i]:
                    duh.log_print(f"x,y in pred: inst {i}: ({pred_centroids[i,0]:.1f}, {pred_centroids[i,1]:.1f})")
 
        # Case 0: No DLC prediction on current frame
        if np.sum(valid_pred_mask) == 0:
            if debug_print:
                duh.log_print("SKIP, No valid prediction.")
            changes_applied = applying_last_order(last_order, corrected_pred_data, frame_idx, changes_applied, debug_print)
            continue

        # Case 1: idTrackerai detection valid and active
        if idt_mode:
            valid_idt_mask = np.all(~np.isnan(remapped_idt[frame_idx]), axis=1)
            idt_centroids = remapped_idt[frame_idx].copy()

        if idt_mode and np.sum(valid_pred_mask) == np.sum(valid_idt_mask):
            if debug_print:
                for i in range(instance_count):
                    if valid_idt_mask[i]:
                        print(f"x,y in idt: inst {i}: ({idt_centroids[i,0]:.1f}, {idt_centroids[i,1]:.1f})")

        # Case 2: idTracker invalid — use prior DLC as reference
        else: # # Build last_known_centroids from prior frames
            idt_centroids = np.full((instance_count,2),np.nan)
            valid_idt_mask = np.zeros(instance_count, dtype=bool)

            for inst_idx in range(instance_count):
                cand_idx = frame_idx
                while cand_idx > 0:
                    cand_idx -= 1
                    if np.any(~np.isnan(corrected_pred_data[cand_idx, inst_idx, :])):
                        cand_centroids, _ = duh.calculate_pose_centroids(corrected_pred_data, cand_idx)
                        idt_centroids[inst_idx, :] = cand_centroids[inst_idx, :]
                        valid_idt_mask[inst_idx] = True
                        break

            if debug_print:
                duh.log_print(f"[TMOD] Found valid_prior={np.sum(valid_idt_mask)}")

        valid_pred_centroids = pred_centroids[valid_pred_mask]
        valid_idt_centroids = idt_centroids[valid_idt_mask]
        mode_text = "" if idt_mode else "[TMOD]"

        if len(valid_idt_centroids) == 0:
            if debug_print:
                duh.log_print(f"{mode_text} No valid reference.")
            changes_applied = applying_last_order(last_order, corrected_pred_data, frame_idx, changes_applied, debug_print)
            continue

        new_order = hungarian_matching(
            valid_pred_centroids, valid_idt_centroids, valid_pred_mask, valid_idt_mask, max_dist, debug_print)

        if new_order is None:
            if debug_print:
                duh.log_print(f"{mode_text} Failed to build new order with Hungarian.")
            changes_applied = applying_last_order(last_order, corrected_pred_data, frame_idx, changes_applied, debug_print)
            continue
        elif new_order == list(range(instance_count)):
            if debug_print:
                last_order = None # Reset last order
                duh.log_print(f"{mode_text} NO SWAP, already the best solution in Hungarian.")
            continue

        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, new_order, :]
        last_order = new_order
        changes_applied += 1

        if debug_print:
            duh.log_print(f"{mode_text} SWAP, new_order: {new_order}.")

    return corrected_pred_data, changes_applied

def remap_idt_array(pred_positions:np.ndarray, idt_traj_array:np.ndarray) -> np.ndarray:
    total_frames, instance_count, _ = pred_positions.shape

    remapped_idt = idt_traj_array.copy()
    valid_pred = np.all(~np.isnan(pred_positions), axis=2)
    valid_idt = np.all(~np.isnan(idt_traj_array), axis=2) 

    valid_frame_idx = None
    for frame_idx in range(total_frames):
        n_pred = np.sum(valid_pred[frame_idx])
        n_idt  = np.sum(valid_idt[frame_idx])
        if n_pred == n_idt and n_pred == instance_count: # Only use frames with all instances present
            valid_frame_idx = frame_idx
            break

    if valid_frame_idx is None:
        raise ValueError("No frame with valid data in both arrays.")

    p_pred = pred_positions[valid_frame_idx]
    p_idt = idt_traj_array[valid_frame_idx]

    cost_matrix = np.linalg.norm(p_pred[:, np.newaxis, :] - p_idt[np.newaxis, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping_details = []
    for pred_inst, idt_inst in zip(row_ind, col_ind):
        remapped_idt[:, pred_inst, :] = idt_traj_array[:, idt_inst, :]
        mapping_details.append((pred_inst, idt_inst))
    mapping_details.sort(key=lambda x: x[1])

    print(f"Successfully remapped idtracker trajectory using frame {valid_frame_idx}.")
    print(f"Matched using Hungarian algorithm (total cost: {cost_matrix[row_ind, col_ind].sum():.3f}):")
    for idt_inst, pred_inst in mapping_details:
        print(f"  idTracker instance {idt_inst:2d} → Prediction instance {pred_inst:2d}")

    return remapped_idt

def hungarian_matching(valid_pred_centroids:np.ndarray, valid_idt_centroids:np.ndarray,
        valid_pred_mask:np.ndarray, valid_idt_mask:np.ndarray, max_dist:float=10.0, debug_print:bool=False) -> Optional[np.ndarray]:
    """
    Perform identity correction using Hungarian algorithm.

    Maps current DLC detections to reference identities (idTracker or prior frame)
    by solving optimal assignment based on centroid distances.

    Args:
        valid_pred_centroids: (K, 2) — centroids of CURRENT detections
        valid_idt_centroids:  (M, 2) — centroids of REFERENCE (prior/idt) instances
        valid_pred_mask: (N,) bool — which of N total instances are valid in pred
        valid_idt_mask:  (N,) bool — which of N total instances are valid in ref
        max_dist: Max distance to allow match (pixels)

    Returns:
        new_order: List[int] of length N, where:
            new_order[target_identity] = source_instance_index_in_current_frame
            i.e., "Identity j comes from current instance new_order[j]"
    """
    instance_count = valid_pred_mask.shape[0]
    K, M = len(valid_pred_centroids), len(valid_idt_centroids)

    # Case: no valid data
    if K == 0 or M == 0:
        if debug_print:
            duh.log_print(f"[HUN] No valid data for Hungarian matching (K={K}, M={M}). Returning default order.")
        return list(range(instance_count))

    # Reconstruct global indices
    pred_indices = np.where(valid_pred_mask)[0]  # global IDs of valid preds
    idt_indices  = np.where(valid_idt_mask)[0]   # global IDs of valid ref instances

    # Single valid pair — skip Hungarian
    if K == 1 and M == 1:
        dist = np.linalg.norm(valid_pred_centroids[0] - valid_idt_centroids[0])
        if debug_print:
            duh.log_print(f"[HUN] Single pair matching. Distance: {dist:.2f}, Max_dist: {max_dist}")
        if dist < max_dist:
            new_order = list(range(instance_count))
            new_order[idt_indices[0]] = pred_indices[0]
            if debug_print:
                duh.log_print(f"[HUN] Single pair matched. New order: {new_order}")
            return new_order
        else:
            if debug_print:
                duh.log_print(f"[HUN] Single pair not matched (distance too high). Returning default order.")
            return list(range(instance_count))  # no swap

    # All pairs on board, validate before Hungarian
    if K == instance_count and M == instance_count:
        full_set = True
        distances = np.linalg.norm(valid_pred_centroids - valid_idt_centroids, axis=1)
        if debug_print:
            duh.log_print(f"[HUN] All instances present and masks match. Distances: {distances}, Max_dist: {max_dist}")
        if np.all(distances < max_dist):
            if debug_print:
                duh.log_print(f"[HUNG] All identities stable. Returning default order.")
            return list(range(instance_count))  # identities stable
    else:
        full_set = False
        
    # Build cost matrix
    cost_matrix = np.linalg.norm(valid_pred_centroids[:, np.newaxis] - valid_idt_centroids[np.newaxis, :], axis=2)

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        if debug_print:
            duh.log_print(f"[HUN] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}")
    except Exception as e:
        if debug_print:
            duh.log_print(f"[HUN] Hungarian failed: {e}. Returning None.")
        return None  # Hungarian failed
    else:
        if full_set: # Only do the comparison with full set
            current_order = list(range(instance_count))
            if not compare_assignment_costs(cost_matrix, current_order, row_ind, col_ind, improvement_threshold=0.1):
                if debug_print:
                    duh.log_print(f"[HUN] Hungarian failed to improve the assognment costs.")
                return list(range(instance_count))  # already stable

    # Build new_order
    all_inst = range(instance_count)

    processed = {}
    for r, c in zip(row_ind, col_ind):
        target_identity = idt_indices[c]
        source_instance = pred_indices[r]
        processed[target_identity] = source_instance
    if debug_print:
        duh.log_print(f"[HUN] Processed matches: {processed}")

    unprocessed = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
    unassigned = [inst_idx for inst_idx in all_inst if inst_idx not in processed.values()]
    if debug_print:
        duh.log_print(f"[HUN] Unprocessed identities: {unprocessed}, Unassigned instances: {unassigned}")

    for target_identity in unprocessed:  # First loop, find remaining pair without idx change
        if target_identity in unassigned:
            source_instance = target_identity
            processed[target_identity] = source_instance
            unassigned.remove(source_instance)
    if debug_print:
        duh.log_print(f"[HUN] Processed after first loop (self-assignment): {processed}")
    
    unprocessed[:] = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
    if debug_print:
        duh.log_print(f"[HUN] Unprocessed identities after first loop: {unprocessed}")

    for target_identity in unprocessed:  # Second loop, arbitarily reassign
        source_instance = unassigned[-1]
        processed[target_identity] = source_instance
        unassigned.remove(source_instance)
    if debug_print:
        duh.log_print(f"[HUN] Processed after second loop (arbitrary assignment): {processed}")
        
    sorted_processed = {k: processed[k] for k in sorted(processed)}
    new_order = list(sorted_processed.values())
    if debug_print:
        duh.log_print(f"[HUN] Final new_order: {new_order}")

    return new_order

def compare_assignment_costs(cost_matrix: np.ndarray, current_order: list, 
        new_row_ind: np.ndarray, new_col_ind: np.ndarray, improvement_threshold: float = 0.1) -> bool:
    """
    Decide whether to apply the new Hungarian assignment by comparing total costs.

    Args:
        cost_matrix: (K, M) matrix of distances between current detections and prior positions
        current_order: list of length N, current identity mapping (e.g., [0,1] or [1,0])
        new_row_ind: Hungarian result - assigned detection indices
        new_col_ind: Hungarian result - assigned prior (identity) indices
        improvement_threshold: float, minimum relative improvement to accept swap
                             e.g., 0.1 = 10% better cost required

    Returns:
        bool: True if new assignment is significantly better
    """
    K, M = cost_matrix.shape
    N = len(current_order)

    # Build current assignment cost
    current_cost = 0.0
    count = 0
    for j in range(N):
        i = current_order[j] 
        if i < K and j < M and not np.isnan(cost_matrix[i, j]):
            current_cost += cost_matrix[i, j]
            count += 1

    current_cost = current_cost / count if count > 0 else 1e6

    # Build new assignment cost
    new_cost = cost_matrix[new_row_ind, new_col_ind].sum()
    new_count = len(new_row_ind)
    new_cost = new_cost / new_count if new_count > 0 else 1e6

    # Only apply if new assignment is significantly better
    if new_cost < current_cost * (1 - improvement_threshold):
        return True
    else:
        return False

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