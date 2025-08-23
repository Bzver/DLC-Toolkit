import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

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
    aligned_local, rotation_angles = duh.align_poses_by_vector(local_coords, angle_map_data=angle_map_data)
    avg_angle = duh.circular_mean(rotation_angles)
    if set_centroid is not None : # If a specific centroid is provided, use it for alignment
        centroids = set_centroid
    if set_rotation is not None and ~np.isnan(set_rotation): # If a specific angle is provided, use it for alignment
        avg_angle = set_rotation
    average_pose = duh.track_rotation_worker(avg_angle, centroids, aligned_local, conf_scores)

    return average_pose

###################################################################################################################################################

def idt_track_correction(pred_data_array: np.ndarray, idt_traj_array: np.ndarray, progress,
    debug_status: bool = False, max_dist: float = 20.0, lookback_limit: int = 5) -> Tuple[np.ndarray, int]:
    """
    Correct instance identities in DLC predictions using idTracker trajectories,
    with fallback to temporal coherence from prior DLC frames when idTracker fails.

    Parameters
    ----------
    pred_data_array : np.ndarray
        DLC predictions of shape (T, N, 3*keypoints)
    idt_traj_array : np.ndarray
        idTracker trajectories of shape (T, N, 2)
    progress : QProgressBar for GUI progress updates
    debug_status : bool
        Whether to log detailed debug info
    max_dist : float
        Max per-mouse displacement to skip Hungarian (in pixels)
    lookback_limit : int
        Max frames to look back for valid prior (default 5)

    Returns
    -------
    corrected_pred_data : np.ndarray
        Identity-corrected DLC predictions
    changes_applied : int
        Number of frames where identity swap was applied
    """
    assert pred_data_array.shape[1] == idt_traj_array.shape[1], "Instance count must match between prediction and idTracker"

    total_frames, instance_count, _ = pred_data_array.shape
    pred_positions, _ = duh.calculate_pose_centroids(pred_data_array)

    remapped_idt = remap_idt_array(pred_positions, idt_traj_array)
    corrected_pred_data = pred_data_array.copy()

    last_order = list(range(instance_count))
    changes_applied = 0
    debug_print = debug_status

    for frame_idx in range(total_frames):
        progress.setValue(frame_idx)
        if progress.wasCanceled():
            return pred_data_array, 0

        pred_position_curr, _ = duh.calculate_pose_centroids(corrected_pred_data, frame_idx)

        if debug_print:
            duh.log_print(f"---------- frame: {frame_idx} ---------- ")
            duh.log_print(
                f"x,y in pred: inst 0: ({pred_position_curr[0,0]}, {pred_position_curr[0,1]})"
                            f" | inst 1: ({pred_position_curr[1,0]}, {pred_position_curr[1,1]})"
                )
            duh.log_print(
                f"x,y in idt: inst 0: ({remapped_idt[frame_idx,0,0]}, {remapped_idt[frame_idx,0,1]})"
                            f" | inst 1: ({remapped_idt[frame_idx,1,0]}, {remapped_idt[frame_idx,1,1]})"
                )

        valid_pred_curr = np.all(~np.isnan(pred_position_curr), axis=1)
        valid_idt_curr = np.all(~np.isnan(remapped_idt[frame_idx]), axis=1) 

        n_pred = np.sum(valid_pred_curr)
        n_idt = np.sum(valid_idt_curr)

        # Case 1: No DLC prediction on current frame
        if n_pred == 0:
            if debug_print:
                duh.log_print(f"SKIP, No valid prediction to correct in frame {frame_idx}.")
            continue

        # Case 2: idTrackerai detection valid and counts matched
        elif n_pred == n_idt:
            if n_pred == 1: # Single instance, swap if needed
                valid_pred_inst = np.where(valid_pred_curr)[0][0]
                valid_idt_inst = np.where(valid_idt_curr)[0][0]
                if valid_pred_inst == valid_idt_inst:
                    last_order = list(range(instance_count)) # Reset last order
                    if debug_print:
                        duh.log_print(f"NO SWAP, valid_pred_inst: {valid_pred_inst}, valid_idt_inst:{valid_idt_inst}")
                    continue

                new_order = list(range(instance_count))
                new_order[valid_pred_inst] = valid_idt_inst
                new_order[valid_idt_inst] = valid_pred_inst
                last_order = new_order

                corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, new_order, :]

                changes_applied += 1
                if debug_print:
                    duh.log_print(f"SWAP, inst {valid_pred_inst} <-> inst {valid_idt_inst}.")
                continue

            else: # Multiple instance pair in DLC and IDT prediction, prepare for Hungarian
                valid_positions_pred = pred_position_curr[valid_pred_curr]
                valid_positions_idt  = remapped_idt[frame_idx][valid_idt_curr]

        # Case 3: idTracker invalid — use prior DLC as reference
        else:
            if debug_print:
                duh.log_print(f"TEMPORAL MODE: n_pred={n_pred}, n_idt={n_idt}")

            lookback_limit = min(lookback_limit, frame_idx)  # don't go before 0
            ref_frame_idx = None
            pred_position_ref = None

            for offset in range(1, lookback_limit + 1):
                cand_idx = frame_idx - offset
                pos_cand, _ = duh.calculate_pose_centroids(corrected_pred_data, cand_idx)
                valid_cand = np.all(~np.isnan(pos_cand), axis=1)
                n_prior = np.sum(valid_cand)

                if n_prior == instance_count:
                    ref_frame_idx = cand_idx
                    pred_position_ref = pos_cand
                    if debug_print:
                        duh.log_print(f"[TMOD] Found valid reference frame at t={cand_idx} (offset {offset})")
                    break

            if pred_position_ref is None:
                corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, last_order, :]
                changes_applied += 1
                if debug_print:
                    duh.log_print("[TMOD] NO TMOD, No valid full frame found in last 5 frames."
                                f"[TMOD] SWAP, Applying the last order: {last_order}")
                continue

            if n_pred == 1: # Only one mouse detected — assign it to the closest prior mouse
                curr_pos = pred_position_curr[valid_pred_curr][0]  # (x, y)
                ref_positions = pred_position_ref  # (I, 2) — both prior positions
                distances = np.linalg.norm(ref_positions - curr_pos, axis=1)  # (I,)

                closest_ref_idx = np.argmin(distances)
                current_valid_inst = np.where(valid_pred_curr)[0][0]

                # Build new_order: assign closest prior identity to current valid instance
                new_order = list(range(instance_count))
                new_order[closest_ref_idx] = current_valid_inst
                new_order[current_valid_inst] = closest_ref_idx
                last_order = new_order

                corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, new_order, :]
                changes_applied += 1
                if debug_print:
                    duh.log_print(f"[TMOD] n_pred=1: assigned closest prior identity {closest_ref_idx} to instance {current_valid_inst}")
                continue

            if n_pred != instance_count:
                corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, last_order, :]
                changes_applied += 1
                if debug_print:
                    duh.log_print("[TMOD] NO TMOD, insufficient current predictions from Hungarian."
                                f"[TMOD] SWAP, Applying the last order: {last_order}")
                continue

            if debug_print:
                duh.log_print(
                    f"[TMOD] Using prior DLC frame {ref_frame_idx} as reference for Hungarian matching.\n"
                    f"x,y in pred: inst 0: ({pred_position_curr[0,0]}, {pred_position_curr[0,1]})"
                            f" | inst 1: ({pred_position_curr[1,0]}, {pred_position_curr[1,1]})\n"
                    f"x,y in prev: inst 0: ({pred_position_ref[0,0]}, {pred_position_ref[0,1]})"
                                f" | inst 1: ({pred_position_ref[1,0]}, {pred_position_ref[1,1]})"
                    )

            valid_positions_pred = pred_position_curr[valid_pred_curr]
            valid_positions_idt = pred_position_ref[valid_cand]  # all valid
            valid_idt_curr = valid_cand  # needed for indexing later

        new_order = hungarian_matching(
            valid_positions_pred, valid_positions_idt, valid_pred_curr, valid_idt_curr, max_dist)

        if new_order is None:
            corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, last_order, :]
            changes_applied += 1
            if debug_print:
                duh.log_print("[TMOD] NO TMOD, failed to build new order with Hungarian."
                            f"[TMOD] SWAP, Applying the last order: {last_order}")
            continue

        elif new_order == list(range(instance_count)):
            if debug_print:
                duh.log_print("[TMOD] Positions within threshold, no need for Hungarian.")
            continue

        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, new_order, :]
        last_order = new_order
        changes_applied += 1
        if debug_print:
            duh.log_print(f"[TMOD] SWAP, new_order: {new_order}.")

    return corrected_pred_data, changes_applied

def remap_idt_array(pred_positions:np.ndarray, idt_traj_array:np.ndarray) -> np.ndarray: # Version 2
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

def hungarian_matching(valid_positions_pred:np.ndarray, valid_positions_idt:np.ndarray, 
        valid_pred_curr:np.ndarray, valid_idt_curr:np.ndarray, max_dist:float=20.0) -> Optional[np.ndarray]:
    """
    Perform identity correction by solving the optimal assignment problem using the Hungarian algorithm.
    Matches detected instances from DeepLabCut (DLC) predictions to reference instances (from idTracker 
    or prior DLC frame) based on spatial proximity.

    This function handles partial detections (e.g., one mouse occluded) by only matching valid instances,
    then reconstructing the full identity permutation for the entire instance set.

    Args:
    valid_positions_pred / valid_positions_idt:  XY postitions of each valid instance.
        Shape: (valid_instance_count, 2),

    valid_pred_curr / valid_idt_curr: Boolean mask indicating which instances are valid
        Shape: (instance_count,) — e.g., (2,) → [True, False] if only instance 0 is valid.

    max_dist: float, optional
        Maximum average per-instance displacement (in pixels) to skip Hungarian and assume 
        identities are consistent. If all matched distances are < `max_dist`, no swap is applied.

    Returns:
    new_order : Optional[List[int]]
        A list of length `instance_count` representing the new identity mapping:
            new_order[target_identity] = source_instance_index_in_current_frame
    """
    distances = np.linalg.norm(valid_positions_pred - valid_positions_idt, axis=1)
    instance_count = valid_idt_curr.shape[0]

    if np.all(distances < max_dist):
        return list(range(instance_count))

    cost_matrix = np.linalg.norm(valid_positions_pred[:, np.newaxis, :] - valid_positions_idt[np.newaxis, :, :], axis=2)
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except:
        return None

    new_order = list(range(instance_count))
    pred_indices = np.where(valid_pred_curr)[0]
    idt_indices  = np.where(valid_idt_curr)[0]

    for i, j in zip(row_ind, col_ind):
        new_order[idt_indices[j]] = pred_indices[i]

    return new_order

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