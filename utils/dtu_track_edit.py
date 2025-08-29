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
        instance_threshold:int, use_or:Optional[bool]=False, return_frame_list:Optional[bool]=False):
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
    aligned_local, rotation_angles = duh.align_poses_by_vector(local_coords, angle_map_data=angle_map_data)
    avg_angle = duh.circular_mean(rotation_angles)
    if set_centroid is not None : # If a specific centroid is provided, use it for alignment
        centroids = set_centroid
    if set_rotation is not None and ~np.isnan(set_rotation): # If a specific angle is provided, use it for alignment
        avg_angle = set_rotation
    average_pose = duh.track_rotation_worker(avg_angle, centroids, aligned_local, conf_scores)

    return average_pose

###################################################################################################################################################

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

    corrected_pred_data = ghost_prediction_buster(corrected_pred_data, instance_count)

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
            continue

        # Case 1: idTrackerai detection valid and active
        if idt_mode:
            valid_idt_mask = np.all(~np.isnan(remapped_idt[frame_idx]), axis=1)
            idt_centroids = remapped_idt[frame_idx].copy()

            if debug_print:
                for i in range(instance_count):
                    if valid_idt_mask[i]:
                        duh.log_print(f"x,y in idt: inst {i}: ({idt_centroids[i,0]:.1f}, {idt_centroids[i,1]:.1f})")

        # Case 2: idTracker invalid — use prior DLC as reference
        else:
            # # Build last_known_centroids from prior frames
            idt_centroids = np.full((instance_count,2),np.nan)
            valid_idt_mask = np.zeros(instance_count, dtype=bool)

            cand_idx = frame_idx
            for inst_idx in range(instance_count):
                while cand_idx > 0:
                    cand_idx -= 1
                    cand_centroids = duh.calculate_pose_centroids(corrected_pred_data, cand_idx)
                    if np.any(~np.isnan(cand_centroids[inst_idx])):
                        idt_centroids = cand_centroids[inst_idx]
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

        new_order = hungarian_matching(valid_pred_centroids, valid_idt_centroids, valid_pred_mask, valid_idt_mask, max_dist)

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
        valid_pred_mask:np.ndarray, valid_idt_mask:np.ndarray, max_dist:float=10.0) -> Optional[np.ndarray]:
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
        return list(range(instance_count))

    # Reconstruct global indices
    pred_indices = np.where(valid_pred_mask)[0]  # global IDs of valid preds
    idt_indices  = np.where(valid_idt_mask)[0]   # global IDs of valid ref instances

    # Case: single valid pair — skip Hungarian
    if K == 1 and M == 1:
        dist = np.linalg.norm(valid_pred_centroids[0] - valid_idt_centroids[0])
        if dist < max_dist:
            # Identity preserved: ref identity idt_indices[0] ← pred instance pred_indices[0]
            new_order = list(range(instance_count))
            new_order[idt_indices[0]] = pred_indices[0]
            return new_order
        else:
            return list(range(instance_count))  # no swap

    # Build cost matrix
    cost_matrix = np.linalg.norm(valid_pred_centroids[:, np.newaxis] - valid_idt_centroids[np.newaxis, :], axis=2)
    
    # Apply max_dist threshold
    cost_matrix = np.where(cost_matrix <= max_dist, cost_matrix, 1e6)

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        return None  # Hungarian failed

    # Filter matches by max_dist
    valid_match = cost_matrix[row_ind, col_ind] < 1e6
    if not np.any(valid_match):
        return list(range(instance_count))  # no valid match

    row_ind = row_ind[valid_match]
    col_ind = col_ind[valid_match]

    # Build new_order
    new_order = list(range(instance_count))

    for r, c in zip(row_ind, col_ind):
        target_identity = idt_indices[c]
        source_instance = pred_indices[r]
        new_order[target_identity] = source_instance

    return new_order

def ghost_prediction_buster(pred_data_array:np.ndarray, instance_count:int) -> np.ndarray:
    """Filter out ghost predictions: i.e. identical predictions in the exact same coordinates"""
    instances = list(range(instance_count))
    for inst_1_idx, inst_2_idx in combinations(instances, 2):
        pose_diff_all_frames = pred_data_array[:, inst_1_idx, :] - pred_data_array[:, inst_2_idx, :]
        ghost_mask = np.all(pose_diff_all_frames <= 1, axis=1)
        pred_data_array[ghost_mask, inst_2_idx, :] = np.nan
        if np.any(ghost_mask):
            ghost_frame_indices = np.where(ghost_mask)[0]
            print(f"Ghost prediction detected: instances {inst_1_idx} and {inst_2_idx} identical in frames {ghost_frame_indices}")
    return pred_data_array

def applying_last_order(last_order:List[int], corrected_pred_data:np.ndarray, frame_idx:int, changes_applied:int, debug_print:bool):
    if last_order:
        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, last_order, :]
        changes_applied += 1
        if debug_print:
            duh.log_print(f"[TMOD] SWAP, Applying the last order: {last_order}")
    return changes_applied

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