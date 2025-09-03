import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Tuple

from .hungarian import hungarian_matching
from utils.pose import calculate_pose_centroids
from utils.helper import log_print

def track_correction(pred_data_array: np.ndarray,
        idt_traj_array: Optional[np.ndarray],
        progress,
        debug_status: bool = False,
        max_dist: float = 10.0
        ) -> Tuple[np.ndarray, int]:
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
    pred_positions, _ = calculate_pose_centroids(pred_data_array)

    if idt_traj_array is not None:
        idt_mode = True
        assert pred_data_array.shape[1] == idt_traj_array.shape[1], "Instance count must match between prediction and idTracker"
        remapped_idt = _remap_idt_array(pred_positions, idt_traj_array)
    else:
        idt_mode = False

    corrected_pred_data = pred_data_array.copy()

    last_order = None
    changes_applied = 0
    debug_print = debug_status
    if debug_print:
        log_print("----------  Starting IDT Autocorrection  ----------")

    for frame_idx in range(total_frames):
        progress.setValue(frame_idx)
        if progress.wasCanceled():
            return pred_data_array, 0

        pred_centroids, _ = calculate_pose_centroids(corrected_pred_data, frame_idx)
        valid_pred_mask = np.all(~np.isnan(pred_centroids), axis=1)

        if debug_print:
            log_print(f"---------- frame: {frame_idx} ---------- ")
            for i in range(instance_count):
                if valid_pred_mask[i]:
                    log_print(f"x,y in pred: inst {i}: ({pred_centroids[i,0]:.1f}, {pred_centroids[i,1]:.1f})")
 
        # Case 0: No DLC prediction on current frame
        if np.sum(valid_pred_mask) == 0:
            if debug_print:
                log_print("SKIP, No valid prediction.")
            changes_applied = _applying_last_order(last_order, corrected_pred_data, frame_idx, changes_applied, debug_print)
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
                        cand_centroids, _ = calculate_pose_centroids(corrected_pred_data, cand_idx)
                        idt_centroids[inst_idx, :] = cand_centroids[inst_idx, :]
                        valid_idt_mask[inst_idx] = True
                        break

            if debug_print:
                log_print(f"[TMOD] Found valid_prior={np.sum(valid_idt_mask)}")

        valid_pred_centroids = pred_centroids[valid_pred_mask]
        valid_idt_centroids = idt_centroids[valid_idt_mask]
        mode_text = "" if idt_mode else "[TMOD]"

        if len(valid_idt_centroids) == 0:
            if debug_print:
                log_print(f"{mode_text} No valid reference.")
            changes_applied = _applying_last_order(last_order, corrected_pred_data, frame_idx, changes_applied, debug_print)
            continue

        new_order = hungarian_matching(
            valid_pred_centroids, valid_idt_centroids, valid_pred_mask, valid_idt_mask, max_dist, debug_print)

        if new_order is None:
            if debug_print:
                log_print(f"{mode_text} Failed to build new order with Hungarian.")
            changes_applied = _applying_last_order(last_order, corrected_pred_data, frame_idx, changes_applied, debug_print)
            continue
        elif new_order == list(range(instance_count)):
            if debug_print:
                last_order = None # Reset last order
                log_print(f"{mode_text} NO SWAP, already the best solution in Hungarian.")
            continue

        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, new_order, :]
        last_order = new_order
        changes_applied += 1

        if debug_print:
            log_print(f"{mode_text} SWAP, new_order: {new_order}.")

    return corrected_pred_data, changes_applied

def _remap_idt_array(pred_positions:np.ndarray, idt_traj_array:np.ndarray) -> np.ndarray:
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
    
def _applying_last_order(
        last_order:List[int],
        corrected_pred_data:np.ndarray,
        frame_idx:int,
        changes_applied:int,
        debug_print:bool
        ):
    
    if last_order:
        corrected_pred_data[frame_idx, :, :] = corrected_pred_data[frame_idx, last_order, :]
        changes_applied += 1
        if debug_print:
            log_print(f"[TMOD] SWAP, Applying the last order: {last_order}")
    return changes_applied
