import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional

from utils.helper import log_print

def hungarian_matching(
        valid_pred_centroids:np.ndarray,
        valid_idt_centroids:np.ndarray,
        valid_pred_mask:np.ndarray,
        valid_idt_mask:np.ndarray,
        max_dist:float=10.0,
        debug_print:bool=False
        ) -> Optional[np.ndarray]:
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
            log_print(f"[HUN] No valid data for Hungarian matching (K={K}, M={M}). Returning default order.")
        return list(range(instance_count))

    # Reconstruct global indices
    pred_indices = np.where(valid_pred_mask)[0]  # global IDs of valid preds
    idt_indices  = np.where(valid_idt_mask)[0]   # global IDs of valid ref instances

    # Single valid pair — skip Hungarian
    if K == 1 and M == 1:
        dist = np.linalg.norm(valid_pred_centroids[0] - valid_idt_centroids[0])
        if debug_print:
            log_print(f"[HUN] Single pair matching. Distance: {dist:.2f}, Max_dist: {max_dist}")
        if dist < max_dist:
            new_order = list(range(instance_count))
            new_order[idt_indices[0]] = pred_indices[0]
            if debug_print:
                log_print(f"[HUN] Single pair matched. New order: {new_order}")
            return new_order
        else:
            if debug_print:
                log_print(f"[HUN] Single pair not matched (distance too high). Returning default order.")
            return list(range(instance_count))  # no swap

    # All pairs on board, validate before Hungarian
    if K == instance_count and M == instance_count:
        full_set = True
        distances = np.linalg.norm(valid_pred_centroids - valid_idt_centroids, axis=1)
        if debug_print:
            log_print(f"[HUN] All instances present and masks match. Distances: {distances}, Max_dist: {max_dist}")
        if np.all(distances < max_dist):
            if debug_print:
                log_print(f"[HUNG] All identities stable. Returning default order.")
            return list(range(instance_count))  # identities stable
    else:
        full_set = False
        
    # Build cost matrix
    cost_matrix = np.linalg.norm(valid_pred_centroids[:, np.newaxis] - valid_idt_centroids[np.newaxis, :], axis=2)

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        if debug_print:
            log_print(f"[HUN] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}")
    except Exception as e:
        if debug_print:
            log_print(f"[HUN] Hungarian failed: {e}. Returning None.")
        return None  # Hungarian failed
    else:
        if full_set: # Only do the comparison with full set
            current_order = list(range(instance_count))
            if not _compare_assignment_costs(cost_matrix, current_order, row_ind, col_ind, improvement_threshold=0.1):
                if debug_print:
                    log_print(f"[HUN] Hungarian failed to improve the assognment costs.")
                return list(range(instance_count))  # already stable

    # Build new_order
    all_inst = range(instance_count)

    processed = {}
    for r, c in zip(row_ind, col_ind):
        target_identity = idt_indices[c]
        source_instance = pred_indices[r]
        processed[target_identity] = source_instance
    if debug_print:
        log_print(f"[HUN] Processed matches: {processed}")

    unprocessed = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
    unassigned = [inst_idx for inst_idx in all_inst if inst_idx not in processed.values()]
    if debug_print:
        log_print(f"[HUN] Unprocessed identities: {unprocessed}, Unassigned instances: {unassigned}")

    for target_identity in unprocessed:  # First loop, find remaining pair without idx change
        if target_identity in unassigned:
            source_instance = target_identity
            processed[target_identity] = source_instance
            unassigned.remove(source_instance)
    if debug_print:
        log_print(f"[HUN] Processed after first loop (self-assignment): {processed}")
    
    unprocessed[:] = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
    if debug_print:
        log_print(f"[HUN] Unprocessed identities after first loop: {unprocessed}")

    for target_identity in unprocessed:  # Second loop, arbitarily reassign
        source_instance = unassigned[-1]
        processed[target_identity] = source_instance
        unassigned.remove(source_instance)
    if debug_print:
        log_print(f"[HUN] Processed after second loop (arbitrary assignment): {processed}")
        
    sorted_processed = {k: processed[k] for k in sorted(processed)}
    new_order = list(sorted_processed.values())
    if debug_print:
        log_print(f"[HUN] Final new_order: {new_order}")

    return new_order

def _compare_assignment_costs(
        cost_matrix:np.ndarray,
        current_order:list, 
        new_row_ind:np.ndarray,
        new_col_ind:np.ndarray,
        improvement_threshold:float = 0.1
        ) -> bool:
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