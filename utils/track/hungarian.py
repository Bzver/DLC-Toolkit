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
        if full_set:
            current_order = list(range(instance_count))
            if not _compare_distance_improvement(valid_pred_centroids, valid_idt_centroids, current_order, row_ind, col_ind):
                if debug_print:
                    log_print(f"[HUN] Hungarian assignment does not improve geometric distances sufficiently.")
                return list(range(instance_count))  # keep current assignment

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

def _compare_distance_improvement(
    valid_pred_centroids: np.ndarray,
    valid_idt_centroids: np.ndarray,
    current_order: list,
    new_row_ind: np.ndarray,
    new_col_ind: np.ndarray,
    improvement_threshold: float = 0.1
) -> bool:
    """
    Compare assignments by sorting all assignment distances and comparing element-wise.

    Ignores which identity is assigned to which instance — only compares the overall
    distribution of match quality.

    Accepts new assignment if, on average, sorted distances are improved by threshold.
    """
    K, _ = valid_pred_centroids.shape
    M, _ = valid_idt_centroids.shape
    N = min(K, M)

    # Compute current assignment distances
    current_dists = []
    for j in range(len(current_order)):
        if j >= M:
            continue
        i_old = current_order[j]
        if i_old >= K:
            continue
        dist = np.linalg.norm(valid_pred_centroids[i_old] - valid_idt_centroids[j])
        current_dists.append(dist)

    # Compute new assignment distances
    new_dists = []
    for r, c in zip(new_row_ind, new_col_ind):
        if c >= M or r >= K:
            continue
        dist = np.linalg.norm(valid_pred_centroids[r] - valid_idt_centroids[c])
        new_dists.append(dist)

    # Sort both to get identity-agnostic score
    current_dists = np.sort(current_dists)[:N]
    new_dists = np.sort(new_dists)[:N]

    if len(current_dists) == 0 or len(new_dists) == 0:
        return False

    # Compute relative improvement per sorted position
    total_improvement = 0.0
    count = 0

    for i in range(len(current_dists)):
        old_d = current_dists[i]
        new_d = new_dists[i] if i < len(new_dists) else old_d

        if old_d < 1e-9:
            improvement = 1.0 if new_d < old_d else 0.0
        else:
            improvement = (old_d - new_d) / old_d

        total_improvement += improvement
        count += 1

    avg_improvement = total_improvement / count if count > 0 else 0.0

    return avg_improvement >= improvement_threshold