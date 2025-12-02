import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from typing import Optional, Literal, Tuple

from utils.helper import log_print

class Hungarian:
    def __init__(self,
                pred_centroids:np.ndarray,
                ref_centroids:np.ndarray,
                valid_pred_mask:np.ndarray,
                valid_ref_mask:np.ndarray,
                max_dist:float,
                ref_frame_gap:np.ndarray,
                motion_model:Literal["Diffusion", "Ballistic"]="Diffusion",
                debug_print:bool=False,
                ):
        self.pred_centroids = pred_centroids[valid_pred_mask]
        self.ref_centroids = ref_centroids[valid_ref_mask]
        self.pred_mask = valid_pred_mask
        self.ref_mask = valid_ref_mask
        self.max_dist = max_dist
        self.frame_gap = ref_frame_gap[valid_ref_mask]

        self.motion_model = motion_model
        self.debug_print = debug_print

        self.instance_count = valid_pred_mask.shape[0]
        self.inst_list = list(range(self.instance_count))
        self.full_set = np.all(self.pred_mask) and np.all(self.ref_mask)
        
        self.dym_dist = self._dynamic_max_dist()

        self.pred_indices = np.where(self.pred_mask)[0]
        self.ref_indices  = np.where(self.ref_mask)[0]
        
    def hungarian_matching(self) -> Optional[list]:
        """
        Perform identity correction using Hungarian algorithm.

        Returns:
            new_order: List[int] of length N, where:
                new_order[target_identity] = source_instance_index_in_current_frame
                i.e., "Identity j comes from current instance new_order[j]"
        """
        skip, new_order = self._hun_early_return()
        if skip:
            return new_order

        cost_matrix = np.linalg.norm(self.pred_centroids[:, np.newaxis] - self.ref_centroids[np.newaxis, :], axis=2)

        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            log_print(f"[HUN] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}", enabled=self.debug_print)
        except Exception as e:
            log_print(f"[HUN] Hungarian failed: {e}. Returning None.", enabled=self.debug_print)
            return None

        new_order = self._build_new_order(row_ind, col_ind)
        log_print(f"[HUN] Final new_order: {new_order}", enabled=self.debug_print)
        return new_order

    def compare_distance_improvement(self, new_order:list, improvement_threshold:float=0.1) -> Tuple[bool, float]:
        """Wrapper for distance improvement comparison of a given new_order"""
        row_ind, col_ind = [], []

        for i, j in enumerate(new_order):
            if j in self.pred_indices:
                row_ind.append(j)
                col_ind.append(i)

        row_ind = np.array(row_ind, dtype=int)
        col_ind = np.array(col_ind, dtype=int)
  
        avg_improv = self._dist_improv_comp(row_ind, col_ind)
 
        return (avg_improv>improvement_threshold), avg_improv

    def _hun_early_return(self) -> Tuple[bool, Optional[list]]:
        K, M = len(self.pred_centroids), len(self.ref_centroids)

        if K == 0 or M == 0:
            log_print(f"[HUN] No valid data for Hungarian matching (K={K}, M={M}).", enabled=self.debug_print)
            return (K != 0), None

        if K == 1:
            if M == 1:
                dist = np.linalg.norm(self.pred_centroids[0] - self.ref_centroids[0])
                max_dist = self.dym_dist[0]
                
                log_print(f"[HUN] Single pair matching. Distance: {dist:.2f}, Max_dist: {max_dist:.2f}", enabled=self.debug_print)
                if dist < max_dist:
                    new_order = self._build_new_order_simple(self.pred_indices[0], self.ref_indices[0])
                    log_print(f"[HUN] Single pair matched. New order: {new_order}", enabled=self.debug_print)
                    return True, new_order
                else:
                    log_print(f"[HUN] Single pair not matched (Dist: {dist:.2f} >= {max_dist:.2f}).",
                              enabled=self.debug_print)
                    return True, None
            else:
                dist = np.linalg.norm(self.pred_centroids[0] - self.ref_centroids, axis=1)
                ref_dg_array = np.column_stack((dist, self.dym_dist, self.frame_gap))
                valid = ref_dg_array[:, 1] > ref_dg_array[:, 0]
                if np.sum(valid) == 0:
                    log_print("[HUN] No singe pair matched.", enabled=self.debug_print)
                    for i in range(M):
                        log_print(f"Inst {i} | Distances: {dist[i]:.2f}, Max_dist: {self.dym_dist[i]:.2f}", enabled=self.debug_print)
                    ref_idx = self.ref_indices[np.argmin(dist)]
                    new_order = self._build_new_order_simple(self.pred_indices[0], ref_idx)
                    log_print(f"[HUN] Choose the pair with minimal dist. New order: {new_order}", enabled=self.debug_print)
                    return True, new_order
                elif np.sum(valid) == 1:
                    valid_idx_global = self.ref_indices[valid][0]
                    new_order = self._build_new_order_simple(self.pred_indices[0], valid_idx_global)
                    log_print(f"[HUN] Single pair matched. New order: {new_order}", enabled=self.debug_print)
                    return True, new_order
                else:
                    best_cand = np.argmin(ref_dg_array, axis=0)
                    dist_best, _, time_best = best_cand
                    if dist_best == time_best:
                        best_idx_global = self.ref_indices[dist_best]
                        new_order = self._build_new_order_simple(self.pred_indices[0], best_idx_global)
                        log_print(f"[HUN] Single pair matched. New order: {new_order}", enabled=self.debug_print)
                        return True, new_order
                    else:
                        log_print("[HUN] Distance-wise best instance is not the same as time-wise. Returning Distance best as new order.\n"
                                    f"Inst {dist_best} (Dist Best) | Distances: {dist[dist_best]:.2f}, Gap: {ref_dg_array[dist_best, 2]}\n"
                                    f"Inst {time_best} (Time Best) | Distances: {dist[time_best]:.2f}, Gap: {ref_dg_array[time_best, 2]}" ,enabled=self.debug_print)
                        new_order = self._build_new_order_simple(self.pred_indices[0], dist_best)
                        return True, new_order

        return False, None

    def _dynamic_max_dist(self):
        gap_arr = self.frame_gap.copy()
        gap_arr[gap_arr<1] = 1
        match self.motion_model:
            case "Ballistic":
                return self.max_dist * gap_arr
            case "Diffusion":
                return self.max_dist * np.sqrt(gap_arr)

    def _build_new_order(self, row_ind:np.ndarray, col_ind:np.ndarray) -> list:
        all_inst = range(self.instance_count)

        processed = {}
        for r, c in zip(row_ind, col_ind):
            target_identity = self.ref_indices[c]
            source_instance = self.pred_indices[r]
            processed[target_identity] = source_instance

        unprocessed = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
        unassigned = [inst_idx for inst_idx in all_inst if inst_idx not in processed.values()]

        for target_identity in unprocessed:
            if target_identity in unassigned:
                source_instance = target_identity
                processed[target_identity] = source_instance
                unassigned.remove(source_instance)
        
        unprocessed[:] = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]

        for target_identity in unprocessed:
            source_instance = unassigned[-1]
            processed[target_identity] = source_instance
            unassigned.remove(source_instance)
            
        sorted_processed = {k: processed[k] for k in sorted(processed)}
        new_order = list(sorted_processed.values())

        return new_order
    
    def _build_new_order_simple(self, pred_idx, ref_idx):
        new_order = self.inst_list.copy()
        new_order[pred_idx], new_order[ref_idx] = self.inst_list[ref_idx], self.inst_list[pred_idx]
        return new_order

    def _dist_improv_comp(
            self, new_row_ind:np.ndarray, new_col_ind:np.ndarray) -> float:
        """
        Compare assignments by sorting all assignment distances and comparing element-wise.

        Ignores which identity is assigned to which instance — only compares the overall
        distribution of match quality.

        Accepts new assignment if, on average, sorted distances are improved by threshold.
        """
        K = self.pred_centroids.shape[0]
        M = self.ref_centroids.shape[0]
        N = min(K, M)

        # Compute current assignment distances
        current_dists = []
        for j in range(len(self.inst_list)):
            if j >= M:
                continue
            i_old = self.inst_list[j]
            if i_old >= K:
                continue
            dist = np.linalg.norm(self.pred_centroids[i_old] - self.ref_centroids[j])
            current_dists.append(dist)

        if len(current_dists) == 1: # No other choice
            return float('inf')

        # Compute new assignment distances
        new_dists = []
        for r, c in zip(new_row_ind, new_col_ind):
            if c >= M or r >= K:
                continue
            dist = np.linalg.norm(self.pred_centroids[r] - self.ref_centroids[c])
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

        return total_improvement / count if count > 0 else 0.0

class Transylvanian(Hungarian):
    def __init__(self,
                pred_vectors:np.ndarray,
                ref_vectors:np.ndarray,
                valid_pred_mask:np.ndarray,
                valid_ref_mask:np.ndarray,
                max_dist:float,
                ref_frame_gap:np.ndarray,
                debug_print:bool=False,
                ):
        super().__init__(pred_vectors, ref_vectors, valid_pred_mask, valid_ref_mask, max_dist, ref_frame_gap, debug_print)
        self.pred_vec = pred_vectors[valid_pred_mask]
        self.ref_vec = ref_vectors[valid_ref_mask]
        self.debug_print = debug_print

    def vector_hun_match(self):
        cost_matrix = np.linalg.norm(self.pred_vec[:, np.newaxis] - self.ref_vec[np.newaxis, :], axis=2)
        weight_cost_matrix = cost_matrix * np.sqrt(self.frame_gap)
        
        try:
            row_ind, col_ind = linear_sum_assignment(weight_cost_matrix)
            log_print(f"[HUN(VEC)] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}", enabled=self.debug_print)
        except Exception as e:
            log_print(f"[HUN(VEC)] Hungarian failed: {e}. Returning None.", enabled=self.debug_print)
            return None

        new_order = self._build_new_order(row_ind, col_ind)
        log_print(f"[HUN(VEC)] Final new_order: {new_order}", enabled=self.debug_print)
        return new_order