import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional

from utils.helper import log_print

class Hungarian:
    def __init__(self,
                pred_centroids:np.ndarray,
                ref_centroids:np.ndarray,
                valid_pred_mask:np.ndarray,
                valid_ref_mask:np.ndarray,
                max_dist:float=10.0,
                debug_print:bool=False,
                ):
        self.pred_centroids = pred_centroids[valid_pred_mask]
        self.ref_centroids = ref_centroids[valid_ref_mask]
        self.pred_mask = valid_pred_mask
        self.ref_mask = valid_ref_mask
        self.max_dist = max_dist
        self.debug_print = debug_print

        self.instance_count = valid_pred_mask.shape[0]
        self.inst_list = list(range(self.instance_count))
        self.full_set = np.all(self.pred_mask) and np.all(self.ref_mask)
        
        # Global indices
        self.pred_indices = np.where(self.pred_mask)[0]
        self.ref_indices  = np.where(self.ref_mask)[0]
        
    def hungarian_matching(self) -> Optional[np.ndarray]:
        """
        Perform identity correction using Hungarian algorithm.

        Returns:
            new_order: List[int] of length N, where:
                new_order[target_identity] = source_instance_index_in_current_frame
                i.e., "Identity j comes from current instance new_order[j]"
        """

        K, M = len(self.pred_centroids), len(self.ref_centroids)
        if K == 0 or M == 0:
            log_print(f"[HUN] No valid data for Hungarian matching (K={K}, M={M}). Returning default order.", enabled=self.debug_print)
            if K == 0: # Nothing to swap, nothing to be considered ambiguous
                return self.inst_list
            if M == 0: # Sus
                return None

        if K == 1 and M == 1:
            dist = np.linalg.norm(self.pred_centroids[0] - self.ref_centroids[0])
            log_print(f"[HUN] Single pair matching. Distance: {dist:.2f}, Max_dist: {self.max_dist}", enabled=self.debug_print)
            if dist < self.max_dist:
                new_order = self.inst_list.copy()
                new_order[self.ref_indices[0]] = self.pred_indices[0]
                log_print(f"[HUN] Single pair matched. New order: {new_order}", enabled=self.debug_print)
                return new_order
            else: # Sus
                log_print(f"[HUN] Single pair not matched (Dist: {dist:.2f} > {self.max_dist}).", enabled=self.debug_print)
                return None

        if self.full_set:
            distances = np.linalg.norm(self.pred_centroids - self.ref_centroids, axis=1)
            log_print(f"[HUN] All instances present and masks match. Distances: {distances}, Max_dist: {self.max_dist}", enabled=self.debug_print)
            if np.all(distances < self.max_dist):
                log_print("[HUN] All identities stable. Returning default order.", enabled=self.debug_print)
                return self.inst_list

        cost_matrix = np.linalg.norm(self.pred_centroids[:, np.newaxis] - self.ref_centroids[np.newaxis, :], axis=2)

        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            log_print(f"[HUN] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}", enabled=self.debug_print)
        except Exception as e:
            log_print(f"[HUN] Hungarian failed: {e}. Returning None.", enabled=self.debug_print)
            return None  # Hungarian failed, sus
        else:
            if self.full_set:
                if not self._dist_improv_comp(row_ind, col_ind):
                    log_print(f"[HUN] Hungarian assignment does not improve geometric distances sufficiently.", enabled=self.debug_print)
                    return None # Sus

        return self._build_new_order(row_ind, col_ind)

    def hungarian_matching_with_rotation(self, pred_rotation: np.ndarray, ref_rotation: np.ndarray) -> Optional[list]:
        """
        Match using centroid + rotation, where rotation acts as a tie-breaker.
        
        Args:
            pred_rotation: (K,) — rotations for valid predicted instances
            ref_rotation:  (M,) — rotations for valid reference instances
        Returns:
            new_order: list of length N, or None if failed
        """
        K, M = len(self.pred_centroids), len(self.ref_centroids)
        if K == 0 or M == 0:
            return self.inst_list if K == 0 else None

        cent_cost = np.linalg.norm(self.pred_centroids[:, None] - self.ref_centroids[None, :], axis=2)
        rot_cost = np.zeros_like(cent_cost)
        reliability = np.ones_like(cent_cost)
        pred_valid_rot = ~np.isnan(pred_rotation)
        ref_valid_rot  = ~np.isnan(ref_rotation)
        for i in range(K):
            for j in range(M):
                if not (pred_valid_rot[i] and ref_valid_rot[j]):
                    reliability[i, j] = 0.0 

        dtheta = pred_rotation[:, None] - ref_rotation[None, :]
        angular_sim = np.abs(np.cos(dtheta))
        rot_cost = 1.0 - angular_sim 

        min_cent_costs = np.min(cent_cost, axis=1)
        cent_std = np.std(min_cent_costs) if K > 1 else 0.0
        cent_mean = np.mean(min_cent_costs) if K > 0 else 0.0

        if cent_std < 0.3 * cent_mean and cent_mean > 0:  # ambiguous case
            rot_weight = 2.0
        else:
            rot_weight = 0.2

        combined_cost = cent_cost + rot_weight * rot_cost * reliability

        try:
            row_ind, col_ind = linear_sum_assignment(combined_cost)
            log_print(f"[HUN+ROT] Assignment: {list(zip(row_ind, col_ind))}", enabled=self.debug_print)
        except Exception as e:
            log_print(f"[HUN+ROT] Failed: {e}", enabled=self.debug_print)
            return None

        assigned_costs = combined_cost[row_ind, col_ind]
        if np.any(assigned_costs > self.max_dist * 1.5):
            log_print(f"[HUN+ROT] High cost ({assigned_costs}), rejecting.", enabled=self.debug_print)
            return None

        return self._build_new_order(row_ind, col_ind)

    def compare_distance_improvement(self, new_order:list) -> bool:
        """Wrapper for distance improvement comparison of a given new_order"""
        row_ind, col_ind = [], []

        for i, j in enumerate(new_order):
            if j in self.pred_indices:
                row_ind.append(j)
                col_ind.append(i)

        row_ind = np.array(row_ind, dtype=int)
        col_ind = np.array(col_ind, dtype=int)

        return self._dist_improv_comp(row_ind, col_ind)

    def _build_new_order(self, row_ind:np.ndarray, col_ind:np.ndarray) -> list:
        all_inst = range(self.instance_count)

        processed = {}
        for r, c in zip(row_ind, col_ind):
            target_identity = self.ref_indices[c]
            source_instance = self.pred_indices[r]
            processed[target_identity] = source_instance
        log_print(f"[HUN] Processed matches: {processed}", enabled=self.debug_print)

        unprocessed = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
        unassigned = [inst_idx for inst_idx in all_inst if inst_idx not in processed.values()]
        log_print(f"[HUN] Unprocessed identities: {unprocessed}, Unassigned instances: {unassigned}", enabled=self.debug_print)

        for target_identity in unprocessed:  # First loop, find remaining pair without idx change
            if target_identity in unassigned:
                source_instance = target_identity
                processed[target_identity] = source_instance
                unassigned.remove(source_instance)
        log_print(f"[HUN] Processed after first loop (self-assignment): {processed}", enabled=self.debug_print)
        
        unprocessed[:] = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
        log_print(f"[HUN] Unprocessed identities after first loop: {unprocessed}", enabled=self.debug_print)

        for target_identity in unprocessed:  # Second loop, arbitarily reassign
            source_instance = unassigned[-1]
            processed[target_identity] = source_instance
            unassigned.remove(source_instance)
        log_print(f"[HUN] Processed after second loop (arbitrary assignment): {processed}", enabled=self.debug_print)
            
        sorted_processed = {k: processed[k] for k in sorted(processed)}
        new_order = list(sorted_processed.values())
        log_print(f"[HUN] Final new_order: {new_order}", enabled=self.debug_print)

        return new_order
    
    def _dist_improv_comp(
            self, new_row_ind:np.ndarray, new_col_ind:np.ndarray, improvement_threshold:float=0.1
        ) -> bool:
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

        avg_improvement = total_improvement / count if count > 0 else 0.0

        return avg_improvement >= improvement_threshold