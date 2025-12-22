import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from typing import Optional, Tuple

from utils.logger import logger


class Hungarian:
    def __init__(self,
                pred_centroids:np.ndarray,
                ref_centroids:np.ndarray,
                valid_pred_mask:np.ndarray,
                valid_ref_mask:np.ndarray,
                ref_age:np.ndarray,
                max_dist:float,
                ):
        self.pred_centroids = pred_centroids[valid_pred_mask]
        self.ref_centroids = ref_centroids[valid_ref_mask]
        self.pred_mask = valid_pred_mask
        self.ref_mask = valid_ref_mask
        self.max_dist = max_dist
        self.ref_age = ref_age

        self.ref_centroids_full = ref_centroids

        self.instance_count = valid_pred_mask.shape[0]
        self.inst_list = list(range(self.instance_count))
        self.full_set = np.all(self.pred_mask) and np.all(self.ref_mask)

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
        cost_matrix = np.linalg.norm(self.pred_centroids[:, np.newaxis] - self.ref_centroids[np.newaxis, :], axis=2)

        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            logger.debug(f"[HUN] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}")
        except Exception as e:
            logger.debug(f"[HUN] Hungarian failed: {e}. Returning None.")
            return None

        new_order = self._build_new_order(row_ind, col_ind)
        logger.debug(f"[HUN] Final new_order: {new_order}")
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

    def hungarian_skipper(self, lookback_window:int) -> Tuple[bool, Optional[list]]:
        K, M = len(self.pred_centroids), len(self.ref_centroids)

        if K == 0 or M == 0:
            logger.debug(f"[HUN] No valid data for Hungarian matching (K={K}, M={M}).")
            return (K != 0), None

        if K == 1 and M == 1:
            dist = np.linalg.norm(self.pred_centroids[0] - self.ref_centroids[0])
            
            logger.debug(f"[HUN] Single pair matching. Distance: {dist:.2f}, Max_dist: {self.max_dist:.2f}")
            if dist < self.max_dist:
                new_order = self._build_new_order_simple(self.pred_indices[0], self.ref_indices[0])
                logger.debug(f"[HUN] Single pair matched. New order: {new_order}")
                return True, new_order
            elif dist < self.max_dist * 2:
                logger.debug(f"[HUN] Single pair semi matched (Dist: {self.max_dist:.2f} <= {dist:.2f} < {2*self.max_dist:.2f}).")
                alt_idx_found = -1
                dist_alt_found = 0
                for alt_idx in self.inst_list:
                    if alt_idx in self.ref_indices:
                        continue
                    ref_centroids_ancient = self.ref_centroids_full[alt_idx]
                    if np.any(np.isnan(ref_centroids_ancient)):
                        continue
                    dist_alt = np.linalg.norm(self.pred_centroids[0] - ref_centroids_ancient)
                    if dist_alt > self.max_dist * 2:
                        continue
                    if self.ref_age[alt_idx] > lookback_window*2:
                        continue
                    alt_idx_found = alt_idx
                    dist_alt_found = dist_alt

                if alt_idx_found != -1:
                    logger.debug(
                        f"[HUN] Single pair ambiguous (Inst {alt_idx_found} from {self.ref_age[alt_idx_found]} frames away,\
                            dist: {dist_alt_found:.2f} <= {2*self.max_dist:.2f}).")
                    return True, None
                else:
                    new_order = self._build_new_order_simple(self.pred_indices[0], self.ref_indices[0])
                    logger.debug(f"[HUN] Single pair matched. New order: {new_order}")
                    return True, new_order
            else:
                logger.debug(f"[HUN] Single pair not matched (Dist: {dist:.2f} >= {2*self.max_dist:.2f}).")
                return True, None

        if K == 1 and M != 1:
            dist = np.linalg.norm(self.pred_centroids[0] - self.ref_centroids, axis=1)
            valid = dist < self.max_dist

            if np.sum(valid) == 0:
                logger.debug("[HUN] No singe pair matched.")
                for i in range(M):
                    logger.debug(f"Inst {i} | Distances: {dist[i]:.2f}, Max_dist: {self.max_dist:.2f}")
                return True, None
            elif np.sum(valid) == 1:
                valid_idx_global = self.ref_indices[valid][0]
                new_order = self._build_new_order_simple(self.pred_indices[0], valid_idx_global)
                logger.debug(f"[HUN] Single pair matched. New order: {new_order}")
                return True, new_order
            else:
                idx = np.argsort(dist)
                smallest = dist[idx[0]]
                small_2st = dist[idx[1]]
                if (small_2st - smallest) / (small_2st + smallest) < 0.5:
                    logger.debug(f"[HUN] Single pair ambiguous. Smallest dist: {smallest}, small_2st dist: {small_2st}")
                    return True, None
                else:
                    valid_idx_global = self.ref_indices[valid][idx[0]]
                    new_order = self._build_new_order_simple(self.pred_indices[0], valid_idx_global)
                    logger.debug(f"[HUN] Single pair matched. New order: {new_order}")

        if self.danger_close(self.pred_centroids, self.ref_centroids, self.max_dist):
            return True, None

        return False, None

    def danger_close(self, pred_centroids, ref_centroids, max_dist:float) -> bool:
        K, M = len(pred_centroids), len(ref_centroids)
        if K >= 2:
            all_d = []
            for i, j in combinations(range(K), 2):
                d = np.linalg.norm(pred_centroids[i] - pred_centroids[j])
                all_d.append(d)
                if d < max_dist:
                    logger.debug(f"[HUN] Ambiguous pred layout (pred {i}–{j}: {d:.1f} < {max_dist}).")
                    return True
            logger.debug(f"[HUN] All pred pairwise distance: {all_d}")

        if M >= 2:
            all_d = []
            for i, j in combinations(range(M), 2):
                d = np.linalg.norm(ref_centroids[i] - ref_centroids[j])
                all_d.append(d)
                if d < max_dist:
                    logger.debug(f"[HUN] Ambiguous ref layout (ref {i}–{j}: {d:.1f} < {max_dist}).")
                    return True
            logger.debug(f"[HUN] All ref pairwise distance: {all_d}")
        
        return False

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