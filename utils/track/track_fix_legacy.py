import numpy as np
from scipy.optimize import linear_sum_assignment

from PySide6.QtWidgets import QProgressDialog
from typing import Optional, Literal, List, Tuple, Optional

from utils.pose import calculate_pose_centroids
from utils.helper import get_instance_count_per_frame
from utils.logger import logger


class Hungarian:
    def __init__(self,
                pred_centroids:np.ndarray,
                ref_centroids:np.ndarray,
                valid_pred_mask:np.ndarray,
                valid_ref_mask:np.ndarray,
                max_dist:float,
                ref_frame_gap:np.ndarray,
                motion_model:Literal["Diffusion", "Ballistic"]="Diffusion",
                ):
        self.pred_centroids = pred_centroids[valid_pred_mask]
        self.ref_centroids = ref_centroids[valid_ref_mask]
        self.pred_mask = valid_pred_mask
        self.ref_mask = valid_ref_mask
        self.max_dist = max_dist
        self.frame_gap = ref_frame_gap[valid_ref_mask]

        self.ref_centroids_full = ref_centroids
        self.frame_gap_full = ref_frame_gap
        self.motion_model = motion_model

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

    def hungarian_skipper(self) -> Tuple[bool, Optional[list]]:
        K, M = len(self.pred_centroids), len(self.ref_centroids)

        if K == 0 or M == 0:
            logger.debug(f"[HUN] No valid data for Hungarian matching (K={K}, M={M}).")
            return (K != 0), None

        if K == 1 and M == 1:
            dist = np.linalg.norm(self.pred_centroids[0] - self.ref_centroids[0])
            max_dist = self.dym_dist[0]
            
            logger.debug(f"[HUN] Single pair matching. Distance: {dist:.2f}, Max_dist: {max_dist:.2f}")
            if dist < max_dist:
                new_order = self._build_new_order_simple(self.pred_indices[0], self.ref_indices[0])
                logger.debug(f"[HUN] Single pair matched. New order: {new_order}")
                return True, new_order
            elif dist < max_dist * 2:
                logger.debug(f"[HUN] Single pair semi matched (Dist: {max_dist:.2f} <= {dist:.2f} < {2*max_dist:.2f}).")
                alt_idx_found = -1
                dist_alt_found = 0
                for alt_idx in self.inst_list:
                    if alt_idx == self.pred_indices[0]:
                        continue
                    if alt_idx in self.ref_indices:
                        continue
                    ref_centroids_ancient = self.ref_centroids_full[alt_idx]
                    if np.any(np.isnan(ref_centroids_ancient)):
                        continue
                    dist_alt = np.linalg.norm(self.pred_centroids[0] - ref_centroids_ancient)
                    if dist_alt > max_dist * 2:
                        continue
                    alt_idx_found = alt_idx
                    dist_alt_found = dist_alt

                if alt_idx_found != -1:
                    f"[HUN] Single pair ambiguous (Inst {alt_idx_found} from {self.frame_gap_full[alt_idx_found]} frames away, dist: {dist_alt_found:.2f} <= {2*max_dist:.2f})."
                    return True, None
                else:
                    new_order = self._build_new_order_simple(self.pred_indices[0], self.ref_indices[0])
                    logger.debug(f"[HUN] Single pair matched. New order: {new_order}")
                    return True, new_order
            else:
                logger.debug(f"[HUN] Single pair not matched (Dist: {dist:.2f} >= {2*max_dist:.2f}).")
                return True, None
        
        if K == 1 and M != 1:
            dist = np.linalg.norm(self.pred_centroids[0] - self.ref_centroids, axis=1)
            ref_dg_array = np.column_stack((dist, self.dym_dist, self.frame_gap))
            valid = ref_dg_array[:, 1] > ref_dg_array[:, 0]
            if np.sum(valid) == 0:
                logger.debug("[HUN] No singe pair matched.")
                for i in range(M):
                    logger.debug(f"Inst {i} | Distances: {dist[i]:.2f}, Max_dist: {self.dym_dist[i]:.2f}")
                return True, None
            elif np.sum(valid) == 1:
                valid_idx_global = self.ref_indices[valid][0]
                new_order = self._build_new_order_simple(self.pred_indices[0], valid_idx_global)
                logger.debug(f"[HUN] Single pair matched. New order: {new_order}")
                return True, new_order
            else:
                dist_best = np.argmin(ref_dg_array[:, 0])
                time_best = np.where(ref_dg_array[:, 2]==np.min(ref_dg_array[:,2]))[0]
                if len(time_best) > 1:
                    logger.debug(f"[HUN] Multiple Time Best candidates. Reroute to Hungarian.")
                    return False, None
                time_best = time_best[0]
                if dist_best == time_best:
                    best_idx_global = self.ref_indices[dist_best]
                    new_order = self._build_new_order_simple(self.pred_indices[0], best_idx_global)
                    logger.debug(f"[HUN] Single pair matched. New order: {new_order}")
                    return True, new_order
                else:
                    logger.debug("[HUN] Distance-wise best instance is not the same as time-wise. Mark as ambiguous.\n"
                                f"Inst {dist_best} (Dist Best) | Distances: {dist[dist_best]:.2f}, Gap: {ref_dg_array[dist_best, 2]}\n"
                                f"Inst {time_best} (Time Best) | Distances: {dist[time_best]:.2f}, Gap: {ref_dg_array[time_best, 2]}")
                    new_order = self._build_new_order_simple(self.pred_indices[0], dist_best)
                    return True, None

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


class Track_Fixer:
    def __init__(self,
                pred_data_array:np.ndarray,
                exit_zone: Optional[np.ndarray]=None,
                progress:Optional[QProgressDialog]=None,
                ):
        self.progress = progress

        self.corrected_pred_data = pred_data_array.copy()
        self.inst_count_per_frame = get_instance_count_per_frame(self.corrected_pred_data)

        self.total_frames, self.instance_count = self.corrected_pred_data.shape[:2]
        if self.progress:
            self.progress.setRange(0, self.total_frames)

        self.inst_list = list(range(self.instance_count))
        self.inst_array = np.array(self.inst_list)

        self.new_order_array = -np.ones((self.total_frames, self.instance_count), dtype=np.int8)
        self.current_frame_data = np.full_like(self.corrected_pred_data[0], np.nan)

        self.exit_zone = exit_zone
        self.max_tunnel_gap = 10 
        self.tunneling_mouse_id = None 

    def track_correction(self, max_dist:float=10.0, lookback_window=10, start_idx=0) -> Tuple[np.ndarray, List[int], List[int]]:
        try:
            gap_mask = np.all(np.isnan(self.corrected_pred_data), axis=(1,2))
            self._first_pass_centroids(max_dist, lookback_window, start_idx)
            self.new_order_array[gap_mask] = self.inst_array
            self.new_order_array[0:start_idx+1] = self.inst_array
            if np.any(self.new_order_array == -1):
                amogus_frames = np.where(self.new_order_array[:,0]==-1)[0]
                self.new_order_array[amogus_frames] = self.inst_array
                amogus_frames = amogus_frames.tolist()
            else:
                amogus_frames = []
        except Exception as e:
            logger.exception(f"Track correction failed: {e}")
            return self.corrected_pred_data, 0, []

        if self.progress:
            self.progress.close()

        diff = self.new_order_array - np.arange(self.instance_count)
        changed_frames = np.where(np.any(diff != 0, axis=1))[0].tolist()
        fixed_data_array = self.corrected_pred_data

        return fixed_data_array, changed_frames, amogus_frames

    def _first_pass_centroids(self, max_dist:float, lookback_window, start_idx):
        last_order = self.inst_list
        ref_centroids = np.full((self.instance_count, 2), np.nan)
        ref_last_updated = np.full((self.instance_count,), -2 * lookback_window, np.int32)
        valid_ref_mask = np.zeros_like(ref_last_updated, dtype=bool)

        for frame_idx in range(self.total_frames):
            if frame_idx < start_idx:
                continue

            if self.progress:
                self.progress.setValue(frame_idx)
                if self.progress.wasCanceled():
                    raise RuntimeError("User cancelled track fixing operation.")
            
            logger.debug(f"---------- frame: {frame_idx} ---------- ")

            pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
            valid_pred_mask = np.any(~np.isnan(pred_centroids), axis=1)

            if not np.any(valid_pred_mask):
                self.new_order_array[frame_idx] = self.inst_array
                logger.debug("[TMOD] Skipping due to no prediction on current frame.")
                continue

            if (self.exit_zone is not None and self.instance_count == 2 and self.tunneling_mouse_id is not None):
                T = self.tunneling_mouse_id
                O = 1 - T
                x1, y1, x2, y2 = self.exit_zone

                if valid_pred_mask[T]:
                    x, y = pred_centroids[T]
                    in_exit_zone = (x1 <= x <= x2) and (y1 <= y <= y2)

                    if in_exit_zone:
                        logger.debug(f"[TUNNEL] Mouse {T} detected in exit zone, but awaiting Hungarian assignment for identity confirmation.")

                    elif not valid_pred_mask[O]:
                        logger.debug(f"[TUNNEL] Only slot {T} filled (outside zone) → reassigning to slot {O}")
                        self.corrected_pred_data[frame_idx, O] = self.corrected_pred_data[frame_idx, T].copy()
                        self.corrected_pred_data[frame_idx, T] = np.nan
                        pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
                        valid_pred_mask = np.any(~np.isnan(pred_centroids), axis=1)

                    elif ref_last_updated[T] >= frame_idx - lookback_window: # Recently vanished so maybe not vanished at all?
                        logger.debug(f"[TUNNEL] Mouse {T} possibly never entered the tunnel.")
                        self.tunneling_mouse_id = None

                    else:
                        x_O, y_O = pred_centroids[O]
                        in_exit_zone_O = (x1 <= x_O <= x2) and (y1 <= y_O <= y2)

                        if not in_exit_zone_O:
                            logger.debug(f"[TUNNEL] Spurious detection of mouse {T} (outside zone, O present) → discarding.")
                            self.corrected_pred_data[frame_idx, T] = np.nan
                            pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
                            valid_pred_mask = np.any(~np.isnan(pred_centroids), axis=1)

                elif valid_pred_mask[O]:
                    x, y = pred_centroids[O]
                    in_exit_zone = (x1 <= x <= x2) and (y1 <= y <= y2)

                    if in_exit_zone:
                        last_O_pos = ref_centroids[O]
                        if np.any(np.isnan(last_O_pos)):
                            logger.debug(f"[TUNNEL] Only slot {O} in zone, no ref for O → assuming T return.")
                            self.corrected_pred_data[frame_idx, T] = self.corrected_pred_data[frame_idx, O].copy()
                            self.corrected_pred_data[frame_idx, O] = np.nan
                            pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
                            valid_pred_mask = np.any(~np.isnan(pred_centroids), axis=1)
                            self.tunneling_mouse_id = None
                        else:
                            dist_from_last_O = np.linalg.norm(np.array([x, y]) - last_O_pos)
                            gap = frame_idx - ref_last_updated[O]
                            gap = max(1, gap)
                            max_allowed = max_dist * np.sqrt(gap)

                            if dist_from_last_O > max_allowed:
                                logger.debug(f"[TUNNEL] Slot {O} in zone but too far from last O ({dist_from_last_O:.1f} > {max_allowed:.1f}) → assuming T return.")
                                self.corrected_pred_data[frame_idx, T] = self.corrected_pred_data[frame_idx, O].copy()
                                self.corrected_pred_data[frame_idx, O] = np.nan
                                pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
                                valid_pred_mask = np.any(~np.isnan(pred_centroids), axis=1)
                                self.tunneling_mouse_id = None
                            else:
                                logger.debug(f"[TUNNEL] Slot {O} in zone, within motion bounds → treating as O loitering.")

            valid_ref_mask = (ref_last_updated > frame_idx - lookback_window)
            current_visible = np.sum(valid_pred_mask)
            ref_visible = np.sum(valid_ref_mask)

            if (self.exit_zone is not None and self.instance_count == 2 and current_visible == 1 and ref_visible == 2):
                x1, y1, x2, y2 = self.exit_zone
                candidate_tunnel_mouse = []

                for mouse_id in range(2):
                    x, y = ref_centroids[mouse_id]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        candidate_tunnel_mouse.append(mouse_id)

                if candidate_tunnel_mouse:
                    if len(candidate_tunnel_mouse) == 1:
                        self.tunneling_mouse_id = candidate_tunnel_mouse[0]
                    else:
                        self.tunneling_mouse_id = 1 - np.where(valid_pred_mask)[0][0]
                    logger.debug(f"[TUNNEL] Mouse {self.tunneling_mouse_id} entered tunnel (last ref pos in exit zone).")
                else:
                    logger.debug("[TUNNEL] One mouse vanished, but neither ref was in exit zone — ignoring.")

            if not np.any(valid_ref_mask):
                if np.all(np.isnan(ref_centroids)): # No ref at all, i.e. beginning of the vid
                    self.new_order_array[frame_idx] = np.array(last_order)
                else:
                    self.new_order_array[frame_idx] = -1 # Mark as ambiguous
                logger.debug("[TMOD] No valid ref frame found. Rebasing the reference on current frame.")
                ref_centroids[valid_pred_mask] = pred_centroids[valid_pred_mask] # Rebase the prediction on current frame
                ref_last_updated[valid_pred_mask] = frame_idx
                last_order = self.inst_list
                continue

            for i in range(self.instance_count):
                logger.debug(f"x,y in pred: inst {i}: ({pred_centroids[i,0]:.1f}, {pred_centroids[i,1]:.1f})")
            for i in range(self.instance_count):
                logger.debug(f"x,y in ref: inst {i}: ({ref_centroids[i,0]:.1f}, {ref_centroids[i,1]:.1f}) | "
                            f"last updated: {ref_last_updated[i]} | valid: {valid_ref_mask[i]}")

            new_order = None
            hun = Hungarian(pred_centroids, ref_centroids, valid_pred_mask, valid_ref_mask,
                            max_dist=max_dist, ref_frame_gap=frame_idx-ref_last_updated)

            skip_matching, new_order = self._hungarian_skip_check(hun, frame_idx, last_order, max_dist)

            if not skip_matching:
                new_order = hun.hungarian_matching()

                if new_order and new_order != self.inst_list:
                    improved, avg_improv = hun.compare_distance_improvement(new_order)
                    if improved:
                        logger.debug(f"[TMOD] Hungarian order improves overall distance by {avg_improv*100:.2f}% | Threshold: 10%")
                    else:
                        logger.debug(f"[TMOD] Hungarian order fails to sufficiently improve overall distance: {avg_improv*100:.2f}% | Threshold: 10%")
                        new_order = None

                if new_order is None:
                    logger.debug("[TMOD] Failed to build new order.")
                    self.new_order_array[frame_idx] = -1 # Ambiguous
                else:
                    if new_order == self.inst_list:
                        logger.debug("[TMOD] NO SWAP, already the best solution.")
                    else:
                        logger.debug(f"[TMOD] SWAP, new_order: {new_order}.")

            if new_order:
                self.new_order_array[frame_idx] = np.array(new_order)
                self.corrected_pred_data[frame_idx, :, :] = self.corrected_pred_data[frame_idx, new_order, :]
                last_order = new_order

            fixed_pred_centroids = pred_centroids[new_order] if new_order else pred_centroids
            fixed_pred_mask = valid_pred_mask[new_order] if new_order else valid_pred_mask

            if self.tunneling_mouse_id is not None:
                T = self.tunneling_mouse_id
                x1, y1, x2, y2 = self.exit_zone
                x, y = fixed_pred_centroids[T]

                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.tunneling_mouse_id = None
                    logger.debug(f"[TUNNEL] Mouse {T} (post Hungarian correction) returned via exit zone.")

            ref_centroids[fixed_pred_mask] = fixed_pred_centroids[fixed_pred_mask]
            ref_last_updated[fixed_pred_mask] = frame_idx

    def _second_pass_disambiguation(self, ambiguous_frames:List[int], max_dist:float, lookback_window:int):
        final_ambi_frames = ambiguous_frames.copy()
        for i, amb_idx in enumerate(ambiguous_frames):
            if i == 0:
                continue
            last_amb_idx = ambiguous_frames[i-1]
            if amb_idx - last_amb_idx >= lookback_window:
                logger.debug(
                    f"[DAM] Frame {amb_idx}: gap from prev ambig ({last_amb_idx}) = {amb_idx - last_amb_idx} ≥ {lookback_window} → new segment.")
                continue
            if self.inst_count_per_frame[amb_idx] != self.inst_count_per_frame[last_amb_idx]:
                logger.debug(
                    f"[DAM] Frame {amb_idx}: inst count changed ({self.inst_count_per_frame[last_amb_idx]} → {self.inst_count_per_frame[amb_idx]}) → preserved.")
                continue

            cur_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, amb_idx)
            last_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, last_amb_idx)
            cur_indices = sorted(np.where(np.all(~np.isnan(cur_centroids), axis=1))[0])
            last_indices = sorted(np.where(np.all(~np.isnan(last_centroids), axis=1))[0])

            if cur_indices != last_indices:
                logger.debug(
                    f"[DAM] Frame {amb_idx}: valid instances changed ({last_indices} → {cur_indices}) → preserved.")
                continue
            
            pair_dist_max = 0.0
            for inst_idx in cur_indices:
                pair_dist = np.linalg.norm(cur_centroids[inst_idx]-last_centroids[inst_idx])
                pair_dist_max = max(pair_dist_max, pair_dist)

            if pair_dist_max > max_dist:
                logger.debug(
                    f"[DAM] Frame {amb_idx}: max inst displacement = {pair_dist_max:.1f} > {max_dist} (from {last_amb_idx}) → preserved.")
                continue

            final_ambi_frames.remove(amb_idx)
            logger.debug(
                f"[DAM] Frame {amb_idx}: collapsed into segment (prev ambig {last_amb_idx}, max Δ = {pair_dist_max:.1f} ≤ {max_dist}).")

        logger.debug(f"[DAM] Second pass: {len(ambiguous_frames)} -> {len(final_ambi_frames)} ambiguous frames.")
        return final_ambi_frames

    def _hungarian_skip_check(self, hun:Hungarian, frame_idx:int, last_order:List[int], max_dist:float)-> Tuple[bool, List[int]]:
        skip_matching, new_order = False, None

        if last_order != self.inst_list and hun.full_set: # Try last order first if possible
            improved, avg_improv = hun.compare_distance_improvement(last_order, 0.2)
            if improved:
                logger.debug(f"[TMOD] Last order improves overall distance by {avg_improv*100:.2f}% | Threshold: 20%")
                new_order = last_order
                self.new_order_array[frame_idx] = np.array(last_order)
                logger.debug(f"[TMOD] SWAP, Applying the last order: {last_order}")
                skip_matching = True
            else:
                logger.debug(f"[TMOD] Last order fails to sufficiently improve overall distance: {avg_improv*100:.2f}% | Threshold: 20%")

        return skip_matching, new_order