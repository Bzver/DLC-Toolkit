import numpy as np
from collections import defaultdict
from PySide6.QtWidgets import QProgressDialog
from typing import List, Dict, Any, Tuple, Optional

from .hungarian import Hungarian
from utils.pose import (
    calculate_pose_centroids, outlier_removal, outlier_confidence, 
    outlier_bodypart, outlier_enveloped, outlier_size, outlier_pose,
    )
from utils.helper import get_instance_count_per_frame
from utils.logger import logger


class Track_Fixer:
    def __init__(self,
                pred_data_array:np.ndarray,
                canon_pose:Optional[np.ndarray]=None,
                angle_map:Optional[Dict[str, Any]]=None,
                progress:Optional[QProgressDialog]=None,
                ):
        self.canon_pose = canon_pose
        self.angle_map = angle_map
        self.progress = progress

        self.pred_data_cleaned = self._pose_clean(pred_data_array, conservative=True)
        self.corrected_pred_data = self._pose_clean(pred_data_array, conservative=False)
        self.inst_count_per_frame = get_instance_count_per_frame(self.corrected_pred_data)

        self.total_frames, self.instance_count = self.corrected_pred_data.shape[:2]
        if self.progress:
            self.progress.setRange(0, self.total_frames)

        self.inst_list = list(range(self.instance_count))

        self.new_order_dict: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        self.ambiguous_frames = []
        self.current_frame_data = np.full_like(self.corrected_pred_data[0], np.nan)

    def track_correction(self, max_dist:float=10.0, lookback_window=10) -> Tuple[np.ndarray, int, List[int]]:
        try:
            self._first_pass_centroids(max_dist, lookback_window)
            if self.ambiguous_frames:
                amogus_frames = self._second_pass_disambiguation(max_dist, lookback_window)
            else:
                amogus_frames = []
        except Exception as e:
            logger.exception(f"Track correction failed: {e}")
            return self.corrected_pred_data, [], []

        if self.progress:
            self.progress.close()

        changed_frames = []
        fixed_data_array = self.pred_data_cleaned.copy()
        for new_order, frame_list in self.new_order_dict.items():
            if new_order == tuple(self.inst_list):
                continue
            fixed_data_array[frame_list, :, :] = self.pred_data_cleaned[frame_list][:, new_order, :]
            changed_frames.extend(frame_list)

        return fixed_data_array, sorted(set(changed_frames)), sorted(set(amogus_frames))

    def _first_pass_centroids(self, max_dist:float, lookback_window, decay=0.8):
        last_order = self.inst_list
        ref_centroids = np.full((self.instance_count, 2), np.nan)
        ref_centroids_projected = np.full((self.instance_count, 2), np.nan)
        ref_velocities = np.full((self.instance_count, 2), np.nan)
        ref_last_updated = np.full((self.instance_count,), -2 * lookback_window, np.int32)
        valid_ref_mask = np.zeros_like(ref_last_updated, dtype=bool)

        for frame_idx in range(self.total_frames):
            if self.progress:
                self.progress.setValue(frame_idx)
                if self.progress.wasCanceled():
                    raise RuntimeError("User cancelled track fixing operation.")
            
            logger.debug(f"---------- frame: {frame_idx} ---------- ")

            pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
            valid_pred_mask = np.any(~np.isnan(pred_centroids), axis=1)
            if not np.any(valid_pred_mask):
                logger.debug("[TMOD] Skipping due to no prediction on current frame.")
                continue

            valid_ref_mask = (ref_last_updated > frame_idx - lookback_window)
            ref_centroids_projected[:] = ref_centroids[:]

            if not np.any(valid_ref_mask):
                if not np.all(np.isnan(ref_centroids)):
                    self.ambiguous_frames.append(frame_idx)
                logger.debug("[TMOD] No valid ref frame found. Rebasing the reference on current frame.")
                ref_centroids[valid_pred_mask] = pred_centroids[valid_pred_mask] # Rebase the prediction on current frame
                ref_velocities[valid_ref_mask] = 0.0
                ref_last_updated[valid_pred_mask] = frame_idx
                last_order = self.inst_list
                continue

            for inst_idx in range(self.instance_count):
                age = frame_idx - ref_last_updated[inst_idx]
                if not np.any(np.isnan(ref_velocities[inst_idx])):
                    xy_motion = ref_velocities[inst_idx] * ((1-(decay)**age)/(1-decay))
                    ref_centroids_projected[inst_idx] = ref_centroids[inst_idx] + xy_motion
                    logger.debug(f"[MOTION] Inst {inst_idx}: Velocity after decay: x: {xy_motion[0]}, y: {xy_motion[1]}.")

            for i in range(self.instance_count):
                logger.debug(f"x,y in pred: inst {i}: ({pred_centroids[i,0]:.1f}, {pred_centroids[i,1]:.1f})")
            for i in range(self.instance_count):
                proj_x, proj_y = ref_centroids_projected[i]
                logger.debug(f"x,y in ref_proj: inst {i}: ({proj_x:.1f}, {proj_y:.1f}) ")
            for i in range(self.instance_count):
                logger.debug(f"x,y in ref: inst {i}: ({ref_centroids[i,0]:.1f}, {ref_centroids[i,1]:.1f}) | "
                            f"last updated: {ref_last_updated[i]} | valid: {valid_ref_mask[i]}")

            new_order = None
            hun = Hungarian(pred_centroids, ref_centroids_projected, valid_pred_mask, valid_ref_mask,
                            ref_age=frame_idx-ref_last_updated, max_dist=max_dist)

            skip_matching, new_order = self._hungarian_skip_check(hun, frame_idx, last_order, max_dist, lookback_window)

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
                else:
                    if new_order == self.inst_list:
                        logger.debug("[TMOD] NO SWAP, already the best solution.")
                    else:
                        logger.debug(f"[TMOD] SWAP, new_order: {new_order}.")

            if new_order:
                self.new_order_dict[tuple(new_order)].append(frame_idx)
                self.corrected_pred_data[frame_idx, :, :] = self.corrected_pred_data[frame_idx, new_order, :]
                last_order = new_order
                fixed_pred_centroids = pred_centroids[new_order]
                fixed_pred_mask = valid_pred_mask[new_order]
            else:
                fixed_pred_centroids = pred_centroids
                fixed_pred_mask = valid_pred_mask
                self.ambiguous_frames.append(frame_idx)

            ref_velocities = self._get_ref_velocities(fixed_pred_centroids, ref_centroids, ref_velocities, fixed_pred_mask, frame_idx-ref_last_updated)
            ref_centroids[fixed_pred_mask] = fixed_pred_centroids[fixed_pred_mask]
            ref_last_updated[fixed_pred_mask] = frame_idx

    def _second_pass_disambiguation(self, max_dist:float, lookback_window:int):
        final_ambi_frames = self.ambiguous_frames.copy()
        for i, amb_idx in enumerate(self.ambiguous_frames):
            if i == 0:
                continue
            last_amb_idx = self.ambiguous_frames[i-1]
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

        logger.debug(f"[DAM] Second pass: {len(self.ambiguous_frames)} -> {len(final_ambi_frames)} ambiguous frames.")
        return final_ambi_frames

    def _hungarian_skip_check(
            self, hun:Hungarian, frame_idx:int, last_order:List[int], max_dist:float, lookback_window:int
            )-> Tuple[bool, List[int]]:
        skip_matching, new_order = False, None

        if last_order != self.inst_list and hun.full_set: # Try last order first if possible
            improved, avg_improv = hun.compare_distance_improvement(last_order, 0.2)
            if improved:
                logger.debug(f"[TMOD] Last order improves overall distance by {avg_improv*100:.2f}% | Threshold: 20%")
                new_order = last_order
                self.new_order_dict[tuple(last_order)].append(frame_idx)
                logger.debug(f"[TMOD] SWAP, Applying the last order: {last_order}")
                skip_matching = True
            else:
                logger.debug(f"[TMOD] Last order fails to sufficiently improve overall distance: {avg_improv*100:.2f}% | Threshold: 20%")
        else:
            if frame_idx > 0 and self.inst_count_per_frame[frame_idx] != self.inst_count_per_frame[frame_idx - 1]:
                if self.inst_count_per_frame[frame_idx] >= 2 or self.inst_count_per_frame[frame_idx - 1] >= 2:
                    pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
                    prev_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx - 1)
                    if hun.danger_close(
                        pred_centroids[np.any(~np.isnan(pred_centroids), axis=1)],
                        prev_centroids[np.any(~np.isnan(prev_centroids), axis=1)],
                        max_dist=max_dist
                        ):
                        return True, None

            skip_matching, new_order = hun.hungarian_skipper(lookback_window)

        return skip_matching, new_order

    def _get_ref_velocities(self, fixed_pred_centroids, ref_centroids, ref_velocities, fixed_pred_mask, frame_gap) -> np.ndarray:
        if not np.any(fixed_pred_mask):
            return ref_velocities

        for inst_id in self.inst_list:
            if not fixed_pred_mask[inst_id]:
                continue
            temp_gap = frame_gap[inst_id]
            dist_gap = fixed_pred_centroids[inst_id] - ref_centroids[inst_id]
            ref_velocities[inst_id] = dist_gap / temp_gap

        return ref_velocities

    def _pose_clean(self, pred_data_array:np.ndarray, conservative:bool=True) -> np.ndarray:
        num_keypoint = pred_data_array.shape[2] // 3

        if conservative:
            env_mask = outlier_enveloped(pred_data_array, 0.9)
            conf_mask = outlier_confidence(pred_data_array, 0.4)
            bp_mask = outlier_bodypart(pred_data_array, 3)
            combined_mask = conf_mask | bp_mask | env_mask
            corrected_data_array, _, _ = outlier_removal(pred_data_array, combined_mask)
            return corrected_data_array

        no_mask = np.zeros((pred_data_array.shape[0], pred_data_array.shape[1]), dtype=bool)
        conf_mask = outlier_confidence(pred_data_array, 0.5)
        bp_mask = outlier_bodypart(pred_data_array, num_keypoint//2)
        env_mask = outlier_enveloped(pred_data_array, 0.75)
        if self.canon_pose is None or self.angle_map is None:
            size_mask = pose_mask = no_mask
        else:
            size_mask = outlier_size(pred_data_array, self.canon_pose, 0.3, 2)
            pose_mask = outlier_pose(pred_data_array, self.angle_map, 1.5, 2)
        combined_mask = conf_mask | bp_mask | size_mask | pose_mask | env_mask
        corrected_data_array, _, _ = outlier_removal(pred_data_array, combined_mask)

        return corrected_data_array