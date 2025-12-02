import numpy as np

from PySide6.QtWidgets import QProgressDialog
from typing import List, Dict, Any, Tuple, Optional
import traceback

from .hungarian import Hungarian, Transylvanian
from utils.pose import (
    calculate_pose_centroids, outlier_removal, outlier_confidence, outlier_pose, 
    outlier_bodypart, outlier_duplicate, outlier_size, outlier_enveloped,
    )
from utils.helper import log_print, clean_log, build_weighted_pose_vectors

DEBUG = True

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

        self.corrected_pred_data = self._pose_clean(pred_data_array, duplicate_only=True)

        self.total_frames, self.instance_count = self.corrected_pred_data.shape[:2]
        if self.progress:
            self.progress.setRange(0, self.total_frames)

        self.inst_list = list(range(self.instance_count))
        self.inst_array = np.array(self.inst_list)

        self.new_order_array = -np.ones((self.total_frames, self.instance_count), dtype=np.int8)
        self.current_frame_data = np.full_like(self.corrected_pred_data[0], np.nan)

        if DEBUG:
            clean_log()

    def track_correction(self, max_dist:float=10.0, lookback_window=10) -> Tuple[np.ndarray, int, List[int]]:
        try:
            gap_mask = np.all(np.isnan(self.corrected_pred_data), axis=(1,2))
            self._centroids_tf_pass(max_dist, lookback_window)
            self.new_order_array[gap_mask] = self.inst_array
            if np.any(self.new_order_array == -1):
                amogus_frames = np.where(self.new_order_array[:,0]==-1)[0]
                self.new_order_array[amogus_frames] = self.inst_array
                amogus_frames = amogus_frames.tolist()
            else:
                amogus_frames = []
        except Exception as e:
            if self.progress:
                self.progress.close()
            print(f"Track Correction Error: {e}")
            traceback.print_exc()
            return self.corrected_pred_data, 0, []

        if self.progress:
            self.progress.close()

        diff = self.new_order_array - np.arange(self.instance_count)
        changed_frames = np.where(np.any(diff != 0, axis=1))[0].tolist()
        fixed_data_array = self.corrected_pred_data

        return fixed_data_array, changed_frames, amogus_frames

    def _centroids_tf_pass(self, max_dist:float, lookback_window):
        last_order = self.inst_list
        ref_centroids = np.full((self.instance_count, 2), np.nan)
        ref_last_updated = np.full((self.instance_count,), -2 * lookback_window)
        valid_ref_mask = np.zeros_like(ref_last_updated, dtype=bool)

        for frame_idx in range(self.total_frames):
            if self.progress:
                self.progress.setValue(frame_idx)
                if self.progress.wasCanceled():
                    raise RuntimeError("User cancelled track fixing operation.")
            
            log_print(f"---------- frame: {frame_idx} ---------- ", enabled=DEBUG)

            pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
            valid_pred_mask = np.all(~np.isnan(pred_centroids), axis=1)
            if not np.any(valid_pred_mask):
                self.new_order_array[frame_idx] = self.inst_array
                log_print("Skipping due to no prediction on current frame.")
                continue

            full_seek_window = min(frame_idx, 5, lookback_window) # Only look for very recent match
            offset = 0
            found_full_ref = False
            while offset <= full_seek_window: # Seek frames with all the ID if possible
                offset += 1
                cand = frame_idx - offset
                cand_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, cand)
                if np.all(~np.isnan(cand_centroids)):
                    ref_centroids_full = cand_centroids
                    valid_ref_mask_full = np.ones((cand_centroids.shape[0],), dtype=bool)
                    ref_last_updated_full = np.full_like(valid_ref_mask_full, cand, dtype=np.int16)
                    found_full_ref = True
                    log_print(f"Found full-reference frame: {cand}", enabled=DEBUG)
                    break

            valid_ref_mask = (ref_last_updated > frame_idx - lookback_window)

            if not np.any(valid_ref_mask):
                if np.all(np.isnan(ref_centroids)): # No ref at all, i.e. beginning of the vid
                    self.new_order_array[frame_idx] = np.array(last_order)
                else:
                    self.new_order_array[frame_idx] = -1 # Mark as ambiguous
                ref_centroids[valid_pred_mask] = pred_centroids[valid_pred_mask] # Rebase the prediction on current frame
                ref_last_updated[valid_pred_mask] = frame_idx
                last_order = self.inst_list
                continue

            hun_full = None
            if not found_full_ref:
                log_print("No full-reference frame found. Falling back to partial reference window.", enabled=DEBUG)
            else:
                if np.any(valid_ref_mask != valid_ref_mask_full):
                    hun_full = Hungarian(pred_centroids, ref_centroids_full, valid_pred_mask, valid_ref_mask_full,
                                max_dist=max_dist, ref_frame_gap=frame_idx-ref_last_updated_full, debug_print=DEBUG)

            for i in range(self.instance_count):
                log_print(f"x,y in pred: inst {i}: ({pred_centroids[i,0]:.1f}, {pred_centroids[i,1]:.1f})", enabled=DEBUG)
            for i in range(self.instance_count):
                log_print(f"x,y in ref: inst {i}: ({ref_centroids[i,0]:.1f}, {ref_centroids[i,1]:.1f}) | "
                            f"last updated: {ref_last_updated[i]} | valid: {valid_ref_mask[i]}", enabled=DEBUG)
            if hun_full:
                for i in range(self.instance_count):
                    log_print(f"x,y in ref (full): inst {i}: ({ref_centroids_full[i,0]:.1f}, {ref_centroids_full[i,1]:.1f}) | "
                                f"last full frame: {cand} | valid: {valid_ref_mask_full[i]}", enabled=DEBUG)

            hun = Hungarian(pred_centroids, ref_centroids, valid_pred_mask, valid_ref_mask,
                            max_dist=max_dist, ref_frame_gap=frame_idx-ref_last_updated, debug_print=DEBUG)

            skip_matching = False
            if last_order != self.inst_list and hun.full_set: # Try last order first if possible
                improved, avg_improv = hun.compare_distance_improvement(last_order, 0.2)
                if improved:
                    log_print(f"[TMOD] Last order improves overall distance by {avg_improv*100:.2f}% | Threshold: 20%",
                              enabled=DEBUG)
                    new_order = last_order
                    self.new_order_array[frame_idx] = np.array(last_order)
                    log_print(f"[TMOD] SWAP, Applying the last order: {last_order}", enabled=DEBUG)
                    skip_matching = True
                else:
                    log_print(f"[TMOD] Last order fails to sufficiently improve overall distance: {avg_improv*100:.2f}% | Threshold: 20%",
                              enabled=DEBUG)

            if not skip_matching:
                hun_order = hun.hungarian_matching()
                if not hun_order:
                    new_order = None
                elif hun_order == self.inst_list:
                    new_order = hun_order
                    if hun_full:
                        hun_f_order = hun_full.hungarian_matching()
                        if hun_f_order != hun_order:
                            log_print("Mismatch: Hungarian ≠ Full Frame Hungarian, rejecting assignment.", enabled=DEBUG)
                            new_order = None
                else:
                    improved, avg_improv = hun.compare_distance_improvement(hun_order)
                    if improved:
                        log_print(f"[TMOD] Hungarian order improves overall distance by {avg_improv*100:.2f}% | Threshold: 10%",
                                enabled=DEBUG)
                        new_order = hun_order
                    else:
                        log_print(f"[TMOD] Hungarian order fails to sufficiently improve overall distance: {avg_improv*100:.2f}% | Threshold: 10%",
                                enabled=DEBUG)
                        vec_order = self._vector_doublecheck(frame_idx, ref_last_updated, max_dist, lookback_window)
                        log_print(f"Vector double-check result: {vec_order}", enabled=DEBUG)
                        if hun_order != vec_order:
                            log_print("Mismatch: Hungarian ≠ Vector, rejecting assignment.", enabled=DEBUG)
                            new_order = None
                        else:
                            new_order = hun_order

                if new_order is None:
                    log_print("[TMOD] Failed to build new order.", enabled=DEBUG)
                    self.new_order_array[frame_idx] = -1 # Ambiguous
                else:
                    self.new_order_array[frame_idx] = np.array(new_order)
                    if new_order == self.inst_list:
                        log_print("[TMOD] NO SWAP, already the best solution.", enabled=DEBUG)
                    else:
                        log_print(f"[TMOD] SWAP, new_order: {new_order}.", enabled=DEBUG)

            if new_order:
                self.corrected_pred_data[frame_idx, :, :] = self.corrected_pred_data[frame_idx, new_order, :]
                last_order = new_order

            fixed_pred_centroids = pred_centroids[new_order] if new_order else pred_centroids
            fixed_pred_mask = valid_pred_mask[new_order] if new_order else valid_pred_mask
            
            ref_centroids[fixed_pred_mask] = fixed_pred_centroids[fixed_pred_mask]
            ref_last_updated[fixed_pred_mask] = frame_idx
                    
    def _vector_doublecheck(self, frame_idx, ref_last_updated, max_dist, lookback_window) -> Optional[list]:
        if self.angle_map is None:
            log_print("[VEC] Skipping — no angle_map provided.", enabled=DEBUG)
            return

        pose_vectors = build_weighted_pose_vectors(self.corrected_pred_data, self.angle_map)
        ref_vector = np.full_like(pose_vectors[0], np.nan)

        for inst in self.inst_list:
            ref_idx = ref_last_updated[inst]
            if ref_idx >= frame_idx - lookback_window:
                ref_vector[inst, :] = pose_vectors[ref_idx, inst, :]

        cur_vector = pose_vectors[frame_idx]

        valid_pred = np.any(~np.isnan(cur_vector), axis=1)
        valid_ref  = np.any(~np.isnan(ref_vector), axis=1)
        frame_gap = frame_idx - ref_last_updated

        hunvec = Transylvanian(
            cur_vector, ref_vector, valid_pred, valid_ref, max_dist=max_dist, ref_frame_gap=frame_gap, debug_print=DEBUG)
        
        new_order = hunvec.vector_hun_match()
        return new_order

    def _pose_clean(self, pred_data_array:np.ndarray, duplicate_only:bool=False) -> np.ndarray:
        dp_mask = outlier_duplicate(pred_data_array)
        if duplicate_only:
            corrected_data_array, _, _ = outlier_removal(pred_data_array, dp_mask)
            return corrected_data_array

        no_mask = np.zeros((pred_data_array.shape[0], pred_data_array.shape[1]), dtype=bool)
        conf_mask = outlier_confidence(pred_data_array, 0.4)
        bp_mask = outlier_bodypart(pred_data_array, 2)
        env_mask = outlier_enveloped(pred_data_array)
        if self.canon_pose is None or self.angle_map is None:
            size_mask = pose_mask = no_mask
        else:
            size_mask = outlier_size(pred_data_array, self.canon_pose, 0.3, 2.5)
            pose_mask = outlier_pose(pred_data_array, self.angle_map, 1.25, 2)
        combined_mask = conf_mask | bp_mask | dp_mask | size_mask | pose_mask | env_mask
        corrected_data_array, _, _ = outlier_removal(pred_data_array, combined_mask)

        return corrected_data_array