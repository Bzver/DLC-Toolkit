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

        self.pred_data_array = self._pose_clean(pred_data_array, duplicate_only=True)
        self.corrected_pred_data = self._pose_clean(pred_data_array)

        self.total_frames, self.instance_count = self.pred_data_array.shape[:2]
        if self.progress:
            self.progress.setRange(0, self.total_frames)

        self.inst_list = list(range(self.instance_count))
        self.inst_array = np.array(self.inst_list)

        self.new_order_array = -np.ones((self.total_frames, self.instance_count), dtype=np.int8)
        self.current_frame_data = np.full_like(self.pred_data_array[0], np.nan)

        if DEBUG:
            clean_log()

    def track_correction(self, max_dist:float=10.0, lookback_window=10) -> Tuple[np.ndarray, int, List[int]]:
        try:
            gap_mask = np.all(np.isnan(self.pred_data_array), axis=(1,2))
            self.progress.setLabelText("Fixing tracks (centroid pass)…")
            self._centroids_tf_pass(max_dist, lookback_window)
            self.new_order_array[gap_mask] = self.inst_array
            if np.any(self.new_order_array == -1):
                self.progress.setLabelText("Fixing tracks (rotation pass)…")
                self._vector_tf_pass(max_dist, lookback_window)
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
            return self.pred_data_array, 0, []

        if self.progress:
            self.progress.close()

        diff = self.new_order_array - np.arange(self.instance_count)
        changed_frames = np.where(np.any(diff != 0, axis=1))[0].tolist()
        fixed_data_array = self.pred_data_array[np.arange(self.total_frames)[:, None], self.new_order_array]

        return fixed_data_array, changed_frames, amogus_frames

    def _centroids_tf_pass(self, max_dist:float, lookback_window):
        last_order = self.inst_list
        ref_centroids = np.full((self.instance_count, 2), np.nan)
        ref_last_updated = np.full((self.instance_count,), -2 * lookback_window)

        for frame_idx in range(self.total_frames):
            if self.progress:
                self.progress.setValue(frame_idx)
                if self.progress.wasCanceled():
                    raise RuntimeError("User cancelled track fixing operation.")
            
            log_print(f"---------- frame: {frame_idx} [CENTROID MODE] ---------- ", enabled=DEBUG)

            pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
            valid_pred_mask = np.all(~np.isnan(pred_centroids), axis=1)
            if not np.any(valid_pred_mask):
                self.new_order_array[frame_idx] = self.inst_array
                log_print("Skipping due to no prediction on current frame.")
                continue

            self.current_frame_data = self.corrected_pred_data[frame_idx]

            valid_ref_mask = ref_last_updated > frame_idx - lookback_window

            for i in range(self.instance_count):
                log_print(f"x,y in pred: inst {i}: ({pred_centroids[i,0]:.1f}, {pred_centroids[i,1]:.1f})", enabled=DEBUG)
            for i in range(self.instance_count):
                log_print(f"x,y in ref: inst {i}: ({ref_centroids[i,0]:.1f}, {ref_centroids[i,1]:.1f}) | "
                            f"last updated: {ref_last_updated[i]} | valid: {valid_ref_mask[i]}", enabled=DEBUG)

            if not np.any(valid_ref_mask):
                if np.all(np.isnan(ref_centroids)): # No ref at all, i.e. beginning of the vid
                    self.new_order_array[frame_idx] = np.array(last_order)
                else:
                    self.new_order_array[frame_idx] = -1 # Mark as ambiguous
                ref_centroids[valid_pred_mask] = pred_centroids[valid_pred_mask]
                ref_last_updated[valid_pred_mask] = frame_idx
                last_order = self.inst_list
                continue

            hun = Hungarian(pred_centroids, ref_centroids, valid_pred_mask, valid_ref_mask,
                            max_dist=max_dist, ref_frame_gap=frame_idx-ref_last_updated, debug_print=DEBUG)

            skip_matching = False
            if last_order != self.inst_list and hun.full_set: # Try last order first if possible
                if hun.compare_distance_improvement(last_order):
                    new_order = last_order
                    self.new_order_array[frame_idx] = np.array(last_order)
                    self._applying_last_order(last_order)
                    skip_matching = True
                    log_print(f"[TMOD] SWAP, reusing the last order.", enabled=DEBUG)

            if not skip_matching:
                new_order = hun.hungarian_matching()

                if new_order is None:
                    log_print(f"[TMOD] Failed to build new order with Hungarian.", enabled=DEBUG)
                    self.new_order_array[frame_idx] = -1 # Ambiguous
                elif new_order == self.inst_list:
                    last_order = new_order
                    self.new_order_array[frame_idx] = np.array(last_order)
                    log_print(f"[TMOD] NO SWAP, already the best solution.", enabled=DEBUG)
                else:
                    self.current_frame_data[:, :] = self.current_frame_data[new_order, :]
                    last_order = new_order
                    self.new_order_array[frame_idx] = np.array(last_order)
                    log_print(f"[TMOD] SWAP, new_order: {new_order}.", enabled=DEBUG)

            self.corrected_pred_data[frame_idx] = self.current_frame_data
            fixed_pred_centroids = pred_centroids[new_order] if new_order else pred_centroids
            fixed_pred_mask = valid_pred_mask[new_order] if new_order else valid_pred_mask
            
            ref_centroids[fixed_pred_mask] = fixed_pred_centroids[fixed_pred_mask]
            ref_last_updated[fixed_pred_mask] = frame_idx
                    
    def _vector_tf_pass(self, max_dist, lookback_window):
        if self.angle_map is None:
            log_print("[VEC] Skipping — no angle_map provided.", enabled=DEBUG)
            return
    
        pose_vectors = build_weighted_pose_vectors(self.corrected_pred_data, self.angle_map)
        if pose_vectors is None:
            log_print("[VEC] Skipping — fail to get pose_vectors.", enabled=DEBUG)
            return

        amogus = np.where(self.new_order_array[:, 0] == -1)[0]
        log_print(f"[VEC] Resolving {len(amogus)} ambiguous frames with rotation...", enabled=DEBUG)

        self.progress.setRange(0, len(amogus))

        for i, frame_idx in enumerate(amogus):
            if self.progress:
                self.progress.setValue(i)
                if self.progress.wasCanceled():
                    raise RuntimeError("User cancelled track fixing operation.")
            cur_vector = pose_vectors[frame_idx]
            if np.all(np.isnan(cur_vector)):
                self.new_order_array[frame_idx] = self.inst_array
                continue

            log_print(f"---------- frame: {frame_idx} [VECTOR MODE] ---------- ", enabled=DEBUG)

            ref_vector = np.full_like(cur_vector, np.nan)
            frame_gap = np.zeros((ref_vector.shape[0]))
            offset = 0
            while np.any(frame_gap==0) and offset <= lookback_window:
                offset += 1
                cand = frame_idx - offset
                if cand < 0:
                    break
                for inst in range(ref_vector.shape[0]):
                    cand_vector = pose_vectors[cand, inst]
                    if np.any(~np.isnan(ref_vector[inst])) or not np.any(~np.isnan(cand_vector)):
                        continue
                    ref_vector[inst] = cand_vector
                    frame_gap[inst] = offset
                    log_print(f"[VEC] Using pose from frame {cand} for inst {inst}.", enabled=DEBUG)

            if np.all(frame_gap==0):
                log_print(f"[VEC] No resolved reference for frame {frame_idx}. Fallback.", enabled=DEBUG)
                continue

            valid_pred = np.any(~np.isnan(cur_vector), axis=1)
            valid_ref  = np.any(~np.isnan(ref_vector), axis=1)

            hunvec = Transylvanian(
                cur_vector, ref_vector, valid_pred, valid_ref, max_dist=max_dist, ref_frame_gap=frame_gap, debug_print=DEBUG)
            
            new_order = hunvec.vector_hun_match()

            if new_order is None:
                continue

            self.current_frame_data[:, :] = self.current_frame_data[new_order, :]
            pose_vectors[frame_idx, :, :] = pose_vectors[frame_idx, new_order, :]
            self.new_order_array[frame_idx] = np.array(new_order)

    def _applying_last_order(self, last_order:List[int]):
        if last_order is None or last_order == self.inst_list:
            log_print(f"[TMOD]NO SWAP.", enabled=DEBUG)
            return

        self.current_frame_data[:, :] = self.current_frame_data[last_order, :]
        log_print(f"[TMOD] SWAP, Applying the last order: {last_order}", enabled=DEBUG)

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