import numpy as np

from PySide6.QtWidgets import QProgressDialog
from typing import List, Dict, Any, Tuple, Optional

from .hungarian import Hungarian
from utils.pose import (
    calculate_pose_centroids, outlier_removal, outlier_confidence, outlier_pose, 
    outlier_bodypart, outlier_duplicate, outlier_size, outlier_enveloped,
    )
from utils.helper import log_print, clean_log

DEBUG = False

class Track_Fixer:
    def __init__(self,
                pred_data_array:np.ndarray,
                canon_pose:Optional[np.ndarray]=None,
                angle_map:Optional[Dict[str, Any]]=None,
                progress:Optional[QProgressDialog]=None,
                ):
        self.pred_data_array = pred_data_array
        self.canon_pose = canon_pose
        self.angle_map = angle_map
        self.progress = progress

        self.total_frames, self.instance_count = self.pred_data_array.shape[:2]
        self.inst_list = list(range(self.instance_count))
        self.changes_applied = 0
        self.corrected_pred_data = self._pose_cleaning()
        self.current_frame_data = np.full_like(self.pred_data_array[0], np.nan)

        if DEBUG:
            clean_log()

    def track_correction(self, max_dist:float=10.0, lookback_window=10) -> Tuple[np.ndarray, int]:
        last_order = self.inst_list
        debug_print = DEBUG

        ref_centroids = np.full((self.instance_count, 2), np.nan)
        ref_last_updated = np.full(self.instance_count, -2 * lookback_window)

        for frame_idx in range(self.total_frames):
            if self.progress:
                self.progress.setValue(frame_idx)
                if self.progress.wasCanceled():
                    self.progress.close()
                    return self.pred_data_array, 0
            
            log_print(f"---------- frame: {frame_idx} ---------- ", enabled=self.debug_print)

            pred_centroids, _ = calculate_pose_centroids(self.corrected_pred_data, frame_idx)
            valid_pred_mask = np.all(~np.isnan(pred_centroids), axis=1)
            self.current_frame_data = self.corrected_pred_data[frame_idx]

            for i in range(self.instance_count):
                log_print(f"x,y in pred: inst {i}: ({pred_centroids[i,0]:.1f}, {pred_centroids[i,1]:.1f})", enabled=self.debug_print)
                    
            valid_ref_mask = ref_last_updated > frame_idx - lookback_window

            for i in range(self.instance_count):
                    log_print(f"x,y in ref: inst {i}: ({ref_centroids[i,0]:.1f}, {ref_centroids[i,1]:.1f}) | "
                              f"last updated: {ref_last_updated[i]} | valid: {valid_ref_mask[i]}", enabled=self.debug_print)

            if not np.any(valid_ref_mask):
                ref_centroids[valid_pred_mask] = pred_centroids[valid_pred_mask]
                ref_last_updated[valid_pred_mask] = frame_idx
                continue

            hun = Hungarian(pred_centroids, ref_centroids, valid_pred_mask, valid_ref_mask, max_dist, debug_print)

            skip_matching = False
            if last_order != self.inst_list and hun.full_set: # Try last order first if possible
                if hun.compare_distance_improvement(last_order):
                    new_order = last_order
                    self._applying_last_order(last_order)
                    skip_matching = True
                    log_print(f"[TMOD] SWAP, reusing the last order.", enabled=self.debug_print)

            if not skip_matching:
                new_order = hun.hungarian_matching()

                if new_order is None:
                    log_print(f"[TMOD] Failed to build new order with Hungarian.", enabled=self.debug_print)
                    self._applying_last_order(last_order)
                elif new_order == self.inst_list:
                    last_order = new_order
                    log_print(f"[TMOD] NO SWAP, already the best solution.", enabled=self.debug_print)
                else:
                    self.current_frame_data[:, :] = self.current_frame_data[new_order, :]
                    last_order = new_order
                    self.changes_applied += 1
                    log_print(f"[TMOD] SWAP, new_order: {new_order}.", enabled=self.debug_print)

            self.corrected_pred_data[frame_idx] = self.current_frame_data
            fixed_pred_centroids = pred_centroids[new_order] if new_order else pred_centroids
            fixed_pred_mask = valid_pred_mask[new_order] if new_order else valid_pred_mask
            
            ref_centroids[fixed_pred_mask] = fixed_pred_centroids[fixed_pred_mask]
            ref_last_updated[fixed_pred_mask] = frame_idx
                    
        if self.progress:
            self.progress.close()

        return self.corrected_pred_data, self.changes_applied

    def _pose_cleaning(self) -> np.ndarray:
        no_mask = np.zeros((self.total_frames, self.instance_count), dtype=bool)
        conf_mask = outlier_confidence(self.pred_data_array, 0.4)
        bp_mask = outlier_bodypart(self.pred_data_array, 2)
        dp_mask = outlier_duplicate(self.pred_data_array)
        env_mask = outlier_enveloped(self.pred_data_array)
        if self.canon_pose is None or self.angle_map is None:
            size_mask = pose_mask = no_mask
        else:
            size_mask = outlier_size(self.pred_data_array, self.canon_pose, 0.5, 2.5)
            pose_mask = outlier_pose(self.pred_data_array, self.angle_map, 1.0, 2)
        combined_mask = conf_mask | bp_mask | dp_mask | size_mask | pose_mask | env_mask

        corrected_data_array, _, _ = outlier_removal(self.pred_data_array, combined_mask)
        return corrected_data_array

    def _applying_last_order(self, last_order:List[int]):
        if last_order is None or last_order == self.inst_list:
            log_print(f"[TMOD]NO SWAP.", enabled=DEBUG)
            return

        self.current_frame_data[:, :] = self.current_frame_data[last_order, :]
        self.changes_applied += 1
        log_print(f"[TMOD] SWAP, Applying the last order: {last_order}", enabled=DEBUG)