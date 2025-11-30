import numpy as np
import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtWidgets import QMessageBox, QDialog

from typing import Callable, Optional

from utils.track import (
    Track_Fixer, interpolate_track_all, delete_track, swap_track, interpolate_track,
    )
from utils.helper import get_instances_on_current_frame, get_instance_count_per_frame
from utils.pose import (
    rotate_selected_inst, generate_missing_inst, generate_missing_kp_for_inst,
    calculate_pose_centroids, calculate_pose_rotations
)
from ui import Progress_Indicator_Dialog, Selectable_Instance, Track_Fix_Dialog
from core.tool import Uno_Stack, Parallel_Review_Dialog

DEBUG = False

class Keypoint_Edit_Manager:
    def __init__(self, track_edited_callback:Callable[[], None], parent=None):
        self.main = parent
        self.track_edited_callback = track_edited_callback
        self.reset_kem()

    def reset_kem(self):
        self.total_frames = 0
        self.pred_data_array = None
        self.last_selected_idx = None
        self.current_prediction_file = None
        self.uno = Uno_Stack()

    def set_pred_data(self, pred_data_array:np.ndarray):
        self.pred_data_array = pred_data_array
        self.total_frames = self.pred_data_array.shape[0]
        self.uno.save_state_for_undo(self.pred_data_array)

    def get_current_frame_data(self, frame_idx):
        return self.pred_data_array[frame_idx, :, :].copy()

    def check_pred_data(self):
        if self.pred_data_array is not None:
            return True
        else:
            QMessageBox.warning(self.main, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return False

    ##############################################################################################

    def correct_track_using_temporal(self, dlc_data, extractor, canon_pose, angle_map_data):
        centroids = np.stack([calculate_pose_centroids(self.pred_data_array, idx)[0] for idx in range(self.total_frames)])
        speeds_px_frame = np.linalg.norm(np.diff(centroids, axis=0), axis=2).flatten()
        speeds_px_frame = speeds_px_frame[~np.isnan(speeds_px_frame)]

        dialog = Track_Fix_Dialog(self.main)
        dialog.set_histogram(speeds_px_frame, max_dist_px_frame=10.0)
        if dialog.exec() == QDialog.Accepted:
            max_dist, lookback = dialog.get_values()
        else:
            return

        self._save_state_for_undo()

        progress = Progress_Indicator_Dialog(
            0, self.total_frames, "Temporal Track Fixing", "Fixing track using temporal consistency...", self.main)

        tf = Track_Fixer(self.pred_data_array, canon_pose, angle_map_data, progress)
        self.pred_data_array, changed_frames, amongus_frames = tf.track_correction(max_dist, lookback)

        if not changed_frames:
            QMessageBox.information(self.main, "No Changes Applied", "No changes were applied.")
            return

        if amongus_frames:
            reply = QMessageBox.question(
                        self.main, f"Ambiguous Frames", f"{len(amongus_frames)} frames are ambiguous, start manual correction now?",
                        QMessageBox.Yes | QMessageBox.No,
                    )
            if reply == QMessageBox.Yes:
                dialog = Parallel_Review_Dialog(dlc_data, extractor,  self.pred_data_array, [], (changed_frames, amongus_frames), True, parent=self.main)
                dialog.pred_data_exported.connect(self._get_pred_data_from_manual_review)
                dialog.exec()
                return
            
        msg = f"Applied {len(changed_frames)} changes to track."
        QMessageBox.information(self.main, "Track Correction Finished", msg)
        self.track_edited_callback()

    def _get_pred_data_from_manual_review(self, pred_data_array):
        self.pred_data_array = pred_data_array
        self.track_edited_callback()

    ###################################################################################################################################################  

    def update_kp_pos(self, frame_idx, instance_id, keypoint_id, new_x, new_y):
        self._save_state_for_undo()

        current_conf = self.pred_data_array[frame_idx, instance_id, keypoint_id*3+2]
        self.pred_data_array[frame_idx, instance_id, keypoint_id*3] += new_x
        self.pred_data_array[frame_idx, instance_id, keypoint_id*3+1] += new_y
        if pd.isna(current_conf) and not (pd.isna(new_x) or pd.isna(new_y)):
            self.pred_data_array[frame_idx, instance_id, keypoint_id*3+2] = 1.0

    def update_inst_pos(self, frame_idx, instance_id, dx, dy):
        self._save_state_for_undo()
        
        for kp_idx in range(self.pred_data_array.shape[2]//3): # Update all keypoints for the given instance in the current frame
            x_coord_idx, y_coord_idx = kp_idx * 3, kp_idx * 3 + 1
            current_x = self.pred_data_array[frame_idx, instance_id, x_coord_idx]
            current_y = self.pred_data_array[frame_idx, instance_id, y_coord_idx]

            if not pd.isna(current_x) and not pd.isna(current_y):
                self.pred_data_array[frame_idx, instance_id, x_coord_idx] = current_x + dx
                self.pred_data_array[frame_idx, instance_id, y_coord_idx] = current_y + dy

    def del_kp(self, frame_idx, instance_id, keypoint_id):
        self._save_state_for_undo()
        self.pred_data_array[frame_idx, instance_id, keypoint_id*3:keypoint_id*3+3] = np.nan

    def rot_inst(self, frame_idx, instance_idx, angle_delta):
        self.pred_data_array = rotate_selected_inst(self.pred_data_array, frame_idx, instance_idx, angle_delta)
        self.track_edited_callback()

    def del_outlier(self, outlier_mask:np.ndarray):
        self._save_state_for_undo()
        self.pred_data_array[outlier_mask] = np.nan
        self.track_edited_callback()

    def update_roi(self) -> list:
        self.inst_count_per_frame_pred = get_instance_count_per_frame(self.pred_data_array)
        return list(np.where(np.diff(self.inst_count_per_frame_pred)!=0)[0]+1)

    ###################################################################################################################################################  
    
    def del_trk(self, frame_idx:int, selected_instance:Selectable_Instance):
        if not self.check_pred_data():
            return
        selected_instance_idx = self._instance_multi_select(frame_idx, selected_instance)
        if selected_instance_idx is None:
            return
        self._save_state_for_undo()
        try:
            self.pred_data_array = delete_track(self.pred_data_array, frame_idx, selected_instance_idx)
        except ValueError as e:
            QMessageBox.warning(self.main, "Deletion Error", str(e))
            return
        self.track_edited_callback()
        
    def swp_trk(self, frame_idx, swap_range=None):
        if not self.check_pred_data():
            return
        self._save_state_for_undo()
        try:
            self.pred_data_array = swap_track(self.pred_data_array, frame_idx, swap_range)
        except ValueError as e:
            QMessageBox.warning(self.main, "Swap Error", str(e))
            return
        self.track_edited_callback()

    def intp_trk(self, frame_idx:int, selected_instance:Optional[Selectable_Instance]):
        if not self.check_pred_data():
            return
        selected_instance_idx = self._instance_multi_select(frame_idx, selected_instance)
        if selected_instance_idx is None:
            return
        self._save_state_for_undo()
        iter_frame_idx = frame_idx + 1
        frames_to_interpolate = []
        while np.all(np.isnan(self.pred_data_array[iter_frame_idx, selected_instance_idx, :])):
            frames_to_interpolate.append(iter_frame_idx)
            iter_frame_idx += 1
            if iter_frame_idx >= self.total_frames:
                QMessageBox.information(self.main, "Interpolation Failed", "No valid subsequent keypoint data found for this instance to interpolate to.")
                return

        if not frames_to_interpolate:
            QMessageBox.information(self.main, "Interpolation Info", "No gaps found to interpolate for the selected instance.")
            return
        
        frames_to_interpolate.sort()
        self.pred_data_array = interpolate_track(self.pred_data_array, frames_to_interpolate, selected_instance_idx)
        self.track_edited_callback()

    def gen_inst(self, frame_idx:int, instance_count:int, angle_map_data, canon_pose):
        if not self.check_pred_data():
            return
        self._save_state_for_undo()

        current_frame_inst = get_instances_on_current_frame(self.pred_data_array, frame_idx)
        missing_instances = [inst for inst in range(instance_count) if inst not in current_frame_inst]
        if missing_instances is None:
            QMessageBox.information(self.main, "No Missing Instances", "No missing instances found in the current frame to fill.")
            return

        self.pred_data_array = generate_missing_inst(
            self.pred_data_array, frame_idx, missing_instances, angle_map_data=angle_map_data, canon_pose=canon_pose)
        self.track_edited_callback()

    def rot_inst_prep(self, frame_idx:int, selected_instance:Optional[Selectable_Instance], angle_map_data):
        selected_instance_idx = self._instance_multi_select(frame_idx, selected_instance)
        if selected_instance_idx is None:
            return None, None
        _, local_coords = calculate_pose_centroids(self.pred_data_array, frame_idx)
        local_x = local_coords[selected_instance_idx, 0::2]
        local_y = local_coords[selected_instance_idx, 1::2]
        current_rotation = np.degrees(calculate_pose_rotations(local_x, local_y, angle_map_data))
        if np.isnan(current_rotation) or np.isinf(current_rotation):
            current_rotation = 0.0
        else:
            current_rotation = current_rotation % 360.0 
        self._save_state_for_undo()
        return selected_instance_idx, current_rotation

    def intp_ms_kp(
            self, frame_idx:int, selected_instance:Optional[Selectable_Instance], angle_map_data, canon_pose):
        if not self.check_pred_data():
            return
        selected_instance_idx = self._instance_multi_select(frame_idx, selected_instance)
        if selected_instance_idx is None:
            return
        self._save_state_for_undo()
        self.pred_data_array = generate_missing_kp_for_inst(
            self.pred_data_array,
            frame_idx,
            selected_instance_idx,
            angle_map_data,
            canon_pose)
        self.track_edited_callback()

    def interpolate_all_for_inst(self, selected_instance:Optional[Selectable_Instance]):
        if selected_instance is None:
            QMessageBox.information(self.main, "No Instance Selected", "Please select a track to interpolate all frames for one instance.")
            return False
        max_gap_allowed, ok = QtWidgets.QInputDialog.getInt(
            self.main,"Set Max Gap For Interpolation","Will not interpolate gap beyond this limit, set to 0 to ignore the limit.",
            value=10, minValue=0, maxValue=1000
        )
        if not ok:
            QMessageBox.information(self.main, "Input Cancelled", "Max Gap input cancelled.")
            return False

        self._save_state_for_undo()
        self.pred_data_array = interpolate_track_all(self.pred_data_array, selected_instance.instance_id, max_gap_allowed)
        self.track_edited_callback()
        return True

    def _instance_multi_select(self, frame_idx:int, selected_instance:Optional[Selectable_Instance]) -> Optional[int]:
        current_frame_inst = get_instances_on_current_frame(self.pred_data_array, frame_idx)
        if current_frame_inst is None:
            return
        if len(current_frame_inst) == 1:
            return current_frame_inst[0]
        if selected_instance is None:
            if self.last_selected_idx is None:
                QMessageBox.information(self.main, "No Instance Seleted",
                    "When there are more than one instance present, "
                    "you need to click one of the instance bounding box to specify which to delete.")
                return
            else:
                return self.last_selected_idx
        return selected_instance.instance_id

    ###################################################################################################################################################  

    def undo(self):
        data_array = self.uno.undo()
        if data_array is not None:
            self.pred_data_array = data_array

    def redo(self):
        data_array = self.uno.redo()
        if data_array is not None:
            self.pred_data_array = data_array
    
    def _save_state_for_undo(self):
        self.uno.save_state_for_undo(self.pred_data_array)