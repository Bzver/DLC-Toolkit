import os
import numpy as np

from PySide6.QtWidgets import QMessageBox, QFileDialog

from typing import Callable, Literal, List, Optional

from utils.helper import infer_head_tail_indices
from utils.pose import calculate_canonical_pose
from .palette import NAV_COLOR_PALETTE as nvp, NAV_COLOR_PALETTE_COUNTING as nvpc
from .io import (
    Prediction_Loader, Exporter,
    remove_confidence_score, append_new_video_to_dlc_config
)
from .dataclass import Plot_Config, Loaded_DLC_Data, Export_Settings

class Data_Manager:
    HexColor = str

    def __init__(self, refresh_callback:Callable[[], None], parent=None):
        self.main = parent
        self.refresh_callback = refresh_callback

        self.reset_dm_vars()

    def reset_dm_vars(self):
        self.total_frames = 0
        self.current_frame_idx = 0

        self.video_file, self.video_name, self.project_dir = None, None, None
        self.dlc_data, self.canon_pose = None, None

        self.labeled_frame_list, self.frame_list = [], []
        self.refined_frame_list, self.approved_frame_list, self.rejected_frame_list = [], [], []
        self.animal_0_list, self.animal_1_list, self.animal_n_list = [], [], []

        self.label_data_array = None

        self.plot_config = Plot_Config(
            plot_opacity =1.0, point_size = 6.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False)
        self.blob_config = None

    def update_video_path(self, video_path:str):
        self.video_file = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_file))[0]

    def pred_file_dialog(self):
        file_dialog = QFileDialog(self.main)
        prediction_path, _ = file_dialog.getOpenFileName(self.main, "Select Prediction", "", "HDF5 Files (*.h5);;All Files (*)")
        if not prediction_path:
            return

        prediction_filename = str(os.path.basename(prediction_path))
        if prediction_filename.startswith("CollectedData_"):
            is_label_data = True
            QMessageBox.information(self.main, "Labeled Data Selected", "Labeled data selected, now loading DLC config.")
        else:
            is_label_data = False
            QMessageBox.information(self.main, "Prediction Selected", "Prediction selected, now loading DLC config.")

        dlc_config = self.config_file_dialog()
        if dlc_config:
            self.load_pred_to_dm(dlc_config, prediction_path, is_label_data)

    def config_file_dialog(self) -> Optional[str]:
        file_dialog = QFileDialog(self.main)
        dlc_config, _ = file_dialog.getOpenFileName(self.main, "Select DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        return dlc_config

    def load_pred_to_dm(self, dlc_config:Loaded_DLC_Data, prediction_path:str, is_label_data:bool=False):
        data_loader = Prediction_Loader(dlc_config, prediction_path)

        if is_label_data:
            if not self.dlc_data:
                try:
                    self.dlc_data = data_loader.load_data(metadata_only=True)
                except Exception as e:
                    QMessageBox.critical(self.main, "Error Loading Prediction",
                                         f"Unexpected error during prediction loading: {e}.")
                
                self.dlc_data.pred_data_array = np.full(
                    (self.total_frames, self.dlc_data.instance_count, self.dlc_data.num_keypoint*3),
                    np.nan)
                
                self.dlc_data.pred_frame_count = self.total_frames
                self.dlc_data.prediction_filepath = prediction_path

            self.process_labeled_frame(prediction_path)
        else:
            try:
                self.dlc_data = data_loader.load_data()
            except Exception as e:
                QMessageBox.critical(self.main, "Error Loading Prediction",
                                     f"Unexpected error during prediction loading: {e}.")
            self.process_labeled_frame()

        self._init_canon_pose()

    def load_metadata_to_dm(self, dlc_config:Loaded_DLC_Data):
        data_loader = Prediction_Loader(dlc_config)
        try:
            self.dlc_data = data_loader.load_data(metadata_only=True)
            self.dlc_data.pred_frame_count = self.total_frames
        except Exception as e:
            QMessageBox.critical(self.main, "Error Loading DLC Config",
                                    f"Unexpected error during DLC Config loading: {e}.")

    def reload_pred_to_dm(self, prediction_path:str):
        data_loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, prediction_path)
        self.dlc_data = data_loader.load_data()
        self.approved_frame_list[:] = list(set(self.approved_frame_list) - set(self.refined_frame_list))
        self.rejected_frame_list[:] = list(set(self.rejected_frame_list) - set(self.refined_frame_list))

    def process_labeled_frame(self, lbf:str=None):
        if lbf is None:
            dlc_dir = os.path.dirname(self.dlc_data.dlc_config_filepath)
            self.project_dir = os.path.join(dlc_dir, "labeled-data", self.video_name)

            if not os.path.isdir(self.project_dir):
                return
        
            scorer = self.dlc_data.scorer
            label_data_filename = f"CollectedData_{scorer}.h5"
            label_data_filepath = os.path.join(self.project_dir, label_data_filename)
        else:
            label_data_filepath = lbf
            self.project_dir = os.path.dirname(lbf)

        self.label_data_array = np.full_like(self.dlc_data.pred_data_array, np.nan)

        data_loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, label_data_filepath)
        label_data = data_loader.load_data()
        label_array = label_data.pred_data_array
        self.label_data_array[range(label_array.shape[0])] = label_array

        self.labeled_frame_list = np.where(
            np.any(~np.isnan(self.label_data_array), axis=(1, 2))
            )[0].tolist()

        self.frame_list = list(set(self.frame_list) - set(self.labeled_frame_list))
        self.refined_frame_list = list(set(self.refined_frame_list) - set(self.labeled_frame_list))
        self.approved_frame_list = list(set(self.approved_frame_list) - set(self.labeled_frame_list))
        self.rejected_frame_list = list(set(self.rejected_frame_list) - set(self.labeled_frame_list))
        self.frame_list.sort()
        self.refresh_callback()

    ###################################################################################################################################################

    def toggle_frame_status(self):
        if self.current_frame_idx in self.labeled_frame_list:
            QMessageBox.information(self.main, "Already Labeled", "The frame is already in the labeled dataset, skipping...")
            return

        if self.current_frame_idx in self.refined_frame_list:
            reply = QMessageBox.question(
                self.main, "Confirm Unmarking",
                "This frame is already refined, do you still want to remove it from the exported lists?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.refined_frame_list.remove(self.current_frame_idx)
                self.frame_list.remove(self.current_frame_idx)
            return

        if not self.current_frame_idx in self.frame_list:
            self.frame_list.append(self.current_frame_idx)
        else: # Remove the mark status if already marked
            self.frame_list.remove(self.current_frame_idx)
            if self.current_frame_idx in self.approved_frame_list:
                self.approved_frame_list.remove(self.current_frame_idx)
            if self.current_frame_idx in self.rejected_frame_list:
                self.rejected_frame_list.remove(self.current_frame_idx)

        self.refresh_callback()

    def get_frame_cat(self) -> List[str]:
        frame_set = set(self.frame_list)
        refined_set = set(self.refined_frame_list)
        approved_set = set(self.approved_frame_list)
        rejected_set = set(self.rejected_frame_list)
        marked_set = refined_set | approved_set | rejected_set
        
        frame_options = {
            "All Marked Frames": self.frame_list,
            "Refined Frames": self.refined_frame_list,
            "Approved Frames": self.approved_frame_list,
            "Rejected Frames": self.rejected_frame_list,
        }

        frame_categories = [label for label, frame_list in frame_options.items() if frame_list]
        marked_set = set(self.refined_frame_list) | set(self.approved_frame_list) | set(self.rejected_frame_list)

        if refined_set:
            all_except_refined = frame_set - refined_set
            if all_except_refined:
                frame_categories.append("All Marked Frames (Except For Refined)")

        if marked_set:
            remaining_frames = frame_set - marked_set
            if remaining_frames:
                frame_categories.append("Remaining Frames")

        return frame_categories
    
    def clear_frame_cat(self, 
            frame_category:Literal[
                "All Marked Frames",
                "Refined Frames",
                "Approved Frames",
                "Rejected Frames",
                "Remaining Frames",
                "All Marked Frames (Except for Refined)"
            ]
            ):
        actions = {
            "All Marked Frames": lambda: (
                self.frame_list.clear(),
                self.refined_frame_list.clear(),
                self.approved_frame_list.clear(),
                self.rejected_frame_list.clear()
            ),
            "Refined Frames": self.refined_frame_list.clear,
            "Approved Frames": self.approved_frame_list.clear,
            "Rejected Frames": self.rejected_frame_list.clear,
        }

        if frame_category in actions:
            actions[frame_category]()
        elif frame_category == "Remaining Frames":
            kept_frames = set(self.refined_frame_list) | set(self.approved_frame_list) | set(self.rejected_frame_list)
            self.frame_list[:] = list(kept_frames)
        elif frame_category == "All Marked Frames (Except for Refined)":
            self.frame_list[:] = self.refined_frame_list.copy()
            self.approved_frame_list.clear()
            self.rejected_frame_list.clear()

        self.refresh_callback()

    def clear_old_cat(self, clear_old:bool):
        if not clear_old:
            return
        if not self.refined_frame_list:
            self.clear_frame_cat("All Marked Frames")
        else:
            self.clear_frame_cat("All Marked Frames (Except for Refined)")

    def get_inference_list(self) -> List[int]:
        return list(set(self.frame_list) - set(self.approved_frame_list) - set(self.rejected_frame_list) - set(self.refined_frame_list))

    ###################################################################################################################################################

    def determine_nav_color_fview(self) -> HexColor:
        frame_lists = [
            self.frame_list,
            self.rejected_frame_list,
            self.approved_frame_list,
            self.refined_frame_list,
            self.labeled_frame_list
        ]
        return nvp[self._get_max_priority(frame_lists, range(1, 6))]

    def determine_nav_color_counting(self) -> HexColor:
        frame_lists = [
            self.animal_0_list,
            self.animal_1_list,
            self.animal_n_list
        ]
        return nvpc[self._get_max_priority(frame_lists, range(1, 4))]

    def _get_max_priority(self, frame_lists:List[List[int]], priorities:range) -> int:
        """Get the highest priority for current frame."""
        color_code = 0
        for frame_list, priority in zip(frame_lists, priorities):
            if self.current_frame_idx in frame_list:
                color_code = max(color_code, priority)
        return color_code

    ###################################################################################################################################################
        
    def _init_canon_pose(self):
        head_idx, tail_idx = infer_head_tail_indices(self.dlc_data.keypoints)
        if head_idx is None or tail_idx is None:
            return
        if np.all(np.isnan(self.dlc_data.pred_data_array)) and self.label_data_array is None:
            return

        if not np.all(np.isnan(self.dlc_data.pred_data_array)):
            canon_pred_to_use = self.dlc_data.pred_data_array
        else:
            canon_pred_to_use = self.label_data_array
        
        self.canon_pose, _ = calculate_canonical_pose(canon_pred_to_use, head_idx, tail_idx)

    ###################################################################################################################################################
        
    def save_to_dlc(self):
        dlc_dir = os.path.dirname(self.dlc_data.dlc_config_filepath)
        exp_set = Export_Settings(video_filepath=self.video_file, video_name=self.video_name,
                                  save_path=self.project_dir, export_mode="Append")

        if not self.project_dir:
            exp_set.save_path = os.path.join(dlc_dir, "labeled-data", self.video_name)
            os.makedirs(exp_set.save_path, exist_ok=True)

        if not self.refined_frame_list:
            exporter = Exporter(self.dlc_data, exp_set, self.frame_list)
        else:
            exp_set.export_mode = "Merge"
            pred_data_array_for_export = remove_confidence_score(self.dlc_data.pred_data_array)
            exporter = Exporter(self.dlc_data, exp_set, self.refined_frame_list, pred_data_array_for_export)
        
        if self.dlc_data:
            try:
                exporter.export_data_to_DLC()
                QMessageBox.information(self.main, "Success", "Successfully exported frames and prediction to DLC.")
            except Exception as e:
                QMessageBox.critical(self.main, "Error Save Data", f"Error saving data to DLC: {e}")
            append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, self.video_name)

            if exp_set.export_mode == "Merge":
                self.process_labeled_frame()
        else:
            reply = QMessageBox.question(
                self.main,
                "No Prediction Loaded",
                "No prodiction has been loaded. Would you like export frames only?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.save_to_dlc_frame_only(exporter)
            else:
                self.pred_file_dialog()

    def merge_data(self):
        if not self.refined_frame_list:
            QMessageBox.warning(self.main, "No Refined Frame", "No frame has been refined, please refine some marked frames first.")
            return
        if not self.labeled_frame_list:
            self.save_to_dlc()
            return

        reply = QMessageBox.question(
            self.main, "Confirm Merge",
            "This action will merge the selected data into the labeled dataset. "
            "Please ensure you have reviewed and refined the predictions on the marked frames.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            exp_set = Export_Settings(video_filepath=self.video_file, video_name=self.video_name,
                                      save_path=self.project_dir, export_mode="Merge")

            self.label_data_array[self.refined_frame_list, :, :] = self.dlc_data.pred_data_array[self.refined_frame_list, :, :]
            merge_frame_list = list(set(self.labeled_frame_list) | set(self.refined_frame_list))
            label_data_array_export = remove_confidence_score(self.label_data_array)

            exporter = Exporter(dlc_data=self.dlc_data, export_settings=exp_set,
                                frame_list=merge_frame_list, pred_data_array=label_data_array_export)
            
            try:
                exporter.export_data_to_DLC()
                QMessageBox.information(self, "Success", "Successfully exported frames and prediction to DLC.")
            except Exception as e:
                QMessageBox.critical(self, "Error Merge Data", f"Error merging data to DLC: {e}")

            self.process_labeled_frame()

    def save_to_dlc_frame_only(self, exporter:Exporter):
        QMessageBox.information(self.main, "Frame Only Mode", "Choose the directory of DLC project.")
        dlc_dir = QFileDialog.getExistingDirectory(
                    self.main, "Select Project Folder",
                    os.path.dirname(self.video_file),
                    QFileDialog.ShowDirsOnly
                )
        if not dlc_dir: # When user close the file selection window
            return
        try:
            exporter.export_data_to_DLC(frame_only=True)
            QMessageBox.information(self.main, "Success", "Successfully exported marked frames to DLC for labeling!")
        except Exception as e:
            QMessageBox.critical(self.main, "Error Export Frames", f"Error exporting marked frames to DLC: {e}")
