import os
import pickle
import yaml
import json
import numpy as np

from PySide6.QtWidgets import QMessageBox, QFileDialog, QDialog

from typing import Callable, Literal, List, Optional
import traceback

from utils.helper import infer_head_tail_indices, build_angle_map
from utils.pose import calculate_canonical_pose
from .palette import (
    NAV_COLOR_PALETTE as nvp, NAV_COLOR_PALETTE_COUNTING as nvpc,
    NAV_COLOR_PALETTE_FLAB as nvpl)
from .io import (
    Prediction_Loader, Exporter,
    remove_confidence_score, append_new_video_to_dlc_config, determine_save_path,
    backup_existing_prediction, save_prediction_to_existing_h5, prediction_to_csv
)
from ui import Head_Tail_Dialog
from .dataclass import Plot_Config, Export_Settings

class Data_Manager:
    HexColor = str

    def __init__(self, 
                init_vid_callback:Callable[[str], None],
                refresh_callback:Callable[[], None],
                parent=None):
        self.main = parent
        self.init_vid_callback = init_vid_callback
        self.refresh_callback = refresh_callback
        self.reset_dm()

    def reset_dm(self):
        self.total_frames, self.current_frame_idx  = 0, 0

        self.video_file, self.video_name, self.project_dir = None, None, None
        self.dlc_data, self.canon_pose = None, None

        self.refined_frame_list, self.frame_list = [], []
        self.plot_config = Plot_Config(
            plot_opacity =1.0, point_size = 6.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False,
            plot_labeled = True, plot_pred = True, navigate_labeled = False, auto_snapping = False, navigate_roi = False)
        
        # fview only
        self.blob_config = None
        self.labeled_frame_list, self.approved_frame_list, self.rejected_frame_list = [], [], []
        self.animal_0_list, self.animal_1_list, self.animal_n_list, self.blob_merged_list = [], [], [], []
        self.label_data_array, self.blob_array = None, None

        # flabel only
        self.prediction = None  # To track modified prediction file
        self.angle_map_data = None
        self.inst_count_per_frame_pred = None

        self.roi_frame_list, self.outlier_frame_list = [], []

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
            if self.dlc_data:
                self._process_labeled_frame()
            else:
                self.load_dlc_label()   
        else:
            QMessageBox.information(self.main, "Prediction Selected", "Prediction selected, now loading DLC config.")
            dlc_config = self.config_file_dialog()
            if dlc_config:
                self.load_pred_to_dm(dlc_config, prediction_path)
        return True

    def config_file_dialog(self) -> Optional[str]:
        file_dialog = QFileDialog(self.main)
        dlc_config_path, _ = file_dialog.getOpenFileName(self.main, "Select DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        return dlc_config_path

    def load_pred_to_dm(self, dlc_config_path:str, prediction_path:str):
        data_loader = Prediction_Loader(dlc_config_path, prediction_path)           

        try:
            self.dlc_data = data_loader.load_data()
        except Exception as e:
            QMessageBox.critical(self.main, "Error Loading Prediction",
                                    f"Unexpected error during prediction loading: {e}.")

        self._init_loaded_data()

    def load_metadata_to_dm(self, dlc_config_path:str):
        data_loader = Prediction_Loader(dlc_config_path)
        try:
            self.dlc_data = data_loader.load_data(metadata_only=True)
            self.dlc_data.pred_frame_count = self.total_frames
        except Exception as e:
            QMessageBox.critical(self.main, "Error Loading DLC Config",
                                    f"Unexpected error during DLC Config loading: {e}.")

    def load_dlc_label(self, image_folder:str, prediction_path:Optional[str]=None):
        """Load DLC Label without a preexisting prediction"""
        if not prediction_path:
            h5_candidates = [f for f in os.listdir(image_folder) if f.startswith("CollectedData_") and f.endswith(".h5")]
            if not h5_candidates:
                QMessageBox.warning(self.main, "No H5 File", "No 'CollectedData_*.h5' file found in the selected folder.")
                return
            prediction_path = os.path.join(image_folder, h5_candidates[0])

        self.prediction = prediction_path
        dlc_dir = os.path.dirname(os.path.dirname(image_folder))
        dlc_config = os.path.join(dlc_dir, "config.yaml")

        # Set video file to folder path (for naming)
        self.video_file = image_folder
        self.video_name = os.path.basename(image_folder)

        data_loader = Prediction_Loader(dlc_config, prediction_path)
        try:
            self.dlc_data = data_loader.load_data(force_load_pred=True)
        except Exception as e:
            QMessageBox.critical(self.main, "Error Loading Prediction", f"Failed to load prediction: {e}")
            return
      
        self._init_loaded_data()
        self.refresh_callback()

    def _init_loaded_data(self):
        self.prediction = self.dlc_data.prediction_filepath
        self._process_labeled_frame()
        self._init_canon_pose()

    def auto_loader(self):
        """Automaticaly load coreesponding prediction file and config when there has not been one"""
        if self.prediction:
           return None, None
        video_folder = os.path.dirname(self.video_file)
        pred_candidates = []
        for f in os.listdir(video_folder):
            if f.endswith(".h5") and self.video_name in f:
                full_path = os.path.join(video_folder, f)
                pred_candidates.append(full_path)
        if not pred_candidates:
            return None, None
        newest_pred = max(pred_candidates, key=os.path.getmtime)
        print(f"Automatically fetched the newest prediction: {newest_pred}")

        dlc_sub_folders = ["dlc-models-pytorch", "evaluation-results-pytorch", "labeled-data", "training-datasets", "videos"]
        found = False
        for fn in dlc_sub_folders:
            if fn in self.video_file:
                found = True
                break
        if not found:
            return None, None
        dlc_dir = self.video_file.split(fn)[0]
        dlc_config = os.path.join(dlc_dir, "config.yaml")
        print(f"DLC config found: {dlc_config}")
        return dlc_config, newest_pred

    ###################################################################################################################################################

    def toggle_frame_status_fview(self):
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

    def toggle_frame_status_flabel(self):
        if self.current_frame_idx in self.frame_list and self.current_frame_idx not in self.refined_frame_list:
            self.refined_frame_list.append(self.current_frame_idx)

    def mark_all_refined_flabel(self):
        self.refined_frame_list = self.frame_list.copy()

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
        return nvp[self._get_max_priority(frame_lists)]

    def determine_nav_color_counting(self) -> HexColor:
        frame_lists = [
            self.animal_0_list,
            self.animal_1_list,
            self.animal_n_list,
            self.blob_merged_list,
        ]
        return nvpc[self._get_max_priority(frame_lists)]

    def determine_nav_color_flabel(self) -> HexColor:
        frame_lists = [
            self.frame_list,
            self.refined_frame_list
        ]
        return nvp[self._get_max_priority(frame_lists)]
    
    def determine_nav_color_fro(self) -> HexColor:
        frame_lists = [
            self.roi_frame_list,
            self.outlier_frame_list,
        ]
        return nvpl[self._get_max_priority(frame_lists)]

    def _get_max_priority(self, frame_lists:List[List[int]], priorities:Optional[range]=None) -> int:
        """Get the highest priority for current frame."""
        pr = range(1, len(frame_lists) + 1) if priorities is None else priorities
        color_code = 0
        for frame_list, priority in zip(frame_lists, pr):
            if self.current_frame_idx in frame_list:
                color_code = max(color_code, priority)
        return color_code

    def get_title_text(self, labeler:bool=False, kp_edit:bool=False):
        title_text = ""
        if labeler:
            if self.refined_frame_list and self.frame_list:
                title_text += f"    Manual Refining Progress: {len(self.refined_frame_list)} | {len(self.frame_list)} Frames Refined    "
            if kp_edit and self.current_frame_idx:
                title_text += "    ----- KEYPOINTS EDITING MODE -----    "
        elif self.frame_list:
            title_text += f"    Marked Frame Count: {len(self.frame_list)}    "
            
        return title_text

    ###################################################################################################################################################

    def _process_labeled_frame(self, label_file:str=None):
        """Load labeled frames as an separate overlay to the current prediction file."""
        if label_file is None:
            dlc_dir = os.path.dirname(self.dlc_data.dlc_config_filepath)
            self.project_dir = os.path.join(dlc_dir, "labeled-data", self.video_name)

            if not os.path.isdir(self.project_dir):
                return
        
            scorer = self.dlc_data.scorer
            label_data_filename = f"CollectedData_{scorer}.h5"
            label_data_filepath = os.path.join(self.project_dir, label_data_filename)
        else:
            label_data_filepath = label_file
            self.project_dir = os.path.dirname(label_file)

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

    def _init_canon_pose(self):
        head_idx, tail_idx = infer_head_tail_indices(self.dlc_data.keypoints)

        if head_idx is None or tail_idx is None:
            dialog = Head_Tail_Dialog(self.dlc_data.keypoints, self)
            if dialog.exec() == QDialog.Accepted:
                head_idx, tail_idx = dialog.get_selected_indices()
            else:
                QMessageBox.warning(self.main, "Head/Tail Not Set", 
                    "Canonical pose and angle map will not be available.")
                self.canon_pose = None
                self.angle_map_data = None
                return

        self.canon_pose, all_frame_pose = calculate_canonical_pose(self.dlc_data.pred_data_array, head_idx, tail_idx)
        self.angle_map_data = build_angle_map(self.canon_pose, all_frame_pose, head_idx, tail_idx)

    ###################################################################################################################################################

    def save_workspace(self):
        """Save the current workspace state (all vars from reset_dm_vars) to a pickle file."""
        default_name = f"{self.video_name}_workspace.pkl"
        file_path = os.path.join(os.path.dirname(self.video_file), default_name)

        workspace_state = {
            'total_frames': self.total_frames,
            'current_frame_idx': self.current_frame_idx,
            'video_file': self.video_file,
            'video_name': self.video_name,
            'project_dir': self.project_dir,
            'dlc_data': self.dlc_data,
            'canon_pose': self.canon_pose,
            'labeled_frame_list': self.labeled_frame_list,
            'frame_list': self.frame_list,
            'refined_frame_list': self.refined_frame_list,
            'approved_frame_list': self.approved_frame_list,
            'rejected_frame_list': self.rejected_frame_list,
            'animal_0_list': self.animal_0_list,
            'animal_1_list': self.animal_1_list,
            'animal_n_list': self.animal_n_list,
            'blob_merged_list': self.blob_merged_list,
            'label_data_array': self.label_data_array,
            'plot_config': self.plot_config,
            'blob_config': self.blob_config,
            'prediction': self.prediction,
            'angle_map_data': self.angle_map_data,
            'inst_count_per_frame_pred': self.inst_count_per_frame_pred,
            'blob_array': self.blob_array,
            'roi_frame_list': self.roi_frame_list,
            'outlier_frame_list': self.outlier_frame_list,
        }

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(workspace_state, f)
        except Exception as e:
            QMessageBox.critical(self.main, "Error Saving Workspace", f"Failed to save workspace:\n{e}")

    def load_workspace(self):
        """Load a previously saved workspace state."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main, "Load Workspace", "", "Pickle Files (*.pkl);;YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not file_path:
            return

        if file_path.endswith(".yaml"):
            self._load_workspace_legacy(file_path)
            return

        try:
            with open(file_path, 'rb') as f:
                workspace_state = pickle.load(f)

            # Restore all attributes
            self.total_frames = workspace_state.get('total_frames', 0)
            self.current_frame_idx = workspace_state.get('current_frame_idx', 0)
            self.video_file = workspace_state.get('video_file')
            self.video_name = workspace_state.get('video_name')
            self.project_dir = workspace_state.get('project_dir')
            self.dlc_data = workspace_state.get('dlc_data')
            self.canon_pose = workspace_state.get('canon_pose')
            self.labeled_frame_list = workspace_state.get('labeled_frame_list', [])
            self.frame_list = workspace_state.get('frame_list', [])
            self.refined_frame_list = workspace_state.get('refined_frame_list', [])
            self.approved_frame_list = workspace_state.get('approved_frame_list', [])
            self.rejected_frame_list = workspace_state.get('rejected_frame_list', [])
            self.animal_0_list = workspace_state.get('animal_0_list', [])
            self.animal_1_list = workspace_state.get('animal_1_list', [])
            self.animal_n_list = workspace_state.get('animal_n_list', [])
            self.blob_merged_list = workspace_state.get('blob_merged_list', [])
            self.label_data_array = workspace_state.get('label_data_array')
            self.plot_config = workspace_state.get('plot_config')
            self.blob_config = workspace_state.get('blob_config')
            self.prediction = workspace_state.get('prediction')
            self.angle_map_data = workspace_state.get('angle_map_data')
            self.inst_count_per_frame_pred = workspace_state.get('inst_count_per_frame_pred')
            self.blob_array = workspace_state.get('blob_array')
            self.roi_frame_list = workspace_state.get('roi_frame_list', [])
            self.outlier_frame_list = workspace_state.get('outlier_frame_list', [])

            self.init_vid_callback(self.video_file)
            if self.dlc_data is not None:
                self._init_loaded_data()
            self.refresh_callback()
        except Exception as e:
            QMessageBox.critical(self.main, "Error Loading Workspace", f"Failed to load workspace:\n{e}")
            traceback.print_exc()

    def _load_workspace_legacy(self, file_path:str):
        with open(file_path, "r") as fmkf:
            fmk = yaml.safe_load(fmkf)

        if not "frame_list" in fmk.keys():
            QMessageBox.warning(self.main, "File Error", "Not a extractor status file, make sure to load the correct file.")
            return
        
        video_file = fmk["video_path"]

        if not os.path.isfile(video_file):
            QMessageBox.warning(self.main, "Warning", "Video path in file is not valid, has the video been moved?")
            return
        
        self.update_video_path(video_file)
        self.init_vid_callback(video_file)

        dlc_config = fmk["dlc_config"]
        prediction = fmk["prediction"]

        if dlc_config and prediction:
            self.load_pred_to_dm(dlc_config, prediction)

        self.frame_list = fmk["frame_list"]
        if "refined_frame_list" in fmk.keys():
            self.refined_frame_list = fmk["refined_frame_list"]
        if "approved_frame_list" in fmk.keys():
            self.approved_frame_list = fmk["approved_frame_list"]
        if "rejected_frame_list" in fmk.keys():
            self.rejected_frame_list = fmk["rejected_frame_list"]
            
        self._init_loaded_data()
        self.refresh_callback()

    ###################################################################################################################################################

    def save_pred(self, pred_data_array:np.ndarray, is_label_file:bool=False):
        if is_label_file:
            backup_existing_prediction(self.prediction) # Only backup for label file as overwriting in situ
            pred_data_array = remove_confidence_score(pred_data_array)
            save_path = self.prediction
        else:
            save_path = determine_save_path(self.prediction, suffix="_track_labeler_modified_")

        status, msg = save_prediction_to_existing_h5(
            save_path,
            pred_data_array,
            self.dlc_data.keypoints,
            self.dlc_data.multi_animal
            )
        
        return save_path, status, msg

    def save_pred_to_csv(self):
        save_path = os.path.dirname(self.prediction)
        pred_file = os.path.basename(self.prediction).split(".")[0]
        exp_set = Export_Settings(self.video_file, self.video_name, save_path, "CSV")
        try:
            prediction_to_csv(self.dlc_data, self.dlc_data.pred_data_array, exp_set)
            QMessageBox.information(self.main, "Save Successful",
                f"Successfully saved modified prediction in csv to: {os.path.join(save_path, pred_file)}.csv")
        except Exception as e:
            QMessageBox.critical(self.main, "Saving Error", f"An error occurred during csv saving: {e}")
            print(f"An error occurred during csv saving: {e}")

    def reload_pred_to_dm(self, prediction_path:str):
        data_loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, prediction_path)
        self.prediction = prediction_path
        self.dlc_data = data_loader.load_data()
        self.approved_frame_list[:] = list(set(self.approved_frame_list) - set(self.refined_frame_list))
        self.rejected_frame_list[:] = list(set(self.rejected_frame_list) - set(self.refined_frame_list))
        self._init_canon_pose()

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
                self._process_labeled_frame()
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
                QMessageBox.information(self.main, "Success", "Successfully exported frames and prediction to DLC.")
            except Exception as e:
                QMessageBox.critical(self.main, "Error Merge Data", f"Error merging data to DLC: {e}")

            self._process_labeled_frame()

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

    def export_lists_json(self):
        list_name = f"{self.video_name}_frame_lists.json"
        file_path = os.path.join(os.path.dirname(self.video_file), list_name)

        data = {
            'refined_frame_list': self.refined_frame_list,
            'frame_list': self.frame_list,
            'labeled_frame_list': self.labeled_frame_list,
            'approved_frame_list': self.approved_frame_list,
            'rejected_frame_list': self.rejected_frame_list,
            'animal_0_list': self.animal_0_list,
            'animal_1_list': self.animal_1_list,
            'animal_n_list': self.animal_n_list,
            'blob_merged_list': self.blob_merged_list,
            'roi_frame_list': self.roi_frame_list,
            'outlier_frame_list': self.outlier_frame_list,
        }

        data = {k: v for k, v in data.items() if v}

        def default_handler(obj):
            if hasattr(obj, 'item'):  # catches np.int64, np.float32, etc.
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} not serializable")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=default_handler)
            print(f"Frame lists exported to: {file_path}")
        except Exception as e:
            print(f"Error exporting frame lists: {e}")