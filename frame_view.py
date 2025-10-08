import os
import yaml
import pandas as pd
import numpy as np
import cv2

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QFileDialog

import traceback

from utils import helper as duh, pose as dupe
from core.io import (
    Prediction_Loader, Exporter, Frame_Extractor,
    remove_confidence_score, append_new_video_to_dlc_config
)
from core.dataclass import Export_Settings, Plot_Config, Nav_Callback
from ui import (
    Menu_Widget, Clear_Mark_Dialog, Video_Player_Widget
    )
from core.tool import (
    Prediction_Plotter, Mark_Generator, Canonical_Pose_Dialog,
    Plot_Config_Menu, Blob_Counter, navigate_to_marked_frame,
    )

class Frame_View(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frame Viewer")
        self.setGeometry(100, 100, 1200, 960)

        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        extractor_menu_config = {
            "File": {
                "buttons": [
                    ("Load Video", self.load_video),
                    ("Load Prediction", self.load_prediction),
                    ("Load Workspace", self.load_workspace),
                ]
            },
            "View": {
                "buttons": [
                    ("Toggle Labeled Predictions Visiblity", self.toggle_labeled_vis, {"checkable": True, "checked": True}),
                    ("Toggle Predictions Visiblity", self.toggle_pred_vis, {"checkable": True, "checked": True}),
                    ("Toggle Navigating Labeled Frames", self.toggle_labeled_nav, {"checkable": True, "checked": False}),
                    ("Toggle Animal Counting", self.toggle_animal_counting, {"checkable": True, "checked": False}),
                    ("View Canonical Pose", self.view_canonical_pose),
                    ("Count Animals Options", self.count_animals_options),
                ]
            },
            "Mark": {
                "buttons": [
                    ("Mark / Unmark Current Frame (X)", self.toggle_frame_status),
                    ("Clear Frame Marks of Category", self.show_clear_mark_dialog),
                    ("Automatic Mark Generation", self.toggle_mark_gen_menu),
                    ("Plot Config Menu", self.open_plot_config_menu),
                ]
            },
            "Edit": {
                "buttons": [
                    ("Call Labeler - Track Correction", lambda: self.call_labeler(track_only=True)),
                    ("Call Labeler - Edit Marked Frames", lambda: self.call_labeler(track_only=False)),
                    ("Call DeepLabCut - Run Predictions of Marked Frames", self.dlc_inference_marked),
                    ("Call DeepLabCut - Run Predictions on Entire Video", self.dlc_inference_all),
                ]
            },
            "Export": {
                "display_name": "Save",
                "buttons": [
                    ("Save the Current Workspace", self.save_workspace),
                    ("Export to DeepLabCut", self.save_to_dlc),
                    ("Export Marked Frame Indices to Clipboard", self.export_marked_to_clipboard),
                    ("Merge with Existing Label in DeepLabCut", self.merge_data)
                ]
            }
        }
        self.menu_widget.add_menu_from_config(extractor_menu_config)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.app_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Video display area
        nav_callback = Nav_Callback(
            change_frame_callback = self._change_frame,
            nav_prev_callback = self._navigate_prev,
            nav_next_callback = self._navigate_next,
        )
        self.vid_play = Video_Player_Widget(
            slider_callback = self._handle_frame_change_from_comp,
            nav_callback = nav_callback,
            parent = self,
            )

        self.app_layout.addWidget(self.vid_play)

        self._setup_shortcut()        
        self.reset_state()

    def _setup_shortcut(self):
        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self._change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self._change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self._change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self._change_frame(10))
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self.toggle_frame_status)
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self._navigate_prev)
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self._navigate_next)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.vid_play.sld.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_workspace)

    def reset_state(self):
        self.video_file, self.video_name, self.project_dir = None, None, None
        self.dlc_data, self.canon_pose = None, None

        self.labeled_frame_list, self.frame_list = [], []
        self.refined_frame_list, self.approved_frame_list, self.rejected_frame_list = [], [], []
        self.animal_0_list, self.animal_1_list, self.animal_n_list = [], [], []
        self.label_data_array = None

        self.extractor, self.current_frame = None, None

        self.is_counting = False
        self.open_mark_gen, self.open_config = False, False
        self.is_saved = True
        self.last_saved = []
        self.plot_labeled, self.plot_pred = True, True
        self.navigate_labeled = False

        self.plot_config = Plot_Config(
            plot_opacity =1.0, point_size = 6.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False)
        
        self.blob_counter = None

    def load_video(self):
        file_dialog = QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if video_path:
            self.reset_state()
            self.video_file = video_path
            self.initialize_loaded_video()

    def initialize_loaded_video(self):
        self.video_name = os.path.basename(self.video_file).split(".")[0]

        self.extractor = Frame_Extractor(video_path=self.video_file)
        self.blob_counter = Blob_Counter(frame_extractor=self.extractor, parent=self)
        self.blob_counter.frame_processed.connect(self._plot_current_frame)
        self.blob_counter.video_counted.connect(self._handle_counter_from_counter)

        if self.is_counting:
            self.vid_play.set_left_panel_widget(self.blob_counter)

        self.total_frames = self.extractor.get_total_frames()

        self.current_frame_idx = 0
        self.vid_play.sld.set_slider_range(self.total_frames)
        self._refresh_slider()
        self.display_current_frame()
        self.navigation_title_controller()
        print(f"Video loaded: {self.video_file}")

    def load_prediction(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        file_dialog = QFileDialog(self)
        prediction_path, _ = file_dialog.getOpenFileName(self, "Select Prediction", "", "HDF5 Files (*.h5);;All Files (*)")

        if not prediction_path:
            return

        prediction_filename = os.path.basename(prediction_path)
        if prediction_filename.startswith("CollectedData_"):
            is_label_data = True
            QMessageBox.information(self, "Labeled Data Selected", "Labeled data selected, now loading DLC config.")
        else:
            is_label_data = False
            QMessageBox.information(self, "Prediction Selected", "Prediction selected, now loading DLC config.")

        file_dialog = QFileDialog(self)
        dlc_config, _ = file_dialog.getOpenFileName(self, "Select DLC Config", "", "YAML Files (config.yaml);;All Files (*)")

        if not dlc_config:
            return

        data_loader = Prediction_Loader(dlc_config, prediction_path)

        if is_label_data:
            if not self.dlc_data:
                try:
                    self.dlc_data = data_loader.load_data(metadata_only=True)
                except Exception as e:
                    QMessageBox.critical(self, "Error Loading Prediction",
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
                QMessageBox.critical(self, "Error Loading Prediction",
                                     f"Unexpected error during prediction loading: {e}.")
            self.process_labeled_frame()

        self.initialize_canon_pose()
        self.display_current_frame()

    def load_labeled(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        file_dialog = QFileDialog(self)
        labeled_path, _ = file_dialog.getOpenFileName(self, "Load Prediction", "", "HDF5 Files (*.h5);;All Files (*)")

        if not labeled_path:
            return

    def load_workspace(self):
        file_dialog = QFileDialog(self)
        marked_frame_path, _ = file_dialog.getOpenFileName(self, "Load Status", "", "YAML Files (*.yaml);;All Files (*)")

        if marked_frame_path:
            self.reset_state()
            with open(marked_frame_path, "r") as fmkf:
                fmk = yaml.safe_load(fmkf)

            if not "frame_list" in fmk.keys():
                QMessageBox.warning(self, "File Error", "Not a extractor status file, make sure to load the correct file.")
                return
            
            video_file = fmk["video_path"]

            if not os.path.isfile(video_file):
                QMessageBox.warning(self, "Warning", "Video path in file is not valid, has the video been moved?")
                return
            
            self.video_file = video_file
            self.initialize_loaded_video()
            self.frame_list = fmk["frame_list"]
            print(f"Marked frames loaded: {self.frame_list}")

            dlc_config = fmk["dlc_config"]
            prediction = fmk["prediction"]

            if dlc_config and prediction:
                data_loader = Prediction_Loader(dlc_config, prediction)

            try:
                self.dlc_data = data_loader.load_data()
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Prediction", f"Unexpected error during prediction loading: {e}.")

            self.initialize_canon_pose()

            if "refined_frame_list" in fmk.keys():
                self.refined_frame_list = fmk["refined_frame_list"]
            
            if "approved_frame_list" in fmk.keys():
                self.approved_frame_list = fmk["approved_frame_list"]

            if "rejected_frame_list" in fmk.keys():
                self.rejected_frame_list = fmk["rejected_frame_list"]
                
            self._refresh_slider()
            self.determine_save_status()
            self.process_labeled_frame()
            self.display_current_frame()

    def process_labeled_frame(self, lbf=None):
        if lbf is None:
            dlc_dir = os.path.dirname(self.dlc_data.dlc_config_filepath)
            self.project_dir = os.path.join(dlc_dir, "labeled-data", self.video_name)

            self.labeled_frame_list = []
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

        self._refresh_slider()
        self.navigation_title_controller()

    def initialize_plotter(self):
        current_frame_data = np.full((self.dlc_data.instance_count, self.dlc_data.num_keypoint*3), np.nan)
        self.plotter = Prediction_Plotter(
            dlc_data = self.dlc_data,
            current_frame_data = current_frame_data,
            plot_config = self.plot_config,
            frame_cv2 = self.current_frame)
        
    def initialize_canon_pose(self):
        head_idx, tail_idx = duh.infer_head_tail_indices(self.dlc_data.keypoints)
        if head_idx is None or tail_idx is None:
            return
        if not np.all(np.isnan(self.dlc_data.pred_data_array)):
            self.canon_pose, _ = dupe.calculate_canonical_pose(self.dlc_data.pred_data_array, head_idx, tail_idx)
        elif self.label_data_array is None:
            return
        else:
            self.canon_pose, _ = dupe.calculate_canonical_pose(self.label_data_array, head_idx, tail_idx)

    def count_animals_options(self):
        if self.current_frame is None or self.extractor is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        if self.blob_counter:
            self.blob_counter.show()

###################################################################################################################################################

    def display_current_frame(self):
        if not self.extractor:
            self.vid_play.display.setText("No video loaded")

        frame = self.extractor.get_frame(self.current_frame_idx)
        if frame is None:
            self.vid_play.display.setText("Failed to load current frame.")
            return
        
        self.current_frame = frame
        if self.is_counting:
            self.blob_counter.set_current_frame(frame=frame)
        else:
            self._plot_current_frame(frame)

    def _plot_current_frame(self, frame, count=None):
        if self.dlc_data is not None:
            if not hasattr(self, "plotter"):
                self.initialize_plotter()

            if self.plot_pred:
                self.plotter.frame_cv2 = frame
                self.plotter.current_frame_data = self.dlc_data.pred_data_array[self.current_frame_idx,:,:]
                frame = self.plotter.plot_predictions()

            if self.current_frame_idx in self.labeled_frame_list and self.plot_labeled:
                self.plotter.frame_cv2 = frame
                self.plotter.current_frame_data = self.label_data_array[self.current_frame_idx,:,:]
                old_colors = self.plotter.color.copy()
                self.plotter.color = [(200, 130, 0), (40, 200, 40), (40, 120, 200), (200, 40, 40), (200, 200, 80)]
                frame = self.plotter.plot_predictions()
                self.plotter.color = old_colors

        # Convert OpenCV image to QPixmap
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)

        # Scale pixmap to fit label
        scaled_pixmap = pixmap.scaled(self.vid_play.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.vid_play.display.setPixmap(scaled_pixmap)
        self.vid_play.display.setText("")
        self.vid_play.sld.set_current_frame(self.current_frame_idx) # Update slider handle's position

    ###################################################################################################################################################

    def navigation_title_controller(self):
        title_text = f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1} | Video: {self.video_name}"
        if self.frame_list:
            title_text += f" | Marked Frame Count: {len(self.frame_list)}"
        self.vid_play.nav.setTitle(title_text)
        if self.current_frame_idx in self.labeled_frame_list:
            self.vid_play.nav.setTitleColor("#1F32D7")  # Blue
        elif self.current_frame_idx in self.refined_frame_list:
            self.vid_play.nav.setTitleColor("#009979")  # Teal/Green 
        elif self.current_frame_idx in self.approved_frame_list:
            self.vid_play.nav.setTitleColor("#68b3ff")  # Sky Blue
        elif self.current_frame_idx in self.rejected_frame_list:
            self.vid_play.nav.setTitleColor("#F749C6")  # Pink
        elif self.current_frame_idx in self.frame_list:
            self.vid_play.nav.setTitleColor("#E28F13")  # Amber
        else:
            self.vid_play.nav.setTitleColor("black")

    def _refresh_slider(self):
        self.vid_play.sld.set_frame_category("marked_frames", self.frame_list, "#E28F13")
        self.vid_play.sld.set_frame_category("refined_frames", self.refined_frame_list, "#009979", priority=6)
        self.vid_play.sld.set_frame_category("approved_frames", self.approved_frame_list, "#68b3ff", priority=6)
        self.vid_play.sld.set_frame_category("rejected_frames", self.rejected_frame_list, "#F749C6", priority=6)
        self.vid_play.sld.set_frame_category("labeled_frames", self.labeled_frame_list, "#2A39C4", priority=7)
        if self.is_counting:
            self.vid_play.sld.set_frame_category("zero_animal_frames", self.animal_0_list, "#FF1100", priority=10)
            self.vid_play.sld.set_frame_category("one_animal_frames", self.animal_1_list, "#0400FF", priority=10)
            self.vid_play.sld.set_frame_category("muliple_animal_frames", self.animal_n_list, "#00FF2A", priority=10)

    ###################################################################################################################################################

    def toggle_frame_status(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        if self.current_frame_idx in self.labeled_frame_list:
            QMessageBox.information(self, "Already Labeled", "The frame is already in the labeled dataset, skipping...")
            return

        if self.current_frame_idx in self.refined_frame_list:
            reply = QMessageBox.question(
                self, "Confirm Unmarking",
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

        self.determine_save_status()
        self._refresh_slider()
        self.navigation_title_controller()

    def _change_frame(self, delta):
        if self.extractor:
            new_frame_idx = self.current_frame_idx + delta
            if 0 <= new_frame_idx < self.total_frames:
                self.current_frame_idx = new_frame_idx
                self.display_current_frame()
                self.navigation_title_controller()

    def _navigate_prev(self):
        list_to_nav = self.labeled_frame_list if self.navigate_labeled else self.frame_list
        navigate_to_marked_frame(
            self, list_to_nav, self.current_frame_idx, self._handle_frame_change_from_comp, "prev")

    def _navigate_next(self):
        list_to_nav = self.labeled_frame_list if self.navigate_labeled else self.frame_list
        navigate_to_marked_frame(
            self, list_to_nav, self.current_frame_idx, self._handle_frame_change_from_comp, "next")

    ###################################################################################################################################################

    def open_plot_config_menu(self):
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        if not self.dlc_data:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        self.open_config = not self.open_config

        if self.open_config:
            self.open_mark_gen = False
            plot_config_widget = Plot_Config_Menu(plot_config=self.plot_config, skip_opacity=True)
            plot_config_widget.config_changed.connect(self._handle_config_from_config)
            self.vid_play.set_right_panel_widget(plot_config_widget)
        else:
            self.vid_play.set_right_panel_widget(None)

    def toggle_mark_gen_menu(self):
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        self.open_mark_gen = not self.open_mark_gen

        if self.open_mark_gen:
            self.open_config = False
            mark_gen = Mark_Generator(self.total_frames, self.dlc_data, self.canon_pose, parent=self)
            mark_gen.clear_old.connect(self._on_clear_old_command)
            mark_gen.frame_list_new.connect(self._handle_frame_marks_from_comp)
            self.vid_play.set_right_panel_widget(mark_gen)
        else:
            self.vid_play.set_right_panel_widget(None)

    def show_clear_mark_dialog(self):
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

        if not frame_categories:
            return

        mark_clear_dialog = Clear_Mark_Dialog(frame_categories, parent=self)
        mark_clear_dialog.frame_category_to_clear.connect(self._clear_category)
        mark_clear_dialog.exec()

    def view_canonical_pose(self):
        dialog = Canonical_Pose_Dialog(self.dlc_data, self.canon_pose)
        dialog.exec()

    def toggle_labeled_vis(self):
        self.plot_labeled = not self.plot_labeled
        self.display_current_frame()

    def toggle_pred_vis(self):
        self.plot_pred = not self.plot_pred
        self.display_current_frame()

    def toggle_labeled_nav(self):
        self.navigate_labeled = not self.navigate_labeled
        self.display_current_frame()

    def toggle_animal_counting(self):
        self.is_counting = not self.is_counting
        
        if self.is_counting:
            self.vid_play.set_left_panel_widget(self.blob_counter)
        else:
            self.vid_play.set_left_panel_widget(None)

        self.display_current_frame()

    def _clear_category(self, frame_category):
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

        self._refresh_slider()
        self.navigation_title_controller()

    def _on_clear_old_command(self, clear_old:bool):
        if not clear_old:
            return
        if not self.refined_frame_list:
            self._clear_category("All Marked Frames")
        else:
            self._clear_category("All Marked Frames (Except for Refined)")

    ###################################################################################################################################################

    def _handle_refined_frames_exported(self, refined_frames):
        self.refined_frame_list = refined_frames
        self._refresh_slider()
        self.display_current_frame()
        self.navigation_title_controller()
        self.determine_save_status()

    def _handle_rerun_frames_exported(self, frame_tuple):
        self.approved_frame_list, self.rejected_frame_list = frame_tuple
        self._refresh_slider()
        self.display_current_frame()
        self.navigation_title_controller()
        self.determine_save_status()

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.current_frame_idx = new_frame_idx
        self.display_current_frame()
        self.navigation_title_controller()
        self.determine_save_status()

    def _handle_frame_marks_from_comp(self, frame_list):
        frame_set = set(self.frame_list) | set(frame_list) - set(self.labeled_frame_list)
        self.frame_list[:] = list(frame_set)
        self._refresh_slider()
        self.display_current_frame()
        self.navigation_title_controller()
        self.determine_save_status()

    def _handle_counter_from_counter(self, count_list):
        count_array = np.array(count_list)
        self.animal_0_list = list(np.where(count_array==0)[0])
        self.animal_1_list = list(np.where(count_array==1)[0])
        self.animal_n_list = list(np.where((count_array!=1) & (count_array!=0))[0])
        self._refresh_slider()

    def _handle_config_from_config(self, new_config:Plot_Config):
        self.plot_config = new_config
        self.display_current_frame()

    ###################################################################################################################################################

    def determine_save_status(self):
        if set(self.last_saved) == set(self.frame_list):
            self.is_saved = True
        else:
            self.is_saved = False

    def pre_saving_sanity_check(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return False
        if not self.frame_list:
            QMessageBox.warning(self, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return False
        return True

    def save_workspace(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return False
        self.last_saved = self.frame_list
        self.is_saved = True
        if self.dlc_data:
            save_yaml = {'video_path': self.video_file,  'frame_list': self.last_saved, 'dlc_config': self.dlc_data.dlc_config_filepath,
                'prediction': self.dlc_data.prediction_filepath, 'refined_frame_list': self.refined_frame_list,
                'approved_frame_list': self.approved_frame_list, 'rejected_frame_list': self.rejected_frame_list}
        else:
            save_yaml = {'video_path': self.video_file,  'frame_list': self.last_saved, 'dlc_config': None,
                'prediction': None, 'refined_frame_list': self.refined_frame_list,
                'approved_frame_list': self.approved_frame_list, 'rejected_frame_list': self.rejected_frame_list }
        output_filepath = os.path.join(os.path.dirname(self.video_file), f"{self.video_name}_frame_extractor.yaml")

        with open(output_filepath, 'w') as file:
            yaml.dump(save_yaml, file)
            
        self.statusBar().showMessage(f"Current workspace files have been saved to {output_filepath}")
        return True

    def call_labeler(self, track_only=False):
        if not self.video_file:
            QMessageBox.warning(self, "Video Not Loaded", "No video is loaded, load a video first!")
            return
        if not self.dlc_data:
            QMessageBox.warning(self, "DLC Data Not Loaded", "No DLC data has been loaded, load them to export to Labeler.")
            return
        
        from frame_label import Frame_Label

        try:
            self.labeler_window = Frame_Label()
            self.labeler_window.video_file = self.video_file
            self.labeler_window.initialize_loaded_video()
            self.labeler_window.dlc_data = self.dlc_data
            if not track_only:
                self.labeler_window.marked_roi_frame_list = self.frame_list
                self.labeler_window.refined_roi_frame_list = self.refined_frame_list
            self.labeler_window.current_frame_idx = self.current_frame_idx
            self.labeler_window.prediction = self.dlc_data.prediction_filepath
            self.labeler_window.initialize_loaded_data()
            self.labeler_window.display_current_frame()
            self.labeler_window.navigation_title_controller()
            if self.frame_list and not track_only:
                self.labeler_window.direct_keypoint_edit()
                self.labeler_window.refined_frames_exported.connect(self._handle_refined_frames_exported)
            self.labeler_window.show()
            self.labeler_window.prediction_saved.connect(self.reload_prediction) # Reload from prediction provided by Labeler
            
        except Exception as e:
            error_message = f"Labeler failed to initialize. Error: {e}"
            detailed_message = f"{error_message}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.warning(self, "Labeler Failed", detailed_message)

    def dlc_inference_marked(self):
        inference_list = list(set(self.frame_list) - set(self.approved_frame_list) - set(self.rejected_frame_list) - set(self.refined_frame_list))

        if not inference_list:
            self.statusBar().showMessage("No unapproved / unrejected/ unrefined marked frames to inference.")
            return
        
        self.call_inference(inference_list)

    def dlc_inference_all(self):
        pass
    
    def call_inference(self, inference_list:list):
        if not self.video_file:
            QMessageBox.warning(self, "Video Not Loaded", "No video is loaded, load a video first!")
            return
        if not self.frame_list:
            QMessageBox.warning(self, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return
        if self.dlc_data is None:
            QMessageBox.information(self, "Load DLC Config", "You need to load DLC config to inference with DLC models.")

            file_dialog = QFileDialog(self)
            dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")

            if not dlc_config:
                return

            data_loader = Prediction_Loader(dlc_config)
            try:
                self.dlc_data = data_loader.load_data(metadata_only=True)
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Prediction", f"Unexpected error during prediction loading: {e}.")

            self.dlc_data.pred_frame_count = self.total_frames

        from core import DLC_Inference
        try:
            self.inference_window = DLC_Inference(dlc_data=self.dlc_data, frame_list=inference_list, video_filepath=self.video_file, parent=self)
            self.inference_window.show()
            self.inference_window.frames_exported.connect(self._handle_rerun_frames_exported)
            self.inference_window.prediction_saved.connect(self.reload_prediction)
        except Exception as e:
            error_message = f"Inference Process failed to initialize. Exception: {e}"
            detailed_message = f"{error_message}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.warning(self, "Inference Failed", detailed_message)
            return

    def reload_prediction(self, prediction_path):
        """Reload prediction data from file and update visualization"""
        data_loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, prediction_path)
        self.dlc_data = data_loader.load_data()
        self.approved_frame_list[:] = list(set(self.approved_frame_list) - set(self.refined_frame_list))
        self.rejected_frame_list[:] = list(set(self.rejected_frame_list) - set(self.refined_frame_list))

        self.display_current_frame()
        self.statusBar().showMessage("Prediction successfully reloaded")

        if hasattr(self, 'labeler_window') and self.labeler_window: # Clean labeler windows
            self.labeler_window.close()
            self.labeler_window = None
        
        if hasattr(self, "inference_window") and self.inference_window:
            self.inference_window.close()
            self.inference_window = None

    def export_marked_to_clipboard(self):
        df = pd.DataFrame([self.frame_list])
        df.to_clipboard(sep=',', index=False, header=False)
        self.statusBar().showMessage("Marked frames exported to clipboard.")

    def save_to_dlc(self):
        if not self.pre_saving_sanity_check():
            return

        dlc_dir = os.path.dirname(self.dlc_data.dlc_config_filepath)

        exp_set = Export_Settings(video_filepath=self.video_file,
                                  video_name=self.video_name,
                                  save_path=self.project_dir,
                                  export_mode="Append")

        self.save_workspace()

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
                QMessageBox.information(self, "Success", "Successfully exported frames and prediction to DLC.")
            except Exception as e:
                QMessageBox.critical(self, "Error Save Data", f"Error saving data to DLC: {e}")
            append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, self.video_name)

            if exp_set.export_mode == "Merge":
                self.process_labeled_frame()
        else:
            reply = QMessageBox.question(
                self,
                "No Prediction Loaded",
                "No prodiction has been loaded. Would you like export frames only?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                QMessageBox.information(self, "Frame Only Mode", "Choose the directory of DLC project.")
                dlc_dir = QFileDialog.getExistingDirectory(
                            self, "Select Project Folder",
                            os.path.dirname(self.video_file),
                            QFileDialog.ShowDirsOnly
                        )
                if not dlc_dir: # When user close the file selection window
                    return
                try:
                    exporter.export_data_to_DLC(frame_only=True)
                    QMessageBox.information(self, "Success", "Successfully exported marked frames to DLC for labeling!")
                except Exception as e:
                    QMessageBox.critical(self, "Error Export Frames", f"Error exporting marked frames to DLC: {e}")
                return
            else:
                self.load_prediction()
                if self.dlc_data is None:
                    return

    def merge_data(self):
        if not self.pre_saving_sanity_check():
            return
        
        if not self.refined_frame_list:
            QMessageBox.warning(self, "No Refined Frame", "No frame has been refined, please refine some marked frames first.")
            return
        
        if not self.labeled_frame_list:
            self.save_to_dlc()
            return

        reply = QMessageBox.question(
            self, "Confirm Merge",
            "This action will merge the selected data into the labeled dataset. "
            "Please ensure you have reviewed and refined the predictions on the marked frames.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return
        else:
            self.save_workspace()

            exp_set = Export_Settings(video_filepath=self.video_file,
                                      video_name=self.video_name,
                                      save_path=self.project_dir,
                                      export_mode="Merge")

            self.label_data_array[self.refined_frame_list, :, :] = self.dlc_data.pred_data_array[self.refined_frame_list, :, :]
            merge_frame_list = list(set(self.labeled_frame_list) | set(self.refined_frame_list))
            label_data_array_export = remove_confidence_score(self.label_data_array)

            exporter = Exporter(dlc_data=self.dlc_data,
                                export_settings=exp_set,
                                frame_list=merge_frame_list,
                                pred_data_array=label_data_array_export)
            
            try:
                exporter.export_data_to_DLC()
                QMessageBox.information(self, "Success", "Successfully exported frames and prediction to DLC.")
            except Exception as e:
                QMessageBox.critical(self, "Error Merge Data", f"Error merging data to DLC: {e}")

            self.process_labeled_frame()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange and self.extractor:
            self.display_current_frame()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        duh.handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_workspace)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = Frame_View()
    window.show()
    app.exec()