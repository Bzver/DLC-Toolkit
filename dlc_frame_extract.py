import os

import h5py
import yaml

import pandas as pd
import numpy as np

import cv2

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QFileDialog

from utils.dtu_io import DLC_Loader, DLC_Exporter
from utils.dtu_widget import Menu_Widget, Progress_Bar_Widget, Nav_Widget
from utils.dtu_comp import Adjust_Property_Dialog
from utils.dtu_dataclass import Export_Settings
import utils.dtu_helper as duh
import utils.dtu_gui_helper as dugh

class DLC_Extractor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Manual Frame Extractor")
        self.setGeometry(100, 100, 1200, 960)

        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        extractor_menu_config = {
            "File": {
                "display_name": "File",
                "buttons": [
                    ("Load Video", self.load_video),
                    ("Load Prediction", self.load_prediction),
                    ("Load Workplace", self.load_workplace)
                ]
            },
            "Edit": {
                "display_name": "Edit",
                "buttons": [
                    ("Mark / Unmark Current Frame (X)", self.toggle_frame_status),
                    ("Adjust Confidence Cutoff", self.show_confidence_dialog),
                    ("Call Refiner - Track Edit Only", lambda: self.call_refiner(track_only=True)),
                    ("Call Refiner", lambda: self.call_refiner(track_only=False))
                ]
            },
            "Export": {
                "display_name": "Save",
                "buttons": [
                    ("Save the Current Workspace", self.save_workspace),
                    ("Export to DLC", self.save_to_dlc),
                    ("Merge with Existing Data", self.merge_data)
                ]
            }
        }
        self.menu_widget.add_menu_from_config(extractor_menu_config)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Video display area
        self.video_label = QtWidgets.QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label, 1)

        self.progress_widget = Progress_Bar_Widget()
        self.layout.addWidget(self.progress_widget)
        self.progress_widget.frame_changed.connect(self._handle_frame_change_from_comp)

        # Navigation controls
        self.nav_widget = Nav_Widget()
        self.layout.addWidget(self.nav_widget)
        self.nav_widget.set_collapsed(True)

        self.nav_widget.frame_changed_sig.connect(self.change_frame)
        self.nav_widget.prev_marked_frame_sig.connect(lambda:self._navigate_marked_frames("prev"))
        self.nav_widget.next_marked_frame_sig.connect(lambda:self._navigate_marked_frames("next"))

        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self.toggle_frame_status)
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(lambda:self._navigate_marked_frames("prev"))
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(lambda:self._navigate_marked_frames("next"))
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.progress_widget.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_workspace)
        
        self.reset_state()

    def reset_state(self):
        self.video_file, self.video_name, self.project_dir = None, None, None
        self.dlc_data = None

        self.data_loader = DLC_Loader(None, None) # Initialize the data loader
        self.exp_set = Export_Settings(
            video_filepath=None, video_name=None, save_path=None, export_mode=None
        ) # Initialize export_settings

        self.labeled_frame_list, self.frame_list, self.refined_frame_list = [], [], []
        self.label_data_array = None

        self.cap, self.current_frame = None, None
        self.confidence_cutoff = 0

        self.is_playing = False
        self.is_saved = True
        self.last_saved = []

        self.nav_widget.set_collapsed(True)
        self.refiner_window = None

    def load_video(self):
        self.reset_state()
        file_dialog = QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if video_path:
            self.video_file = video_path
            self.exp_set.video_filepath = video_path
            self.initialize_loaded_video()
            self.nav_widget.set_collapsed(False)

    def initialize_loaded_video(self):
        self.video_name = os.path.basename(self.video_file).split(".")[0]
        self.exp_set.video_name = self.video_name
        self.cap = cv2.VideoCapture(self.video_file)

        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_file}")
            self.video_label.setText("Error: Could not open video")
            self.cap = None
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.progress_widget.set_slider_range(self.total_frames)
        self.progress_widget.set_frame_category("marked_frames", self.frame_list, "#E28F13")
        self.progress_widget.set_frame_category("refined_frames", self.refined_frame_list, "#009979", priority=7)
        self.progress_widget.set_frame_category("labeled_frames", self.labeled_frame_list, "#1F32D7", priority=9)
        self.display_current_frame()
        self.navigation_title_controller()
        self.nav_widget.set_collapsed(False)
        print(f"Video loaded: {self.video_file}")

    def load_prediction(self):

        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        file_dialog = QFileDialog(self)
        prediction_path, _ = file_dialog.getOpenFileName(self, "Load Prediction", "", "HDF5 Files (*.h5);;All Files (*)")

        if not prediction_path:
            return
        
        self.data_loader.prediction_filepath = prediction_path

        QMessageBox.information(self, "DLC Config Loaded", "Prediction loaded, now loading DLC config.")

        file_dialog = QFileDialog(self)
        dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")

        if not dlc_config:
            return

        self.data_loader.dlc_config_filepath = dlc_config

        self.dlc_data = dugh.load_and_show_message(self, self.data_loader)

        self.process_labeled_frame()
        self.display_current_frame()

    def load_workplace(self):
        file_dialog = QFileDialog(self)
        marked_frame_path, _ = file_dialog.getOpenFileName(self, "Load Status", "", "YAML Files (*.yaml);;All Files (*)")

        if marked_frame_path:
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
            self.exp_set.video_filepath = video_file
            self.initialize_loaded_video()
            self.frame_list = fmk["frame_list"]
            print(f"Marked frames loaded: {self.frame_list}")

            dlc_config = fmk["dlc_config"]
            prediction = fmk["prediction"]

            if dlc_config and prediction:
                self.data_loader.dlc_config_filepath = dlc_config
                self.data_loader.prediction_filepath = prediction

                self.dlc_data = dugh.load_and_show_message(self, self.data_loader)

            if "refined_frame_list" in fmk.keys():
                self.refined_frame_list = fmk["refined_frame_list"]
                
            self.progress_widget.set_frame_category("marked_frames", self.frame_list, "#E28F13")
            self.progress_widget.set_frame_category("refined_frames", self.refined_frame_list, "#009979", priority=7)
            self.determine_save_status()
            self.process_labeled_frame()
            self.display_current_frame()

    def process_labeled_frame(self):
        dlc_dir = os.path.dirname(self.data_loader.dlc_config_filepath)
        self.project_dir = os.path.join(dlc_dir, "labeled-data", self.video_name)
        self.exp_set.save_path = self.project_dir

        if not os.path.isdir(self.project_dir):
            self.labeled_frame_list = []
            return
        
        label_data_files = [ f for f in os.listdir(self.project_dir) if f.startswith("CollectedData_") and f.endswith(".h5") ]
        if not label_data_files:
            return
        
        self.labeled_frame_list = []
        # Initialize an empty label data array that has same size as pred_data_array
        self.label_data_array = np.full_like(self.dlc_data.pred_data_array, np.nan)
        for label_data_file in label_data_files:
            with h5py.File(os.path.join(self.project_dir, label_data_file), "r") as lbf:
                labeled_frame_list = lbf["df_with_missing"]["axis1_level2"].asstr()[()]
                labeled_frame_list = [int(f.split("img")[1].split(".")[0]) for f in labeled_frame_list]
                labeled_frame_list.sort()
                labeled_data_flattened = lbf["df_with_missing"]["block0_values"]
                self.labeled_frame_list.extend(labeled_frame_list)
                
                rows, cols = labeled_data_flattened.shape # Check if labeled_data_flattened already have conf or not
                if cols / self.dlc_data.num_keypoint == 3 * self.dlc_data.instance_count:
                    labeled_data_with_conf = labeled_data_flattened
                else:
                    labeled_data_with_conf = duh.add_mock_confidence_score(labeled_data_flattened)
                labeled_data_unflattened = duh.unflatten_data_array(
                    labeled_data_with_conf, self.dlc_data.instance_count)
                self.label_data_array[labeled_frame_list,:,:] = labeled_data_unflattened
        
        self.frame_list = list(set(self.frame_list) - set(self.labeled_frame_list)) # Clean up the already labeled marked frames
        self.refined_frame_list = list(set(self.refined_frame_list) - set(self.labeled_frame_list))
        self.frame_list.sort()

        self.progress_widget.set_frame_category("labeled_frames", self.labeled_frame_list, "#1F32D7", priority=9)
        self.navigation_title_controller()

    ###################################################################################################################################################

    def display_current_frame(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame = self.plot_predictions(frame) if self.dlc_data is not None else frame
                # Convert OpenCV image to QPixmap
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                # Scale pixmap to fit label
                scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
                self.video_label.setText("")
                self.progress_widget.set_current_frame(self.current_frame_idx) # Update slider handle's position
            else:
                self.video_label.setText("Error: Could not read frame")
        else:
            self.video_label.setText("No video loaded")

    def plot_predictions(self, frame):
        if self.dlc_data is None:
            return frame
        try:
            if self.current_frame_idx in self.labeled_frame_list: # Use labeled data first
                current_frame_data = self.label_data_array[self.current_frame_idx,:,:]
            else:
                current_frame_data = self.dlc_data.pred_data_array[self.current_frame_idx,:,:]
        except IndexError:
            print(f"Frame index {self.current_frame_idx} out of bounds for prediction data.")
            return frame
        colors_rgb = [(0, 165, 255), (51, 255, 51), (255, 153, 51), (51, 51, 255), (102, 255, 255)]
        num_keypoint = self.dlc_data.num_keypoint
        num_instance = self.dlc_data.instance_count

        if num_keypoint != current_frame_data.size // num_instance // 3:
            QMessageBox.warning(
                self, "Error: Keypoint Mismatch",
                "Keypoints in config and in prediction do not match!\n"
                f"Keypoints in config: {self.dlc_data.keypoints} \n"
                f"Keypoints in prediction: {current_frame_data.size // num_instance * 2 // 3}"
                )
            return
        
        for inst in range(num_instance):
            color = colors_rgb[inst % len(colors_rgb)]

            # Initiate an empty dict for storing coordinates
            keypoint_coords = {}
            for i in range(num_keypoint):  # x, y, confidence triplet
                x = current_frame_data[inst, i * 3]
                y = current_frame_data[inst, i * 3 + 1]
                confidence = current_frame_data[inst, i * 3 + 2]

                keypoint = self.dlc_data.keypoints[i]

                if pd.isna(x) or pd.isna(y) or confidence <= self.confidence_cutoff: # Apply confidence cutoff
                    keypoint_coords[keypoint] = None
                    continue # Skip plotting empty coords
                else:
                    keypoint_coords[keypoint] = (int(x),int(y)) # int required by cv2
                
                cv2.circle(frame, (int(x), int(y)), 3, color, -1) # Draw the dot representing the keypoints

            if self.dlc_data.individuals is not None and len(keypoint_coords) >= 2:
                self.plot_bounding_box(keypoint_coords, frame, color, inst)
            if self.dlc_data.skeleton:
                self.plot_skeleton(keypoint_coords, frame, color)

        return frame
    
    def plot_bounding_box(self, keypoint_coords, frame, color, inst):
        # Calculate bounding box coordinates
        x_coords = [keypoint_coords[p][0] for p in keypoint_coords if keypoint_coords[p] is not None]
        y_coords = [keypoint_coords[p][1] for p in keypoint_coords if keypoint_coords[p] is not None]

        if not x_coords or not y_coords: # Skip if the mice has no keypoint
            return frame
            
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1] - 1, max_x + padding)
        max_y = min(frame.shape[0] - 1, max_y + padding)
            
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 1) # Draw the bounding box

        # Add individual label
        cv2.putText(frame, f"Instance: {self.dlc_data.individuals[inst]}", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        center_x = min_x + max_x / 2
        center_y = min_y + max_y / 2

        # Plot keypoint labels
        text_size = 0.3
        text_color = color

        for keypoint in keypoint_coords:
            if keypoint_coords[keypoint] is None: # Skip empty keypoint
                return

            (x, y) = keypoint_coords[keypoint]
            keypoint_label = keypoint
            
            # Calculate vector from mouse center to keypoint
            vec_x = x - center_x
            vec_y = y - center_y
            
            # Normalize vector
            norm = (vec_x**2 + vec_y**2)**0.5
            if norm == 0: # Avoid division by zero if keypoint is at the center
                norm = 1
            unit_vec_x = vec_x / norm
            unit_vec_y = vec_y / norm

            # Position text label away from keypoint and center
            offset_distance = 20 # Distance from keypoint to text label
            text_x = x + unit_vec_x * offset_distance
            text_y = y + unit_vec_y * offset_distance

            cv2.putText(frame, str(keypoint_label), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1, cv2.LINE_AA) # Add the label

        return frame
    
    def plot_skeleton(self, keypoint_coords, frame, color):
        for start_kp, end_kp in self.dlc_data.skeleton:
            start_coord = keypoint_coords.get(start_kp)
            end_coord = keypoint_coords.get(end_kp)
            if start_coord and end_coord:
                cv2.line(frame, start_coord, end_coord, color, 2)
        return frame
            
    ###################################################################################################################################################

    def change_frame(self, delta):
        if self.cap and self.cap.isOpened():
            new_frame_idx = self.current_frame_idx + delta
            if 0 <= new_frame_idx < self.total_frames:
                self.current_frame_idx = new_frame_idx
                self.display_current_frame()
                self.navigation_title_controller()

    def navigation_title_controller(self):
        title_text = f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1} | Video: {self.video_name}"
        if self.frame_list:
            title_text += f" | Marked Frame Count: {len(self.frame_list)}"
        self.nav_widget.setTitle(title_text)
        if self.current_frame_idx in self.labeled_frame_list:
            self.nav_widget.setTitleColor("#1F32D7")  # Blue
        elif self.current_frame_idx in self.refined_frame_list:
            self.nav_widget.setTitleColor("#009979")  # Teal/Green
        elif self.current_frame_idx in self.frame_list:
            self.nav_widget.setTitleColor("#E28F13")  # Amber/Orange
        else:
            self.nav_widget.setTitleColor("black")

    ###################################################################################################################################################

    def toggle_frame_status(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        if self.current_frame_idx in self.labeled_frame_list:
            QMessageBox.information(self, "Already Labeled", 
                "The frame is already in the labeled dataset, skipping...")
            return

        if self.current_frame_idx in self.refined_frame_list:
            reply = QMessageBox.question(
                self, "Confirm Unmarking",
                "This frame is already refined, do you still want to remove it from the exported lists>",
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

        self.determine_save_status()
        self.progress_widget.set_frame_category("marked_frames", self.frame_list, "#E28F13")
        self.navigation_title_controller()

    def _navigate_marked_frames(self, mode):
        dugh.navigate_to_marked_frame(self, self.frame_list, self.current_frame_idx, self._handle_frame_change_from_comp, mode)

    ###################################################################################################################################################

    def show_confidence_dialog(self):
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        if not self.dlc_data:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        dialog = Adjust_Property_Dialog(
            property_name="Confidence Cutoff", property_val=self.confidence_cutoff, range=(0.00, 1.00), parent=self)
        dialog.property_changed.connect(self._update_application_cutoff)
        dialog.show() # .show() instead of .exec() for a non-modal dialog

    def _update_application_cutoff(self, new_cutoff):
        self.confidence_cutoff = new_cutoff
        self.display_current_frame() # Redraw with the new cutoff

    ###################################################################################################################################################

    def _handle_refined_frames_exported(self, refined_frames):
        self.refined_frame_list = refined_frames
        self.progress_widget.set_frame_category("refined_frames", self.refined_frame_list, "#009979", priority=7)
        self.display_current_frame()
        self.determine_save_status()

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.current_frame_idx = new_frame_idx
        self.display_current_frame()
        self.navigation_title_controller()

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
        if not self.labeled_frame_list:
            QMessageBox.information(self, "No Previous Label Loaded", "Can't merge when no previous label is loaded.")
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
                'prediction': self.dlc_data.prediction_filepath, 'refined_frame_list': self.refined_frame_list}
        else:
            save_yaml = {'video_path': self.video_file,  'frame_list': self.last_saved, 'dlc_config': None,
                'prediction': None, 'refined_frame_list': self.refined_frame_list}
        output_filepath = os.path.join(os.path.dirname(self.video_file), f"{self.video_name}_frame_extractor.yaml")

        with open(output_filepath, 'w') as file:
            yaml.dump(save_yaml, file)
            
        QMessageBox.information(self, "Success", f"Current workplace files have been saved to {output_filepath}")
        return True

    def call_refiner(self, track_only=False):
        from dlc_track_refiner import DLC_Track_Refiner
        if not self.video_file:
            QMessageBox.warning(self, "Video Not Loaded", "No video is loaded, load a video first!")
            return
        if not self.dlc_data:
            QMessageBox.information(self, "DLC Data Not Loaded", "No DLC data has been loaded, load them to export to Refiner.")
            return
        try:
            self.refiner_window = DLC_Track_Refiner()
            self.refiner_window.video_file = self.video_file
            self.refiner_window.initialize_loaded_video()
            self.refiner_window.dlc_data = self.dlc_data
            if not track_only:
                self.refiner_window.marked_roi_frame_list = self.frame_list
                self.refiner_window.refined_roi_frame_list = self.refined_frame_list
            self.refiner_window.current_frame_idx = self.current_frame_idx
            self.refiner_window.prediction = self.dlc_data.prediction_filepath
            self.refiner_window.initialize_loaded_data()
            self.refiner_window.display_current_frame()
            self.refiner_window.navigation_title_controller()
            if self.frame_list and not track_only:
                self.refiner_window.direct_keypoint_edit()
                self.refiner_window.refined_frames_exported.connect(self._handle_refined_frames_exported)
            self.refiner_window.show()
            self.refiner_window.prediction_saved.connect(self.reload_prediction) # Reload from prediction provided by Refiner
            
        except Exception as e:
            QMessageBox.warning(self, "Refiner Failed", f"Refiner failed to initialize. Exception: {e}")
            return

    def reload_prediction(self, prediction_path):
        """Reload prediction data from file and update visualization"""
        self.data_loader.prediction_filepath = prediction_path
        self.dlc_data, msg = self.data_loader.load_data()
        self.display_current_frame()
        QMessageBox.information(self, "Success", msg)

        if hasattr(self, 'refiner_window') and self.refiner_window: # Clean refiner windows
            self.refiner_window.close()
            self.refiner_window = None

    def save_to_dlc(self):
        if not self.pre_saving_sanity_check():
            return

        self.save_workspace()

        self.exp_set.export_mode = "Append"
        exporter = DLC_Exporter(self.dlc_data, self.exp_set, self.frame_list)

        if not self.dlc_data:
            reply = QMessageBox.question(
                self,
                "No Prediction Loaded",
                "No prodiction has been loaded. Would you like export frames only?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                QMessageBox.information(self, "Frame Only Mode", "Choose the directory of DLC project to save to DLC.")
                dlc_dir = QFileDialog.getExistingDirectory(
                            self, "Select Project Folder",
                            os.path.dirname(self.video_file),
                            QFileDialog.ShowDirsOnly
                        )
                if not dlc_dir: # When user close the file selection window
                    return
                self.project_dir = os.path.join(dlc_dir, "labeled-data", self.video_name)
                dugh.export_and_show_message(self, exporter, frame_only=True)
                return
            else:
                self.load_prediction()
                if self.dlc_data is None:
                    return

        dugh.export_and_show_message(self, exporter, frame_only=True)

    def merge_data(self):
        if not self.pre_saving_sanity_check():
            return
        
        if not self.refined_frame_list:
            QMessageBox.warning(self, "No Refined Frame", "No frame has been refined, please refine some marked frames first.")
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

            self.exp_set.export_mode = "Merge"

            self.label_data_array[self.refined_frame_list, :, :] = self.dlc_data.pred_data_array[self.refined_frame_list, :, :]
            merge_frame_list = list(set(self.labeled_frame_list) | set(self.refined_frame_list))
            label_data_array_with_conf = self.label_data_array[merge_frame_list, :, :]
            label_data_array_export = duh.remove_mock_confidence_score(label_data_array_with_conf)

            exporter = DLC_Exporter(self.dlc_data, self.exp_set, merge_frame_list, label_data_array_export)
            dugh.export_and_show_message(self, exporter, frame_only=False)

            self.process_labeled_frame()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.display_current_frame()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        dugh.handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_workspace)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DLC_Extractor()
    window.show()
    app.exec()