import os

import h5py
import yaml

import pandas as pd
import numpy as np
from itertools import islice
import bisect

import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QPushButton

#################   W   ##################   I   ##################   P   ##################   

DLC_CONFIG_DEBUG = "D:/Project/DLC-Models/NTD/config.yaml"
VIDEO_FILE_DEBUG = "D:/Project/DLC-Models/NTD/videos/jobs/20250626C1-first3h-conv/20250626C1-first3h-D.mp4"
PRED_FILE_DEBUG = "D:/Project/DLC-Models/NTD/videos/jobs/20250626C1-first3h-conv/20250626C1-first3h-DDLC_HrnetW32_bezver-SD-20250605M-cam52025-06-26shuffle1_detector_090_snapshot_080_el.h5"

class DLC_Track_Refiner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Track Refiner")
        self.setGeometry(100, 100, 1200, 960)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.load_video_button = QPushButton("Load Video")
        self.load_DLC_config_button = QPushButton("Load DLC Config")
        self.load_prediction_button = QPushButton("Load Prediction")
        self.save_prediction_button = QPushButton("Save Prediction")

        self.button_layout.addWidget(self.load_video_button)
        self.button_layout.addWidget(self.load_DLC_config_button)
        self.button_layout.addWidget(self.load_prediction_button)
        self.button_layout.addWidget(self.save_prediction_button)
        self.layout.addLayout(self.button_layout)

        # Video display area
        self.video_label = QtWidgets.QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label, 1)

        # Progress bar
        self.progress_layout = QtWidgets.QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.undo_button = QPushButton("⮌")
        self.undo_button.setFixedWidth(20)
        self.redo_button = QPushButton("⮎")
        self.redo_button.setFixedWidth(20)
        self.progress_slider = Slider_With_Marks(Qt.Horizontal)
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        self.progress_layout.addWidget(self.undo_button)
        self.progress_layout.addWidget(self.redo_button)
        self.playback_timer = QTimer()
        # self.playback_timer.timeout.connect(self.autoplay_video)
        self.is_playing = False
        self.layout.addLayout(self.progress_layout)

        # Navigation controls
        self.navigation_group_box = QtWidgets.QGroupBox("Video Navigation")
        self.navigation_layout = QtWidgets.QGridLayout(self.navigation_group_box)
        self.prev_10_frames_button = QPushButton("Prev 10 Frames (Shift + ←)")
        self.prev_frame_button = QPushButton("Prev Frame (←)")
        self.next_frame_button = QPushButton("Next Frame (→)")
        self.next_10_frames_button = QPushButton("Next 10 Frames (Shift + →)")

        self.prev_instance_change_button = QPushButton("◄ Prev ROI (↓)")
        self.next_instance_change_button = QPushButton("► Next ROI (↑)")
        self.swap_track_button = QPushButton("Swap Track")
        self.delete_track_button = QPushButton("Delete Track")

        self.navigation_layout.addWidget(self.prev_10_frames_button, 0, 0)
        self.navigation_layout.addWidget(self.prev_frame_button, 0, 1)
        self.navigation_layout.addWidget(self.next_frame_button, 0, 2)
        self.navigation_layout.addWidget(self.next_10_frames_button, 0, 3)

        self.navigation_layout.addWidget(self.prev_instance_change_button, 1, 1)
        self.navigation_layout.addWidget(self.next_instance_change_button, 1, 2)
        self.navigation_layout.addWidget(self.swap_track_button, 1, 0)
        self.navigation_layout.addWidget(self.delete_track_button, 1, 3)

        self.layout.addWidget(self.navigation_group_box)

        # Connect buttons to events
        self.load_video_button.clicked.connect(self.load_video)
        self.load_DLC_config_button.clicked.connect(self.load_DLC_config)
        self.load_prediction_button.clicked.connect(self.load_prediction)
        self.save_prediction_button.clicked.connect(self.save_prediction)

        self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.play_button.clicked.connect(self.toggle_playback)
        self.undo_button.clicked.connect(self.undo_changes)
        self.redo_button.clicked.connect(self.redo_changes)

        self.prev_10_frames_button.clicked.connect(lambda: self.change_frame(-10))
        self.prev_frame_button.clicked.connect(lambda: self.change_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self.change_frame(1))
        self.next_10_frames_button.clicked.connect(lambda: self.change_frame(10))

        self.prev_instance_change_button.clicked.connect(self.prev_instance_change)
        self.next_instance_change_button.clicked.connect(self.next_instance_change)
        self.swap_track_button.clicked.connect(self.swap_track)
        self.delete_track_button.clicked.connect(self.delete_track)

        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_W), self).activated.connect(self.swap_track)
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self.delete_track)
        QShortcut(QKeySequence(Qt.Key_Z | Qt.ShiftModifier), self).activated.connect(self.undo_changes)
        QShortcut(QKeySequence(Qt.Key_Y | Qt.ShiftModifier), self).activated.connect(self.redo_changes)
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self.prev_instance_change)
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self.next_instance_change)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_frame_mark)

        self.reset_state()
        self.is_debug = True

    def load_video(self):
        if self.is_debug:
            self.original_vid = VIDEO_FILE_DEBUG
            self.initialize_loaded_video()
            self.load_dlc_file(DLC_CONFIG_DEBUG)
            self.load_prediction(PRED_FILE_DEBUG)
            return
        self.reset_state()
        file_dialog = QtWidgets.QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if video_path:
            self.original_vid = video_path
            self.initialize_loaded_video()
            
    def initialize_loaded_video(self):
        self.navigation_group_box.show()
        self.video_name = os.path.basename(self.original_vid).split(".")[0]
        self.cap = cv2.VideoCapture(self.original_vid)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.original_vid}")
            self.video_label.setText("Error: Could not open video")
            self.cap = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.progress_slider.setRange(0, self.total_frames - 1) # Initialize slider range
        self.progress_slider.set_marked_frames(self.frame_list) # Update marked frames
        self.progress_slider.set_labeled_frames(self.labeled_frame_list)
        self.display_current_frame()
        self.navigation_box_title_controller()
        print(f"Video loaded: {self.original_vid}")

    def load_dlc_file(self, dlc_config=None):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        if dlc_config is None:
            file_dialog = QtWidgets.QFileDialog(self)
            dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        self.dlc_dir = os.path.dirname(dlc_config)
        with open(dlc_config, "r") as conf:
            cfg = yaml.safe_load(conf)
        self.multi_animal = cfg["multianimalproject"]
        self.keypoints = cfg["bodyparts"] if not self.multi_animal else cfg["multianimalbodyparts"]
        self.skeleton = cfg["skeleton"]
        self.individuals = cfg["individuals"]
        self.instance_count = len(self.individuals) if self.individuals is not None else 1
        self.project_dir = os.path.join(self.dlc_dir,"labeled-data", self.video_name)
        self.display_current_frame()

    def load_prediction(self, prediction_path=None):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        if self.dlc_dir is None:
            QMessageBox.warning(self, "No DLC Config", "No dlc config has been loaded, please load it first.")
            return
        if prediction_path is None:
            file_dialog = QtWidgets.QFileDialog(self)
            prediction_path, _ = file_dialog.getOpenFileName(self, "Load Prediction", "", "HDF5 Files (*.h5);;All Files (*)")
        self.prediction = prediction_path
        print(f"Prediction loaded: {self.prediction}")
        with h5py.File(self.prediction, "r") as pred_file:
            if not "tracks" in pred_file.keys():
                print("Error: Prediction file not valid, no 'tracks' key found in prediction file.")
                return False
            self.pred_data = pred_file["tracks"]["table"][:]
            pred_data_values = np.array([item[1] for item in pred_data_raw])
            pred_frame_count = self.pred_data.size
            self.pred_data_array = np.full((pred_frame_count, self.instance_count, len(self.keypoints)),np.nan)
            if pred_frame_count != self.total_frames:
                QMessageBox.warning(self, "Error: Frame Mismatch", "Total frames in video and in prediction do not match!")
                print(f"Frames in config: {self.total_frames} \n Frames in prediction: {pred_frame_count}")
            self.display_current_frame()

    ###################################################################################################################################################

    def display_current_frame(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame = self.plot_predictions(frame) if self.pred_data is not None else frame
                # Convert OpenCV image to QPixmap
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                # Scale pixmap to fit label
                scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
                self.video_label.setText("") # Clear "No video loaded" text
                self.progress_slider.setValue(self.current_frame_idx) # Update slider position
            else:
                self.video_label.setText("Error: Could not read frame")
        else:
            self.video_label.setText("No video loaded")

    def plot_predictions(self, frame):
        if self.pred_data is None:
            return frame
        try:
            current_frame_data = self.pred_data[self.current_frame_idx][1]
        except IndexError:
            print(f"Frame index {self.current_frame_idx} out of bounds for prediction data.")
            return frame
        colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)] # BGR
        if self.keypoints is None: 
            num_keypoints = current_frame_data.size // self.instance_count // 3 # Consider the confidence col
        elif len(self.keypoints) != current_frame_data.size // self.instance_count // 3:
            QMessageBox.warning(self, "Error: Keypoint Mismatch", "Keypoints in config and in prediction do not match! Falling back to prediction parameters!")
            print(f"Keypoints in config: {len(self.keypoints)} \n Keypoints in prediction: {current_frame_data.size // self.instance_count * 2 // 3}")
            self.keypoints = None   # Falling back to prediction parameters
            self.skeleton = None
            num_keypoints = current_frame_data.size // self.instance_count // 3
        else:
            num_keypoints = len(self.keypoints)

        # Iterate over each individual (animal)
        for inst in range(self.instance_count):
            color = colors[inst % len(colors)]
            # Initiate an empty dict for storing coordinates
            keypoint_coords = {}
            for i in range(num_keypoints):
                x = current_frame_data[inst * num_keypoints * 3 + i * 3] # x, y, confidence triplet
                y = current_frame_data[inst * num_keypoints * 3 + i * 3 + 1]
                confidence = current_frame_data[inst * num_keypoints * 3 + i * 3 + 2]
                if pd.isna(confidence):
                    confidence = 0 # Set confidence value to 0 for NaN confidence
                keypoint = i if self.keypoints is None else self.keypoints[i]
                text_size = 0.5 if self.keypoints is None else 0.3
                text_color = color if self.keypoints is None else (255, 255, 86)
                if pd.isna(x) or pd.isna(y) or confidence <= self.confidence_cutoff: # Apply confidence cutoff
                    keypoint_coords[keypoint] = None
                    continue # Skip plotting empty coords
                else:
                    keypoint_coords[keypoint] = (int(x),int(y))
                
                cv2.circle(frame, (int(x), int(y)), 3, color, -1) # Draw the dot representing the keypoints
                cv2.putText(frame, str(keypoint), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1, cv2.LINE_AA) # Add the label

            if self.individuals is not None and len(keypoint_coords) >= 2:
                self.plot_bounding_box(keypoint_coords, frame, color, inst)
            if self.skeleton:
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

        #Add individual label
        cv2.putText(frame, f"Instance: {self.individuals[inst]}", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        return frame
    
    def plot_skeleton(self, keypoint_coords, frame, color):
        for start_kp, end_kp in self.skeleton:
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
                self.navigation_box_title_controller()

    def set_frame_from_slider(self, value):
        if self.cap and self.cap.isOpened():
            self.current_frame_idx = value
            self.display_current_frame()
            self.navigation_box_title_controller()

    def autoplay_video(self):
        if self.cap and self.cap.isOpened():
            if self.current_frame_idx < self.total_frames - 1:
                self.current_frame_idx += 1
                self.display_current_frame()
                self.navigation_box_title_controller()
            else:
                self.playback_timer.stop()
                self.play_button.setText("▶")
                self.is_playing = False

    def toggle_playback(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        if not self.is_playing:
            self.playback_timer.start(1000/100) # 100 fps
            self.play_button.setText("■")
            self.is_playing = True
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

    def navigation_box_title_controller(self):
        self.navigation_group_box.setTitle(f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1} | Video: {self.video_name}")
        if self.current_frame_idx in self.frame_list:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: #F04C4C;}""")
        else:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: black;}""")

    ###################################################################################################################################################




    def reset_state(self):
        self.original_vid, self.prediction, self.dlc_config, self.video_name = None, None, None, None
        self.keypoints, self.skeleton, self.individuals, self.project_dir = None, None, None, None

        self.instance_count = 1
        self.multi_animal = False
        self.pred_data = None

        self.labeled_frame_list, self.frame_list = [], []

        self.cap, self.current_frame = None, None

        self.is_playing = False
        self.is_saved = True
        self.last_saved = []

        self.progress_slider.setRange(0, 0)
        self.navigation_group_box.hide()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DLC_Track_Refiner()
    window.show()
    app.exec()