import os

import pandas as pd
from itertools import islice
import bisect

import h5py
import yaml

import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox

class dlcFrameFinder(QtWidgets.QMainWindow):  # GUI for manually select the frames
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Manual Frame Extractor")
        self.setGeometry(100, 100, 1200, 960)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.load_video_button = QtWidgets.QPushButton("Load Video")
        self.load_prediction_button = QtWidgets.QPushButton("Load Prediction")
        self.load_marked_frames_button = QtWidgets.QPushButton("Load Marked Frames")
        self.save_to_dlc_button = QtWidgets.QPushButton("Save to DLC")

        self.button_layout.addWidget(self.load_video_button)
        self.button_layout.addWidget(self.load_prediction_button)
        self.button_layout.addWidget(self.load_marked_frames_button)
        self.button_layout.addWidget(self.save_to_dlc_button)

        self.layout.addLayout(self.button_layout)

        # Video display area
        self.video_label = QtWidgets.QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label, 1)

        # Progress bar
        self.progress_layout = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.progress_slider = custom_slider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0) # Will be set dynamically
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.autoplay_video)
        self.is_playing = False
        self.layout.addLayout(self.progress_layout)

        # Navigation controls
        self.navigation_group_box = QtWidgets.QGroupBox("Video Navigation")
        self.navigation_layout = QtWidgets.QGridLayout(self.navigation_group_box)

        self.prev_10_frames_button = QtWidgets.QPushButton("Prev 10 Frames (Shift + ←)")
        self.prev_frame_button = QtWidgets.QPushButton("Prev Frame (←)")
        self.next_frame_button = QtWidgets.QPushButton("Next Frame (→)")
        self.next_10_frames_button = QtWidgets.QPushButton("Next 10 Frames (Shift + →)")

        self.prev_marked_frame_button = QtWidgets.QPushButton("◄ Prev Marked (↓)")
        self.next_marked_frame_button = QtWidgets.QPushButton("► Next Marked (↑)")
        self.mark_frame_button = QtWidgets.QPushButton("Mark / Unmark Current Frame (X)")
        self.check_current_mark_button = QtWidgets.QPushButton("Check & Save Current Marked")

        self.navigation_layout.addWidget(self.prev_10_frames_button, 0, 0)
        self.navigation_layout.addWidget(self.prev_frame_button, 0, 1)
        self.navigation_layout.addWidget(self.next_frame_button, 0, 2)
        self.navigation_layout.addWidget(self.next_10_frames_button, 0, 3)

        self.navigation_layout.addWidget(self.mark_frame_button, 1, 0)
        self.navigation_layout.addWidget(self.prev_marked_frame_button, 1, 1)
        self.navigation_layout.addWidget(self.next_marked_frame_button, 1, 2)
        self.navigation_layout.addWidget(self.check_current_mark_button, 1, 3)

        self.layout.addWidget(self.navigation_group_box)
        self.navigation_group_box.hide()

        # Connect buttons to events
        self.load_video_button.clicked.connect(self.load_video)
        self.load_prediction_button.clicked.connect(self.load_prediction)
        self.load_marked_frames_button.clicked.connect(self.load_marked_frames)

        self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.play_button.clicked.connect(self.toggle_playback)

        self.prev_10_frames_button.clicked.connect(lambda: self.change_frame(-10))
        self.prev_frame_button.clicked.connect(lambda: self.change_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self.change_frame(1))
        self.next_10_frames_button.clicked.connect(lambda: self.change_frame(10))

        self.prev_marked_frame_button.clicked.connect(self.prev_marked_frame)
        self.next_marked_frame_button.clicked.connect(self.next_marked_frame)
        self.mark_frame_button.clicked.connect(self.change_current_frame_status)
        self.check_current_mark_button.clicked.connect(self.display_marked_frames_list)

        self.save_to_dlc_button.clicked.connect(self.save_to_dlc)

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self.change_current_frame_status)
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self.prev_marked_frame)
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self.next_marked_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_L | Qt.ShiftModifier), self).activated.connect(self.load_dlc_file)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_frame_mark)
        
        self.original_vid = None
        self.prediction = None
        self.dlc_dir = None
        self.video_name = None

        self.pred_data = None

        self.multi_animal = False
        self.keypoints = None
        self.skeleton = None
        self.individuals = None
        self.instance_count = 1
        self.project_dir = None
        self.labeled_frame_list = []

        self.cap = None
        self.current_frame = None
        self.frame_list = []

        self.is_saved = True
        self.last_saved = []

    def load_video(self):
        self.reset_state()
        file_dialog = QtWidgets.QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if video_path:
            self.original_vid = video_path
            self.initialize_loaded_video()
            self.navigation_group_box.show()

    def initialize_loaded_video(self):
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

    def load_prediction(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        file_dialog = QtWidgets.QFileDialog(self)
        prediction_path, _ = file_dialog.getOpenFileName(self, "Load Prediction", "", "HDF5 Files (*.h5);;All Files (*)")
        if prediction_path:
            self.prediction = prediction_path
            print(f"Prediction loaded: {self.prediction}")
            if self.dlc_dir is None:
                self.dlc_loader_query()
            with h5py.File(self.prediction, "r") as pred_file:
                if not "tracks" in pred_file.keys():
                    print("Error: Prediction file not valid, no 'tracks' key found in prediction file.")
                    return False
                elif not "table" in pred_file["tracks"].keys():
                    print("Errpr: Prediction file not valid, no prediction table found in 'tracks'.")
                    return False
                self.pred_data = pred_file["tracks"]["table"][:]
                pred_frame_count = self.pred_data.size
                if pred_frame_count != self.total_frames:
                    QMessageBox.warning(self, "Error: Frame Mismatch", "Total frames in video and in prediction do not match!")
                    print(f"Frames in config: {self.total_frames} \n Frames in prediction: {pred_frame_count}")
                self.display_current_frame()

    def dlc_loader_query(self):
        dlc_loader = QMessageBox(self)
        dlc_loader.setWindowTitle("Load DLC Config? (optional)")
        dlc_loader.setText("Do you want to also load DLC config? (necessary for displaying keypoint labels, skeleton, and more)")
        okay_btn = dlc_loader.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
        no_btn = dlc_loader.addButton("No", QMessageBox.RejectRole)
        dlc_loader.setDefaultButton(no_btn)
        dlc_loader.exec()
        clicked_button = dlc_loader.clickedButton()
        if clicked_button == okay_btn:
            self.load_dlc_file()
        else:
            QMessageBox.information(self, "OK", "No DLC config has been loaded, press Shift + L if you change your mind.")

    def load_dlc_file(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        file_dialog = QtWidgets.QFileDialog(self)
        dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        if dlc_config:
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
            if not os.path.isdir(self.project_dir):
                self.labeled_frame_list = []
                return
            label_data_files = [ f for f in os.listdir(self.project_dir) if f.startswith("CollectedData_") and f.endswith(".h5") ]
            if not label_data_files:
                return
            label_frame_list = []
            for label_data_file in label_data_files:
                with h5py.File(os.path.join(self.project_dir, label_data_file), "r") as lbf:
                    label_frame_list.extend(lbf["df_with_missing"]["axis1_level2"].asstr()[()])
                continue
            self.labeled_frame_list = [ int(f.split("img")[1].split(".")[0]) for f in label_frame_list ]
            self.display_current_frame()

    def load_marked_frames(self):
        file_dialog = QtWidgets.QFileDialog(self)
        marked_frame_path, _ = file_dialog.getOpenFileName(self, "Load Marked Frames", "", "YAML Files (*.yaml);;All Files (*)")
        if marked_frame_path:
            with open(marked_frame_path, "r") as fmkf:
                fmk = yaml.safe_load(fmkf)
            if not "frame_list" in fmk.keys():
                QMessageBox.warning(self, "File Error", "Not a marked frame file, make sure to load the correct file.")
                return
            self.frame_list = fmk["frame_list"]
            print(f"Marked frames loaded: {self.frame_list}")
            if self.original_vid is None:
                self.original_vid = fmk["video_path"]
                self.initialize_loaded_video()
            else:
                self.progress_slider.set_marked_frames(self.frame_list)
            self.determine_save_status()

    def change_current_frame_status(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        if not self.current_frame_idx in self.frame_list:
            self.frame_list.append(self.current_frame_idx)
        else: # Remove the mark status if already marked
            self.frame_list.remove(self.current_frame_idx)
        self.determine_save_status()
        self.progress_slider.set_marked_frames(self.frame_list)
        self.navigation_box_title_controller()

    def display_marked_frames_list(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        if self.frame_list:
            self.frame_list.sort()
            frame_list_msg = QMessageBox(self)
            frame_list_msg.setWindowTitle("Current Marked Frames")
            frame_list_msg.setText(str(self.frame_list))
            save_btn = frame_list_msg.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
            close_btn = frame_list_msg.addButton("Close", QMessageBox.RejectRole)
            frame_list_msg.setDefaultButton(close_btn)
            frame_list_msg.exec()
            clicked_button = frame_list_msg.clickedButton()
            if clicked_button == save_btn:
                self.save_frame_mark()
        else:
            QMessageBox.information(self, "No Marked Frames", "There are no frames marked yet.")

    def prev_marked_frame(self):
        if not self.frame_list:
            QMessageBox.information(self, "No Marked Frames", "No marked frames to navigate.")
            return
        
        self.frame_list.sort()
        try:
            current_idx_in_marked = self.frame_list.index(self.current_frame_idx) - 1
        except ValueError:
            # Current frame is not marked, find the closest previous marked frame
            current_idx_in_marked = bisect.bisect_left(self.frame_list, self.current_frame_idx) - 1

        if current_idx_in_marked >= 0:
            self.current_frame_idx = self.frame_list[current_idx_in_marked]
            self.display_current_frame()
            self.current_marked_idx_in_list = current_idx_in_marked
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No previous marked frame found.")

    def next_marked_frame(self):
        if not self.frame_list:
            QMessageBox.information(self, "No Marked Frames", "No marked frames to navigate.")
            return
        self.frame_list.sort()
        try:
            current_idx_in_marked = self.frame_list.index(self.current_frame_idx) + 1
        except ValueError:
            # Current frame is not marked, find the closest next marked frame
            current_idx_in_marked = bisect.bisect_right(self.frame_list, self.current_frame_idx)

        if current_idx_in_marked < len(self.frame_list):
            self.current_frame_idx = self.frame_list[current_idx_in_marked]
            self.display_current_frame()
            self.current_marked_idx_in_list = current_idx_in_marked
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No next marked frame found.")

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
                x = current_frame_data[inst * num_keypoints * 3 + i * 3] # every third col in confidence col lol
                y = current_frame_data[inst * num_keypoints * 3 + i * 3 + 1]
                keypoint = i if self.keypoints is None else self.keypoints[i]
                text_size = 0.5 if self.keypoints is None else 0.3
                text_color = color if self.keypoints is None else (255, 255, 86)
                if pd.isna(x) or pd.isna(y):
                    keypoint_coords[keypoint] = None
                    continue # Skip plotting empty coords
                else:
                    keypoint_coords[keypoint] = (int(x),int(y))
                
                cv2.circle(frame, (int(x), int(y)), 3, color, -1) # Draw the dot
                cv2.putText(frame, str(keypoint), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1, cv2.LINE_AA) # Add the label

            if self.skeleton: # Draw the skeleton
                for start_kp, end_kp in self.skeleton:
                    start_coord = keypoint_coords.get(start_kp)
                    end_coord = keypoint_coords.get(end_kp)
                    if start_coord and end_coord:
                        cv2.line(frame, start_coord, end_coord, color, 2)
        return frame

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
        if self.current_frame_idx in self.labeled_frame_list:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: #1F32D7;}""")
        elif self.current_frame_idx in self.frame_list:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: #E28F13;}""")
        else:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: black;}""")

    def determine_save_status(self):
        if set(self.last_saved) == set(self.frame_list):
            self.is_saved = True
        else:
            self.is_saved = False

    def save_frame_mark(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        self.last_saved = self.frame_list
        self.is_saved = True
        save_yaml = {'video_path': self.original_vid, 'frame_list': self.last_saved}
        output_filepath = os.path.join(os.path.dirname(self.original_vid), f"{self.video_name}_frame_list.yaml")
        with open(output_filepath, 'w') as file:
            yaml.dump(save_yaml, file)
        QMessageBox.information(self, "Success", f"Marked frames have been saved to {output_filepath}")

    def save_to_dlc(self):
        frame_only_mode = False
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return False
        if not self.frame_list:
            QMessageBox.warning(self, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return False
        if self.dlc_dir is None:
            QMessageBox.warning(self, "No DLC Config", "DLC config is required for saving to DLC.")
            self.load_dlc_file()
            if self.dlc_dir is None: # When user close the file selection window
                return False
        if self.prediction is None:
            reply = QMessageBox.question(
                self,
                "No Prediction Loaded",
                "No prodiction has been loaded. Would you like export frames only?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                frame_only_mode = True
            else:
                self.load_prediction()
                if self.prediction is None:
                    return False       
        try:
            # Initialize DLC extractor backend
            extractor = dlcFrameExtractor(
                self.original_vid,
                self.prediction,
                self.frame_list,
                self.dlc_dir,
                self.project_dir,
                self.video_name,
                self.pred_data,
                self.keypoints,
                self.individuals,
                multi_animal=self.multi_animal
            )
            # Perform the extraction
            os.makedirs(self.project_dir, exist_ok=True)
            if not frame_only_mode:
                success = extractor.extract_frame_and_label()
            else:
                success = extractor.extract_frame()        
            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully saved {len(self.frame_list)} frames to:\n"
                    f"{os.path.join(self.dlc_dir, 'labeled-data', self.video_name)}"
                )
                return True
            else:
                QMessageBox.warning(self, "Error", "Failed to save frames to DLC format.")
                return False
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while saving to DLC:\n{str(e)}"
            )
            return False

    def reset_state(self):
        self.progress_slider.setRange(0, 0)
        self.navigation_group_box.hide()
        self.original_vid = None
        self.prediction = None
        self.dlc_dir = None
        self.video_name = None
        self.multianimal = False
        self.keypoints = None
        self.skeleton = None
        self.individuals = None
        self.instance_count = 1
        self.pred_data = None
        self.labeled_folder = None
        self.labeled_frame_list = []
        self.cap = None
        self.current_frame = None
        self.frame_list = []
        self.is_saved = True
        self.last_saved = []
        self.is_playing = False

    def closeEvent(self, event: QCloseEvent):
        if not self.is_saved:
            # Create a dialog to confirm saving
            close_call = QMessageBox(self)
            close_call.setWindowTitle("Marked Frame Unsaved")
            close_call.setText("Do you want to save your changes before closing?")
            close_call.setIcon(QMessageBox.Icon.Question)

            save_btn = close_call.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = close_call.addButton("Don't Save", QMessageBox.ButtonRole.DestructiveRole)
            close_btn = close_call.addButton("Close", QMessageBox.RejectRole)
            
            close_call.setDefaultButton(close_btn)

            close_call.exec()
            clicked_button = close_call.clickedButton()
            
            if clicked_button == save_btn:
                self.save_frame_mark()
                if self.is_saved:
                    event.accept()
                else:
                    event.ignore()
            elif clicked_button == discard_btn:
                event.accept()  # Close without saving
            else:
                event.ignore()  # Cancel the close action
        else:
            event.accept()  # No unsaved changes, close normally

#########################################################################################################################################################################################

class custom_slider(QtWidgets.QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.marked_frames = set()
        self.labeled_frames = set()
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B1B1B1, stop:1 #B1B1B1);
                margin: 2px 0;
            }
            
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 10px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

    def set_marked_frames(self, marked_frames):
        self.marked_frames = set(marked_frames)
        self.update()

    def set_labeled_frames(self, labeled_frame):
        self.labeled_frames = set(labeled_frame)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.marked_frames and not self.labeled_frames:
            return

        self.paintEvent_painter(self.marked_frames,"#E28F13")
        self.paintEvent_painter(self.labeled_frames,"#1F32D7")
        
    def paintEvent_painter(self, frames, color):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # Get slider geometry
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, 
            opt, 
            QtWidgets.QStyle.SC_SliderGroove, 
            self
        )
        # Calculate available width and range
        min_val = self.minimum()
        max_val = self.maximum()
        available_width = groove_rect.width()
        # Draw each frame on slider
        for frame in frames:
            if frame < min_val or frame > max_val:
                continue  
            pos = QtWidgets.QStyle.sliderPositionFromValue(
                min_val, 
                max_val, 
                frame, 
                available_width,
                opt.upsideDown
            ) + groove_rect.left()
            # Draw marker
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(color))
            painter.drawRect(
                int(pos) - 1,  # Center the mark
                groove_rect.top(),
                3,  # Width
                groove_rect.height()
            )
        painter.end()

#########################################################################################################################################################################################

class dlcFrameExtractor:  # Backend for extracting frames for labeling in DLC
    def __init__(self, original_vid, prediction, frame_list, dlc_dir, project_dir, video_name, pred_data, keypoints, individuals, multi_animal=False):
        self.original_vid = original_vid
        self.prediction = prediction
        self.frame_list = frame_list
        self.dlc_dir = dlc_dir
        self.project_dir = project_dir
        self.video_name = video_name
        self.multi_animal = multi_animal
        self.pred_data = pred_data
        self.keypoints = keypoints
        self.individuals = individuals

        self.data = []

    def extract_frame_and_label(self):
        if not os.path.isfile(self.original_vid):
            print(f"Original video not found at {self.original_vid}")
            return False
        elif not os.path.isfile(self.prediction):
            print(f"Prediction file not found at {self.prediction}")
            return False
        else:
            frame_extraction = self.extract_frame()
            label_extraction = self.extract_label()
            if frame_extraction and label_extraction:
                print("Extraction successful.")
                return True
            else:
                fail_message = ""
                fail_message += " extract frame" if not frame_extraction else ""
                fail_message += " extract label" if not label_extraction else ""
                print(f"Extraction failed. Error:{fail_message}")
        
    def extract_frame(self):
        cap = cv2.VideoCapture(self.original_vid)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.original_vid}")
            return False
        
        frames_to_extract = set(self.frame_list)

        for frame in frames_to_extract:
            image_path = f"img{str(int(frame)).zfill(8)}.png"
            image_output_path = os.path.join(self.project_dir,image_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_output_path, frame)
        return True

    def extract_label(self):
        for frame in self.frame_list:
            frame_idx = self.pred_data[frame][0]
            frame_data = self.pred_data[frame][1]
            filtered_frame_data = [val for i, val in enumerate(frame_data) if (i % 3 != 2)] # Remove likelihood
            self.data.append([frame_idx] + filtered_frame_data)
            
        if not self.prediction_to_csv():
            print("Error exporting predictions to csv.")
            return False
        else:
            print("Prediction of selected frames successfully exported to csv.")
        if not self.csv_to_h5():
            print("Error transforming to h5.")
            return False
        else:
            print("Exported csv transformed into h5.")
        return True

    def prediction_to_csv(self): # Adapted from DeepLabCut
        data_frames = []
        columns = ["frame"]
        bodyparts_row = ["bodyparts"]
        coords_row = ["coords"]

        num_keypoints = len(self.keypoints)

        if not self.multi_animal:
            columns += ([f"{kp}_x" for kp in self.keypoints] + [f"{kp}_y" for kp in self.keypoints])
            bodyparts_row += [ f"{kp}" for kp in self.keypoints for _ in (0, 1) ]
            coords_row += (["x", "y"] * self.keypoints)
            max_instances = 1
        else:
            individuals_row = ["individuals"]
            max_instances = len(self.individuals)
            self.individuals = [str(k) for k in range(1,max_instances+1)]
            for m in range(max_instances):
                columns += ([f"{kp}_x" for kp in self.keypoints] + [f"{kp}_y" for kp in self.keypoints])
                bodyparts_row += [ f"{kp}" for kp in self.keypoints for _ in (0, 1) ]
                coords_row += (["x", "y"] * num_keypoints)
                for _ in range(num_keypoints*2):
                    individuals_row += [self.individuals[m]]
        scorer_row = ["scorer"] + ["machine-labeled"] * (len(columns) - 1)

        labels_df = pd.DataFrame(self.data, columns=columns)
        labels_df["frame"] = labels_df["frame"].apply(
            lambda x: (
                f"labeled-data/{self.video_name}/"
                f"img{str(int(x)).zfill(8)}.png"
            )
        )
        labels_df = labels_df.groupby("frame", as_index=False).first()
        data_frames.append(labels_df)
        combined_df = pd.concat(data_frames, ignore_index=True)

        header_df = pd.DataFrame(
        [row for row in [scorer_row, individuals_row, bodyparts_row, coords_row] if row != individuals_row or self.multi_animal],
            columns=combined_df.columns
        )

        final_df = pd.concat([header_df, combined_df], ignore_index=True)
        final_df.columns = [None] * len(final_df.columns)

        final_df.to_csv(
            os.path.join(self.project_dir, f"MachineLabelsRefine.csv"),
            index=False,
            header=None,
        )
        return True

    def csv_to_h5(self): # Adapted from DeepLabCut
        try:
            fn = os.path.join(self.project_dir, f"MachineLabelsRefine.csv")
            with open(fn) as datafile:
                head = list(islice(datafile, 0, 5))
            if self.multi_animal:
                header = list(range(4))
            else:
                header = list(range(3))
            if head[-1].split(",")[0] == "labeled-data":
                index_col = [0, 1, 2]
            else:
                index_col = 0
            data = pd.read_csv(fn, index_col=index_col, header=header)
            data.columns = data.columns.set_levels(["machine-labeled"], level="scorer")
            self.guarantee_multiindex_rows(data)
            data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
            data.to_csv(fn)
            return True
        except FileNotFoundError:
            print("Attention:", self.project_dir, "does not appear to have labeled data!")

    def guarantee_multiindex_rows(self, df): # Adapted from DeepLabCut
        # Make paths platform-agnostic if they are not already
        if not isinstance(df.index, pd.MultiIndex):  # Backwards compatibility
            path = df.index[0]
            try:
                sep = "/" if "/" in path else "\\"
                splits = tuple(df.index.str.split(sep))
                df.index = pd.MultiIndex.from_tuples(splits)
            except TypeError:  #  Ignore numerical index of frame indices
                pass
        # Ensure folder names are strings
        try:
            df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
        except AttributeError:
            pass

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = dlcFrameFinder()
    window.show()
    app.exec()