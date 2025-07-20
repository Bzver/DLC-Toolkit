import os
import shutil

import h5py
import yaml

import pandas as pd
import numpy as np
import bisect

import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QPainter, QColor, QPen, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QPushButton, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QMenu, QToolButton

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

        # Menu bars
        self.menu_layout = QtWidgets.QHBoxLayout()
        self.file_menu = QMenu("File", self)

        self.load_video_action = self.file_menu.addAction("Load Video")
        self.load_dlc_config_action = self.file_menu.addAction("Load DLC Config")
        self.load_prediction_action = self.file_menu.addAction("Load Prediction")
        self.save_prediction_action = self.file_menu.addAction("Save Prediction")

        self.file_button = QToolButton()
        self.file_button.setText("File")
        self.file_button.setMenu(self.file_menu)
        self.file_button.setPopupMode(QToolButton.InstantPopup)

        self.refiner_menu = QMenu("Adv. Refine", self)

        self.purge_inst_by_conf_action = self.refiner_menu.addAction("Delete All Track Below Set Confidence")
        self.precision_deletion_action = self.refiner_menu.addAction("Delete Track Between Set Frames")
        self.precision_interpolate_action = self.refiner_menu.addAction("Interpolate Track Between Set Frames")
        self.precision_fill_action = self.refiner_menu.addAction("Fill Track Between Set Frames")

        self.refiner_button = QToolButton()
        self.refiner_button.setText("Adv. Refine")
        self.refiner_button.setMenu(self.refiner_menu)
        self.refiner_button.setPopupMode(QToolButton.InstantPopup)

        self.menu_layout.addWidget(self.file_button, alignment=Qt.AlignLeft)
        self.menu_layout.addWidget(self.refiner_button, alignment=Qt.AlignLeft)
        self.menu_layout.addStretch(1)
        self.layout.addLayout(self.menu_layout)

        # Graphics view for interactive elements and video display
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setMouseTracking(True) # Enable mouse tracking for hover effects
        self.graphics_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.graphics_view.setStyleSheet("background-color: black;") # Set background for empty view
        self.layout.addWidget(self.graphics_view, 1)

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
        self.playback_timer.timeout.connect(self.autoplay_video)
        self.is_playing = False
        self.layout.addLayout(self.progress_layout)

        # Navigation controls and refiner controls
        self.control_layout = QtWidgets.QHBoxLayout()

        self.navigation_group_box = QtWidgets.QGroupBox("Video Navigation")
        self.navigation_layout = QtWidgets.QGridLayout(self.navigation_group_box)
        self.prev_10_frames_button = QPushButton("Prev 10 Frames (Shift + ←)")
        self.next_10_frames_button = QPushButton("Next 10 Frames (Shift + →)")
        self.prev_frame_button = QPushButton("Prev Frame (←)")
        self.next_frame_button = QPushButton("Next Frame (→)")
        self.prev_instance_change_button = QPushButton("◄ Prev ROI (↑)")
        self.next_instance_change_button = QPushButton("► Next ROI (↓)")

        self.navigation_layout.addWidget(self.prev_10_frames_button, 0, 0)
        self.navigation_layout.addWidget(self.next_10_frames_button, 1, 0)
        self.navigation_layout.addWidget(self.prev_frame_button, 0, 1)
        self.navigation_layout.addWidget(self.next_frame_button, 1, 1)
        self.navigation_layout.addWidget(self.prev_instance_change_button, 0, 2)
        self.navigation_layout.addWidget(self.next_instance_change_button, 1, 2)

        self.refiner_group_box = QtWidgets.QGroupBox("Track Refiner")
        self.refiner_layout = QtWidgets.QGridLayout(self.refiner_group_box)

        self.swap_track_button = QPushButton("Swap Track (W)")
        self.swap_track_button.setToolTip("Shift + W for swapping all the frames instance before next ROI.")
        self.delete_track_button = QPushButton("Delete Track (X)")
        self.delete_track_button.setToolTip("Shift + X for deleting all the frames instance before next ROI.")
        self.interpolate_track_button = QPushButton("Interpolate Track (T)")
        self.fill_track_button = QPushButton("Retroactive Fill (F)")

        self.refiner_layout.addWidget(self.swap_track_button, 0, 0)
        self.refiner_layout.addWidget(self.delete_track_button, 0, 1)
        self.refiner_layout.addWidget(self.interpolate_track_button, 1, 0)
        self.refiner_layout.addWidget(self.fill_track_button, 1, 1)

        self.control_layout.addWidget(self.navigation_group_box)
        self.control_layout.addWidget(self.refiner_group_box)
        self.layout.addLayout(self.control_layout)
        
        # Connect QActions to events
        self.load_video_action.triggered.connect(self.load_video)
        self.load_dlc_config_action.triggered.connect(self.load_DLC_config)
        self.load_prediction_action.triggered.connect(self.load_prediction)
        self.save_prediction_action.triggered.connect(self.save_prediction)

        self.purge_inst_by_conf_action.triggered.connect(self.purge_inst_by_conf)

        # Connect buttons to events
        self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.play_button.clicked.connect(self.toggle_playback)
        self.undo_button.clicked.connect(self.undo_changes)
        self.redo_button.clicked.connect(self.redo_changes)

        self.prev_10_frames_button.clicked.connect(lambda: self.change_frame(-10))
        self.prev_frame_button.clicked.connect(lambda: self.change_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self.change_frame(1))
        self.next_10_frames_button.clicked.connect(lambda: self.change_frame(10))

        self.prev_instance_change_button.clicked.connect(lambda:self.prev_instance_change("frame"))
        self.next_instance_change_button.clicked.connect(lambda:self.next_instance_change("frame"))
        self.swap_track_button.clicked.connect(lambda:self.swap_track("point"))
        self.delete_track_button.clicked.connect(lambda:self.delete_track("point"))
        self.interpolate_track_button.clicked.connect(self.interpolate_track)
        self.fill_track_button.clicked.connect(self.fill_track)

        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self.prev_instance_change)
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self.next_instance_change)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)

        QShortcut(QKeySequence(Qt.Key_W), self).activated.connect(lambda:self.swap_track("point"))
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(lambda:self.delete_track("point"))
        QShortcut(QKeySequence(Qt.Key_W | Qt.ShiftModifier), self).activated.connect(lambda:self.swap_track("batch"))
        QShortcut(QKeySequence(Qt.Key_X | Qt.ShiftModifier), self).activated.connect(lambda:self.delete_track("batch"))
        QShortcut(QKeySequence(Qt.Key_T), self).activated.connect(self.interpolate_track)
        QShortcut(QKeySequence(Qt.Key_F), self).activated.connect(self.fill_track)

        QShortcut(QKeySequence(Qt.Key_Z | Qt.ControlModifier), self).activated.connect(self.undo_changes)
        QShortcut(QKeySequence(Qt.Key_Y | Qt.ControlModifier), self).activated.connect(self.redo_changes)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_prediction)
        
        self.graphics_view.mousePressEvent = self.graphics_view_mouse_press_event
        self.graphics_scene.parent = lambda: self # Allow items to access the main window

        self.reset_state()
        self.is_debug = False

    def load_video(self):
        if self.is_debug:
            self.original_vid = VIDEO_FILE_DEBUG
            self.initialize_loaded_video()
            self.config_loader_DLC(DLC_CONFIG_DEBUG)
            self.prediction = PRED_FILE_DEBUG
            self.prediction_loader()
            return
        self.reset_state()
        file_dialog = QtWidgets.QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if video_path:
            self.original_vid = video_path
            self.initialize_loaded_video()
            
    def initialize_loaded_video(self):
        self.navigation_group_box.show()
        self.refiner_group_box.show()
        self.video_name = os.path.basename(self.original_vid).split(".")[0]
        self.cap = cv2.VideoCapture(self.original_vid)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error Open Video", f"Error: Could not open video {self.original_vid}")
            self.cap = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.progress_slider.setRange(0, self.total_frames - 1) # Initialize slider range
        self.display_current_frame()
        self.navigation_box_title_controller()
        print(f"Video loaded: {self.original_vid}")

    def load_DLC_config(self):
        file_dialog = QtWidgets.QFileDialog(self)
        dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        self.config_loader_DLC(dlc_config)

    def config_loader_DLC(self, dlc_file):
        try:
            with open(dlc_file, "r") as conf:
                cfg = yaml.safe_load(conf)
                if cfg:
                    QMessageBox.information(self, "Success", "DLC Config loaded successfully!")
            self.multi_animal = cfg["multianimalproject"]
            self.keypoints = cfg["bodyparts"] if not self.multi_animal else cfg["multianimalbodyparts"]
            self.skeleton = cfg["skeleton"]
            self.individuals = cfg["individuals"]
            self.instance_count = len(self.individuals) if self.individuals is not None else 1
            self.num_keypoints = len(self.keypoints)
            self.keypoint_to_idx = {name: idx for idx, name in enumerate(self.keypoints)}
        except Exception as e:
            QMessageBox.warning(self, "Loading Failed", f"DLC config is not loaded: {e}")
            pass
        
    def load_prediction(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        if self.keypoints is None:
            QMessageBox.warning(self, "No DLC Config", "No dlc config has been loaded, please load it first.")
            return
        file_dialog = QtWidgets.QFileDialog(self)
        prediction_path, _ = file_dialog.getOpenFileName(self, "Load Prediction", "", "HDF5 Files (*.h5);;All Files (*)")
        self.prediction = prediction_path
        self.prediction_loader()
        print(f"Prediction loaded: {self.prediction}")

    def prediction_loader(self):
        with h5py.File(self.prediction, "r") as pred_file:
            if not "tracks" in pred_file.keys():
                print("Error: Prediction file not valid, no 'tracks' key found in prediction file.")
                return False
            QMessageBox.information(self, "Loading Prediction","Loading and parsing prediction file, this could take a few seconds, please wait...")
            self.pred_data = pred_file["tracks"]["table"]
            pred_data_values = np.array([item[1] for item in self.pred_data])
            pred_frame_count = self.pred_data.size
            self.pred_data_array = np.full((pred_frame_count, self.instance_count, self.num_keypoints*3),np.nan)
            for inst in range(self.instance_count): # Sort inst out
                    self.pred_data_array[:,inst,:] = pred_data_values[:, inst*self.num_keypoints*3:(inst+1)*self.num_keypoints*3]
            self.check_instance_count_per_frame()
            if pred_frame_count != self.total_frames:
                QMessageBox.warning(self, "Error: Frame Mismatch", "Total frames in video and in prediction do not match!")
                print(f"Frames in config: {self.total_frames} \n Frames in prediction: {pred_frame_count}")
            self.display_current_frame()

    ###################################################################################################################################################

    def display_current_frame(self):
        self.selected_box = None # Ensure the selected instance is unselected
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.graphics_scene.clear() # Clear previous graphics items

                # Convert OpenCV image to QPixmap and add to scene
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                
                # Add pixmap to the scene
                pixmap_item = self.graphics_scene.addPixmap(pixmap)
                pixmap_item.setZValue(-1)

                self.graphics_scene.setSceneRect(0, 0, w, h)
                self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
                
                # Plot predictions (keypoints, bounding boxes, skeleton)
                if self.pred_data is not None:
                    self.plot_predictions(frame.copy())
                
                self.progress_slider.setValue(self.current_frame_idx) # Update slider position
                self.graphics_view.update() # Force update of the graphics view

            else: # If video frame cannot be read, clear scene and display error
                self.graphics_scene.clear()
                error_text_item = self.graphics_scene.addText("Error: Could not read frame")
                error_text_item.setDefaultTextColor(QColor(255, 255, 255)) # White text
                self.graphics_view.fitInView(error_text_item.boundingRect(), Qt.KeepAspectRatio)
        else: # If no video loaded, clear scene and display message
            self.graphics_scene.clear()
            no_video_text_item = self.graphics_scene.addText("No video loaded")
            no_video_text_item.setDefaultTextColor(QColor(255, 255, 255)) # White text
            self.graphics_view.fitInView(no_video_text_item.boundingRect(), Qt.KeepAspectRatio)

    def plot_predictions(self, frame):
        self.current_selectable_boxes = [] # Store selectable boxes for the current frame
        color_rgb = [(255, 165, 0), (128, 0, 128), (0, 128, 128), (128, 128, 0), (0, 0, 128)]

        # Iterate over each individual (animal)
        for inst in range(self.instance_count):
            color = color_rgb[inst % len(color_rgb)]
            
            # Initiate an empty dict for storing coordinates
            keypoint_coords = dict()
            for kp_idx in range(self.num_keypoints):
                kp = self.pred_data_array[self.current_frame_idx,inst,kp_idx*3:kp_idx*3+3]
                if pd.isna(kp[0]):
                    continue
                x, y, conf = kp[0], kp[1], kp[2]
                keypoint_coords[kp_idx] = (int(x),int(y),float(conf))
                # Draw the dot representing the keypoints
                ellipse = QtWidgets.QGraphicsEllipseItem(x - 3, y - 3, 6, 6)
                ellipse.setBrush(QtGui.QBrush(QtGui.QColor(*color)))
                self.graphics_scene.addItem(ellipse)

            if self.individuals is not None and len(keypoint_coords) >= 2:
                self.plot_bounding_box(keypoint_coords, frame, color, inst)
            if self.skeleton:
                self.plot_skeleton(keypoint_coords, frame, color)

        return frame
    
    def plot_bounding_box(self, keypoint_coords, frame, color, inst):
        # Calculate bounding box coordinates
        x_coords = [keypoint_coords[p][0] for p in keypoint_coords if keypoint_coords[p] is not None]
        y_coords = [keypoint_coords[p][1] for p in keypoint_coords if keypoint_coords[p] is not None]
        kp_confidence = [keypoint_coords[p][2] for p in keypoint_coords if keypoint_coords[p] is not None]

        if not x_coords or not y_coords: # Skip if the mice has no keypoint
            return frame
            
        kp_inst_mean = sum(kp_confidence) / len(kp_confidence)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1] - 1, max_x + padding)
        max_y = min(frame.shape[0] - 1, max_y + padding)

        # Draw bounding box using QGraphicsRectItem
        rect_item = Selectable_Instance(min_x, min_y, max_x - min_x, max_y - min_y, inst, default_color_rgb=color)
        self.graphics_scene.addItem(rect_item)
        self.current_selectable_boxes.append(rect_item)
        rect_item.clicked.connect(self.handle_box_selection) # Connect the signal

        # Add individual label
        text_item = QtWidgets.QGraphicsTextItem(f"Inst: {self.individuals[inst]} | Conf:{kp_inst_mean:.4f}")
        text_item.setPos(min_x, min_y - 20) # Adjust position to be above the bounding box
        text_item.setDefaultTextColor(QtGui.QColor(*color))
        self.graphics_scene.addItem(text_item)
        return frame
    
    def plot_skeleton(self, keypoint_coords, frame, color):
        for start_kp, end_kp in self.skeleton:
            start_kp_idx = self.keypoint_to_idx[start_kp]
            end_kp_idx = self.keypoint_to_idx[end_kp]
            start_coord = keypoint_coords.get(start_kp_idx)
            end_coord = keypoint_coords.get(end_kp_idx)
            if start_coord and end_coord:
                line = QtWidgets.QGraphicsLineItem(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                line.setPen(QtGui.QPen(QtGui.QColor(*color), 2))
                self.graphics_scene.addItem(line)
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
        if self.current_frame_idx in self.roi_frame_list:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: #F04C4C;}""")
        else:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: black;}""")

    ###################################################################################################################################################

    def check_instance_count_per_frame(self):
        nan_mask = np.isnan(self.pred_data_array)
        empty_instance = np.all(nan_mask, axis=2)
        non_empty_instance_numerical = (~empty_instance)*1
        instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
        roi_frames = np.where(np.diff(instance_count_per_frame)!=0)[0]+1

        self.roi_frame_list = list(roi_frames)
        self.progress_slider.set_marked_frames(self.roi_frame_list) # Update ROI frames

    def prev_instance_change(self, mode="frame"):
        if not self.roi_frame_list:
            QMessageBox.information(self, "No Instance Change", "No frames with instance count change to navigate.")
            return
        
        self.roi_frame_list.sort()
        try:
            current_idx_in_roi = self.roi_frame_list.index(self.current_frame_idx) - 1
        except ValueError:
            current_idx_in_roi = bisect.bisect_left(self.roi_frame_list, self.current_frame_idx) - 1

        if current_idx_in_roi >= 0:
            if mode == "idx":
                return self.roi_frame_list[current_idx_in_roi]
            else:
                self.current_frame_idx = self.roi_frame_list[current_idx_in_roi]
            self.display_current_frame()
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No previous ROI frame found.")

    def next_instance_change(self, mode="frame"):
        if not self.roi_frame_list:
            QMessageBox.information(self, "No Instance Change", "No frames with instance count change to navigate.")
            return
        
        self.roi_frame_list.sort()
        try:
            current_idx_in_roi = self.roi_frame_list.index(self.current_frame_idx) + 1
        except ValueError:
            current_idx_in_roi = bisect.bisect_right(self.roi_frame_list, self.current_frame_idx)

        if current_idx_in_roi < len(self.roi_frame_list):
            if mode == "idx":
                return self.roi_frame_list[current_idx_in_roi]
            else:
                self.current_frame_idx = self.roi_frame_list[current_idx_in_roi]
            self.display_current_frame()
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No next ROI frame found.")
            return
        
    ###################################################################################################################################################

    def purge_inst_by_conf(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "No Prediction Data", "Please load a prediction file first.")
            return

        confidence_threshold, ok = QtWidgets.QInputDialog.getDouble(
            self,"Set Confidence Threshold","Delete all instances below this confidence:",
            value=0.5,minValue=0.0,maxValue=1.0,decimals=2
        )
        if ok:
            reply = QMessageBox.question(
                self,"Confirm Deletion",
                f"Are you sure you want to delete all instances with confidence below {confidence_threshold:.2f}?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                confidence_scores = self.pred_data_array[:, :, 2:self.num_keypoints*3:3]
                inst_conf_all = np.mean(confidence_scores, axis=2)
                low_conf_mask = inst_conf_all < confidence_threshold
                f_idx, i_idx = np.where(low_conf_mask)
                self.pred_data_array[f_idx, i_idx, :] = np.nan
                self.display_current_frame()
                self.check_instance_count_per_frame()
            else:
                QMessageBox.information(self, "Deletion Cancelled", "Deletion cancelled by user.")
        else:
            QMessageBox.information(self, "Input Cancelled", "Confidence input cancelled.")

    ###################################################################################################################################################


    def delete_track(self, mode="point"):
        if self.pred_data_array is None: # Silent fail
            return
        current_frame_inst = self.get_current_frame_inst()
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "Track Not Interpolated", "No track is selected.")
            return
        if self.selected_box:
            instance_for_track_deletion = self.selected_box.instance_id
        else:
            instance_for_track_deletion = current_frame_inst[0]
        self._save_state_for_undo() # Save state before modification
        if mode == "point": # Only removing the current frame
            self.pred_data_array[self.current_frame_idx, instance_for_track_deletion, :] = np.nan
        else:
            next_roi_frame_idx = self.next_instance_change("idx")
            if next_roi_frame_idx:
                self.pred_data_array[self.current_frame_idx:next_roi_frame_idx, instance_for_track_deletion, :] = np.nan
        self.selected_box = None
        self.check_instance_count_per_frame()
        self.display_current_frame()
        self.determine_save_status()

    def swap_track(self, mode="point"):
        if self.pred_data_array is None:
            return
        self._save_state_for_undo() # Save state before modification
        if self.instance_count == 2: # 2 instances need no selection
            if mode == "point":
                self.pred_data_array[self.current_frame_idx, 0, :], self.pred_data_array[self.current_frame_idx, 1, :] = \
                self.pred_data_array[self.current_frame_idx, 1, :].copy(), self.pred_data_array[self.current_frame_idx, 0, :].copy()
            else: # Till the end of times
                self.pred_data_array[self.current_frame_idx:, 0, :], \
                self.pred_data_array[self.current_frame_idx:, 1, :] = \
                self.pred_data_array[self.current_frame_idx:, 1, :].copy(), \
                self.pred_data_array[self.current_frame_idx:, 0, :].copy()
            self.selected_box = None
            self.check_instance_count_per_frame()
            self.display_current_frame()
            self.determine_save_status()
        else:
            if not self.selected_box:
                QMessageBox.information(self, "Track Not Swapped", "No track is swapped.")
                return
            raise NotImplementedError

    def interpolate_track(self):
        if self.pred_data_array is None:
            return
        current_frame_inst = self.get_current_frame_inst()
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "Track Not Interpolated", "No track is selected.")
            return
        if self.selected_box:
            instance_for_track_interpolate = self.selected_box.instance_id
        else:
            instance_for_track_interpolate = current_frame_inst[0]
        self._save_state_for_undo() # Save state before modification
        iter_frame_idx = self.current_frame_idx + 1
        frames_to_interpolate = []
        while np.all(np.isnan(self.pred_data_array[iter_frame_idx, instance_for_track_interpolate, :])):
            frames_to_interpolate.append(iter_frame_idx)
            iter_frame_idx += 1
            if iter_frame_idx >= self.total_frames:
                QMessageBox.information(self, "Interpolation Failed", "No valid subsequent keypoint data found for this instance to interpolate to.")
                return
        if frames_to_interpolate:
            frames_to_interpolate.sort()
            start_kp = self.pred_data_array[frames_to_interpolate[0]-1, instance_for_track_interpolate, :]
            end_kp = self.pred_data_array[frames_to_interpolate[-1]+1, instance_for_track_interpolate, :]
            if np.all(np.isnan(start_kp)) or np.all(np.isnan(end_kp)):
                QMessageBox.information(self, "Instance not found", "Selected keypoint not found in the current frame or the next ROI frame.")
                return
            self.pred_data_array[frames_to_interpolate[0]-1:frames_to_interpolate[-1]+2, instance_for_track_interpolate, :]\
                = np.linspace(start_kp, end_kp, num=len(frames_to_interpolate)+2, axis=0)
            self.selected_box = None
            self.check_instance_count_per_frame()
            self.display_current_frame()
            self.determine_save_status()

    def fill_track(self): # Retroactively fill frame from the last vaid kp from previous frames
        if self.pred_data_array is None:
            return
        self._save_state_for_undo() # Save state before modification
        current_frame_inst = set(self.get_current_frame_inst())
        instance_for_track_fill = set([ inst for inst in range(self.instance_count) ]) - current_frame_inst
        if len(instance_for_track_fill) > 1: # Multiple empty instance
            # Construct the question message and buttons dynamically
            question_text = "Multiple missing instances on the current frame. Which instance would you like to duplicate from the previous ROI frame?"
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Multiple Instance")
            msg_box.setText(question_text)
            msg_box.setIcon(QMessageBox.Icon.Question)
            buttons = []
            for inst_id in instance_for_track_fill:
                button_text = f"Instance {self.individuals[inst_id]}" if self.individuals else f"Instance {inst_id}"
                button = msg_box.addButton(button_text, QMessageBox.ButtonRole.ActionRole)
                buttons.append((button, inst_id))
            cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg_box.setDefaultButton(cancel_button)
            msg_box.exec()
            clicked_button = msg_box.clickedButton()
            selected_instance = None
            for button, inst_id in buttons:
                if clicked_button == button:
                    selected_instance = inst_id
                    break
            if selected_instance is not None:
                instance_for_track_fill = selected_instance
            else:
                QMessageBox.information(self, "Selection Cancelled", "No instance was selected. Operation cancelled.")
                return # Exit the function if no instance is selected or cancelled
        else:
            instance_for_track_fill = list(instance_for_track_fill)[0]
        # Find the last non-empty frame for inst, the copy the kp of that frame to all the empty frames in between and the current one
        iter_frame_idx = self.current_frame_idx
        frames_to_fill = []
        while np.all(np.isnan(self.pred_data_array[iter_frame_idx, instance_for_track_fill, :])):
            frames_to_fill.append(iter_frame_idx)
            iter_frame_idx -= 1
            if iter_frame_idx < 0:
                QMessageBox.information(self, "No Previous Data", "No valid previous keypoint data found for this instance.")
                return
        if frames_to_fill:
            frames_to_fill.sort()
            self.pred_data_array[frames_to_fill[0]:frames_to_fill[-1]+1, instance_for_track_fill, :]\
                    = self.pred_data_array[iter_frame_idx, instance_for_track_fill, :].copy()
            self.selected_box = None
            self.check_instance_count_per_frame()
            self.display_current_frame()
            self.determine_save_status()

    ###################################################################################################################################################

    def get_current_frame_inst(self):
        current_frame_inst = []
        for inst in [ inst for inst in range(self.instance_count) ]:
            if np.any(~np.isnan(self.pred_data_array[self.current_frame_idx, inst, :])):
                current_frame_inst.append(inst)
        return current_frame_inst

    def handle_box_selection(self, clicked_box):
        if self.selected_box and self.selected_box != clicked_box and self.selected_box.scene() is not None:
            self.selected_box.toggle_selection() # Deselect previously selected box
        clicked_box.toggle_selection() # Toggle selection of the clicked box
        if clicked_box.is_selected:
            self.selected_box = clicked_box
            print(f"Selected Instance: {clicked_box.instance_id}")
        else:
            self.selected_box = None
            print("No instance selected.")

    def graphics_view_mouse_press_event(self, event):
        item = self.graphics_view.itemAt(event.position().toPoint())
        if item and isinstance(item, Selectable_Instance):
            pass
        else: # If no item was clicked, deselect any currently selected box
            if self.selected_box:
                self.selected_box.toggle_selection()
                self.selected_box = None
                print("No instance selected.")
        QtWidgets.QGraphicsView.mousePressEvent(self.graphics_view, event)

    ###################################################################################################################################################

    def determine_save_status(self):
        if self.pred_data is None or np.all(self.last_saved_pred_array == self.pred_data_array):
            self.is_saved = True
            self.save_prediction_action.setEnabled(False)
        else:
            self.is_saved = False
            self.save_prediction_action.setEnabled(True)

    def undo_changes(self):
        if self.undo_stack:
            self.redo_stack.append(self.pred_data_array.copy())
            self.pred_data_array = self.undo_stack.pop()
            self.check_instance_count_per_frame()
            self.display_current_frame()
            self.determine_save_status()
            print("Undo performed.")
        else:
            QMessageBox.information(self, "Undo", "Nothing to undo.")

    def redo_changes(self):
        if self.redo_stack:
            self.undo_stack.append(self.pred_data_array.copy())
            self.pred_data_array = self.redo_stack.pop()
            self.check_instance_count_per_frame()
            self.display_current_frame()
            self.determine_save_status()
            print("Redo performed.")
        else:
            QMessageBox.information(self, "Redo", "Nothing to redo.")

    def _save_state_for_undo(self):
        if self.pred_data_array is not None:
            self.redo_stack = [] # Clear redo stack when a new action is performed
            self.undo_stack.append(self.pred_data_array.copy())
            if len(self.undo_stack) > self.max_undo_stack_size:
                self.undo_stack.pop(0) # Remove the oldest state

    def save_prediction(self):
        if self.is_saved:
            QMessageBox.information(self, "Save Cancelled", "No change needed to be saved.")
            return
        # Made a copy of the original data and save upon that
        pred_file_dir = os.path.dirname(self.prediction)
        pred_file_name_without_ext = os.path.splitext(os.path.basename(self.prediction))[0]
        trrf_suffix = "_track_refiner_modified_"
        
        if not trrf_suffix in pred_file_name_without_ext:
            save_idx = 0
            base_name = pred_file_name_without_ext
        else:
            base_name, save_idx_str = pred_file_name_without_ext.split(trrf_suffix)
            try:
                save_idx = int(save_idx_str) + 1
            except ValueError:
                save_idx = 0 # Fallback if suffix is malformed
        
        pred_file_to_save_path = os.path.join(pred_file_dir,f"{base_name}{trrf_suffix}{save_idx}.h5")
        
        shutil.copy(self.prediction, pred_file_to_save_path)
        print(f"Copied original prediction to: {pred_file_to_save_path}")
        new_data = []
        num_frames = self.pred_data_array.shape[0]
        for frame_idx in range(num_frames):
            frame_data = self.pred_data_array[frame_idx, :, :].flatten()
            new_data.append((frame_idx, frame_data))
        try:
            if new_data:
                num_vals_per_frame = new_data[0][1].shape[0]
                dtype = np.dtype([('index', 'i8'), ('data', 'f8', (num_vals_per_frame,))])
            else:
                print("No data to save. Skipping HDF5 write.")
                return
            with h5py.File(pred_file_to_save_path, "a") as pred_file_to_save: # Open the copied HDF5 file in write mode
                if 'tracks/table' in pred_file_to_save:
                    del pred_file_to_save['tracks/table']
                structured_data = np.array([(idx, arr) for idx, arr in new_data], dtype=dtype)
                pred_file_to_save.create_dataset('tracks/table', data=structured_data)
            self.prediction = pred_file_to_save_path
            self.last_saved_pred_array = self.pred_data_array.copy()
            self.prediction_loader()
            self.determine_save_status()
            QMessageBox.information(self, "Save Successful", f"Successfully saved modified prediction to: {self.prediction}")
        except Exception as e:
            print(f"An error occurred during HDF5 saving: {e}")
            pass

    def reset_state(self):
        self.original_vid, self.prediction, self.dlc_config, self.video_name = None, None, None, None
        self.keypoints, self.skeleton, self.individuals, self.num_keypoints = None, None, None, None
        self.keypoint_to_idx = None

        self.instance_count = 1
        self.multi_animal = False
        self.pred_data, self.pred_data_array, self.last_saved_pred_array = None, None, None

        self.roi_frame_list = []

        self.cap, self.current_frame = None, None

        self.is_playing = False
        self.is_saved = True

        self.progress_slider.setRange(0, 0)
        self.save_prediction_action.setEnabled(False)
        self.navigation_group_box.hide()
        self.refiner_group_box.hide()

        self.selected_box = None # To keep track of the currently selected box
        
        self.undo_stack = [] # Clear undo stack on reset
        self.redo_stack = [] # Clear redo stack on reset
        self.max_undo_stack_size = 10

    def closeEvent(self, event: QCloseEvent):
        if not self.is_saved:
            # Create a dialog to confirm saving
            close_call = QMessageBox(self)
            close_call.setWindowTitle("Prediction Unsaved")
            close_call.setText("Do you want to save your changes before closing?")
            close_call.setIcon(QMessageBox.Icon.Question)

            save_btn = close_call.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = close_call.addButton("Don't Save", QMessageBox.ButtonRole.DestructiveRole)
            close_btn = close_call.addButton("Close", QMessageBox.RejectRole)
            
            close_call.setDefaultButton(close_btn)

            close_call.exec()
            clicked_button = close_call.clickedButton()
            
            if clicked_button == save_btn:
                self.save_prediction()
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

#######################################################################################################################################################

class Slider_With_Marks(QtWidgets.QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.marked_frames = set()
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

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.marked_frames:
            return

        self.paintEvent_painter(self.marked_frames,"#F04C4C")
        
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

class Selectable_Instance(QtCore.QObject, QGraphicsRectItem):
    clicked = Signal(object) # Signal to emit when this box is clicked

    def __init__(self, x, y, width, height, instance_id, default_color_rgb, parent=None):
        QtCore.QObject.__init__(self, parent)
        QGraphicsRectItem.__init__(self, x, y, width, height, parent)
        self.instance_id = instance_id
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self.default_pen = QPen(QColor(*default_color_rgb), 1) # Use passed color
        self.selected_pen = QPen(QColor(255, 0, 0), 2) # Red, 2px
        self.hover_pen = QPen(QColor(255, 255, 0), 1) # Yellow, 1px

        self.setPen(self.default_pen)
        self.is_selected = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self) # Emit the signal
            event.accept()
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        if not self.is_selected:
            self.setPen(self.hover_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self.is_selected:
            self.setPen(self.default_pen)
        super().hoverLeaveEvent(event)

    def toggle_selection(self):
        self.is_selected = not self.is_selected
        self.update_visual()

    def update_visual(self):
        if self.is_selected:
            self.setPen(self.selected_pen)
        else:
            self.setPen(self.default_pen)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DLC_Track_Refiner()
    window.show()
    app.exec()