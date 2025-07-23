import os
import shutil

import h5py
import yaml

import pandas as pd
import numpy as np
import bisect

import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QTimer, QEvent, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QPainter, QColor, QPen, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QPushButton, QGraphicsView, QGraphicsRectItem, QMenu, QToolButton, QGraphicsEllipseItem

DLC_CONFIG_DEBUG = "D:/Project/DLC-Models/NTD/config.yaml"
VIDEO_FILE_DEBUG = "D:/Project/A-SOID/Data/20250709/20250709-first3h-S-conv.mp4"
PRED_FILE_DEBUG = "D:/Project/A-SOID/Data/20250709/20250709-first3h-S-convDLC_HrnetW32_bezver-SD-20250605M-cam52025-06-26shuffle1_detector_090_snapshot_080_el_tr.h5"

# Todo:
#   Add instance generation in keypoint edit mode
#   Add support to export to csv
#   Add support for scenario where individual counts exceed 2

class DLC_Track_Refiner(QtWidgets.QMainWindow):
    prediction_saved = Signal(str) # Signal to emit the path of the saved prediction file

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Track Refiner")
        self.setGeometry(100, 100, 1200, 960)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.is_debug = True
        if self.is_debug:
            self.setWindowTitle("DLC Track Refiner ----- DEBUG MODE")

        # Menu bars
        self.menu_layout = QtWidgets.QHBoxLayout()
        self.load_menu = QMenu("File", self)

        self.load_video_action = self.load_menu.addAction("Load Video")
        self.load_dlc_config_action = self.load_menu.addAction("Load DLC Config")
        self.load_prediction_action = self.load_menu.addAction("Load Prediction")

        self.load_button = QToolButton()
        self.load_button.setText("File")
        self.load_button.setMenu(self.load_menu)
        self.load_button.setPopupMode(QToolButton.InstantPopup)

        self.refiner_menu = QMenu("Adv. Refine", self)

        self.direct_keypoint_edit_action = self.refiner_menu.addAction("Direct Keypoint Edit (Q)")
        self.purge_inst_by_conf_action = self.refiner_menu.addAction("Delete All Track Below Set Confidence")
        self.interpolate_all_action = self.refiner_menu.addAction("Interpolate All Frames for One Inst")
        self.designate_no_mice_zone_action = self.refiner_menu.addAction("Remove All Prediction Inside Area")
        self.segment_auto_correct_action = self.refiner_menu.addAction("Segmental Auto Correct")

        self.refiner_button = QToolButton()
        self.refiner_button.setText("Adv. Refine")
        self.refiner_button.setMenu(self.refiner_menu)
        self.refiner_button.setPopupMode(QToolButton.InstantPopup)

        self.save_menu = QMenu("Save", self)

        self.save_prediction_action = self.save_menu.addAction("Save Prediction")
        self.save_prediction_as_csv_action = self.save_menu.addAction("Save Prediction Into CSV") # 2 B Implemented

        self.save_button = QToolButton()
        self.save_button.setText("Save")
        self.save_button.setMenu(self.save_menu)
        self.save_button.setPopupMode(QToolButton.InstantPopup)

        self.menu_layout.addWidget(self.load_button, alignment=Qt.AlignLeft)
        self.menu_layout.addWidget(self.refiner_button, alignment=Qt.AlignLeft)
        self.menu_layout.addWidget(self.save_button, alignment=Qt.AlignLeft)
        self.menu_layout.addStretch(1)
        self.layout.addLayout(self.menu_layout)

        # Graphics view for interactive elements and video display
        self.graphics_scene = QtWidgets.QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.graphics_view.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.graphics_view, 1)

        # Progress bar
        self.progress_layout = QtWidgets.QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.visibility_button = QPushButton("👁")
        self.visibility_button.setToolTip("Set keypoint label text visibility (V)")
        self.visibility_button.setFixedWidth(20)
        self.magnifier_button = QPushButton("🔍︎")
        self.magnifier_button.setToolTip("Toggle zoom mode (Z)")
        self.magnifier_button.setFixedWidth(20)
        self.undo_button = QPushButton("↻")
        self.undo_button.setToolTip("Undo (Ctrl + Z)")
        self.undo_button.setFixedWidth(20)
        self.redo_button = QPushButton("↺")
        self.redo_button.setToolTip("Redo (Ctrl + Y)")
        self.redo_button.setFixedWidth(20)
        self.progress_slider = Slider_With_Marks(Qt.Horizontal)
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        self.progress_layout.addWidget(self.visibility_button)
        self.progress_layout.addWidget(self.magnifier_button)
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
        self.prev_roi_frame_button = QPushButton("◄ Prev ROI (↑)")
        self.next_roi_frame_button = QPushButton("► Next ROI (↓)")

        self.navigation_layout.addWidget(self.prev_10_frames_button, 0, 0)
        self.navigation_layout.addWidget(self.next_10_frames_button, 1, 0)
        self.navigation_layout.addWidget(self.prev_frame_button, 0, 1)
        self.navigation_layout.addWidget(self.next_frame_button, 1, 1)
        self.navigation_layout.addWidget(self.prev_roi_frame_button, 0, 2)
        self.navigation_layout.addWidget(self.next_roi_frame_button, 1, 2)

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

        self.purge_inst_by_conf_action.triggered.connect(self.purge_inst_by_conf)
        self.interpolate_all_action.triggered.connect(self.interpolate_all)
        self.segment_auto_correct_action.triggered.connect(self.segment_auto_correct)
        self.designate_no_mice_zone_action.triggered.connect(self.designate_no_mice_zone)

        self.save_prediction_action.triggered.connect(self.save_prediction)
        self.save_prediction_as_csv_action.triggered.connect(self.save_prediction_as_csv)

        # Connect buttons to events
        self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.play_button.clicked.connect(self.toggle_playback)
        self.undo_button.clicked.connect(self.undo_changes)
        self.redo_button.clicked.connect(self.redo_changes)
        self.visibility_button.clicked.connect(self.adjust_text_opacity)
        self.magnifier_button.clicked.connect(self.toggle_zoom_mode)

        self.prev_10_frames_button.clicked.connect(lambda: self.change_frame(-10))
        self.prev_frame_button.clicked.connect(lambda: self.change_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self.change_frame(1))
        self.next_10_frames_button.clicked.connect(lambda: self.change_frame(10))

        self.prev_roi_frame_button.clicked.connect(lambda:self.prev_roi_frame("frame"))
        self.next_roi_frame_button.clicked.connect(lambda:self.next_roi_frame("frame"))
        self.swap_track_button.clicked.connect(lambda:self.swap_track("point"))
        self.delete_track_button.clicked.connect(lambda:self.delete_track("point"))
        self.interpolate_track_button.clicked.connect(self.interpolate_track)
        self.fill_track_button.clicked.connect(self.fill_track)

        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self.prev_roi_frame)
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self.next_roi_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)

        QShortcut(QKeySequence(Qt.Key_W), self).activated.connect(lambda:self.swap_track("point"))
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(lambda:self.delete_track("point"))
        QShortcut(QKeySequence(Qt.Key_W | Qt.ShiftModifier), self).activated.connect(lambda:self.swap_track("batch"))
        QShortcut(QKeySequence(Qt.Key_X | Qt.ShiftModifier), self).activated.connect(lambda:self.delete_track("batch"))
        QShortcut(QKeySequence(Qt.Key_T), self).activated.connect(self.interpolate_track)
        QShortcut(QKeySequence(Qt.Key_F), self).activated.connect(self.fill_track)
        QShortcut(QKeySequence(Qt.Key_Q), self).activated.connect(self.direct_keypoint_edit)
        QShortcut(QKeySequence(Qt.Key_Backspace), self).activated.connect(self.delete_dragged_keypoint)

        QShortcut(QKeySequence(Qt.Key_Z | Qt.ControlModifier), self).activated.connect(self.undo_changes)
        QShortcut(QKeySequence(Qt.Key_Y | Qt.ControlModifier), self).activated.connect(self.redo_changes)
        QShortcut(QKeySequence(Qt.Key_V), self).activated.connect(self.adjust_text_opacity)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_prediction)
        QShortcut(QKeySequence(Qt.Key_Z), self).activated.connect(self.toggle_zoom_mode)
        
        self.graphics_view.mousePressEvent = self.graphics_view_mouse_press_event
        self.graphics_view.mouseMoveEvent = self.graphics_view_mouse_move_event
        self.graphics_view.mouseReleaseEvent = self.graphics_view_mouse_release_event
        self.graphics_scene.parent = lambda: self # Allow items to access the main window

        self.reset_state()

    def reset_state(self):
        self.video_file, self.prediction, self.dlc_config, self.video_name = None, None, None, None
        self.keypoints, self.skeleton, self.individuals, self.num_keypoints = None, None, None, None
        self.keypoint_to_idx = None

        self.instance_count = 1
        self.instance_count_per_frame = None
        self.multi_animal = False
        self.pred_data, self.pred_data_array = None, None

        self.roi_frame_list, self.marked_roi_frame_list = [], []

        self.cap, self.current_frame = None, None

        self.is_playing = False

        self.progress_slider.setRange(0, 0)
        self.navigation_group_box.hide()
        self.refiner_group_box.hide()
        self.text_label_opacity = 1.0

        self.selected_box = None

        self.is_drawing_zone = False
        self.start_point, self.current_rect_item = None, None

        self.is_kp_edit = False
        self.dragged_keypoint, self.dragged_bounding_box = None, None
        
        self.undo_stack, self.redo_stack = [], []
        self.max_undo_stack_size = 50
        self.is_initialize = True
        self.is_saved = True

        self.is_zoom_mode = False
        self.zoom_factor = 1.0

    def load_video(self):
        if self.is_debug:
            self.video_file = VIDEO_FILE_DEBUG
            self.initialize_loaded_video()
            self.config_loader_dlc(DLC_CONFIG_DEBUG)
            self.prediction = PRED_FILE_DEBUG
            self.prediction_loader()
            return
        self.reset_state()
        file_dialog = QtWidgets.QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if video_path:
            self.video_file = video_path
            self.initialize_loaded_video()
            
    def initialize_loaded_video(self):
        self.navigation_group_box.show()
        self.refiner_group_box.show()
        self.video_name = os.path.basename(self.video_file).split(".")[0]
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error Open Video", f"Error: Could not open video {self.video_file}")
            self.cap = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.progress_slider.setRange(0, self.total_frames - 1) # Initialize slider range
        self.display_current_frame()
        self.reset_zoom()
        self.navigation_box_title_controller()
        print(f"Video loaded: {self.video_file}")

    def load_DLC_config(self):
        file_dialog = QtWidgets.QFileDialog(self)
        dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        self.config_loader_dlc(dlc_config)

    def config_loader_dlc(self, dlc_file):
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
            if self.is_initialize:
                QMessageBox.information(self, "Loading Prediction","Loading and parsing prediction file, this could take a few seconds, please wait...")
                self.is_initialize = False
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
            self.reset_zoom()

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
                
                new_transform = QtGui.QTransform()
                new_transform.scale(self.zoom_factor, self.zoom_factor)
                self.graphics_view.setTransform(new_transform)

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

    def plot_predictions(self, frame):
        self.current_selectable_boxes = [] # Store selectable boxes for the current frame
        color_rgb = [(255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)]

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
                keypoint_coords[kp_idx] = (float(x),float(y),float(conf))
                # Draw the dot representing the keypoints
                keypoint_item = Draggable_Keypoint(x - 3, y - 3, 6, 6, inst, kp_idx, default_color_rgb=color)

                if isinstance(keypoint_item, Draggable_Keypoint):
                    keypoint_item.setFlag(QGraphicsEllipseItem.ItemIsMovable, self.is_kp_edit)

                self.graphics_scene.addItem(keypoint_item)
                keypoint_item.setZValue(1) # Ensure keypoints are on top of the video frame
                keypoint_item.keypoint_moved.connect(self.update_keypoint_position)
                keypoint_item.keypoint_drag_started.connect(self.set_dragged_keypoint)

            self.plot_keypoint_label(keypoint_coords, frame, color)

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

        # Calculate mouse center
        mouse_center_x = (min_x + max_x) / 2
        mouse_center_y = (min_y + max_y) / 2
        mouse_center = (mouse_center_x, mouse_center_y)

        # Draw bounding box using QGraphicsRectItem
        rect_item = Selectable_Instance(min_x, min_y, max_x - min_x, max_y - min_y, inst, default_color_rgb=color)
        if isinstance(rect_item, Selectable_Instance):
                rect_item.setFlag(QGraphicsRectItem.ItemIsMovable, self.is_kp_edit)
        self.graphics_scene.addItem(rect_item)
        self.current_selectable_boxes.append(rect_item)
        rect_item.clicked.connect(self.handle_box_selection) # Connect the signal
        # Connect the bounding_box_moved signal to the update method in DLC_Track_Refiner
        rect_item.bounding_box_moved.connect(self.update_instance_position)

        # Add individual label and keypoint labels
        text_item_inst = QtWidgets.QGraphicsTextItem(f"Inst: {self.individuals[inst]} | Conf:{kp_inst_mean:.4f}")
        text_item_inst.setPos(min_x, min_y - 20) # Adjust position to be above the bounding box
        text_item_inst.setDefaultTextColor(QtGui.QColor(*color))
        text_item_inst.setOpacity(self.text_label_opacity)
        text_item_inst.setFlag(QtWidgets.QGraphicsTextItem.ItemIgnoresTransformations) # Keep text size constant
        self.graphics_scene.addItem(text_item_inst)

    def plot_keypoint_label(self, keypoint_coords, frame, color):
        # Plot keypoint labels
        for kp_idx, (x, y, conf) in keypoint_coords.items():
            keypoint_label = self.keypoints[kp_idx]

            text_item = QtWidgets.QGraphicsTextItem(f"{keypoint_label}")

            font = text_item.font() # Get the default font of the QGraphicsTextItem
            fm = QtGui.QFontMetrics(font)
            text_rect = fm.boundingRect(keypoint_label)
            
            text_width = text_rect.width()
            text_height = text_rect.height()

            text_x = x - text_width / 2 + 5
            text_y = y - text_height / 2 + 5

            text_item.setPos(text_x, text_y)
            text_item.setDefaultTextColor(QtGui.QColor(*color))
            text_item.setOpacity(self.text_label_opacity)
            text_item.setFlag(QtWidgets.QGraphicsTextItem.ItemIgnoresTransformations) # Keep text size constant
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
                self.reset_zoom()
                self.navigation_box_title_controller()

    def set_frame_from_slider(self, value):
        if self.cap and self.cap.isOpened():
            self.current_frame_idx = value
            self.display_current_frame()
            self.reset_zoom()
            self.navigation_box_title_controller()

    def autoplay_video(self):
        if self.cap and self.cap.isOpened():
            if self.current_frame_idx < self.total_frames - 1:
                self.current_frame_idx += 1
                self.display_current_frame()
                self.reset_zoom()
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
        title = f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1} | Video: {self.video_name}"
        if self.is_kp_edit and self.current_frame_idx:
            title += " ----- KEYPOINTS EDITING MODE ----- "
        self.navigation_group_box.setTitle(title)
        if self.current_frame_idx in self.roi_frame_list:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: #F04C4C;}""")
        else:
            self.navigation_group_box.setStyleSheet("""QGroupBox::title {color: black;}""")

    def adjust_text_opacity(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Keypoint Label Visibility")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(0, 100) # Scale 0.00 to 1.00 to 0 to 100
        slider.setValue(int(self.text_label_opacity * 100))
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        slider.setTickInterval(10)
        layout.addWidget(slider)
        slider.valueChanged.connect(self._update_text_opacity) # Connect slider to update opacity and redraw
        dialog.exec()

    def _update_text_opacity(self, value):
        self.text_label_opacity = value / 100.0
        self.display_current_frame()

    def toggle_zoom_mode(self):
        self.is_zoom_mode = not self.is_zoom_mode
        if self.is_kp_edit:
            self.is_kp_edit = False
        if self.is_zoom_mode:
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphics_view.wheelEvent = self.graphics_view_mouse_wheel_event
        else:
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.graphics_view.wheelEvent = super(QGraphicsView, self.graphics_view).wheelEvent

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)

    ###################################################################################################################################################

    def check_instance_count_per_frame(self):
        nan_mask = np.isnan(self.pred_data_array)
        empty_instance = np.all(nan_mask, axis=2)
        # Convert the boolean mask to numerical (0 for empty, 1 for non-empty) and sum per frame
        non_empty_instance_numerical = (~empty_instance) * 1
        self.instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
        if self.marked_roi_frame_list:
            self.roi_frame_list = list(self.marked_roi_frame_list)
        else:
            roi_frames = np.where(np.diff(self.instance_count_per_frame)!=0)[0]+1
            self.roi_frame_list = list(roi_frames)
        self.progress_slider.set_roi_frames(self.roi_frame_list) # Update ROI frames

        if self.is_debug:
            print("\n--- Instance Counting Details ---")
            f_idx = self.current_frame_idx
            print(f"Frame {f_idx}: (Expected Count: {self.instance_count_per_frame[f_idx]})")
            
            # Get the NaN mask for the current frame's instances
            current_frame_nan_mask = nan_mask[f_idx, :, :] # Shape: (num_instances, num_keypoints * 3)

            for i_idx in range(self.pred_data_array.shape[1]): # Iterate through instances for the current frame
                if not empty_instance[f_idx, i_idx]: # If this instance is NOT empty (i.e., it's being counted)
                    
                    # Extract the NaN status for this specific instance, then reshape to (num_keypoints, 3)
                    instance_nan_status = current_frame_nan_mask[i_idx, :].reshape(self.num_keypoints, 3)
                    
                    non_nan_keypoints_found = []
                    for k_idx in range(self.num_keypoints): # Iterate through individual keypoints
                        # If NOT all 3 values (x, y, conf) for this keypoint are NaN, then it's contributing
                        if not np.all(instance_nan_status[k_idx, :]):
                            # Try to get keypoint label, fall back to index if not available
                            keypoint_label = (self.keypoints[k_idx] # Changed from self.keypoint_labels
                                            if hasattr(self, 'keypoints') and k_idx < len(self.keypoints) 
                                            else f"Keypoint_{k_idx}")
                            non_nan_keypoints_found.append(keypoint_label)
                    
                    print(f"  Instance {i_idx} is counted because it has non-NaN keypoints: {', '.join(non_nan_keypoints_found)}")
            print("-----------------------------------\n")

    def prev_roi_frame(self, mode="frame"):
        if not self.roi_frame_list:
            QMessageBox.information(self, "No Instance Change", "No ROI frames to navigate.")
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
            self.reset_zoom()
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No previous ROI frame found.")

    def next_roi_frame(self, mode="frame"):
        if not self.roi_frame_list:
            QMessageBox.information(self, "No Instance Change", "No frames with ROI frames to navigate.")
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
            self.reset_zoom()
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No next ROI frame found.")
            return
        
    ###################################################################################################################################################

    def purge_inst_by_conf(self):
        if not self._you_shall_not_pass():
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
                self._save_state_for_undo() # Save state before modification
                confidence_scores = self.pred_data_array[:, :, 2:self.num_keypoints*3:3]
                inst_conf_all = np.mean(confidence_scores, axis=2)
                low_conf_mask = inst_conf_all < confidence_threshold
                f_idx, i_idx = np.where(low_conf_mask)
                self.pred_data_array[f_idx, i_idx, :] = np.nan
                self.is_saved = False # Mark as unsaved
                self.display_current_frame()
                self.reset_zoom()
                self.check_instance_count_per_frame()
            else:
                QMessageBox.information(self, "Deletion Cancelled", "Deletion cancelled by user.")
        else:
            QMessageBox.information(self, "Input Cancelled", "Confidence input cancelled.")

    def interpolate_all(self):
        if not self._you_shall_not_pass():
            return
        
        if not self.selected_box:
            QMessageBox.information(self, "No Track Selected", "Please select a track to interpolate all frames for one instance.")
            return

        instance_to_interpolate = self.selected_box.instance_id
        self._save_state_for_undo() # Save state before modification

        for kp_idx in range(self.num_keypoints):
            # Extract x, y, confidence for the current keypoint and instance across all frames
            x_coords = self.pred_data_array[:, instance_to_interpolate, kp_idx*3]
            y_coords = self.pred_data_array[:, instance_to_interpolate, kp_idx*3+1]
            conf_values = self.pred_data_array[:, instance_to_interpolate, kp_idx*3+2]

            # Convert to pandas Series for interpolation
            x_series = pd.Series(x_coords)
            y_series = pd.Series(y_coords)
            conf_series = pd.Series(conf_values)

            # Interpolate NaNs
            x_interpolated = x_series.interpolate(method='linear', limit_direction='both').values
            y_interpolated = y_series.interpolate(method='linear', limit_direction='both').values
            conf_interpolated = conf_series.interpolate(method='linear', limit_direction='both').values

            # Update the pred_data_array
            self.pred_data_array[:, instance_to_interpolate, kp_idx*3] = x_interpolated
            self.pred_data_array[:, instance_to_interpolate, kp_idx*3+1] = y_interpolated
            self.pred_data_array[:, instance_to_interpolate, kp_idx*3+2] = conf_interpolated

        self.is_saved = False
        self.selected_box = None
        self.check_instance_count_per_frame()
        self.display_current_frame()
        self.reset_zoom()
        QMessageBox.information(self, "Interpolation Complete", f"All frames interpolated for instance {self.individuals[instance_to_interpolate]}.")
    
    def designate_no_mice_zone(self):
        if not self._you_shall_not_pass():
            return
        self.is_drawing_zone = True
        self.graphics_view.setCursor(Qt.CrossCursor)
        QMessageBox.information(self, "Designate No Mice Zone", "Click and drag on the video to select a zone. Release to apply.")

    def segment_auto_correct(self):
        QMessageBox.information(self, "Segmental Auto Correct",
        """
  This function works for scenarios where only one instance persistently remains in view while another goes in and out.\n
  It will identify segments where only one instance is detected for more than 100 frames.\n
  Throughout and proceeding these segments, the track associated with the remaining instance will be swapped to instance 0.\n
        """
        )

        if not self._you_shall_not_pass():
            return

        if self.instance_count < 2: # Need at least two instances for swapping to make sense
            QMessageBox.information(self, "Info", "Less than two instances configured. Segmental auto-correction is not applicable.")
            return

        self.check_instance_count_per_frame()

        segments_to_correct = []
        num_corrections_applied = 0
        current_segment_start = -1
        min_segment_length = 50

        for i in range(len(self.instance_count_per_frame)):
            if self.instance_count_per_frame[i] <= 1:
                if current_segment_start == -1:
                    current_segment_start = i
            else:
                if current_segment_start != -1:
                    segment_length = i - current_segment_start
                    if segment_length >= min_segment_length: # Use >= for segments of exactly 100 frames
                        segments_to_correct.append((current_segment_start, i - 1))
                    current_segment_start = -1
        
        # Handle the last segment if it extends to the end of the video
        if current_segment_start != -1:
            segment_length = len(self.instance_count_per_frame) - current_segment_start
            if segment_length >= min_segment_length:
                segments_to_correct.append((current_segment_start, len(self.instance_count_per_frame) - 1))

        if not segments_to_correct:
            QMessageBox.information(self, "Info", "No segments found where less than two instance is persistently detected for more than 100 frames.")
            return

        self._save_state_for_undo() # Save state before making changes

        for start_frame, end_frame in segments_to_correct:
            for frame_idx in range(start_frame, end_frame + 1): # Swap non 'instance 0' with 'instance 0' for all frames in the segment
                if self.instance_count_per_frame[frame_idx] == 0: # Skip swapping for empty predictions
                    continue
                current_present_at_frame = np.where(~np.all(np.isnan(self.pred_data_array[frame_idx]), axis=1))[0]
                if current_present_at_frame[0] != 0: # Ensure that at this specific frame, the instance to be swapped is not instance 0
                    self.pred_data_array[frame_idx, 0, :], self.pred_data_array[frame_idx, 1, :] = \
                    self.pred_data_array[frame_idx, 1, :].copy(), self.pred_data_array[frame_idx, 0, :].copy()
                last_present_instance = current_present_at_frame[0]

            # Apply the swap from (end_frame + 1) to the end of the video, IF the last instance detected was not 0
            if last_present_instance is not None and last_present_instance != 0:
                print(f"Applying global swap from frame {end_frame + 1} to end.")
                self.pred_data_array[end_frame + 1:, 0, :], \
                self.pred_data_array[end_frame + 1:, 1, :] = \
                self.pred_data_array[end_frame + 1:, 1, :].copy(), \
                self.pred_data_array[end_frame + 1:, 0, :].copy()
            
            num_corrections_applied += 1

        QMessageBox.information(self, "Success", f"Segmental auto-correction applied to {num_corrections_applied} segments.")
        self.is_saved = False
        self.check_instance_count_per_frame()
        self.display_current_frame()
        self.reset_zoom()

    def direct_keypoint_edit(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return
        
        self.is_kp_edit = not self.is_kp_edit # Toggle the mode
        if self.is_zoom_mode: # Cancel the zoom mode when editing
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.graphics_view.wheelEvent = super(QGraphicsView, self.graphics_view).wheelEvent
        self.navigation_box_title_controller() # Update title to reflect mode
        
        # Enable/disable draggable property of items based on self.is_kp_edit
        for item in self.graphics_scene.items():
            if isinstance(item, Draggable_Keypoint):
                item.setFlag(QGraphicsEllipseItem.ItemIsMovable, self.is_kp_edit)
            elif isinstance(item, Selectable_Instance):
                item.setFlag(QGraphicsRectItem.ItemIsMovable, self.is_kp_edit)
        
        if self.is_kp_edit:
            QMessageBox.information(self, "Keypoint Editing Mode", 
                    "Keypoint editing mode is ON.\n" 
                    "You can now drag keypoints and bounding boxes to adjust positions.\n"
                    "If you want to delete a keypoint, simply press Backspace when holding it.")
        else:
            QMessageBox.information(self, "Keypoint Editing Mode", "Keypoint editing mode is OFF.")

    def update_keypoint_position(self, instance_id, keypoint_id, new_x, new_y):
        self._save_state_for_undo()
        # Keep the confidence value as is.
        current_conf = self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3+2]
        self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3] += new_x
        self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3+1] += new_y
        # Ensure confidence is not NaN if x,y are valid
        if pd.isna(current_conf) and not (pd.isna(new_x) or pd.isna(new_y)):
            self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3+2] = 1.0 # Default confidence
        print(f"{self.keypoints[keypoint_id]} of instance {instance_id} moved by ({new_x}, {new_y})")
        self.is_saved = False
        QtCore.QTimer.singleShot(0, self.display_current_frame)

    def update_instance_position(self, instance_id, dx, dy):
        self._save_state_for_undo()
        # Update all keypoints for the given instance in the current frame
        for kp_idx in range(self.num_keypoints):

            x_coord_idx = kp_idx * 3
            y_coord_idx = kp_idx * 3 + 1
            
            current_x = self.pred_data_array[self.current_frame_idx, instance_id, x_coord_idx]
            current_y = self.pred_data_array[self.current_frame_idx, instance_id, y_coord_idx]

            if not pd.isna(current_x) and not pd.isna(current_y):
                self.pred_data_array[self.current_frame_idx, instance_id, x_coord_idx] = current_x + dx
                self.pred_data_array[self.current_frame_idx, instance_id, y_coord_idx] = current_y + dy
        print(f"Instance {instance_id} moved by ({dx}, {dy})")
        self.is_saved = False
        QtCore.QTimer.singleShot(0, self.display_current_frame)

    def delete_dragged_keypoint(self):
        if self.is_kp_edit and self.dragged_keypoint:
            self._save_state_for_undo()
            instance_id = self.dragged_keypoint.instance_id
            keypoint_id = self.dragged_keypoint.keypoint_id
            # Set the keypoint coordinates and confidence to NaN
            self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3:keypoint_id*3+3] = np.nan
            print(f"{self.keypoints[keypoint_id]} of instance {instance_id} deleted.")
            self.dragged_keypoint = None # Clear the dragged keypoint
            self.display_current_frame()

    def set_dragged_keypoint(self, keypoint_item):
        self.dragged_keypoint = keypoint_item

    ###################################################################################################################################################

    def delete_track(self, mode="point"):
        if not self._you_shall_not_pass():
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
            next_roi_frame_idx = self.next_roi_frame("idx")
            if next_roi_frame_idx:
                self.pred_data_array[self.current_frame_idx:next_roi_frame_idx, instance_for_track_deletion, :] = np.nan
            else: # If no next ROI, delete till end of video
                self.pred_data_array[self.current_frame_idx:, instance_for_track_deletion, :] = np.nan

        self.selected_box = None
        self.is_saved = False # Mark as unsaved
        self.check_instance_count_per_frame()
        self.display_current_frame()

    def swap_track(self, mode="point"):
        if not self._you_shall_not_pass():
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
            self.is_saved = False # Mark as unsaved
            self.check_instance_count_per_frame()
            self.display_current_frame()
        else:
            if not self.selected_box:
                QMessageBox.information(self, "Track Not Swapped", "No track is selected for swapping. Please select one of the track.")
                return
            QMessageBox.information(self, "Not Implemented", "Swapping for more than 2 instances is not yet implemented.")
            raise NotImplementedError

    def interpolate_track(self):
        if not self._you_shall_not_pass():
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
            start_frame_for_interpol = frames_to_interpolate[0] - 1
            end_frame_for_interpol = frames_to_interpolate[-1] + 1

            start_kp_data = self.pred_data_array[start_frame_for_interpol, instance_for_track_interpolate, :]
            end_kp_data = self.pred_data_array[end_frame_for_interpol, instance_for_track_interpolate, :]
            
            if np.all(np.isnan(start_kp_data)) or np.all(np.isnan(end_kp_data)):
                QMessageBox.information(self, "Instance not found", "Selected keypoint not found in the start or end frame for interpolation.")
                return

            # Interpolate all 3 values (x, y, confidence)
            interpolated_values = np.linspace(start_kp_data, end_kp_data, num=len(frames_to_interpolate)+2, axis=0)
            
            # Apply interpolation
            self.pred_data_array[start_frame_for_interpol : end_frame_for_interpol + 1, instance_for_track_interpolate, :] = interpolated_values
            
            self.selected_box = None
            self.is_saved = False # Mark as unsaved
            self.check_instance_count_per_frame()
            self.display_current_frame()
        else:
            QMessageBox.information(self, "Interpolation Info", "No gaps found to interpolate for the selected instance.")

    def fill_track(self): # Retroactively fill frame from the last vaid kp from previous frames
        if not self._you_shall_not_pass():
            return
        self._save_state_for_undo() # Save state before modification
        current_frame_inst = set(self.get_current_frame_inst())
        
        # Find instances that are missing in the current frame
        missing_instances = [inst for inst in range(self.instance_count) if inst not in current_frame_inst]
        
        if not missing_instances:
            QMessageBox.information(self, "No Missing Instances", "No missing instances found in the current frame to fill.")
            return
        
        instance_for_track_fill = None

        if len(missing_instances) > 1: # Multiple empty instances
            # Construct the question message and buttons dynamically
            question_text = "Multiple missing instances on the current frame. Which instance would you like to duplicate from the previous valid frame?"
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Multiple Instance")
            msg_box.setText(question_text)
            msg_box.setIcon(QMessageBox.Icon.Question)
            buttons = []
            for inst_id in missing_instances:
                button_text = f"Instance {self.individuals[inst_id]}" if self.individuals else f"Instance {inst_id}"
                button = msg_box.addButton(button_text, QMessageBox.ButtonRole.ActionRole)
                buttons.append((button, inst_id))
            cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg_box.setDefaultButton(cancel_button)
            msg_box.exec()
            clicked_button = msg_box.clickedButton()
            
            for button, inst_id in buttons:
                if clicked_button == button:
                    instance_for_track_fill = inst_id
                    break
            if instance_for_track_fill is None:
                QMessageBox.information(self, "Selection Cancelled", "No instance was selected. Operation cancelled.")
                return # Exit the function if no instance is selected or cancelled
        else: # Only one missing instance
            instance_for_track_fill = missing_instances[0]

        # Find the last non-empty frame for the selected instance
        iter_frame_idx = self.current_frame_idx - 1
        frames_to_fill = [self.current_frame_idx] # Start by including the current frame
        
        found_previous_data = False
        while iter_frame_idx >= 0:
            if np.all(np.isnan(self.pred_data_array[iter_frame_idx, instance_for_track_fill, :])):
                frames_to_fill.append(iter_frame_idx)
                iter_frame_idx -= 1
            else:
                found_previous_data = True
                break
        
        if not found_previous_data:
            QMessageBox.information(self, "No Previous Data", "No valid previous keypoint data found for this instance.")
            return
            
        # The keypoint data to copy from
        source_kp_data = self.pred_data_array[iter_frame_idx, instance_for_track_fill, :].copy()

        if frames_to_fill:
            frames_to_fill.sort() # Ensure frames are in ascending order
            for frame_idx_to_fill in frames_to_fill:
                self.pred_data_array[frame_idx_to_fill, instance_for_track_fill, :] = source_kp_data
            
            self.selected_box = None
            self.is_saved = False # Mark as unsaved
            self.check_instance_count_per_frame()
            self.display_current_frame()
        else:
            QMessageBox.information(self, "Fill Info", "No frames found to fill for the selected instance.")


    ###################################################################################################################################################

    def _you_shall_not_pass(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return False
        if self.is_kp_edit:
            QMessageBox.warning(self, "Not Allowed", "Please finish editing keypoints before using this function.")
            return False
        return True

    def get_current_frame_inst(self):
        current_frame_inst = []
        for inst in [ inst for inst in range(self.instance_count) ]:
            if np.any(~np.isnan(self.pred_data_array[self.current_frame_idx, inst, :])):
                current_frame_inst.append(inst)
        return current_frame_inst

    def handle_box_selection(self, clicked_box):
        if self.is_kp_edit: # no box select to prevent interference
            self.selected_box = None
            return
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
        if self.is_zoom_mode:
            QtWidgets.QGraphicsView.mousePressEvent(self.graphics_view, event)
            return
        if self.is_drawing_zone:
            self.start_point = self.graphics_view.mapToScene(event.position().toPoint())
            if self.current_rect_item:
                self.graphics_scene.removeItem(self.current_rect_item)
            self.current_rect_item = QGraphicsRectItem(self.start_point.x(), self.start_point.y(), 0, 0)
            self.current_rect_item.setPen(QPen(QColor(255, 0, 0), 2)) # Red pen for drawing
            self.graphics_scene.addItem(self.current_rect_item)
        else:
            # If not in drawing zone mode, allow default behavior for item selection/dragging
            item = self.graphics_view.itemAt(event.position().toPoint())
            
            # Check if the clicked item is a draggable keypoint or selectable instance and we are in edit mode
            if (isinstance(item, Draggable_Keypoint) or isinstance(item, Selectable_Instance)) and self.is_kp_edit:
                pass # Let the item's own mousePressEvent handle it
            elif isinstance(item, Selectable_Instance): # Allow box selection even outside of direct keypoint edit mode
                pass
            else: # If no interactive item was clicked, deselect any currently selected box
                if self.selected_box:
                    self.selected_box.toggle_selection()
                    self.selected_box = None
                    print("No instance selected.")
            QtWidgets.QGraphicsView.mousePressEvent(self.graphics_view, event)

    def graphics_view_mouse_move_event(self, event):
        if self.is_zoom_mode:
            QtWidgets.QGraphicsView.mouseMoveEvent(self.graphics_view, event)
            return
        if self.is_drawing_zone and self.start_point:
            current_point = self.graphics_view.mapToScene(event.position().toPoint())
            rect = QtCore.QRectF(self.start_point, current_point).normalized()
            self.current_rect_item.setRect(rect)
        QtWidgets.QGraphicsView.mouseMoveEvent(self.graphics_view, event)

    def graphics_view_mouse_release_event(self, event):
        if self.is_drawing_zone and self.start_point and self.current_rect_item:
            self.is_drawing_zone = False
            self.graphics_view.setCursor(Qt.ArrowCursor)
            
            rect = self.current_rect_item.rect()
            self.graphics_scene.removeItem(self.current_rect_item) # Remove the temporary drawing rectangle
            self.current_rect_item = None
            self.start_point = None

            # Convert scene coordinates to image coordinates
            x1, y1, x2, y2 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())

            self._save_state_for_undo()
            self._clean_inconsistent_nans() # Cleanup ghost points (NaN for x,y yet non-nan in confidence)

            all_x_kps = self.pred_data_array[:,:,0::3]
            all_y_kps = self.pred_data_array[:,:,1::3]

            x_in_range = (all_x_kps >= x1) & (all_x_kps <= x2)
            y_in_range = (all_y_kps >= y1) & (all_y_kps <= y2)
            points_in_bbox_mask = x_in_range & y_in_range

            self.pred_data_array[np.repeat(points_in_bbox_mask, 3, axis=-1)] = np.nan
            
            self.is_saved = False
            self.check_instance_count_per_frame()
            self.display_current_frame()
            QMessageBox.information(self, "No Mice Zone Applied", "Keypoints within the selected zone have been set to NaN.")
        
    def graphics_view_mouse_release_event(self, event):
        if self.is_zoom_mode: # Allow panning, but prevent other interactions
            QtWidgets.QGraphicsView.mouseReleaseEvent(self.graphics_view, event)
            return
        QtWidgets.QGraphicsView.mouseReleaseEvent(self.graphics_view, event)

    def graphics_view_mouse_wheel_event(self, event):
        if self.is_zoom_mode:
            zoom_in_factor = 1.15
            zoom_out_factor = 1 / zoom_in_factor

            mouse_pos_view = event.position()
            mouse_pos_scene = self.graphics_view.mapToScene(QtCore.QPoint(int(mouse_pos_view.x()), int(mouse_pos_view.y())))

            transform = self.graphics_view.transform()

            if event.angleDelta().y() > 0: # Zoom in
                self.zoom_factor *= zoom_in_factor
            else: # Zoom out
                self.zoom_factor *= zoom_out_factor
            
            self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0)) # Limit zoom to prevent extreme values

            new_transform = QtGui.QTransform()
            new_transform.scale(self.zoom_factor, self.zoom_factor)
            self.graphics_view.setTransform(new_transform)

            self.graphics_view.centerOn(mouse_pos_scene)
        else:
            super(QGraphicsView, self.graphics_view).wheelEvent(event)

    ###################################################################################################################################################

    def _clean_inconsistent_nans(self):
        print("Performing Operation Clean Sweep to inconsistent NaN keypoints...")
        nan_mask = np.isnan(self.pred_data_array)
        x_is_nan = nan_mask[:, :, 0::3]
        y_is_nan = nan_mask[:, :, 1::3]
        keypoints_to_fully_nan = x_is_nan | y_is_nan
        full_nan_sweep_mask = np.repeat(keypoints_to_fully_nan, 3, axis=-1)
        self.pred_data_array[full_nan_sweep_mask] = np.nan
        print("Inconsistent NaN sweep completed.")

    def undo_changes(self):
        if self.undo_stack:
            self.redo_stack.append(self.pred_data_array.copy())
            self.pred_data_array = self.undo_stack.pop()
            self.check_instance_count_per_frame()
            self.display_current_frame()
            self.is_saved = False
            print("Undo performed.")
        else:
            QMessageBox.information(self, "Undo", "Nothing to undo.")

    def redo_changes(self):
        if self.redo_stack:
            self.undo_stack.append(self.pred_data_array.copy())
            self.pred_data_array = self.redo_stack.pop()
            self.check_instance_count_per_frame()
            self.display_current_frame()
            self.is_saved = False
            print("Redo performed.")
        else:
            QMessageBox.information(self, "Redo", "Nothing to redo.")

    def _save_state_for_undo(self):
        self.is_saved = False
        if self.pred_data_array is not None:
            self.redo_stack = [] # Clear redo stack when a new action is performed
            self.undo_stack.append(self.pred_data_array.copy())
            if len(self.undo_stack) > self.max_undo_stack_size:
                self.undo_stack.pop(0) # Remove the oldest state

    def save_prediction(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "No Prediction Data", "Please load a prediction file first.")
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
                    pred_file_to_save['tracks/table'][...] = new_data
            self.prediction = pred_file_to_save_path
            self.prediction_loader()
            self.is_saved = True
            
            QMessageBox.information(self, "Save Successful", f"Successfully saved modified prediction to: {self.prediction}")
            self.prediction_saved.emit(self.prediction) # Emit the signal with the saved file path
        except Exception as e:
            QMessageBox.critical(self, "Saving Error", f"An error occurred during HDF5 saving: {e}")
            print(f"An error occurred during HDF5 saving: {e}")

    def save_prediction_as_csv(self):
        QMessageBox.information(self, "Not Implemented", "Sorry! This method is yet to be implemented, DM me if you need it real bad. :D")

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.reset_zoom()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        if not self.is_debug and self.prediction is not None and not self.is_saved:
            # Create a dialog to confirm saving
            close_call = QMessageBox(self)
            close_call.setWindowTitle("Close Application?")
            close_call.setText("Do you want to save your current prediction before closing?")
            close_call.setIcon(QMessageBox.Icon.Question)

            save_btn = close_call.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = close_call.addButton("Don't Save", QMessageBox.ButtonRole.DestructiveRole)
            close_btn = close_call.addButton("Cancel", QMessageBox.RejectRole)
            
            close_call.setDefaultButton(close_btn)
            close_call.exec()
            clicked_button = close_call.clickedButton()
            
            if clicked_button == save_btn:
                self.save_prediction()
                if self.is_saved: # Only accept if save was successful
                    event.accept()
                else: # If save failed or was cancelled by user from within save_prediction
                    event.ignore()
            elif clicked_button == discard_btn:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept() 

#######################################################################################################################################################

class Slider_With_Marks(QtWidgets.QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.roi_frames = set()
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

    def set_roi_frames(self, roi_frames):
        self.roi_frames = set(roi_frames)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.roi_frames:
            return
        self.paintEvent_painter(self.roi_frames,"#F04C4C")
        
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

#######################################################################################################################################################

class Draggable_Keypoint(QtCore.QObject, QGraphicsEllipseItem):
    # Signal to emit when the keypoint is moved
    keypoint_moved = Signal(int, int, float, float) # instance_id, keypoint_id, new_x, new_y
    keypoint_drag_started = Signal(object) # Emits the Draggable_Keypoint object itself

    def __init__(self, x, y, width, height, instance_id, keypoint_id, default_color_rgb, parent=None):
        QtCore.QObject.__init__(self, None)
        QGraphicsEllipseItem.__init__(self, x, y, width, height, parent)
        self.instance_id = instance_id
        self.keypoint_id = keypoint_id
        self.default_color_rgb = default_color_rgb
        self.setBrush(QtGui.QBrush(QtGui.QColor(*default_color_rgb)))
        self.setPen(QtGui.QPen(QtGui.QColor(*default_color_rgb), 1))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, False) # Initially not movable, enabled by direct_keypoint_edit
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.original_pos = self.pos() # Store initial position on press

    def hoverEnterEvent(self, event):
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0))) # Yellow on hover
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QtGui.QBrush(QtGui.QColor(*self.default_color_rgb))) # Revert to default
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsEllipseItem.ItemIsMovable:
            self.keypoint_drag_started.emit(self)
            self.original_pos = self.pos() # Store position at the start of the drag
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsEllipseItem.ItemIsMovable:
            new_pos = self.pos()
            if new_pos != self.original_pos:
                center_x = new_pos.x() + self.rect().width() / 2
                center_y = new_pos.y() + self.rect().height() / 2
                self.keypoint_moved.emit(self.instance_id, self.keypoint_id, center_x, center_y)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self.scene():
            # The actual data array update will happen on mouse release
            return value
        return super().itemChange(change, value)

class Selectable_Instance(QtCore.QObject, QGraphicsRectItem):
    clicked = Signal(object) # Signal to emit when this box is clicked
    # Signal to emit when the bounding box is moved
    bounding_box_moved = Signal(int, float, float) # instance_id, dx, dy

    def __init__(self, x, y, width, height, instance_id, default_color_rgb, parent=None):
        QtCore.QObject.__init__(self, parent)
        QGraphicsRectItem.__init__(self, x, y, width, height, parent)
        self.instance_id = instance_id
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, False) # Initially not movable, enabled by direct_keypoint_edit
        self.setAcceptHoverEvents(True)

        self.default_pen = QPen(QColor(*default_color_rgb), 1) # Use passed color
        self.selected_pen = QPen(QColor(255, 0, 0), 2) # Red, 2px
        self.hover_pen = QPen(QColor(255, 255, 0), 1) # Yellow, 1px

        self.setPen(self.default_pen)
        self.is_selected = False
        self.last_mouse_pos = None # To track mouse movement for dragging

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self) # Emit the signal for selection
            if self.flags() & QGraphicsRectItem.ItemIsMovable:
                self.last_mouse_pos = event.scenePos() # Store the initial mouse position for dragging
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.flags() & QGraphicsRectItem.ItemIsMovable and self.last_mouse_pos is not None:
            current_pos = event.scenePos()
            dx = current_pos.x() - self.last_mouse_pos.x()
            dy = current_pos.y() - self.last_mouse_pos.y()
            
            self.setPos(self.pos().x() + dx, self.pos().y() + dy)
            self.last_mouse_pos = current_pos # Update last position for next move event

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsRectItem.ItemIsMovable and self.last_mouse_pos is not None:

            if hasattr(self, 'initial_pos_on_press'):
                dx = self.pos().x() - self.initial_pos_on_press.x()
                dy = self.pos().y() - self.initial_pos_on_press.y()
                if dx != 0 or dy != 0:
                    self.bounding_box_moved.emit(self.instance_id, dx, dy)
                del self.initial_pos_on_press # Clean up
            
            self.last_mouse_pos = None # Reset
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionChange and self.flags() & QGraphicsRectItem.ItemIsMovable:
            if not hasattr(self, 'initial_pos_on_press'):
                self.initial_pos_on_press = self.pos()
        return super().itemChange(change, value)

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