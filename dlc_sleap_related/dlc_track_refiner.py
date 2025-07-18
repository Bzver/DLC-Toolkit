import os

import h5py
import yaml

import pandas as pd
import numpy as np
import bisect

import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QPainter, QColor, QPen
from PySide6.QtWidgets import QMessageBox, QPushButton, QGraphicsScene, QGraphicsView, QGraphicsRectItem

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
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_prediction)
        
        self.graphics_view.mousePressEvent = self.graphics_view_mouse_press_event # Override mousePressEvent for the view
        self.graphics_scene.parent = lambda: self # Allow items to access the main window

        self.reset_state()
        self.is_debug = True
        self.selected_box = None # To keep track of the currently selected box

    def load_video(self):
        if self.is_debug:
            self.original_vid = VIDEO_FILE_DEBUG
            self.initialize_loaded_video()
            self.load_DLC_config(DLC_CONFIG_DEBUG)
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
        self.display_current_frame()
        self.navigation_box_title_controller()
        print(f"Video loaded: {self.original_vid}")

    def load_DLC_config(self, dlc_config=None):
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
        self.num_keypoints = len(self.keypoints)
        self.keypoint_to_idx = {name: idx for idx, name in enumerate(self.keypoints)}

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
            pred_data_values = np.array([item[1] for item in self.pred_data])
            pred_frame_count = self.pred_data.size
            self.pred_data_array = np.full((pred_frame_count, self.instance_count, self.num_keypoints*3),np.nan)
            for inst in range(self.instance_count): # Sort inst out
                    self.pred_data_array[:,inst,:] = pred_data_values[:, inst*self.num_keypoints*3:(inst+1)*self.num_keypoints*3]
            self.check_instance_count_per_frame()
            if pred_frame_count != self.total_frames:
                QMessageBox.warning(self, "Error: Frame Mismatch", "Total frames in video and in prediction do not match!")
                print(f"Frames in config: {self.total_frames} \n Frames in prediction: {pred_frame_count}")
            self.last_saved_pred_array = self.pred_data_array
            self.display_current_frame()

    ###################################################################################################################################################

    def display_current_frame(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Clear previous graphics items
                self.graphics_scene.clear()

                # Convert OpenCV image to QPixmap and add to scene
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                
                # Add pixmap to the scene
                pixmap_item = self.graphics_scene.addPixmap(pixmap)
                pixmap_item.setZValue(-1) # Ensure pixmap is behind other items

                # Set scene rect to original frame dimensions
                self.graphics_scene.setSceneRect(0, 0, w, h)
                # Fit view to the scene, maintaining aspect ratio
                self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
                
                # Plot predictions (keypoints, bounding boxes, skeleton)
                if self.pred_data is not None:
                    self.plot_predictions(frame.copy()) # Pass a copy to avoid modifying original frame for QPixmap
                
                self.progress_slider.setValue(self.current_frame_idx) # Update slider position
                self.graphics_view.update() # Force update of the graphics view

            else:
                # If video frame cannot be read, clear scene and display error
                self.graphics_scene.clear()
                error_text_item = self.graphics_scene.addText("Error: Could not read frame")
                error_text_item.setDefaultTextColor(QColor(255, 255, 255)) # White text
                self.graphics_view.fitInView(error_text_item.boundingRect(), Qt.KeepAspectRatio)
        else:
            # If no video loaded, clear scene and display message
            self.graphics_scene.clear()
            no_video_text_item = self.graphics_scene.addText("No video loaded")
            no_video_text_item.setDefaultTextColor(QColor(255, 255, 255)) # White text
            self.graphics_view.fitInView(no_video_text_item.boundingRect(), Qt.KeepAspectRatio)

    def plot_predictions(self, frame):
        self.current_selectable_boxes = [] # Store selectable boxes for the current frame
        color_bgr = [(255, 165, 0), (128, 0, 128), (0, 128, 128), (128, 128, 0), (0, 0, 128)] # BGR

        # Iterate over each individual (animal)
        for inst in range(self.instance_count):
            color = color_bgr[inst % len(color_bgr)]
            
            # Initiate an empty dict for storing coordinates
            keypoint_coords = dict()
            for kp_idx in range(self.num_keypoints):
                kp = self.pred_data_array[self.current_frame_idx,inst,kp_idx*3:kp_idx*3+3]
                if pd.isna(kp[0]):
                    continue
                x, y = kp[0], kp[1]
                keypoint_coords[kp_idx] = (int(x),int(y))
                cv2.circle(frame, (int(x), int(y)), 3, color, -1) # Draw the dot representing the keypoints

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
            
        b, g, r = color
        color_rgb = (r,g,b)

        # Draw bounding box using QGraphicsRectItem
        rect_item = Selectable_Instance(min_x, min_y, max_x - min_x, max_y - min_y, inst, default_color_rgb=color_rgb)
        self.graphics_scene.addItem(rect_item)
        self.current_selectable_boxes.append(rect_item)
        rect_item.clicked.connect(self.handle_box_selection) # Connect the signal

        # Add individual label using OpenCV for now (can be converted to QGraphicsTextItem later if needed)
        cv2.putText(frame, f"Instance: {self.individuals[inst]}", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        return frame
    
    def plot_skeleton(self, keypoint_coords, frame, color):
        for start_kp, end_kp in self.skeleton:
            start_kp_idx = self.keypoint_to_idx[start_kp]
            end_kp_idx = self.keypoint_to_idx[end_kp]
            start_coord = keypoint_coords.get(start_kp_idx)
            end_coord = keypoint_coords.get(end_kp_idx)
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

    def prev_instance_change(self):
        if not self.roi_frame_list:
            QMessageBox.information(self, "No Instance Change", "No frames with instance count change to navigate.")
            return
        
        self.roi_frame_list.sort()
        try:
            current_idx_in_roi = self.roi_frame_list.index(self.current_frame_idx) - 1
        except ValueError:
            current_idx_in_roi = bisect.bisect_left(self.roi_frame_list, self.current_frame_idx) - 1

        if current_idx_in_roi >= 0:
            self.current_frame_idx = self.roi_frame_list[current_idx_in_roi]
            self.display_current_frame()
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No previous ROI frame found.")

    def next_instance_change(self):
        if not self.roi_frame_list:
            QMessageBox.information(self, "No Instance Change", "No frames with instance count change to navigate.")
            return
        
        self.roi_frame_list.sort()
        try:
            current_idx_in_roi = self.roi_frame_list.index(self.current_frame_idx) + 1
        except ValueError:
            # Current frame is not marked, find the closest previous marked frame
            current_idx_in_roi = bisect.bisect_right(self.roi_frame_list, self.current_frame_idx)

        if current_idx_in_roi < len(self.roi_frame_list):
            self.current_frame_idx = self.roi_frame_list[current_idx_in_roi]
            self.display_current_frame()
            self.navigation_box_title_controller()
        else:
            QMessageBox.information(self, "Navigation", "No next ROI frame found.")

    def delete_track(self):
        pass

    def swap_track(self):
        pass

    def handle_box_selection(self, clicked_box):
        if self.selected_box and self.selected_box != clicked_box:
            self.selected_box.toggle_selection() # Deselect previously selected box
        
        clicked_box.toggle_selection() # Toggle selection of the clicked box
        
        if clicked_box.is_selected:
            self.selected_box = clicked_box
            print(f"Selected Instance: {clicked_box.instance_id}")
        else:
            self.selected_box = None
            print("No instance selected.")

    def graphics_view_mouse_press_event(self, event):
        # Called when the mouse is pressed on the QGraphicsView
        item = self.graphics_view.itemAt(event.position().toPoint())
        if item and isinstance(item, Selectable_Instance):
            pass
        else: # If no item was clicked, deselect any currently selected box
            if self.selected_box:
                self.selected_box.toggle_selection()
                self.selected_box = None
                print("No instance selected.")
        QtWidgets.QGraphicsView.mousePressEvent(self.graphics_view, event)

    def determine_save_status(self):
        if np.all(self.last_saved_pred_array == self.pred_data_array):
            self.is_saved = True
        else:
            self.is_saved = False

    def undo_changes():
        pass

    def redo_changes():
        pass

    def save_prediction():
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
        self.navigation_group_box.hide()

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