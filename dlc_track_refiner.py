import os

import pandas as pd
import numpy as np

import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QPainter, QColor, QPen, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QPushButton, QGraphicsView, QGraphicsRectItem

from utils.dtu_widget import Menu_Widget, Progress_Bar_Widget, Nav_Widget
from utils.dtu_comp import Selectable_Instance, Draggable_Keypoint, Adjust_Property_Dialog
from utils.dtu_io import DLC_Loader
from utils.dtu_dataclass import Export_Settings
import utils.dtu_io as dio
import utils.dtu_helper as duh
import utils.dtu_gui_helper as dugh
import utils.dtu_track_edit as dute

DLC_CONFIG_DEBUG = "D:/Project/DLC-Models/NTD/config.yaml"
VIDEO_FILE_DEBUG = "D:/Project/DLC-Models/NTD/videos/20250709-first3h-S-conv.mp4"
PRED_FILE_DEBUG = "D:/Project/A-SOID/Data/20250709/20250709-first3h-S-convDLC_HrnetW32_bezver-SD-20250605M-cam52025-06-26shuffle1_detector_370_snapshot_150_el.h5"

# Todo:
#   Add support for use cases where individual counts exceed 2

class DLC_Track_Refiner(QtWidgets.QMainWindow):
    prediction_saved = Signal(str) # Signal to emit the path of the saved prediction file
    refined_frames_exported = Signal(list) # New signal to emit the refined_roi_frame_list

    def __init__(self):
        super().__init__()

        self.is_debug = False
        self.setWindowTitle(duh.format_title("DLC Track Refiner", self.is_debug))
        self.setGeometry(100, 100, 1200, 960)

        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        refiner_menu_config = {
            "File": {
                "display_name": "File",
                "buttons": [
                    ("Load Video", self.load_video),
                    ("Load Prediction", self.load_prediction)
                ]
            },
            "Refiner": {
                "display_name": "Refine",
                "buttons": [
                    ("Swap Track On Current Frame (W)", lambda:self._swap_track_wrapper("point")),
                    ("Delete Selected Track On Current Frame (X)", lambda:self._delete_track_wrapper("point")),
                    ("Interpolate Track (T)", self._interpolate_track_wrapper),
                    ("Generate Instance (G)", self._generate_track_wrapper),
                    ("Swap Track Until The End (Shift + W)", lambda:self._swap_track_wrapper("batch")),
                    ("Delete Selected Track Until Next ROI (Shift + X)", lambda:self._delete_track_wrapper("batch")),
                    ("Interpolate Missing Keypoints (Shift + T)", self._interpolate_missing_kp_wrapper)
                ]
            },
            "AdvRefiner": {
                "display_name": "Adv. Refine",
                "buttons": [
                    ("Direct Keypoint Edit (Q)", self.direct_keypoint_edit),
                    ("Delete All Track Below Set Confidence", self.purge_inst_by_conf),
                    ("Remove All Prediction Inside Area", self.designate_no_mice_zone),
                    ("Interpolate All Frames for One Inst", self.interpolate_all),
                    ("Fix Track Using Idtrackerai Trajectory", self.correct_track_using_idtrackerai)
                ]
            },
            "Preference": {
                "display_name": "Preference",
                "buttons": [
                    ("Adjust Point Size", self.adjust_point_size),
                    ("Adjust Plot Visibility", self.adjust_plot_opacity)
                ]
            },
            "Save": {
                "display_name": "Save",
                "buttons": [
                    ("Mark All As Refined", self.mark_all_as_refined),
                    ("Save Prediction", self.save_prediction),
                    ("Save Prediction Into CSV", self.save_prediction_as_csv)
                ]
            }
        }
        if self.is_debug:
            refiner_menu_config["File"]["buttons"].append(("Debug Load", self.debug_load))
        self.menu_widget.add_menu_from_config(refiner_menu_config)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

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
        self.progress_widget = Progress_Bar_Widget()
        self.progress_layout.addWidget(self.progress_widget)
        self.progress_widget.frame_changed.connect(self._handle_frame_change_from_comp)

        self.magnifier_button = QPushButton("🔍︎")
        self.magnifier_button.setToolTip("Toggle zoom mode (Z)")
        self.magnifier_button.setFixedWidth(20)
        self.undo_button = QPushButton("↻")
        self.undo_button.setToolTip("Undo (Ctrl + Z)")
        self.undo_button.setFixedWidth(20)
        self.redo_button = QPushButton("↺")
        self.redo_button.setToolTip("Redo (Ctrl + Y)")
        self.redo_button.setFixedWidth(20)

        self.progress_layout.addWidget(self.magnifier_button)
        self.progress_layout.addWidget(self.undo_button)
        self.progress_layout.addWidget(self.redo_button)
        self.layout.addLayout(self.progress_layout)

        self.nav_widget = Nav_Widget(mark_name="ROI Frame")
        self.layout.addWidget(self.nav_widget)
        self.nav_widget.set_collapsed(True)

        # Connect buttons to events
        self.undo_button.clicked.connect(self.undo_changes)
        self.redo_button.clicked.connect(self.redo_changes)
        self.magnifier_button.clicked.connect(self.toggle_zoom_mode)

        self.graphics_view.mousePressEvent = self.graphics_view_mouse_press_event
        self.graphics_view.mouseMoveEvent = self.graphics_view_mouse_move_event
        self.graphics_view.mouseReleaseEvent = self.graphics_view_mouse_release_event
        self.graphics_scene.parent = lambda: self # Allow items to access the main window

        self.setup_shortcut()
        self.reset_state()

    def setup_shortcut(self):
        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(lambda:self._navigate_roi_frames("prev"))
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(lambda:self._navigate_roi_frames("next"))
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.progress_widget.toggle_playback)

        QShortcut(QKeySequence(Qt.Key_W), self).activated.connect(lambda:self._swap_track_wrapper("point"))
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(lambda:self._delete_track_wrapper("point"))
        QShortcut(QKeySequence(Qt.Key_W | Qt.ShiftModifier), self).activated.connect(lambda:self._swap_track_wrapper("batch"))
        QShortcut(QKeySequence(Qt.Key_X | Qt.ShiftModifier), self).activated.connect(lambda:self._delete_track_wrapper("batch"))
        QShortcut(QKeySequence(Qt.Key_T), self).activated.connect(self._interpolate_track_wrapper)
        QShortcut(QKeySequence(Qt.Key_T | Qt.ShiftModifier), self).activated.connect(self._interpolate_missing_kp_wrapper)
        QShortcut(QKeySequence(Qt.Key_G), self).activated.connect(self._generate_track_wrapper)
        QShortcut(QKeySequence(Qt.Key_Q), self).activated.connect(self.direct_keypoint_edit)
        QShortcut(QKeySequence(Qt.Key_Backspace), self).activated.connect(self.delete_dragged_keypoint)

        QShortcut(QKeySequence(Qt.Key_Z | Qt.ControlModifier), self).activated.connect(self.undo_changes)
        QShortcut(QKeySequence(Qt.Key_Y | Qt.ControlModifier), self).activated.connect(self.redo_changes)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_prediction)
        QShortcut(QKeySequence(Qt.Key_Z), self).activated.connect(self.toggle_zoom_mode)

    def reset_state(self):
        self.video_file, self.prediction, self.video_name = None, None, None
        self.dlc_data, self.keypoint_to_idx = None, None
        self.instance_count_per_frame, self.pred_data_array = None, None
        self.data_loader = DLC_Loader(None, None)

        self.roi_frame_list, self.marked_roi_frame_list, self.refined_roi_frame_list = [], [], []

        self.cap, self.current_frame = None, None

        self.is_playing = False

        self.nav_widget.set_collapsed(True)
        self.plot_opacity = 1.0
        self.point_size = 6

        self.selected_box = None

        self.is_drawing_zone = False
        self.start_point, self.current_rect_item = None, None

        self.is_kp_edit = False
        self.dragged_keypoint, self.dragged_bounding_box = None, None
        
        self.undo_stack, self.redo_stack = [], []
        self.max_undo_stack_size = 50
        self.is_saved = True

        self.is_zoom_mode = False
        self.zoom_factor = 1.0

        self._refresh_slider()

    def load_video(self):
        self.reset_state()
        file_dialog = QtWidgets.QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if video_path:
            self.video_file = video_path
            self.initialize_loaded_video()
            
    def initialize_loaded_video(self):
        self.video_name = os.path.basename(self.video_file).split(".")[0]
        self.nav_widget.set_collapsed(False)
        self.cap = cv2.VideoCapture(self.video_file)
        
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error Open Video", f"Error: Could not open video {self.video_file}")
            self.cap = None
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.progress_widget.set_slider_range(self.total_frames) # Initialize slider range
        self.display_current_frame()
        self.reset_zoom()
        self.navigation_title_controller()
        print(f"Video loaded: {self.video_file}")

    def load_prediction(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Video", "No video has been loaded, please load a video first.")
            return
        
        file_dialog = QtWidgets.QFileDialog(self)
        prediction_path, _ = file_dialog.getOpenFileName(self, "Load Prediction", "", "HDF5 Files (*.h5);;All Files (*)")

        if not prediction_path:
            return
        
        self.data_loader.prediction_filepath = prediction_path

        QMessageBox.information(self, "DLC Config Loaded", "Prediction loaded , now loading DLC config.")

        file_dialog = QtWidgets.QFileDialog(self)
        dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")

        if not dlc_config:
            return

        self.data_loader.dlc_config_filepath = dlc_config

        self.dlc_data = dugh.load_and_show_message(self, self.data_loader)
        self.initialize_loaded_data()

    def debug_load(self):
        self.video_file = VIDEO_FILE_DEBUG
        self.initialize_loaded_video()
        self.data_loader.dlc_config_filepath = DLC_CONFIG_DEBUG
        self.data_loader.prediction_filepath = self.prediction = PRED_FILE_DEBUG
        self.dlc_data = dugh.load_and_show_message(self, self.data_loader)
        self.initialize_loaded_data()

    def initialize_loaded_data(self):
        self.prediction = self.dlc_data.prediction_filepath
        self.pred_data_array = self.dlc_data.pred_data_array
        self.keypoint_to_idx = {name: idx for idx, name in enumerate(self.dlc_data.keypoints)}

        self.check_instance_count_per_frame()

        if self.dlc_data.pred_frame_count != self.total_frames:
            QMessageBox.warning(self, "Error: Frame Mismatch",
                    "Total frames in video and in prediction do not match!"
                    f"Frames in config: {self.total_frames} \n Frames in prediction: {self.dlc_data.pred_frame_count}")
            
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
                if self.pred_data_array is not None:
                    self.plot_predictions(frame.copy())

                self.graphics_view.update() # Force update of the graphics view

                self.progress_widget.set_current_frame(self.current_frame_idx) # Update slider handle's position

            else: # If video frame cannot be read, clear scene and display error
                self.graphics_scene.clear()
                error_text_item = self.graphics_scene.addText("Error: Could not read frame")
                error_text_item.setDefaultTextColor(QColor(255, 255, 255)) # White text
                self.graphics_view.fitInView(error_text_item.boundingRect(), Qt.KeepAspectRatio)

    def plot_predictions(self, frame):
        self.current_selectable_boxes = [] # Store selectable boxes for the current frame
        color_rgb = [(255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)]

        # Iterate over each individual (animal)
        for inst in range(self.dlc_data.instance_count):
            color = color_rgb[inst % len(color_rgb)]
            
            # Initiate an empty dict for storing coordinates
            keypoint_coords = dict()
            for kp_idx in range(self.dlc_data.num_keypoint):
                kp = self.pred_data_array[self.current_frame_idx,inst,kp_idx*3:kp_idx*3+3]
                if pd.isna(kp[0]):
                    continue
                x, y, conf = kp[0], kp[1], kp[2]
                keypoint_coords[kp_idx] = (float(x),float(y),float(conf))
                # Draw the dot representing the keypoints
                keypoint_item = Draggable_Keypoint(x - self.point_size / 2, y - self.point_size / 2, self.point_size, self.point_size, inst, kp_idx, default_color_rgb=color)
                keypoint_item.setOpacity(self.plot_opacity)

                if isinstance(keypoint_item, Draggable_Keypoint):
                    keypoint_item.setFlag(QtWidgets.QGraphicsEllipseItem.ItemIsMovable, self.is_kp_edit)

                self.graphics_scene.addItem(keypoint_item)
                keypoint_item.setZValue(1) # Ensure keypoints are on top of the video frame
                keypoint_item.keypoint_moved.connect(self.update_keypoint_position)
                keypoint_item.keypoint_drag_started.connect(self.set_dragged_keypoint)

            self.plot_keypoint_label(keypoint_coords, frame, color)

            if self.dlc_data.individuals is not None and len(keypoint_coords) >= 2:
                self.plot_bounding_box(keypoint_coords, frame, color, inst)
            if self.dlc_data.skeleton:
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
        rect_item.setOpacity(self.plot_opacity)
        if isinstance(rect_item, Selectable_Instance):
                rect_item.setFlag(QGraphicsRectItem.ItemIsMovable, self.is_kp_edit)
        self.graphics_scene.addItem(rect_item)
        self.current_selectable_boxes.append(rect_item)
        rect_item.clicked.connect(self._handle_box_selection) # Connect the signal
        # Connect the bounding_box_moved signal to the update method in DLC_Track_Refiner
        rect_item.bounding_box_moved.connect(self.update_instance_position)

        # Add individual label and keypoint labels
        text_item_inst = QtWidgets.QGraphicsTextItem(f"Inst: {self.dlc_data.individuals[inst]} | Conf:{kp_inst_mean:.4f}")
        text_item_inst.setPos(min_x, min_y - 20) # Adjust position to be above the bounding box
        text_item_inst.setDefaultTextColor(QtGui.QColor(*color))
        text_item_inst.setOpacity(self.plot_opacity)
        text_item_inst.setFlag(QtWidgets.QGraphicsTextItem.ItemIgnoresTransformations) # Keep text size constant
        self.graphics_scene.addItem(text_item_inst)

    def plot_keypoint_label(self, keypoint_coords, frame, color):
        # Plot keypoint labels
        for kp_idx, (x, y, conf) in keypoint_coords.items():
            keypoint_label = self.dlc_data.keypoints[kp_idx]

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
            text_item.setOpacity(self.plot_opacity)
            text_item.setFlag(QtWidgets.QGraphicsTextItem.ItemIgnoresTransformations) # Keep text size constant
            self.graphics_scene.addItem(text_item)

        return frame
    
    def plot_skeleton(self, keypoint_coords, frame, color):
        for start_kp, end_kp in self.dlc_data.skeleton:
            start_kp_idx = self.keypoint_to_idx[start_kp]
            end_kp_idx = self.keypoint_to_idx[end_kp]
            start_coord = keypoint_coords.get(start_kp_idx)
            end_coord = keypoint_coords.get(end_kp_idx)
            if start_coord and end_coord:
                line = QtWidgets.QGraphicsLineItem(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                line.setPen(QtGui.QPen(QtGui.QColor(*color), self.point_size / 3))
                line.setOpacity(self.plot_opacity)
                self.graphics_scene.addItem(line)
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
        if self.refined_roi_frame_list and self.marked_roi_frame_list:
            title_text += f"Manual Refining Progress: {len(self.refined_roi_frame_list)} / {len(self.marked_roi_frame_list)} Frames Refined"
        if self.is_kp_edit and self.current_frame_idx:
            title_text += " ----- KEYPOINTS EDITING MODE ----- "
        self.nav_widget.setTitle(title_text)
        if self.current_frame_idx in self.refined_roi_frame_list:
            self.nav_widget.setTitleColor("#009979")  # Teal/Green for refined
        elif self.current_frame_idx in self.roi_frame_list:
            self.nav_widget.setTitleColor("#F04C4C")  # Red for ROI
        else:
            self.nav_widget.setTitleColor("black")  # Default black

    def toggle_zoom_mode(self):
        self.is_zoom_mode = not self.is_zoom_mode
        if self.is_zoom_mode:
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphics_view.wheelEvent = self.graphics_view_mouse_wheel_event
        else:
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.graphics_view.wheelEvent = super(QGraphicsView, self.graphics_view).wheelEvent
        self.navigation_title_controller()

    def reset_zoom(self):
        if self.zoom_factor == 1.0: # Don't do anything when there has not been any zoom in/out yet
            return
        self.zoom_factor = 1.0
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)

    ###################################################################################################################################################

    def adjust_point_size(self):
        dialog = Adjust_Property_Dialog(
            property_name="Point Size", property_val=self.point_size, range=(0.1, 5), parent=self)
        dialog.property_changed.connect(self._update_point_size)
        dialog.show()

    def adjust_plot_opacity(self):
        dialog = Adjust_Property_Dialog(
            property_name="Point Opacity", property_val=self.point_size, range=(0.00, 1.00), parent=self)
        dialog.property_changed.connect(self._update_plot_opacity)
        dialog.show()

    def _update_point_size(self, value):
        self.point_size = value
        self.display_current_frame()

    def _update_plot_opacity(self, value):
        self.plot_opacity = value
        self.display_current_frame()

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

        self._refresh_slider()

        if self.is_debug:
            print("\n--- Instance Counting Details ---")
            f_idx = self.current_frame_idx
            print(f"Frame {f_idx}: (Expected Count: {self.instance_count_per_frame[f_idx]})")
            print("-----------------------------------\n")

    def _navigate_roi_frames(self, mode):
        dugh.navigate_to_marked_frame(self, self.roi_frame_list, self.current_frame_idx, self._handle_frame_change_from_comp, mode)
        
    ###################################################################################################################################################

    def purge_inst_by_conf(self):
        if not self._track_edit_blocker():
            return
        confidence_threshold, ok = QtWidgets.QInputDialog.getDouble(
            self,"Set Confidence Threshold","Delete all instances below this confidence:",
            value=0.5,minValue=0.0,maxValue=1.0,decimals=2
        )

        if not ok:
            QMessageBox.information(self, "Input Cancelled", "Confidence input cancelled.")
            return

        # Prompt for the body part discovery percentage
        bodypart_threshold, ok_bp = QtWidgets.QInputDialog.getDouble(
            self, "Set Body Part Threshold", "Delete all instances with fewer than this percentage of body parts discovered:",
            value=20.0, minValue=0.0, maxValue=100.0, decimals=0
        )

        if not ok_bp:
            QMessageBox.information(self, "Input Cancelled", "Body part threshold input cancelled.")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete all instances with confidence below {confidence_threshold:.2f} "
            f"OR with fewer than {int(bodypart_threshold)}% of body parts discovered?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self._save_state_for_undo()  # Save state before modification

            self.pred_data_array, removed_frames_count, removed_instances_count = dute.purge_by_conf_and_bp(
                self.pred_data_array, self.dlc_data.num_keypoint, confidence_threshold, bodypart_threshold)
            QMessageBox.information(self, "Deletion Complete", f"Deleted {removed_instances_count} instances from {removed_frames_count} frames.")

            self._on_track_data_changed()
            self.reset_zoom()
        else:
            QMessageBox.information(self, "Deletion Cancelled", "Deletion cancelled by user.")

    def interpolate_all(self):
        if not self._track_edit_blocker():
            return
        
        if not self.selected_box:
            QMessageBox.information(self, "No Track Selected", "Please select a track to interpolate all frames for one instance.")
            return

        instance_to_interpolate = self.selected_box.instance_id
        self._save_state_for_undo() # Save state before modification

        for kp_idx in range(self.dlc_data.num_keypoint):
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

        self._on_track_data_changed()
        self.reset_zoom()
        QMessageBox.information(self, "Interpolation Complete", f"All frames interpolated for instance {self.dlc_data.individuals[instance_to_interpolate]}.")
    
    def designate_no_mice_zone(self):
        if not self._track_edit_blocker():
            return
        self.is_drawing_zone = True
        self.graphics_view.setCursor(Qt.CrossCursor)
        QMessageBox.information(self, "Designate No Mice Zone", "Click and drag on the video to select a zone. Release to apply.")

    def correct_track_using_idtrackerai(self):
        if not self._track_edit_blocker():
            return
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not folder_path:
            return
        
        idt_csv = os.path.join(folder_path, "trajectories", "trajectories_csv", "trajectories.csv")
        if not os.path.isfile(idt_csv):
            QMessageBox.warning(self, "", "")
        df_idt = pd.read_csv(idt_csv, header=0)
        idt_traj_array = duh.parse_idt_df_into_ndarray(df_idt)

        dialog = "Fixing track from idTracker.ai trajectories..."
        title = f"Fix Track Using idTracker.ai"
        progress = dugh.get_progress_dialog(self, 0, self.total_frames, title, dialog)

        self.pred_data_array, changes_applied = dute.idt_track_correction(self.pred_data_array, idt_traj_array, progress)

        progress.close()

        if not changes_applied:
            QMessageBox.information(self, "No Changes Applied", "No changes were applied.")
            return

        QMessageBox.information(self, "Track Correction Finished", f"Applied {changes_applied} changes to the current track.")

        self._on_track_data_changed()

    def direct_keypoint_edit(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return
        
        self.is_kp_edit = not self.is_kp_edit # Toggle the mode
        self.navigation_title_controller() # Update title to reflect mode
        
        # Enable/disable draggable property of items based on self.is_kp_edit
        for item in self.graphics_scene.items():
            if isinstance(item, Draggable_Keypoint):
                item.setFlag(QtWidgets.QGraphicsEllipseItem.ItemIsMovable, self.is_kp_edit)
            elif isinstance(item, Selectable_Instance):
                item.setFlag(QGraphicsRectItem.ItemIsMovable, self.is_kp_edit)
        
        # Non-intrusive feedback
        if self.is_kp_edit:
            self.statusBar().showMessage(
                "Keypoint editing mode is ON. Drag to adjust. Press Backspace to delete a keypoint."
            )
        else:
            self.statusBar().showMessage("Keypoint editing mode is OFF.")

        self.navigation_title_controller()

    def update_keypoint_position(self, instance_id, keypoint_id, new_x, new_y):
        self._save_state_for_undo()
        if self.current_frame_idx in self.marked_roi_frame_list and self.current_frame_idx not in self.refined_roi_frame_list:
            self.refined_roi_frame_list.append(self.current_frame_idx)
        current_conf = self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3+2]
        self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3] += new_x
        self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3+1] += new_y
        # Ensure confidence is not NaN if x,y are valid
        if pd.isna(current_conf) and not (pd.isna(new_x) or pd.isna(new_y)):
            self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3+2] = 1.0
        print(f"{self.dlc_data.keypoints[keypoint_id]} of instance {instance_id} moved by ({new_x}, {new_y})")
        self.is_saved = False
        QtCore.QTimer.singleShot(0, self.display_current_frame)
        self._refresh_slider()
        self.navigation_title_controller()

    def update_instance_position(self, instance_id, dx, dy):
        self._save_state_for_undo()
        if self.current_frame_idx in self.marked_roi_frame_list and self.current_frame_idx not in self.refined_roi_frame_list:
            self.refined_roi_frame_list.append(self.current_frame_idx)
        # Update all keypoints for the given instance in the current frame
        for kp_idx in range(self.dlc_data.num_keypoint):

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
        self._refresh_slider()
        self.navigation_title_controller()

    def delete_dragged_keypoint(self):
        if self.is_kp_edit and self.dragged_keypoint:
            self._save_state_for_undo()
            if self.current_frame_idx in self.marked_roi_frame_list and self.current_frame_idx not in self.refined_roi_frame_list:
                self.refined_roi_frame_list.append(self.current_frame_idx)
            instance_id = self.dragged_keypoint.instance_id
            keypoint_id = self.dragged_keypoint.keypoint_id
            # Set the keypoint coordinates and confidence to NaN
            self.pred_data_array[self.current_frame_idx, instance_id, keypoint_id*3:keypoint_id*3+3] = np.nan
            print(f"{self.dlc_data.keypoints[keypoint_id]} of instance {instance_id} deleted.")
            self.dragged_keypoint = None # Clear the dragged keypoint
            self._refresh_slider()
            self.display_current_frame()

    def set_dragged_keypoint(self, keypoint_item):
        self.dragged_keypoint = keypoint_item

    ###################################################################################################################################################

    def _track_edit_blocker(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return False
        if self.is_kp_edit:
            QMessageBox.warning(self, "Not Allowed", "Please finish editing keypoints before using this function.")
            return False
        return True
    
    def _delete_track_wrapper(self, mode, deletion_range=None):
        if not self._track_edit_blocker():
            return

        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "No Track Seleted",
                "When there are more than one instance present, "
                "you need to click one of the instance bounding box to specify which to delete.")
            return
        
        self._save_state_for_undo()
        selected_instance_idx = self.selected_box.instance_id if self.selected_box else current_frame_inst[0]
        try:
            self.pred_data_array = dute.delete_track(self.pred_data_array, self.current_frame_idx,
                                        self.roi_frame_list, selected_instance_idx, mode, deletion_range)
        except ValueError as e:
            QMessageBox.warning(self, "Deletion Error", str(e))
            return
        
        self._on_track_data_changed()
        
    def _swap_track_wrapper(self, mode, swap_range=None):
        if not self._track_edit_blocker():
            return
        
        if self.dlc_data.instance_count > 2:
            QMessageBox.information(self, "Not Implemented",
                "Swapping while instance count is larger than 2 has not been implemented.")
            return

        self._save_state_for_undo()
        try:
            self.pred_data_array = dute.swap_track(self.pred_data_array, self.current_frame_idx, mode, swap_range)
        except ValueError as e:
            QMessageBox.warning(self, "Deletion Error", str(e))
            return
        
        self._on_track_data_changed()

    def _interpolate_track_wrapper(self):
        if not self._track_edit_blocker():
            return
        
        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "Track Not Interpolated", "No track is selected.")
            return
        
        selected_instance_idx = self.selected_box.instance_id if self.selected_box else current_frame_inst[0]
        self._save_state_for_undo()
        
        iter_frame_idx = self.current_frame_idx + 1
        frames_to_interpolate = []
        while np.all(np.isnan(self.pred_data_array[iter_frame_idx, selected_instance_idx, :])):
            frames_to_interpolate.append(iter_frame_idx)
            iter_frame_idx += 1
            if iter_frame_idx >= self.total_frames:
                QMessageBox.information(self, "Interpolation Failed", "No valid subsequent keypoint data found for this instance to interpolate to.")
                return
       
        if not frames_to_interpolate:
            QMessageBox.information(self, "Interpolation Info", "No gaps found to interpolate for the selected instance.")
            return
        
        frames_to_interpolate.sort()
        self.pred_data_array = dute.interpolate_track(self.pred_data_array, frames_to_interpolate, selected_instance_idx)
        self._on_track_data_changed()

    def _generate_track_wrapper(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return False
        
        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        missing_instances = [inst for inst in range(self.dlc_data.instance_count) if inst not in current_frame_inst]
        if not missing_instances:
            QMessageBox.information(self, "No Missing Instances", "No missing instances found in the current frame to fill.")
            return
        
        self.pred_data_array = dute.generate_track(
            self.pred_data_array, self.current_frame_idx, missing_instances, self.dlc_data.num_keypoint)
        self._on_track_data_changed()

    def _interpolate_missing_kp_wrapper(self):
        if not self._track_edit_blocker():
            return
        
        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "Track Not Interpolated", "No track is selected.")
            return
        
        selected_instance_idx = self.selected_box.instance_id if self.selected_box else current_frame_inst[0]
        self._save_state_for_undo()

        self.pred_data_array = dute.interpolate_missing_keypoints(self.pred_data_array, self.current_frame_idx, selected_instance_idx)
        self._on_track_data_changed()

    ###################################################################################################################################################  
    
    def _on_track_data_changed(self):
        self.selected_box = None
        self.is_saved = False
        self.check_instance_count_per_frame()
        self.display_current_frame()

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.current_frame_idx = new_frame_idx
        self.display_current_frame()
        self.navigation_title_controller()

    def _refresh_slider(self):
        self.progress_widget.set_frame_category("Refined frames", self.refined_roi_frame_list, "#009979", priority=7)
        self.progress_widget.set_frame_category("ROI frames", self.roi_frame_list, "#F04C4C") # Update ROI frames

    def _handle_box_selection(self, clicked_box):
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

    ###################################################################################################################################################
    
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
            self.pred_data_array = dute.clean_inconsistent_nans(self.pred_data_array) # Cleanup ghost points (NaN for x,y yet non-nan in confidence)

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
    
    def mark_all_as_refined(self):
        if not self.marked_roi_frame_list:
            QMessageBox.information(
                self,  "Action Not Available", 
                "To begin keypoint refinement, you must first select and designate frames" \
                " using the Extractor tool. The Refiner tool is open, but only for track " \
                "refinement, which uses the automatically tagged frames to assist with track " \
                "swapping, interpolation, and other tracking tasks."
            )
            return

        self.refined_roi_frame_list = self.marked_roi_frame_list
        self._refresh_slider()
        self.navigation_title_controller()

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
        
        pred_file_to_save_path = dio.determine_save_path(self.prediction, suffix="_track_refiner_modified_")
        status, msg = dio.save_prediction_to_h5(pred_file_to_save_path, self.pred_data_array)
        
        if not status:
            QMessageBox.critical(self, "Saving Error", f"An error occurred during saving: {msg}")
            print(f"An error occurred during saving: {msg}")
            return
        
        self.reload_prediction(pred_file_to_save_path)

        QMessageBox.information(self, "Save Successful", str(msg))

        self.is_saved = True
        self.prediction_saved.emit(self.prediction) # Emit the signal with the saved file path
        self.refined_frames_exported.emit(self.refined_roi_frame_list)

    def reload_prediction(self, pred_file_to_save_path):
        self.prediction = pred_file_to_save_path
        self.data_loader.dlc_config_filepath = self.dlc_data.dlc_config_filepath
        self.data_loader.prediction_filepath = self.prediction
        self.dlc_data, msg = self.data_loader.load_data()

    def save_prediction_as_csv(self):
        save_path = os.path.dirname(self.prediction)
        pred_file = os.path.basename(self.prediction).split(".")[0]
        exp_set = Export_Settings(self.video_file, self.video_name, save_path, "CSV")
        try:
            dio.prediction_to_csv(self.dlc_data, self.dlc_data.pred_data_array, exp_set)
            msg = f"Successfully saved modified prediction in csv to: {os.path.join(save_path, pred_file)}.csv"
            QMessageBox.information(self, "Save Successful", str(msg))
        except Exception as e:
            QMessageBox.critical(self, "Saving Error", f"An error occurred during csv saving: {e}")
            print(f"An error occurred during csv saving: {e}")

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.reset_zoom()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        dugh.handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_prediction)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DLC_Track_Refiner()
    window.show()
    app.exec()