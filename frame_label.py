import os
import ast
import pandas as pd
import numpy as np
import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QPainter, QColor, QPen, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QGraphicsView, QGraphicsRectItem

import ui
from ui import (
    Menu_Widget, Progress_Bar_Widget, Nav_Widget,
    Adjust_Property_Dialog, Pose_Rotation_Dialog, Canonical_Pose_Dialog, Head_Tail_Dialog,
    Prediction_Plotter, Selectable_Instance, Draggable_Keypoint
)
from utils.dataclass import Export_Settings, Plot_Config, Refiner_Plotter_Callbacks
import utils.io as dio
import utils.helper as duh
import utils.track_edit as dute

DLC_CONFIG_DEBUG = "D:/Project/DLC-Models/NTD/config.yaml"
VIDEO_FILE_DEBUG = "D:/Project/DLC-Models/NTD/videos/job/20250709S-340.mp4"
PRED_FILE_DEBUG = "D:/Project/DLC-Models/NTD/videos/job/20250709S-340DLC_HrnetW32_bezver-SD-20250605M-cam52025-06-26shuffle3_detector_510_snapshot_390_el.h5"

# Todo:
#   Add support for use cases where individual counts exceed 2

class Frame_Label(QtWidgets.QMainWindow):
    prediction_saved = Signal(str)          # Signal to emit the path of the saved prediction file
    refined_frames_exported = Signal(list)  # Signal to emit the refined_roi_frame_list back to Extractor

    def __init__(self):
        super().__init__()

        self.is_debug = False
        self.setWindowTitle(ui.format_title("Frame Labeler", self.is_debug))
        self.setGeometry(100, 100, 1200, 960)

        self.setup_menu()

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.setup_graphicscene()

        self.progress_widget = Progress_Bar_Widget()
        self.progress_widget.frame_changed.connect(self._handle_frame_change_from_comp)
        self.layout.addWidget(self.progress_widget)

        self.nav_widget = Nav_Widget(mark_name="ROI Frame")
        self.layout.addWidget(self.nav_widget)
        self.nav_widget.set_collapsed(True)

        # Connect buttons to events
        self.nav_widget.frame_changed_sig.connect(self.change_frame)
        self.nav_widget.prev_marked_frame_sig.connect(lambda:self._navigate_roi_frames("prev"))
        self.nav_widget.next_marked_frame_sig.connect(lambda:self._navigate_roi_frames("next"))

        self.setup_shortcut()
        self.reset_state()

    def setup_menu(self):
        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        refiner_menu_config = {
            "File": {
                "buttons": [
                    ("Load Video", self.load_video),
                    ("Load Prediction", self.load_prediction)
                ]
            },
            "Refiner": {
                "display_name": "Refine",
                "buttons": [
                    ("Direct Keypoint Edit (Q)", self.direct_keypoint_edit),
                    {
                        "submenu": "Interpolate",
                        "display_name": "Interpolate",
                        "items": [
                            ("Interpolate Selected Instance on Current Frame (T)", self._interpolate_track_wrapper),
                            ("Interpolate Missing Keypoints for Selected Instance (Shift + T)", self._interpolate_missing_kp_wrapper),
                            ("Interpolate Selected Instance Across All Frames", self.interpolate_all),     
                        ]
                    },
                    {
                        "submenu": "Delete",
                        "display_name": "Delete",
                        "items": [
                            ("Delete Selected Instance On Current Frame (X)", lambda:self._delete_track_wrapper("point")),
                            ("Delete Selected Instance Until Next ROI (Shift + X)", lambda:self._delete_track_wrapper("batch")),
                            ("Delete Instances Below Set Confidence Across All Frames ", self.purge_inst_by_conf),
                            ("Delete All Prediction Inside Selected Area", self.designate_no_mice_zone),
                            ("Delete Abnormal Instances", self.purge_ghost_tracks)
                        ]
                    },
                    {
                        "submenu": "Swap",
                        "display_name": "Swap",
                        "items": [
                            ("Swap Instances On Current Frame (W)", lambda:self._swap_track_wrapper("point")),
                            ("Swap Until The End (Shift + W)", lambda:self._swap_track_wrapper("batch"))
                        ]
                    },
                    {
                        "submenu": "Correct",
                        "display_name": "Auto Correction",
                        "items": [
                            ("Correct Track Using Temporal Consistency", self.correct_track_using_temporal),
                            ("Correct Track Using Idtrackerai Trajectories", self.correct_track_using_idtrackerai),
                        ]
                    },
                    ("Generate Instance (G)", self._generate_track_wrapper),
                    ("Rotate Selected Instance (R)", self._rotate_track_wrapper),
                ]
            },
            "View": {
                "buttons": [
                    ("Toggle Zoom Mode (Z)", self.toggle_zoom_mode, {"checkable": True, "checked": False}),
                    ("Toggle Snap to Instances (E)", self.toggle_snap_to_instances, {"checkable": True, "checked": False}),
                    ("Reset Zoom", self.reset_zoom),
                    ("Hide Text Labels", self.toggle_hide_text_labels, {"checkable": True, "checked": False}),
                    ("View Canonical Pose", self.view_canonical_pose)
                ]
            },
            "Preference": {
                "buttons": [
                    ("Undo Changes", self.undo_changes),
                    ("Redo Changes", self.redo_changes),
                    ("Remove Current Frame From Refine Task", self.this_frame_is_beyond_savable),
                    ("Mark All As Refined", self.mark_all_as_refined),
                    {
                        "submenu": "Adjust",
                        "display_name": "Adjust",
                        "items": [
                            ("Point Size", self.adjust_point_size),
                            ("Plot Visibility", self.adjust_plot_opacity)
                        ]
                    },
                ]
            },
            "Save": {
                "buttons": [
                    ("Save Prediction", self.save_prediction),
                    ("Save Prediction Into CSV", self.save_prediction_as_csv)
                ]
            }
        }
        if self.is_debug:
            refiner_menu_config["DEBUG"] = {
                "buttons": [
                    ("Debug Load", self.debug_load),
                    ("Paste Frame Marks From Clipboard", self.load_marked_roi_from_clipboard),
                    ("Empty Marked ROI", self.clear_marked_roi)
                ]
            }
        self.menu_widget.add_menu_from_config(refiner_menu_config)

    def setup_graphicscene(self):
        """Graphics view for interactive elements and video display"""
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

        self.graphics_view.mousePressEvent = self.graphics_view_mouse_press_event
        self.graphics_view.mouseMoveEvent = self.graphics_view_mouse_move_event
        self.graphics_view.mouseReleaseEvent = self.graphics_view_mouse_release_event
        self.graphics_scene.parent = lambda: self # Allow items to access the main window

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
        QShortcut(QKeySequence(Qt.Key_R), self).activated.connect(self._rotate_track_wrapper)
        QShortcut(QKeySequence(Qt.Key_Q), self).activated.connect(self.direct_keypoint_edit)
        QShortcut(QKeySequence(Qt.Key_Backspace), self).activated.connect(self.delete_dragged_keypoint)

        QShortcut(QKeySequence(Qt.Key_Z | Qt.ControlModifier), self).activated.connect(self.undo_changes)
        QShortcut(QKeySequence(Qt.Key_Y | Qt.ControlModifier), self).activated.connect(self.redo_changes)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_prediction)
        QShortcut(QKeySequence(Qt.Key_Z), self).activated.connect(self.toggle_zoom_mode)
        QShortcut(QKeySequence(Qt.Key_E), self).activated.connect(self.toggle_snap_to_instances)

    def reset_state(self):
        self.video_file, self.prediction, self.video_name = None, None, None
        self.dlc_data, self.canon_pose, self.angle_map_data = None, None, None
        self.instance_count_per_frame, self.pred_data_array = None, None
        self.data_loader = dio.DLC_Loader(None, None)

        self.roi_frame_list, self.marked_roi_frame_list, self.refined_roi_frame_list = [], [], []
        self.cap, self.current_frame = None, None
        self.selected_box, self.dragged_keypoint = None, None
        self.start_point, self.current_rect_item = None, None

        self.is_kp_edit, self.is_drawing_zone, self.is_zoom_mode = False, False, False

        self.undo_stack, self.redo_stack = [], []
        self.max_undo_stack_size = 100
        self.is_saved = True

        self.plot_config = Plot_Config(
            plot_opacity=1.0, point_size = 6.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False)
        self.plotter_callback = Refiner_Plotter_Callbacks(
            keypoint_coords_callback = self._update_keypoint_position, keypoint_object_callback = self.set_dragged_keypoint,
            box_coords_callback = self._update_instance_position, box_object_callback = self._handle_box_selection
        )
        self.zoom_factor = 1.0

        self.auto_snapping = False

        self.nav_widget.set_collapsed(True)
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

        self.dlc_data = ui.load_and_show_message(self, self.data_loader)
        self.initialize_loaded_data()

    def debug_load(self):
        self.video_file = VIDEO_FILE_DEBUG
        self.initialize_loaded_video()
        self.data_loader.dlc_config_filepath = DLC_CONFIG_DEBUG
        self.data_loader.prediction_filepath = self.prediction = PRED_FILE_DEBUG
        self.dlc_data = ui.load_and_show_message(self, self.data_loader)
        self.initialize_loaded_data()

    def initialize_loaded_data(self):
        self.prediction = self.dlc_data.prediction_filepath
        self.pred_data_array = self.dlc_data.pred_data_array
        head_idx, tail_idx = duh.infer_head_tail_indices(self.dlc_data.keypoints)

        if head_idx is None or tail_idx is None:
            dialog = Head_Tail_Dialog(self.dlc_data.keypoints)
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                head_idx, tail_idx = dialog.get_selected_indices()

        self.canon_pose, all_frame_pose = duh.calculate_canonical_pose(self.pred_data_array, head_idx, tail_idx)
        self.angle_map_data = duh.build_angle_map(self.canon_pose, all_frame_pose, head_idx, tail_idx)

        self._update_roi_list()

        self.plotter = Prediction_Plotter(
            dlc_data=self.dlc_data, current_frame_data=self.pred_data_array[self.current_frame_idx, ...],
            graphics_scene=self.graphics_scene, plot_config=self.plot_config, plot_callback=self.plotter_callback)

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
                
                if self.auto_snapping:
                    view_width = self.graphics_scene.sceneRect().width()
                    view_height = self.graphics_scene.sceneRect().height()
                    current_frame_data = self.pred_data_array[self.current_frame_idx, ...]
                    if not np.all(np.isnan(current_frame_data)):
                        self.zoom_factor, center_x, center_y = \
                            duh.calculate_snapping_zoom_level(current_frame_data, view_width, view_height)
                        self.graphics_view.centerOn(center_x, center_y)

                new_transform = QtGui.QTransform()
                new_transform.scale(self.zoom_factor, self.zoom_factor)
                self.graphics_view.setTransform(new_transform)

                # Plot predictions (keypoints, bounding boxes, skeleton)
                if self.pred_data_array is not None:
                    self.plotter.current_frame_data = self.pred_data_array[self.current_frame_idx, ...]
                    self.plotter.plot_config = self.plot_config
                    self.plotter.plot_predictions()

                self.graphics_view.update() # Force update of the graphics view
                self.progress_widget.set_current_frame(self.current_frame_idx) # Update slider handle's position

            else: # If video frame cannot be read, clear scene and display error
                self.graphics_scene.clear()
                error_text_item = self.graphics_scene.addText("Error: Could not read frame")
                error_text_item.setDefaultTextColor(QColor(255, 255, 255)) # White text
                self.graphics_view.fitInView(error_text_item.boundingRect(), Qt.KeepAspectRatio)

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

    def toggle_snap_to_instances(self):
        self.auto_snapping = not self.auto_snapping
        self.display_current_frame()

    def toggle_hide_text_labels(self):
        self.plot_config.hide_text_labels = not self.plot_config.hide_text_labels
        self.display_current_frame()

    def adjust_point_size(self):
        dialog = Adjust_Property_Dialog(
            property_name="Point Size", property_val=self.plot_config.point_size, range=(0.1, 10.0), parent=self)
        dialog.property_changed.connect(self._update_point_size)
        dialog.show()

    def adjust_plot_opacity(self):
        dialog = Adjust_Property_Dialog(
            property_name="Point Opacity", property_val=self.plot_config.plot_opacity, range=(0.00, 1.00), parent=self)
        dialog.property_changed.connect(self._update_plot_opacity)
        dialog.show()

    def _update_point_size(self, value):
        self.plot_config.point_size = value
        self.display_current_frame()

    def _update_plot_opacity(self, value):
        self.plot_config.plot_opacity = value
        self.display_current_frame()

    ###################################################################################################################################################

    def _update_roi_list(self):
        self.instance_count_per_frame = dute.get_instance_count_per_frame(self.pred_data_array)
        if self.marked_roi_frame_list:
            self.roi_frame_list = list(self.marked_roi_frame_list)
        else:
            roi_frames = np.where(np.diff(self.instance_count_per_frame)!=0)[0]+1
            self.roi_frame_list = list(roi_frames)

        self._refresh_slider()

    def load_marked_roi_from_clipboard(self):
        df = pd.read_clipboard(header=None, sep='\0', engine='python', names=['data'])
        if pd.isna(df['data'].iloc[0]):
            return
        
        raw_text = str(df['data'].iloc[0]).strip()

        if not raw_text.startswith("["):
            raw_text = "[" + raw_text
        if not raw_text.endswith("]"):
            raw_text += "]"

        frame_list = ast.literal_eval(raw_text)
        if not frame_list:
            return
        
        new_frame_set = set([int(x) for x in frame_list if isinstance(x, (int, float)) and not pd.isna(x)])
        self.marked_roi_frame_list[:] = list(new_frame_set)
        self._update_roi_list()
        self._refresh_slider()

    def clear_marked_roi(self):
        self.marked_roi_frame_list.clear()
        self._update_roi_list()
        self._refresh_slider()

    def _navigate_roi_frames(self, mode):
        ui.navigate_to_marked_frame(self, self.roi_frame_list, self.current_frame_idx, self._handle_frame_change_from_comp, mode)
        
    ###################################################################################################################################################

    def purge_inst_by_conf(self):
        if not self._track_edit_blocker():
            return
        confidence_threshold, ok = QtWidgets.QInputDialog.getDouble(
            self,"Set Confidence Threshold","Delete all instances below this confidence:",
            value=0.5, minValue=0.0, maxValue=1.0, decimals=2
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
                self.pred_data_array, confidence_threshold, bodypart_threshold)
            QMessageBox.information(self, "Deletion Complete", f"Deleted {removed_instances_count} instances from {removed_frames_count} frames.")

            self._on_track_data_changed()
            self.reset_zoom()
        else:
            QMessageBox.information(self, "Deletion Cancelled", "Deletion cancelled by user.")

    def purge_ghost_tracks(self):
        if not self._track_edit_blocker():
            return
        
        self._save_state_for_undo()
        self.pred_data_array = dute.ghost_prediction_buster(self.pred_data_array, self.canon_pose, debug_print=self.is_debug)
        self._on_track_data_changed()
        self.reset_zoom()

    def interpolate_all(self):
        if not self._track_edit_blocker():
            return
        
        if not self.selected_box:
            QMessageBox.information(self, "No Instance Selected", "Please select a track to interpolate all frames for one instance.")
            return
        
        max_gap_allowed, ok = QtWidgets.QInputDialog.getInt(
            self,"Set Max Gap For Interpolation","Will not interpolate gap beyond this limit, set to 0 to ignore the limit.",
            value=10, minValue=0, maxValue=1000
        )

        if not ok:
            QMessageBox.information(self, "Input Cancelled", "Max Gap input cancelled.")
            return

        instance_to_interpolate = self.selected_box.instance_id
        self._save_state_for_undo() # Save state before modification

        self.pred_data_array = dute.interpolate_track_all(self.pred_data_array, instance_to_interpolate, max_gap_allowed)

        self._on_track_data_changed()
        self.reset_zoom()
        QMessageBox.information(self, "Interpolation Complete", f"All frames interpolated for instance {self.dlc_data.individuals[instance_to_interpolate]}.")
    
    def designate_no_mice_zone(self):
        if not self._track_edit_blocker():
            return
        self.is_drawing_zone = True
        self.graphics_view.setCursor(Qt.CrossCursor)
        QMessageBox.information(self, "Designate No Mice Zone", "Click and drag on the video to select a zone. Release to apply.")

    def correct_track_using_temporal(self):
        if not self._track_edit_blocker():
            return

        self._save_state_for_undo()

        dialog = "Fixing track using temporal consistency..."
        title = f"Fix Track Using Temporal"
        progress = ui.get_progress_dialog(self, 0, self.total_frames, title, dialog)

        self.pred_data_array = dute.ghost_prediction_buster(self.pred_data_array, self.canon_pose)
        self.pred_data_array, _, _ = dute.purge_by_conf_and_bp(
            self.pred_data_array, confidence_threshold=0.5, bodypart_threshold=0)

        self.pred_data_array, changes_applied = dute.track_correction(
            pred_data_array=self.pred_data_array,
            idt_traj_array=None,
            progress=progress,
            debug_status=self.is_debug
            )
        progress.close()

        if not changes_applied:
            QMessageBox.information(self, "No Changes Applied", "No changes were applied.")
            return

        QMessageBox.information(self, "Track Correction Finished", f"Applied {changes_applied} changes to the current track.")

        self._on_track_data_changed()

    def correct_track_using_idtrackerai(self):
        if not self._track_edit_blocker():
            return
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not folder_path:
            return
        
        idt_csv = os.path.join(folder_path, "trajectories", "trajectories_csv", "trajectories.csv")
        conf_csv = os.path.join(folder_path, "trajectories", "trajectories_csv", "id_probabilities.csv")
        if not os.path.isfile(idt_csv):
            QMessageBox.warning(self, "", "")
        df_idt = pd.read_csv(idt_csv, header=0)
        df_conf = pd.read_csv(conf_csv, header=0)
        idt_traj_array = dio.parse_idt_df_into_ndarray(df_idt, df_conf)

        self._save_state_for_undo()

        dialog = "Fixing track from idTracker.ai trajectories..."
        title = f"Fix Track Using idTracker.ai"
        progress = ui.get_progress_dialog(self, 0, self.total_frames, title, dialog)

        self.pred_data_array, changes_applied = dute.track_correction(
            pred_data_array=self.pred_data_array,
            idt_traj_array=idt_traj_array,
            progress=progress,
            debug_status=self.is_debug,
            )

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
        self.plot_config.edit_mode = self.is_kp_edit
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
        self.display_current_frame() # Refresh the frame to get the kp edit status through to plotter

    def _update_keypoint_position(self, instance_id, keypoint_id, new_x, new_y):
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

    def _update_instance_position(self, instance_id, dx, dy):
        self._save_state_for_undo()
        if self.current_frame_idx in self.marked_roi_frame_list and self.current_frame_idx not in self.refined_roi_frame_list:
            self.refined_roi_frame_list.append(self.current_frame_idx)
        
        for kp_idx in range(self.dlc_data.num_keypoint): # Update all keypoints for the given instance in the current frame
            x_coord_idx, y_coord_idx = kp_idx * 3, kp_idx * 3 + 1
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

    def set_dragged_keypoint(self, keypoint_item:Draggable_Keypoint):
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
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return

        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "No Instance Seleted",
                "When there are more than one instance present, "
                "you need to click one of the instance bounding box to specify which to delete.")
            return
        
        self._save_state_for_undo()
        selected_instance_idx = self.selected_box.instance_id if self.selected_box else current_frame_inst[0]
        try:
            self.pred_data_array = dute.delete_track(self.pred_data_array, self.current_frame_idx,
                                        selected_instance_idx, mode, deletion_range)
        except ValueError as e:
            QMessageBox.warning(self, "Deletion Error", str(e))
            return
        
        self._on_track_data_changed()
        
    def _swap_track_wrapper(self, mode, swap_range=None):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
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
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return
        
        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "Track Not Interpolated", "No Instance is selected.")
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

        self._save_state_for_undo()

        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        missing_instances = [inst for inst in range(self.dlc_data.instance_count) if inst not in current_frame_inst]
        if not missing_instances:
            QMessageBox.information(self, "No Missing Instances", "No missing instances found in the current frame to fill.")
            return
        
        self.pred_data_array = dute.generate_track(self.pred_data_array, self.current_frame_idx, missing_instances,
            angle_map_data=self.angle_map_data)
        self._on_track_data_changed()

    def _rotate_track_wrapper(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return
        
        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "Track Not Rotated", "No Instance is selected.")
            return
        
        selected_instance_idx = self.selected_box.instance_id if self.selected_box else current_frame_inst[0]

        _, local_coords = duh.calculate_pose_centroids(self.pred_data_array, self.current_frame_idx)
        local_x = local_coords[selected_instance_idx, 0::2]
        local_y = local_coords[selected_instance_idx, 1::2]
        current_rotation = np.degrees(duh.calculate_pose_rotations(local_x, local_y, angle_map_data=self.angle_map_data))
        if np.isnan(current_rotation) or np.isinf(current_rotation):
            current_rotation = 0.0
        else:
            current_rotation = current_rotation % 360.0 

        self._save_state_for_undo()

        self.rotation_dialog = Pose_Rotation_Dialog(selected_instance_idx, current_rotation, parent=self)
        self.rotation_dialog.rotation_changed.connect(self._on_rotation_changed)
        self.rotation_dialog.show()

    def _interpolate_missing_kp_wrapper(self):
        if self.pred_data_array is None:
            QMessageBox.warning(self, "Error", "Prediction data not loaded. Please load a prediction file first.")
            return
        
        current_frame_inst = duh.get_current_frame_inst(self.dlc_data, self.pred_data_array, self.current_frame_idx)
        if len(current_frame_inst) > 1 and not self.selected_box:
            QMessageBox.information(self, "Track Not Interpolated", "No Instance is selected.")
            return
        
        selected_instance_idx = self.selected_box.instance_id if self.selected_box else current_frame_inst[0]
        self._save_state_for_undo()

        self.pred_data_array = dute.interpolate_missing_keypoints(self.pred_data_array, self.current_frame_idx, selected_instance_idx,
            angle_map_data=self.angle_map_data)
        self._on_track_data_changed()

    ###################################################################################################################################################  
    
    def _on_track_data_changed(self):
        self.selected_box = None
        self.is_saved = False
        self._update_roi_list()
        self.display_current_frame()

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.current_frame_idx = new_frame_idx
        self.display_current_frame()
        self.navigation_title_controller()

    def _on_rotation_changed(self, selected_instance_idx, angle_delta: float):
        angle_delta = np.radians(angle_delta)
        self.pred_data_array = dute.rotate_track(self.pred_data_array, self.current_frame_idx,
            selected_instance_idx, angle=angle_delta)
        self._on_track_data_changed()

    def _refresh_slider(self):
        self.progress_widget.set_frame_category("Refined frames", self.refined_roi_frame_list, "#009979", priority=7)
        self.progress_widget.set_frame_category("ROI frames", self.roi_frame_list, "#F04C4C") # Update ROI frames

    def _handle_box_selection(self, clicked_box:Selectable_Instance):
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
            self._update_roi_list()
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

    def view_canonical_pose(self):
        dialog = Canonical_Pose_Dialog(self.dlc_data, self.canon_pose)
        dialog.exec()

    def mark_all_as_refined(self):
        if not self.manual_refinement_check():
            return

        self.refined_roi_frame_list = self.marked_roi_frame_list
        self._refresh_slider()
        self.navigation_title_controller()

    def this_frame_is_beyond_savable(self):
        if not self.manual_refinement_check():
            return
        
        if self.current_frame_idx in self.marked_roi_frame_list:
            self.marked_roi_frame_list.remove(self.current_frame_idx)
            self.roi_frame_list.remove(self.current_frame_idx)
        
        if self.current_frame_idx in self.refined_roi_frame_list:
            self.refined_roi_frame_list.remove(self.current_frame_idx)

        self._refresh_slider()
        self.navigation_title_controller()

    def manual_refinement_check(self):
        if not self.marked_roi_frame_list:
            QMessageBox.information(
                self,  "Action Not Available", 
                "To begin keypoint refinement, you must first select and designate frames" \
                " using the Extractor tool. The Refiner tool is open, but only for track " \
                "refinement, which uses the automatically tagged frames to assist with track " \
                "swapping, interpolation, and other tracking tasks."
            )
            return False

        return True

    def undo_changes(self):
        if self.undo_stack:
            self.redo_stack.append(self.pred_data_array.copy())
            self.pred_data_array = self.undo_stack.pop()
            self._update_roi_list()
            self.display_current_frame()
            self.is_saved = False
            print("Undo performed.")
        else:
            QMessageBox.information(self, "Undo", "Nothing to undo.")

    def redo_changes(self):
        if self.redo_stack:
            self.undo_stack.append(self.pred_data_array.copy())
            self.pred_data_array = self.redo_stack.pop()
            self._update_roi_list()
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
        status, msg = dio.save_prediction_to_existing_h5(pred_file_to_save_path, self.pred_data_array)
        
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
        ui.handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_prediction)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = Frame_Label()
    window.show()
    app.exec()