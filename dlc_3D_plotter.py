import os
import glob

import scipy.io as sio
import pickle

import numpy as np
import pandas as pd
from itertools import combinations
import cv2

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from utils.dtu_io import DLC_Loader
from utils.dtu_widget import Menu_Widget, Progress_Bar_Widget, Nav_Widget
from utils.dtu_comp import Clickable_Video_Label, Adjust_Property_Dialog
from utils.dtu_triangulation import Data_Processor_3D
import utils.dtu_io as dio
import utils.dtu_helper as duh
import utils.dtu_gui_helper as dugh
import utils.dtu_track_edit as dute
import utils.dtu_triangulation as dutri

import traceback

# Todo: Add support fot sleap-anipose / anipose toml calibration file

DLC_CONFIG_DEBUG = "D:/Project/DLC-Models/COM3D/config.yaml"
CALIB_FILE_DEBUG = "D:/Project/SDANNCE-Models/4CAM-250620/SD-20250705-MULTI/sync_dannce.mat"
VIDEO_FOLDER_DEBUG = "D:/Project/SDANNCE-Models/4CAM-250620/SD-20250705-MULTI/Videos"

class DLC_3D_plotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_debug = False
        self.setWindowTitle(duh.format_title("DLC 3D Plotter", self.is_debug))
        self.setGeometry(100, 100, 1600, 960)

        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        plotter_3d_menu_config = {
            "File": {
                "display_name": "Load",
                "buttons": [
                    ("Load DLC Configs", self.load_dlc_config),
                    ("Load Calibrations", self.load_calibrations),
                    ("Load Videos and Predictions", self.open_video_folder_dialog),
                    ("Load Workspace", self.load_workspace)
                ]
            },
            "Detect": {
                "display_name": "Swap Detect",
                "buttons": [
                    ("Calculate Track Swap Score", lambda:self.calculate_identity_swap_score(mode="full")),
                    ("Calculate Temporal Velocity", self.calculate_temporal_vel),
                    ("Adjust Confidence Cutoff", self.show_confidence_dialog),
                    ("Adjust Deviance Threshold", self.show_deviance_dialog),
                    ("Adjust Velocity Threshold", self.show_velocity_dialog),
                ]
            },
            "View": {
                "display_name": "View",
                "buttons": [
                    ("Change Marked Frame View Mode", self.change_mark_view_mode),
                    ("Reset Marked Frames", self.reset_marked_frames),
                    ("Check Camera Geometry", self.plot_camera_geometry),
                    ("Auto 3D View Perspective", self.toggle_auto_3d_perspective, {"checkable": True, "checked": True})
                ]
            },
            "Track": {
                "display_name": "Track Refine",
                "buttons": [
                    ("Automatic Track Correction", self.automatic_track_correction),
                    ("Manual Swap Selected View (X)", self.manual_swap_frame_view),
                    ("Call Track Refiner", self.call_track_refiner)
                ]
            },
            "Save": {
                "display_name": "Save",
                "buttons": [
                    ("Save Workspace", self.save_workspace),
                    ("Save Swapped Track", self.save_swapped_prediction),
                    ("Export COM to SDANNCE", self.export_COM_to_SDANNCE)
                ]
            },
        }
        if self.is_debug:
            plotter_3d_menu_config["File"]["buttons"].append(("Debug Load", self.debug_load))
        self.menu_widget.add_menu_from_config(plotter_3d_menu_config)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.display_layout = QtWidgets.QHBoxLayout()
        self.video_layout = QtWidgets.QGridLayout()
        
        self.video_labels = [] # Store video labels in a list for easy access
        for row in range(2):
            for col in range(2):
                cam_idx = row * 2 + col # 0-indexed camera index
                label = Clickable_Video_Label(cam_idx, self) # Use the custom label
                label.setText(f"Video {cam_idx }")
                label.setAlignment(Qt.AlignCenter) # Center the "Video X" text
                label.setFixedSize(480, 360) # Set a fixed size for video display
                label.setStyleSheet("border: 1px solid gray;") # Add a border for visibility
                self.video_layout.addWidget(label, row, col)
                self.video_labels.append(label)
                label.clicked.connect(self.set_selected_camera) # Connect the clicked signal

        # Store 3D plot
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setMouseTracking(True)
        self.plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.display_layout.addLayout(self.video_layout)
        self.display_layout.addLayout(self.plot_layout)
        self.layout.addLayout(self.display_layout, 1)

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
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(lambda:self._navigate_marked_frames("prev"))
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(lambda:self._navigate_marked_frames("next"))
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.progress_widget.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_workspace)
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self.manual_swap_frame_view)
        self.canvas.mpl_connect("scroll_event", self.on_scroll_3d_plot)

        self.reset_state()

    def reset_state(self):
        self.num_cam = None

        self.confidence_cutoff = 0.6
        self.deviance_threshold = 50
        self.velocity_threshold = 20

        self.data_loader = DLC_Loader(None, None)
        self.dlc_data = None
        self.keypoint_to_idx = {}

        self.base_folder, self.calibration_filepath, self.dlc_config_filepath = None, None, None
        self.video_list, self.cap_list, self.prediction_list, self.camera_params = [], [], [], []
        self.cam_pos, self.cam_dir = None, None

        self.pred_data_array = None # Combined prediction data for all cameras
        self.swap_detection_score_array = None
        self.temporal_dist_array_all = None

        self.num_cam_from_calib = None
        self.correction_progress = None

        self.plot_lim = 150
        self.instance_color = [
            (255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)] # RGB
        
        self.current_frame_idx = 0
        self.total_frames = 0
        self.selected_cam_idx = None

        self.refiner_window = None

        self.view_mode_choice = ["ROI Frames", "Failed Frames", "Skipped Frames"]
        self.current_view_mode_idx = 0

        self.is_saved = True
        self.auto_perspective = True

        self.roi_frame_list, self.failed_frame_list, self.skipped_frame_list = [], [], []
        self.check_range = 100
        self._refresh_slider()

        self.ax.clear()
        self.ax.set_title("3D Skeleton Plot - No DLC data loaded")
        self.canvas.draw_idle()

    def debug_load(self):
        self.dlc_config_loader(DLC_CONFIG_DEBUG)
        self.calibration_loader(CALIB_FILE_DEBUG)
        self.load_video_folder(VIDEO_FOLDER_DEBUG)

    def load_workspace(self):
        self.reset_state()

        workspace_filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Workspace", "", "Workspace Files (*.pickle)")
        if not workspace_filepath:
            return
        
        print(f"LOADER | Workspace loaded: {workspace_filepath}")
        
        try:
            with open(workspace_filepath, "rb") as f:
                state = pickle.load(f)

            self.calibration_filepath = state["calibration_filepath"]
            self.dlc_config_filepath = state["dlc_config_filepath"]
            self.base_folder = state["base_folder"]

            self.calibration_loader(self.calibration_filepath)
            self.dlc_config_loader(self.dlc_config_filepath)
            self.load_video_folder(self.base_folder)

            for key, value in state.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"LOADER | Warning: Loaded unknown attribute '{key}' — not restored.")

            self.display_current_frame()
            self._refresh_slider()
            self.navigation_title_controller()

            QMessageBox.information(self, "Load Successful", f"Workspace loaded from {workspace_filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not load workspace: {e}")

    def open_video_folder_dialog(self):
        if self.dlc_data is None: # Check if dlc_data is loaded
            QMessageBox.warning(self, "Warning", "DLC config is not loaded, load DLC config first!")
            print("LOADER | DLC config is not loaded, load DLC config first!")
            self.load_dlc_config()
            if self.dlc_data is None: # User closed DLC loading window or failed to load
                return
        if self.num_cam_from_calib is None:
            QMessageBox.warning(self, "Warning", "Calibrations are not loaded, load calibrations first!")
            print("LOADER | Calibrations are not loaded, load calibrations first!")
            self.load_calibrations()
            if self.num_cam_from_calib is None: # User closed calibration loading window or failed to load
                return

        self.reset_state()

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder_path:
            self.load_video_folder(folder_path)

    def load_dlc_config(self):
        file_dialog = QtWidgets.QFileDialog(self)
        dlc_config_filepath, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        if dlc_config_filepath:
            self.dlc_config_loader(dlc_config_filepath)

    def dlc_config_loader(self, dlc_config_filepath: str):
        self.data_loader.dlc_config_filepath = dlc_config_filepath
        try:
            self.dlc_data = dugh.load_and_show_message(self, self.data_loader, metadata_only=True)
            self.keypoint_to_idx = {name: idx for idx, name in enumerate(self.dlc_data.keypoints)}
        except:
            QMessageBox.critical(self, "Error", "Failed to load DLC config.")
            traceback.print_exc()
        self.dlc_config_filepath = dlc_config_filepath # Store the DLC config file path for saving

    def load_calibrations(self):
        calib_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Calibration File", "", "Calibration Files (*.mat)")
        if calib_file:
            print(f"LOADER | Calibration loaded: {calib_file}")
            self.calibration_loader(calib_file)

    def calibration_loader(self, calibration_file):
        try:
            calib = sio.loadmat(calibration_file)
            if not self.is_debug:
                QMessageBox.information(self, "Success", "Calibrations loaded successfully!")
            self.num_cam_from_calib = calib["params"].size
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Calibration file not found: {calibration_file}")
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load calibration file: {e}")
            traceback.print_exc()
            return
        self.calibration_filepath = calibration_file # Store the calibration file path for later saving process
        self.camera_params = [{} for _ in range(self.num_cam_from_calib)]
        self.parse_calibration_mat(calib)
        self.plot_camera_geometry()

    def load_video_folder(self, folder_path):
        if self.data_loader.dlc_config_filepath is None:
            QMessageBox.warning(self, "Warning", "DLC config is not loaded. Please load them first.")
            return

        self.num_cam = len([f for f in os.listdir(folder_path) if f.startswith("Camera")])
        self.video_list = [None] * self.num_cam
        self.cap_list = [None] * self.num_cam
        folder_list = [None] * self.num_cam
        self.prediction_list = [None] * self.num_cam

        print(f"LOADER | Loading videos from: {folder_path}")
        
        # Determine total frames based on the longest video
        temp_total_frames = 0
        for i in range(self.num_cam):  # Loop through expected camera folders
            folder = os.path.join(folder_path, f"Camera{i+1}")
            video_file = os.path.join(folder, "0.mp4")
            self.video_list[i] = video_file
            
            cap = cv2.VideoCapture(video_file) # Try to open the video capture
            if not cap.isOpened():
                print(f"LOADER | Warning: Could not open video file: {video_file}")
                cap = None
            else:
                num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_frame > temp_total_frames:
                    temp_total_frames = num_frame
            self.cap_list[i] = cap # Add the capture object (or None) to the list
            folder_list[i] = folder
        
        self.total_frames = temp_total_frames # Set the global total_frames
        self.progress_widget.set_slider_range(self.total_frames)

        # Initialize the swap detection score array
        self.swap_detection_score_array = np.full((self.total_frames, 3), np.nan)
        self.swap_detection_score_array[:, 0] = np.arange(self.total_frames)

        # Initialize the temporal distance array
        self.temporal_dist_array_all = np.full((self.total_frames, self.dlc_data.instance_count), np.nan)

        if not any(self.cap_list): # Check if at least one video was loaded
            QMessageBox.warning(self, "Error", "No video files were loaded successfully.")
            traceback.print_exc()
            return

        # Initialize the pred_data_array to accept all data streams
        self.pred_data_array = np.full((self.total_frames, self.num_cam, self.dlc_data.instance_count, self.dlc_data.num_keypoint * 3), np.nan)

        for k in range(self.num_cam):
            if folder_list[k] is not None:
                folder = folder_list[k]
                h5_files = glob.glob(os.path.join(folder, "*.h5"))
                if not h5_files:
                    QMessageBox.warning(self, "Warning", f"No .h5 prediction file found in {folder}")
                    continue
                h5_files.sort()
                h5_file = h5_files[-1] # Take the newest one
                self.prediction_list[k] = h5_file
                self.load_prediction(cam_idx=k, prediction_filepath=h5_file)

        self.base_folder = folder_path # Store the base folder path for saving later
        self.current_frame_idx = 0
        self.progress_widget.set_slider_range(self.total_frames)
        self.nav_widget.set_collapsed(False)
        self.navigation_title_controller()
        self.display_current_frame() # Display the first frames

    def load_prediction(self, cam_idx:int, prediction_filepath:str):
        self.data_loader.prediction_filepath = prediction_filepath
        temp_dlc_data = dugh.load_and_show_message(self, self.data_loader, mute=True)

        if not temp_dlc_data:
            return

        if temp_dlc_data.pred_data_array.shape[0] > self.total_frames:
            QMessageBox.warning(self, "Warning", "Reloaded prediction has more frames than total frames. Truncating.")
            self.pred_data_array[:, cam_idx, :, :] = temp_dlc_data.pred_data_array[:self.total_frames, :, :]
        else:
            self.pred_data_array[:temp_dlc_data.pred_data_array.shape[0], cam_idx, :, :] = temp_dlc_data.pred_data_array
            # Pad with NaNs if the reloaded prediction is shorter
            if temp_dlc_data.pred_data_array.shape[0] < self.total_frames:
                    self.pred_data_array[temp_dlc_data.pred_data_array.shape[0]:, cam_idx, :, :] = np.nan

        if self.dlc_data.pred_frame_count is None:
            self.dlc_data.pred_frame_count = temp_dlc_data.pred_frame_count

    def parse_calibration_mat(self, calib):
        cam_pos = [None] * self.num_cam_from_calib
        cam_dir = [None] * self.num_cam_from_calib
        frame_count = [None] * self.num_cam_from_calib
        for i in range(self.num_cam_from_calib):
            self.camera_params[i]["RDistort"] = calib["params"][i,0][0,0]["RDistort"][0]
            self.camera_params[i]["TDistort"] = calib["params"][i,0][0,0]["TDistort"][0]
            K = calib["params"][i,0][0,0]["K"].T
            r = calib["params"][i,0][0,0]["r"].T
            t = calib["params"][i,0][0,0]["t"].flatten()
            cam_pos[i] = -np.dot(r.T, t)
            cam_dir[i] = r[2, :]
            self.camera_params[i]["K"] = K
            self.camera_params[i]["P"] = dutri.get_projection_matrix(K,r,t)
            frame_count[i] = len(calib["sync"][i,0][0,0]["data_sampleID"][0])
        self.cam_pos = np.array(cam_pos)
        self.cam_dir = np.array(cam_dir)

    ###################################################################################################################################################

    def call_track_refiner(self): 
        from dlc_track_refiner import DLC_Track_Refiner

        if not hasattr(self, "video_list") or not hasattr(self, "prediction_list") or self.dlc_data is None:
            QMessageBox.warning(self, "Warning", "Predictions or DLC config are not loaded, load predictions and config first!")
            return

        # The cam_idx is now directly passed from the clicked video label
        if self.selected_cam_idx is None:
            QMessageBox.information(self, "No Camera View Selected", "Please select a camera view first.")
            return
        
        selected_value = self.selected_cam_idx 
        self.refiner_window = DLC_Track_Refiner()
        self.refiner_window.video_file = self.video_list[selected_value]
        self.refiner_window.initialize_loaded_video()
        self.dlc_data.pred_data_array = self.pred_data_array[:,selected_value,:,:].copy()
        self.refiner_window.dlc_data = self.dlc_data
        self.refiner_window.initialize_loaded_data()
        self.refiner_window.current_frame_idx = self.current_frame_idx
        self.refiner_window.prediction = self.prediction_list[selected_value]
        self.refiner_window.display_current_frame()
        self.refiner_window.navigation_title_controller()
        self.refiner_window.show()
        self.refiner_window.prediction_saved.connect(self.reload_prediction)

    def reload_prediction(self, pred_file_path):
        """Reload prediction data from file and update visualization"""
        if pred_file_path == "reload_all":
            self.load_video_folder(self.base_folder)
            return
        try:
            # Find which camera this prediction belongs to
            cam_idx = None
            for i, pred in enumerate(self.prediction_list):
                if pred and os.path.dirname(pred) == os.path.dirname(pred_file_path):
                    cam_idx = i
                    break
            
            if cam_idx is not None:
                self.load_prediction(cam_idx, pred_file_path)

                self.prediction_list[cam_idx] = pred_file_path

                # Update visualization
                self.display_current_frame()
                QMessageBox.information(self, "Success", "Prediction reloaded successfully!")
                
                # Close the refiner window if it exists
                if hasattr(self, 'refiner_window') and self.refiner_window:
                    self.refiner_window.close()
                    self.refiner_window = None
            else:
                QMessageBox.warning(self, "Warning", "Could not match prediction to camera")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reload prediction: {str(e)}")
            traceback.print_exc() # Print full traceback for debugging

    ###################################################################################################################################################

    def display_current_frame(self):
        for i, cap in enumerate(self.cap_list):
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Check if pred_data_array is initialized and has data for the current frame and camera
                    if self.pred_data_array is not None and \
                       not np.all(np.isnan(self.pred_data_array[self.current_frame_idx, i, :, :])):
                        frame = self.plot_2d_points(frame, i) 

                    target_width = self.video_labels[i].width() # Get the target size from the QLabel
                    target_height = self.video_labels[i].height()

                    # Resize the frame to the target size
                    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

                    rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qt_image)
                    self.video_labels[i].setPixmap(pixmap)
                    self.video_labels[i].setText("")
                else:
                    self.video_labels[i].setText(f"End of Video {i+1} / Error")
                    self.video_labels[i].setPixmap(QtGui.QPixmap()) # Clear any previous image
            else:
                self.video_labels[i].setText(f"Video {i+1} Not Loaded/Available")
                self.video_labels[i].setPixmap(QtGui.QPixmap())
            
            self.progress_widget.set_current_frame(self.current_frame_idx) # Update slider handle's position

        self.plot_3d_points()
        self._refresh_selected_cam()

    def plot_2d_points(self, frame, cam_idx):
        if self.dlc_data is None:
            return frame # Cannot plot if DLC data is not loaded

        for inst in range(self.dlc_data.instance_count):
            color_rgb = self.instance_color[inst % len(self.instance_color)]
            # Convert RGB to BGR for OpenCV
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            keypoint_coords = dict()
            for kp_idx in range(self.dlc_data.num_keypoint):
                kp = self.pred_data_array[self.current_frame_idx,cam_idx,inst,kp_idx*3:kp_idx*3+3]
                if pd.isna(kp[0]) or kp[2] < self.confidence_cutoff:
                    continue
                x, y = kp[0], kp[1]
                keypoint_coords[kp_idx] = (int(x),int(y))
                cv2.circle(frame, (int(x), int(y)), 3, color_bgr, -1) # Draw the dot representing the keypoints

            if self.dlc_data.individuals is not None and len(keypoint_coords) >= 2: # Only plot bounding box with more than one points
                self.plot_bounding_box(keypoint_coords, frame, color_bgr, inst)
            if self.dlc_data.skeleton:
                self.plot_2d_skeleton(keypoint_coords, frame, color_bgr)
        return frame

    def plot_bounding_box(self, keypoint_coords, frame, color, inst):
        # Calculate bounding box coordinates
        x_coords = [int(keypoint_coords[p][0]) for p in keypoint_coords if keypoint_coords[p] is not None]
        y_coords = [int(keypoint_coords[p][1]) for p in keypoint_coords if keypoint_coords[p] is not None]

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
        if self.dlc_data.individuals: # Check if individuals list exists
            cv2.putText(frame, f"Instance: {self.dlc_data.individuals[inst]}", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        return frame
    
    def plot_2d_skeleton(self, keypoint_coords, frame, color):
        if self.dlc_data is None:
            return frame # Cannot plot if DLC data is not loaded

        for start_kp, end_kp in self.dlc_data.skeleton:
            start_kp_idx = self.keypoint_to_idx.get(start_kp)
            end_kp_idx = self.keypoint_to_idx.get(end_kp)
            
            if start_kp_idx is not None and end_kp_idx is not None:
                start_coord = keypoint_coords.get(start_kp_idx)
                end_coord = keypoint_coords.get(end_kp_idx)
                if start_coord and end_coord:
                    cv2.line(frame, start_coord, end_coord, color, 2)
        return frame
    
    def plot_3d_points(self):
        if self.dlc_data is None:
            self.ax.clear()
            self.ax.set_title("3D Skeleton Plot - No DLC data loaded")
            self.canvas.draw_idle()
            return

        if self.dlc_data is None or self.pred_data_array is None:
            point_3d_array = np.full((1, 1, 3), np.nan)
        else:
            data_processor_3d = Data_Processor_3D(
                self.dlc_data, self.camera_params, self.pred_data_array, self.confidence_cutoff, self.num_cam)
            point_3d_array = data_processor_3d.get_3d_pose_array(self.current_frame_idx, return_confidence=False)

        if point_3d_array is None:
            return

        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"3D Skeleton Plot - Frame {self.current_frame_idx}")
        self.ax.set_xlim([-self.plot_lim, self.plot_lim])
        self.ax.set_ylim([-self.plot_lim, self.plot_lim])
        self.ax.set_zlim([-self.plot_lim // 5, self.plot_lim])

        for inst in range(self.dlc_data.instance_count):
            color = self.instance_color[inst % len(self.instance_color)]
            # Plot 3D keypoints
            for kp_idx in range(self.dlc_data.num_keypoint):
                point_3d = point_3d_array[inst, kp_idx, :]
                if not pd.isna(point_3d[0]): # Check if the point is not NaN
                    self.ax.scatter(point_3d[0], point_3d[1], point_3d[2], color=np.array(color)/255, s=50)

            # Plot 3D skeleton
            if self.dlc_data.skeleton:
                for start_kp, end_kp in self.dlc_data.skeleton:
                    start_kp_idx = self.keypoint_to_idx.get(start_kp)
                    end_kp_idx = self.keypoint_to_idx.get(end_kp)
                    
                    if start_kp_idx is not None and end_kp_idx is not None:
                        start_point = point_3d_array[inst, start_kp_idx, :]
                        end_point = point_3d_array[inst, end_kp_idx, :]
                        if not pd.isna(start_point[0]) and not pd.isna(end_point[0]): # Check if both points are not NaN
                            self.ax.plot([start_point[0], end_point[0]],
                                         [start_point[1], end_point[1]],
                                         [start_point[2], end_point[2]],
                                         color=np.array(color)/255)
        self.canvas.draw_idle() # Redraw the 3D canvas

    def plot_camera_geometry(self):
        """Plots the relative geometry on a given Axes3D object."""
        # Ensure the plot is cleared before drawing camera geometry
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"3D Camera Geometry")
        self.ax.set_xlim([-self.plot_lim * 3, self.plot_lim * 3])
        self.ax.set_ylim([-self.plot_lim * 3, self.plot_lim * 3])
        self.ax.set_zlim([-self.plot_lim * 3 // 5, self.plot_lim * 3])

        for i in range(self.num_cam_from_calib):
            self.ax.scatter(*self.cam_pos[i], s=100, label=f"Camera {i+1} Pos")
            self.ax.quiver(*self.cam_pos[i], *self.cam_dir[i], length=100, color='blue', normalize=True)
        # Plot ground plane
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        self.canvas.draw_idle() 

    ###################################################################################################################################################

    def calculate_identity_swap_score(self, mode, parent_progress=None, mute=False):
        if not self.dlc_data:
            return False

        if self.dlc_data.instance_count == 1:
            QMessageBox.information(self, "Single Instance", "Only one instance detected, no swap detection needed.")
            return False

        try:
            config = duh.get_config_from_calculation_mdode(mode, self.current_frame_idx, self.check_range, self.total_frames)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Mode", f"{e}")
            return False

        if config.show_progress:
            dialog = "Calculating swap detection score..."
            title = f"Calculating Identity Score In {mode}"
            progress = dugh.get_progress_dialog(self, config.start_frame, self.total_frames, title, dialog, parent_progress)
        else:
            progress = None

        calculated_frame_count = 0

        for frame_idx in range(config.start_frame, self.total_frames):
            if calculated_frame_count >= config.frame_count_max:
                break

            if progress:
                progress.setValue(frame_idx)
                if progress.wasCanceled():
                    return False

            QtWidgets.QApplication.processEvents()  # Keep UI responsive
        
            data_processor_3d = Data_Processor_3D(self.dlc_data, self.camera_params, self.pred_data_array, self.confidence_cutoff, self.num_cam)
            keypoint_data_tr, valid_view = data_processor_3d.get_keypoint_data_for_frame(
                frame_idx, instance_threshold=self.dlc_data.instance_count, view_threshold=3)

            if not keypoint_data_tr:
                if mode == "full": # Only applying during the initial sweep
                    self.skipped_frame_list.append(frame_idx)
                continue

            swap_score = dutri.calculate_identity_swap_score_per_frame(
                keypoint_data_tr, valid_view, self.dlc_data.instance_count, self.dlc_data.num_keypoint, self.num_cam)
            
            self.swap_detection_score_array[frame_idx, 1] = swap_score
            calculated_frame_count += 1

            if config.until_next_error and swap_score > self.deviance_threshold and calculated_frame_count > config.frame_count_min:
                break

        if progress:
            progress.close()
        if not mute and mode == "full":
            QMessageBox.information(self, "Swap Detection Score Calculated", "Identity swap detection completed.")

        self._populate_roi_frame_list()
        self.navigation_title_controller()
        return True
    
    def calculate_temporal_vel(self, frame_idx_r=None, check_window=5):
        if not self.dlc_data:
            return False

        progress = None
        if frame_idx_r is None:
            dialog = "Calculating temporal velocity for all the frames..."
            title = f"Calculating Temporal Velocity"
            progress = dugh.get_progress_dialog(self, 0, self.total_frames, title, dialog)

        data_processor_3d = Data_Processor_3D(self.dlc_data, self.camera_params, self.pred_data_array, self.confidence_cutoff, self.num_cam)

        if progress:
            for frame_idx in range(self.total_frames):
                progress.setValue(frame_idx)
                self.temporal_dist_array_all[frame_idx, :] = data_processor_3d.calculate_temporal_velocity(frame_idx)
            progress.close()
        else:
            end_frame = frame_idx_r + check_window + 1
            if end_frame > self.total_frames:
                end_frame = self.total_frames
            for frame_idx in range(frame_idx_r, end_frame):
                self.temporal_dist_array_all[frame_idx, :] = data_processor_3d.calculate_temporal_velocity(frame_idx, check_window)

        self._populate_roi_frame_list()
        self.navigation_title_controller()

    def _populate_roi_frame_list(self):
        deviance_mask = self.swap_detection_score_array[:, 1] >= self.deviance_threshold
        temporal_mask = self.temporal_dist_array_all[:, 1] >= self.velocity_threshold
        combined_mask = deviance_mask | temporal_mask
        self.roi_frame_list = np.where(combined_mask)[0].tolist() # Get the indices of frames with significant deviations
        self._refresh_slider()

    def automatic_track_correction(self):
        if not self.dlc_data:
            QMessageBox.information(self, "No Data", "Load prediction data first!") 
            return

        if np.isnan(self.swap_detection_score_array[:, 1]).all():
            if not self.calculate_identity_swap_score(mode="full", mute=True):
                return

        if self.dlc_data.instance_count != 2:
            QMessageBox.information(self, "Unimplemented", "The function is only for two instance only.")
            return

        self.correction_progress = dugh.get_progress_dialog(self, 0, self.total_frames,
            title="Automatic Correction Progress", dialog="Commencing automatic track correction...")

        main_window_center = self.geometry().center()
        x = main_window_center.x() - self.correction_progress.width() // 2
        y = main_window_center.y() - self.correction_progress.height() // 2
        self.correction_progress.move(x, y)

        self.failed_frame_list = [] # Reset the failed frame list
        for frame_idx in range(self.total_frames):
            if frame_idx in self.skipped_frame_list:
                continue
            
            if not self.attempt_track_correction(frame_idx):
                QMessageBox.warning(self, "Correction Cancelled", "Track correction was cancelled by the user.")
                return

        self.correction_progress.close()

        self.calculate_temporal_vel()
        self.refresh_failed_frame_list()

        if self.failed_frame_list:
            QMessageBox.warning(self, "Correction Partially Successful", f"Failed to correct frames: {', '.join(map(str, self.failed_frame_list))}.")
        else:
            QMessageBox.information(self, "Correction Successful", "All marked frames corrected successfully.") 

        self.is_saved = False

    def attempt_track_correction(self, frame_idx):
        if frame_idx not in self.roi_frame_list:
            return True
        
        if self.correction_progress.wasCanceled():
            return False
        
        self.correction_progress.setValue(frame_idx)
        self.current_frame_idx = frame_idx
        self.display_current_frame()
        
        backup_pred_data_array = self.pred_data_array.copy() # Backup current prediction data
        
        for swap_num in range(1, 5): # Try 1, 2, 3 ,4 -way swap
            if self.attempt_multiple_way_swap_solution(frame_idx, swap_num=swap_num):
                self.calculate_identity_swap_score(mode="remap")
                self._refresh_slider()
                return True

        self.pred_data_array = backup_pred_data_array # Restore backup for a second round

        for swap_num in range(1, 5): # Second chance, with easy_mode requirements
            if self.attempt_multiple_way_swap_solution(frame_idx, swap_num=swap_num, easy_mode=True):
                self.calculate_identity_swap_score(mode="remap", parent_progress=self.correction_progress)
                self._refresh_slider()
                return True
            
        self.pred_data_array = backup_pred_data_array # Restore backup if FUBAR
        self.failed_frame_list.append(frame_idx)
        return True

    def attempt_multiple_way_swap_solution(self, frame_idx:int, swap_num:int, easy_mode:bool=False):
        print(f"AUTOCORRECT | Attempting to find a {swap_num}-way swap solution for frame {frame_idx}...")
        all_camera_pairs = list(combinations(range(self.num_cam), swap_num))
        backup_pred_data_array = self.pred_data_array.copy()

        for cam_combo in all_camera_pairs:
            self.pred_data_array = backup_pred_data_array.copy()

            for cam_idx in cam_combo:
                self.pred_data_array = dute.track_swap_3D_plotter(self.pred_data_array, frame_idx, cam_idx)

            self.selected_cam_idx = cam_combo[0] # Set the first camera as selected for display
            self._post_correction_operations()

            swap_score_calculation_result = self._validate_swap_score_post_correction(frame_idx)
            temporal_calculation_result = self._validate_velocity_post_correction(frame_idx)

            if easy_mode:
                condition = self._validate_swap_score_post_correction(frame_idx, easy_mode=True)
                appendix = "(SECOND RUN)"
            else:
                condition = swap_score_calculation_result and temporal_calculation_result
                appendix = ""

            if condition:
                print(f"AUTOCORRECT | Successfully applied {swap_num}-way swap for cameras {cam_combo} on frame {frame_idx}. {appendix}")
                return True

        print(f"AUTOCORRECT | {swap_num} swap solution failed for frame {frame_idx}. {appendix}")
        return False

    def manual_swap_frame_view(self):
        if not self.dlc_data:
            return
        
        if self.selected_cam_idx is None:
            QMessageBox.information(self, "No Camera View Selected", "Please select a camera view first.")
            return
        
        if self.dlc_data.instance_count != 2:
            QMessageBox.information(self, "Unimplemented", "The function is only for two instance only.")
            return
        
        self.pred_data_array = dute.track_swap_3D_plotter(self.pred_data_array, self.current_frame_idx, self.selected_cam_idx)
        self.display_current_frame() # Refresh the display to show the swapped frame
        self.calculate_identity_swap_score(mode="manual_check")
        self.calculate_temporal_vel(frame_idx_r=self.current_frame_idx)

        self.refresh_failed_frame_list()
        self.navigation_title_controller()
        self._refresh_slider()
        self.is_saved = False

    def _validate_swap_score_post_correction(self, frame_idx, easy_mode=False):
        self.calculate_identity_swap_score(mode="auto_check", parent_progress=self.correction_progress)
        
        score = self.swap_detection_score_array[frame_idx, 1]

        if easy_mode:
            if score >= self.deviance_threshold * 2:
                return False
            return True  # Allow even if above normal threshold
        else:
            if score >= self.deviance_threshold:
                return False
            return True
        
    def _validate_velocity_post_correction(self, frame_idx):
        self.calculate_temporal_vel(frame_idx_r=frame_idx)
        result_list = []
        for inst_idx in range(self.dlc_data.instance_count):
            result = self.temporal_dist_array_all[frame_idx, inst_idx] > self.velocity_threshold
            result_list.append(result)

        if all(result_list):
            return False
        
        return True

    def reset_marked_frames(self):
        """Reset all marked frames and clear the lists."""
        self.roi_frame_list = []
        self.failed_frame_list = []
        self.skipped_frame_list = []
        self.swap_detection_score_array = np.full((self.total_frames, 2), np.nan)
        self.navigation_title_controller()
        self._refresh_slider()

    def refresh_failed_frame_list(self):
        """Refresh the failed frame list to only include frames that are also in the ROI frame list."""
        self.dialog_deviance = None
        self.failed_frame_list = [frame for frame in self.failed_frame_list if frame in self.roi_frame_list]
        self.navigation_title_controller()
        self._refresh_slider()

    def change_mark_view_mode(self):
        self.current_view_mode_idx = (self.current_view_mode_idx + 1) % len(self.view_mode_choice)
        self.navigation_title_controller()

    def _navigate_marked_frames(self, direction):
        """Navigate through marked frames based on the current view mode."""
        if self.current_view_mode_idx == 0:  # ROI frames
            if not self.roi_frame_list:
                QMessageBox.information(self, "No ROI Frames", "No ROI frames available to navigate.")
                return
            dugh.navigate_to_marked_frame(self, self.roi_frame_list, self.current_frame_idx, self._handle_frame_change_from_comp, direction)
        elif self.current_view_mode_idx == 1:  # Failed frames
            if not self.failed_frame_list:
                QMessageBox.information(self, "No Failed Frames", "No failed frames available to navigate.")
                return
            dugh.navigate_to_marked_frame(self, self.failed_frame_list, self.current_frame_idx, self._handle_frame_change_from_comp, direction)
        elif self.current_view_mode_idx == 2:  # Multi-swap frames
            if not self.skipped_frame_list:
                QMessageBox.information(self, "No Skipped Frames", "No frames of multiple instance swap available to navigate.")
                return
            dugh.navigate_to_marked_frame(self, self.skipped_frame_list, self.current_frame_idx, self._handle_frame_change_from_comp, direction)

    def _post_correction_operations(self):
        self.display_current_frame()
        self._adjust_3D_plot_view_angle()
        self.calculate_identity_swap_score(mode="auto_check")
        self.navigation_title_controller()

    ###################################################################################################################################################

    def set_selected_camera(self, cam_idx):
        if hasattr(self, 'cap_list'):
            self.selected_cam_idx = cam_idx
            self._refresh_selected_cam()
            if self.auto_perspective:
                self._adjust_3D_plot_view_angle()

    def change_frame(self, delta):
        self.selected_cam_idx = None # Clear the selected cam upon frame switch
        new_frame_idx = self.current_frame_idx + delta
        if 0 <= new_frame_idx < self.total_frames:
            self.current_frame_idx = new_frame_idx
            self.display_current_frame()
            self.navigation_title_controller()

        self.canvas.draw_idle()

    def navigation_title_controller(self):
        title_text = f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1} | View Mode: {self.view_mode_choice[self.current_view_mode_idx]}"\
        
        if self.swap_detection_score_array is not None and self.swap_detection_score_array.shape[0] > 0:
            deviance_scores = self.swap_detection_score_array[:, 1]
            title_text += f" | Deviance Score: {deviance_scores[self.current_frame_idx]:.2f}"
        
        if self.temporal_dist_array_all is not None:
            for inst_idx in range(self.dlc_data.instance_count):
                inst_vel = self.temporal_dist_array_all[self.current_frame_idx, inst_idx]
                title_text += f" | Instance {inst_idx} Velocity: {inst_vel}"

        self.nav_widget.setTitle(title_text)
        if self.current_frame_idx in self.failed_frame_list:
            self.nav_widget.setTitleColor("#FF0000")  # Red
        elif self.current_frame_idx in self.skipped_frame_list:
            self.nav_widget.setTitleColor("#3D3D3D")  # Dark Gray
        elif self.current_frame_idx in self.roi_frame_list:
            self.nav_widget.setTitleColor("#F79F1C")  # Amber/Orange
        else:
            self.nav_widget.setTitleColor("black")

    ###################################################################################################################################################

    def toggle_auto_3d_perspective(self):
        self.auto_perspective = not self.auto_perspective
        self.ax.view_init(elev=30, azim=-60)
        self.canvas.draw_idle()

    def show_confidence_dialog(self):
        if self.pred_data_array is None:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        dialog = Adjust_Property_Dialog(
            property_name="Confidence Cutoff", property_val=self.confidence_cutoff, range=(0.00, 1.00), parent=self)
        dialog.property_changed.connect(self._update_confidence_cutoff)
        dialog.show() # .show() instead of .exec() for a non-modal dialog

    def show_deviance_dialog(self):
        if self.pred_data_array is None:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        dialog = Adjust_Property_Dialog(
            property_name="Deviance Threshold", property_val=self.deviance_threshold, range=(0, 300), parent=self)
        dialog.property_changed.connect(self._update_deviance_threshold)
        dialog.finished.connect(self.refresh_failed_frame_list)
        dialog.show()
        self.dialog_deviance = dialog

    def show_velocity_dialog(self):
        if self.pred_data_array is None:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        dialog = Adjust_Property_Dialog(
            property_name="Velocity Threshold", property_val=self.velocity_threshold, range=(0, 100), parent=self)
        dialog.property_changed.connect(self._update_velocity_threshold)
        dialog.show() # .show() instead of .exec() for a non-modal dialog

    def _update_confidence_cutoff(self, new_cutoff):
        self.confidence_cutoff = new_cutoff
        self.display_current_frame() # Redraw with the new cutoff

    def _update_deviance_threshold(self, new_threshold):
        self.deviance_threshold = new_threshold
        self._populate_roi_frame_list()
        self.navigation_title_controller()

    def _update_velocity_threshold(self, new_threshold):
        self.velocity_threshold = new_threshold
        self._populate_roi_frame_list()
        self.navigation_title_controller()

    def _refresh_slider(self):
        self.progress_widget.set_frame_category("ROI frames", self.roi_frame_list, "#F79F1C") # Update ROI frames
        self.progress_widget.set_frame_category("Failed frames", self.failed_frame_list, "#FF0000", priority=7)
        self.progress_widget.set_frame_category("Skippedframes", self.skipped_frame_list, "#3D3D3D", priority=6)

    def _refresh_selected_cam(self):
        for i in range(4): # Will change to a flexible range later
            if i == self.selected_cam_idx:
                self.video_labels[i].setStyleSheet("border: 2px solid red;")
            else:
                self.video_labels[i].setStyleSheet("border: 1px solid gray;")

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.current_frame_idx = new_frame_idx
        self.display_current_frame()
        self.navigation_title_controller()

    def _adjust_3D_plot_view_angle(self):
        if self.selected_cam_idx is None:
            return
        cam_pos = self.cam_pos[self.selected_cam_idx]
        elev, azim = duh.acquire_view_perspective_for_cur_cam(cam_pos)
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def on_scroll_3d_plot(self, event):
        """Handle matplotlib scroll events for zooming the 3D plot."""
        zoom_factor = 1.2
        if event.button == 'up':  # Zoom in (or typically 1 for wheel up)
            self.plot_lim = max(50, self.plot_lim / zoom_factor)
        elif event.button == 'down':  # Zoom out (or typically -1 for wheel down)
            self.plot_lim = min(500, self.plot_lim * zoom_factor)
        
        self.plot_3d_points()
        self.canvas.draw_idle() # Ensure the canvas redraws

    ###################################################################################################################################################

    def get_saveable_state(self):
        return {
            'base_folder': self.base_folder,
            'calibration_filepath': self.calibration_filepath,
            'dlc_config_filepath': self.dlc_config_filepath,
            'pred_data_array': self.pred_data_array,
            'current_frame_idx': self.current_frame_idx,
            'confidence_cutoff': self.confidence_cutoff,
            'deviance_threshold': self.deviance_threshold,
            'velocity_threshold': self.velocity_threshold,
            'roi_frame_list': self.roi_frame_list,
            'failed_frame_list': self.failed_frame_list,
            'skipped_frame_list': self.skipped_frame_list,
            'swap_detection_score_array': self.swap_detection_score_array,
            'temporal_dist_array_all': self.temporal_dist_array_all,
        }

    def save_workspace(self):
        if not hasattr(self, 'pred_data_array') or self.pred_data_array is None:
            QMessageBox.information(self, "No Data", "Load prediction data first!")
            return

        save_path = os.path.join(self.base_folder, "workspace_save.pickle")

        try:
            state = self.get_saveable_state() 
            with open(save_path, "wb") as f:
                pickle.dump(state, f)
            QMessageBox.information(self, "Save Successful", f"Workspace saved to {save_path}")
            self.is_saved = True
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"An error occurred while saving: {e}")
            
    def save_swapped_prediction(self):
        if not self.dlc_data:
            QMessageBox.information(self, "No Data", "Load prediction data first!")
            return
        
        error_views = []
        for prediction in self.prediction_list:
            if prediction is None:
                continue # Skip if no prediction loaded for this camera

            prediction_idx = self.prediction_list.index(prediction)
            pred_file_to_save_path = dio.determine_save_path(prediction, suffix="_3D_plotter_")

            pred_data_array = self.pred_data_array[:, prediction_idx, :, :].copy() # Extract the prediction data for this camera
            status, msg = dio.save_prediction_to_h5(pred_file_to_save_path, pred_data_array)
            if not status:
                error_views.append((prediction_idx, msg))

        if error_views:
            error_msg = [f"Camera {error_view[0]}: {error_view[1]}" for error_view in error_views]
            error_msg_for_msgbox = "\n".join(error_msg)
            self.save_workspace() # Save the workspace to ensure no progress is lost
            QMessageBox.critical(self, "Save Failed", f"Failed to save prediction for following cameras and errors: "
                f"{error_msg_for_msgbox}"
                f"Workspace saved to {self.base_folder} for resuming later.")
        else:
            QMessageBox.information(self, "Save Successful", msg)
            self.is_saved = True
            self.reload_prediction(pred_file_path="reload_all")

    def export_COM_to_SDANNCE(self):
        # Create a dialog for the user to select the COM bodypart
        item, ok = QtWidgets.QInputDialog.getItem(
            self, "Select COM Bodypart", "Choose a bodypart to use as Center of Mass (COM):",
            self.dlc_data.keypoints, 0, False
        )

        if not ok or not item:
            QMessageBox.warning(self, "No Selection", "COM export cancelled by user.")
            return

        com_idx = self.keypoint_to_idx[item]
        dialog = "Gathering 3D COM (Center of Mass) data for export..."
        title = f"Gather 3D COM Data For Export"
        progress = dugh.get_progress_dialog(self, 0, self.total_frames, title, dialog)

        data_processor_3d = Data_Processor_3D(self.dlc_data, self.camera_params, self.pred_data_array, self.confidence_cutoff, self.num_cam)
        com_for_export = np.full((self.total_frames, 3, self.dlc_data.instance_count), np.nan)
        for frame_idx in range(self.total_frames):
            point_3d_array_current_frame = data_processor_3d.get_3d_pose_array(frame_idx, return_confidence=False)
            com_for_export[frame_idx, :, :] = point_3d_array_current_frame[:, com_idx, :].T
            progress.setValue(frame_idx)
        
        progress.close()

        print("Interpolating missing 3D COM data...")
        for inst_idx in range(self.dlc_data.instance_count):
            com_data = com_for_export[:, :, inst_idx]  # Shape: (T, 3)
            df = pd.DataFrame(com_data, columns=['x', 'y', 'z'])
            # Linear interpolation across time (axis=0)
            df_interpol = df.interpolate(method='linear', axis=0, limit_direction='both')
            df_filled = df_interpol.ffill().bfill()
            com_for_export[:, :, inst_idx] = df_filled.values

        for inst_idx in range(self.dlc_data.instance_count):
            mat_data = {
                "com": com_for_export[:, :, inst_idx],
                "sampleID": range(self.total_frames)
            }
            mat_filename = f"instance{inst_idx}com3d.mat"
            mat_save_path = os.path.join(self.base_folder, mat_filename)
            sio.savemat(mat_save_path, mat_data)
            print(f"Saved: {mat_filename} for instance{inst_idx}")

        mat_data = {
            "com": com_for_export[:, :, :],
            "sampleID": range(self.total_frames)
        }
        mat_filename = f"com3d.mat"
        mat_save_path = os.path.join(self.base_folder, mat_filename)
        sio.savemat(mat_save_path, mat_data)
        print(f"Saved combined mat in {mat_filename} ")

        QMessageBox.information(self, "Export Complete", 
            f"3D COM data '{item}' exported successfully for {self.dlc_data.instance_count} instances.")

    def closeEvent(self, event: QCloseEvent):
        # Ensure all VideoCapture objects are released when the window closes
        if hasattr(self, 'cap_list'):
            for cap in self.cap_list:
                if cap and cap.isOpened():
                    cap.release()
        dugh.handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_workspace)

###################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = DLC_3D_plotter()
    main_window.show()
    app.exec()
