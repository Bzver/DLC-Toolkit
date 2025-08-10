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
from utils.dtu_widget import Menu_Widget, Progress_Widget, Nav_Widget
from utils.dtu_comp import Clickable_Video_Label, Adjust_Property_Dialog
from utils.dtu_dataclass import Session_3D_Plotter
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
                    ("Adjust Confidence Cutoff", self.show_confidence_dialog),
                    ("Adjust Deviance Threshold", self.show_deviance_dialog),
                    ("Track Swap Score Calculation", lambda:self.calculate_identity_swap_score(mode="full")),
                    ("Refresh Failed Frames Ater Deviance Adjustment", self.refresh_failed_frame_list)
                ]
            },
            "View": {
                "display_name": "View",
                "buttons": [
                    ("Change Marked Frame View Mode", self.change_mark_view_mode),
                    ("Reset Marked Frames", self.reset_marked_frames),
                    ("Check Camera Geometry", self.plot_camera_geometry),
                    ("View 3D Plot in Selected Camera's Perspective", self.adjust_3D_plot_view_angle)
                ]
            },
            "Track": {
                "display_name": "Track Refine",
                "buttons": [
                    ("Auto Track Swap Correct", self.automatic_track_correction),
                    ("Manual Swap Selected View (X)", self.manual_swap_frame_view),
                    ("Call Track Refiner", self.call_track_refiner)
                ]
            },
            "Save": {
                "display_name": "Save",
                "buttons": [
                    ("Save Workspace", self.save_workspace),
                    ("Save Swapped Track", self.save_swapped_prediction)
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
                label.setText(f"Video {cam_idx + 1}")
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

        self.progress_widget = Progress_Widget()
        self.layout.addWidget(self.progress_widget)
        self.progress_widget.frame_changed.connect(self._handle_frame_change_from_comp)

        # Navigation controls
        self.nav_widget = Nav_Widget()
        self.layout.addWidget(self.nav_widget)
        self.nav_widget.hide()

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

        self.confidence_cutoff = 0.6 # Initialize confidence cutoff
        self.deviance_threshold = 100 # Initialize deviancy threshold

        self.data_loader = DLC_Loader(None, None)
        self.dlc_data = None
        self.keypoint_to_idx = {}

        self.base_folder, self.calibration_filepath, self.dlc_config_filepath = None, None, None
        self.video_list, self.cap_list, self.prediction_list, self.camera_params = [], [], [], []
        self.cam_pos, self.cam_dir = None, None

        self.pred_data_array = None # Combined prediction data for all cameras

        self.num_cam_from_calib = None

        self.plot_lim = 300
        self.instance_color = [
            (255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)] # RGB
        
        self.current_frame_idx = 0
        self.total_frames = 0
        self.selected_cam_idx = None

        self.refiner_window = None

        self.view_mode_choice = ["ROI Frames", "Failed Frames", "Multi-swap Frames"]
        self.current_view_mode_idx = 0

        self.is_saved = True

        self.roi_frame_list, self.failed_frame_list, self.sus_frame_list = [], [], []
        self._refresh_slider()

        self.ax.clear()
        self.ax.set_title("3D Skeleton Plot - No DLC data loaded")
        self.canvas.draw_idle()

    def debug_load(self):
        self.dlc_config_loader(DLC_CONFIG_DEBUG)
        self.calibration_loader(CALIB_FILE_DEBUG)
        self.load_video_folder(VIDEO_FOLDER_DEBUG)

    def load_workspace(self):
        """Load all the saved variables from a previously saved HDF5 file."""
        self.reset_state()
        workspace_filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Workspace", "", "Workspace Files (*.pickle)")
        if not workspace_filepath:
            return
        
        print(f"Workspace loaded: {workspace_filepath}")
            
        try:
            with open(workspace_filepath, "rb") as f:
                session_data = pickle.load(f)

            self.calibration_loader(session_data.base_folder)
            self.dlc_config_loader(session_data.dlc_config_filepath)
            self.load_video_folder(session_data.calibration_filepath)

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"An error occurred while loading: {e}")
            traceback.print_exc()

            self.pred_data_array = session_data.pred_data_array
            self.current_frame_idx = session_data.current_frame_idx
            self.confidence_cutoff = session_data.confidence_cutoff
            self.deviance_threshold = session_data.deviance_threshold
            self.roi_frame_list = session_data.roi_frame_list
            self.failed_frame_list = session_data.failed_frame_list
            self.sus_frame_list = session_data.sus_frame_list
            self.swap_detection_score_array = session_data.swap_detection_score_array

        self.display_current_frame()
        self._refresh_slider()
        self.navigation_title_controller()

        QMessageBox.information(self, "Load Successful", f"Workspace loaded from {workspace_filepath}")

    def open_video_folder_dialog(self):
        if self.dlc_data is None: # Check if dlc_data is loaded
            QMessageBox.warning(self, "Warning", "DLC config is not loaded, load DLC config first!")
            print("DLC config is not loaded, load DLC config first!")
            self.load_dlc_config()
            if self.dlc_data is None: # User closed DLC loading window or failed to load
                return
        if self.num_cam_from_calib is None:
            QMessageBox.warning(self, "Warning", "Calibrations are not loaded, load calibrations first!")
            print("Calibrations are not loaded, load calibrations first!")
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
            print(f"Calibration loaded: {calib_file}")
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

        print(f"Loading videos from: {folder_path}")
        
        # Determine total frames based on the longest video
        temp_total_frames = 0
        for i in range(self.num_cam):  # Loop through expected camera folders
            folder = os.path.join(folder_path, f"Camera{i+1}")
            video_file = os.path.join(folder, "0.mp4")
            self.video_list[i] = video_file
            
            cap = cv2.VideoCapture(video_file) # Try to open the video capture
            if not cap.isOpened():
                print(f"Warning: Could not open video file: {video_file}")
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
        self.nav_widget.show()
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
            return # Cannot plot if DLC data is not loaded

        point_3d_array = self.data_loader_for_3d_plot()
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
        self.ax.set_xlim([-self.plot_lim, self.plot_lim])
        self.ax.set_ylim([-self.plot_lim, self.plot_lim])
        self.ax.set_zlim([-self.plot_lim // 5, self.plot_lim])

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

    def data_loader_for_3d_plot(self, undistorted_images=False):
        if self.dlc_data is None or self.pred_data_array is None:
            return np.full((1, 1, 3), np.nan) # Return empty array if no data

        point_3d_array = np.full((self.dlc_data.instance_count, self.dlc_data.num_keypoint, 3), np.nan)
        instances_detected_per_camera = self._validate_multiview_instances()
        keypoint_data_for_triangulation = self._acquire_keypoint_data(instances_detected_per_camera, undistorted_images)

        for inst in range(self.dlc_data.instance_count):
            # iterate through each keypoint to perform triangulation
            for kp_idx in range(self.dlc_data.num_keypoint):
                # Convert dictionaries to lists while maintaining camera order
                projs_dict = keypoint_data_for_triangulation[inst][kp_idx]['projs']
                pts_2d_dict = keypoint_data_for_triangulation[inst][kp_idx]['2d_pts']
                confs_dict = keypoint_data_for_triangulation[inst][kp_idx]['confs']
                
                # Get sorted camera indices to maintain order
                cam_indices = sorted(projs_dict.keys())
                projs = [projs_dict[i] for i in cam_indices]
                pts_2d = [pts_2d_dict[i] for i in cam_indices]
                confs = [confs_dict[i] for i in cam_indices]
                num_valid_views = len(projs)

                if num_valid_views >= 2:
                    point_3d_array[inst, kp_idx, :] = dutri.triangulate_point(num_valid_views, projs, pts_2d, confs)

        return point_3d_array

    def _validate_multiview_instances(self):
        if self.dlc_data.instance_count < 2:
            return [1] * self.num_cam # All cameras valid for the single instance
        
        instances_detected_per_camera = [0] * self.num_cam
        for cam_idx_check in range(self.num_cam):
            detected_instances_in_cam = 0

            for inst_check in range(self.dlc_data.instance_count):
                has_valid_data = False
                for kp_idx_check in range(self.dlc_data.num_keypoint):

                    if not pd.isna(self.pred_data_array[self.current_frame_idx,cam_idx_check, inst_check, kp_idx_check*3]):
                        has_valid_data = True
                        break

                if has_valid_data:
                    detected_instances_in_cam += 1
            instances_detected_per_camera[cam_idx_check] = detected_instances_in_cam

        return instances_detected_per_camera

    def _acquire_keypoint_data(self, instances_detected_per_camera, undistorted_images=False, frame_idx=None):
        # Dictionary to store per-keypoint data across cameras:
        keypoint_data_for_triangulation = {
            inst:
            {
                kp_idx: {'projs': {}, '2d_pts': {}, 'confs': {}}
                for kp_idx in range(self.dlc_data.num_keypoint)
            }
            for inst in range(self.dlc_data.instance_count)
        }

        for inst_idx in range(self.dlc_data.instance_count):
            for cam_idx in range(self.num_cam):
                if self.dlc_data.instance_count > 1 and instances_detected_per_camera[cam_idx] < 2:
                    continue # Skip if this camera has not detect enough instances

                # Ensure camera_params are available for the current camera index
                if cam_idx >= len(self.camera_params) or not self.camera_params[cam_idx]:
                    print(f"Warning: Camera parameters not available for camera {cam_idx}. Skipping.")
                    continue

                RDistort = self.camera_params[cam_idx]['RDistort']
                TDistort = self.camera_params[cam_idx]['TDistort']
                K = self.camera_params[cam_idx]['K']
                P = self.camera_params[cam_idx]['P']

                # Get all keypoint data (flattened) for the current frame, camera, and instance
                frame_to_use = frame_idx if frame_idx is not None else self.current_frame_idx
                keypoint_data_all_kps_flattened = self.pred_data_array[frame_to_use, cam_idx, inst_idx, :]

                if not undistorted_images:
                    keypoint_data_all_kps_flattened = dutri.undistort_points(keypoint_data_all_kps_flattened, K, RDistort, TDistort)
                
                # Shape the flattened data back into (num_keypoints, 3) for easier iteration
                keypoint_data_all_kps_reshaped = keypoint_data_all_kps_flattened.reshape(-1, 3)

                # Iterate through each keypoint's (x,y,conf) for the current camera
                for kp_idx in range(self.dlc_data.num_keypoint):
                    point_2d = keypoint_data_all_kps_reshaped[kp_idx, :2] # (x, y)
                    confidence = keypoint_data_all_kps_reshaped[kp_idx, 2] # confidence

                    # Only add data if the confidence is above a threshold
                    if confidence >= self.confidence_cutoff:
                        keypoint_data_for_triangulation[inst_idx][kp_idx]['projs'][cam_idx] = P
                        keypoint_data_for_triangulation[inst_idx][kp_idx]['2d_pts'][cam_idx] = point_2d
                        keypoint_data_for_triangulation[inst_idx][kp_idx]['confs'][cam_idx] = confidence

        return keypoint_data_for_triangulation

    ###################################################################################################################################################

    def calculate_identity_swap_score(self, mode="full", parent_progress=None):
        if not self.dlc_data:
            return False

        if self.dlc_data.instance_count == 1:
            QMessageBox.information(self, "Single Instance", "Only one instance detected, no swap detection needed.")
            return False

        show_progress = True
        if mode == "full": # Create progress dialog
            start_frame = 0
            end_frame = self.total_frames
            window_title = "Identity Swap Calculation"
        elif mode == "check":
            start_frame = self.current_frame_idx
            end_frame = self.current_frame_idx + 1000
            if end_frame > self.total_frames:
                end_frame = self.total_frames
            show_progress = False
        elif self.current_frame_idx < self.total_frames - 1000: # remap mode
            start_frame = self.current_frame_idx + 1000 
            end_frame = self.total_frames
            window_title = f"Remap Swap at {start_frame - 1000}"
        else:
            return # No need for remapping when the remaining part is already remapped

        if show_progress:
            progress = QtWidgets.QProgressDialog("Calculating swap detection score...", "Cancel", start_frame, end_frame, self)
            progress.setWindowTitle(window_title)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)

            if parent_progress:
                # Position it below and slightly to the side of the parent dialog
                x = parent_progress.x() + 25
                y = parent_progress.y() + parent_progress.height() + 25
                progress.move(x, y)

        for frame in range(start_frame, end_frame):
            if show_progress:
                if progress.wasCanceled():
                    return False
                progress.setValue(frame)
            QtWidgets.QApplication.processEvents()  # Keep UI responsive
        
            valid_view = []
            skip_frame = False
            instances_detected_per_camera = self._validate_multiview_instances()

            for cam_idx in range(self.num_cam):
                if instances_detected_per_camera[cam_idx] < 2:
                    break
                else:
                    valid_view.append(cam_idx)

            if len(valid_view) < 4: # Only use frames with sufficient detections
                if self.is_debug:
                    print(f"DEBUG: Skippng frame {frame} due to insufficient detections. Only detected Camera {valid_view}.")
                continue

            for inst in range(self.dlc_data.instance_count):
                keypoint_data_for_triangulation = \
                    self._acquire_keypoint_data(instances_detected_per_camera, undistorted_images=False, frame_idx=frame)

                if not keypoint_data_for_triangulation:
                    skip_frame = True
                    break

            if skip_frame:
                if self.is_debug:
                    print(f"DEBUG: Skippng frame {frame} due to insufficient keypoints.")
                continue

            all_camera_pairs = list(combinations(valid_view, 2))

            # Initiate a numpy array for storing the 3D keypoints
            kp_3d_all_pair = np.full((len(all_camera_pairs), self.dlc_data.instance_count, self.dlc_data.num_keypoint, 3), np.nan)

            for pair_idx, (cam1_idx, cam2_idx) in enumerate(all_camera_pairs):
                for inst in range(self.dlc_data.instance_count): 
                    for kp_idx in range(self.dlc_data.num_keypoint):
                        if (cam1_idx in keypoint_data_for_triangulation[inst][kp_idx]['projs'] and 
                        cam2_idx in keypoint_data_for_triangulation[inst][kp_idx]['projs']):
                            
                            proj1 = keypoint_data_for_triangulation[inst][kp_idx]['projs'][cam1_idx]
                            proj2 = keypoint_data_for_triangulation[inst][kp_idx]['projs'][cam2_idx]
                            pts_2d1 = keypoint_data_for_triangulation[inst][kp_idx]['2d_pts'][cam1_idx]
                            pts_2d2 = keypoint_data_for_triangulation[inst][kp_idx]['2d_pts'][cam2_idx]
                            conf1 = keypoint_data_for_triangulation[inst][kp_idx]['confs'][cam1_idx]
                            conf2 = keypoint_data_for_triangulation[inst][kp_idx]['confs'][cam2_idx]

                            kp_3d_all_pair[pair_idx, inst, kp_idx] = \
                                dutri.triangulate_point_simple(proj1, proj2, pts_2d1, pts_2d2, conf1, conf2)

            mean_3d_kp = np.nanmean(kp_3d_all_pair, axis=0) # [inst, kp, xyz]  
            # Calculate the Euclidean distance between each pair's result and the mean
            diffs = np.linalg.norm(kp_3d_all_pair - mean_3d_kp, axis=-1) # [pair, inst, kp]
            total_diffs_per_pair = np.nansum(diffs, axis=(1, 2)) # [pair]

            if np.nanstd(total_diffs_per_pair) == 0:
                continue

            deviant_pair_idx = np.nanargmax(total_diffs_per_pair)
            deviant_pair = all_camera_pairs[deviant_pair_idx]
            deviant_cameras_scores = np.zeros(self.num_cam)
            
            deviant_cameras_scores[deviant_pair[0]] += 1
            deviant_cameras_scores[deviant_pair[1]] += 1
            
            swap_camera_idx = np.argmax(deviant_cameras_scores)
            
            self.swap_detection_score_array[frame, 1] = swap_camera_idx
            self.swap_detection_score_array[frame, 2] = total_diffs_per_pair[deviant_pair_idx]

        if show_progress:
            progress.close()
        if mode == "full":
            QMessageBox.information(self, "Swap Detection Score Calculated", "Identity swap detection completed.")
        self.populate_roi_frame_list()
        return True

    def populate_roi_frame_list(self):
        deviance_mask = self.swap_detection_score_array[:, 2] >= self.deviance_threshold
        self.roi_frame_list = np.where(deviance_mask)[0].tolist() # Get the indices of frames with significant deviations
        self._refresh_slider()

    def automatic_track_correction(self):
        if not self.dlc_data:
            QMessageBox.information(self, "No Data", "Load prediction data first!") 
            return

        if np.isnan(self.swap_detection_score_array[:,2]).all():
            if not self.calculate_identity_swap_score(mode="full"):
                return

        if self.dlc_data.instance_count != 2:
            QMessageBox.information(self, "Unimplemented", "The function is only for two instance only.")
            return
        
        correction_progress = QtWidgets.QProgressDialog("Commencing automatic track correction...", "Cancel", 0, self.total_frames, self)
        correction_progress.setWindowTitle("Automatic Correction Progress")
        correction_progress.setWindowModality(Qt.WindowModal)
        correction_progress.setValue(0)

        main_window_center = self.geometry().center()
        x = main_window_center.x() - correction_progress.width() // 2
        y = main_window_center.y() - correction_progress.height() // 2
        correction_progress.move(x, y)

        self.failed_frame_list = [] # Reset the failed frame list
        self.sus_frame_list = [] # Reset the suspicious frame list
        for frame_idx in range(self.total_frames):
            if not self.attempt_track_correction(frame_idx, correction_progress):
                QMessageBox.warning(self, "Correction Cancelled", "Track correction was cancelled by the user.")
                break

        correction_progress.close()
        if self.failed_frame_list:
            QMessageBox.warning(self, "Correction Partially Failed", f"Failed to correct frames: {', '.join(map(str, self.failed_frame_list))}.")
        else:
            QMessageBox.information(self, "Correction Successful", "All marked frames corrected successfully.") 

        self.is_saved = False

    def attempt_track_correction(self, frame_idx, correction_progress):
        if frame_idx not in self.roi_frame_list:
            return True
        if self.failed_frame_list and frame_idx <= self.failed_frame_list[-1] + 10:
            print(f"Skipping frame {frame_idx} as it is within 10 frames of the last failed correction.")
            return True
        if correction_progress.wasCanceled():
            return False
        
        correction_progress.setValue(frame_idx)

        self.current_frame_idx = frame_idx
        self.display_current_frame()
        print(f"Attempting automatic correction for frame {frame_idx}...")
        
        culprit_view_idx = int(self.swap_detection_score_array[frame_idx, 1])
        self.selected_cam_idx = culprit_view_idx
        self._refresh_selected_cam()

        backup_pred_data_array = self.pred_data_array.copy() # Backup current prediction data
        self.pred_data_array = dute.track_swap_3D_plotter(self.pred_data_array, frame_idx, self.selected_cam_idx)
        self.display_current_frame()
        self.adjust_3D_plot_view_angle()
        self.calculate_identity_swap_score(mode="check")
        
        max_retry = self.num_cam
        retry_count = 0

        while not self.validate_swap_effect(frame_idx):
            self.pred_data_array = backup_pred_data_array.copy() # Restore backup
            print(f"Frame {frame_idx} correction failed, retrying with next camera view...")
            retry_count += 1
            if retry_count >= max_retry:
                break

            self.selected_cam_idx = (self.selected_cam_idx + 1) % self.num_cam
            print(f"Trying {self.selected_cam_idx} for swapping...")
            self.pred_data_array = dute.track_swap_3D_plotter(self.pred_data_array, frame_idx, self.selected_cam_idx)
            self.display_current_frame()
            self.adjust_3D_plot_view_angle()
            self.calculate_identity_swap_score(mode="check")

        if retry_count < max_retry:
            self.calculate_identity_swap_score(mode="remap", parent_progress=correction_progress)
            self._refresh_slider()
        else:
            self.pred_data_array = backup_pred_data_array.copy() # Restore backup if max retries reached
            print("Try to swap two views simulatenously...")
            all_camera_pairs = list(combinations(range(self.num_cam), 2))
            for cam1_idx, cam2_idx in all_camera_pairs:
                self.pred_data_array = dute.track_swap_3D_plotter(self.pred_data_array, frame_idx, cam1_idx)
                self.pred_data_array = dute.track_swap_3D_plotter(self.pred_data_array, frame_idx, cam2_idx)
                self.selected_cam_idx = cam1_idx # Set the first camera as selected for display
                self.display_current_frame()
                self.adjust_3D_plot_view_angle()
                self.calculate_identity_swap_score(mode="check")

                if self.validate_swap_effect(frame_idx):
                    self.sus_frame_list.append(frame_idx)
                    self._refresh_slider()
                    print(f"Successfully swapped cameras {cam1_idx} and {cam2_idx} for frame {frame_idx}.")
                    break

            self.pred_data_array = backup_pred_data_array.copy() # Restore backup if no valid swap found
            self.failed_frame_list.append(frame_idx)
        return True

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
        self.calculate_identity_swap_score(mode="check")
        self._refresh_slider()
        self.is_saved = False

    def validate_swap_effect(self, frame_idx):
        end_check_idx = frame_idx + 100 if frame_idx + 100 < self.total_frames else self.total_frames
        error_frame_count = 0
        for check_frame_idx in range(frame_idx, end_check_idx):
            if check_frame_idx in self.roi_frame_list:
                error_frame_count += 1

        if error_frame_count > 90:
            return False
        
        return True

    def reset_marked_frames(self):
        """Reset all marked frames and clear the lists."""
        self.roi_frame_list = []
        self.failed_frame_list = []
        self.sus_frame_list = []
        self.swap_detection_score_array = np.full((self.total_frames, 3), np.nan)
        self.navigation_title_controller()
        self._refresh_slider()

    def refresh_failed_frame_list(self):
        """Refresh the failed frame list to only include frames that are also in the ROI frame list."""
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
            if not self.sus_frame_list:
                QMessageBox.information(self, "No Multi-Swap Frames", "No frames of multiple instance swap available to navigate.")
                return
            dugh.navigate_to_marked_frame(self, self.sus_frame_list, self.current_frame_idx, self._handle_frame_change_from_comp, direction)

    ###################################################################################################################################################

    def set_selected_camera(self, cam_idx):
        if hasattr(self, 'cap_list'):
            self.selected_cam_idx = cam_idx
            print(f"Selected Camera Index: {self.selected_cam_idx}")
            self.display_current_frame() # Refresh display to update border

    def change_frame(self, delta):
        self.selected_cam_idx = None # Clear the selected cam upon frame switch
        new_frame_idx = self.current_frame_idx + delta
        if 0 <= new_frame_idx < self.total_frames:
            self.current_frame_idx = new_frame_idx
            self.display_current_frame()
            self.navigation_title_controller()

        self.canvas.draw_idle()

    def navigation_title_controller(self):
        title_text = f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1} | View Mode: {self.view_mode_choice[self.current_view_mode_idx]}"
        if self.swap_detection_score_array is not None and self.swap_detection_score_array.shape[0] > 0:
            deviance_scores = self.swap_detection_score_array[:, 2]
            title_text += f" | Deviance Score: {deviance_scores[self.current_frame_idx]:.2f} (Threshold: {self.deviance_threshold})"
        self.nav_widget.setTitle(title_text)
        if self.current_frame_idx in self.failed_frame_list:
            self.nav_widget.setStyleSheet("""QGroupBox::title {color: #FF0000;}""")
        elif self.current_frame_idx in self.sus_frame_list:
            self.nav_widget.setStyleSheet("""QGroupBox::title {color: #FF00FF;}""")
        elif self.current_frame_idx in self.roi_frame_list:
            self.nav_widget.setStyleSheet("""QGroupBox::title {color: #F79F1C;}""")
        else:
            self.nav_widget.setStyleSheet("""QGroupBox::title {color: black;}""")

    ###################################################################################################################################################

    def show_confidence_dialog(self):
        if not self.pred_data_array:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        dialog = Adjust_Property_Dialog(
            property_name="Confidence Cutoff", property_val=self.confidence_cutoff, range_mult=100, parent=self)
        dialog.property_changed.connect(self._update_confidence_cutoff)
        dialog.show() # .show() instead of .exec() for a non-modal dialog

    def show_deviance_dialog(self):
        if not self.pred_data_array:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        dialog = Adjust_Property_Dialog(
            property_name="Deviance Threshold", property_val=self.deviance_threshold, range_mult=0.1, parent=self)
        dialog.property_changed.connect(self._update_deviance_threshold)
        dialog.show() # .show() instead of .exec() for a non-modal dialog

    def _update_confidence_cutoff(self, new_cutoff):
        self.confidence_cutoff = new_cutoff
        self.display_current_frame() # Redraw with the new cutoff

    def _update_deviance_threshold(self, new_threshold):
        self.deviance_threshold = new_threshold
        self.populate_roi_frame_list()
        self.navigation_title_controller()

    def _refresh_slider(self):
        self.progress_widget.set_frame_category("ROI frames", self.roi_frame_list, "#F79F1C") # Update ROI frames
        self.progress_widget.set_frame_category("Failed frames", self.failed_frame_list, "#FF0000", priority=7)
        self.progress_widget.set_frame_category("Multi-Swap frames", self.sus_frame_list, "#FF00FF", priority=6)

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

    def adjust_3D_plot_view_angle(self):
        if self.selected_cam_idx is None:
            QMessageBox.information(self, "No Camera Selected", "Please select a camera view first by clicking on one of the video frames.")
            return

        cam_pos = self.cam_pos[self.selected_cam_idx]
        direction_to_target = cam_pos

        hypot = np.linalg.norm(direction_to_target[:2]) # Length of the vector's projection on the xy plane
        elevation = np.arctan2(direction_to_target[2], hypot)

        elev_deg = np.degrees(elevation)

        # Calculate azimuth (angle in the xy plane)
        azimuth = np.arctan2(direction_to_target[1], direction_to_target[0])
        azim_deg = np.degrees(azimuth)

        self.ax.view_init(elev=elev_deg, azim=azim_deg)
        self.canvas.draw_idle()

    def on_scroll_3d_plot(self, event):
        """Handle matplotlib scroll events for zooming the 3D plot."""
        zoom_factor = 1.2
        if event.button == 'up':  # Zoom in (or typically 1 for wheel up)
            self.plot_lim = max(50, self.plot_lim / zoom_factor)
        elif event.button == 'down':  # Zoom out (or typically -1 for wheel down)
            self.plot_lim = min(1000, self.plot_lim * zoom_factor)
        
        self.plot_3d_points()
        self.canvas.draw_idle() # Ensure the canvas redraws

    ###################################################################################################################################################

    def save_workspace(self):
        """Save all the self variables in a hdf5 file in case saving goes awry or user wants to resume later"""
        if not self.dlc_data:
            QMessageBox.information(self, "No Data", "Load prediction data first!")
            return
        
        try:
            save_path = os.path.join(self.base_folder, "workspace_save.pickle")
            session_data = Session_3D_Plotter(
                base_folder=self.base_folder, calibration_filepath=self.calibration_filepath, dlc_config_filepath=self.dlc_config_filepath,
                pred_data_array=self.pred_data_array, current_frame_idx=self.current_frame_idx,
                confidence_cutoff=self.confidence_cutoff, deviance_threshold=self.deviance_threshold,
                roi_frame_list=self.roi_frame_list, failed_frame_list=self.failed_frame_list,
                sus_frame_list=self.sus_frame_list, swap_detection_score_array=self.swap_detection_score_array
            )
            with open(save_path, "wb") as f:
                pickle.dump(session_data, f)

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
