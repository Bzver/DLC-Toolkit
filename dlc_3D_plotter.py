import os
import glob

import scipy.io as sio

import numpy as np
import pandas as pd
import cv2

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from utils.dtu_comp import Clickable_Video_Label
from utils.dtu_io import DLC_Loader
from utils.dtu_widget import Menu_Widget, Progress_Widget, Nav_Widget
import utils.dtu_helper as duh
import utils.dtu_gui_helper as dugh
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
                "display_name": "File",
                "buttons": [
                    ("Load DLC Configs", self.load_dlc_config),
                    ("Load Calibrations", self.load_calibrations),
                    ("Load Videos and Predictions", self.open_video_folder_dialog)
                ]
            },
            "Edit": {
                "display_name": "Edit",
                "buttons": [
                    ("Mark / Unmark Current Frame (X)", self.wip_unimplemented),
                    ("Adjust Confidence Cutoff", self.wip_unimplemented),
                    ("Refine Tracks", self.call_track_refiner)
                ]
            }
        }
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
        self.nav_widget.prev_marked_frame_sig.connect(self.wip_unimplemented)
        self.nav_widget.next_marked_frame_sig.connect(self.wip_unimplemented)

        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.progress_widget.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self.wip_unimplemented)

        self.canvas.mpl_connect("scroll_event", self.on_scroll_3d_plot)

        self.reset_state()

    def wip_unimplemented(self):
        QMessageBox.information(self, "Unimplemented", "This function has yet to be implemented.")
        pass

    def reset_state(self):
        self.num_cam = None

        self.confidence_cutoff = 0.6 # Initialize confidence cutoff

        self.data_loader = DLC_Loader(None, None)
        self.dlc_data = None
        self.keypoint_to_idx = {}

        self.pred_data_array = None # Combined prediction data for all cameras

        self.num_cam_from_calib = None

        self.plot_lim = 300
        self.instance_color = [(255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)] # RGB
        
        self.current_frame_idx = 0
        self.total_frames = 0
        self.selected_cam_idx = None

        self.refiner_window = None

    def open_video_folder_dialog(self):
        if self.is_debug:
            self.dlc_config_loader(DLC_CONFIG_DEBUG)
            self.calibration_loader(CALIB_FILE_DEBUG)
            self.load_video_folder(VIDEO_FOLDER_DEBUG)
            return
        
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
        
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder_path:
            self.load_video_folder(folder_path)

    def load_dlc_config(self):
        file_dialog = QtWidgets.QFileDialog(self)
        dlc_config_filepath, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        if dlc_config_filepath:
            self.dlc_config_loader(dlc_config_filepath)

    def dlc_config_loader(self, dlc_config_filepath: str):
        """
        Loads DLC configuration using DLC_Data_Loader.
        """
        self.data_loader.dlc_config_filepath = dlc_config_filepath
        try:
            self.dlc_data = dugh.load_and_show_message(self, self.data_loader, metadata_only=True)
            self.keypoint_to_idx = {name: idx for idx, name in enumerate(self.dlc_data.keypoints)}
        except:
            QMessageBox.critical(self, "Error", "Failed to load DLC config.")
            traceback.print_exc()

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

        self.current_frame_idx = 0
        self.progress_widget.set_slider_range(self.total_frames)
        self.nav_widget.show()
        self.display_current_frame() # Display the first frames

    def load_prediction(self, cam_idx:int, prediction_filepath:str):
        """
        Loads prediction data for a specific camera using DLC_Data_Loader
        and populates the main pred_data_array.
        """
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

            # Update border color based on selection
            if i == self.selected_cam_idx:
                self.video_labels[i].setStyleSheet("border: 2px solid red;")
            else:
                self.video_labels[i].setStyleSheet("border: 1px solid gray;")

        self.plot_3d_points()

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
        self.ax.set_zlim([-self.plot_lim, self.plot_lim])

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
        self.ax.set_title(f"3D Camera Geometry - Frame {self.current_frame_idx}")
        self.ax.set_xlim([-self.plot_lim, self.plot_lim])
        self.ax.set_ylim([-self.plot_lim, self.plot_lim])
        self.ax.set_zlim([-self.plot_lim, self.plot_lim])

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

        # Determine how many instances each camera detects in the current frame
        if self.dlc_data.instance_count > 1:
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
        else:
            instances_detected_per_camera = [1] * self.num_cam # All cameras valid for the single instance

        for inst in range(self.dlc_data.instance_count):
            # Dictionary to store per-keypoint data across cameras:
            keypoint_data_for_triangulation = {
                kp_idx: {'projs': [], '2d_pts': [], 'confs': []}
                for kp_idx in range(self.dlc_data.num_keypoint)
            }

            for cam_idx in range(self.num_cam):
                if self.dlc_data.instance_count > 1 and instances_detected_per_camera[cam_idx] == 1:
                    continue # Skip if this camera only detected one instance in multi-instance scenario

                # Ensure camera_params are available for the current camera index
                if cam_idx >= len(self.camera_params) or not self.camera_params[cam_idx]:
                    print(f"Warning: Camera parameters not available for camera {cam_idx}. Skipping.")
                    continue

                RDistort = self.camera_params[cam_idx]['RDistort']
                TDistort = self.camera_params[cam_idx]['TDistort']
                K = self.camera_params[cam_idx]['K']
                P = self.camera_params[cam_idx]['P']

                # Get all keypoint data (flattened) for the current frame, camera, and instance
                keypoint_data_all_kps_flattened = self.pred_data_array[self.current_frame_idx, cam_idx, inst, :]

                if not undistorted_images:
                    keypoint_data_all_kps_flattened = dutri.undistort_points(keypoint_data_all_kps_flattened, K, RDistort, TDistort)
                
                # Shape the flattened data back into (num_keypoints, 3) for easier iteration
                keypoint_data_all_kps_reshaped = keypoint_data_all_kps_flattened.reshape(-1, 3)

                # Iterate through each keypoint's (x,y,conf) for the current camera
                for kp_idx in range(self.dlc_data.num_keypoint):
                    point_2d = keypoint_data_all_kps_reshaped[kp_idx, :2] # (x, y)
                    confidence = keypoint_data_all_kps_reshaped[kp_idx, 2] # confidence

                    # Only add data if the confidence is above a threshold (or another validity check)
                    if confidence >= self.confidence_cutoff:
                        keypoint_data_for_triangulation[kp_idx]['projs'].append(P)
                        keypoint_data_for_triangulation[kp_idx]['2d_pts'].append(point_2d)
                        keypoint_data_for_triangulation[kp_idx]['confs'].append(confidence)

            # iterate through each keypoint to perform triangulation
            for kp_idx in range(self.dlc_data.num_keypoint):
                projs = keypoint_data_for_triangulation[kp_idx]['projs']
                pts_2d = keypoint_data_for_triangulation[kp_idx]['2d_pts']
                confs = keypoint_data_for_triangulation[kp_idx]['confs']
                num_valid_views = len(projs)

                if num_valid_views >= 2:
                    point_3d_array[inst, kp_idx, :] = dutri.triangulate_point(num_valid_views, projs, pts_2d, confs)

        return point_3d_array

    ###################################################################################################################################################

    def set_selected_camera(self, cam_idx):
        if hasattr(self, 'cap_list'):
            self.selected_cam_idx = cam_idx
            print(f"Selected Camera Index: {self.selected_cam_idx}")
            self.display_current_frame() # Refresh display to update border
        else:
            pass

    def change_frame(self, delta):
        self.selected_cam_idx = None # Clear the selected cam upon frame switch
        new_frame_idx = self.current_frame_idx + delta
        if 0 <= new_frame_idx < self.total_frames:
            self.current_frame_idx = new_frame_idx
            self.display_current_frame()
            self.navigation_title_controller()

        self.canvas.draw_idle()

    def navigation_title_controller(self):
        self.nav_widget.setTitle(f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1}")

    ###################################################################################################################################################

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.current_frame_idx = new_frame_idx
        self.display_current_frame()
        self.navigation_title_controller()

    def on_scroll_3d_plot(self, event):
        """Handle matplotlib scroll events for zooming the 3D plot."""
        print(f"Matplotlib scroll event received - button: {event.button}, step: {event.step}")
        
        zoom_factor = 1.2
        if event.button == 'up':  # Zoom in (or typically 1 for wheel up)
            self.plot_lim = max(50, self.plot_lim / zoom_factor)
        elif event.button == 'down':  # Zoom out (or typically -1 for wheel down)
            self.plot_lim = min(1000, self.plot_lim * zoom_factor)
        
        print(f"Zoom level changed to: {self.plot_lim}")
        self.plot_3d_points()
        self.canvas.draw_idle() # Ensure the canvas redraws

    def closeEvent(self, event: QCloseEvent):
        # Ensure all VideoCapture objects are released when the window closes
        if hasattr(self, 'cap_list'):
            for cap in self.cap_list:
                if cap and cap.isOpened():
                    cap.release()
        event.accept()

###################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = DLC_3D_plotter()
    main_window.show()
    app.exec()