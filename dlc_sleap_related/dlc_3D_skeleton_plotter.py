import os
import glob

import h5py
import yaml
import scipy.io as sio

import numpy as np
import pandas as pd
import cv2

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QPushButton
from PySide6.QtCore import Signal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Todo:
#        Add a reprojection button to reproject the 3D coords back to the views
#        Move video ploting and reprojection to another thread or optional (toggled via a button)

DLC_CONFIG_DEBUG = "D:/Project/DLC-Models/COM3D/config.yaml"
CALIB_FILE_DEBUG = "D:/Project/SDANNCE-Models/4CAM-250620/SD-20250705-MULTI/sync_dannce.mat"
VIDEO_FOLDER_DEBUG = "D:/Project/SDANNCE-Models/4CAM-250620/SD-20250705-MULTI/Videos"

class ClickableVideoLabel(QtWidgets.QLabel):
    clicked = Signal(int) # Signal to emit cam_idx when clicked

    def __init__(self, cam_idx, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.setMouseTracking(True) # Enable mouse tracking for hover effects if needed

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.cam_idx)
        super().mousePressEvent(event)

class DLC_3D_plotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC 3D Plotter")
        self.setGeometry(100, 100, 1600, 960)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.load_dlc_config_button = QPushButton("1. Load DLC Configs")
        self.load_calibrations_button = QPushButton("2. Load Calibrations")
        self.load_video_folder_button = QPushButton("3. Load Videos & Predictions")

        self.button_layout.addWidget(self.load_dlc_config_button)
        self.button_layout.addWidget(self.load_calibrations_button)
        self.button_layout.addWidget(self.load_video_folder_button)
        self.layout.addLayout(self.button_layout)

        self.display_layout = QtWidgets.QHBoxLayout()
        self.video_layout = QtWidgets.QGridLayout()
        
        self.video_labels = [] # Store video labels in a list for easy access
        for row in range(2):
            for col in range(2):
                cam_idx = row * 2 + col # 0-indexed camera index
                label = ClickableVideoLabel(cam_idx, self) # Use the custom label
                label.setText(f"Video {cam_idx + 1}")
                label.setAlignment(Qt.AlignCenter) # Center the "Video X" text
                label.setFixedSize(480, 360) # Set a fixed size for video display
                label.setStyleSheet("border: 1px solid gray;") # Add a border for visibility
                self.video_layout.addWidget(label, row, col)
                self.video_labels.append(label)

        #Store plot and a slider for adjust plot size
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.size_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 301)
        self.size_slider.setTracking(True)
        self.plot_layout.addWidget(self.size_slider)
        self.size_slider.setValue(300)
        self.size_slider.hide()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.display_layout.addLayout(self.video_layout)
        self.display_layout.addLayout(self.plot_layout)
        self.layout.addLayout(self.display_layout, 1)

        # Progress bar
        self.progress_layout = QtWidgets.QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40) # Slightly wider button
        self.progress_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0) # Will be set dynamically
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        self.playback_timer = QTimer(self) # Pass self to QTimer
        self.playback_timer.timeout.connect(self.autoplay_video)
        self.is_playing = False
        self.layout.addLayout(self.progress_layout)

        # Navigation controls
        self.navigation_group_box = QtWidgets.QGroupBox("Video Navigation")
        self.navigation_layout = QtWidgets.QHBoxLayout(self.navigation_group_box)

        self.prev_10_frames_button = QPushButton("Prev 10 Frames (Shift + ←)")
        self.prev_frame_button = QPushButton("Prev Frame (←)")
        self.next_frame_button = QPushButton("Next Frame (→)")
        self.next_10_frames_button = QPushButton("Next 10 Frames (Shift + →)")
        self.reset_3d_view_button = QPushButton("Reset 3D View Angle (r)")

        self.navigation_layout.addWidget(self.prev_10_frames_button)
        self.navigation_layout.addWidget(self.prev_frame_button)
        self.navigation_layout.addWidget(self.next_frame_button)
        self.navigation_layout.addWidget(self.next_10_frames_button)
        self.navigation_layout.addWidget(self.reset_3d_view_button)

        self.layout.addWidget(self.navigation_group_box)
        self.navigation_group_box.hide() # Hide until videos are loaded

        # Connect buttons to events
        self.load_dlc_config_button.clicked.connect(self.load_dlc_config)
        self.load_video_folder_button.clicked.connect(self.open_video_folder_dialog)
        self.load_calibrations_button.clicked.connect(self.load_calibrations)

        for label in self.video_labels:
            label.clicked.connect(self.set_selected_camera)

        self.size_slider.sliderMoved.connect(self.set_plot_lim_from_slider)
        self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.play_button.clicked.connect(self.toggle_playback)

        self.prev_10_frames_button.clicked.connect(lambda: self.change_frame(-10))
        self.prev_frame_button.clicked.connect(lambda: self.change_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self.change_frame(1))
        self.next_10_frames_button.clicked.connect(lambda: self.change_frame(10))

        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_R), self).activated.connect(self.reset_3d_view)

        self.num_cam = None

        self.confidence_cutoff = 0.6 # Initialize confidence cutoff

        self.multi_animal, self.keypoints, self.skeleton, self.individuals = False, None, None, None
        self.instance_count = 1

        self.num_cam_from_calib = None

        self.plot_lim = 300
        self.instance_color = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)] # BGR
        
        self.current_frame_idx = 0      # Single frame index for all synchronized videos
        self.total_frames = 0      # Max frames across all videos
        self.selected_cam_idx = 0  # Selected view, default to camera 0

        self.is_debug = True

    def open_video_folder_dialog(self):
        if self.is_debug:
            self.load_dlc_config(DLC_CONFIG_DEBUG)
            self.load_calibrations(CALIB_FILE_DEBUG)
            self.load_video_folder(VIDEO_FOLDER_DEBUG)
            return
        if self.keypoints is None:
            QMessageBox.warning(self, "Warning", "DLC config is not loaded, load DLC config first!")
            print("DLC config is not loaded, load DLC config first!")
            self.load_dlc_config()
            if self.keypoints is None: # User close DLC loading window
                return
        if self.num_cam_from_calib is None:
            QMessageBox.warning(self, "Warning", "Calibrations are not loaded, load calibrations first!")
            print("Calibrations are not loaded, load calibrations first!")
            self.load_calibrations()
            if self.num_cam_from_calib is None: # User close calibration loading window
                return
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder_path:
            self.load_video_folder(folder_path)
   
    def load_dlc_config(self, dlc_config=None):
        if dlc_config is None:
            file_dialog = QtWidgets.QFileDialog(self)
            dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        with open(dlc_config, "r") as conf:
            cfg = yaml.safe_load(conf)
        if cfg:
            print(f"DLC Config loaded: {dlc_config}")
            QMessageBox.information(self, "Success", "DLC Config loaded successfully!")
        self.multi_animal = cfg["multianimalproject"]
        self.keypoints = cfg["bodyparts"] if not self.multi_animal else cfg["multianimalbodyparts"]
        self.skeleton = cfg["skeleton"]
        self.individuals = cfg["individuals"]
        self.instance_count = len(self.individuals) if self.individuals is not None else 1
        self.num_keypoints = len(self.keypoints)
        self.keypoint_to_idx = {name: idx for idx, name in enumerate(self.keypoints)}

    def load_calibrations(self, calib_file=None):
        if calib_file is None:
            calib_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Calibration File", "", "Calibration Files (*.mat)")
        calibration_file = calib_file
        print(f"Calibration loaded: {calibration_file}")
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
            return
        self.camera_params = [{} for _ in range(self.num_cam_from_calib)]
        self.load_calibration_mat(calib)
        self.plot_camera_geometry()

    def load_video_folder(self, folder_path):
        self.num_cam = len([f for f in os.listdir(folder_path) if f.startswith("Camera")])
        self.video_list = [None] * self.num_cam
        self.cap_list = [None] * self.num_cam
        folder_list = [None] * self.num_cam
        print(f"Loading videos from: {folder_path}")
        for i in range(self.num_cam):  # Loop through expected camera folders
            folder = os.path.join(folder_path, f"Camera{i+1}")
            video_file = os.path.join(folder, "0.mp4")
            self.video_list[i] = video_file
            
            cap = cv2.VideoCapture(video_file) # Try to open the video capture
            if not cap.isOpened():
                print(f"Warning: Could not open video file: {video_file}")
                cap = None
            elif self.total_frames == 0: # Fallback if not loaded frame counts from calibration file
                # Update max_frames based on the longest video
                num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_frame > self.total_frames:
                    self.total_frames = num_frame
                print(num_frame)

            self.cap_list[i] = cap # Add the capture object (or None) to the list
            folder_list[i] = folder

        if not self.cap_list:
            QMessageBox.warning(self, "Error", "No video files were loaded successfully.")
            return

        self.pred_data_array = np.full((self.total_frames, self.num_cam, self.instance_count, self.num_keypoints*3), np.nan)
        for k in range(self.num_cam):
            if folder_list[k] is not None:
                folder = folder_list[k]
                self.load_prediction(k, folder)

        self.current_frame_idx = 0
        self.progress_slider.setRange(0, self.total_frames - 1)
        self.progress_slider.setValue(0)
        self.navigation_group_box.show()
        self.display_current_frame() # Display the first frames

    def load_prediction(self, cam_idx, folder):
        h5_files = glob.glob(os.path.join(folder, "*.h5"))
        if not h5_files:
            print(f"Warning: No .h5 prediction file found in {folder}")
            return False
        try:
            with h5py.File(h5_files[0], 'r') as pred_file:
                if "tracks" not in pred_file.keys():
                    print(f"Error: Prediction file {h5_files[0]} not valid, no 'tracks' key found.")
                    return False

                pred_data_raw = pred_file["tracks"]["table"][:]
                pred_data_values = np.array([item[1] for item in pred_data_raw])
                for inst in range(self.instance_count): # Sort inst out
                    self.pred_data_array[:,cam_idx,inst,:] = pred_data_values[:, inst*self.num_keypoints*3:(inst+1)*self.num_keypoints*3]
                
        except Exception as e:
            print(f"Error loading H5 file {h5_files[0]}: {e}")
            return False

    def load_calibration_mat(self, calib):
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
            self.camera_params[i]["P"] = self.get_projection_matrix(K,r,t)
            frame_count[i] = len(calib["sync"][i,0][0,0]["data_sampleID"][0])
        self.cam_pos = np.array(cam_pos)
        self.cam_dir = np.array(cam_dir)
        self.total_frames = max(frame_count)

    ###################################################################################################################################################

    def display_current_frame(self):
        for i, cap in enumerate(self.cap_list):
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = cap.read()
                if ret:
                    if np.all(np.isnan(self.pred_data_array[self.current_frame_idx,i,:,:])):
                        return frame
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
            
            # Update border color based on selection
            if i == self.selected_cam_idx:
                self.video_labels[i].setStyleSheet("border: 2px solid red;")
            else:
                self.video_labels[i].setStyleSheet("border: 1px solid gray;")

        self.progress_slider.setValue(self.current_frame_idx) # Update the slider after all frames are displayed
        self.plot_3d_points()

    def plot_2d_points(self, frame, cam_idx):
        # Iterate over each individual (animal)
        for inst in range(self.instance_count):
            color_rgb = self.instance_color[inst % len(self.instance_color)]
            # Convert RGB to BGR for OpenCV
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            keypoint_coords = dict()
            for kp_idx in range(self.num_keypoints):
                kp = self.pred_data_array[self.current_frame_idx,cam_idx,inst,kp_idx*3:kp_idx*3+3]
                if pd.isna(kp[0]) or kp[2] < self.confidence_cutoff:
                    continue
                x, y = kp[0], kp[1]
                keypoint_coords[kp_idx] = (int(x),int(y))
                cv2.circle(frame, (int(x), int(y)), 3, color_bgr, -1) # Draw the dot representing the keypoints

            if self.individuals is not None and len(keypoint_coords) >= 2: # Only plot bounding box with more than one points
                self.plot_bounding_box(keypoint_coords, frame, color_bgr, inst)
            if self.skeleton:
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
        cv2.putText(frame, f"Instance: {self.individuals[inst]}", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        return frame
    
    def plot_2d_skeleton(self, keypoint_coords, frame, color):
        for start_kp, end_kp in self.skeleton:
            start_kp_idx = self.keypoint_to_idx[start_kp]
            end_kp_idx = self.keypoint_to_idx[end_kp]
            start_coord = keypoint_coords.get(start_kp_idx)
            end_coord = keypoint_coords.get(end_kp_idx)
            if start_coord and end_coord:
                cv2.line(frame, start_coord, end_coord, color, 2)
        return frame
    
    def plot_3d_points(self):
        point_3d_array = self.data_loader_for_3d_plot()
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"3D Skeleton Plot - Frame {self.current_frame_idx}")
        self.ax.set_xlim([-self.plot_lim, self.plot_lim])
        self.ax.set_ylim([-self.plot_lim, self.plot_lim])
        self.ax.set_zlim([-self.plot_lim, self.plot_lim])
        self.size_slider.show()

        for inst in range(self.instance_count):
            color = self.instance_color[inst % len(self.instance_color)]
            # Plot 3D keypoints
            for kp_idx in range(self.num_keypoints):
                point_3d = point_3d_array[inst, kp_idx, :]
                if point_3d[0] is not None:
                    self.ax.scatter(point_3d[0], point_3d[1], point_3d[2], color=np.array(color)/255, s=50)

            # Plot 3D skeleton
            if self.skeleton:
                for start_kp, end_kp in self.skeleton:
                    start_kp_idx = self.keypoint_to_idx[start_kp]
                    end_kp_idx = self.keypoint_to_idx[end_kp]
                    start_point = point_3d_array[inst, start_kp_idx, :]
                    end_point = point_3d_array[inst, end_kp_idx, :]
                    if start_point is not None and end_point is not None:
                        self.ax.plot([start_point[0], end_point[0]],
                                     [start_point[1], end_point[1]],
                                     [start_point[2], end_point[2]],
                                     color=np.array(color)/255)
        self.canvas.draw_idle() # Redraw the 3D canvas

    def plot_camera_geometry(self):
        """Plots the relative geometry on a given Axes3D object."""
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
        point_3d_array = np.full((self.instance_count, self.num_keypoints, 3), np.nan)

        # Determine how many instances each camera detects in the current frame
        if self.instance_count > 1:
            instances_detected_per_camera = [0] * self.num_cam
            for cam_idx_check in range(self.num_cam):
                detected_instances_in_cam = 0
                for inst_check in range(self.instance_count):
                    has_valid_data = False
                    for kp_idx_check in range(self.num_keypoints):
                        if self.pred_data_array[self.current_frame_idx,cam_idx_check, inst_check, kp_idx_check*3] is not None:
                            has_valid_data = True
                            break
                    if has_valid_data:
                        detected_instances_in_cam += 1
                instances_detected_per_camera[cam_idx_check] = detected_instances_in_cam
        else:
            instances_detected_per_camera = [1] * self.num_cam # All cameras valid for the single instance

        for inst in range(self.instance_count):
            # Dictionary to store per-keypoint data across cameras:
            keypoint_data_for_triangulation = {
                kp_idx: {'projs': [], '2d_pts': [], 'confs': []}
                for kp_idx in range(self.num_keypoints)
            }

            for cam_idx in range(self.num_cam):
                if self.instance_count > 1 and instances_detected_per_camera[cam_idx] == 1:
                    continue # Skip if this camera only detected one instance in multi-instance scenario

                RDistort = self.camera_params[cam_idx]['RDistort']
                TDistort = self.camera_params[cam_idx]['TDistort']
                K = self.camera_params[cam_idx]['K']
                P = self.camera_params[cam_idx]['P']

                # Get all keypoint data (flattened) for the current frame, camera, and instance
                keypoint_data_all_kps_flattened = self.pred_data_array[self.current_frame_idx, cam_idx, inst, :]

                if not undistorted_images:
                    keypoint_data_all_kps_flattened = self.undistort_points(keypoint_data_all_kps_flattened, K, RDistort, TDistort)
                
                # Shape the flattened data back into (num_keypoints, 3) for easier iteration
                keypoint_data_all_kps_reshaped = keypoint_data_all_kps_flattened.reshape(-1, 3)

                # Iterate through each keypoint's (x,y,conf) for the current camera
                for kp_idx in range(self.num_keypoints):
                    point_2d = keypoint_data_all_kps_reshaped[kp_idx, :2] # (x, y)
                    confidence = keypoint_data_all_kps_reshaped[kp_idx, 2] # confidence

                    # Only add data if the confidence is above a threshold (or another validity check)
                    if confidence >= self.confidence_cutoff:
                        keypoint_data_for_triangulation[kp_idx]['projs'].append(P)
                        keypoint_data_for_triangulation[kp_idx]['2d_pts'].append(point_2d)
                        keypoint_data_for_triangulation[kp_idx]['confs'].append(confidence)

            # iterate through each keypoint to perform triangulation
            for kp_idx in range(self.num_keypoints):
                projs = keypoint_data_for_triangulation[kp_idx]['projs']
                pts_2d = keypoint_data_for_triangulation[kp_idx]['2d_pts']
                confs = keypoint_data_for_triangulation[kp_idx]['confs']
                num_valid_views = len(projs)

                if num_valid_views >= 2:
                    point_3d_array[inst, kp_idx, :] = self.triangulate_point(num_valid_views, projs, pts_2d, confs)

        return point_3d_array


    @staticmethod
    def triangulate_point(num_views, projs, pts_2d, confs):
        """
        Triangulates a single 3D point from multiple 2D camera views using the Direct Linear Transformation (DLT) method.
        Each 2D point's contribution to the system of equations is weighted by its confidence.

        Args:
            num_views (int): The number of camera views providing observations for this point.
            projs (list of np.array): A list of 3x4 projection matrices, one for each camera view.
            pts_2d (list of tuple/np.array): A list of 2D image points (u, v), one for each camera view.
            confs (list of float): A list of confidence values, one for each 2D point. Used as weights in the triangulation.

        Returns:
            np.array: The triangulated 3D point in Euclidean coordinates (x, y, z).
        """
        A = []
        for i in range(num_views):
            P_i = projs[i]
            u, v = pts_2d[i]
            w = confs[i] # Weight by confidence

            P_i = np.array(P_i) # Ensure P_i is a numpy array for slicing

            # Equations for DLT:
            # u * P_i[2,:] - P_i[0,:] = 0
            # v * P_i[2,:] - P_i[1,:] = 0
            # Apply weight 'w' to each row
            A.append(w * (u * P_i[2,:] - P_i[0,:]))
            A.append(w * (v * P_i[2,:] - P_i[1,:]))

        A = np.array(A) # Solve Ax = 0 using SVD
        U, S, Vt = np.linalg.svd(A) # The 3D point is the last column of V (or last row of Vt)
        
        point_4d_hom = Vt[-1] 
        point_3d = (point_4d_hom / point_4d_hom[3]).flatten()[:3] # Convert from homogeneous to Euclidean coordinates (x/w, y/w, z/w)
        return point_3d

    @staticmethod
    def get_projection_matrix(K, R, t):
        """
        Computes the projection matrix from camera intrinsic and extrinsic parameters.

        The projection matrix P combines the camera's intrinsic properties (K)
        with its extrinsic pose (rotation R and translation t) relative to the
        world coordinate system. It maps 3D world points to 2D image points.

        Args:
            K (np.array): The camera intrinsic matrix (3x3).
                        Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            R (np.array): The 3x3 rotation matrix representing the camera's orientation
                        in the world coordinate system.
            t (np.array): The 3x1 or (3,) translation vector representing the camera's
                        position in the world coordinate system.

        Returns:
            np.array: The 3x4 projection matrix P = K * [R | t].
        """
        # Ensure t is a 3x1 column vector
        if t.shape == (3,):
            t = t.reshape(3, 1)
        elif t.shape == (3, 1):
            pass
        else:
            raise ValueError("Translation vector 't' must be of shape (3,) or (3,1)")

        # Concatenate R and t to form the extrinsic matrix [R | t]
        extrinsic_matrix = np.hstack((R, t))
        
        # Projection matrix
        P = K @ extrinsic_matrix
        return P

    @staticmethod
    def undistort_points(points_xy_conf, K, RDistort, TDistort):
        """
        Undistorts 2D image points given camera intrinsic matrix and distortion coefficients.

        Args:
            points_xy_conf (list or np.array): A 1D array or list of (x, y, confidence) triplets.
                                            Example: [x1, y1, conf1, x2, y2, conf2, ...]
            K (np.array): The camera intrinsic matrix (3x3).
                        Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            RDistort (list or np.array): Radial distortion coefficients [k1, k2].
            TDistort (list or np.array): Tangential distortion coefficients [p1, p2].

        Returns:
            np.array: A 1D array of undistorted (x, y, confidence) triplets.
                    Example: [undistorted_x1, undistorted_y1, conf1, ...]
        """
        if points_xy_conf.size % 3 != 0:
            raise ValueError("Input 'points_xy_conf' must contain triplets of (x, y, confidence).")
        
        num_points = points_xy_conf.size // 3
        dist_coeffs = np.array([RDistort[0], RDistort[1], TDistort[0], TDistort[1], 0])
        points_xy_conf = np.array(points_xy_conf, dtype=np.float32)

        reshaped_points = points_xy_conf.reshape(-1, 3) # Separate (x, y) coordinates from confidences
        xy_coords = reshaped_points[:, :2]    # Shape (N, 2)
        confidences = reshaped_points[:, 2]   # Shape (N,)

        points_xy_conf = xy_coords.reshape(-1, 1, 2).astype(np.float32) # Reshape points for OpenCV: (N, 1, 2)
        undistorted_pts = cv2.undistortPoints(points_xy_conf, K, dist_coeffs, P=K)

        undistorted_pts_clean = undistorted_pts.reshape(-1, 2) #Reshape the undistorted (x, y) back to a simple (N, 2) array

        output_combined = np.empty((num_points, 3), dtype=np.float32) #Create an empty array to hold the final (x, y, conf) triplets
        output_combined[:, :2] = undistorted_pts_clean
        output_combined[:, 2] = confidences

        return output_combined.flatten() # Reshape back to 1D array

    ###################################################################################################################################################

    def set_selected_camera(self, cam_idx):
        if hasattr(self, 'cap_list'):
            self.selected_cam_idx = cam_idx
            print(f"Selected Camera Index: {self.selected_cam_idx}")
            self.display_current_frame() # Refresh display to update border
        else:
            pass

    def change_frame(self, delta):
        new_frame_idx = self.current_frame_idx + delta
        if 0 <= new_frame_idx < self.total_frames:
            self.current_frame_idx = new_frame_idx
            self.display_current_frame()
            self.navigation_box_title_controller()

    def set_frame_from_slider(self, value):
        self.current_frame_idx = value
        self.display_current_frame()
        self.navigation_box_title_controller()

    def set_plot_lim_from_slider(self):
        self.plot_lim = self.size_slider.value()
        self.plot_3d_points()
        self.canvas.draw_idle()

    def autoplay_video(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.display_current_frame()
            self.navigation_box_title_controller()
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

    def toggle_playback(self):
        if not self.is_playing:
            self.playback_timer.start(1000/50) # 50 fps
            self.play_button.setText("■")
            self.is_playing = True
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

    def reset_3d_view(self):
        self.ax.view_init(elev=20, azim=-60) # Set to default view angle
        self.canvas.draw_idle() 

    def navigation_box_title_controller(self):
        self.navigation_group_box.setTitle(f"Video Navigation | Frame: {self.current_frame_idx} / {self.total_frames-1}")

    ###################################################################################################################################################

    def closeEvent(self, event: QCloseEvent):
        # Ensure all VideoCapture objects are released when the window closes
        if hasattr(self, 'cap_list'):
            for cap in self.cap_list:
                if cap and cap.isOpened():
                    cap.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = DLC_3D_plotter()
    main_window.show()
    app.exec()
