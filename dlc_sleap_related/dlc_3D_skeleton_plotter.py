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
        self.size_slider.setRange(1, 300)
        self.size_slider.setTracking(True)
        self.plot_layout.addWidget(self.size_slider)
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

        self.num_cam = 4
        self.video_list = [None] * self.num_cam
        self.cap_list = [None] * self.num_cam
        self.pred_data_list = [None] * self.num_cam

        self.confidence_cutoff = 0.6 # Initialize confidence cutoff

        self.multi_animal, self.keypoints, self.skeleton, self.individuals = False, None, None, None
        self.instance_count = 1

        self.cam_pos, self.cam_dir = [None] * self.num_cam, [None] * self.num_cam

        self.camera_params = [{} for _ in range(self.num_cam)]
        self.keypoint_coords = {}
        self.keypoint_coords_3d = {}

        self.plot_lim = 300
        self.instance_color = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)] # BGR
        
        self.current_frame_idx = 0      # Single frame index for all synchronized videos
        self.total_frames = 0      # Max frames across all videos
        self.selected_cam_idx = 0  # Default to camera 0

    def open_video_folder_dialog(self):
        if self.keypoints is None:
            print("DLC config is not loaded, load DLC config first!")
            self.load_dlc_config()
            if self.keypoints is None: # User close DLC loading window
                return
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder_path:
            self.load_video_folder(folder_path)

    def load_video_folder(self, folder_path):
        print(f"Loading videos from: {folder_path}")
        max_frames = 0
        for i in range(self.num_cam):  # Loop through expected camera folders
            folder = os.path.join(folder_path, f"Camera{i+1}")
            video_file = os.path.join(folder, "0.mp4")
            self.video_list[i] = video_file
            
            cap = cv2.VideoCapture(video_file) # Try to open the video capture
            if not cap.isOpened():
                print(f"Warning: Could not open video file: {video_file}")
                cap = None
            else:
                # Update max_frames based on the longest video
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_frames > max_frames:
                    max_frames = num_frames

            self.cap_list[i] = cap # Add the capture object (or None) to the list

            self.load_prediction(i, folder, cap, num_frames)

        if not self.cap_list:
            QMessageBox.warning(self, "Error", "No video files were loaded successfully.")
            return

        self.total_frames = max_frames
        self.current_frame_idx = 0
        self.progress_slider.setRange(0, self.total_frames - 1)
        self.progress_slider.setValue(0)
        self.navigation_group_box.show()
        self.display_current_frame() # Display the first frames

    def load_prediction(self, cam_idx, folder, cap, num_frame):
        h5_files = glob.glob(os.path.join(folder, "*.h5"))
        if not h5_files:
            print(f"Warning: No .h5 prediction file found in {folder}")
            return
        
        try:
            with h5py.File(h5_files[0], 'r') as pred_file:
                if "tracks" not in pred_file.keys():
                    print(f"Error: Prediction file {h5_files[0]} not valid, no 'tracks' key found.")
                    self.pred_data_list[cam_idx] = None
                    return
                
                pred_data = pred_file["tracks"]["table"][:]
                self.pred_data_list[cam_idx] = pred_data                    
                
                # Check for frame count mismatch only if video was loaded successfully
                if cap is not None:
                    pred_frame_count = pred_data.shape[0] # Assuming first dimension is frame count
                    if pred_frame_count != num_frame:
                        QMessageBox.warning(self, "Error: Frame Mismatch", f"Frames in video {cam_idx+1} ({num_frame}) and prediction ({pred_frame_count}) do not match!")

        except Exception as e:
            print(f"Error loading H5 file {h5_files[0]}: {e}")
            return
            
    def load_dlc_config(self):
        file_dialog = QtWidgets.QFileDialog(self)
        dlc_config, _ = file_dialog.getOpenFileName(self, "Load DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        if dlc_config:
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
            # Initialize data dict
            for cam_idx in range(self.num_cam):
                self.keypoint_coords[cam_idx] = {} # Initialize for cam_idx
                for inst in range(self.instance_count):
                    self.keypoint_coords[cam_idx][inst] = {} # Initialize for inst
                    self.keypoint_coords_3d[inst] = {}
                    for keypoint in self.keypoints:
                        self.keypoint_coords[cam_idx][inst][keypoint] = {}
                        self.keypoint_coords_3d[inst][keypoint] = {}

    def load_calibrations(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Calibration File", "", "Calibration Files (*.mat)")
        if file_path:
            calibration_file = file_path
            print(f"Calibration loaded: {calibration_file}")
            QMessageBox.information(self, "Success", "Calibrations loaded successfully!")
            calib = sio.loadmat(calibration_file)
            num_cam_from_calib = calib["params"].size
            for i in range(num_cam_from_calib):
                self.load_calibration_mat(i, calib)
            self.plot_camera_geometry()

    def load_calibration_mat(self, cam_idx, calib):
        self.camera_params[cam_idx]["RDistort"] = calib["params"][cam_idx,0][0,0]["RDistort"][0]
        self.camera_params[cam_idx]["TDistort"] = calib["params"][cam_idx,0][0,0]["TDistort"][0]
        K = calib["params"][cam_idx,0][0,0]["K"].T
        r = calib["params"][cam_idx,0][0,0]["r"].T
        t = calib["params"][cam_idx,0][0,0]["t"].flatten()
        self.cam_pos[cam_idx] = -np.dot(r.T, t)
        self.cam_dir[cam_idx] = r[:, 2]
        self.camera_params[cam_idx]["K"] = K
        self.camera_params[cam_idx]["P"] = self.get_projection_matrix(K,r,t)

    ###################################################################################################################################################

    def display_current_frame(self):
        for i, cap in enumerate(self.cap_list):
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = cap.read()
                if ret:
                    if self.pred_data_list[i] is None:
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
        self.data_loader_for_plot(cam_idx)
        # Iterate over each individual (animal)
        for inst in range(self.instance_count):
            color = self.instance_color[inst % len(self.instance_color)]
            keypoint_coords = dict()
            for keypoint in self.keypoints:
                kp = self.keypoint_coords[cam_idx][inst][keypoint]
                if kp is None:
                    continue
                x, y = kp[0], kp[1]
                keypoint_coords[keypoint] = (int(x),int(y))
                cv2.circle(frame, (int(x), int(y)), 3, color, -1) # Draw the dot representing the keypoints

            if self.individuals is not None and len(self.keypoint_coords[cam_idx][inst]) >= 2: # Only plot bounding box with more than one points
                self.plot_bounding_box(keypoint_coords, frame, color, inst)
            if self.skeleton:
                self.plot_2d_skeleton(keypoint_coords, frame, color)

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
            start_coord = keypoint_coords.get(start_kp)
            end_coord = keypoint_coords.get(end_kp)
            if start_coord and end_coord:
                cv2.line(frame, start_coord, end_coord, color, 2)
        return frame
    
    def plot_3d_points(self):
        self.data_loader_for_3d_plot()
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
            for keypoint_name, point_3d in self.keypoint_coords_3d[inst].items():
                if point_3d is not None:
                    self.ax.scatter(point_3d[0], point_3d[1], point_3d[2], color=np.array(color)/255, s=50)

            # Plot 3D skeleton
            if self.skeleton:
                for start_kp, end_kp in self.skeleton:
                    start_point = self.keypoint_coords_3d[inst].get(start_kp)
                    end_point = self.keypoint_coords_3d[inst].get(end_kp)
                    if start_point is not None and end_point is not None:
                        self.ax.plot([start_point[0], end_point[0]],
                                     [start_point[1], end_point[1]],
                                     [start_point[2], end_point[2]],
                                     color=np.array(color)/255)
        self.canvas.draw_idle() # Redraw the 3D canvas

    def plot_camera_geometry(self):
        """Plots the relative geometry on a given Axes3D object."""
        for i in range(self.num_cam):
            self.ax.scatter(*self.cam_pos[i], s=100, label=f"Camera {i+1} Pos")
            self.ax.quiver(*self.cam_pos[i], *self.cam_dir[i], length=100, color='blue', normalize=True)
        self.canvas.draw_idle() 

    ###################################################################################################################################################

    def set_selected_camera(self, cam_idx):
        self.selected_cam_idx = cam_idx
        print(f"Selected Camera Index: {self.selected_cam_idx}")
        self.display_current_frame() # Refresh display to update border

    def data_loader_for_plot(self, cam_idx):
        try:
            pred_data = self.pred_data_list[cam_idx]
            current_frame_data = pred_data[self.current_frame_idx][1] # 1 for extracting keypoints data from ( frame_idx, [ keypoint_1_x, keypoint_1_y, ... ])
        except IndexError:
            print(f"Frame index {self.current_frame_idx} of Camera {cam_idx} out of bounds for prediction data.")
            return
        
        num_keypoints_from_pred = current_frame_data.size // self.instance_count // 3  # Consider the confidence col
        if num_keypoints_from_pred != len(self.keypoints):
            QMessageBox.warning(self, "Error: Keypoint Mismatch", "Keypoints in config and in prediction do not match! Falling back to prediction parameters!")
            print(f"Keypoints in config: {len(self.keypoints)} \n Keypoints in prediction: {num_keypoints_from_pred}")
            return

        for inst in range(self.instance_count):
            for i in range(len(self.keypoints)):
                keypoint = self.keypoints[i]
                x = current_frame_data[inst * len(self.keypoints) * 3 + i * 3] # x, y, confidence triplet
                y = current_frame_data[inst * len(self.keypoints) * 3 + i * 3 + 1]
                confidence = current_frame_data[inst * len(self.keypoints) * 3 + i * 3 + 2]
                
                if pd.isna(confidence):
                    confidence = 0 # Set confidence value to
                    confidence = 0 # Set confidence value to 0 for NaN confidence
                if pd.isna(x) or pd.isna(y) or confidence <= self.confidence_cutoff: # Apply confidence cutoff
                    self.keypoint_coords[cam_idx][inst][keypoint] = None
                    continue
                else:
                    self.keypoint_coords[cam_idx][inst][keypoint] = (float(x),float(y),float(confidence))

    def data_loader_for_3d_plot(self, undistorted_images=False):
        self.keypoint_coords_3d = [{} for _ in range(self.instance_count)]
        # Determine how many instances each camera detects in the current frame
        if self.instance_count > 1: # This check is only relevant if multiple instances are expected
            instances_detected_per_camera = [0] * self.num_cam
            for cam_idx_check in range(self.num_cam):
                detected_instances_in_cam = 0
                for inst_check in range(self.instance_count):
                    # Check if this instance has any valid keypoint data for this camera in the current frame
                    has_valid_data = False
                    for keypoint_check in self.keypoints:
                        if self.keypoint_coords[cam_idx_check][inst_check][keypoint_check] is not None:
                            has_valid_data = True
                            break
                    if has_valid_data:
                        detected_instances_in_cam += 1
                instances_detected_per_camera[cam_idx_check] = detected_instances_in_cam

        for inst in range(self.instance_count):
            for keypoint in self.keypoints:
                valid_projection_matrices = []
                valid_points_2d = []
                valid_confidences = []
                valid_cam_view = 0

                for cam_idx in range(self.num_cam):
                    # If multiple instances are expected but this camera only detects one, skip its data for triangulation
                    if self.instance_count > 1 and instances_detected_per_camera[cam_idx] == 1:
                        continue

                    point_2d_data = self.keypoint_coords[cam_idx][inst][keypoint]
                    if point_2d_data is None:
                        continue

                    RDistort = self.camera_params[cam_idx]['RDistort']
                    TDistort = self.camera_params[cam_idx]['TDistort']
                    K = self.camera_params[cam_idx]['K']
                    P = self.camera_params[cam_idx]['P']

                    point_2d_no_confidence = (point_2d_data[0], point_2d_data[1]) # Remove confidence
                    confidence = point_2d_data[2]

                    if not undistorted_images:  # Undistort points
                        point_2d_undistorted = self.undistort_points(point_2d_no_confidence, K, RDistort, TDistort)
                        valid_points_2d.append(point_2d_undistorted)
                    else:
                        valid_points_2d.append(point_2d_no_confidence)

                    valid_projection_matrices.append(P)
                    valid_confidences.append(confidence)
                    valid_cam_view += 1

                if valid_cam_view >= 2:
                    self.triangulation_coords()

    def triangulation_coords(self, valid_cam_view, valid_projection_matrices, valid_points_2d, valid_confidences, inst, keypoint):
        """
        Implement a triangulation method that is both fast and take account of the confidence value of individual points
        Weighted Linear Triangulation (WLT)
        Construct the A matrix for Ax = 0, where x is the 3D point (X, Y, Z, 1)
        For each camera i, and its projection matrix P_i = [p_i1 p_i2 p_i3 p_i4]
        u_i * p_i3 - p_i1 = 0
        v_i * p_i3 - p_i2 = 0
        """
        A = []
        for i in range(valid_cam_view):
            P_i = valid_projection_matrices[i]
            u, v = valid_points_2d[i]
            w = valid_confidences[i] # Weight by confidence

            # Ensure P_i is a numpy array for slicing
            P_i = np.array(P_i)

            # Equations for DLT:
            # u * P_i[2,:] - P_i[0,:] = 0
            # v * P_i[2,:] - P_i[1,:] = 0
            # Apply weight 'w' to each row
            A.append(w * (u * P_i[2,:] - P_i[0,:]))
            A.append(w * (v * P_i[2,:] - P_i[1,:]))

        A = np.array(A) # Solve Ax = 0 using SVD
        U, S, Vt = np.linalg.svd(A) # The 3D point is the last column of V (or last row of Vt)
        
        point_4d_hom = Vt[-1]
        # Convert from homogeneous to Euclidean coordinates
        point_3d = point_4d_hom[:3] / point_4d_hom[3]
        # If valid_cam_view < 2, point_3d remains None, which is handled by the initialization.
        
        # Convert from homogeneous to Euclidean coordinates (x/w, y/w, z/w)
        point_3d = (point_4d_hom / point_4d_hom[3]).flatten()[:3]

        self.keypoint_coords_3d[inst][keypoint] = point_3d

    def get_projection_matrix(self, K, R, t):
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

    def undistort_points(self, points, K, RDistort, TDistort):
        dist_coeffs = np.array([RDistort[0], RDistort[1], TDistort[0], TDistort[1], 0])
        points_array = np.array(points)
        points_array = points_array.reshape(-1, 1, 2).astype(np.float32) # Reshape points for OpenCV: (N, 1, 2)
        undistorted_pts = cv2.undistortPoints(points_array, K, dist_coeffs, P=K)
        return tuple(undistorted_pts.flatten()) # Reshape back to (N, 2) and convert to tuple

    ###################################################################################################################################################

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
        for cap in self.cap_list:
            if cap and cap.isOpened():
                cap.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = DLC_3D_plotter()
    main_window.show()
    app.exec()
