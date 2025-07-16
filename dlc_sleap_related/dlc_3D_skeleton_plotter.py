import os
import glob

import h5py
import yaml
import scipy.io as sio

import cv2
import pandas as pd
import numpy as np

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QPushButton

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

#######################################    W     #################      I     #########################    P   #############################################

class DLC_3D_plotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC 3D Plotter")
        self.setGeometry(100, 100, 1600, 960)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.load_video_folder_button = QPushButton("Load Videos & Predictions")
        self.load_dlc_config_button = QPushButton("Load DLC Configs")
        self.load_calibrations_button = QPushButton("Load Calibrations")

        self.button_layout.addWidget(self.load_video_folder_button)
        self.button_layout.addWidget(self.load_dlc_config_button)
        self.button_layout.addWidget(self.load_calibrations_button)
        self.layout.addLayout(self.button_layout)

        self.display_layout = QtWidgets.QHBoxLayout()
        self.video_layout = QtWidgets.QGridLayout()
        
        self.video_labels = [] # Store video labels in a list for easy access
        for row in range(2):
            for col in range(2):
                label = QtWidgets.QLabel(f"Video {row*2 + col + 1}")
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
        self.load_video_folder_button.clicked.connect(self.open_video_folder_dialog)
        self.load_dlc_config_button.clicked.connect(self.load_dlc_config)
        self.load_calibrations_button.clicked.connect(self.load_calibrations)

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
        self.keypoint_coords = [None] * self.num_cam
        
        self.camera_params = [{} for _ in range(self.num_cam)]

        self.confidence_cutoff = 0.6 # Initialize confidence cutoff

        self.multi_animal, self.keypoints, self.skeleton, self.individuals = False, None, None, None
        self.instance_count = 1

        self.num_keypoints = None
        self.cam_pos, self.cam_dir = [None] * self.num_cam, [None] * self.num_cam

        self.plot_lim = 300
        
        self.current_frame_idx = 0      # Single frame index for all synchronized videos
        self.total_frames = 0      # Max frames across all videos

    def open_video_folder_dialog(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder_path:
            self.load_video_folder(folder_path)

    def load_video_folder(self, folder_path):
        print(f"Loading videos from: {folder_path}")
        max_frames = 0
        for i in range(self.num_cam):  # Loop through expected camera folders
            folder = os.path.join(folder_path, f"Camera{i+1}")
            video_file = os.path.join(folder, "0.mp4")
            self.video_list.append(video_file)
            
            cap = cv2.VideoCapture(video_file) # Try to open the video capture
            if not cap.isOpened():
                print(f"Warning: Could not open video file: {video_file}")
                cap = None
            else:
                # Update max_frames based on the longest video
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_frames > max_frames:
                    max_frames = num_frames

            self.cap_list.append(cap) # Add the capture object (or None) to the list

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
            self.multi_animal = cfg["multianimalproject"]
            self.keypoints = cfg["bodyparts"] if not self.multi_animal else cfg["multianimalbodyparts"]
            self.skeleton = cfg["skeleton"]
            self.individuals = cfg["individuals"]
            self.instance_count = len(self.individuals) if self.individuals is not None else 1
            self.display_current_frame()

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
        self.cam_pos[cam_idx](-np.dot(r.T, t))
        self.cam_dir[cam_idx](r[2, :])
        self.camera_params[cam_idx]["K"] = K
        self.camera_params[cam_idx]["P"] = self.get_projection_matrix(K,r,t)

    ###################################################################################################################################################

    def display_current_frame(self):
        for i, cap in enumerate(self.cap_list):
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = self.plot_2d_points(frame, i) if self.pred_data_list[i] is not None else frame

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
        self.progress_slider.setValue(self.current_frame_idx) # Update the slider after all frames are displayed
        if self.camera_params[1] != None:  # Update the 3D plot after loading calibration
            self.plot_3d_points()

    def plot_2d_points(self, frame, cam_idx):
        try:
            pred_data = self.pred_data_list[cam_idx]
            current_frame_data = pred_data[self.current_frame_idx][1]
        except IndexError:
            print(f"Frame index {self.current_frame_idx} out of bounds for prediction data.")
            return frame
        colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)] # BGR
        if self.keypoints is None:
            self.num_keypoints = current_frame_data.size // self.instance_count // 3 # Consider the confidence col
        elif len(self.keypoints) != current_frame_data.size // self.instance_count // 3:
            QMessageBox.warning(self, "Error: Keypoint Mismatch", "Keypoints in config and in prediction do not match! Falling back to prediction parameters!")
            print(f"Keypoints in config: {len(self.keypoints)} \n Keypoints in prediction: {current_frame_data.size // self.instance_count * 2 // 3}")
            self.keypoints = None   # Falling back to prediction parameters
            self.skeleton = None
            self.num_keypoints = current_frame_data.size // self.instance_count // 3
        else:
            self.num_keypoints = len(self.keypoints)

        # Iterate over each individual (animal)
        for inst in range(self.instance_count):
            color = colors[inst % len(colors)]
            # Initiate an empty dict for storing coordinates
            for i in range(self.num_keypoints):
                x = current_frame_data[inst * self.num_keypoints * 3 + i * 3] # x, y, confidence triplet
                y = current_frame_data[inst * self.num_keypoints * 3 + i * 3 + 1]
                confidence = current_frame_data[inst * self.num_keypoints * 3 + i * 3 + 2]
                if pd.isna(confidence):
                    confidence = 0 # Set confidence value to 0 for NaN confidence
                keypoint = i if self.keypoints is None else self.keypoints[i]
                if pd.isna(x) or pd.isna(y) or confidence <= self.confidence_cutoff: # Apply confidence cutoff
                    self.keypoint_coords[cam_idx][inst][keypoint] = None
                    continue # Skip plotting empty coords
                else:
                    self.keypoint_coords[cam_idx][inst][keypoint] = (float(x),float(y),float(confidence))
                
                cv2.circle(frame, (int(x), int(y)), 3, color, -1) # Draw the dot representing the keypoints

            if self.individuals is not None and len(self.keypoint_coords[cam_idx][inst]) >= 2: # Only plot bounding box with more than one points
                self.plot_bounding_box(self.keypoint_coords[cam_idx][inst], frame, color, inst)
            if self.skeleton:
                self.plot_2d_skeleton(self.keypoint_coords[cam_idx][inst], frame, color)

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

            points_2d_per_instance = np.full((num_keypoints, len(self.cap_list), 2), np.nan, dtype=np.float32)
            
            for cam_idx, pred_data in enumerate(self.pred_data_list):
                if pred_data is not None and self.current_frame_idx < pred_data.shape[0]:
                    current_frame_data_cam = pred_data[self.current_frame_idx][1]
                    
                    for i in range(num_keypoints):
                        # Ensure we don't go out of bounds for current_frame_data_cam
                        data_idx_start = inst * num_keypoints * 3 + i * 3
                        if data_idx_start + 2 < current_frame_data_cam.size:
                            x = current_frame_data_cam[data_idx_start]
                            y = current_frame_data_cam[data_idx_start + 1]
                            confidence = current_frame_data_cam[data_idx_start + 2]

                            if not pd.isna(x) and not pd.isna(y) and confidence > self.confidence_cutoff:
                                points_2d_per_instance[i, cam_idx, :] = [x, y]

            # Triangulate points for the current instance
            points_3d_instance = self.triangulate_points_3d(points_2d_per_instance)
            
            # Plot the 3D skeleton for the current instance
            # plot_3d_skeleton() to be implemented

        # Prepare skeleton for 3D plotting
        skeleton_3d_plot_format = {'joints_idx': [], 'joint_names': self.keypoints, 'color': [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]}
        if self.skeleton and self.keypoints:
            for start_kp, end_kp in self.skeleton:
                try:
                    start_idx = self.keypoints.index(start_kp)
                    end_idx = self.keypoints.index(end_kp)
                    skeleton_3d_plot_format['joints_idx'].append((start_idx, end_idx))
                except ValueError:
                    print(f"Warning: Keypoint '{start_kp}' or '{end_kp}' not found in config. Skipping skeleton segment.")
        
        self.canvas.draw_idle() # Redraw the 3D 0canvas

    def plot_camera_geometry(self):
        """Plots the relative geometry on a given Axes3D object."""
        for i in range(self.num_cam):
            self.ax.scatter(*self.cam_pos[i], s=100, label=f"Camera {i+1} Pos")
            self.ax.quiver(*self.cam_pos[i], *self.cam_dir[i], length=100, color='blue', normalize=True)
        self.canvas.draw_idle() 

    ###################################################################################################################################################

    def triangulate_points_3d(self, point_2d_list, undistorted_images=False):

        for cam_idx, point_2d in enumerate(point_2d_list):
            RDistort = self.camera_params[cam_idx]['RDistort']
            TDistort = self.camera_params[cam_idx]['TDistort']
            K = self.camera_params[cam_idx]['K']
            P = self.camera_params[cam_idx]['P']
            
            # Undistort points if necessary
            if not undistorted_images:
                point_2d_undistorted = self.undistort_points(point_2d, K, RDistort, TDistort)
                valid_views_2d.append(point_2d_undistorted.flatten())
            else:
                valid_views_2d.append(pt_2d.flatten())
            
            projection_matrices.append(P)

        # Perform triangulation if at least two valid views are available
        if len(valid_views_2d) >= 2:
            # For OpenCV's triangulatePoints, we need exactly two views.
            # If more than two, we can pick the first two or implement a more robust
            # multi-view triangulation (e.g., RANSAC-based).
            # For simplicity, we'll use the first two valid views.
            
            # Reshape points to (2, N) for cv2.triangulatePoints
            pts1 = valid_views_2d[0].reshape(2, 1)
            pts2 = valid_views_2d[1].reshape(2, 1)
            
            P1 = projection_matrices[0]
            P2 = projection_matrices[1]
            
            # Triangulate points
            # DLT algorithm is used by default
            # Returns homogeneous coordinates (x, y, z, w)
            point_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
            
            # Convert from homogeneous to Euclidean coordinates (x/w, y/w, z/w)
            point_3d = (point_4d_hom / point_4d_hom[3]).flatten()[:3]
            points_3d[marker_idx, :] = point_3d

        return points_3d
    
    def get_projection_matrix(K, R, t):
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

    def undistort_points(points, K, RDistort, TDistort):
        dist_coeffs = np.array([RDistort[0], RDistort[1], TDistort[0], TDistort[1], 0])
        points = points.reshape(-1, 1, 2).astype(np.float32) # Reshape points for OpenCV: (N, 1, 2)
        undistorted_pts = cv2.undistortPoints(points, K, dist_coeffs, P=K)
        return undistorted_pts.reshape(-1, 2) # Reshape back to (N, 2)

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
            self.playback_timer.start(1000/100) # 100 fps
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
