import os
import tempfile
import yaml
import subprocess

import numpy as np
import cv2

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QGuiApplication
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QHBoxLayout

from typing import List, Literal

from ui import Clickable_Video_Label, Video_Slider_Widget, Progress_Indicator_Dialog
from core.dataclass import Loaded_DLC_Data, Export_Settings
from core.io import (
    Exporter, Prediction_Loader, Frame_Extractor,
    save_prediction_to_existing_h5, determine_save_path, save_predictions_to_new_h5,
)
from core.tool import Prediction_Plotter
from utils.helper import (
    log_print, handle_unsaved_changes_on_close, crop_coords_to_array, infer_head_tail_indices, build_angle_map)
from utils.pose import calculate_pose_centroids, calculate_canonical_pose
from utils.track import Hungarian, Track_Fixer

DEBUG = False

class DLC_Inference(QtWidgets.QDialog):
    prediction_saved = Signal(str)
    frames_exported = Signal(tuple)
    crop_coords_requested = Signal(list)

    def __init__(
        self,
        dlc_data:Loaded_DLC_Data,
        frame_list:List[int],
        video_filepath:str,
        parent=None
        ):
        """
        Initializes the DLC inference dialog for re-running predictions on selected frames 
        using a trained DeepLabCut model.

        Args:
            dlc_data (Loaded_DLC_Data): Object containing DLC project configuration, 
                prediction data, and metadata.
            frame_list (List[int]): List of frame indices to inference.
            video_filepath (str): Path to the source video file, used for frame extraction.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Run Predictions in DLC")
        self.dlc_data = dlc_data
        self.frame_list = frame_list
        self.video_filepath = video_filepath
        self.setFixedWidth(600)

        self.frame_list.sort()
        self.is_saved = True
        self.auto_cropping = False
        self.crop_coords = None

        video_name = os.path.basename(self.video_filepath).split(".")[0]
        self.temp_directory = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_directory.name
        self.export_set = Export_Settings(
            video_filepath=self.video_filepath, video_name=video_name, save_path=self.temp_dir, export_mode="Append")
        self.extractor = Frame_Extractor(self.export_set.video_filepath)
        
        if self.dlc_data.pred_data_array is None:
            self.fresh_pred = True
        else:
            self.fresh_pred = False

        layout = QVBoxLayout(self)
        self.setup_container = self._build_setup_container()
        if self.setup_container is None:
            return
        layout.addWidget(self.setup_container)
        
        self.wait_container = self._build_wait_container()
        layout.addWidget(self.wait_container)

        if not self.fresh_pred:
            self.video_container = self._build_video_container()
            if self.setup_container is None:
                return
            layout.addWidget(self.video_container)
            self.video_container.setVisible(False)

        self.wait_container.setVisible(False)

        self.setLayout(layout)

    #######################################################################################################################

    def _build_setup_container(self):
        """Container to wrap up all the setup process UI"""
        iteration_idx, iteration_folder = self._check_iteration_integrity()
        if iteration_folder is None:
            self.emergency_exit(f"Iteration {iteration_idx} folder not found.")
            return
        
        self.iteration_folder = iteration_folder

        available_shuffles = self._check_available_shuffles()
        if available_shuffles is None:
            self.emergency_exit("No shuffles found in iteration folder.")
            return
        
        available_shuffles.sort()
        self.shuffle_idx = int(available_shuffles[-1])
        shuffle_metadata_text, model_status = self._check_shuffle_metadata()

        setup_container = QtWidgets.QWidget()
        container_layout = QVBoxLayout(setup_container)

        shuffle_frame = QHBoxLayout()
        shuffle_label = QtWidgets.QLabel(f"Shuffle: ")
        self.shuffle_spinbox = QtWidgets.QSpinBox()
        self.shuffle_spinbox.setRange(0, self.shuffle_idx+2)
        self.shuffle_spinbox.setValue(self.shuffle_idx)
        self.shuffle_spinbox.valueChanged.connect(self._shuffle_spinbox_changed)

        self.shuffle_config_label = QtWidgets.QLabel(f"{shuffle_metadata_text}")
        if not model_status:
            self.shuffle_config_label.setStyleSheet("color: red;")

        shuffle_frame.addWidget(shuffle_label)
        shuffle_frame.addWidget(self.shuffle_spinbox)
        shuffle_frame.addWidget(self.shuffle_config_label)
        container_layout.addLayout(shuffle_frame)

        individual_frame = QHBoxLayout()
        max_individual_label = QtWidgets.QLabel(f"Number of animals in marked frames: ")
        self.max_individual_val = len(self.dlc_data.individuals)
        self.max_individual_spinbox = QtWidgets.QSpinBox()
        self.max_individual_spinbox.setRange(1, 20)
        self.max_individual_spinbox.setValue(self.max_individual_val)
        self.max_individual_spinbox.valueChanged.connect(self._individual_spinbox_changed)

        individual_frame.addWidget(max_individual_label)
        individual_frame.addWidget(self.max_individual_spinbox)
        container_layout.addLayout(individual_frame)

        self.auto_cropping_checkbox = QtWidgets.QCheckBox("Auto Crop")
        self.auto_cropping_checkbox.setChecked(self.auto_cropping)
        self.auto_cropping_checkbox.toggled.connect(self._auto_cropping_changed)

        button_frame = QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Extract Frames and Rerun Predictions in DLC")
        self.start_button.clicked.connect(self.inference_workflow)
        config_button = QtWidgets.QPushButton("Edit Pytorch Config")
        config_button.clicked.connect(self.show_pytorch_config_menu)
        button_frame.addWidget(self.auto_cropping_checkbox)
        button_frame.addWidget(config_button)
        button_frame.addWidget(self.start_button)
        container_layout.addLayout(button_frame)

        return setup_container

    def show_pytorch_config_menu(self):
        shuffle_folder = self._get_shuffle_folder()
        pytorch_yaml_path = os.path.join(shuffle_folder, "train", "pytorch_config.yaml")
        
        if os.path.exists(pytorch_yaml_path):
            if os.name == 'nt':
                subprocess.run(['explorer', '/select,', pytorch_yaml_path.replace('/', '\\')])
            elif sys.platform == 'darwin':
                subprocess.run(['open', '-R', pytorch_yaml_path])
            else:
                subprocess.run(['xdg-open', os.path.dirname(pytorch_yaml_path)])

    def _check_iteration_integrity(self):
        dlc_config_filepath = self.dlc_data.dlc_config_filepath
        dlc_folder = os.path.dirname(self.dlc_data.dlc_config_filepath)
        with open(dlc_config_filepath, 'r') as dcf:
            dlc_config = yaml.load(dcf, Loader=yaml.SafeLoader)
        iteration_idx = dlc_config["iteration"]
        iteration_folder = os.path.join(dlc_folder, "dlc-models-pytorch", f"iteration-{iteration_idx}")
        if not os.path.isdir(iteration_folder):
            return iteration_idx, None
        
        return iteration_idx, iteration_folder

    def _check_available_shuffles(self):
        available_shuffles = [f.split("shuffle")[1] for f in os.listdir(self.iteration_folder) if "shuffle" in f]
        if not available_shuffles:
            return None

        return available_shuffles
    
    def _check_shuffle_metadata(self):
        available_detector_models = []
        available_models = []

        shuffle_folder = self._get_shuffle_folder()
        if not shuffle_folder:
            return f"This shuffle does not exist in {self.iteration_folder}!", False

        shuffle_train_folder = os.path.join(shuffle_folder, "train")
        if not os.path.isdir(shuffle_train_folder):
            return "This shuffle does not seem to have trained models!", False

        for file in os.listdir(shuffle_train_folder):
            if not file.endswith(".pt"):
                continue

            if "detector" in file:
                available_detector_models.append(file)
                continue

            if "snapshot" in file:
                available_models.append(file)

        shuffle_config_filepath = os.path.join(shuffle_train_folder, "pytorch_config.yaml")
        with open(shuffle_config_filepath, 'r') as scf:
            shuffle_config = yaml.load(scf, Loader=yaml.SafeLoader)

        method = shuffle_config["method"]
        model_name = shuffle_config["model"]["backbone"]["model_name"]

        config_text = f"Model Name: {model_name}"
        
        if not available_models:
            return "This shuffle does not seem to have trained models!", False

        if method == "td":
            if not available_detector_models:
                return "This shuffle is using top-down method yet has no detector models!", False
            
            dectector_type = shuffle_config["detector"]["model"]["variant"]
            config_text += f" | Detector Type: {dectector_type}"

        return config_text, True

    def _get_shuffle_folder(self):
        shuffle_folder = None
        for f in os.listdir(self.iteration_folder):
            fullpath = os.path.join(self.iteration_folder, f)
            if f"shuffle{self.shuffle_idx}" in f and os.path.isdir(fullpath):
                shuffle_folder = fullpath
        return shuffle_folder

    def _shuffle_spinbox_changed(self, value):
        self.shuffle_idx = value
        text, status = self._check_shuffle_metadata()
        self.shuffle_config_label.setText(text)
        if not status:
            self.shuffle_config_label.setStyleSheet("color: red;")
            self.start_button.setEnabled(False)
        else:
            self.shuffle_config_label.setStyleSheet("color: black;")
            self.start_button.setEnabled(True)

    def _individual_spinbox_changed(self, value):
        self.max_individual_val = value

    def _auto_cropping_changed(self, checked:bool):
        self.auto_cropping = checked

    #######################################################################################################################

    def _build_video_container(self):
        """Container to display prediction plots, left pane -> old, right pane -> new"""
        self.total_marked_frames = len(self.frame_list)
        self.total_frames = self.dlc_data.pred_data_array.shape[0]

        self.backup_data_array = self.dlc_data.pred_data_array.copy()
        self.frame_status = ["Unprocessed"] * self.total_marked_frames
        self.unprocessed_list = list(range(self.total_marked_frames))
        self.approved_list, self.rejected_list = [], []
        self.current_frame_idx = 0

        video_container = QtWidgets.QWidget()
        container_layout = QVBoxLayout(video_container)

        # Frame info display
        frame_info_layout = QHBoxLayout()
        self.global_frame_label = QtWidgets.QLabel(f"Global: {self.frame_list[0]} / {self.total_frames-1}")
        self.selected_frame_label = QtWidgets.QLabel(f"Selected: 0 / {self.total_marked_frames-1}")

        font = QtGui.QFont()
        font.setBold(True)
        self.global_frame_label.setFont(font)
        self.selected_frame_label.setFont(font)
        self.global_frame_label.setStyleSheet("color: black;")
        self.selected_frame_label.setStyleSheet("color: #1E90FF;")
        frame_info_layout.addStretch()
        frame_info_layout.addWidget(self.global_frame_label)
        frame_info_layout.addWidget(self.selected_frame_label)
        frame_info_layout.addStretch()
        
        container_layout.addLayout(frame_info_layout)

        # Label displayed just above video
        video_label_layout = QHBoxLayout()

        old_label = QtWidgets.QLabel("Old Predictions")
        old_label.setAlignment(Qt.AlignCenter)
        old_label.setFont(font)
        old_label.setStyleSheet("color: #4B4B4B; background: #F0F0F0; padding: 6px; border-radius: 4px;")
        old_label.setFixedHeight(30)

        new_label = QtWidgets.QLabel("New Predictions")
        new_label.setAlignment(Qt.AlignCenter)
        new_label.setFont(font)
        new_label.setStyleSheet("color: #4B4B4B; background: #F0F0F0; padding: 6px; border-radius: 4px;")
        new_label.setFixedHeight(30)

        video_label_layout.addWidget(old_label)
        video_label_layout.addWidget(new_label)

        container_layout.addLayout(video_label_layout)

        # Video display layout
        self.video_layout = QHBoxLayout()
        self.selected_cam = None
        self.video_labels = []
        for col in range(2):
            label = Clickable_Video_Label(col, self) # Use the custom label
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(720, 540)
            label.setStyleSheet("border: 1px solid gray;")
            self.video_layout.addWidget(label, col)
            self.video_labels.append(label)
            label.clicked.connect(self._set_selected_camera)

        container_layout.addLayout(self.video_layout)

        self.progress_widget= Video_Slider_Widget()
        self.progress_widget.set_total_frames(self.total_marked_frames)
        self.progress_widget.set_current_frame(0)
        self.progress_widget.set_frame_category("Unprocessed", self.unprocessed_list)
        self.progress_widget.frame_changed.connect(self._handle_frame_change_from_comp)
        container_layout.addWidget(self.progress_widget)

        # Approval controls
        approval_box = QtWidgets.QGroupBox("Frame Approval")
        approval_layout = QHBoxLayout()

        self.reject_button = QtWidgets.QPushButton("Reject (R)")
        self.approve_button = QtWidgets.QPushButton("Approve (A)")
        self.approve_all_button = QtWidgets.QPushButton("Approve All (Ctrl + A)")
        self.apply_button = QtWidgets.QPushButton("Apply Changes (Ctrl + S)")
        self.approve_all_button.setToolTip("Mark all remaining unprocessed frames as Approved")
        self.apply_button.setEnabled(False)

        self.reject_button.clicked.connect(lambda: self.mark_frame_status("Rejected"))
        self.approve_button.clicked.connect(lambda: self.mark_frame_status("Approved"))
        self.approve_all_button.clicked.connect(self.approve_all_remaining_frames)
        self.apply_button.clicked.connect(self.save_prediction)

        approval_layout.addWidget(self.reject_button)
        approval_layout.addWidget(self.approve_button)
        approval_layout.addWidget(self.approve_all_button)
        approval_layout.addWidget(self.apply_button)
        approval_box.setLayout(approval_layout)
        container_layout.addWidget(approval_box)

        self._build_shortcut()

        return video_container

    def _build_shortcut(self):
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.progress_widget.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_A), self).activated.connect(lambda: self.mark_frame_status("Approved"))
        QShortcut(QKeySequence(Qt.Key_R), self).activated.connect(lambda: self.mark_frame_status("Rejected"))
        QShortcut(QKeySequence(Qt.CTRL | Qt.Key_A), self).activated.connect(self.approve_all_remaining_frames)
        QShortcut(QKeySequence(Qt.CTRL | Qt.Key_S), self).activated.connect(self.save_prediction)
    
    def _display_current_frame(self):
        global_frame_idx = self.frame_list[self.current_frame_idx]
        image_filename = f"img{global_frame_idx:08d}.png"
        image_path = os.path.join(self.temp_dir, image_filename)
        if self.crop_coords is None:
            frame = cv2.imread(image_path)
        else:
            frame = self.extractor.get_frame(global_frame_idx)

        for i in range(len(self.video_labels)):
            frame_view = frame.copy()
            if frame_view is None:
                self.video_labels[i].setText(f"Image {image_filename} Not Loaded/Available")
                self.video_labels[i].setPixmap(QtGui.QPixmap())
                continue

            # Determine which prediction data array to use
            pred_data_to_use = None
            if i == 0 and self.dlc_data.pred_data_array is not None:
                pred_data_to_use = self.dlc_data.pred_data_array
            elif i == 1 and self.new_data_array is not None:
                pred_data_to_use = self.new_data_array

            current_frame_data = pred_data_to_use[global_frame_idx, :, :]

            if not hasattr(self, "plotter"):
                self.plotter = Prediction_Plotter(dlc_data=self.dlc_data, current_frame_data=current_frame_data, frame_cv2=frame)

            self.plotter.current_frame_data = current_frame_data
            self.plotter.frame_cv2 = frame_view

            if current_frame_data is not None and not np.all(np.isnan(current_frame_data)):
                frame_view = self.plotter.plot_predictions()

            target_width = self.video_labels[i].width() # Get the target size from the QLabel
            target_height = self.video_labels[i].height()

            # Resize the frame to the target size
            resized_frame = cv2.resize(frame_view, (target_width, target_height), interpolation=cv2.INTER_AREA)

            rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.video_labels[i].setPixmap(pixmap)
            self.video_labels[i].setText("")
            
        self.progress_widget.set_current_frame(self.current_frame_idx) # Update slider handle's position
        self._update_button_states()
    
    def change_frame(self, delta):
        self.selected_cam = None # Clear the selected cam upon frame switch
        new_frame_idx = self.current_frame_idx + delta
        if 0 <= new_frame_idx < self.total_marked_frames:
            self.current_frame_idx = new_frame_idx
            self._display_current_frame()
            self._navigation_title_controller()
            self._update_button_states()

    def mark_frame_status(self, status:Literal["Rejected","Approved"]):
        if self.current_frame_idx < 0 or self.current_frame_idx >= len(self.frame_status):
            return

        self.frame_status[self.current_frame_idx] = status
        self.apply_button.setEnabled("Approved" in self.frame_status)
        global_frame_idx = self.frame_list[self.current_frame_idx]
        
        pred_data_to_use = self.new_data_array if status == "Approved" else self.backup_data_array
        self.dlc_data.pred_data_array[global_frame_idx, :, :] = pred_data_to_use[global_frame_idx, :, :]

        self._display_current_frame()
        self._update_button_states()
        self._refresh_lists()

    def approve_all_remaining_frames(self):
        for i in range(len(self.frame_status)):
            if self.frame_status[i] == "Unprocessed":
                self.frame_status[i] = "Approved"
                self.dlc_data.pred_data_array[self.frame_list[i], :, :] = self.new_data_array[self.frame_list[i], :, :]

        self.apply_button.setEnabled(True)
        self._display_current_frame()
        self._update_button_states()
        self._refresh_lists()

    def save_prediction(self):
        self._refresh_lists()

        if self.unprocessed_list:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Unprocessed Frames - Save and Exit?")
            msg_box.setText(
                f"You have {len(self.unprocessed_list)} unprocessed frame(s).\n\n"
                "This action will save your progress and close the window.\n"
                "All unprocessed frames will remain as original and will not be marked as approved.\n\n"
                "Are you sure you want to continue?"
            )
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            reply = msg_box.exec_()

            if reply == QMessageBox.No:
                return False 

        pred_file_to_save_path = determine_save_path(self.dlc_data.prediction_filepath, suffix="_rerun_")
        status, msg = save_prediction_to_existing_h5(pred_file_to_save_path, self.dlc_data.pred_data_array)
        
        if not status:
            QMessageBox.critical(self, "Saving Error", f"An error occurred during saving: {msg}")
            print(f"An error occurred during saving: {msg}")
            return False

        self.is_saved = True
        approve_list_global = [self.frame_list[idx] for idx in self.approved_list]
        rejected_list_global = [self.frame_list[idx] for idx in self.rejected_list]
        list_tuple = (approve_list_global, rejected_list_global)
        self.prediction_saved.emit(pred_file_to_save_path)
        self.frames_exported.emit(list_tuple)

        return True

    def _navigation_title_controller(self):
        global_frame_idx = self.frame_list[self.current_frame_idx]
        self.global_frame_label.setText(f"Global: {global_frame_idx} / {self.total_frames}")
        self.selected_frame_label.setText(f"Selected: {self.current_frame_idx} / {self.total_marked_frames}")
        
    def _set_selected_camera(self, cam_idx):
        self.selected_cam = cam_idx
        for i in range(2):
            if i == self.selected_cam:
                self.video_labels[i].setStyleSheet("border: 2px solid red;")
            else:
                self.video_labels[i].setStyleSheet("border: 1px solid gray;")

    def _handle_frame_change_from_comp(self, new_frame_idx:int):
        self.current_frame_idx = new_frame_idx
        self._display_current_frame()
        self._navigation_title_controller()
        self._update_button_states()

    def _refresh_lists(self):
        self.is_saved = False
        self.approved_list, self.rejected_list, self.unprocessed_list = [], [], []
        for frame_idx in range(len(self.frame_list)):
            if self.frame_status[frame_idx] == "Approved":
                self.approved_list.append(frame_idx)
            elif self.frame_status[frame_idx] == "Rejected":
                self.rejected_list.append(frame_idx)
            else:
                self.unprocessed_list.append(frame_idx)

        self.progress_widget.set_frame_category("Approved", self.approved_list, color="#0066cc", priority=6)
        self.progress_widget.set_frame_category("Rejected", self.rejected_list, color="#F749C6", priority=6)
        self.progress_widget.set_frame_category("Unprocessed", self.unprocessed_list)
        self.progress_widget.commit_categories()

    #######################################################################################################################

    def _build_wait_container(self):
        wait_container = QtWidgets.QWidget()
        wait_layout = QHBoxLayout(wait_container)
        wait_layout.setContentsMargins(10, 10, 10, 10)
        wait_layout.setSpacing(10)

        icon_label = QtWidgets.QLabel()
        icon = QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation)
        pixmap = icon.pixmap(48, 48)
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignTop)
        icon_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        wait_layout.addWidget(icon_label)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        title_label = QtWidgets.QLabel("<b>DeepLabCut Analysis Started</b>")
        title_label.setWordWrap(True)

        info_label = QtWidgets.QLabel(
            "This action usually takes between a few seconds and one minute, "
            "depending on number of marked frames unless mode is inferencing all. "
            "Check the terminal to see the progress."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: gray; }")

        text_layout.addWidget(title_label)
        text_layout.addWidget(info_label)
        text_layout.addStretch()
        wait_layout.addLayout(text_layout)

        return wait_container

    def _update_button_states(self):
        current_status = self.frame_status[self.current_frame_idx]
        self.approve_button.setEnabled(current_status != "Approved")
        self.reject_button.setEnabled(current_status != "Rejected")
        self.approve_all_button.setEnabled(any(s == "Unprocessed" for s in self.frame_status))

    #######################################################################################################################

    def inference_workflow(self):
        if self.auto_cropping and self.crop_coords is None:
            self.crop_coords_requested.emit(self.frame_list)
            return

        extract_success = self._extract_marked_frame_images(self.crop_coords)

        self.setup_container.setVisible(False)
        self.wait_container.setVisible(True)
        self.center()
        QtWidgets.QApplication.processEvents()

        analyze_success = self._analyze_frame_images()
        if not extract_success or not analyze_success:
            QMessageBox(self, "Error", 
                "Error during frame image extraction and analysis, check terminal for detail.")
            self.emergency_exit()
        
        if self.fresh_pred:
            temp_pred_filename = self._load_and_remap_new_prediction()
            video_path = os.path.dirname(self.video_filepath)
            pred_filename = temp_pred_filename.replace("image_predictions_", self.export_set.video_name)
            pred_filepath = os.path.join(video_path, pred_filename)
            self.dlc_data.prediction_filepath = pred_filepath # So that it will be picked up by prediction_to_csv later
            self.export_set.save_path = video_path
            status, msg = save_predictions_to_new_h5(
                dlc_data=self.dlc_data,
                pred_data_array=self.new_data_array,
                export_settings=self.export_set)
        
            if not status:
                QMessageBox.critical(self, "Saving Error", f"An error occurred during saving: {msg}")
                print(f"An error occurred during saving: {msg}")
                return
            
            print(f"Prediction saved to {pred_filepath}")
            
            list_tuple = (self.frame_list, [])
            self.prediction_saved.emit(pred_filepath)
            self.frames_exported.emit(list_tuple)
            self.accept()
            return

        self.wait_container.setVisible(False)
        self.setFixedWidth(1400)
        self.video_container.setVisible(True)
        self.center()

        self._load_and_remap_new_prediction()

        self._crossref_existing_pred()
        self._display_current_frame()

    def _extract_marked_frame_images(self, crop_coords=None) -> bool:
        try:
            progress = Progress_Indicator_Dialog(0, 100, "Frame Extraction", "Extracting frames from video", parent=self)
            exporter = Exporter(
                dlc_data=self.dlc_data, export_settings=self.export_set, frame_list=self.frame_list, progress_callback=progress, crop_coords=crop_coords)
            exporter.export_data_to_DLC(frame_only=True)
            return True
        except Exception as e:
            print(f"Failed to extracted frame images. Exception: {e}.")
            return False

    def _analyze_frame_images(self) -> bool:
        try:
            import deeplabcut
            deeplabcut.analyze_images(
                config=self.dlc_data.dlc_config_filepath,
                images=self.temp_dir,
                shuffle=self.shuffle_idx,
                max_individuals=self.max_individual_val)
            return True
        except Exception as e:
            print(f"Failed to analyze extracted frame images using deeplabcut. Exception: {e}.")
            return False

    def _load_and_remap_new_prediction(self) -> str:
        h5_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".h5")]
        pred_filepath = os.path.join(self.temp_dir, h5_files[-1])
        dlc_config_filepath = self.dlc_data.dlc_config_filepath
        loader = Prediction_Loader(dlc_config_filepath, pred_filepath)
        loaded_data = loader.load_data()
        new_data_array = np.full(
            (self.dlc_data.pred_frame_count, self.dlc_data.instance_count, self.dlc_data.num_keypoint*3), np.nan)
        temp_data_array = loaded_data.pred_data_array

        if self.crop_coords is not None:
            coords_array = crop_coords_to_array(self.crop_coords, temp_data_array.shape, self.frame_list)
            temp_data_array = temp_data_array + coords_array

        temp_data_array = self._correct_new_pred(temp_data_array)

        new_data_array[self.frame_list, :, :] = temp_data_array
        self.new_data_array = new_data_array
        return h5_files[-1]

    def _correct_new_pred(self, temp_data_array:np.ndarray) -> np.ndarray:
        head_idx, tail_idx = infer_head_tail_indices(self.dlc_data.keypoints)
        if head_idx is None or tail_idx is None:
            canon_pose, angle_map_data = None, None
        else:
            canon_pose, all_frame_pose = calculate_canonical_pose(temp_data_array, head_idx, tail_idx)
            angle_map_data = build_angle_map(canon_pose, all_frame_pose, head_idx, tail_idx)

        dialog = "Fixing track using temporal consistency..."
        title = f"Fix Track Using Temporal"
        progress = Progress_Indicator_Dialog(0, temp_data_array.shape[0], title, dialog, self)
        tf = Track_Fixer(temp_data_array, canon_pose, angle_map_data, progress)
        temp_data_array, _ = tf.track_correction()
        return temp_data_array

    def _crossref_existing_pred(self):
        for frame_idx in self.frame_list:
            pred_centroids, _ = calculate_pose_centroids(self.new_data_array, frame_idx)
            ref_centroids, _ = calculate_pose_centroids(self.dlc_data.pred_data_array, frame_idx)

            valid_pred_mask = np.all(~np.isnan(pred_centroids), axis=1)
            valid_ref_mask = np.all(~np.isnan(ref_centroids), axis=1)

            log_print(f"------ Processing Frame {frame_idx} ------", enabled=DEBUG)

            hun = Hungarian(pred_centroids, ref_centroids, valid_pred_mask, valid_ref_mask, debug_print=DEBUG)
            corrected_order = hun.hungarian_matching()
            
            if corrected_order:
                self.new_data_array[frame_idx, :, :] = self.new_data_array[frame_idx, corrected_order, :]

    #######################################################################################################################

    def center(self):
        screen = QGuiApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def emergency_exit(self, reason:str="Check terminal for reason."):
        QMessageBox.warning(self, "Rerun Failed", reason)
        self.reject()

    def closeEvent(self, event):
        handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_prediction)
        if self.temp_directory is not None:
            self.temp_directory.cleanup()