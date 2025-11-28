import os
import shutil
import tempfile
import yaml
import numpy as np
import cv2

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QHBoxLayout, QPushButton

import traceback
from typing import List, Literal, Tuple, Optional

from .undo_redo import Uno_Stack
from ui import Clickable_Video_Label, Video_Slider_Widget, Progress_Indicator_Dialog, Shortcut_Manager
from core.dataclass import Loaded_DLC_Data, Export_Settings
from core.io import (
    Exporter, Prediction_Loader, Frame_Extractor,
    save_prediction_to_existing_h5, determine_save_path, save_predictions_to_new_h5,
)
from core.tool import Prediction_Plotter
from utils.helper import (
    log_print, handle_unsaved_changes_on_close, crop_coord_to_array, infer_head_tail_indices, build_angle_map, get_roi_cv2)
from utils.pose import calculate_pose_centroids, calculate_canonical_pose
from utils.track import Hungarian, Track_Fixer

DEBUG = False

class DLC_Inference(QtWidgets.QDialog):
    prediction_saved = Signal(str)
    frames_exported = Signal(tuple)
    roi_set = Signal(object)

    def __init__(
        self,
        dlc_data:Loaded_DLC_Data,
        frame_list:List[int],
        video_filepath:str,
        roi:Optional[np.ndarray]=None,
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

        try:
            x1, y1, x2, y2 = roi
            self.crop_coord = x1, y1, x2, y2
        except:
            self.crop_coord = None

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

        self.max_individual_val = len(self.dlc_data.individuals)
        self.max_individual_spinbox = Spinbox_With_Label(
            label_text = "Number of animals in marked frames: ",
            spinbox_range = (1,20),
            initial_val = self.max_individual_val,
            parent = self
        )
        self.max_individual_spinbox.value_changed.connect(self._individual_spinbox_changed)

        container_layout.addWidget(self.max_individual_spinbox)

        self.auto_cropping_checkbox = QtWidgets.QCheckBox("Crop")
        self.auto_cropping_checkbox.setChecked(self.auto_cropping)
        self.auto_cropping_checkbox.toggled.connect(self._auto_cropping_changed)

        button_frame = QHBoxLayout()

        self.batch_size_changed = False
        self.batchsize_label_spinbox = Spinbox_With_Label(
            label_text = "Batch Size: ", spinbox_range = (1,10000), initial_val = self.batch_size, parent = self
        )
        self.batchsize_label_spinbox.value_changed.connect(self._batch_size_spinbox_changed)

        self.detector_batchsize_label_spinbox = Spinbox_With_Label(
            label_text = "Detector Batch Size: ", spinbox_range = (1,10000), initial_val = self.detector_batch_size, parent = self
        )
        self.detector_batchsize_label_spinbox.value_changed.connect(self._det_batch_size_spinbox_changed)
        self._determine_det_spinbox_vis(shuffle_metadata_text)

        self.start_button = QPushButton("Run Inference")
        self.start_button.clicked.connect(self.inference_workflow)
        button_frame.addWidget(self.auto_cropping_checkbox)
        button_frame.addWidget(self.batchsize_label_spinbox)
        button_frame.addWidget(self.detector_batchsize_label_spinbox)
        button_frame.addWidget(self.start_button)
        container_layout.addLayout(button_frame)
 
        return setup_container

    def _check_iteration_integrity(self):
        dlc_config_filepath = self.dlc_data.dlc_config_filepath
        dlc_folder = os.path.dirname(self.dlc_data.dlc_config_filepath)
        with open(dlc_config_filepath, 'r') as dcf:
            dlc_config = yaml.load(dcf, Loader=yaml.SafeLoader)
        iteration_idx = dlc_config["iteration"]
        self.batch_size = dlc_config["batch_size"]
        self.detector_batch_size = dlc_config.get("detector_batch_size", 1)
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
            self._determine_det_spinbox_vis(text)

    def _individual_spinbox_changed(self, value):
        self.max_individual_val = value

    def _batch_size_spinbox_changed(self, value):
        self.batch_size = value
        self.batch_size_changed = True

    def _det_batch_size_spinbox_changed(self, value):
        self.detector_batch_size = value
        self.batch_size_changed = True

    def _auto_cropping_changed(self, checked:bool):
        self.auto_cropping = checked

    def _determine_det_spinbox_vis(self, text:str):
        self.detector_batchsize_label_spinbox.setVisible("Detector" in text)

    #######################################################################################################################

    def _build_video_container(self):
        self.total_frames = self.dlc_data.pred_data_array.shape[0]
        self.total_marked_frames = len(self.frame_list)

        self.backup_data_array = self.dlc_data.pred_data_array.copy()
        self.frame_status_array = np.zeros((self.total_marked_frames,))
        self.current_frame_idx = 0
        self.uno = Uno_Stack()
        self.uno.save_state_for_undo(self.frame_status_array)

        video_container = QtWidgets.QWidget()
        container_layout = QVBoxLayout(video_container)

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
        self._refresh_slider()
        self.progress_widget.frame_changed.connect(self._handle_frame_change_from_comp)
        container_layout.addWidget(self.progress_widget)

        # Approval controls
        approval_box = QtWidgets.QGroupBox("Frame Approval")
        approval_layout = QHBoxLayout()

        self.reject_button = QPushButton("Reject (R)")
        self.approve_button = QPushButton("Approve (A)")
        self.approve_all_button = QPushButton("Approve All")
        self.reject_all_button = QPushButton("Reject All")
        self.apply_button = QPushButton("Apply Changes (Ctrl + S)")
        self.approve_all_button.setToolTip("Mark all remaining unprocessed frames as Approved")
        self.apply_button.setEnabled(False)

        self.reject_button.clicked.connect(lambda: self.mark_frame_status("Rejected"))
        self.approve_button.clicked.connect(lambda: self.mark_frame_status("Approved"))
        self.approve_all_button.clicked.connect(self.approve_all_remaining_frames)
        self.reject_all_button.clicked.connect(self.reject_all_remaining_frames)
        self.apply_button.clicked.connect(self.save_prediction)

        approval_layout.addWidget(self.reject_button)
        approval_layout.addWidget(self.approve_button)
        approval_layout.addWidget(self.approve_all_button)
        approval_layout.addWidget(self.reject_all_button)
        approval_layout.addWidget(self.apply_button)
        approval_box.setLayout(approval_layout)
        container_layout.addWidget(approval_box)

        self._build_shortcut()

        return video_container

    def _build_shortcut(self):
        self.shortcut_man = Shortcut_Manager(self)
        shortcut_setting = {
            "prev_frame":{"key": "Left", "callback": lambda: self._change_frame(-1)},
            "next_frame":{"key": "Right", "callback": lambda: self._change_frame(1)},
            "prev_fast":{"key": "Shift+Left", "callback": lambda: self._change_frame(-10)},
            "next_fast":{"key": "Shift+Right", "callback": lambda: self._change_frame(10)},
            "playback":{"key": "Space", "callback": self._toggle_playback},
            "approve":{"key": "A", "callback": lambda: self.mark_frame_status("Approved")},
            "reject":{"key": "R", "callback": lambda: self.mark_frame_status("Rejected")},
            "undo": {"key": "Ctrl+Z", "callback": self._undo_changes},
            "redo": {"key": "Ctrl+Y", "callback": self._redo_changes},
            "save":{"key": "Ctrl+S", "callback": self.save_prediction},
        }
        self.shortcut_man.add_shortcuts_from_config(shortcut_config=shortcut_setting)
    
    def _display_current_frame(self):
        global_frame_idx = self.frame_list[self.current_frame_idx]
        image_filename = f"img{global_frame_idx:08d}.png"
        image_path = os.path.join(self.temp_dir, image_filename)
        if self.crop_coord is None:
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
    
    def _change_frame(self, delta):
        self.selected_cam = None # Clear the selected cam upon frame switch
        new_frame_idx = self.current_frame_idx + delta
        if 0 <= new_frame_idx < self.total_marked_frames:
            self.current_frame_idx = new_frame_idx
            self._display_current_frame()
            self._navigation_title_controller()
            self._update_button_states()

    def _toggle_playback(self):
        self.progress_widget.toggle_playback()

    def mark_frame_status(self, status:Literal["Rejected","Approved"]):
        self._save_state_for_undo()
        self.frame_status_array[self.current_frame_idx] = 1 if status == "Approved" else 2
        self.apply_button.setEnabled(np.any(self.frame_status_array==1))
        global_frame_idx = self.frame_list[self.current_frame_idx]
        pred_data_to_use = self.new_data_array if status == "Approved" else self.backup_data_array
        self.dlc_data.pred_data_array[global_frame_idx, :, :] = pred_data_to_use[global_frame_idx, :, :]
        self._refresh_ui()

    def approve_all_remaining_frames(self):
        self._save_state_for_undo()
        unproc_mask, global_mask = self._acquire_unproc_mask()
        self.dlc_data.pred_data_array[global_mask] = self.new_data_array[global_mask]
        self.frame_status_array[unproc_mask] = 1
        self.apply_button.setEnabled(True)
        self._refresh_ui()

    def reject_all_remaining_frames(self):
        self._save_state_for_undo()
        unproc_mask, global_mask = self._acquire_unproc_mask()
        self.dlc_data.pred_data_array[global_mask] = self.new_data_array[global_mask]
        self.frame_status_array[unproc_mask] = 2
        self._refresh_ui()

    def save_prediction(self):
        self._refresh_slider()

        if np.any(self.frame_status_array==0):
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Unprocessed Frames - Save and Exit?")
            msg_box.setText(
                f"You have {np.sum(self.frame_status_array==0)} unprocessed frame(s).\n\n"
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
        approve_list_global = np.array(self.frame_list)[self.frame_status_array==1].tolist()
        rejected_list_global = np.array(self.frame_list)[self.frame_status_array==2].tolist()
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

    def _refresh_slider(self):
        self.is_saved = False
        self.progress_widget.clear_frame_category()
        self.progress_widget.set_frame_category_array(self.frame_status_array, {0:"#383838", 1:"#68b3ff", 2:"#F749C6"})
        self.progress_widget.commit_categories()

    def _acquire_unproc_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        unproc_mask = self.frame_status_array == 0
        global_mask = np.zeros((self.total_frames), dtype=bool)
        global_mask[self.frame_list] = unproc_mask
        return unproc_mask, global_mask

    def _refresh_ui(self):
        self._display_current_frame()
        self._update_button_states()
        self._refresh_slider()

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
        current_status = self.frame_status_array[self.current_frame_idx]
        self.approve_button.setEnabled(current_status != 1)
        self.reject_button.setEnabled(current_status != 2)
        self.approve_all_button.setEnabled(np.any(self.frame_status_array==0))
        self.reject_all_button.setEnabled(np.any(self.frame_status_array==0))

    #######################################################################################################################

    def inference_workflow(self):
        if self.batch_size_changed:
            config_path = self.dlc_data.dlc_config_filepath
            dlc_dir = os.path.dirname(self.dlc_data.dlc_config_filepath)
            config_backup = os.path.join(dlc_dir, "config_bak.yaml")
            print("Backup up the original config.yaml as config_bak.yaml")
            shutil.copy(config_path ,config_backup)

            with open(config_path, 'r') as f:
                try:
                    config_org = yaml.load(f, Loader=yaml.SafeLoader)
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML file: {e}")

                config_org["batch_size"] = self.batch_size
                config_org["detector_batch_size"] = self.detector_batch_size

            with open(config_path, 'w') as file:
                yaml.dump(config_org, file, default_flow_style=False, sort_keys=False)
                print(f"DeepLabCut config in {config_path} has been updated.")

        self.total_frames = self.extractor.get_total_frames()

        if self.auto_cropping and self.crop_coord is None:
            frame = self.extractor.get_frame(0)
            roi = get_roi_cv2(frame)
            if roi is not None:
                self.crop_coord = np.array(roi)
                self.roi_set.emit(self.crop_coord)
            else:
                QMessageBox.information(self, "Crop Region Not Set", "User cancel the ROI selection.")
                return

        inference_video_path = None
        try:
            if len(self.frame_list)  > 0.9 * self.total_frames and self.crop_coord is None:
                inference_video_path = self.export_set.video_filepath
            elif len(self.frame_list) > 5000:
                inference_video_path = os.path.join(self.temp_dir, "temp_extract.mp4")
                self._extract_marked_frame_as_video(self.crop_coord)
            else:
                self._extract_marked_frame_images(self.crop_coord)
        except Exception as e:
            QMessageBox(self, "Error", f"Error during frame image extraction. Error:{e}")
            self.emergency_exit()
            return

        self.setup_container.setVisible(False)
        self.wait_container.setVisible(True)
        self.center()
        QtWidgets.QApplication.processEvents()

        try:
            if inference_video_path:
                self._analyze_frame_videos(inference_video_path)
            else:
                self._analyze_frame_images()
        except Exception as e:
            QMessageBox(self, "Error", f"Error during frame analysis. Error:{e}")
            self.emergency_exit()
            return
        
        if self.fresh_pred:
            temp_pred_filename = self._load_and_remap_new_prediction()
            video_path = os.path.dirname(self.video_filepath)
            if "image_predictions_" in temp_pred_filename:
                pred_filename = temp_pred_filename.replace("image_predictions_", self.export_set.video_name)
            else:
                pred_filename = temp_pred_filename.replace("temp_extract", self.export_set.video_name)

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

    def _extract_marked_frame_images(self, crop_coord=None):
        progress = Progress_Indicator_Dialog(0, 100, "Frame Extraction", "Extracting frames from video", parent=self)
        exporter = Exporter(
            dlc_data=self.dlc_data, export_settings=self.export_set, frame_list=self.frame_list, progress_callback=progress, crop_coord=crop_coord)
        exporter.export_data_to_DLC(frame_only=True)

    def _extract_marked_frame_as_video(self, crop_coord=None):
        progress = Progress_Indicator_Dialog(0, 100, "Frame Extraction", "Extracting frames from video", parent=self)
        exporter = Exporter(
            dlc_data=self.dlc_data, export_settings=self.export_set, frame_list=self.frame_list, progress_callback=progress, crop_coord=crop_coord)
        exporter.export_frame_to_video()

    def _analyze_frame_images(self):
        import deeplabcut
        deeplabcut.analyze_images(
            config=self.dlc_data.dlc_config_filepath,
            images=self.temp_dir,
            shuffle=self.shuffle_idx,
            max_individuals=self.max_individual_val)
        
    def _analyze_frame_videos(self, inference_video_path):
        import deeplabcut
        deeplabcut.analyze_videos(
            config=self.dlc_data.dlc_config_filepath,
            videos=[inference_video_path],
            shuffle=self.shuffle_idx
        )

    def _load_and_remap_new_prediction(self) -> str:
        h5_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".h5")]
        if not h5_files:
            raise RuntimeError("Failed to find new prediction.")
        pred_filepath = os.path.join(self.temp_dir, h5_files[-1])
        dlc_config_filepath = self.dlc_data.dlc_config_filepath
        loader = Prediction_Loader(dlc_config_filepath, pred_filepath)
        loaded_data = loader.load_data()
        new_data_array = np.full(
            (self.dlc_data.pred_frame_count, self.dlc_data.instance_count, self.dlc_data.num_keypoint*3), np.nan)
        temp_data_array = loaded_data.pred_data_array

        if self.crop_coord is not None:
            coords_array = crop_coord_to_array(self.crop_coord, temp_data_array.shape, self.frame_list)
            temp_data_array = temp_data_array + coords_array

        if self.dlc_data.pred_data_array is not None or not np.any(self.dlc_data.pred_data_array): # Only do correction on new data
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
        temp_data_array, _, _ = tf.track_correction()
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

    def _save_state_for_undo(self):
        self.uno.save_state_for_undo(self.frame_status_array)

    def _undo_changes(self):
        data_array = self.uno.undo()
        self._undo_redo_worker(data_array)

    def _redo_changes(self):
        data_array = self.uno.redo()
        self._undo_redo_worker(data_array)

    def _undo_redo_worker(self, data_array):
        if data_array is not None and np.any(self.frame_status_array - data_array != 0):
            for frame_idx in np.where(self.frame_status_array - data_array)[0]:
                global_idx = self.frame_list[frame_idx]
                if self.frame_status_array[frame_idx] == 1:
                    self.dlc_data.pred_data_array[global_idx] = self.backup_data_array[global_idx]
                if data_array[frame_idx] == 1:
                    self.dlc_data.pred_data_array[global_idx] = self.new_data_array[global_idx]

            self.frame_status_array = data_array
            self._refresh_ui()

    #######################################################################################################################

    def center(self):
        screen = QGuiApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def emergency_exit(self, reason:str="Check terminal for reason."):
        traceback.print_exc()
        QMessageBox.warning(self, "Rerun Failed", reason)
        self.reject()

    def closeEvent(self, event):
        handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_prediction)
        if self.temp_directory is not None:
            self.temp_directory.cleanup()

class Spinbox_With_Label(QtWidgets.QWidget):
    value_changed = Signal(int)

    def __init__(self, label_text:str, spinbox_range:Tuple[int, int], initial_val:int, parent=None):
        super().__init__(parent)
        widget_layout = QHBoxLayout(self)
        spinbox_label = QtWidgets.QLabel(label_text)
        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setRange(*spinbox_range)
        self.spinbox.setValue(initial_val)
        self.spinbox.valueChanged.connect(self._spinbox_changed)

        widget_layout.addWidget(spinbox_label)
        widget_layout.addWidget(self.spinbox)

    def _spinbox_changed(self, val):
        self.value_changed.emit(val)