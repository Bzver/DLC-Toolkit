import os
import shutil
import tempfile
import yaml
import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QDialog

from typing import List, Optional, Tuple

from ui import Spinbox_With_Label, Progress_Indicator_Dialog
from core.io import (
    Exporter, Prediction_Loader, Frame_Extractor, Frame_Extractor_Img,
    save_predictions_to_new_h5, timestamp_new_prediction,
    )
from .reviewer import Parallel_Review_Dialog
from utils.helper import crop_coord_to_array, get_roi_cv2, validate_crop_coord
from utils.logger import logger, Loggerbox
from utils.dataclass import Loaded_DLC_Data


class DLC_Inference(QDialog):
    prediction_saved = Signal(str)
    frames_exported = Signal(tuple)
    roi_set = Signal(object)

    def __init__(
        self,
        dlc_data:Loaded_DLC_Data,
        frame_list:List[int],
        video_filepath:str,
        roi:Optional[np.ndarray]=None,
        mask:Optional[np.ndarray]=None,
        parent=None
        ):
        super().__init__(parent)
        self.setWindowTitle("Run Predictions in DLC")
        self.dlc_data = dlc_data
        self.frame_list = frame_list
        self.video_filepath = video_filepath
        self.setFixedWidth(600)

        self.frame_list.sort()
        self.cropping = False
        self.masking = False
        self.grayscaling = False

        self.crop_coord = validate_crop_coord(roi)
        self.mask_region = mask
        self.video_name, _ = os.path.splitext(os.path.basename(self.video_filepath))

        self.temp_directory = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_directory.name
        self.extractor = Frame_Extractor(self.video_filepath)

        layout = QVBoxLayout(self)
        setup_container = self._build_setup_container()
        if setup_container is None:
            return
        layout.addLayout(setup_container)

    def _build_setup_container(self):
        iteration_idx, iteration_folder = self._check_iteration_integrity()
        if iteration_folder is None:
            self._panic_exit(f"Iteration {iteration_idx} folder not found.")
            return
        
        self.iteration_folder = iteration_folder

        available_shuffles = self._check_available_shuffles()
        if available_shuffles is None:
            self._panic_exit("No shuffles found in iteration folder.")
            return
        
        available_shuffles.sort()
        self.shuffle_idx = int(available_shuffles[-1])
        shuffle_metadata_text, model_status = self._check_shuffle_metadata()

        container_layout = QVBoxLayout()

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

        self.cropping_checkbox = QtWidgets.QCheckBox("Crop")
        self.cropping_checkbox.setChecked(self.cropping)
        self.cropping_checkbox.toggled.connect(self._cropping_changed)

        self.masking_checkbox = QtWidgets.QCheckBox("Mask")
        self.masking_checkbox.setChecked(self.masking)
        self.masking_checkbox.toggled.connect(self._masking_changed)

        self.grayscaling_checkbox = QtWidgets.QCheckBox("Grayscale")
        self.grayscaling_checkbox.setChecked(self.grayscaling)
        self.grayscaling_checkbox.toggled.connect(self._grayscaling_changed)

        button_frame = QHBoxLayout()

        self.batch_size_changed = False
        self.batchsize_label_spinbox = Spinbox_With_Label(label_text = "Batch Size: ", spinbox_range = (1,10000), initial_val = self.batch_size, parent = self)
        self.batchsize_label_spinbox.value_changed.connect(self._batch_size_spinbox_changed)

        self.detector_batchsize_label_spinbox = Spinbox_With_Label(
            label_text = "Detector Batch Size: ", spinbox_range = (1,10000), initial_val = self.detector_batch_size, parent = self
        )
        self.detector_batchsize_label_spinbox.value_changed.connect(self._det_batch_size_spinbox_changed)
        self._determine_det_spinbox_vis(shuffle_metadata_text)

        self.start_button = QPushButton("Run Inference")
        self.start_button.clicked.connect(self._inference_pipe)
        button_frame.addWidget(self.cropping_checkbox)
        button_frame.addWidget(self.masking_checkbox)
        button_frame.addWidget(self.grayscaling_checkbox)
        button_frame.addWidget(self.batchsize_label_spinbox)
        button_frame.addWidget(self.detector_batchsize_label_spinbox)
        button_frame.addWidget(self.start_button)
        container_layout.addLayout(button_frame)
 
        return container_layout

    def _check_iteration_integrity(self):
        dlc_config_filepath = self.dlc_data.dlc_config_filepath
        dlc_folder = os.path.dirname(self.dlc_data.dlc_config_filepath)
        with open(dlc_config_filepath, 'r') as dcf:
            dlc_config = yaml.safe_load(dcf)
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
            shuffle_config = yaml.safe_load(scf)

        method = shuffle_config["method"]
        if shuffle_config["model"]["backbone"]["type"] == "CondPreNet":
            self.model_name = "CondPreNet_" + shuffle_config["model"]["backbone"]["backbone"]["model_name"]
        else:
            self.model_name = shuffle_config["model"]["backbone"]["model_name"]

        config_text = f"Model Name: {self.model_name}"
        
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

    def _masking_changed(self, checked:bool):
        self.masking = checked

    def _cropping_changed(self, checked:bool):
        self.cropping = checked

    def _grayscaling_changed(self, checked:bool):
        self.grayscaling = checked

    def _determine_det_spinbox_vis(self, text:str):
        self.detector_batchsize_label_spinbox.setVisible("Detector" in text)

    def _inference_pipe(self, headless:bool=False):
        if self.batch_size_changed:
            self._update_config()

        self.total_frames = self.extractor.get_total_frames()

        if self.masking and self.mask_region is None:
            Loggerbox.info(self, "Mask Region Not Calculated", "Click Toggle Smart Masking in the Main-View submenu first.")
            return

        if self.cropping and self.crop_coord is None:
            frame = self.extractor.get_frame(0)
            roi = get_roi_cv2(frame)
            if roi is not None:
                self.crop_coord = np.array(roi)
                self.roi_set.emit(self.crop_coord)
            else:
                Loggerbox.info(self, "Crop Region Not Set", "User cancel the ROI selection.")
                return

        inference_video_path = None
        try:
            if len(self.frame_list)  > int(0.9 * self.total_frames) and not (self.cropping or self.masking or self.grayscaling):
                inference_video_path = self.video_filepath
                self.extractor_reviewer = Frame_Extractor(inference_video_path)
            elif len(self.frame_list) < 500 and not self.model_name.startswith("CondPreNet_"):
                self._extract_marked_frame_images()
                self.extractor_reviewer = Frame_Extractor_Img(self.temp_dir)
            else:
                inference_video_path = os.path.join(self.temp_dir, "temp_extract.mp4")
                self._extract_marked_frame_as_video()
                self.extractor_reviewer = Frame_Extractor(inference_video_path)
        except Exception as e:
            self._panic_exit(reason=f"Error during frame image extraction. Error:{e}", exception=e)
            return

        QtWidgets.QApplication.processEvents()

        try:
            if not headless:
                self.on_hold_dialog = On_Hold_Dialog(self)
                self.on_hold_dialog.show()
            self.hide()
            QtWidgets.QApplication.processEvents()
            if inference_video_path:
                self._analyze_frame_videos(inference_video_path)
            else:
                self._analyze_frame_images()
        except Exception as e:
            self._panic_exit(reason=f"Error during frame analysis. Error:{e}", exception=e)
            return
        
        if not headless:
            self.on_hold_dialog.accept()

        self.show()
        self._process_new_pred(headless)

    def _update_config(self):
        config_path = self.dlc_data.dlc_config_filepath
        dlc_dir = os.path.dirname(self.dlc_data.dlc_config_filepath)
        config_backup = os.path.join(dlc_dir, "config_bak.yaml")
        logger.info("[INFER] Backup up the original config.yaml as config_bak.yaml")
        shutil.copy(config_path ,config_backup)

        with open(config_path, 'r') as f:
            try:
                config_org = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

            config_org["batch_size"] = self.batch_size
            config_org["detector_batch_size"] = self.detector_batch_size

        with open(config_path, 'w') as file:
            yaml.dump(config_org, file, default_flow_style=False, sort_keys=False)
            logger.info(f"[INFER] DeepLabCut config in {config_path} has been updated.")

    def _extract_marked_frame_images(self):
        progress = Progress_Indicator_Dialog(0, 100, "Frame Extraction", "Extracting frames from video", parent=self)
        exporter = Exporter(
            dlc_data=self.dlc_data,
            save_folder=self.temp_dir,
            video_filepath=self.video_filepath,
            frame_list=self.frame_list,
            progress_callback=progress,
            crop_coord=self.crop_coord if self.cropping else None,
            mask=self.mask_region,
            grayscaling=self.grayscaling
            )
        corrected_indices = exporter.export_data_to_DLC(frame_only=True)
        if corrected_indices:
            self.frame_list = corrected_indices

    def _extract_marked_frame_as_video(self):
        progress = Progress_Indicator_Dialog(0, 100, "Frame Extraction", "Extracting frames from video", parent=self)
        exporter = Exporter(
            dlc_data=self.dlc_data,
            save_folder=self.temp_dir,
            video_filepath=self.video_filepath,
            frame_list=self.frame_list,
            progress_callback=progress,
            crop_coord=self.crop_coord if self.cropping else None,
            mask=self.mask_region,
            grayscaling=self.grayscaling
            )
        corrected_indices = exporter.export_frame_to_video()
        if corrected_indices:
            self.frame_list = corrected_indices

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
        
        temp_data_array = loaded_data.pred_data_array
        new_data_array = np.full(
            (self.dlc_data.pred_frame_count, temp_data_array.shape[1], temp_data_array.shape[2]), np.nan)

        if self.crop_coord is not None:
            coords_array = crop_coord_to_array(self.crop_coord, temp_data_array.shape)
            temp_data_array = temp_data_array + coords_array

        expected_length = len(self.frame_list)
        actual_length = temp_data_array.shape[0]
        if actual_length < expected_length:
            logger.warning(f"[INFER] Inferenced data is shorter than expected ( {actual_length} < {expected_length} ),"
                           "appending with nans. Predictions might be misaligned.")
            temp_data_array = np.pad(
                temp_data_array, pad_width=((0, expected_length - actual_length), (0, 0), (0, 0)), mode='constant', constant_values=np.nan)

        new_data_array[self.frame_list, :, :] = temp_data_array

        self.new_data_array = new_data_array
        return h5_files[-1]

    def _process_new_pred(self, headless:bool=False):
        temp_pred_filename = self._load_and_remap_new_prediction()
        video_path = os.path.dirname(self.video_filepath)
        if "image_predictions_" in temp_pred_filename:
            pred_filename = temp_pred_filename.replace("image_predictions_", self.video_name)
        else:
            pred_filename = temp_pred_filename.replace("temp_extract", self.video_name)

        pred_filepath = os.path.join(video_path, pred_filename)
        self.save_path = timestamp_new_prediction(pred_filepath)

        if self.dlc_data.pred_data_array is None or not np.any(~np.isnan(self.dlc_data.pred_data_array)):
            pred_data_array=self.new_data_array
            list_tuple = (self.frame_list, [])
            self._save_pred_to_file(pred_data_array, list_tuple)
        elif headless or np.all(np.isnan(self.dlc_data.pred_data_array[self.frame_list])):
            old_data_array = self.dlc_data.pred_data_array
            old_data_array[self.frame_list, ...] = self.new_data_array[self.frame_list, ...]
            list_tuple = (self.frame_list, [])
            self._save_pred_to_file(old_data_array, list_tuple)
        else:
            self.hide()
            QtWidgets.QApplication.processEvents()
            self.reviewer = Parallel_Review_Dialog(self.dlc_data, self.extractor_reviewer, self.new_data_array, self.frame_list,
                                                   crop_coord=self.crop_coord, grayscaling=self.grayscaling_checkbox.isChecked(), parent=self)
            self.reviewer.pred_data_exported.connect(self._save_pred_to_file)
            self.reviewer.exec()

    def _save_pred_to_file(self, pred_data_array:np.ndarray, list_tuple:Tuple[List[int], List[int]]):
        try:
            save_predictions_to_new_h5(dlc_data=self.dlc_data, pred_data_array=pred_data_array, save_path=self.save_path)
        except Exception as e:
            Loggerbox.error(self, "Saving Error", f"An error occurred during saving: {e}", exc=e)
            return
        
        logger.info(f"[INFER] Prediction saved to {self.save_path}")

        self.prediction_saved.emit(self.save_path)
        self.frames_exported.emit(list_tuple)
        self.accept()

    def _panic_exit(self, reason:str="Check terminal for reason.", exception:Optional[Exception]=None):
        Loggerbox.error(self, "Rerun Failed", reason, exception)
        if hasattr(self, "on_hold_dialog"):
            self.on_hold_dialog.accept()
        self.reject()

    def closeEvent(self, event):
        if hasattr(self, "extractor_reviewer"):
            self.extractor_reviewer.close()


class On_Hold_Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        wait_layout = QHBoxLayout()
        wait_layout.setContentsMargins(10, 10, 10, 10)
        wait_layout.setSpacing(20)

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
        info_label = QtWidgets.QLabel("DeeplabCut inference started,\n Check terminal for progress.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: gray; }")
        text_layout.addWidget(title_label)
        text_layout.addWidget(info_label)
        text_layout.addStretch()
        wait_layout.addLayout(text_layout)

        self.setLayout(wait_layout)