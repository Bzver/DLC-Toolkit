import os
import tempfile

import yaml

from PySide6 import QtWidgets
import deeplabcut

from typing import List

from .dtu_dataclass import Loaded_DLC_Data, Export_Settings
from .dtu_io import DLC_Exporter

class DLC_RERUN(QtWidgets.QDialog):
    def __init__(self, dlc_data:Loaded_DLC_Data, frame_list:List[int], video_filepath:str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Re-run Predictions With Selected Frames in DLC")
        self.dlc_data = dlc_data
        self.frame_list = frame_list

        iteration_idx, iteration_folder = self.check_iteration_integrity()
        if iteration_folder is None:
            self.emergency_exit(f"Iteration {iteration_idx} folder not found.")
            return

        available_shuffles = self.check_shuffle_availability(iteration_folder)
        if available_shuffles is None:
            self.emergency_exit("No shuffles found in iteration folder.")
            return
        
        self.shuffle_idx = int(available_shuffles[-1])
        shuffle_metadata_text, model_status = self.check_shuffle_metadata(iteration_folder)

        layout = QtWidgets.QVBoxLayout(self)

        # Shuffle controls
        shuffle_frame = QtWidgets.QHBoxLayout()
        shuffle_label = QtWidgets.QLabel(f"Shuffle: ")
        self.shuffle_spinbox = QtWidgets.QSpinBox()
        self.shuffle_spinbox.setRange(0, 100)
        self.shuffle_spinbox.setValue(self.shuffle_idx)

        self.shuffle_config_label = QtWidgets.QLabel(f"{shuffle_metadata_text}")
        if not model_status:
            self.shuffle_config_label.setStyleSheet("color: red;")

        shuffle_frame.addWidget(shuffle_label)
        shuffle_frame.addWidget(self.shuffle_spinbox)
        shuffle_frame.addWidget(self.shuffle_config_label)

        # Max animal settings
        individual_frame = QtWidgets.QHBoxLayout()
        max_individual_label = QtWidgets.QLabel(f"Number of animals in marked frames: ")
        self.max_individual_val = len(self.dlc_data.individuals)
        self.max_individual_spinbox = QtWidgets.QSpinBox()
        self.max_individual_spinbox.setRange(1, 20)
        self.max_individual_spinbox.setValue(self.max_individual_val)

        individual_frame.addWidget(max_individual_label)
        individual_frame.addWidget(self.max_individual_spinbox)

        # Button for start
        self.start_button = QtWidgets.QPushButton("Extract Frames and Rerun Predictions in DLC")
        self.start_button.clicked.connect(self.rerun_workflow)

        layout.addLayout(shuffle_frame)
        layout.addLayout(individual_frame)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.export_set = Export_Settings(
            video_filepath=video_filepath, video_name="", save_path=self.temp_dir, export_mode="Append")

    def check_iteration_integrity(self):
        dlc_config_filepath = self.dlc_data.dlc_config_filepath
        dlc_folder = os.path.dirname(self.dlc_data.dlc_config_filepath)
        with open(dlc_config_filepath, 'r') as dcf:
            dlc_config = yaml.load(dcf, Loader=yaml.SafeLoader)
        iteration_idx = dlc_config["iteration"]
        iteration_folder = os.path.join(dlc_folder, "dlc-models-pytorch", f"iteration-{iteration_idx}")
        if not os.path.isdir(iteration_folder):
            return iteration_idx, None
        
        return iteration_idx, iteration_folder

    def check_shuffle_availability(self, iteration_folder):
        available_shuffles = [f.split("shuffle")[1] for f in os.listdir(iteration_folder) if "shuffle" in f]
        if not available_shuffles:
            return None

        return available_shuffles
    
    def check_shuffle_metadata(self, iteration_folder):
        available_detector_models = []
        available_models = []
        for f in os.listdir(iteration_folder):
            fullpath = os.path.join(iteration_folder, f)
            if f"shuffle{self.shuffle_idx}" in f and os.path.isdir(fullpath):
                shuffle_folder = fullpath

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

    #######################################################################################################################

    def rerun_workflow(self):
        self.extract_marked_frame_images()
        self.analyze_frame_images()

    def extract_marked_frame_images(self):
        exporter = DLC_Exporter(dlc_data=self.dlc_data, export_settings=self.export_set, frame_list=self.frame_list)
        exporter.export_data_to_DLC(frame_only=True)

    def analyze_frame_images(self):
        deeplabcut.analyze_images(
            config=self.dlc_data.dlc_config_filepath,
            images=self.temp_dir,
            shuffle=self.shuffle_idx)

    #######################################################################################################################

    def emergency_exit(self, reason:str):
        QtWidgets.QMessageBox.warning(self, "Rerun Not Possible", reason)
        self.reject()

    def closeEvent(self, event):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()