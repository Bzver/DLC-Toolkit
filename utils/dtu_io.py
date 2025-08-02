import os

import h5py
import yaml

import numpy as np

import cv2

from typing import List, Tuple, Optional, Any
from numpy.typing import NDArray

from .dtu_dataclass import Loaded_DLC_Data, Export_Settings
from . import dtu_helper as duh

class DLC_Loader:
    """A class to load DeepLabCut configuration and prediction data."""
    def __init__(self, dlc_config_filepath: str, prediction_filepath: str):
        self.dlc_config_filepath = dlc_config_filepath
        self.prediction_filepath = prediction_filepath

    def load_data(self, metadata_only: bool = False) -> Tuple[Optional[Loaded_DLC_Data], str]:
        """
        Loads both the DLC config and, optionally, prediction data.
        Returns a (Loaded_DLC_Data object, "Success") tuple on success, or (None, "Error Message") on failure.
        """
        config_data, msg = self._load_config_data()
        if not config_data:
            return None, msg

        if metadata_only:
            loaded_data = Loaded_DLC_Data(
                **config_data,
                prediction_filepath=None,
                pred_data_array=None,
                pred_frame_count=None
            )
            return loaded_data, "DLC config loaded successfully!"

        pred_data, msg = self._load_prediction_data(config_data)
        if not pred_data:
            return None, msg
        
        # Merge dictionaries for a clean object creation
        loaded_data = Loaded_DLC_Data(**config_data, **pred_data)
        return loaded_data, "DLC config and prediction loaded successfully!"

    def _load_config_data(self) -> Tuple[Optional[dict[str, Any]], str]:
        """
        Internal method to load DLC configuration from the YAML file.
        Returns a dictionary of config data on success, or (None, "Error Message") on failure.
        """
        if not os.path.isfile(self.dlc_config_filepath):
            return None, f"DLC config file not found at: {self.dlc_config_filepath}"
        
        try:
            with open(self.dlc_config_filepath, "r") as conf:
                cfg = yaml.safe_load(conf)
            
            multi_animal = cfg.get("multianimalproject", False)
            keypoints = cfg.get("bodyparts", []) if not multi_animal else cfg.get("multianimalbodyparts", [])
            individuals = cfg.get("individuals")
            instance_count = len(individuals) if individuals else 1
            
            config_dict = {
                "dlc_config_filepath": self.dlc_config_filepath,
                "scorer": cfg.get("scorer", None),
                "multi_animal": multi_animal,
                "keypoints": keypoints,
                "skeleton": cfg.get("skeleton", []),
                "individuals": individuals,
                "instance_count": instance_count,
                "num_keypoint": len(keypoints)
            }
            return config_dict, "Success"
        except Exception as e:
            return None, f"Error loading DLC config: {e}"

    def _load_prediction_data(self, config_data: dict) -> Tuple[Optional[dict[str, Any]], str]:
        """
        Internal method to load prediction data from HDF5 file.
        Returns a dictionary of prediction data on success, or (None, "Error Message") on failure.
        """
        if not os.path.isfile(self.prediction_filepath):
            return None, f"Prediction file not found at: {self.prediction_filepath}"
        
        if "num_keypoint" not in config_data or "instance_count" not in config_data:
            return None, "Config data is incomplete. Please load config first."

        num_keypoint = config_data["num_keypoint"]
        instance_count = config_data["instance_count"]

        try:
            with h5py.File(self.prediction_filepath, "r") as pred_file:
                if "tracks" not in pred_file:
                    return None, "Error: Prediction file not valid, no 'tracks' key found."

                prediction_raw = pred_file["tracks"]["table"]
                pred_data_values = np.array([item[1] for item in prediction_raw])
                pred_frame_count = prediction_raw.size

                expected_cols = instance_count * num_keypoint * 3
                if pred_data_values.shape[1] != expected_cols:
                    error_msg = (f"Prediction data columns ({pred_data_values.shape[1]}) "
                                 f"do not match expected ({expected_cols}) based on config. "
                                 "Check config or prediction file.")
                    return None, error_msg

            pred_data_array = duh.unflatten_data_array(pred_data_values, instance_count)

            pred_data_dict = {
                "prediction_filepath": self.prediction_filepath,
                "pred_data_array": pred_data_array,
                "pred_frame_count": pred_frame_count
            }
            return pred_data_dict, "Success"
        except Exception as e:
            return None, f"Error loading prediction data: {e}"
        
class DLC_Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    def __init__(self, dlc_data: Loaded_DLC_Data, export_settings: Export_Settings,
        frame_list: List[int], pred_data_array: NDArray=None):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list

    def export_data_to_DLC(self, frame_only:bool=False) -> Tuple[bool, str]:
        status, msg = self._extract_frame()
        if not status:
            return False, msg

        if frame_only:
            msg = "Successfully exported marked frames to DLC for labeling!"
            return status, msg
    
        status, msg = self._extract_pred()
        if not status:
            return False, msg
        
        msg = "Successfully exported frames and prediction to DLC!"
        return status, msg

    def _extract_frame(self) -> Tuple[bool, str]:
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            return False, f"Error: Could not open video {self.video_file}"
        
        frames_to_extract = set(self.frame_list)

        for frame in frames_to_extract:
            image_path = f"img{str(int(frame)).zfill(8)}.png"
            image_output_path = os.path.join(self.save_path, image_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_output_path, frame)
        return True, "Success"

    def _extract_pred(self, pred_data_array) -> Tuple[bool, str]:
        if not pred_data_array:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
        
        if not duh.prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list):
            return False, "Error exporting predictions to csv."

        if not duh.csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal):
            return False, "Error transforming to h5."
        
        return True, f"Label extracted to {self.export_settings.save_path}"