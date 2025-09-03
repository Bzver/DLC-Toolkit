import os
import numpy as np

import h5py
import yaml
from utils import helper as duh

from typing import Optional, Any, Dict

from .h5_op import validate_h5_keys
from utils.dataclass import Loaded_DLC_Data

class Prediction_Loader:
    """A class to load DeepLabCut configuration and prediction data."""
    def __init__(self, dlc_config_filepath: str, prediction_filepath: str):
        self.dlc_config_filepath = dlc_config_filepath
        self.prediction_filepath = prediction_filepath

    def load_data(self, metadata_only: bool = False) -> Optional[Loaded_DLC_Data]:
        config_data = self._load_config_data()
        
        if metadata_only:
            loaded_data = Loaded_DLC_Data(
                **config_data,
                prediction_filepath=None,
                pred_data_array=None,
                pred_frame_count=None
            )
            return loaded_data

        pred_data = self._load_prediction_data(config_data)
        
        # Merge dictionaries for a clean object creation
        loaded_data = Loaded_DLC_Data(**config_data, **pred_data)
        return loaded_data

    def _load_config_data(self) -> Dict[str, Any]:
        """Internal method to load DLC configuration from the YAML file."""
        if not os.path.isfile(self.dlc_config_filepath):
            raise FileNotFoundError(f"DLC config file not found at: {self.dlc_config_filepath}")
        
        try:
            with open(self.dlc_config_filepath, "r") as conf:
                cfg = yaml.safe_load(conf)
            
            multi_animal = cfg.get("multianimalproject", False)
            keypoints = cfg.get("bodyparts", []) if not multi_animal else cfg.get("multianimalbodyparts", [])
            individuals = cfg.get("individuals")
            instance_count = len(individuals) if individuals else 1
            keypoint_to_idx = {name: idx for idx, name in enumerate(keypoints)}
            
            config_dict = {
                "dlc_config_filepath": self.dlc_config_filepath,
                "scorer": cfg.get("scorer", None),
                "multi_animal": multi_animal,
                "keypoints": keypoints,
                "skeleton": cfg.get("skeleton", []),
                "individuals": individuals,
                "instance_count": instance_count,
                "num_keypoint": len(keypoints),
                "keypoint_to_idx": keypoint_to_idx
            }

            return config_dict
        except Exception as e:
            raise RuntimeError(f"Error loading DLC config: {e}") from e

    def _load_prediction_data(self, config_data: dict) -> Dict[str, Any]:
        """Internal method to load prediction data from HDF5 file."""
        if not os.path.isfile(self.prediction_filepath):
            raise FileNotFoundError(f"Prediction file not found at: {self.prediction_filepath}")
        
        if "num_keypoint" not in config_data or "instance_count" not in config_data:
            raise ValueError("Config data is incomplete. Please load config first.")

        num_keypoint = config_data["num_keypoint"]
        instance_count = config_data["instance_count"]

        try:
            with h5py.File(self.prediction_filepath, "r") as pred_file:
                key, subkey = validate_h5_keys(pred_file)
                if not key or not subkey:
                    raise ValueError(f"No valid key found in '{self.prediction_filepath}'")
                
                prediction_raw = pred_file[key][subkey]

                if subkey == "block0_values": # Already an array
                    pred_data_values = np.array(prediction_raw)
                    pred_frame_count = prediction_raw.shape[0]
                else:
                    pred_data_values = np.array([item[1] for item in prediction_raw])
                    pred_frame_count = len(prediction_raw)

                expected_cols = instance_count * num_keypoint * 3
                if pred_data_values.shape[1] != expected_cols:
                    raise ValueError(
                        f"Prediction data has {pred_data_values.shape[1]} columns, "
                        f"but {expected_cols} columns were expected based on config "
                        f"(instances={instance_count}, keypoints={num_keypoint}). "
                        "Please verify the prediction file and configuration match."
                    )

            pred_data_array = duh.unflatten_data_array(pred_data_values, instance_count)

            pred_data_dict = {
                "prediction_filepath": self.prediction_filepath,
                "pred_data_array": pred_data_array,
                "pred_frame_count": pred_frame_count
            }
            return pred_data_dict
        except Exception as e:
            raise RuntimeError(f"Error loading prediction data: {e}")