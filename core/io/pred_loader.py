import os
import numpy as np

import h5py
import yaml

from typing import Any, Dict, Optional

from .h5_op import validate_h5_keys, fix_h5_kp_order
from .io_helper import (
    unflatten_data_array, add_mock_confidence_score, nuke_negative_val_in_loaded_pred, load_crop_notations)
from core.dataclass import Loaded_DLC_Data

class Prediction_Loader:
    def __init__(
            self,
            dlc_config_filepath: str,
            prediction_filepath:Optional[str]=None):
        
        self.dlc_config_filepath = dlc_config_filepath
        self.prediction_filepath = prediction_filepath
        self.crop_dict = None

        project_dir = os.path.dirname(self.prediction_filepath)
        crop_notation_filepath = os.path.join(project_dir, "crop.yaml")
        if os.path.isfile(crop_notation_filepath):
            self.crop_dict = load_crop_notations(crop_notation_filepath)

    def load_data(self, metadata_only: bool = False, force_load_pred:bool = False) -> Loaded_DLC_Data:
        config_data = self._load_config_data()
        
        if metadata_only:
            loaded_data = Loaded_DLC_Data(
                **config_data,
                prediction_filepath=None,
                pred_data_array=None,
                pred_frame_count=None
            )
            return loaded_data

        if os.path.basename(self.prediction_filepath).startswith("CollectedData_") and not force_load_pred:
            pred_data = self._load_labeled_data(config_data)
        else:
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

    def _load_prediction_data(self, config_data:Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to load prediction data from HDF5 file."""
        if not os.path.isfile(self.prediction_filepath):
            raise FileNotFoundError(f"Prediction file not found at: {self.prediction_filepath}")

        try:
            with h5py.File(self.prediction_filepath, "r") as pred_file:
                key, subkey = validate_h5_keys(pred_file)
                if not key or not subkey:
                    raise ValueError(f"No valid key found in '{self.prediction_filepath}'")
                
                prediction_raw = pred_file[key][subkey]
                
                instance_count = config_data["instance_count"]
                num_keypoint = config_data["num_keypoint"]
                if subkey == "block0_values": # Already an array
                    pred_data_values = fix_h5_kp_order(pred_file, key, config_data)
                    
                    pred_frame_count = pred_data_values.shape[0]
                else:
                    pred_data_values = np.array([item[1] for item in prediction_raw])
                    pred_frame_count = len(prediction_raw)

                expected_cols = instance_count * num_keypoint * 3
                half_expected_cols = instance_count * num_keypoint * 2

                if pred_data_values.shape[1] != expected_cols:
                    if pred_data_values.shape[1] == half_expected_cols:
                        pred_data_values = add_mock_confidence_score(pred_data_values)
                    else:
                        raise ValueError(
                            f"Prediction data has {pred_data_values.shape[1]} columns, "
                            f"but {expected_cols} columns were expected based on config "
                            f"(instances={instance_count}, keypoints={num_keypoint}). "
                            "Please verify the prediction file and configuration match."
                        )

            pred_data_array = unflatten_data_array(pred_data_values, instance_count)
            pred_data_array = nuke_negative_val_in_loaded_pred(pred_data_array)

            pred_data_dict = {
                "prediction_filepath": self.prediction_filepath,
                "pred_data_array": pred_data_array,
                "pred_frame_count": pred_frame_count
            }
            return pred_data_dict
        except Exception as e:
            raise RuntimeError(f"Error loading prediction data: {e}")
        
    def _load_labeled_data(self, config_data:Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to load label data from CollectedData_*.h5 file."""
        if not os.path.isfile(self.prediction_filepath):
            raise FileNotFoundError(f"Prediction file not found at: {self.prediction_filepath}")
        
        keypoints = config_data["keypoints"]
        instance_count = config_data["instance_count"]

        try:
            with h5py.File(self.prediction_filepath, "r") as lbf:
                key, subkey = validate_h5_keys(lbf)
                if not key or not subkey:
                    raise ValueError(f"No valid key found in '{self.prediction_filepath}'")
                
                num_keypoint = len(keypoints)

                if subkey == "block0_values": # Already an array
                    pred_data_values = fix_h5_kp_order(lbf, key, config_data)
                    pred_frame_count = pred_data_values.shape[0]
                else:
                    raise ValueError("'block0_values' not found in labeled HDF5.")
                
                expected_cols_labeled = instance_count * num_keypoint * 2 

                if pred_data_values.shape[1] == expected_cols_labeled:
                    pred_data_values = add_mock_confidence_score(pred_data_values)
                    pred_data_unflattened = unflatten_data_array(pred_data_values, instance_count)
                else:
                    raise ValueError(
                        f"Label data has {pred_data_values.shape[1]} columns, "
                        f"but {expected_cols_labeled} columns were expected based on config "
                        f"(instances={instance_count}, keypoints={num_keypoint}). "
                        "Please verify the label data file and configuration match."
                    )

                if "axis1_level2" in lbf[key]:
                    labeled_frame_list = lbf[key]["axis1_level2"].asstr()[()]
                    labeled_frame_list = [int(f.split("img")[1].split(".")[0]) for f in labeled_frame_list]
                    labeled_frame_list.sort()
                    pred_frame_count = max(labeled_frame_list) + 1
                
                    pred_data_array = np.full((pred_frame_count, instance_count, num_keypoint*3),np.nan)
                    pred_data_array[labeled_frame_list, ...] = pred_data_unflattened
                    pred_data_array = nuke_negative_val_in_loaded_pred(pred_data_array)
                else:
                    raise ValueError("'axis1_level2' not found in labeled HDF5.")

            if self.crop_dict:
                for crop_coord, frame_list in self.crop_dict.items():
                    pred_data_array[frame_list, :, 0::3] += crop_coord[0]
                    pred_data_array[frame_list, :, 1::3] += crop_coord[1]

            pred_data_dict = {
                "prediction_filepath": self.prediction_filepath,
                "pred_data_array": pred_data_array,
                "pred_frame_count": pred_frame_count
            }
            return pred_data_dict
        except Exception as e:
            raise RuntimeError(f"Error loading label data: {e}")