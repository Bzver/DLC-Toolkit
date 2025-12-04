import os
import numpy as np

import h5py
import yaml

from typing import Any, Dict, Optional

from utils.logger import logger

from .h5_op import validate_h5_keys, fix_h5_kp_order
from .io_helper import (
    unflatten_data_array, add_mock_confidence_score, nuke_negative_val_in_loaded_pred, load_crop_notations)
from utils.dataclass import Loaded_DLC_Data

class Prediction_Loader:
    def __init__(
            self,
            dlc_config_filepath: str,
            prediction_filepath:Optional[str]=None):
        logger.info(f"[PLOADER] Initializing Prediction_Loader with config: {dlc_config_filepath}, prediction: {prediction_filepath}")
        
        self.dlc_config_filepath = dlc_config_filepath
        self.prediction_filepath = prediction_filepath
        self.crop_dict = None

        if self.prediction_filepath:
            project_dir = os.path.dirname(self.prediction_filepath)
            crop_notation_filepath = os.path.join(project_dir, "crop.yaml")
            if os.path.isfile(crop_notation_filepath):
                self.crop_dict = load_crop_notations(crop_notation_filepath)
                logger.debug(f"[PLOADER] Loaded crop notations from {crop_notation_filepath}")
            else:
                logger.debug(f"[PLOADER] No crop.yaml found at {crop_notation_filepath}")
        logger.info("[PLOADER] Prediction_Loader initialization complete.")

    def load_data(self, metadata_only: bool = False, force_load_pred:bool = False) -> Loaded_DLC_Data:
        logger.info(f"[PLOADER] Loading data (metadata_only: {metadata_only}, force_load_pred: {force_load_pred}).")
        config_data = self._load_config_data()
        
        if metadata_only:
            logger.debug("[PLOADER] Loading metadata only.")
            loaded_data = Loaded_DLC_Data(
                **config_data,
                prediction_filepath=None,
                pred_data_array=None,
                pred_frame_count=None
            )
            logger.info("[PLOADER] Metadata loaded successfully.")
            return loaded_data

        if os.path.basename(self.prediction_filepath).startswith("CollectedData_") and not force_load_pred:
            logger.debug("[PLOADER] Loading labeled data.")
            pred_data = self._load_labeled_data(config_data)
        else:
            logger.debug("[PLOADER] Loading prediction data.")
            pred_data = self._load_prediction_data(config_data)
        
        # Merge dictionaries for a clean object creation
        loaded_data = Loaded_DLC_Data(**config_data, **pred_data)
        logger.info("[PLOADER] All data loaded successfully.")
        return loaded_data

    def _load_config_data(self) -> Dict[str, Any]:
        """Internal method to load DLC configuration from the YAML file."""
        logger.info(f"[PLOADER] Loading config data from {self.dlc_config_filepath}")
        if not os.path.isfile(self.dlc_config_filepath):
            raise FileNotFoundError(f"DLC config file not found at: {self.dlc_config_filepath}")
        
        try:
            with open(self.dlc_config_filepath, "r") as conf:
                cfg = yaml.safe_load(conf)
            logger.debug(f"[PLOADER] Successfully loaded YAML config from {self.dlc_config_filepath}")
            
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
            logger.info("[PLOADER] Config data parsed successfully.")
            logger.debug(f"[PLOADER] Config: {config_dict}")
            return config_dict
        except Exception as e:
            raise RuntimeError(f"Error loading DLC config: {e}") from e

    def _load_prediction_data(self, config_data:Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to load prediction data from HDF5 file."""
        logger.info(f"[PLOADER] Loading prediction data from {self.prediction_filepath}")
        if not os.path.isfile(self.prediction_filepath):
            raise FileNotFoundError(f"Prediction file not found at: {self.prediction_filepath}")

        try:
            with h5py.File(self.prediction_filepath, "r") as pred_file:
                key, subkey = validate_h5_keys(pred_file)
                if not key or not subkey:
                    logger.error(f"[PLOADER] No valid key found in '{self.prediction_filepath}'")
                    raise ValueError(f"No valid key found in '{self.prediction_filepath}'")
                logger.debug(f"[PLOADER] HDF5 keys found: key={key}, subkey={subkey}")
                
                prediction_raw = pred_file[key][subkey]
                
                instance_count = config_data["instance_count"]
                num_keypoint = config_data["num_keypoint"]
                logger.debug(f"[PLOADER] Instance count: {instance_count}, Num keypoints: {num_keypoint}")

                if subkey == "block0_values": # Already an array
                    pred_data_values = fix_h5_kp_order(pred_file, key, config_data)
                    pred_frame_count = pred_data_values.shape[0]
                    logger.debug(f"[PLOADER] Loaded 'block0_values'. Frame count: {pred_frame_count}")
                else:
                    pred_data_values = np.array([item[1] for item in prediction_raw])
                    pred_frame_count = len(prediction_raw)
                    logger.debug(f"[PLOADER] Loaded raw prediction data. Frame count: {pred_frame_count}")

                expected_cols = instance_count * num_keypoint * 3
                half_expected_cols = instance_count * num_keypoint * 2
                logger.debug(f"[PLOADER] Expected columns: {expected_cols}, Half expected columns: {half_expected_cols}")

                if pred_data_values.shape[1] != expected_cols:
                    if pred_data_values.shape[1] == half_expected_cols:
                        logger.warning("[PLOADER] Prediction data missing confidence scores. Adding mock scores.")
                        pred_data_values = add_mock_confidence_score(pred_data_values)
                    else:
                        raise ValueError(
                            f"Prediction data has {pred_data_values.shape[1]} columns, "
                            f"but {expected_cols} columns were expected based on config "
                            f"(instances={instance_count}, keypoints={num_keypoint}). "
                            "Please verify the prediction file and configuration match."
                        )
                logger.debug(f"[PLOADER] Prediction data shape after validation: {pred_data_values.shape}")

            pred_data_array = unflatten_data_array(pred_data_values, instance_count)
            pred_data_array = nuke_negative_val_in_loaded_pred(pred_data_array)
            logger.debug("[PLOADER] Prediction data unflattened and negative values nuked.")

            pred_data_dict = {
                "prediction_filepath": self.prediction_filepath,
                "pred_data_array": pred_data_array,
                "pred_frame_count": pred_frame_count
            }
            logger.info("[PLOADER] Prediction data loaded and processed successfully.")
            return pred_data_dict
        except Exception as e:
            raise RuntimeError(f"Error loading prediction data: {e}")
        
    def _load_labeled_data(self, config_data:Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to load label data from CollectedData_*.h5 file."""
        logger.info(f"[PLOADER] Loading labeled data from {self.prediction_filepath}")
        if not os.path.isfile(self.prediction_filepath):
            raise FileNotFoundError(f"Prediction file not found at: {self.prediction_filepath}")
        
        keypoints = config_data["keypoints"]
        instance_count = config_data["instance_count"]
        logger.debug(f"[PLOADER] Keypoints: {len(keypoints)}, Instance count: {instance_count}")

        try:
            with h5py.File(self.prediction_filepath, "r") as lbf:
                key, subkey = validate_h5_keys(lbf)
                if not key or not subkey:
                    raise ValueError(f"No valid key found in '{self.prediction_filepath}'")
                logger.debug(f"[PLOADER] HDF5 keys found for labeled data: key={key}, subkey={subkey}")
                
                num_keypoint = len(keypoints)

                if subkey == "block0_values": # Already an array
                    pred_data_values = fix_h5_kp_order(lbf, key, config_data)
                    pred_frame_count = pred_data_values.shape[0]
                    logger.debug(f"[PLOADER] Loaded 'block0_values' for labeled data. Frame count: {pred_frame_count}")
                else:
                    logger.error(f"[PLOADER] 'block0_values' not found in labeled HDF5: {self.prediction_filepath}")
                    raise ValueError("'block0_values' not found in labeled HDF5.")
                
                expected_cols_labeled = instance_count * num_keypoint * 2
                logger.debug(f"[PLOADER] Expected columns for labeled data: {expected_cols_labeled}")

                if pred_data_values.shape[1] == expected_cols_labeled:
                    logger.debug("[PLOADER] Labeled data missing confidence scores. Adding mock scores.")
                    pred_data_values = add_mock_confidence_score(pred_data_values)
                    pred_data_unflattened = unflatten_data_array(pred_data_values, instance_count)
                else:
                    raise ValueError(
                        f"Label data has {pred_data_values.shape[1]} columns, "
                        f"but {expected_cols_labeled} columns were expected based on config "
                        f"(instances={instance_count}, keypoints={num_keypoint}). "
                        "Please verify the label data file and configuration match."
                    )
                logger.debug(f"[PLOADER] Labeled data shape after validation: {pred_data_values.shape}")

                if "axis1_level2" in lbf[key]:
                    labeled_frame_list = lbf[key]["axis1_level2"].asstr()[()]
                    labeled_frame_list = [int(f.split("img")[1].split(".")[0]) for f in labeled_frame_list]
                    labeled_frame_list.sort()
                    pred_frame_count = max(labeled_frame_list) + 1
                    logger.debug(f"[PLOADER] Labeled frame list loaded. Max frame: {pred_frame_count-1}")
                
                    pred_data_array = np.full((pred_frame_count, instance_count, num_keypoint*3),np.nan)
                    pred_data_array[labeled_frame_list, ...] = pred_data_unflattened
                    pred_data_array = nuke_negative_val_in_loaded_pred(pred_data_array)
                    logger.debug("[PLOADER] Labeled data array constructed and negative values nuked.")
                else:
                    raise ValueError("'axis1_level2' not found in labeled HDF5.")

            if self.crop_dict:
                logger.debug("[PLOADER] Applying crop notations to labeled data.")
                for crop_coord, frame_list in self.crop_dict.items():
                    pred_data_array[frame_list, :, 0::3] += crop_coord[0]
                    pred_data_array[frame_list, :, 1::3] += crop_coord[1]
                logger.debug("[PLOADER] Crop notations applied.")

            pred_data_dict = {
                "prediction_filepath": self.prediction_filepath,
                "pred_data_array": pred_data_array,
                "pred_frame_count": pred_frame_count
            }
            logger.info("[PLOADER] Labeled data loaded and processed successfully.")
            return pred_data_dict
        except Exception as e:
            raise RuntimeError(f"Error loading label data: {e}")