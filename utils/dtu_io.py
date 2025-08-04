import os
import shutil

import h5py
import yaml

import numpy as np
import pandas as pd
from itertools import islice

import cv2

from typing import List, Tuple, Optional, Any
from numpy.typing import NDArray

from .dtu_dataclass import Loaded_DLC_Data, Export_Settings
from . import dtu_helper as duh

def prediction_to_csv(dlc_data:Loaded_DLC_Data, pred_data_array: NDArray, 
        export_settings: Export_Settings, frame_list: List[int]=None) -> bool:
    
    pred_data_flattened = pred_data_array.reshape(pred_data_array.shape[0], -1)

    if pred_data_flattened.shape[1] // dlc_data.num_keypoint == 3 * dlc_data.instance_count:
        has_conf = True
    elif pred_data_flattened.shape[1] // dlc_data.num_keypoint == 2 * dlc_data.instance_count:
        has_conf = False
    else:
        print(f"Pred data has incomplatible shape: {pred_data_array.shape}")
        return False

    if not frame_list:
        frame_list = list(range(pred_data_flattened.shape[0]))

    frame_col = np.array(frame_list).reshape(-1, 1)
    pred_data_processed = np.concatenate((frame_col, pred_data_flattened), axis=1)

    header_df, columns = construct_header_row(dlc_data, has_conf)

    labels_df = pd.DataFrame(pred_data_processed, columns=columns)

    if export_settings.export_mode != "CSV":
        labels_df["frame"] = labels_df["frame"].apply(
            lambda x: (
                f"labeled-data/{export_settings.video_name}/"
                f"img{str(int(x)).zfill(8)}.png"
            )
        )
    labels_df = labels_df.groupby("frame", as_index=False).first()

    final_df = pd.concat([header_df, labels_df], ignore_index=True)
    final_df.columns = [None] * len(final_df.columns)

    if export_settings.export_mode == "Append":
        csv_name = "MachineLabelsRefine"
    else:
        prediction_filename = os.path.basename(dlc_data.prediction_filepath)
        csv_name = prediction_filename.split(".h5")[0]

    save_filepath = os.path.join(export_settings.save_path, f"{csv_name}.csv")

    if os.path.isfile(save_filepath):
        backup_idx = 0
        backup_dir = os.path.join(export_settings.save_path, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_filepath = os.path.join(backup_dir, f"{csv_name}_backup{backup_idx}.csv")

        while os.path.isfile(backup_filepath):
            backup_idx += 1
            backup_filepath = os.path.join(backup_dir, f"{csv_name}_backup{backup_idx}.csv")

        shutil.copy(save_filepath , backup_filepath)

    final_df.to_csv(save_filepath, index=False, header=None)
    return True

def csv_to_h5(project_dir:str, multi_animal:bool, scorer:str="machine-labeled", csv_name:str="MachineLabelsRefine") -> bool:
    try:
        fn = os.path.join(project_dir, f"{csv_name}.csv")
        with open(fn) as datafile:
            head = list(islice(datafile, 0, 5))
        if multi_animal:
            header = list(range(4))
        else:
            header = list(range(3))
        if head[-1].split(",")[0] == "labeled-data":
            index_col = [0, 1, 2]
        else:
            index_col = 0
        data = pd.read_csv(fn, index_col=index_col, header=header)
        data.columns = data.columns.set_levels([f"{scorer}"], level="scorer")
        guarantee_multiindex_rows(data)
        data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
        data.to_csv(fn)
        return True
    except FileNotFoundError:
        print("Attention:", project_dir, "does not appear to have labeled data!")

def construct_header_row(dlc_data:Loaded_DLC_Data, has_conf:bool=False) -> Tuple[NDArray, List[str]]:
    keypoints = dlc_data.keypoints
    num_keypoint = dlc_data.num_keypoint
    instance_count = dlc_data.instance_count
    individuals = dlc_data.individuals

    columns = ["frame"]

    bodyparts_row = ["bodyparts"]
    coords_row = ["coords"]

    if has_conf:
        suffixes = ["_x", "_y", "_likelihood"]
        coords = ["x", "y", "likelihood"]
        count = 3
    else:
        suffixes = ["_x", "_y"]
        coords = ["x", "y"]
        count = 2

    if dlc_data.multi_animal:
        if not individuals:
            individuals = [str(k) for k in range(1, instance_count + 1)]
        individuals_row = ["individuals"]
    
        for m in range(instance_count):
            columns += [f"{kp}{s}" for kp in keypoints for s in suffixes]
            bodyparts_row += [kp for kp in keypoints for _ in range(count)]
            coords_row += coords * num_keypoint
            individuals_row += [individuals[m]] * (num_keypoint * count)
    else:
        columns += [f"{kp}{s}" for kp in keypoints for s in suffixes]
        bodyparts_row += [kp for kp in keypoints for _ in range(count)]
        coords_row += coords * num_keypoint

    scorer_row = ["scorer"] + ["machine-labeled"] * (len(columns) - 1)
    header_df = pd.DataFrame(
    [row for row in [scorer_row, individuals_row, bodyparts_row, coords_row] if row != individuals_row or dlc_data.multi_animal],
        columns=columns
    )
    
    return header_df, columns

def guarantee_multiindex_rows(df): # Adapted from DeepLabCut
    # Make paths platform-agnostic if they are not already
    if not isinstance(df.index, pd.MultiIndex):  # Backwards compatibility
        path = df.index[0]
        try:
            sep = "/" if "/" in path else "\\"
            splits = tuple(df.index.str.split(sep))
            df.index = pd.MultiIndex.from_tuples(splits)
        except TypeError:  #  Ignore numerical index of frame indices
            pass
    # Ensure folder names are strings
    try:
        df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
    except AttributeError:
        pass

######################################################################################################################################

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
    
######################################################################################################################################
    
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
        
        if not prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list):
            return False, "Error exporting predictions to csv."

        if not csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal):
            return False, "Error transforming to h5."
        
        return True, f"Label extracted to {self.export_settings.save_path}"