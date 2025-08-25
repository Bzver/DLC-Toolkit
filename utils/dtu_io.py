import os
import shutil

import h5py
import yaml

import numpy as np
import pandas as pd
from itertools import islice

import cv2

from typing import List, Tuple, Optional, Any

from .dtu_dataclass import Loaded_DLC_Data, Export_Settings
from . import dtu_helper as duh

import traceback

def prediction_to_csv(dlc_data:Loaded_DLC_Data, pred_data_array: np.ndarray, 
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
    elif export_settings.export_mode == "Merge":
        csv_name = f"CollectedData_{dlc_data.scorer}"
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
        print(f"Expected file: {csv_name}.csv not found in f{project_dir}!")

def construct_header_row(dlc_data:Loaded_DLC_Data, has_conf:bool=False) -> Tuple[np.ndarray, List[str]]:
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

def determine_save_path(prediction_filepath:str, suffix:str) -> str:
    pred_file_dir = os.path.dirname(prediction_filepath)
    pred_file_name_without_ext = os.path.splitext(os.path.basename(prediction_filepath))[0]
    
    if not suffix in pred_file_name_without_ext:
        save_idx = 0
        base_name = pred_file_name_without_ext
    else:
        base_name, save_idx_str = pred_file_name_without_ext.split(suffix)
        try:
            save_idx = int(save_idx_str) + 1
        except ValueError:
            save_idx = 0 # Fallback if suffix is malformed
    
    pred_file_to_save_path = os.path.join(pred_file_dir,f"{base_name}{suffix}{save_idx}.h5")

    shutil.copy(prediction_filepath, pred_file_to_save_path)
    print(f"Saved modified prediction to: {pred_file_to_save_path}")
    return pred_file_to_save_path

def convert_prediction_array_to_save_format(pred_data_array: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    new_data = []
    num_frames = pred_data_array.shape[0]

    for frame_idx in range(num_frames):
        frame_data = pred_data_array[frame_idx, :, :].flatten()
        new_data.append((frame_idx, frame_data))

    return new_data

def save_prediction_to_h5(prediction_filepath: str, pred_data_array: np.ndarray) -> Tuple[bool, str]:
    try:
        with h5py.File(prediction_filepath, "a") as pred_file:
            if 'tracks/table' in pred_file:
                pred_file['tracks/table'][...] = convert_prediction_array_to_save_format(pred_data_array)
            else:
                return False, f"No 'tracks' key in the {prediction_filepath}? How is it possible!!?"
        return True, f"Successfully saved prediction to {prediction_filepath}."
    except Exception as e:
        print(f"Error saving prediction to HDF5: {e}")
        traceback.print_exc()
        return False, e
    
def append_new_video_to_dlc_config(config_path, video_name):
    dlc_dir = os.path.dirname(config_path)
    config_backup = os.path.join(dlc_dir, "config_bak.yaml")
    print("Backup up the original config.yaml as config_bak.yaml")
    shutil.copy(config_path ,config_backup)

    video_filepath = os.path.join(dlc_dir, "videos", f"{video_name}.mp4")

    # Load original config
    with open(config_path, 'r') as f:
        try:
            config_org = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    if video_filepath in config_org["video_sets"]:
        print(f"Video {video_filepath} already exists in video_sets. Skipping update.")
        return

    with open(config_path, 'r') as f:
        config_org["video_sets"][video_filepath] = {"crop": "0, 0, 0, 0"}
        print("Appended new video_sets to the originals.")
    with open(config_path, 'w') as file:
        yaml.dump(config_org, file, default_flow_style=False, sort_keys=False)
        print(f"DeepLabCut config in {config_path} has been updated.")

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
                pred_header = ["tracks", "df_with_missing", "predictions"]
                found_keys = [key for key in pred_header if key in pred_file]
                if not found_keys:
                    return None, f"Error: Prediction file not valid, no key found. Acceptable keys: {pred_header} ."

                key = found_keys[-1]
                subkey = "block0_values" if key == "predictions" else "table"

                prediction_raw = pred_file[key][subkey]

                if key == "predictions": # Already an array
                    pred_data_values = np.array(prediction_raw)
                    pred_frame_count = prediction_raw.shape[0]
                else:
                    pred_data_values = np.array([item[1] for item in prediction_raw])
                    pred_frame_count = len(prediction_raw)

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
        frame_list: List[int], pred_data_array: np.ndarray=None):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array

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
        cap = cv2.VideoCapture(self.export_settings.video_filepath)
        if not cap.isOpened():
            return False, f"Error: Could not open video {self.export_settings.video_filepath}"
        
        frames_to_extract = set(self.frame_list)

        for frame in frames_to_extract:
            image_path = f"img{str(int(frame)).zfill(8)}.png"
            image_output_path = os.path.join(self.export_settings.save_path, image_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_output_path, frame)
        return True, "Success"

    def _extract_pred(self) -> Tuple[bool, str]:
        if self.pred_data_array is None:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
        else:
            pred_data_array = self.pred_data_array[self.frame_list, :, :]
        
        if not prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list):
            return False, "Error exporting predictions to csv."

        csv_name = f"CollectedData_{self.dlc_data.scorer}"
        if not csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal, self.dlc_data.scorer, csv_name=csv_name):
            return False, "Error transforming to h5."
        
        return True, f"Label extracted to {self.export_settings.save_path}"