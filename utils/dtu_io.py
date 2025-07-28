import os
import shutil

import h5py
import yaml

import numpy as np
import pandas as pd
from itertools import islice

import cv2

from PySide6.QtWidgets import QMessageBox

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoadedDLCData:
    # DLC config data
    dlc_config_filepath: str # Full path of dlc_config file
    scorer: str
    multi_animal: bool
    keypoints: List[str]
    skeleton: List[List[str]]
    individuals: Optional[List[str]] # Will be None if not multi_animal, or a list of individual names
    instance_count: int
    num_keypoint: int

    # Prediction data
    prediction_filepath: str # Full path of prediction file
    pred_data_array: np.ndarray # Shape (pred_frame_count, instance_count, num_keypoint * 3)
    pred_frame_count: int

class DLC_Data_Loader:
    def __init__(self, parent, dlc_config_filepath, prediction_filepath):
        self.gui = parent
        self.dlc_config_filepath = dlc_config_filepath
        self.prediction_filepath = prediction_filepath
        self._multi_animal = False
        self._keypoints, self._skeleton, self._individuals, = None, None, None
        self._scorer, self._num_keypoint, self._instance_count = None, None, None
        self._pred_frame_count, self._pred_data_array = None, None

    def dlc_config_loader(self) -> bool:
        """
        Internal method to load DLC configuration.
        Populates internal attributes. Returns True on success, False on failure.
        """
        if not os.path.isfile(self.dlc_config_filepath):
            QMessageBox.warning(self.gui, "DLC Config Not Found", f"DLC config cannot be fetched from given path: {self.dlc_config_filepath}")
            return False
        try:
            with open(self.dlc_config_filepath, "r") as conf:
                cfg = yaml.safe_load(conf)
            self._scorer = cfg.get("scorer", None)
            self._multi_animal = cfg.get("multianimalproject", False)
            self._keypoints = cfg.get("bodyparts", []) if not self._multi_animal else cfg.get("multianimalbodyparts", [])
            self._skeleton = cfg.get("skeleton", [])
            self._individuals = cfg.get("individuals") # Can be None if not multi_animal
            self._instance_count = len(self._individuals) if self._individuals is not None else 1
            self._num_keypoint = len(self._keypoints)
            return True
        except Exception as e:
            QMessageBox.critical(self.gui, "DLC Config Error", f"Error loading DLC config: {e}")
            return False

    def prediction_loader(self) -> bool:
        """
        Internal method to load prediction data from HDF5 file.
        Populates internal attributes. Returns True on success, False on failure.
        """
        if not os.path.isfile(self.prediction_filepath):
            QMessageBox.warning(self.gui, "Prediction File Not Found", f"Prediction file cannot be fetched from given path: {self.prediction_filepath}")
            return False

        if not self.dlc_config_filepath or not self._num_keypoint or not self._instance_count:
            QMessageBox.warning(self.gui, "No DLC Config", "No DLC config has been loaded, please load it first.")
            return False

        try:
            with h5py.File(self.prediction_filepath, "r") as pred_file:
                if "tracks" not in pred_file: # Use 'in' for checking keys
                    QMessageBox.warning(self.gui, "Prediction File Error", "Error: Prediction file not valid, no 'tracks' key found.")
                    return False

                prediction_raw = pred_file["tracks"]["table"]
                pred_data_values = np.array([item[1] for item in prediction_raw])
                self._pred_frame_count = prediction_raw.size

                # Validate dimensions before creating array
                expected_cols = self._instance_count * self._num_keypoint * 3
                if pred_data_values.shape[1] != expected_cols:
                    QMessageBox.warning(self.gui, "Prediction Data Mismatch",
                        f"Prediction data columns ({pred_data_values.shape[1]}) do not match expected ({expected_cols}) based on config. Check config or prediction file."
                    )
                    return False

            self._pred_data_array = DLC_Data_Loader.unflatten_data_array(pred_data_values, self._instance_count)

            return True
        except Exception as e:
            QMessageBox.critical(self.gui, "Prediction Loading Error", f"Error loading prediction data: {e}")
            return False

    def get_loaded_dlc_data(self) -> Optional[LoadedDLCData]:
        """
        Exports the loaded and processed DLC data as a LoadedDLCData dataclass.
        Returns None if data could not be loaded or processed.
        """
        check_list = [self._keypoints, self._instance_count, self._pred_data_array, self._pred_frame_count]
        # Check if all necessary internal attributes are populated
        if any(item is None for item in check_list):
            error_msg = "Missing variables:"
            for item in check_list:
                if item is None:
                    error_msg += f" {item}"
            QMessageBox.warning(self.gui, "Data Incomplete", f"Internal data is incomplete. {error_msg}")
            return None

        return LoadedDLCData(
            dlc_config_filepath = self.dlc_config_filepath,
            scorer = self._scorer,
            multi_animal=self._multi_animal,
            keypoints=self._keypoints,
            skeleton=self._skeleton,
            individuals=self._individuals,
            instance_count=self._instance_count,
            num_keypoint=self._num_keypoint,
            prediction_filepath = self.prediction_filepath,
            pred_data_array=self._pred_data_array.copy(), # Return a copy of the numpy array
            pred_frame_count=self._pred_frame_count
        )
    
    @staticmethod
    def add_mock_confidence_score(data_array):
        array_dim = len(data_array.shape) # Always check for dimension first
        if array_dim == 2:
            rows, cols = data_array.shape
            new_array = np.full((rows, cols // 2 * 3), np.nan)
            new_array[:,0::3] = data_array[:,0::2]
            new_array[:,1::3] = data_array[:,1::2]

            x_nan_mask = np.isnan(new_array[:, 0::3])
            y_nan_mask = np.isnan(new_array[:, 1::3])
            xy_not_nan_mask = ~(x_nan_mask | y_nan_mask)
            new_array[:, 2::3][xy_not_nan_mask] = 1.0

        if array_dim == 3: # Unflattened (frame_idx, instance, bodyparts)
            dim_1, dim_2, dim_3 = data_array.shape
            new_array = np.full((dim_1, dim_2, dim_3 // 2 * 3), np.nan)
            new_array[:,:,0::3] = data_array[:,:,0::2]
            new_array[:,:,1::3] = data_array[:,:,1::2]

            x_nan_mask = np.isnan(new_array[:, :, 0::3])
            y_nan_mask = np.isnan(new_array[:, :, 1::3])
            xy_not_nan_mask = ~(x_nan_mask | y_nan_mask)
            new_array[:, :, 2::3][xy_not_nan_mask] = 1.0

        return new_array
    
    @staticmethod
    def unflatten_data_array(data_array, inst_count:int):
        rows, cols = data_array.shape
        new_array = np.full((rows, inst_count, cols // inst_count), np.nan)

        for inst_idx in range(inst_count):
            start_col = inst_idx * cols // inst_count
            end_col = (inst_idx + 1) * cols // inst_count
            new_array[:, inst_idx, :] = data_array[:, start_col:end_col]
        return new_array
    
    def remove_mock_confidence_score(data_array):
        array_dim = len(data_array.shape) # Always check for dimension first
        if array_dim == 2:
            rows, cols = data_array.shape
            new_array = np.full((rows, cols // 3 * 2), np.nan)
            new_array[:,0::2] = data_array[:,0::3]
            new_array[:,1::2] = data_array[:,1::3]
        if array_dim == 3: # Unflattened (frame_idx, instance, bodyparts)
            dim_1, dim_2, dim_3 = data_array.shape
            new_array = np.full((dim_1, dim_2, dim_3 // 3 * 2), np.nan)
            new_array[:,:,0::2] = data_array[:,:,0::3]
            new_array[:,:,1::2] = data_array[:,:,1::3]
        return new_array

class DLC_Exporter:
    def __init__(self, video_file, video_name, frame_list, dlc_data, project_dir):
        self.video_file = video_file
        self.video_name = video_name
        self.frame_list = frame_list
        self.project_dir = project_dir

        self.dlc_data = dlc_data
        self.prediction = dlc_data.prediction_filepath
        self.pred_data_array = dlc_data.pred_data_array.copy()

    def extract_frame_and_label(self):
        if not os.path.isfile(self.video_file):
            print(f"Original video not found at {self.video_file}")
            return False
        elif not os.path.isfile(self.prediction):
            print(f"Prediction file not found at {self.prediction}")
            return False
        else:
            frame_extraction = DLC_Exporter.extract_frame(self.video_file, self.frame_list, self.project_dir)
            label_extraction = self.extract_label()
            if frame_extraction and label_extraction:
                print("Extraction successful.")
                return True
            else:
                fail_message = ""
                fail_message += " extract frame" if not frame_extraction else ""
                fail_message += " extract label" if not label_extraction else ""
                print(f"Extraction failed. Error:{fail_message}")

    def extract_label(self):
        pred_data_array_filtered = self.pred_data_array[self.frame_list, :, :]
        
        if not DLC_Exporter.prediction_to_csv(self.dlc_data, pred_data_array_filtered, self.project_dir, marked_frames=self.frame_list, src_video_name=self.video_name):
            print("Error exporting predictions to csv.")
            return False
        else:
            print("Prediction of selected frames successfully exported to csv.")
        if not DLC_Exporter.csv_to_h5(self.project_dir, self.dlc_data.multi_animal):
            print("Error transforming to h5.")
            return False
        else:
            print("Exported csv transformed into h5.")
        return True
    
    @staticmethod
    def extract_frame(video_file, frame_list, export_dir):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            return False
        
        frames_to_extract = set(frame_list)

        for frame in frames_to_extract:
            image_path = f"img{str(int(frame)).zfill(8)}.png"
            image_output_path = os.path.join(export_dir, image_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_output_path, frame)
        return True

    @staticmethod
    def prediction_to_csv(dlc_data, pred_data_array, save_path:str, marked_frames:List=None,
        src_video_name:str=None, prediction_filename:str="MachineLabelsRefine"):
        
        pred_data_flattened = pred_data_array.reshape(pred_data_array.shape[0], -1)

        if pred_data_flattened.shape[1] // dlc_data.num_keypoint == 3 * dlc_data.instance_count:
            has_conf = True
        elif pred_data_flattened.shape[1] // dlc_data.num_keypoint == 2 * dlc_data.instance_count:
            has_conf = False
        else:
            print(f"Pred data has incomplatible shape: {pred_data_array.shape}")
            return False

        if not marked_frames:
            frame_list = list(range(pred_data_flattened.shape[0]))
        else:
            frame_list = marked_frames
        frame_col = np.array(frame_list).reshape(-1, 1)
        pred_data_processed = np.concatenate((frame_col, pred_data_flattened), axis=1)

        header_df, columns = DLC_Exporter.construct_header_row(dlc_data, has_conf)

        labels_df = pd.DataFrame(pred_data_processed, columns=columns)
        if marked_frames and src_video_name: # Skip it when called from the refiner <- no marked frames provided
            labels_df["frame"] = labels_df["frame"].apply(
                lambda x: (
                    f"labeled-data/{src_video_name}/"
                    f"img{str(int(x)).zfill(8)}.png"
                )
            )
        labels_df = labels_df.groupby("frame", as_index=False).first()

        final_df = pd.concat([header_df, labels_df], ignore_index=True)
        final_df.columns = [None] * len(final_df.columns)

        save_filepath = os.path.join(save_path, f"{prediction_filename}.csv")

        if os.path.isfile(save_filepath): #Backup the target file if it exists
            backup_idx = 0
            backup_dir = os.path.join(save_path, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            backup_filepath = os.path.join(backup_dir, f"{prediction_filename}_backup{backup_idx}.csv")

            while os.path.isfile(backup_filepath):
                backup_idx += 1
                backup_filepath = os.path.join(backup_dir, f"{prediction_filename}_backup{backup_idx}.csv")

            shutil.copy(save_filepath , backup_filepath)

        final_df.to_csv(save_filepath, index=False, header=None)
        return True

    @staticmethod
    def csv_to_h5(project_dir:str, multi_animal:bool, scorer="machine-labeled", csv_name="MachineLabelsRefine"):  # Adapted from deeplabcut.utils.conversioncode
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
            DLC_Exporter.guarantee_multiindex_rows(data)
            data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
            data.to_csv(fn)
            return True
        except FileNotFoundError:
            print("Attention:", project_dir, "does not appear to have labeled data!")

    @staticmethod
    def construct_header_row(dlc_data, has_conf):
        keypoints = dlc_data.keypoints
        num_keypoint = dlc_data.num_keypoint
        instance_count = dlc_data.instance_count
        individuals = dlc_data.individuals

        columns = ["frame"]

        bodyparts_row = ["bodyparts"]
        coords_row = ["coords"]

        if has_conf:
            if not dlc_data.multi_animal:
                columns += ([f"{kp}_x" for kp in keypoints] + [f"{kp}_y" for kp in keypoints] + [f"{kp}_likelihood" for kp in keypoints])
                bodyparts_row += [ f"{kp}" for kp in keypoints for _ in (0, 1, 2) ]
                coords_row += (["x", "y", "likelihood"] * keypoints)
            else:
                individuals_row = ["individuals"]
                if not individuals: # Fallback for the event that individual is empty
                    individuals = [str(k) for k in range(1,instance_count+1)]
                for m in range(instance_count):
                    columns += ([f"{kp}_x" for kp in keypoints] + [f"{kp}_y" for kp in keypoints] + [f"{kp}_likelihood" for kp in keypoints])
                    bodyparts_row += [ f"{kp}" for kp in keypoints for _ in (0, 1, 2) ]
                    coords_row += (["x", "y", "likelihood"] * num_keypoint)
                    for _ in range(num_keypoint * 3):
                        individuals_row += [individuals[m]]
        else:
            if not dlc_data.multi_animal:
                columns += ([f"{kp}_x" for kp in keypoints] + [f"{kp}_y" for kp in keypoints])
                bodyparts_row += [ f"{kp}" for kp in keypoints for _ in (0, 1) ]
                coords_row += (["x", "y",] * keypoints)
            else:
                individuals_row = ["individuals"]
                if not individuals: # Fallback for the event that individual is empty
                    individuals = [str(k) for k in range(1,instance_count+1)]
                for m in range(instance_count):
                    columns += ([f"{kp}_x" for kp in keypoints] + [f"{kp}_y" for kp in keypoints])
                    bodyparts_row += [ f"{kp}" for kp in keypoints for _ in (0, 1) ]
                    coords_row += (["x", "y"] * num_keypoint)
                    for _ in range(num_keypoint * 2):
                        individuals_row += [individuals[m]]

        scorer_row = ["scorer"] + ["machine-labeled"] * (len(columns) - 1)
        header_df = pd.DataFrame(
        [row for row in [scorer_row, individuals_row, bodyparts_row, coords_row] if row != individuals_row or dlc_data.multi_animal],
            columns=columns
        )
        
        return header_df, columns

    @staticmethod
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

if __name__ == "__main__":
    folder = "D:/Project/DLC-Models/NTD/labeled-data/20250626C1-first3h-D"
    csv_name = "CollectedData_bezver"
    if DLC_Exporter.csv_to_h5(folder, True, scorer="bezver", csv_name=csv_name):
        print(f"Successfully transfer {csv_name}.csv into h5.")