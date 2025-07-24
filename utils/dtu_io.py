import os

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
    def __init__(self, parent, dlc_config_filepath, prediction_filepath, initialize_status=False):
        self.gui = parent
        self.dlc_config_filepath = dlc_config_filepath
        self.prediction_filepath = prediction_filepath
        self._is_initializing = initialize_status
        self._multi_animal = False
        self._keypoints, self._skeleton, self._individuals, = None, None, None
        self._num_keypoint, self._instance_count = None, None
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
            self._multi_animal = cfg.get("multianimalproject", False) # Use .get with default
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

        if not self.dlc_config_filepath:
            QMessageBox.warning(self.gui, "No DLC Config", "No DLC config has been loaded, please load it first.")
            return False

        try:
            with h5py.File(self.prediction_filepath, "r") as pred_file:
                if "tracks" not in pred_file: # Use 'in' for checking keys
                    QMessageBox.warning(self.gui, "Prediction File Error", "Error: Prediction file not valid, no 'tracks' key found.")
                    return False

                if self._is_initializing:
                    QMessageBox.information(self.gui, "Loading Prediction","Loading and parsing prediction file, this could take a few seconds, please wait...")

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

                self._pred_data_array = np.full(
                    (self._pred_frame_count, self._instance_count, self._num_keypoint * 3), np.nan
                )

                for inst_idx in range(self._instance_count):
                    start_col = inst_idx * self._num_keypoint * 3
                    end_col = (inst_idx + 1) * self._num_keypoint * 3
                    self._pred_data_array[:, inst_idx, :] = pred_data_values[:, start_col:end_col]

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

class DLC_Frame_Extractor:  # Backend for extracting frames for labeling in DLC
    def __init__(self, video_file, prediction, frame_list, dlc_dir, project_dir, video_name, pred_data, keypoints, individuals, multi_animal=False):
        self.video_file = video_file
        self.prediction = prediction
        self.frame_list = frame_list
        self.dlc_dir = dlc_dir
        self.project_dir = project_dir
        self.video_name = video_name
        self.multi_animal = multi_animal
        self.pred_data = pred_data
        self.keypoints = keypoints
        self.individuals = individuals

        self.data = []

    def extract_frame_and_label(self):
        if not os.path.isfile(self.video_file):
            print(f"Original video not found at {self.video_file}")
            return False
        elif not os.path.isfile(self.prediction):
            print(f"Prediction file not found at {self.prediction}")
            return False
        else:
            frame_extraction = self.extract_frame()
            label_extraction = self.extract_label()
            if frame_extraction and label_extraction:
                print("Extraction successful.")
                return True
            else:
                fail_message = ""
                fail_message += " extract frame" if not frame_extraction else ""
                fail_message += " extract label" if not label_extraction else ""
                print(f"Extraction failed. Error:{fail_message}")
        
    def extract_frame(self):
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_file}")
            return False
        
        frames_to_extract = set(self.frame_list)

        for frame in frames_to_extract:
            image_path = f"img{str(int(frame)).zfill(8)}.png"
            image_output_path = os.path.join(self.project_dir,image_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_output_path, frame)
        return True

    def extract_label(self):
        for frame in self.frame_list:
            frame_idx = self.pred_data[frame][0]
            frame_data = self.pred_data[frame][1]
            filtered_frame_data = [val for i, val in enumerate(frame_data) if (i % 3 != 2)] # Remove likelihood
            self.data.append([frame_idx] + filtered_frame_data)

        if not self.prediction_to_csv():
            print("Error exporting predictions to csv.")
            return False
        else:
            print("Prediction of selected frames successfully exported to csv.")
        if not self.csv_to_h5():
            print("Error transforming to h5.")
            return False
        else:
            print("Exported csv transformed into h5.")
        return True

    def prediction_to_csv(self): # Adapted from agosztolai's pull request: https://github.com/DeepLabCut/DeepLabCut/pull/2977
        data_frames = []
        columns = ["frame"]
        bodyparts_row = ["bodyparts"]
        coords_row = ["coords"]

        num_keypoints = len(self.keypoints)

        if not self.multi_animal:
            columns += ([f"{kp}_x" for kp in self.keypoints] + [f"{kp}_y" for kp in self.keypoints])
            bodyparts_row += [ f"{kp}" for kp in self.keypoints for _ in (0, 1) ]
            coords_row += (["x", "y"] * self.keypoints)
            max_instances = 1
        else:
            individuals_row = ["individuals"]
            max_instances = len(self.individuals)
            self.individuals = [str(k) for k in range(1,max_instances+1)]
            for m in range(max_instances):
                columns += ([f"{kp}_x" for kp in self.keypoints] + [f"{kp}_y" for kp in self.keypoints])
                bodyparts_row += [ f"{kp}" for kp in self.keypoints for _ in (0, 1) ]
                coords_row += (["x", "y"] * num_keypoints)
                for _ in range(num_keypoints*2):
                    individuals_row += [self.individuals[m]]
        scorer_row = ["scorer"] + ["machine-labeled"] * (len(columns) - 1)

        labels_df = pd.DataFrame(self.data, columns=columns)
        labels_df["frame"] = labels_df["frame"].apply(
            lambda x: (
                f"labeled-data/{self.video_name}/"
                f"img{str(int(x)).zfill(8)}.png"
            )
        )
        labels_df = labels_df.groupby("frame", as_index=False).first()
        data_frames.append(labels_df)
        combined_df = pd.concat(data_frames, ignore_index=True)

        header_df = pd.DataFrame(
        [row for row in [scorer_row, individuals_row, bodyparts_row, coords_row] if row != individuals_row or self.multi_animal],
            columns=combined_df.columns
        )

        final_df = pd.concat([header_df, combined_df], ignore_index=True)
        final_df.columns = [None] * len(final_df.columns)

        final_df.to_csv(
            os.path.join(self.project_dir, f"MachineLabelsRefine.csv"),
            index=False,
            header=None,
        )
        return True

    def csv_to_h5(self):  # Adapted from deeplabcut.utils.conversioncode
        try:
            fn = os.path.join(self.project_dir, f"MachineLabelsRefine.csv")
            with open(fn) as datafile:
                head = list(islice(datafile, 0, 5))
            if self.multi_animal:
                header = list(range(4))
            else:
                header = list(range(3))
            if head[-1].split(",")[0] == "labeled-data":
                index_col = [0, 1, 2]
            else:
                index_col = 0
            data = pd.read_csv(fn, index_col=index_col, header=header)
            data.columns = data.columns.set_levels(["machine-labeled"], level="scorer")
            self.guarantee_multiindex_rows(data)
            data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
            data.to_csv(fn)
            return True
        except FileNotFoundError:
            print("Attention:", self.project_dir, "does not appear to have labeled data!")

    def guarantee_multiindex_rows(self, df): # Adapted from DeepLabCut
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