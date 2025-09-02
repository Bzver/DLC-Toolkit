import os
import shutil
from itertools import islice

import pandas as pd
import numpy as np

from typing import List, Tuple

from utils.dataclass import Loaded_DLC_Data, Export_Settings

def prediction_to_csv(
        dlc_data:Loaded_DLC_Data,
        pred_data_array:np.ndarray, 
        export_settings:Export_Settings,
        frame_list: List[int]=None
        ) -> bool:
    
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

def csv_to_h5(
        project_dir:str,
        multi_animal:bool,
        scorer:str="machine-labeled",
        csv_name:str="MachineLabelsRefine"
        ) -> bool:
    
    try:
        fn = os.path.join(project_dir, f"{csv_name}.csv")
        with open(fn) as datafile:
            total_lines = sum(1 for _ in datafile)
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
        
        expected_rows = total_lines - len(header)
        if len(data) == expected_rows - 1: # First nan frame is dropped, add it back
            print("Adding missing first frame with NaN values...")
            nan_row = pd.DataFrame(
                np.nan, index=[data.index[0] - 1] if index_col != False else [0], columns=data.columns
            )
            data = pd.concat([nan_row, data])
            data.sort_index(inplace=True)

        data.columns = data.columns.set_levels([f"{scorer}"], level="scorer")
        guarantee_multiindex_rows(data)
        data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
        data.to_csv(fn)
        return True
    except FileNotFoundError:
        print(f"Expected file: {csv_name}.csv not found in f{project_dir}!")

def construct_header_row(
        dlc_data:Loaded_DLC_Data,
        has_conf:bool=False
        ) -> Tuple[np.ndarray, List[str]]:
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