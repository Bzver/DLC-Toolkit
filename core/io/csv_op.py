import os
from itertools import islice

import pandas as pd
import numpy as np

from typing import List, Tuple, Optional

from .io_helper import backup_existing_prediction, remove_confidence_score
from utils.logger import logger
from utils.dataclass import Loaded_DLC_Data


def prediction_to_csv(
        dlc_data:Loaded_DLC_Data,
        pred_data_array:np.ndarray, 
        save_path:str,
        frame_list: Optional[List[int]]=None,
        keep_conf:bool=False,
        to_dlc:bool=False,
        no_scorer_row=False
        ):
    
    pred_data_flattened = pred_data_array.reshape(pred_data_array.shape[0], -1) # [F, I, K] to [F, I*K]

    if pred_data_flattened.shape[1] // dlc_data.num_keypoint == 3 * dlc_data.instance_count and not keep_conf:
        pred_data_flattened = remove_confidence_score(pred_data_flattened)

    if not frame_list:
        frame_list = list(range(pred_data_flattened.shape[0]))

    frame_col = np.array(frame_list).reshape(-1, 1)
    pred_data_processed = np.concatenate((frame_col, pred_data_flattened), axis=1)

    header_df, columns = construct_header_row(dlc_data, keep_conf, no_scorer_row)

    labels_df = pd.DataFrame(pred_data_processed, columns=columns)

    if to_dlc:
        dir_pro = os.path.basename(os.path.dirname(save_path))
        labels_df["frame"] = labels_df["frame"].apply(
            lambda x: (
                f"labeled-data/{dir_pro}/"
                f"img{str(int(x)).zfill(8)}.png"
            )
        )
    labels_df = labels_df.groupby("frame", as_index=False).first()

    final_df = pd.concat([header_df, labels_df], ignore_index=True)
    final_df.columns = [None] * len(final_df.columns)
    backup_existing_prediction(save_path)

    final_df.to_csv(save_path, index=False, header=None)

def csv_to_h5(
        csv_path:str,
        multi_animal:bool,
        scorer:str="machine-labeled",
        ):
    with open(csv_path) as datafile:
        total_lines = sum(1 for _ in datafile)
    
    with open(csv_path) as datafile: # Reopen the file to read the head
        head = list(islice(datafile, 0, 5))
    if multi_animal:
        header = list(range(4))
    else:
        header = list(range(3))
        
    if head[-1].split(",")[0] == "labeled-data":
        index_col = [0, 1, 2]
    else:
        index_col = 0
    
    data = pd.read_csv(csv_path, index_col=index_col, header=header)
    
    expected_rows = total_lines - len(header)
    if len(data) == expected_rows - 1: # First nan frame is dropped, add it back
        logger.warning("Adding missing first frame with NaN values...")
        nan_row = pd.DataFrame(
            np.nan, index=[data.index[0] - 1] if index_col != False else [0], columns=data.columns
        )
        data = pd.concat([nan_row, data])
        data.sort_index(inplace=True)

    data.columns = data.columns.set_levels([f"{scorer}"], level="scorer")
    guarantee_multiindex_rows(data)
    data.to_hdf(csv_path.replace(".csv", ".h5"), key="df_with_missing", mode="w")
    data.to_csv(csv_path)

def construct_header_row(
        dlc_data:Loaded_DLC_Data,
        has_conf:bool=False,
        no_scorer_row:bool=False,
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

    if no_scorer_row:
        header_df = pd.DataFrame(
        [row for row in [individuals_row, bodyparts_row, coords_row] if row != individuals_row or dlc_data.multi_animal],
            columns=columns
        )
    else:
        header_df = pd.DataFrame(
        [row for row in [scorer_row, individuals_row, bodyparts_row, coords_row] if row != individuals_row or dlc_data.multi_animal],
            columns=columns
        )

    return header_df, columns

def guarantee_multiindex_rows(df): # Adopted from DeepLabCut
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