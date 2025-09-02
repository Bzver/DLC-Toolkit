import pandas as pd
import numpy as np

from typing import List, Tuple

def parse_idt_df_into_ndarray(
        df_idtracker:pd.DataFrame,
        df_confidence:pd.DataFrame,
        confidence_threshold:float=0.5
        ) ->np.ndarray:
    """
    Convert a DataFrame with idtracker.ai-like format to a 3D numpy array.
    
    Input: 
        df_idtracker: pd.DataFrame with columns ("time", "x1", "y1", "x2", "y2", ....
        df_confidence: pd.DataFrame with columns ("time", "id_probabilities1", "id_probabilities2", ...)
        confidence_threshold: float, minimum confidence value to keep coordinates (default: 0.5)
    
    Output: 
        np.ndarray of shape (num_frames, num_individuals, 2), last dim: x,y
    """
    df_idt = df_idtracker.drop(columns="time")
    df_conf = df_confidence.drop(columns="time")

    x_cols = sorted([col for col in df_idt.columns if col.startswith('x')], key=lambda x: int(x[1:]))
    y_cols = sorted([col for col in df_idt.columns if col.startswith('y')], key=lambda x: int(x[1:]))
    conf_cols = sorted(df_conf.columns, key=lambda x: int(x.split('id_probabilities')[1]))

    assert len(x_cols) == len(y_cols), "Mismatch between x and y coordinate columns"
    assert len(x_cols) == len(conf_cols), "Mismatch between coordinate and confidence columns"
    assert len(df_idt) == len(df_conf), "Mismatch in number of frames between coordinate and confidence dataframes"

    num_frames = len(df_idt)
    num_individuals = len(x_cols)
    coords_array = np.full((num_frames, num_individuals, 2), np.nan)

    for i, (x_col, y_col, conf_col) in enumerate(zip(x_cols, y_cols, conf_cols)):
        confidence_values = df_conf[conf_col].values
        coords_array[:, i, 0] = df_idt[x_col].values  # x coordinates
        coords_array[:, i, 1] = df_idt[y_col].values  # y coordinates

        mask = confidence_values >= confidence_threshold
        coords_array[~mask, i, :] = np.nan

    return coords_array

def convert_prediction_array_to_save_format(pred_data_array: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    new_data = []
    num_frames = pred_data_array.shape[0]

    for frame_idx in range(num_frames):
        frame_data = pred_data_array[frame_idx, :, :].flatten()
        new_data.append((frame_idx, frame_data))

    return new_data