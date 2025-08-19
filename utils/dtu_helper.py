import pandas as pd
import numpy as np
import bisect

from typing import List, Optional, Tuple
from .dtu_dataclass import Loaded_DLC_Data, Swap_Calculation_Config

def format_title(base_title: str, debug_status: bool) -> str:
    if debug_status:
        return f"{base_title} --- DEBUG MODE"
    return base_title

def get_config_from_calculation_mdode(mode:str, frame_idx:int, check_range:int, total_frames:int) -> Swap_Calculation_Config:
    """
    Determines the calculation parameters based on a given mode.

    Args:
        mode (str): The calculation mode ("full", "auto_check", "manual_check", "remap").
        frame_idx (int): The starting frame index for the check.
        check_range (int): A base value for the frame check range.

    Returns:
        Swap_Calculation_Config: An object containing the configuration parameters.
    """

    modes_config = {
        "full": Swap_Calculation_Config(
            show_progress=True,
            start_frame=0,
            frame_count_min=0,
            frame_count_max=total_frames,
            until_next_error=False,
        ),
        "auto_check": Swap_Calculation_Config(
            show_progress=False,
            start_frame=frame_idx,
            frame_count_min=0,
            frame_count_max=check_range,
            until_next_error=False,
        ),
        "manual_check": Swap_Calculation_Config(
            show_progress=False,
            start_frame=frame_idx,
            frame_count_min=check_range,
            frame_count_max=check_range * 10,
            until_next_error=True,
        ),
        "remap": Swap_Calculation_Config(
            show_progress=True,
            start_frame=frame_idx,
            frame_count_min=check_range,
            frame_count_max=total_frames,
            until_next_error=True,
        )
    }

    if mode not in modes_config:
        # Handle the case of an invalid mode
        raise ValueError(f"Invalid mode: '{mode}'. Expected one of {list(modes_config.keys())}")

    return modes_config[mode]

###########################################################################################

def get_prev_frame_in_list(frame_list:List[int], current_frame_idx:int) -> Optional[int]:
    try:
        current_idx_in_list = frame_list.index(current_frame_idx)
        prev_idx = current_idx_in_list - 1
    except ValueError:
        insertion_point = bisect.bisect_left(frame_list, current_frame_idx)
        prev_idx = insertion_point - 1

    if prev_idx >= 0:
        return frame_list[prev_idx]
    
    return None

def get_next_frame_in_list(frame_list:List[int], current_frame_idx:int) -> Optional[int]:
    try:
        current_idx_in_list = frame_list.index(current_frame_idx)
        next_idx = current_idx_in_list + 1
    except ValueError:
        insertion_point = bisect.bisect_right(frame_list, current_frame_idx)
        next_idx = insertion_point

    if next_idx < len(frame_list):
        return frame_list[next_idx]
    
    return None

def get_current_frame_inst(dlc_data:Loaded_DLC_Data, pred_data_array:np.ndarray, current_frame_idx:int) -> List[int]:
    current_frame_inst = []
    for inst in [ inst for inst in range(dlc_data.instance_count) ]:
        if np.any(~np.isnan(pred_data_array[current_frame_idx, inst, :])):
            current_frame_inst.append(inst)
    return current_frame_inst

###########################################################################################

def add_mock_confidence_score(array:np.ndarray) -> np.ndarray:
    array_dim = len(array.shape) # Always check for dimension first
    if array_dim not in (2, 3):
        raise ValueError("Input array must be 2D or 3D.")

    if array_dim == 2:
        rows, cols = array.shape
        new_array = np.full((rows, cols // 2 * 3), np.nan)
        new_array[:,0::3] = array[:,0::2]
        new_array[:,1::3] = array[:,1::2]

        x_nan_mask = np.isnan(new_array[:, 0::3])
        y_nan_mask = np.isnan(new_array[:, 1::3])
        xy_not_nan_mask = ~(x_nan_mask | y_nan_mask)
        new_array[:, 2::3][xy_not_nan_mask] = 1.0

    if array_dim == 3: # Unflattened (frame_idx, instance, bodyparts)
        dim_1, dim_2, dim_3 = array.shape
        new_array = np.full((dim_1, dim_2, dim_3 // 2 * 3), np.nan)
        new_array[:,:,0::3] = array[:,:,0::2]
        new_array[:,:,1::3] = array[:,:,1::2]

        x_nan_mask = np.isnan(new_array[:, :, 0::3])
        y_nan_mask = np.isnan(new_array[:, :, 1::3])
        xy_not_nan_mask = ~(x_nan_mask | y_nan_mask)
        new_array[:, :, 2::3][xy_not_nan_mask] = 1.0

    return new_array

def unflatten_data_array(array:np.ndarray, inst_count:int) -> np.ndarray:
    rows, cols = array.shape
    new_array = np.full((rows, inst_count, cols // inst_count), np.nan)

    for inst_idx in range(inst_count):
        start_col = inst_idx * cols // inst_count
        end_col = (inst_idx + 1) * cols // inst_count
        new_array[:, inst_idx, :] = array[:, start_col:end_col]
    return new_array

def remove_confidence_score(array:np.ndarray):
    array_dim = len(array.shape) # Always check for dimension first
    if array_dim == 2:
        rows, cols = array.shape
        new_array = np.full((rows, cols // 3 * 2), np.nan)
        new_array[:,0::2] = array[:,0::3]
        new_array[:,1::2] = array[:,1::3]
    if array_dim == 3: # Unflattened (frame_idx, instance, bodyparts)
        dim_1, dim_2, dim_3 = array.shape
        new_array = np.full((dim_1, dim_2, dim_3 // 3 * 2), np.nan)
        new_array[:,:,0::2] = array[:,:,0::3]
        new_array[:,:,1::2] = array[:,:,1::3]
    return new_array

###########################################################################################

def acquire_view_perspective_for_cur_cam(cam_pos:np.ndarray) -> Tuple[float, float]:
    hypot = np.linalg.norm(cam_pos[:2]) # Length of the vector's projection on the xy plane
    elevation = np.arctan2(cam_pos[2], hypot)
    elev_deg = np.degrees(elevation)
    # Calculate azimuth (angle in the xy plane)
    azimuth = np.arctan2(cam_pos[1], cam_pos[0])
    azim_deg = np.degrees(azimuth)
    return elev_deg, azim_deg

###########################################################################################

def get_non_completely_nan_slice_count(arr_3D:np.ndarray) -> int:
    if arr_3D.ndim != 3:
        raise("Error: Input array must be a 3D array!")
    not_nan = ~np.isnan(arr_3D)
    has_non_nan = np.any(not_nan, axis=(1,2))
    count = np.sum(has_non_nan)
    return count

def circular_mean(angles: np.ndarray) -> float:
    """Compute mean of angles in radians."""
    x = np.nanmean(np.cos(angles))
    y = np.nanmean(np.sin(angles))
    return np.arctan2(y, x)

###########################################################################################

def parse_idt_df_into_ndarray(df_idtracker:pd.DataFrame) ->np.ndarray:
    """
    Convert a DataFrame with idtracker.ai-like format to a 3D numpy array.
    
    Input: 
        df_idtracker: pd.DataFrame with columns ("time", "x1", "y1", "x2", "y2", ...)
                     where each pair (xN, yN) represents the coordinates of individual N.
                     Index may represent frame/time.
    
    Output: 
        np.ndarray of shape (num_frames, num_individuals, 2), where the last dimension is [x, y] coordinates.
    """
    df = df_idtracker.drop(columns="time")

    x_cols = sorted([col for col in df.columns if col.startswith('x')], key=lambda x: int(x[1:]))
    y_cols = sorted([col for col in df.columns if col.startswith('y')], key=lambda x: int(x[1:]))

    assert len(x_cols) == len(y_cols), "Mismatch between x and y coordinate columns"

    num_frames = len(df)
    num_individuals = len(x_cols)
    coords_array = np.full((num_frames, num_individuals, 2), np.nan)

    for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
        coords_array[:, i, 0] = df[x_col].values  # x coordinates
        coords_array[:, i, 1] = df[y_col].values  # y coordinates

    return coords_array