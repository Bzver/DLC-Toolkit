import numpy as np
import bisect

from typing import List, Optional
from numpy.typing import NDArray
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

def get_current_frame_inst(dlc_data:Loaded_DLC_Data, pred_data_array:NDArray, current_frame_idx:int) -> List[int]:
    current_frame_inst = []
    for inst in [ inst for inst in range(dlc_data.instance_count) ]:
        if np.any(~np.isnan(pred_data_array[current_frame_idx, inst, :])):
            current_frame_inst.append(inst)
    return current_frame_inst

###########################################################################################

def add_mock_confidence_score(array:NDArray) -> NDArray:
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

def unflatten_data_array(array:NDArray, inst_count:int) -> NDArray:
    rows, cols = array.shape
    new_array = np.full((rows, inst_count, cols // inst_count), np.nan)

    for inst_idx in range(inst_count):
        start_col = inst_idx * cols // inst_count
        end_col = (inst_idx + 1) * cols // inst_count
        new_array[:, inst_idx, :] = array[:, start_col:end_col]
    return new_array

def remove_mock_confidence_score(array:NDArray):
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