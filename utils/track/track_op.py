import pandas as pd
import numpy as np
from typing import Optional, List

def delete_track(
        pred_data_array:np.ndarray,
        current_frame_idx:int,
        selected_instance_idx:int,
        deletion_range:Optional[List[int]]=None
        ) -> np.ndarray:

    frames_to_delete = deletion_range if deletion_range else current_frame_idx
    pred_data_array[frames_to_delete, selected_instance_idx, :] = np.nan

    return pred_data_array

def swap_track(
        pred_data_array:np.ndarray,
        current_frame_idx:int,
        swap_target:Optional[List[int]]=None,
        swap_range:Optional[List[int]]=None
        ) -> np.ndarray:
    if not swap_range:
        frames_to_swap = current_frame_idx
    elif swap_range[0] == -1:
        frames_to_swap = range(current_frame_idx, pred_data_array.shape[0])
    else:
        frames_to_swap = swap_range

    if not swap_target:
        swap_target = [0, 1]

    x, y = swap_target
    pred_data_array[frames_to_swap, y, :], pred_data_array[frames_to_swap, x, :] = \
    pred_data_array[frames_to_swap, x, :].copy(), pred_data_array[frames_to_swap, y, :].copy()

    return pred_data_array

def interpolate_track(
        pred_data_array:np.ndarray,
        frames_to_interpolate:List[int],
        selected_instance_idx:int
        ) -> np.ndarray:
    
    start_frame_for_interpol = frames_to_interpolate[0] - 1
    end_frame_for_interpol = frames_to_interpolate[-1] + 1

    start_kp_data = pred_data_array[start_frame_for_interpol, selected_instance_idx, :]
    end_kp_data = pred_data_array[end_frame_for_interpol, selected_instance_idx, :]
    interpolated_values = np.linspace(start_kp_data, end_kp_data, num=len(frames_to_interpolate)+2, axis=0)
    pred_data_array[start_frame_for_interpol : end_frame_for_interpol + 1, selected_instance_idx, :] = interpolated_values
    
    return pred_data_array

def interpolate_track_all(
        pred_data_array:np.ndarray,
        selected_instance_idx:int,
        max_gap:int
        ) -> np.ndarray:

    if max_gap == 0:
        return _interpolation_pd_operator(pred_data_array, selected_instance_idx)

    all_nans = np.isnan(pred_data_array[:, selected_instance_idx, :])

    padded = np.concatenate(([False], np.all(all_nans, axis=1), [False]))
    gap_starts = np.where(np.diff(padded.astype(np.int8)) == 1)[0]
    gap_ends = np.where(np.diff(padded.astype(np.int8)) == -1)[0]

    assert len(gap_starts) == len(gap_ends), f"Mismatched starts and ends. gap_starts: {len(gap_starts)}, gap_ends: {len(gap_ends)}"

    gap_indices = set()
    for start, end in zip(gap_starts, gap_ends):
        if end - start > max_gap:
            gap_indices.update(range(start, end))

    pred_data_array = _interpolation_pd_operator(pred_data_array, selected_instance_idx)
    pred_data_array[list(sorted(gap_indices)), selected_instance_idx, :] = np.nan
        
    return pred_data_array

def _interpolation_pd_operator(
        pred_data_array:np.ndarray,
        selected_instance_idx:int,
        interpolation_range:slice=slice(None),
        ) -> np.ndarray:
    
    num_keypoint = pred_data_array.shape[2] // 3
    
    for kp_idx in range(num_keypoint):
        x_coords = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3]
        y_coords = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+1]
        conf_values = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+2]

        x_series = pd.Series(x_coords)
        y_series = pd.Series(y_coords)
        conf_series = pd.Series(conf_values)

        x_interpolated = x_series.interpolate(method='linear', limit_direction='both').values
        y_interpolated = y_series.interpolate(method='linear', limit_direction='both').values
        conf_interpolated = conf_series.interpolate(method='linear', limit_direction='both').values

        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3] = x_interpolated
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+1] = y_interpolated
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+2] = conf_interpolated
    
    return pred_data_array