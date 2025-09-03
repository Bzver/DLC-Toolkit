import pandas as pd
import numpy as np
from typing import Optional, List

def delete_track(
        pred_data_array:np.ndarray,
        current_frame_idx:int,
        selected_instance_idx:int,
        deletion_range:Optional[List[int]]=None
        ) -> Optional[np.ndarray]:

    frames_to_delete = deletion_range if deletion_range else current_frame_idx
    pred_data_array[frames_to_delete, selected_instance_idx, :] = np.nan

    return pred_data_array

def swap_track(
        pred_data_array:np.ndarray,
        current_frame_idx:int,
        swap_range:Optional[List[int]]=None
        ) -> Optional[np.ndarray]:
    
    if not swap_range:
        frames_to_swap = current_frame_idx

    elif swap_range[0] == -1:
        frames_to_swap = range(current_frame_idx, pred_data_array.shape[0])

    else:
        frames_to_swap = swap_range

    pred_data_array[frames_to_swap, 0, :], pred_data_array[frames_to_swap, 1, :] = \
    pred_data_array[frames_to_swap, 1, :].copy(), pred_data_array[frames_to_swap, 0, :].copy()

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

    if max_gap == 0: # Interpolate all gaps regardless of length (no gap limit)
        return _interpolation_pd_operator(pred_data_array, selected_instance_idx, slice(None))

    # First find all nan gaps
    all_nans = np.isnan(pred_data_array[:, selected_instance_idx, :])
    all_nan_frames = np.all(all_nans, axis=1)
    
    # Identify gap lengths
    padded = np.concatenate(([False], all_nan_frames, [False]))
    deltas = np.diff(padded.astype(np.int8))  # or np.int64
    gap_starts = np.where(deltas == 1)[0]
    gap_ends = np.where(deltas == -1)[0]

    assert len(gap_starts) == len(gap_ends), f"Mismatched starts and ends. gap_starts: {len(gap_starts)}, gap_ends: {len(gap_ends)}"

    # Build gap DataFrame and filter gaps longer than max_gap
    gap_data = []
    for start, end in zip(gap_starts, gap_ends):
        gap_data.append({'Gap_Start': start, 'Gap_End': end, 'Length': end - start})
    gaps_df = pd.DataFrame(gap_data)

    if gaps_df.empty:
        print(f"No gap found for instance {selected_instance_idx}.")
        return pred_data_array
        
    for _, row in gaps_df.iterrows():
        start, end, length = int(row['Gap_Start']), int(row['Gap_End']), row['Length']
        if length > max_gap:
            continue

        total_frames = pred_data_array.shape[0]
        context = 10
        window_start = max(0, start - context)
        window_end   = min(total_frames, end + context)    
        gap_slice = slice(window_start, window_end)
        pred_data_array = _interpolation_pd_operator(pred_data_array, selected_instance_idx, gap_slice)
        
    return pred_data_array

def _interpolation_pd_operator(
        pred_data_array:np.ndarray,
        selected_instance_idx:int,
        interpolation_range:slice
        ) -> np.ndarray:
    
    num_keypoint = pred_data_array.shape[2] // 3
    
    for kp_idx in range(num_keypoint):
        # Extract x, y, confidence for the current keypoint and instance across all frames
        x_coords = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3]
        y_coords = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+1]
        conf_values = pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+2]

        # Convert to pandas Series for interpolation
        x_series = pd.Series(x_coords)
        y_series = pd.Series(y_coords)
        conf_series = pd.Series(conf_values)

        # Interpolate NaNs
        x_interpolated = x_series.interpolate(method='linear', limit_direction='both').values
        y_interpolated = y_series.interpolate(method='linear', limit_direction='both').values
        conf_interpolated = conf_series.interpolate(method='linear', limit_direction='both').values

        # Update the pred_data_array
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3] = x_interpolated
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+1] = y_interpolated
        pred_data_array[interpolation_range, selected_instance_idx, kp_idx*3+2] = conf_interpolated
    
    return pred_data_array