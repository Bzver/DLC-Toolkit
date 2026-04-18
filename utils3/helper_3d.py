import numpy as np
from typing import Tuple
from utils.track import swap_track


def acquire_cam_perspective(cam_pos:np.ndarray) -> Tuple[float, float]:
    hypot = np.linalg.norm(cam_pos[:2])
    elevation = np.arctan2(cam_pos[2], hypot)
    elev_deg = np.degrees(elevation)
    
    azimuth = np.arctan2(cam_pos[1], cam_pos[0])
    azim_deg = np.degrees(azimuth)
    return elev_deg, azim_deg

def track_swap_3D(comb_data_array:np.ndarray, frame_idx:int, selected_cam_idx:int) -> np.ndarray:
    pred_data_array_to_swap = comb_data_array[:, selected_cam_idx, :, :]
    pred_data_array_swapped = swap_track(pred_data_array_to_swap, frame_idx, mode="batch")
    comb_data_array[:, selected_cam_idx, :, :] = pred_data_array_swapped
    return comb_data_array