import numpy as np
from itertools import combinations
from typing import Tuple, Literal

from .triangulation import triangulate_point_simple
from utils.track import swap_track
from utils.dataclass import Swap_Calculation_Config

def calculate_identity_swap_score_per_frame(
        keypoint_data_tr:dict,
        valid_view:int,
        instance_count:int,
        num_keypoint:int
        ) -> float:
    """
    Computes the identity swap score for a single frame by evaluating 3D keypoint consistency 
    across all pairs of valid camera views. The score is based on the maximum deviation 
    (total Euclidean distance) of a camera pair's triangulated 3D keypoints from the mean 
    3D position across all pairs.

    Args:
        keypoint_data_tr (dict): Dictionary containing triangulation-ready keypoint data. 
            Structure: [instance][keypoint_idx] contains 'projs' (projection matrices), 
            '2d_pts' (2D image coordinates), and 'confs' (confidence scores) for each camera.
        valid_view (int): Number of valid camera views available for this frame. 
            Used to generate all possible camera pairs.
        instance_count (int): Number of instances (e.g., people) detected in the frame.
        num_keypoint (int): Number of keypoints per instance.

    Returns:
        float: The maximum total deviation score among all camera pairs, representing 
               the identity swap likelihood. Returns NaN if no valid triangulations exist.
    """
    all_camera_pairs = list(combinations(valid_view, 2))

    kp_3d_all_pair = np.full((len(all_camera_pairs), instance_count, num_keypoint, 3), np.nan)

    for pair_idx, (cam1_idx, cam2_idx) in enumerate(all_camera_pairs):
        for inst in range(instance_count): 
            for kp_idx in range(num_keypoint):
                if (cam1_idx in keypoint_data_tr[inst][kp_idx]['projs'] and 
                cam2_idx in keypoint_data_tr[inst][kp_idx]['projs']):
                    
                    proj1 = keypoint_data_tr[inst][kp_idx]['projs'][cam1_idx]
                    proj2 = keypoint_data_tr[inst][kp_idx]['projs'][cam2_idx]
                    pts_2d1 = keypoint_data_tr[inst][kp_idx]['2d_pts'][cam1_idx]
                    pts_2d2 = keypoint_data_tr[inst][kp_idx]['2d_pts'][cam2_idx]
                    conf1 = keypoint_data_tr[inst][kp_idx]['confs'][cam1_idx]
                    conf2 = keypoint_data_tr[inst][kp_idx]['confs'][cam2_idx]

                    kp_3d_all_pair[pair_idx, inst, kp_idx] = \
                        triangulate_point_simple(proj1, proj2, pts_2d1, pts_2d2, conf1, conf2)

    mean_3d_kp = np.nanmean(kp_3d_all_pair, axis=0)

    # Calculate the Euclidean distance between each pair's result and the mean
    diffs = np.linalg.norm(kp_3d_all_pair - mean_3d_kp, axis=-1) # [pair, inst, kp]
    total_diffs_per_pair = np.nansum(diffs, axis=(1, 2)) # [pair]

    if len(total_diffs_per_pair) > 0:
        if not np.all(np.isnan(total_diffs_per_pair)):
            deviant_pair_idx = np.nanargmax(total_diffs_per_pair)
        else:
            return np.nan, np.nan
        
    return total_diffs_per_pair[deviant_pair_idx]

def get_config_from_mode(
        mode:Literal["full","auto_check","manual_check","remap"],
        frame_idx:int,
        check_range:int,
        total_frames:int
        ) -> Swap_Calculation_Config:
    """
    Generates a configuration object for swap calculation based on the specified mode. 
    Different modes define distinct behaviors for processing range, progress display, 
    and stopping conditions.

    Args:
        mode (str): Operation mode. One of:
                    - "full": Process all frames from the beginning.
                    - "auto_check": Check called after an automatic swap operation to validate
                    effectiveness of the swap.
                    - "manual_check": Check called after a manual swap operation to update swap status 
                    in the proceeding frames.
                    - "remap": Check called after a successful automatic swap operation to update swap status 
                    in the proceeding frames.
        frame_idx (int): Starting frame index for modes that require a starting point, which usually is the 
                         current frame index.
        check_range (int): Base range value used to determine frame count limits for non-full modes.
        total_frames (int): Total number of frames in the video, used to cap ranges.

    Returns:
        Swap_Calculation_Config: Configuration object specifying:
            - show_progress (bool): Whether to display progress bar.
            - start_frame (int): Frame to begin processing.
            - frame_count_min (int): Minimum number of frames to process.
            - frame_count_max (int): Maximum number of frames to process.
            - until_next_error (bool): Whether to stop at the next detection error.
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

def acquire_view_perspective_for_selected_cam(cam_pos:np.ndarray) -> Tuple[float, float]:
    """
    Computes the elevation and azimuth angles of a camera's position relative to the origin, 
    providing a spherical coordinate representation of its viewing direction.

    Args:
        cam_pos (np.ndarray): 3D Cartesian coordinates of the camera position [x, y, z].

    Returns:
        Tuple[float, float]: 
            - elev_deg (float): Elevation angle in degrees, indicating vertical orientation. 
                                Positive values above the xy-plane, negative below.
            - azim_deg (float): Azimuth angle in degrees, indicating horizontal orientation 
                                measured from the positive x-axis toward the positive y-axis.
    """
    hypot = np.linalg.norm(cam_pos[:2]) # Length of the vector's projection on the xy plane
    elevation = np.arctan2(cam_pos[2], hypot)
    elev_deg = np.degrees(elevation)
    
    # Calculate azimuth (angle in the xy plane)
    azimuth = np.arctan2(cam_pos[1], cam_pos[0])
    azim_deg = np.degrees(azimuth)
    return elev_deg, azim_deg

def track_swap_3D(pred_data_array:np.ndarray, frame_idx:int, selected_cam_idx:int) -> np.ndarray:
    pred_data_array_to_swap = pred_data_array[:, selected_cam_idx, :, :]
    pred_data_array_swapped = swap_track(pred_data_array_to_swap, frame_idx, mode="batch")
    pred_data_array[:, selected_cam_idx, :, :] = pred_data_array_swapped
    return pred_data_array