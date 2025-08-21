import pandas as pd
import numpy as np
import bisect
from itertools import combinations

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

from typing import List, Optional, Tuple, Union
from .dtu_dataclass import Loaded_DLC_Data, Swap_Calculation_Config

from . import dtu_triangulation as dutri

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

def log_print(*args, **kwargs):
    log_file = "D:/Project/debug_log.txt"
    with open(log_file, 'a', encoding='utf-8') as f:
        print(*args, file=f, **kwargs)

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

#########################################################################################################################################################1

def calculate_identity_swap_score_per_frame(keypoint_data_tr:dict, valid_view:int, instance_count:int, num_keypoint:int) -> float:
    
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
                        dutri.triangulate_point_simple(proj1, proj2, pts_2d1, pts_2d2, conf1, conf2)

    with np.errstate(invalid='ignore'):
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

def calculate_pose_centroids(pred_data_array:np.ndarray, frame_slice:Union[slice, int]=slice(None)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the centroid (mean x, mean y) of pose keypoints for each instance and frame,
    and compute the relative position of each keypoint with respect to the centroid.

    Args:
        pred_data_array (np.ndarray): 3D array of shape (frames, instances, keypoints*3).
        frame_slice (Union[slice, int]): Slice or index to select specific frames.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pose_centroids: Array of shape (instances, 2) or (N, instances, 2) with mean x and y coordinates.
            - local_coords: Array of shape (instances, keypoints*2) or (N, instances, keypoints*2) with relative positions.
    """
    _, instance_count, xyconf = pred_data_array.shape
    num_keypoint = xyconf // 3

    x_vals = pred_data_array[frame_slice, :, 0::3]
    y_vals = pred_data_array[frame_slice, :, 1::3]

    if isinstance(frame_slice, (int, np.integer)): # Determine output shape
        output_shape_pose = (instance_count, 2)
        output_shape_rltv = (instance_count, num_keypoint * 2)
    else:
        output_shape_pose = (x_vals.shape[0], instance_count, 2)
        output_shape_rltv = (x_vals.shape[0], instance_count, num_keypoint * 2)

    pose_centroids = np.full(output_shape_pose, np.nan, dtype=np.float64)
    local_coords = np.full(output_shape_rltv, np.nan, dtype=np.float64)

    pose_centroids[..., 0] = np.nanmean(x_vals, axis=-1)
    pose_centroids[..., 1] = np.nanmean(y_vals, axis=-1)

    local_coords[..., 0::2] = x_vals - pose_centroids[..., 0, np.newaxis]
    local_coords[..., 1::2] = y_vals - pose_centroids[..., 1, np.newaxis]
        
    return pose_centroids, local_coords

def calculate_pose_rotations(local_x: np.ndarray, local_y: np.ndarray) -> Union[np.ndarray, float]:
    """
    Calculate the rotation angles for poses based on relative keypoint positions.
    Uses the vector from the most-central keypoint (lowest spread) to the most-peripheral 
    keypoint (highest spread) as the reference direction to compute orientation.

    Args:
        local_x (np.ndarray): Relative x-coordinates of keypoints, shape (K,) or (N, K).
        local_y (np.ndarray): Relative y-coordinates of keypoints, shape (K,) or (N, K).

    Returns:
        Union[np.ndarray, float]: Rotation angle(s) in radians.
                                  - If input is (K,), returns float.
                                  - If input is (N, K), returns (N,) array.
    """
    orig_ndim = local_x.ndim
    if orig_ndim == 1:
        local_x = local_x[np.newaxis, :]  # (1, K)
        local_y = local_y[np.newaxis, :]  # (1, K)

    N, K = local_x.shape

    squared_dist = local_x**2 + local_y**2  # (N, K)
    avg_dist_per_keypoint = np.nanmean(squared_dist, axis=0)  # (K,)
    rmse_from_centroid = np.sqrt(avg_dist_per_keypoint)  # (K,)

    # Choose anchor and reference keypoints based on RMSE
    anchor_keypoint_idx = np.argmin(rmse_from_centroid)
    ref_keypoint_idx = np.argmax(rmse_from_centroid)

    # Compute vector from anchor to reference keypoint in each frame
    dx = local_x[:, ref_keypoint_idx] - local_x[:, anchor_keypoint_idx]  # (N,)
    dy = local_y[:, ref_keypoint_idx] - local_y[:, anchor_keypoint_idx]  # (N,)

    # Compute angle of this vector
    angles = np.arctan2(dy, dx)  # (N,)
    angles = angles.item() if orig_ndim == 1 else angles  # Return as scalar if input was 1D
    return angles
    
def align_poses_by_vector(local_coords:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotates all poses to a common orientation using an anchor-reference vector.
    
    Args:
        local_coords (np.ndarray): Relative positions of keypoints, shape (N, K*2).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Rotated relative positions, shape (N, K*2).
            - Angles of rotation used to align the poses in radians, shape (N,).
    
    """
    relative_x, relative_y = local_coords[:, 0::2], local_coords[:, 1::2]
    angles = calculate_pose_rotations(relative_x, relative_y)
    
    cos_angles, sin_angles = np.cos(-angles), np.sin(-angles)
    rotated_rltv = np.empty(local_coords.shape)
    rotated_relative_x = relative_x * cos_angles[:, np.newaxis] - relative_y * sin_angles[:, np.newaxis]
    rotated_relative_y = relative_x * sin_angles[:, np.newaxis] + relative_y * cos_angles[:, np.newaxis]
    rotated_rltv[:, 0::2], rotated_rltv[:, 1::2] = rotated_relative_x, rotated_relative_y
    
    return rotated_rltv, angles

def track_rotation_worker(angle:float, centroids:np.ndarray, local_coords:np.ndarray, confs:np.ndarray) -> np.ndarray:
    """
    Rotate average relative keypoint positions by a given angle around the average centroid, then compute a canonical aligned pose.
    
    Args:
        angle (float): Rotation angle in radians.
        centroids (np.ndarray): Average centroids of shape (2,) or (N, 2) for same instance across multiple frames.
        local_coords (np.ndarray): Local coordinates of shape (K*2,) or (N, K*2) for same instance across multiple frames.
        confs (np.ndarray): Confidence scores of shape (K,) or (N, K) for same instance across multiple frames.

    Returns:
        np.ndarray: Averaged & rotated single pose in global coordinates, shape (K*3,).
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    if centroids.ndim == 1:
        avg_centroid = centroids
        avg_local_x, avg_local_y = local_coords[..., 0::2], local_coords[..., 1::2]
        avg_confs = confs
    else:
        avg_centroid = np.nanmean(centroids, axis=0)
        avg_local_x, avg_local_y = np.nanmean(local_coords[..., 0::2], axis=0), np.nanmean(local_coords[..., 1::2], axis=0)
        avg_confs = np.nanmean(confs, axis=0)

    global_x = avg_local_x * cos_a - avg_local_y * sin_a + avg_centroid[0]
    global_y = avg_local_x * sin_a + avg_local_y * cos_a + avg_centroid[1]
    average_pose = np.empty(local_coords.shape[-1] // 2 * 3)
    average_pose[0::3], average_pose[1::3], average_pose[2::3] = global_x, global_y, avg_confs
    return average_pose

#########################################################################################################################################################1

class Data_Processor_3D:
    def __init__(self, dlc_data:Loaded_DLC_Data, camera_params:dict, pred_data_array:np.ndarray, confidence_cutoff:float,
                 num_cam:int, undistorted_images:bool=False):
        
        self.dlc_data = dlc_data
        self.camera_params = camera_params
        self.pred_data_array = pred_data_array
        self.confidence_cutoff = confidence_cutoff
        self.num_cam = num_cam
        self.undistorted_images = undistorted_images

    def get_3d_pose_array(self, frame_idx:int, return_confidence:bool=True):
        if frame_idx not in range(self.pred_data_array.shape[0]):
            print(f"TRIANGULATION | Frame {frame_idx} is not within the range of prediction data!")
            return None

        data_size = 4 if return_confidence else 3
        point_3d_array = np.full((self.dlc_data.instance_count, self.dlc_data.num_keypoint, data_size), np.nan)
        keypoint_data_tr, valid_view = self.get_keypoint_data_for_frame(frame_idx, instance_threshold=1, view_threshold=2)
        if not keypoint_data_tr:
            print(f"TRIANGULATION | Failed to get valid data from triangulation in {frame_idx}")
            return None

        for inst in range(self.dlc_data.instance_count):
            # iterate through each keypoint to perform triangulation
            for kp_idx in range(self.dlc_data.num_keypoint):
                # Convert dictionaries to lists while maintaining camera order
                projs_dict = keypoint_data_tr[inst][kp_idx]['projs']
                pts_2d_dict = keypoint_data_tr[inst][kp_idx]['2d_pts']
                confs_dict = keypoint_data_tr[inst][kp_idx]['confs']
                
                # Get sorted camera indices to maintain order
                cam_indices = sorted(projs_dict.keys())
                projs = [projs_dict[i] for i in cam_indices]
                pts_2d = [pts_2d_dict[i] for i in cam_indices]
                confs = [confs_dict[i] for i in cam_indices]
                num_valid_views = len(projs)

                if num_valid_views >= 2:
                    if return_confidence:
                        point_3d_array[inst, kp_idx, :3], point_3d_array[inst, kp_idx, 3] = dutri.triangulate_point(
                            num_valid_views, projs, pts_2d, confs, return_confidence)
                    else:
                        point_3d_array[inst, kp_idx, :] = dutri.triangulate_point(num_valid_views, projs, pts_2d, confs, return_confidence)

        return point_3d_array

    def get_keypoint_data_for_frame(self, frame_idx, instance_threshold, view_threshold):
        instances_detected_per_camera = self._validate_multiview_instances(frame_idx)
        valid_view = [cam_idx for cam_idx, count in enumerate(instances_detected_per_camera) if count >= instance_threshold]

        if len(valid_view) < view_threshold:
            return None, valid_view

        # Check for valid keypoint data for each instance
        keypoint_data_tr = self._acquire_keypoint_data(instances_detected_per_camera, frame_idx)

        if not keypoint_data_tr:
            return None, valid_view
            
        return keypoint_data_tr, valid_view

    def calculate_temporal_velocity(
        self,
        frame_idx: int,
        check_window: int = 5,
        min_valid_frames: int = 2,
        smoothing: bool = True,
        sigma: float = 0.8
        ) -> np.ndarray:
        """
        Calculates an averaged velocity-based motion score for each instance,
        using linear regression over recent trajectories within a time window.
        
        Args:
            frame_idx (int): Current frame index.
            check_window (int): Number of past frames to include (excludes current).
            min_valid_frames (int): Minimum number of valid frames to compute velocity.
            smoothing (bool): Whether to smooth velocity per keypoint over time.
            sigma (float): Gaussian smoothing sigma if smoothing is True.

        Returns:
            np.ndarray: 1D array of shape (instance_count,) with average speed per instance.
        """
        if check_window < 1:
            return np.full(self.dlc_data.instance_count, np.nan)

        # Get frame range
        start_frame = max(0, frame_idx - check_window)
        frame_indices = np.arange(start_frame, frame_idx + 1)
        n_frames = len(frame_indices)

        if n_frames <= 1:
            return np.full(self.dlc_data.instance_count, np.nan)

        # Load pose data with optional confidence
        pose_array = np.full((n_frames, self.dlc_data.instance_count, self.dlc_data.num_keypoint, 3), np.nan)
        conf_array = np.full((n_frames, self.dlc_data.instance_count, self.dlc_data.num_keypoint), np.nan)

        for i, f_idx in enumerate(frame_indices):
            pose_conf = self.get_3d_pose_array(f_idx)
            if pose_conf is not None:
                pose_array[i] = pose_conf[..., :3]
                conf_array[i] = pose_conf[..., 3]

        # Time vector for regression (shape: T,)
        times = np.arange(n_frames).astype(float)

        avg_speeds = np.full(self.dlc_data.instance_count, np.nan)

        for inst_idx in range(self.dlc_data.instance_count):
            speeds = []  # collect keypoint speeds

            for kp_idx in range(self.dlc_data.num_keypoint):
                traj = pose_array[:, inst_idx, kp_idx, :]  # (T, 3)
                mask = np.all(np.isfinite(traj), axis=1)

                if np.sum(mask) < min_valid_frames:
                    continue

                # Extract valid data
                valid_times = times[mask]
                valid_traj = traj[mask]

                # Weight by confidence if available
                sample_weights = None
                w = conf_array[mask, inst_idx, kp_idx]
                sample_weights = (w - w.min() + 1e-8)  # avoid zero weights

                # Fit linear model: pos(t) = v * t + b
                try:
                    model = LinearRegression().fit(
                        valid_times.reshape(-1, 1), valid_traj,
                        sample_weight=sample_weights
                    )
                    velocity_vector = model.coef_  # (3,) — velocity in x,y,z
                    speed = np.linalg.norm(velocity_vector)

                    speeds.append(speed)
                except Exception:
                    continue  # skip if fit fails

            # Average across keypoints
            if speeds:
                if smoothing:
                    speeds = gaussian_filter1d(np.array(speeds), sigma=sigma)
                avg_speeds[inst_idx] = np.mean(speeds)

        return avg_speeds

    def _validate_multiview_instances(self, frame_idx):
        if self.dlc_data.instance_count < 2:
            return [1] * self.num_cam # All cameras valid for the single instance

        instances_detected_per_camera = [0] * self.num_cam
        for cam_idx_check in range(self.num_cam):
            detected_instances_in_cam = 0

            for inst_check in range(self.dlc_data.instance_count):
                has_valid_data = False
                for kp_idx_check in range(self.dlc_data.num_keypoint):

                    if not pd.isna(self.pred_data_array[frame_idx, cam_idx_check, inst_check, kp_idx_check*3]):
                        has_valid_data = True
                        break

                if has_valid_data:
                    detected_instances_in_cam += 1
            instances_detected_per_camera[cam_idx_check] = detected_instances_in_cam

        return instances_detected_per_camera

    def _acquire_keypoint_data(self, instances_detected_per_camera, frame_idx):
        # Dictionary to store per-keypoint data across cameras:
        keypoint_data_tr = {
            inst:
            {
                kp_idx: {'projs': {}, '2d_pts': {}, 'confs': {}}
                for kp_idx in range(self.dlc_data.num_keypoint)
            }
            for inst in range(self.dlc_data.instance_count)
        }

        for inst_idx in range(self.dlc_data.instance_count):
            for cam_idx in range(self.num_cam):
                if self.dlc_data.instance_count > 1 and instances_detected_per_camera[cam_idx] < 2:
                    continue # Skip if this camera has not detect enough instances

                # Ensure camera_params are available for the current camera index
                if cam_idx >= len(self.camera_params) or not self.camera_params[cam_idx]:
                    print(f"TRIANGULATION | Warning: Camera parameters not available for camera {cam_idx}. Skipping.")
                    continue

                RDistort = self.camera_params[cam_idx]['RDistort']
                TDistort = self.camera_params[cam_idx]['TDistort']
                K = self.camera_params[cam_idx]['K']
                P = self.camera_params[cam_idx]['P']

                # Get all keypoint data (flattened) for the current frame, camera, and instance
                keypoint_data_all_kps_flattened = self.pred_data_array[frame_idx, cam_idx, inst_idx, :]

                if not self.undistorted_images:
                    keypoint_data_all_kps_flattened = dutri.undistort_points(keypoint_data_all_kps_flattened, K, RDistort, TDistort)
                
                # Shape the flattened data back into (num_keypoints, 3) for easier iteration
                keypoint_data_all_kps_reshaped = keypoint_data_all_kps_flattened.reshape(-1, 3)

                # Iterate through each keypoint's (x,y,conf) for the current camera
                for kp_idx in range(self.dlc_data.num_keypoint):
                    point_2d = keypoint_data_all_kps_reshaped[kp_idx, :2] # (x, y)
                    confidence = keypoint_data_all_kps_reshaped[kp_idx, 2] # confidence

                    # Only add data if the confidence is above a threshold
                    if confidence >= self.confidence_cutoff:
                        keypoint_data_tr[inst_idx][kp_idx]['projs'][cam_idx] = P
                        keypoint_data_tr[inst_idx][kp_idx]['2d_pts'][cam_idx] = point_2d
                        keypoint_data_tr[inst_idx][kp_idx]['confs'][cam_idx] = confidence

        return keypoint_data_tr