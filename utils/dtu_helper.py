import pandas as pd
import numpy as np
import bisect
from itertools import combinations

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

from typing import List, Optional, Tuple, Union, Dict
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
    try:
        log_file = "D:/Project/debug_log.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs)
    except:
        return

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

def parse_idt_df_into_ndarray(df_idtracker:pd.DataFrame, df_confidence:pd.DataFrame,
        confidence_threshold:float=0.5) ->np.ndarray:
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

def calculate_snapping_zoom_level(current_frame_data:np.ndarray, view_width:float, view_height:float
        )->Tuple[float,float,float]:
    x_vals_current_frame = current_frame_data[:, 0::3]
    y_vals_current_frame = current_frame_data[:, 1::3]

    if np.all(np.isnan(x_vals_current_frame)):
        return
    
    min_x, max_x = np.nanmin(x_vals_current_frame), np.nanmax(x_vals_current_frame)
    min_y, max_y = np.nanmin(y_vals_current_frame), np.nanmax(y_vals_current_frame)

    padding_factor = 1.1 # 10% padding
    width, height = max(1.0, max_x - min_x), max(1.0, max_y - min_y)
    padded_width, padded_height = width * padding_factor, height * padding_factor
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    # Calculate new zoom level
    if padded_width > 0 and padded_height > 0:
        zoom_x, zoom_y = view_width / padded_width, view_height / padded_height
        new_zoom_level = min(zoom_x, zoom_y)
    else:
        new_zoom_level = 1.0

    # Apply zoom limits
    new_zoom_level = max(0.1, min(new_zoom_level, 10.0))

    return new_zoom_level, center_x, center_y

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

def calculate_pose_rotations(local_x: np.ndarray, local_y: np.ndarray, angle_map_data:Dict[str, any]) -> Union[np.ndarray, float]:
    """
    Calculate the rotation angles for poses based on relative keypoint positions.
    Uses the vector from the most-central keypoint (lowest spread) to the most-peripheral 
    keypoint (highest spread) as the reference direction to compute orientation.

    Args:
        local_x : Relative x-coordinates of keypoints, shape (K,) or (N, K).
        local_y : Relative y-coordinates of keypoints, shape (K,) or (N, K).
        angle_map_data : Dict with 'head_idx', 'tail_idx', 'angle_map'.
            - angle map: list of (i, j, offset, weight) of each connection

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
    angles = np.full(N, np.nan)

    head_idx = angle_map_data['head_idx']
    tail_idx = angle_map_data['tail_idx']

    # Use head-tail first
    xh, yh = local_x[:, head_idx], local_y[:, head_idx]
    xt, yt = local_x[:, tail_idx], local_y[:, tail_idx]

    valid = np.isfinite(xh) & np.isfinite(yh) & np.isfinite(xt) & np.isfinite(yt)
    if np.any(valid): # Compute angle from tail to head (anterior direction)
        dx = xh[valid] - xt[valid]
        dy = yh[valid] - yt[valid]
        head_tail_angles = np.arctan2(dy, dx)
        angles[valid] = head_tail_angles

    # Fill gaps using angle map with weight voting
    total_weights = np.zeros(N)
    weighted_sum = np.zeros(N)

    for entry in angle_map_data.get('angle_map', []):
        i, j = entry['i'], entry['j']
        offset = entry['offset']
        weight = entry['weight']

        if not (0 <= i < K and 0 <= j < K):
            continue

        xi, yi = local_x[:, i], local_y[:, i]
        xj, yj = local_y[:, j], local_y[:, j]
        connection_valid_mask = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(xj) & np.isfinite(yj)
        if not np.any(connection_valid_mask):
            continue

        # Compute observed vector angle (from i to j) for valid connections
        obs_angle_subset = np.arctan2(yj[connection_valid_mask] - yi[connection_valid_mask],
                                      xj[connection_valid_mask] - xi[connection_valid_mask])
        inferred_angle_subset = obs_angle_subset - offset
        inferred_angle_subset = np.arctan2(np.sin(inferred_angle_subset), np.cos(inferred_angle_subset))

        # Use weighted averaging in complex space to avoid angle wrapping issues
        inferred_angle_complex_subset = np.exp(1j * inferred_angle_subset)
        temp_inferred_complex_full = np.full(N, np.nan + 0j, dtype=complex)
        temp_inferred_complex_full[connection_valid_mask] = inferred_angle_complex_subset

        global_target_mask = connection_valid_mask & np.isnan(angles)

        if np.any(global_target_mask):
            weighted_sum[global_target_mask] += weight * np.angle(temp_inferred_complex_full[global_target_mask])
            total_weights[global_target_mask] += weight

    # Apply weighted average for frames that were filled via consensus
    final_mask = (total_weights > 0) & np.isnan(angles)
    angles[final_mask] = np.arctan2(
        np.sin(weighted_sum[final_mask] / total_weights[final_mask]),
        np.cos(weighted_sum[final_mask] / total_weights[final_mask])
    )

    return angles[0].item() if orig_ndim == 1 else angles
    
def align_poses_by_vector(local_coords:np.ndarray, angle_map_data:Dict[str, any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotates all poses to a common orientation using an anchor-reference vector.
    
    Args:
        local_coords (np.ndarray): Relative positions of keypoints, shape (N, K*2).
        angle_map_data : Dict with 'head_idx', 'tail_idx', 'angle_map'.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Rotated relative positions, shape (N, K*2).
            - Angles of rotation used to align the poses in radians, shape (N,).
    
    """
    relative_x, relative_y = local_coords[:, 0::2], local_coords[:, 1::2]
    angles = calculate_pose_rotations(relative_x, relative_y, angle_map_data=angle_map_data)
    
    cos_angles, sin_angles = np.cos(-angles), np.sin(-angles)
    rotated_rltv = np.empty(local_coords.shape)
    rotated_relative_x = relative_x * cos_angles[:, np.newaxis] - relative_y * sin_angles[:, np.newaxis]
    rotated_relative_y = relative_x * sin_angles[:, np.newaxis] + relative_y * cos_angles[:, np.newaxis]
    rotated_rltv[:, 0::2], rotated_rltv[:, 1::2] = rotated_relative_x, rotated_relative_y
    
    return rotated_rltv, angles

def track_rotation_worker(angle:float, centroids:np.ndarray, local_coords:np.ndarray, confs:np.ndarray) -> np.ndarray:
    """
    Rotate average relative keypoint positions by a given angle around the average centroid, then compute an aligned pose.
    
    Args:
        angle (float): Rotation angle in radians.
        centroids (np.ndarray): Average centroids of shape (2,) or (N, 2) for same instance across multiple frames.
        local_coords (np.ndarray): Local coordinates of shape (K*2,) or (N, K*2) for same instance across multiple frames.
        confs (np.ndarray): Confidence scores of shape (K,) or (N, K) for same instance across multiple frames.

    Returns:
        np.ndarray: Averaged & rotated single pose in global coordinates, shape (K*3,).
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    avg_centroid = centroids if centroids.ndim == 1 else np.nanmean(centroids, axis=0)
    avg_confs = confs if confs.ndim == 1 else np.nanmean(confs, axis=0)
    if local_coords.ndim == 1:
        avg_local_x, avg_local_y = local_coords[..., 0::2], local_coords[..., 1::2]
    else:
        avg_local_x, avg_local_y = np.nanmean(local_coords[..., 0::2], axis=0), np.nanmean(local_coords[..., 1::2], axis=0)

    global_x = avg_local_x * cos_a - avg_local_y * sin_a + avg_centroid[0]
    global_y = avg_local_x * sin_a + avg_local_y * cos_a + avg_centroid[1]
    average_pose = np.empty(local_coords.shape[-1] // 2 * 3)
    average_pose[0::3], average_pose[1::3], average_pose[2::3] = global_x, global_y, avg_confs
    return average_pose

def calculate_canonical_pose(pred_data_array:np.ndarray):
    F, I, XYCONF = pred_data_array.shape
    pred_data_combined = pred_data_array.reshape(F*I, 1, XYCONF)
    _, local_coords = calculate_pose_centroids(pred_data_combined)
    all_frame_poses = np.squeeze(local_coords)
    average_pose = np.nanmean((all_frame_poses), axis=0)
    canon_pose = average_pose.reshape(XYCONF//3, 2) # (kp,2)
    return canon_pose, all_frame_poses

#########################################################################################################################################################1

def infer_head_tail_indices(keypoint_names:List[str]) -> Tuple[int,int]:
    """
    Infer head and tail keypoint indices from keypoint names with robust handling
    of capitalization, underscores, and common anatomical naming patterns.
    
    Args:
        keypoint_names: list of all the keypoint names

    Returns:
        idx of supposed head and tail keypoint
    """
    # Define priority-ordered keywords (lowercase, without underscores)
    head_keywords_priority = [
        'nose', 'head', 'forehead', 'front', 'snout', 'face', 'mouth', 'muzzle', 'spinF', 'neck', 'eye', 'ear', 'cheek', 'chin'
    ]
    tail_keywords_priority = [
        'tailbase', 'base_tail', 'tail_base', 'butt', 'hip', 'rump', 'thorax_back', 'ass', 'pelvis', 'tail', 'spineM', 'cent'
    ]

    def normalize(name): # Normalize keypoint names: lowercase, remove non-alphanumeric, collapse underscores
        return ''.join(c.lower() for c in name if c.isalnum())

    normalized_names = [normalize(name) for name in keypoint_names]

    # Search with priority: return first match in priority list
    head_idx = None
    for kw in head_keywords_priority:
        normalized_kw = normalize(kw)
        for idx, norm_name in enumerate(normalized_names):
            if normalized_kw in norm_name:
                head_idx = idx
                break
        if head_idx is not None:
            break

    tail_idx = None
    for kw in tail_keywords_priority:
        normalized_kw = normalize(kw)
        for idx, norm_name in enumerate(normalized_names):
            if normalized_kw in norm_name:
                tail_idx = idx
                break
        if tail_idx is not None:
            break

    if head_idx is None:
        print("Warning: Could not infer head keypoint from keypoint names.")
    if tail_idx is None:
        print("Warning: Could not infer tail keypoint from keypoint names.")

    return head_idx, tail_idx

def get_head_tail_indices_from_canon_pose(canon_pose:np.ndarray, head_idx:int, tail_idx:int) -> Tuple[int,int]:
    if head_idx is None and tail_idx is None: # PCA fallback
        valid = ~np.isnan(canon_pose).all(axis=1)
        points = canon_pose[valid]
        if len(points) == 2:
            head_idx, tail_idx = 0, 1
        else:
            cov = np.cov(points.T)
            _, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, -1]
            proj = canon_pose @ axis
            head_idx = int(np.nanargmax(proj))
            tail_idx = int(np.nanargmin(proj))
    elif head_idx is None: # Take d
        tail_vec = canon_pose[head_idx]
        dist_sq = np.sum((canon_pose - tail_vec)**2, axis=1)
        tail_idx = int(np.nanargmax(dist_sq))
    elif tail_idx is None:
        head_vec = canon_pose[tail_idx]
        dist_sq = np.sum((canon_pose - head_vec)**2, axis=1)
        head_idx = int(np.nanargmax(dist_sq))

    if head_idx == tail_idx:
        raise ValueError("Could not resolve valid head/tail indices")

    return head_idx, tail_idx

def build_angle_map(canon_pose:np.ndarray, all_frame_poses:np.ndarray , head_idx:int, tail_idx:int) -> dict:
    canonical_vec = canon_pose[head_idx] - canon_pose[tail_idx]
    num_keypoint = canon_pose.shape[0]
    if np.linalg.norm(canonical_vec) < 1e-6:
        canonical_body_angle = 0.0
    else:
        canonical_body_angle = np.arctan2(canonical_vec[1], canonical_vec[0])

    # Build angle map for every possible connection
    angle_map = []  # (i, j, expected_offset, weight)
    all_angles = np.arctan2(all_frame_poses[:, 1::2], all_frame_poses[:, 0::2])  # (N, K)

    for i in range(num_keypoint):
        for j in range(num_keypoint):
            if i == j:
                continue

            # Vector from i to j in canon pose
            vec = canon_pose[j] - canon_pose[i]
            if np.linalg.norm(vec) < 1e-6:
                continue

            # Expected angle of this vector
            raw_angle = np.arctan2(vec[1], vec[0])

            # Offset relative to canonical body angle
            offset = np.arctan2(
                np.sin(raw_angle - canonical_body_angle),
                np.cos(raw_angle - canonical_body_angle)
            )  # Wrap to [-π, π]

            # Measure angular variation (in radians)
            ij_angles = all_angles[:, j] - all_angles[:, i]  # (N,)
            ij_angles = np.arctan2(np.sin(ij_angles), np.cos(ij_angles))  # Unwrap
            var = np.nanvar(ij_angles)

            # Weight: high if stable and aligned with body
            length = np.linalg.norm(vec)
            stability = 1.0 / (1.0 + var) if var > 0 else 1.0
            alignment = abs(np.dot(vec / np.linalg.norm(vec), canonical_vec / np.linalg.norm(canonical_vec)))

            weight = length * stability * alignment

            angle_map.append({"i": i, "j": j,"offset": offset,"weight": weight})

    # Sort by weight (most reliable first)
    angle_map.sort(key=lambda x: x["weight"], reverse=True)
    
    angle_map_data = {"head_idx": head_idx, "tail_idx": tail_idx, "angle_map": angle_map}

    return angle_map_data

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