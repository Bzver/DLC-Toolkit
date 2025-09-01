import pandas as pd
import numpy as np

from typing import List, Tuple, Union, Dict

from .dataclass import Loaded_DLC_Data

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
    
    min_x = np.nanmin(x_vals_current_frame)
    max_x = np.nanmax(x_vals_current_frame)
    min_y = np.nanmin(y_vals_current_frame)
    max_y = np.nanmax(y_vals_current_frame)

    padding_factor = 1.25 # 25% padding
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    padded_width = width * padding_factor
    padded_height = height * padding_factor
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate new zoom level
    if padded_width > 0 and padded_height > 0:
        zoom_x = view_width / padded_width
        zoom_y = view_height / padded_height
        new_zoom_level = min(zoom_x, zoom_y)
    else:
        new_zoom_level = 1.0

    # Apply zoom limits
    new_zoom_level = max(0.1, min(new_zoom_level, 10.0))

    return new_zoom_level, center_x, center_y

#########################################################################################################################################################1

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
    
def align_poses_by_vector(local_coords:np.ndarray, angles:np.ndarray) -> np.ndarray:
    """
    Rotates all poses to a common orientation using an anchor-reference vector.
    
    Args:
        local_coords (np.ndarray): Relative positions of keypoints, shape (N, K*2).
        angles (np.ndarray): Angles of rotation used to align the poses in radians, shape (N,).

    Returns:
        - Rotated relative positions, shape (N, K*2).
    """
    relative_x, relative_y = local_coords[:, 0::2], local_coords[:, 1::2]
    cos_angles, sin_angles = np.cos(-angles), np.sin(-angles)
    rotated_rltv = np.empty(local_coords.shape)
    rotated_relative_x = relative_x * cos_angles[:, np.newaxis] - relative_y * sin_angles[:, np.newaxis]
    rotated_relative_y = relative_x * sin_angles[:, np.newaxis] + relative_y * cos_angles[:, np.newaxis]
    rotated_rltv[:, 0::2], rotated_rltv[:, 1::2] = rotated_relative_x, rotated_relative_y
    
    return rotated_rltv

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

def calculate_canonical_pose(pred_data_array:np.ndarray, head_idx:int, tail_idx:int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a canonical (aligned and averaged) pose by aligning individual animal poses to a common orientation
    based on the head-to-tail body axis.

    Args:
        pred_data_array (np.ndarray):, prediction array of shape (F, I, 3*K)
        head_idx / tail_idx (int): Index of the keypoint representing the head /tail, used to define the two ends
        of the body axis.

    Returns:
        canon_pose (np.ndarray): Canonical average pose of shape (K, 2)
        aligned_frame_poses (np.ndarray):  Aligned individual poses of shape (M, 2*K)
        """
    F, I, XYCONF = pred_data_array.shape
    pred_data_combined = pred_data_array.reshape(F*I, 1, XYCONF)
    _, local_coords = calculate_pose_centroids(pred_data_combined)
    all_frame_poses = np.squeeze(local_coords)

    no_head_mask = np.any(np.isnan(all_frame_poses[:, head_idx*2:head_idx*2+2]), axis=1)
    no_tail_mask = np.any(np.isnan(all_frame_poses[:, tail_idx*2:tail_idx*2+2]), axis=1)
    valid_frame_poses = all_frame_poses[~(no_head_mask | no_tail_mask)]

    dy = valid_frame_poses[:, head_idx*2+1] - valid_frame_poses[:, tail_idx*2+1]
    dx = valid_frame_poses[:, head_idx*2] - valid_frame_poses[:, tail_idx*2]
    valid_angles = np.arctan2(dy, dx)

    aligned_frame_poses = align_poses_by_vector(valid_frame_poses, valid_angles)

    average_pose = np.nanmean((aligned_frame_poses), axis=0)
    canon_pose = average_pose.reshape(XYCONF//3, 2) # (kp,2)
    
    return canon_pose, aligned_frame_poses

def calculate_bbox(inst_x:np.ndarray, inst_y:np.ndarray, padding:float=10.0) -> Tuple[float, float, float, float]:
    min_x, min_y = np.nanmin(inst_x) - padding, np.nanmin(inst_y) - padding
    max_x, max_y = np.nanmax(inst_x) + padding, np.nanmax(inst_y) + padding
    return min_x, min_y, max_x, max_y

def get_current_frame_inst(dlc_data:Loaded_DLC_Data, pred_data_array:np.ndarray, current_frame_idx:int) -> List[int]:
    current_frame_inst = []
    for inst in [ inst for inst in range(dlc_data.instance_count) ]:
        if np.any(~np.isnan(pred_data_array[current_frame_idx, inst, :])):
            current_frame_inst.append(inst)
    return current_frame_inst

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

def log_print(*args, **kwargs):
    try:
        log_file = "D:/Project/debug_log.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs)
    except:
        pass