import numpy as np
import pandas as pd
from itertools import combinations

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

import cv2

from typing import List, Tuple, Union
from numpy.typing import NDArray

from .dtu_dataclass import Loaded_DLC_Data

def triangulate_point(
    num_views: int,
    projs: List[np.ndarray],
    pts_2d: List[Tuple[float, float]],
    confs: List[float],
    return_confidence: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Triangulates a single 3D point using Direct Linear Transformation (DLT) with confidence weighting.
    Optionally returns a confidence score based on weighted reprojection error and system condition.

    Args:
        num_views (int): Number of camera views.
        projs (list of np.array): List of 3x4 camera projection matrices.
        pts_2d (list of tuple): List of 2D image points (u, v).
        confs (list of float): Confidence values for each 2D point (e.g., detector confidence).
        return_confidence (bool): If True, returns both 3D point and a confidence score.

    Returns:
        If return_confidence is False:
            np.array: 3D point (x, y, z)
        If return_confidence is True:
            tuple: (point_3d: np.array (3,), confidence: float in [0, 1])
    """
    # === Step 1: Build DLT system with confidence weighting ===
    A = []
    valid_indices = []
    for i in range(num_views):
        if np.isnan(pts_2d[i][0]) or np.isnan(pts_2d[i][1]) or confs[i] <= 0:
            continue  # Skip invalid or zero-confidence views
        P_i = np.array(projs[i])
        u, v = pts_2d[i]
        w = confs[i]

        A.append(w * (u * P_i[2, :] - P_i[0, :]))
        A.append(w * (v * P_i[2, :] - P_i[1, :]))
        valid_indices.append(i)

    if len(A) < 4:  # Need at least 2 views (4 equations)
        if return_confidence:
            return np.full(3, np.nan), 0.0
        else:
            return np.full(3, np.nan)

    A = np.array(A)
    
    # === Step 2: Solve via SVD ===
    U, S, Vt = np.linalg.svd(A)
    point_4d_hom = Vt[-1]
    if abs(point_4d_hom[3]) < 1e-10:
        # Poorly conditioned
        if return_confidence:
            return np.full(3, np.nan), 0.0
        else:
            return np.full(3, np.nan)
    
    point_3d = (point_4d_hom / point_4d_hom[3])[:3]

    if not return_confidence:
        return point_3d

    # === Step 3: Compute Reprojection Error (Confidence Basis) ===
    reprojection_errors = []
    weighted_errors = []
    total_weight = 0.0

    for i in valid_indices:
        P_i = np.array(projs[i])
        u_obs, v_obs = pts_2d[i]
        conf = confs[i]

        # Reproject 3D point
        p_3d_hom = np.append(point_3d, 1.0)
        proj = P_i @ p_3d_hom
        u_proj, v_proj, w_proj = proj
        if abs(w_proj) < 1e-10:
            continue
        u_proj /= w_proj
        v_proj /= w_proj

        error = np.hypot(u_proj - u_obs, v_proj - v_obs)  # pixel error
        reprojection_errors.append(error)
        weighted_errors.append(conf * error)
        total_weight += conf

    if len(reprojection_errors) == 0:
        return point_3d, 0.0

    # Mean weighted reprojection error
    mean_weighted_error = sum(weighted_errors) / total_weight if total_weight > 0 else np.inf

    # === Step 4: Compute Condition Number (Numerical Stability) ===
    cond_num = S[0] / (S[-1] + 1e-10)  # Avoid division by zero
    condition_score = np.clip(1.0 / (cond_num / 100.0), 0, 1)  # Normalize to ~[0,1]

    # === Step 5: Map reprojection error to [0,1] confidence ===
    # Example: error > 10px → 0, error < 1px → 1
    max_allowed_error = 10.0  # pixels
    reproj_conf = np.clip(1.0 - (mean_weighted_error / max_allowed_error), 0.0, 1.0)

    # Combine with condition score
    final_confidence = 0.7 * reproj_conf + 0.3 * condition_score

    return point_3d, final_confidence

def triangulate_point_simple(
        proj1:NDArray,           proj2:NDArray, 
        pts_2d1:Tuple[float],    pts_2d2:Tuple[float], 
        conf1:float,             conf2:float
        ) -> NDArray:
    """
    Triangulates a single 3D point from two 2D camera views.

    Args:
        proj1 (np.array): 3x4 projection matrix for the first camera view.
        proj2 (np.array): 3x4 projection matrix for the second camera view.
        pts_2d1 (tuple): 2D image point (u, v) from the first camera view.
        pts_2d2 (tuple): 2D image point (u, v) from the second camera view.
        conf1 (float): Confidence value for the first 2D point.
        conf2 (float): Confidence value for the second 2D point.

    Returns:
        np.array: The triangulated 3D point in Euclidean coordinates (x, y, z).
    """
    num_views = 2
    projs = [proj1, proj2]
    pts_2d = [pts_2d1, pts_2d2]
    confs = [conf1, conf2]
    
    point_3d = triangulate_point(num_views, projs, pts_2d, confs, return_confidence=False)
    return point_3d

def get_projection_matrix(K, R, t):

    """
    Computes the projection matrix from camera intrinsic and extrinsic parameters.

    The projection matrix P combines the camera's intrinsic properties (K)
    with its extrinsic pose (rotation R and translation t) relative to the
    world coordinate system. It maps 3D world points to 2D image points.

    Args:
        K (np.array): The camera intrinsic matrix (3x3).
                    Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        R (np.array): The 3x3 rotation matrix representing the camera's orientation
                    in the world coordinate system.
        t (np.array): The 3x1 or (3,) translation vector representing the camera's
                    position in the world coordinate system.

    Returns:
        np.array: The 3x4 projection matrix P = K * [R | t].
    """
    # Ensure t is a 3x1 column vector
    if t.shape == (3,):
        t = t.reshape(3, 1)
    elif t.shape == (3, 1):
        pass
    else:
        raise ValueError("Translation vector 't' must be of shape (3,) or (3,1)")

    # Concatenate R and t to form the extrinsic matrix [R | t]
    extrinsic_matrix = np.hstack((R, t))
    
    # Projection matrix
    P = K @ extrinsic_matrix
    return P

def undistort_points(points_xy_conf, K, RDistort, TDistort):
    """
    Undistorts 2D image points given camera intrinsic matrix and distortion coefficients.

    Args:
        points_xy_conf (list or np.array): A 1D array or list of (x, y, confidence) triplets.
                                        Example: [x1, y1, conf1, x2, y2, conf2, ...]
        K (np.array): The camera intrinsic matrix (3x3).
                    Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        RDistort (list or np.array): Radial distortion coefficients [k1, k2].
        TDistort (list or np.array): Tangential distortion coefficients [p1, p2].

    Returns:
        np.array: A 1D array of undistorted (x, y, confidence) triplets.
                Example: [undistorted_x1, undistorted_y1, conf1, ...]
    """
    if points_xy_conf.size % 3 != 0:
        raise ValueError("Input 'points_xy_conf' must contain triplets of (x, y, confidence).")
    
    num_points = points_xy_conf.size // 3
    dist_coeffs = np.array([RDistort[0], RDistort[1], TDistort[0], TDistort[1], 0])
    points_xy_conf = np.array(points_xy_conf, dtype=np.float32)

    reshaped_points = points_xy_conf.reshape(-1, 3) # Separate (x, y) coordinates from confidences
    xy_coords = reshaped_points[:, :2]    # Shape (N, 2)
    confidences = reshaped_points[:, 2]   # Shape (N,)

    points_xy_conf = xy_coords.reshape(-1, 1, 2).astype(np.float32) # Reshape points for OpenCV: (N, 1, 2)
    undistorted_pts = cv2.undistortPoints(points_xy_conf, K, dist_coeffs, P=K)

    undistorted_pts_clean = undistorted_pts.reshape(-1, 2) #Reshape the undistorted (x, y) back to a simple (N, 2) array

    output_combined = np.empty((num_points, 3), dtype=np.float32) #Create an empty array to hold the final (x, y, conf) triplets
    output_combined[:, :2] = undistorted_pts_clean
    output_combined[:, 2] = confidences

    return output_combined.flatten() # Reshape back to 1D array

#########################################################################################################################################################1

def calculate_identity_swap_score_per_frame(keypoint_data_tr:dict, valid_view:int,
        instance_count:int, num_keypoint:int, num_cam:int) -> float:
    
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

#########################################################################################################################################################1

class Data_Processor_3D:
    def __init__(self, dlc_data:Loaded_DLC_Data, camera_params:dict, pred_data_array:NDArray, confidence_cutoff:float,
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

        point_data_size = 4 if return_confidence else 3
        point_3d_array = np.full((self.dlc_data.instance_count, self.dlc_data.num_keypoint, point_data_size), np.nan)
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
                        point_3d_array[inst, kp_idx, :3], point_3d_array[inst, kp_idx, 3] = triangulate_point(
                            num_valid_views, projs, pts_2d, confs, return_confidence)
                    else:
                        point_3d_array[inst, kp_idx, :] = triangulate_point(num_valid_views, projs, pts_2d, confs, return_confidence)

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
        ) -> NDArray:
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
                    keypoint_data_all_kps_flattened = undistort_points(keypoint_data_all_kps_flattened, K, RDistort, TDistort)
                
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