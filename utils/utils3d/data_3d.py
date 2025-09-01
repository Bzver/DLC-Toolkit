import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

from typing import Optional, Tuple, List

from . import triangulation as tri
from utils.dataclass import Loaded_DLC_Data

class Data_Processor_3D:
    def __init__(
        self,
        dlc_data:Loaded_DLC_Data,
        camera_params:dict,
        pred_data_array:np.ndarray,
        confidence_cutoff:float,
        num_cam:int,
        undistorted_images:bool=False
        ):
        """
        Initializes the 3D data processor with multi-view 2D pose data and camera parameters.

        Args:
            dlc_data (Loaded_DLC_Data): Loaded 2D pose data from DeepLabCut, including 
                instance count, keypoint names, and metadata.
            camera_params (dict): Dictionary containing camera calibration parameters 
                (projection matrices, intrinsic/extrinsic parameters) for each camera.
            pred_data_array (np.ndarray): Array of predicted 2D keypoints with shape 
                (num_frames, num_cameras, num_instances, num_keypoints, 3), where the last 
                dimension includes x, y, and confidence.
            confidence_cutoff (float): Minimum confidence threshold to consider a 2D keypoint 
                valid for triangulation.
            num_cam (int): Number of cameras used in the setup.
            undistorted_images (bool): Whether the input images have been undistorted. 
                Affects how 2D points are used in triangulation.
        """

        self.dlc_data = dlc_data
        self.camera_params = camera_params
        self.pred_data_array = pred_data_array
        self.confidence_cutoff = confidence_cutoff
        self.num_cam = num_cam
        self.undistorted_images = undistorted_images

    def get_3d_pose_array(
            self,
            frame_idx:int,
            return_confidence:bool=True
            ) -> Optional[np.ndarray]:
        """
        Computes 3D poses for all instances in a given frame using multi-view triangulation.

        Args:
            frame_idx (int): Index of the frame to process.
            return_confidence (bool): If True, returns an average confidence score per keypoint 
                along with 3D coordinates.

        Returns:
            np.ndarray or None: Array of shape (instance_count, num_keypoint, 3) or 
                (instance_count, num_keypoint, 4) if confidence is returned. Returns None 
                if triangulation fails due to invalid frame index or lack of valid views.
        """
        if frame_idx not in range(self.pred_data_array.shape[0]):
            print(f"TRIANGULATION | Frame {frame_idx} is not within the range of prediction data!")
            return None

        data_size = 4 if return_confidence else 3
        point_3d_array = np.full((self.dlc_data.instance_count, self.dlc_data.num_keypoint, data_size), np.nan)
        keypoint_data_tr, _ = self.get_keypoint_data_for_frame(frame_idx, instance_threshold=1, view_threshold=2)
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
                        point_3d_array[inst, kp_idx, :3], point_3d_array[inst, kp_idx, 3] = tri.triangulate_point(
                            num_valid_views, projs, pts_2d, confs, return_confidence)
                    else:
                        point_3d_array[inst, kp_idx, :] = tri.triangulate_point(num_valid_views, projs, pts_2d, confs, return_confidence)

        return point_3d_array

    def get_keypoint_data_for_frame(
            self,
            frame_idx:int,
            instance_threshold:int,
            view_threshold:int
            ) ->  Tuple[Optional[dict], List[int]]:
        """
        Prepares triangulation-ready keypoint data by filtering valid cameras and instances.

        Args:
            frame_idx (int): Frame index to extract data from.
            instance_threshold (int): Minimum number of instances that must be detected 
                in a camera view for it to be considered valid.
            view_threshold (int): Minimum number of valid camera views required to proceed 
                with triangulation.

        Returns:
            Tuple[dict or None, list]: 
                - keypoint_data_tr (dict or None): Formatted keypoint data for triangulation, 
                  structured by instance and keypoint, containing projections, 2D points, 
                  and confidences per camera. Returns None if insufficient valid views.
                - valid_view (list): List of camera indices that meet the instance threshold.
        """
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

    def _validate_multiview_instances(self, frame_idx:int) -> List[int]:
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

    def _acquire_keypoint_data(self, instances_detected_per_camera:List[int], frame_idx:int) -> dict:
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
                    keypoint_data_all_kps_flattened = tri.undistort_points(keypoint_data_all_kps_flattened, K, RDistort, TDistort)
                
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