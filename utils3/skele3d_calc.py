import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from PySide6.QtCore import QRunnable, Signal, QObject
from typing import Optional, Tuple, List

from .triangulation import triangulate_point, undistort_points
from utils.dataclass import Loaded_DLC_Data
from utils.logger import logger


class Skele_Signal(QObject):
    pose_computed = Signal(int, object)


class S3D_Runner(QRunnable):
    def __init__(self, frame_idx, dlc_data, cam_params, pred_data_array, confidence_cutoff, num_cam, signal_obj:Skele_Signal):
        super().__init__()
        self.frame_idx = frame_idx
        self.dlc_data = dlc_data
        self.cam_params = cam_params
        self.pred_data_array = pred_data_array
        self.confidence_cutoff = confidence_cutoff
        self.num_cam = num_cam
        self.signal_obj = signal_obj
        self.setAutoDelete(True)
        self.valid = True
    
    def cancel(self):
        self.valid = False
    
    def run(self):
        if not self.valid or self.dlc_data is None or self.pred_data_array is None:
            return
            
        try:
            dp = Data_Processor(
                self.dlc_data, self.cam_params, self.pred_data_array, self.confidence_cutoff, self.num_cam)
            
            point_3d_array = dp.get_3d_pose_array(self.frame_idx, return_confidence=False)

            if self.valid and point_3d_array is not None:
                self.signal_obj.pose_computed.emit(self.frame_idx, point_3d_array)

        except Exception as e:
            logger.error(f"S3D Runner error: {e}", exc_info=e)


class Data_Processor:
    def __init__(
        self,
        dlc_data:Loaded_DLC_Data,
        camera_params:dict,
        pred_data_array:np.ndarray,
        confidence_cutoff:float,
        num_cam:int,
        undistorted_images:bool=False
        ):

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

        if frame_idx not in range(self.pred_data_array.shape[0]):
            logger.warning(f"[TRIANGULATION] Frame {frame_idx} is not within the range of prediction data!")
            return None

        data_size = 4 if return_confidence else 3
        point_3d_array = np.full((self.dlc_data.instance_count, self.dlc_data.num_keypoint, data_size), np.nan)
        keypoint_data_tr, _ = self.get_keypoint_data_for_frame(frame_idx, instance_threshold=1, view_threshold=2)
        if not keypoint_data_tr:
            logger.warning(f"[TRIANGULATION] Failed to get valid data from triangulation in {frame_idx}")
            return None

        for inst in range(self.dlc_data.instance_count):
            for kp_idx in range(self.dlc_data.num_keypoint):
                projs_dict = keypoint_data_tr[inst][kp_idx]['projs']
                pts_2d_dict = keypoint_data_tr[inst][kp_idx]['2d_pts']
                confs_dict = keypoint_data_tr[inst][kp_idx]['confs']
                
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

    def get_keypoint_data_for_frame(
            self,
            frame_idx:int,
            instance_threshold:int,
            view_threshold:int
            ) ->  Tuple[Optional[dict], List[int]]:

        instances_detected_per_camera = self._validate_multiview_instances(frame_idx)
        valid_view = [cam_idx for cam_idx, count in enumerate(instances_detected_per_camera) if count >= instance_threshold]

        if len(valid_view) < view_threshold:
            return None, valid_view

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

        if check_window < 1:
            return np.full(self.dlc_data.instance_count, np.nan)

        start_frame = max(0, frame_idx - check_window)
        frame_indices = np.arange(start_frame, frame_idx + 1)
        n_frames = len(frame_indices)

        if n_frames <= 1:
            return np.full(self.dlc_data.instance_count, np.nan)

        pose_array = np.full((n_frames, self.dlc_data.instance_count, self.dlc_data.num_keypoint, 3), np.nan)
        conf_array = np.full((n_frames, self.dlc_data.instance_count, self.dlc_data.num_keypoint), np.nan)

        for i, f_idx in enumerate(frame_indices):
            pose_conf = self.get_3d_pose_array(f_idx)
            if pose_conf is not None:
                pose_array[i] = pose_conf[..., :3]
                conf_array[i] = pose_conf[..., 3]

        times = np.arange(n_frames).astype(float)

        avg_speeds = np.full(self.dlc_data.instance_count, np.nan)

        for inst_idx in range(self.dlc_data.instance_count):
            speeds = []

            for kp_idx in range(self.dlc_data.num_keypoint):
                traj = pose_array[:, inst_idx, kp_idx, :]
                mask = np.all(np.isfinite(traj), axis=1)

                if np.sum(mask) < min_valid_frames:
                    continue

                valid_times = times[mask]
                valid_traj = traj[mask]

                sample_weights = None
                w = conf_array[mask, inst_idx, kp_idx]
                sample_weights = (w - w.min() + 1e-8)

                try:
                    model = LinearRegression().fit(
                        valid_times.reshape(-1, 1), valid_traj,
                        sample_weight=sample_weights
                    )
                    velocity_vector = model.coef_
                    speed = np.linalg.norm(velocity_vector)

                    speeds.append(speed)
                except Exception:
                    continue

            if speeds:
                if smoothing:
                    speeds = gaussian_filter1d(np.array(speeds), sigma=sigma)
                avg_speeds[inst_idx] = np.mean(speeds)

        return avg_speeds

    def _validate_multiview_instances(self, frame_idx:int) -> List[int]:
        if self.dlc_data.instance_count < 2:
            return [1] * self.num_cam

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
                    continue

                if cam_idx >= len(self.camera_params) or not self.camera_params[cam_idx]:
                    logger.warning(f"[TRIANGULATION] Warning: Camera parameters not available for camera {cam_idx}. Skipping.")
                    continue

                RDistort = self.camera_params[cam_idx]['RDistort']
                TDistort = self.camera_params[cam_idx]['TDistort']
                K = self.camera_params[cam_idx]['K']
                P = self.camera_params[cam_idx]['P']

                keypoint_data_all_kps_flattened = self.pred_data_array[frame_idx, cam_idx, inst_idx, :]

                if not self.undistorted_images:
                    keypoint_data_all_kps_flattened = undistort_points(keypoint_data_all_kps_flattened, K, RDistort, TDistort)
                
                keypoint_data_all_kps_reshaped = keypoint_data_all_kps_flattened.reshape(-1, 3)
                for kp_idx in range(self.dlc_data.num_keypoint):
                    point_2d = keypoint_data_all_kps_reshaped[kp_idx, :2]
                    confidence = keypoint_data_all_kps_reshaped[kp_idx, 2]

                    if confidence >= self.confidence_cutoff:
                        keypoint_data_tr[inst_idx][kp_idx]['projs'][cam_idx] = P
                        keypoint_data_tr[inst_idx][kp_idx]['2d_pts'][cam_idx] = point_2d
                        keypoint_data_tr[inst_idx][kp_idx]['confs'][cam_idx] = confidence

        return keypoint_data_tr