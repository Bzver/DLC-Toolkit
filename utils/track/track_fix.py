import numpy as np
import random
from itertools import combinations

from scipy.optimize import differential_evolution

from PySide6.QtWidgets import QProgressDialog
from typing import List, Dict, Tuple, Optional

from .hungarian import Hungarian
from .sigma_kappa import sigma_estimation, kappa_estimation
from utils.pose import calculate_pose_centroids, calculate_aligned_local, calculate_pose_rotations
from utils.helper import get_instance_count_per_frame
from utils.dataclass import Track_Properties
from utils.logger import logger


class Track_Fixer:
    def __init__(
            self,
            pred_data_array:np.ndarray,
            angle_map:Dict[str, int],
            progress:QProgressDialog,
            crp_weight:Tuple[float, float, float],
            cr_sigma:Optional[Tuple[float, float]] = None,
            kappa:Optional[np.ndarray] = None,
            prefit_window_size:int = 1000,
            learning_rate:float = 0.5,
            lookback_window:int = 5,
            minimum_similarity: float = 0.15,
            gap_threshold: float = 0.05,
            used_starts: List[int]=[]
            ):
        self.pred_data_array = pred_data_array.copy()
        self.angle_map = angle_map
    
        self.crp_weight = crp_weight
        
        self.lr = learning_rate
        self.lookback_window = lookback_window

        self.pred_data_array = pred_data_array.copy()

        self.total_frames, self.instance_count, xyconf = pred_data_array.shape
        self.num_keypoint = xyconf // 3

        self.video_dim = (np.nanmax(pred_data_array[..., 0::3]), np.nanmax(pred_data_array[..., 1::3]))
        self.inst_count_per_frame = get_instance_count_per_frame(pred_data_array)
        self.centroids, self.rotations, self.poses = self._pose_array_to_crp(pred_data_array)
        self.canon_pose = np.nanmean(self.poses.reshape(self.total_frames, self.instance_count, self.num_keypoint, 3)[..., :2], axis=(0, 1))
        self.mice_length = np.linalg.norm(self.canon_pose[self.angle_map["head_idx"]]-self.canon_pose[self.angle_map["tail_idx"]])

        self.cr_sigma = cr_sigma if cr_sigma is not None else sigma_estimation((self.centroids, self.rotations, self.poses), max_disp_px=self.mice_length)
        self.kappa = kappa if kappa is not None else kappa_estimation(self.poses)
        self.min_sim = minimum_similarity
        self.gap_threshold = gap_threshold

        self.progress = progress
        self.progress.setRange(0, self.total_frames)

        self.inst_list = list(range(self.instance_count))
        self.ambiguous_frames = []
        self.used_starts = used_starts
        
        self._precompute_valid_windows(window_step=prefit_window_size)

    def fit_full_video(self):
        return self._weight_repeater(range(self.total_frames))

    def iter_orchestrator(self):
        current_try = 0
        max_try = 5
        blast, ambi = [], []
        pred_data_array = self.pred_data_array.copy()

        while current_try < max_try:
            if not self.window_list:
                self.window_list = self.window_list_org.copy()
                self.used_starts.clear()
            
            window_idx = self.window_list.pop(0)
            if self.valid_starts[window_idx] in self.used_starts:
                continue

            logger.debug(f"[TMOD] Trying window {window_idx} with score {self.window_scores[window_idx]:.3f}")
            slice_window = range(self.valid_starts[window_idx], self.valid_ends[window_idx])
            
            pred_data_array, blast, ambi = self._weight_repeater(slice_window)

            if ambi:
                logger.info(f"[TMOD] Window {window_idx} produced {len(blast)} frames with implausible assignment, {len(ambi)} frames with ambiguious assignment"
                             " → selected for training")
                self.used_starts.append(self.valid_starts[window_idx])
                break
            
            logger.debug(f"[TMOD] Window {window_idx} produced no issues → skipping to next window")

            current_try += 1
        
        if current_try == max_try:
            logger.warning("Failed to find valid window after reaching max attempts, returning the last 'clean' window.")

        return pred_data_array, blast, ambi, list(slice_window), self.used_starts

    def get_params(self):
        return self.crp_weight, self.cr_sigma, self.min_sim, self.gap_threshold, self.kappa

    def _weight_repeater(self, slice_window:range) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        ref = Track_Properties(
            centroids = np.full((self.instance_count, 2), np.nan),
            rotations = np.full((self.instance_count,), np.nan),
            poses = np.full((self.instance_count, self.num_keypoint*3), np.nan),
            last_updated = np.full((self.instance_count,), -2*self.lookback_window),
            validity = np.zeros((self.instance_count,), dtype=bool)
        )
        wr_pred_data = self.pred_data_array.copy()
        centroids, rotations, poses = self.centroids.copy(), self.rotations.copy(), self.poses.copy()
        array_to_sync = [wr_pred_data, centroids, rotations, poses]

        ambiguous_frames = []

        start, end = slice_window[0], slice_window[-1]

        if abs((end-start) - self.total_frames) < 100:
            self.progress.setLabelText("Fitting the entire video.")
        else:
            self.progress.setLabelText(f"Testing the weight with frames between {start} and {end}.")

        self.progress.setRange(start, end)

        for frame_idx in slice_window:
            self.progress.setValue(frame_idx)
            if self.progress.wasCanceled():
                raise RuntimeError("User cancelled the operation.")
            
            logger.debug(f"---------- frame: {frame_idx} ---------- ")

            pred = Track_Properties(
                centroids = centroids[frame_idx],
                rotations = rotations[frame_idx],
                poses = poses[frame_idx],
                last_updated = np.full((self.instance_count,), frame_idx),
                validity = np.any(~np.isnan(centroids[frame_idx]), axis=1)
            )

            if not np.any(pred.validity):
                logger.debug("[TMOD] Skipping due to no prediction on current frame.")
                continue

            ref.validity = (ref.last_updated > frame_idx - self.lookback_window)

            if not np.any(ref.validity):
                if not np.all(np.isnan(ref.centroids)):
                    ambiguous_frames.append(frame_idx)
                logger.debug("[TMOD] No valid ref frame found. Rebasing the reference on current frame.")
                ref = self._rebase_ref(ref, array_to_sync[1:], frame_idx, pred.validity)
                continue

            for i in range(self.instance_count):
                logger.debug(f"x,y in pred: inst {i}: ({pred.centroids[i,0]:.1f}, {pred.centroids[i,1]:.1f}) | angle: {pred.rotations[i]:.2f} ")
            for i in range(self.instance_count):
                logger.debug(f"x,y in ref: inst {i}: ({ref.centroids[i,0]:.1f}, {ref.centroids[i,1]:.1f}) | angle: {ref.rotations[i]:.2f} | "
                            f"last updated: {ref.last_updated[i]} | valid: {ref.validity[i]}")

            hun = Hungarian(pred, ref, oks_kappa=self.kappa, sigma=self.cr_sigma, weight=self.crp_weight,
                            min_sim=self.min_sim, gap_threshold=self.gap_threshold)

            result = hun.hungarian_matching()

            fixed_pred_mask = np.zeros((self.instance_count,), dtype=bool)
            match result:
                case "mission_failed":
                    ambiguous_frames.append(frame_idx)
                case "low_similarity":
                    ambiguous_frames.append(frame_idx)
                    fixed_pred_mask = pred.validity & (~ref.validity)
                case "low_confidence": 
                    ambiguous_frames.append(frame_idx)
                    fixed_pred_mask = pred.validity
                case "success":
                    new_order = hun.get_new_order()
                    match new_order:
                        case self.inst_list: logger.debug("[TMOD] NO SWAP, already the best solution.")
                        case _: logger.debug("[TMOD] SWAP, new_order: {new_order}.")
                    self._sync_changes(array_to_sync, frame_idx, new_order)
                    fixed_pred_mask = pred.validity[new_order]

            if np.any(fixed_pred_mask):
                ref = self._rebase_ref(ref, array_to_sync[1:], frame_idx, fixed_pred_mask)

        return array_to_sync[0], ambiguous_frames

    def process_labels(self, pred_data_array, slice_window, status_array):
        if np.all(status_array == 0):
            return

        slice_corrected = pred_data_array
        self._train_swap(slice_corrected, slice_window, status_array)

    def _train_swap(self, slice_corrected: np.ndarray, slice_window: range, status_array: np.ndarray):
        valid_mask = np.all((status_array > 0) & (status_array < 3), axis=1)
        if not np.any(valid_mask):
            logger.debug("[S_TRAIN] Skipping training.")
            return

        def objective(x):
            w1, w2, log_s1, log_s2, min_sim, gap_thresh = x
            w3 = 1.0 - w1 - w2
            if w1 < 0.6 or w2 < 0 or w3 < 0:
                return 1e6
            
            weights = (float(w1), float(w2), float(w3))
            sigma = (float(np.exp(log_s1)), float(np.exp(log_s2)))

            try:
                error = self._swap_train_assessment(
                    slice_corrected, slice_window, status_array,
                    weights=weights,
                    sigma=sigma,
                    thresholds=(min_sim,gap_thresh)
                )
                return error
            except Exception as e:
                logger.warning(f"[SASS] Evaluation failed: {e}")
                return 1e6

        w1_curr, w2_curr, _ = self.crp_weight
        s1, s2 = self.cr_sigma

        bounds = [
            (0.6, 1.0),           # w1
            (0.0, 0.4),           # w2
            (np.log(s1) - 1.0, np.log(s1) + 1.0),  # log(s1)
            (np.log(s2) - 1.0, np.log(s2) + 1.0),  # log(s2)
            (0.05, 0.30),         # min_sim ∈ [0.05, 0.30]
            (0.01, 0.20),         # gap_threshold ∈ [0.01, 0.20]
        ]

        x0 = [
            w1_curr,
            w2_curr,
            np.log(self.cr_sigma[0]),
            np.log(self.cr_sigma[1]),
            self.min_sim,
            self.gap_threshold
        ]

        result = differential_evolution(
            objective,
            bounds,
            x0 = x0,
            maxiter=100,
            popsize=7,
            seed=42,
            polish=False,
            atol=1e-3,
            updating='deferred',
            workers=1
        )

        best_w1, best_w2, best_s1_log, best_s2_log, best_min_sim, best_gap_thresh = result.x
        best_w3 = 1.0 - best_w1 - best_w2
        best_weight = (float(best_w1), float(best_w2), float(best_w3))
        best_s1, best_s2 =  float(np.exp(best_s1_log)), float(np.exp(best_s2_log))
        best_sigma =  (best_s1, best_s2)
        best_score = result.fun
        logger.info(f"[S_TRAIN] Final params: weights={best_weight}, sigma={best_sigma},"
                    f"min_sim={best_min_sim:.2f}, gap={best_gap_thresh:.2f}")

        if best_score < 0.1:
            self.crp_weight = tuple(
                (1 - self.lr) * old + self.lr * new 
                for old, new in zip(self.crp_weight, best_weight)
            )
            self.cr_sigma = (
                (1 - self.lr) * self.cr_sigma[0] + self.lr * best_s1,
                (1 - self.lr) * self.cr_sigma[1] + self.lr * best_s2
            )
            self.min_sim = (1 - self.lr) * self.min_sim + self.lr * best_min_sim
            self.gap_threshold = (1 - self.lr) * self.gap_threshold + self.lr * best_gap_thresh

            logger.info(f"[S_TRAIN] Updated CRP weights to {self.crp_weight}. Updated CR Sigma to {self.cr_sigma} "
                        f"Updated threshold to {(self.min_sim, self.gap_threshold)}  (error: {best_score:.3f})")
        else:
            logger.warning(f"[S_TRAIN] DE failed to find good weights; keeping current. Best score: {best_score}")

    def _swap_train_assessment(
        self,
        corrected_window: np.ndarray,
        window_slice: range,
        status_array: np.ndarray,
        weights: Tuple[float, float, float],
        sigma: Tuple[float, float],
        thresholds: Tuple[float, float]
    ) -> float:
        F = len(window_slice)
        logger.debug(f"[SASS] Window size: {F} frames, {self.instance_count} instances, "
                     f"weights={weights}, sigma={sigma}, thresholds: {thresholds}")

        raw_centroids = self.centroids[window_slice].copy()
        raw_rotations = self.rotations[window_slice].copy()
        raw_poses = self.poses[window_slice].copy()

        raw_cent_corrected = raw_centroids.copy()

        gt_centroids, gt_rotations, gt_poses = self._pose_array_to_crp(corrected_window)

        ref = Track_Properties(
            centroids = gt_centroids[0],
            rotations = gt_rotations[0],
            poses = gt_poses[0],
            last_updated = np.full(self.instance_count, 0),
            validity = np.any(~np.isnan(gt_centroids[0]), axis=1)
        )

        for f in range(1, F):
            pred = Track_Properties(
                centroids=raw_centroids[f],
                rotations=raw_rotations[f],
                poses=raw_poses[f],
                last_updated=np.full(self.instance_count, f),
                validity = np.any(~np.isnan(raw_centroids[f]), axis=1)
            )

            gt_validity = np.any(~np.isnan(gt_centroids[f]), axis=1)
            if not np.any(pred.validity) or not np.any(ref.validity):
                logger.debug(f"[SASS] Frame {f}: Skipping (no valid pred/ref)")
                ref = self._rebase_ref(ref, [gt_centroids, gt_rotations, gt_poses], f, gt_validity)
                continue

            ref.validity = (ref.last_updated > f - self.lookback_window)

            hun = Hungarian(pred, ref, oks_kappa=self.kappa,
                            sigma=sigma, weight=weights, min_sim=thresholds[0], gap_threshold=thresholds[1])
            result = hun.hungarian_matching()
    
            fixed_pred_mask = np.zeros((self.instance_count,), dtype=bool)
            match result:
                case "low_similarity":
                    fixed_pred_mask = gt_validity & (~ref.validity)
                case "low_confidence": 
                    fixed_pred_mask = gt_validity
                case "success":
                    new_order = hun.get_new_order()
                    match new_order:
                        case self.inst_list: logger.debug("[TMOD] NO SWAP, already the best solution.")
                        case _: logger.debug("[TMOD] SWAP, new_order: {new_order}.")
                    raw_cent_corrected[f, :, :] = raw_centroids[f, new_order, :]
                    fixed_pred_mask = gt_validity

            if np.any(fixed_pred_mask):
                ref = self._rebase_ref(ref, [gt_centroids, gt_rotations, gt_poses], f, fixed_pred_mask)

        raw_cent_corrected = np.nan_to_num(raw_cent_corrected, nan=-1.0)
        gt_centroids = np.nan_to_num(gt_centroids, nan=-1.0)

        total_error = np.sum(np.any(raw_cent_corrected != gt_centroids, axis=-1))
        final_error = total_error / (F*self.instance_count)

        logger.debug(f"[SASS] Final error: {final_error:.4f}.")
        return final_error

    def _precompute_valid_windows(self, window_step: int = 100):
        cumsum = np.concatenate(([0], np.cumsum(self.inst_count_per_frame)))
        starts = np.arange(0, self.total_frames, window_step)
        ends = np.minimum(starts + window_step, self.total_frames)
        lens = ends - starts

        nonzero = lens > 0
        starts, ends, lens = starts[nonzero], ends[nonzero], lens[nonzero]

        sums = cumsum[ends] - cumsum[starts]
        valid = sums >= lens

        if not np.any(valid):
            logger.warning("[TMOD] No windows satisfied medium condition (sum >= len); relaxing to 2*sum >= len")
            valid = (2 * sums) >= lens

        if not np.any(valid):
            raise RuntimeError(f"No non-overlapping windows (step={window_step}) satisfy even minimal instance count fraction.")

        starts = starts[valid]
        ends = ends[valid]

        proximity_threshold = 2 * self.mice_length
        logger.debug(f"[TMOD] Canonical body length: {self.mice_length:.1f}px, proximity threshold: {proximity_threshold:.1f}px")

        centroid_dist = []
        for i, j in combinations(self.inst_list, 2):
            centroid_dist.append(np.linalg.norm(self.centroids[:, i, :]- self.centroids[:, j, :], axis=-1))

        centroid_dist_arr = np.array(centroid_dist)
        min_pairwise_dist = np.nanmin(centroid_dist_arr, axis=0)
        close_frame = min_pairwise_dist < proximity_threshold

        speeds = np.linalg.norm(np.diff(self.centroids, axis=0), axis=-1)
        speeds_full = np.zeros((self.total_frames, len(self.inst_list)))
        speeds_full[1:, :] = speeds

        moving_frame = np.any(speeds_full > 10, axis=1)
        moving_frame = np.nan_to_num(moving_frame, nan=False).astype(bool)

        close_and_moving = close_frame & moving_frame

        scores = []
        for s, e in zip(starts, ends):
            close_count = np.sum(close_and_moving[s:e])
            score = close_count / (e - s)
            scores.append(score)
            logger.debug(f"[TMOD] Window [{s}-{e}]: {close_count}/{e - s} relevant frames, score={score:.3f}")

        sorted_indices = np.argsort(scores)[::-1]
        self.valid_starts = starts[sorted_indices]
        self.valid_ends = ends[sorted_indices]
        self.window_scores = np.array(scores)[sorted_indices]

        total_windows = len(self.valid_starts)
        top_k = min(20, total_windows)

        prioritized = list(range(top_k))
        random.shuffle(prioritized)

        if top_k < total_windows:
            remaining = list(range(top_k, total_windows))
            random.shuffle(remaining)
            prioritized.extend(remaining)

        self.window_list = prioritized
        self.window_list_org = self.window_list.copy()

        logger.info(f"[TMOD] Precomputed {len(self.valid_starts)} windows. Top ambiguity score: {self.window_scores[0]:.2f}")

    def _pose_array_to_crp(self, pred_data_array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        centroids, local_coords = calculate_pose_centroids(pred_data_array)
        local_x, local_y = local_coords[..., 0::2], local_coords[..., 1::2]

        center_start = self.angle_map["center_idx"] * 3
        center_kp = pred_data_array[..., center_start:center_start+2]
        center_available_mask = np.all(~np.isnan(center_kp), axis=-1)

        centroids[center_available_mask] = center_kp[center_available_mask]
        rotations = calculate_pose_rotations(local_x, local_y, self.angle_map)
        poses = calculate_aligned_local(pred_data_array, self.angle_map)
        return centroids, rotations, poses

    @staticmethod
    def _sync_changes(array_list:List[np.ndarray], frame_idx:int, new_order:List[int]):
        for array in array_list:
            if len(array.shape) == 3:
                array[frame_idx, :, :] = array[frame_idx, new_order, :]
            elif len(array.shape) == 2:
                array[frame_idx, :] = array[frame_idx, new_order]
            else:
                raise RuntimeError(f"Incompatible array shape: {array.shape}")

    @staticmethod
    def _rebase_ref(ref:Track_Properties, crp_list:List[np.ndarray], frame_idx:int, mask:np.ndarray) -> Track_Properties:
        centroids, rotations, poses = crp_list
        ref.centroids[mask] = centroids[frame_idx][mask]
        ref.rotations[mask] = rotations[frame_idx][mask]
        ref.poses[mask] = poses[frame_idx][mask]
        ref.last_updated[mask] = frame_idx
        return ref