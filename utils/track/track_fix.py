import numpy as np
from itertools import combinations
from PySide6.QtWidgets import QProgressDialog
from typing import List, Dict, Tuple, Optional

from .hungarian import Hungarian
from .sigma_kappa import sigma_estimation, kappa_estimation, adaptive_sigma_nudge, adaptive_kappa_nudge
from utils.pose import calculate_pose_centroids, calculate_aligned_local, calculate_pose_rotations
from utils.helper import get_instance_count_per_frame
from utils.dataclass import Track_Properties
from utils.logger import logger


class Track_Fixer:
    def __init__(self,
                pred_data_array:np.ndarray,
                angle_map:Dict[str, int],
                progress:QProgressDialog,
                crp_weight:Tuple[float, float, float] = (0.7, 0.15, 0.15),
                blast_weight:Tuple[float, float, float] = (0.4, 0.3, 0.3),
                blast_threshold:Optional[float] = None,
                cr_sigma:Optional[Tuple[float, float]] = None,
                kappa:Optional[np.ndarray] = None,
                prefit_window_size:int = 1000,
                learning_rate:float = 0.5,
                lookback_window:int = 10,
                ):
        self.angle_map = angle_map
        self.crp_weight = crp_weight
        self.blast_weight = blast_weight
        self.blast_threshold = blast_threshold
        self.lr = learning_rate
        self.lookback_window = lookback_window

        self.wr_pred_data = pred_data_array.copy()

        self.total_frames, self.instance_count, xyconf = pred_data_array.shape
        self.num_keypoint = xyconf // 3

        self.video_dim = (np.nanmax(pred_data_array[..., 0::3]), np.nanmax(pred_data_array[..., 1::3]))
        self.inst_count_per_frame = get_instance_count_per_frame(pred_data_array)
        self.centroids, self.rotations, self.poses = self._pose_array_to_crp(pred_data_array)
        self.canon_pose = np.nanmean(self.poses.reshape(self.total_frames, self.instance_count, self.num_keypoint, 3)[..., :2], axis=(0, 1))

        self.kappa_init_manual = False
        if cr_sigma is not None:
            self.cr_sigma = cr_sigma
            self.kappa_init_manual = True
        else:
            self.cr_sigma = sigma_estimation((self.centroids, self.rotations, self.poses))

        if kappa is not None:
            self.kappa = kappa
            self.kappa_init_manual = True
        else:
            self.kappa = kappa_estimation(self.poses)

        self.progress = progress
        self.progress.setRange(0, self.total_frames)

        self.inst_list = list(range(self.instance_count))
        self.ambiguous_frames = []
        self.amb_labels, self.conf_labels, self.blast_labels = [], [], []
        
        self._precompute_valid_windows(window_step=prefit_window_size)

    def fit_full_video(self):
        return self._weight_repeater(range(self.total_frames))

    def iter_orchestrator(self):
        current_try = 0
        max_try = 10
        lost_and_damned = []

        while current_try < max_try:
            if not self.window_list:
                self.window_list = self.window_list_org.copy()
            
            window_idx = self.window_list.pop(0)
            logger.debug(f"[TMOD] Trying window {window_idx} with score {self.window_scores[window_idx]:.3f}")
            slice_window = range(self.valid_starts[window_idx], self.valid_ends[window_idx])
            
            pred_data_array, blasted, ambi = self._weight_repeater(slice_window)

            lost_and_damned = [*blasted, *ambi]
            if lost_and_damned:
                logger.debug(f"[TMOD] Window {window_idx} produced {len(blasted)} blasted + {len(ambi)} ambiguous frames → selected for training")
                break
            
            logger.debug(f"[TMOD] Window {window_idx} produced no issues → skipping to next window")

            current_try += 1
        
        if current_try == max_try:
            logger.warning("Failed to find valid window after reaching max attempts, returning the last 'clean' window.")

        return pred_data_array, blasted, ambi, list(slice_window)

    def get_weights(self):
        return self.crp_weight, self.blast_weight, self.blast_threshold

    def get_sk_vals(self):
        return self.cr_sigma, self.kappa

    def _weight_repeater(self, slice_window:range) -> Tuple[np.ndarray, List[int], List[int]]:
        ref = Track_Properties(
            centroids = np.full((self.instance_count, 2), np.nan),
            rotations = np.full((self.instance_count,), np.nan),
            poses = np.full((self.instance_count, self.num_keypoint*3), np.nan),
            last_updated = np.full((self.instance_count,), -2*self.lookback_window)
        )

        hun = None
        blasted_mask = self._detect_blasted_frames((self.centroids, self.rotations, self.poses))

        ambiguous_frames = []

        centroids, rotations, poses = self.centroids.copy(), self.rotations.copy(), self.poses.copy()
        array_to_sync = [self.wr_pred_data, centroids, rotations, poses]

        start, end = slice_window[0], slice_window[-1]
        blasted_mask[:start] = False
        blasted_mask[end+1:] = False
        self.wr_sliced = self.wr_pred_data[slice_window].copy()

        if start == 0 and end == self.total_frames:
            self.progress.setLabelText("Fitting the entire video.")
        else:
            self.progress.setLabelText(f"Testing the weight with frames between {start} and {end}.")

        self.progress.setRange(start, end)

        for frame_idx in slice_window:
            self.progress.setValue(frame_idx)
            if self.progress.wasCanceled():
                raise RuntimeError("User cancelled the operation.")
            
            logger.debug(f"---------- frame: {frame_idx} ---------- ")
            if blasted_mask[frame_idx]:
                logger.debug("[TMOD] Skipping blasted frame.")
                continue

            pred = Track_Properties(
                centroids = centroids[frame_idx],
                rotations = rotations[frame_idx],
                poses = poses[frame_idx],
                last_updated = np.full((self.instance_count,), frame_idx)
            )

            valid_pred_mask = np.any(~np.isnan(pred.centroids), axis=1)
            if not np.any(valid_pred_mask):
                logger.debug("[TMOD] Skipping due to no prediction on current frame.")
                continue

            valid_ref_mask = (ref.last_updated > frame_idx - self.lookback_window)

            if not np.any(valid_ref_mask):
                if not np.all(np.isnan(ref.centroids)):
                    ambiguous_frames.append(frame_idx)
                logger.debug("[TMOD] No valid ref frame found. Rebasing the reference on current frame.")
                ref = self._rebase_ref(ref, array_to_sync[1:], frame_idx, valid_pred_mask)
                continue

            for i in range(self.instance_count):
                logger.debug(f"x,y in pred: inst {i}: ({pred.centroids[i,0]:.1f}, {pred.centroids[i,1]:.1f}) | angle: {pred.rotations[i]:.2f} ")
            for i in range(self.instance_count):
                logger.debug(f"x,y in ref: inst {i}: ({ref.centroids[i,0]:.1f}, {ref.centroids[i,1]:.1f}) | angle: {ref.rotations[i]:.2f} | "
                            f"last updated: {ref.last_updated[i]} | valid: {valid_ref_mask[i]}")

            if not hun:
                hun = Hungarian(pred, ref, valid_pred_mask, valid_ref_mask, ref_frame_gap=frame_idx-ref.last_updated,
                                oks_kappa=self.kappa, sigma=self.cr_sigma, weight=self.crp_weight)
            else:
                hun.update_predictions(pred, ref, valid_pred_mask, valid_ref_mask, ref_frame_gap=frame_idx-ref.last_updated)

            new_order = None
            if frame_idx > 0 and self.inst_count_per_frame[frame_idx] < self.inst_count_per_frame[frame_idx-1]:
                new_order = self._literal_edge_case(pred, ref, valid_pred_mask, valid_ref_mask)

            if not new_order:
                new_order = hun.hungarian_matching()

                match new_order:
                    case None: logger.debug("[TMOD] Failed to build new order.")
                    case self.inst_list: logger.debug("[TMOD] NO SWAP, already the best solution.")
                    case _: logger.debug(f"[TMOD] SWAP, new_order: {new_order}.")
                        
            if new_order:
                self._sync_changes(array_to_sync, frame_idx, new_order)
                fixed_pred_mask = valid_pred_mask[new_order]
            else:
                ambiguous_frames.append(frame_idx)
                fixed_pred_mask = valid_pred_mask

            ref = self._rebase_ref(ref, array_to_sync[1:], frame_idx, fixed_pred_mask)

        return array_to_sync[0], np.where(blasted_mask)[0].tolist(), ambiguous_frames

    def process_labels(self, pred_data_array, status_array):
        if np.all(status_array == 0):
            return

        slice_corrected = pred_data_array
        pred_clean = slice_corrected[status_array < 3]
        crp = self._pose_array_to_crp(slice_corrected)
        crp_clean = self._pose_array_to_crp(pred_clean)

        if not self.kappa_init_manual:
            alpha = 0.8
            self.kappa_init_manual = True
        else:
            alpha = 0.2

        self.cr_sigma = adaptive_sigma_nudge(crp_clean, self.cr_sigma, alpha)
        self.kappa = adaptive_kappa_nudge(pred_clean, self.kappa, alpha)

        norm_oks, norm_dist, norm_rot = self._compute_blast_features(crp)

        self._train_blast(norm_oks, norm_dist, norm_rot, status_array)
        self._train_swap(slice_corrected, status_array)

    def _train_blast(self, norm_oks, norm_dist, norm_rot, status_array):
        blast_mask = (status_array == 3)
        non_blast_mask = (status_array < 3)

        if not np.any(non_blast_mask):
            logger.warning("[B_TRAIN] All frames marked as blasted — skipping (need non-blast examples)")
            return
            
        if not np.any(blast_mask):
            logger.debug(f"[B_TRAIN] No blast frames, raising the threshold to maximum blast score on clean data.")
            w1, w2, w3 = self.blast_weight
            score = w1 * norm_oks + w2 * norm_dist + w3 * norm_rot
            self.blast_threshold = np.nanmax(score[non_blast_mask]) + 1e-3
            logger.debug(f"[B_TRAIN] Threshold updated to {self.blast_threshold:.3f}")
            return

        logger.debug(f"[B_TRAIN] Blast frames: {np.sum(blast_mask)}, Non-blast: {np.sum(non_blast_mask)}")

        best_weights = self.blast_weight
        best_margin = -np.inf
        grid_count = 0
        improved_count = 0

        for w1 in np.linspace(0.2, 0.8, 5):
            for w2 in np.linspace(0.1, max(0.1, 0.8 - w1), 4):
                w3 = 1.0 - w1 - w2
                if w3 < 0.1: 
                    continue
                weights = (w1, w2, w3)
                grid_count += 1

                score = w1 * norm_oks + w2 * norm_dist + w3 * norm_rot
                blast_scores = score[blast_mask]
                non_blast_scores = score[non_blast_mask]

                if len(blast_scores) == 0 or len(non_blast_scores) == 0:
                    continue

                margin = np.min(blast_scores) - np.max(non_blast_scores)
                logger.debug(f"[B_TRAIN] Weights ({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}): "
                    f"margin={margin:.3f} (blast_min={np.nanmin(blast_scores):.3f}, nonblast_max={np.nanmax(non_blast_scores):.3f})")

                if margin > best_margin:
                    best_margin = margin
                    best_weights = weights
                    improved_count += 1
                    logger.debug(f"[B_TRAIN] → New best margin: {best_margin:.3f} with weights {best_weights}")

        logger.debug(f"[B_TRAIN] Evaluated {grid_count} weight combinations, {improved_count} improvements")

        if best_margin > 0:
            self.blast_weight = tuple((1 - self.lr) * old + self.lr * new for old, new in zip(self.blast_weight, best_weights))
            score = (best_weights[0] * norm_oks + best_weights[1] * norm_dist + best_weights[2] * norm_rot)
            blast_scores = score[blast_mask]
            non_blast_scores = score[non_blast_mask]
            self.blast_threshold = 0.5 * (np.max(non_blast_scores) + np.min(blast_scores))
            logger.info(f"[B_TRAIN] Weights: {best_weights}, threshold: {self.blast_threshold:.2f}, margin: {best_margin:.2f}")
            logger.debug(f"[B_TRAIN] Final stats - Blast scores: μ={np.mean(blast_scores):.3f}±{np.std(blast_scores):.3f}, "
                        f"Non-blast: μ={np.mean(non_blast_scores):.3f}±{np.std(non_blast_scores):.3f}")
        else:
            score = np.mean([norm_oks, norm_dist, norm_rot], axis=0)
            non_blast_scores = score[non_blast_mask]
            self.blast_threshold = np.nanmax(non_blast_scores) + 1e-3
            logger.info(f"[B_TRAIN] No separable margin. Using fallback weights, threshold: {self.blast_threshold:.2f}")
            logger.debug(f"[B_TRAIN] Fallback stats - Non-blast scores: μ={np.mean(non_blast_scores):.3f}±{np.std(non_blast_scores):.3f}")

    def _train_swap(self, slice_corrected: np.ndarray, status_array: np.ndarray):
        corrected = slice_corrected
        raw = self.wr_sliced
        status = status_array

        valid_mask = status < 3
        if not np.any(valid_mask):
            logger.debug("[S_TRAIN] No valid frames for swap training.")
            return

        logger.debug(f"[S_TRAIN] Training on {np.sum(valid_mask)} frames ({np.sum(status==1)} approved, {np.sum(status==2)} rejected)")
        
        best_weight = self.crp_weight
        best_score = float('inf')
        grid_count = 0
        evaluated_count = 0

        for w1 in np.linspace(0.6, 1.0, 5):
            for w2 in np.linspace(0.0, max(0.0, 1.0 - w1), 4):
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                weights = (float(w1), float(w2), float(w3))
                grid_count += 1

                try:
                    error = self._swap_train_assessment(raw, corrected, status, weights)
                    evaluated_count += 1
                    logger.debug(f"[S_TRAIN] Weights {weights}: error={error:.4f}")
                    
                    if error < best_score:
                        best_score = error
                        best_weight = weights
                        logger.debug(f"[S_TRAIN] → New best error: {best_score:.4f} with weights {best_weight}")
                        
                except Exception as e:
                    logger.debug(f"[S_TRAIN] Weight {weights} failed: {e}")
                    continue

        logger.debug(f"[S_TRAIN] Evaluated {evaluated_count}/{grid_count} weight combinations")

        if best_score < 1.0:
            self.crp_weight = tuple((1 - self.lr) * old + self.lr * new for old, new in zip(self.crp_weight, best_weight))
            logger.info(f"[S_TRAIN] Updated CRP weights to {self.crp_weight} (error: {best_score:.3f})")
        else:
            logger.warning("[S_TRAIN] Failed to find valid weights; keeping current.")

    def _swap_train_assessment(
        self,
        raw_window: np.ndarray,
        corrected_window: np.ndarray,
        status_window: np.ndarray,
        weights: Tuple[float, float, float]
    ) -> float:
        """
        Simulate Hungarian assignment with:
        - Blast frames (status == 3) skipped (no assignment, no ref update)
        - Reference reset to GT on non-blast frames
        - Loss computed only on frames where assignment was attempted
        """
        F, I, _ = raw_window.shape
        logger.debug(f"[S_ASSESS] Window size: {F} frames, {I} instances, weights={weights}")

        raw_centroids, raw_rotations, raw_poses = self._pose_array_to_crp(raw_window)
        gt_centroids, gt_rotations, gt_poses = self._pose_array_to_crp(corrected_window)

        ref = None
        first_good = None
        for f in range(F):
            if status_window[f] != 3:
                ref = Track_Properties(
                    centroids=gt_centroids[f].copy(),
                    rotations=gt_rotations[f].copy(),
                    poses=gt_poses[f].copy(),
                    last_updated=np.full(I, f)
                )
                first_good = f
                logger.debug(f"[S_ASSESS] First non-blast frame: {f}")
                break
        if ref is None:
            logger.debug("[S_ASSESS] No non-blast frames found")
            return 1.0 

        total_error = 0.0
        valid_frames = 0
        processed_frames = []

        for f in range(first_good + 1, F):
            if status_window[f] == 3:
                continue

            pred_prop = Track_Properties(
                centroids=raw_centroids[f],
                rotations=raw_rotations[f],
                poses=raw_poses[f],
                last_updated=np.full(I, f)
            )

            valid_pred_mask = np.any(~np.isnan(raw_centroids[f]), axis=1)
            valid_ref_mask = np.any(~np.isnan(ref.centroids), axis=1)

            if not np.any(valid_pred_mask) or not np.any(valid_ref_mask):
                logger.debug(f"[S_ASSESS] Frame {f}: Skipping (no valid pred/ref)")
                # TEACHER FORCE
                ref.centroids = gt_centroids[f].copy()
                ref.rotations = gt_rotations[f].copy()
                ref.poses = gt_poses[f].copy()
                ref.last_updated = np.full(I, f)
                continue

            try:
                hun = Hungarian(
                    pred=pred_prop,
                    ref=ref,
                    valid_pred_mask=valid_pred_mask,
                    valid_ref_mask=valid_ref_mask,
                    ref_frame_gap=np.full(valid_ref_mask.sum(), f - ref.last_updated[valid_ref_mask]),
                    oks_kappa=self.kappa,
                    sigma=self.cr_sigma,
                    weight=weights
                )
                new_order = hun.hungarian_matching(gap_threshold=0.01)
            except Exception as e:
                logger.debug(f"[S_ASSESS] Frame {f}: Hungarian failed - {e}")
                # TEACHER FORCE and continue
                ref.centroids = gt_centroids[f].copy()
                ref.rotations = gt_rotations[f].copy()
                ref.poses = gt_poses[f].copy()
                ref.last_updated = np.full(I, f)
                continue

            if new_order is not None:
                errors = 0
                valid_count = 0
                for identity in range(I):
                    if valid_pred_mask[identity]:
                        valid_count += 1
                        if new_order[identity] != identity:
                            errors += 1
                if valid_count > 0:
                    frame_error = errors / valid_count
                    total_error += frame_error
                    valid_frames += 1
                    processed_frames.append(f)
                    logger.debug(f"[S_ASSESS] Frame {f}: error={frame_error:.3f} ({errors}/{valid_count} mismatches)")
            else:
                logger.debug(f"[S_ASSESS] Frame {f}: No assignment found")

            # TEACHER FORCE
            ref.centroids = gt_centroids[f].copy()
            ref.rotations = gt_rotations[f].copy()
            ref.poses = gt_poses[f].copy()
            ref.last_updated = np.full(I, f)

        final_error = total_error / valid_frames if valid_frames > 0 else 1.0
        logger.debug(f"[S_ASSESS] Final error: {final_error:.4f} over {valid_frames} frames (processed: {processed_frames})")
        return final_error

    def _precompute_valid_windows(self, window_step: int = 100):
        cumsum = np.concatenate(([0], np.cumsum(self.inst_count_per_frame)))
        starts = np.arange(0, self.total_frames, window_step)
        ends = np.minimum(starts + window_step, self.total_frames)
        lens = ends - starts

        nonzero = lens > 0
        starts, ends, lens = starts[nonzero], ends[nonzero], lens[nonzero]

        sums = cumsum[ends] - cumsum[starts]
        valid = (2 * sums) >= (3 * lens)

        if not np.any(valid):
            logger.info("[TMOD] No windows satisfied strict condition (2*sum >= 3*len); relaxing to sum >= len")
            valid = sums >= lens

        if not np.any(valid):
            logger.warning("[TMOD] No windows satisfied medium condition (sum >= len); relaxing to 2*sum >= len")
            valid = (2 * sums) >= lens

        if not np.any(valid):
            raise RuntimeError(f"No non-overlapping windows (step={window_step}) satisfy even minimal instance count fraction.")

        starts = starts[valid]
        ends = ends[valid]

        mice_length = np.linalg.norm(self.canon_pose[self.angle_map["head_idx"]]-self.canon_pose[self.angle_map["tail_idx"]])
        proximity_threshold = 1.5 * mice_length
        logger.debug(f"[TMOD] Canonical body length: {mice_length:.1f}px, proximity threshold: {proximity_threshold:.1f}px")

        centroid_dist = []
        for i, j in combinations(self.inst_list, 2):
            centroid_dist.append(np.linalg.norm(self.centroids[:, i, :]- self.centroids[:, j, :], axis=-1))

        centroid_dist_arr = np.array(centroid_dist)
        min_pairwise_dist = np.nanmin(centroid_dist_arr, axis=0)
        close_frame = min_pairwise_dist < 1.1 * mice_length

        scores = []
        for s, e in zip(starts, ends):
            close_count = np.sum(close_frame[s:e])
            score = close_count / (e - s)
            scores.append(score)
            logger.debug(f"[TMOD] Window [{s}-{e}]: {close_count}/{e - s} close frames, score={score:.3f}")

        sorted_indices = np.argsort(scores)[::-1]
        self.valid_starts = starts[sorted_indices]
        self.valid_ends = ends[sorted_indices]
        self.window_scores = np.array(scores)[sorted_indices]

        total_windows = len(self.valid_starts)
        top_k = min(20, total_windows)

        prioritized = list(range(top_k))

        if top_k < total_windows:
            remaining = list(range(top_k, total_windows))
            prioritized.extend(remaining)

        self.window_list = prioritized
        self.window_list_org = self.window_list.copy()

        logger.info(f"[TMOD] Precomputed {len(self.valid_starts)} windows. Top ambiguity score: {self.window_scores[0]:.2f}")

    def _literal_edge_case(
            self,
            pred:Track_Properties,
            ref:Track_Properties,
            valid_pred_mask:np.ndarray,
            valid_ref_mask:np.ndarray,
            edge_margin:float=0.1) -> Optional[List[int]]:
        """The literal edge case --- Some instances visible in ref are missing in pred (e.g.,presumably exited frame)."""
        ref_ids = np.where(valid_ref_mask)[0]
        missing_ids = [inst for inst in self.inst_list if not valid_pred_mask[inst]]

        confidently_exited = []
        uncertain_missing = []

        for inst in missing_ids:
            last_frame = int(ref.last_updated[inst])
            if last_frame < 0:
                uncertain_missing.append(inst)
                continue

            hist_frames = np.arange(max(0, last_frame - 4), last_frame + 1)
            hist_centroids = self.centroids[hist_frames, inst]  # (T, 2)
            valid_hist = np.all(~np.isnan(hist_centroids), axis=1)
            hist_centroids = hist_centroids[valid_hist]
            if len(hist_centroids) < 2:
                uncertain_missing.append(inst)
                continue

            curr_c = hist_centroids[-1]
            if len(hist_centroids) == 2:
                vel = hist_centroids[-1] - hist_centroids[-2]
            else:
                t = np.arange(len(hist_centroids))
                vx = np.polyfit(t, hist_centroids[:, 0], 1)[0]
                vy = np.polyfit(t, hist_centroids[:, 1], 1)[0]
                vel = np.array([vx, vy])

            x, y = curr_c
            near_left = x < edge_margin * self.video_dim[0]
            near_right = x > (1-edge_margin) * self.video_dim[0]
            near_top = y < edge_margin * self.video_dim[1]
            near_bottom = y > (1-edge_margin) * self.video_dim[1]

            near_edge = near_left or near_right or near_top or near_bottom
            if not near_edge:
                uncertain_missing.append(inst)
                continue

            outward_normal = np.array([0.0, 0.0])
            if near_left:    outward_normal = np.array([-1.0, 0.0])
            if near_right:   outward_normal = np.array([+1.0, 0.0])
            if near_top:     outward_normal = np.array([0.0, -1.0])
            if near_bottom:  outward_normal = np.array([0.0, +1.0])

            speed_outward = np.dot(vel, outward_normal)

            if speed_outward > 1.0:
                logger.debug(f"[EDGE] Inst {inst} exiting: pos=({x:.0f},{y:.0f}), vel={vel}, outward={speed_outward:.1f}")
                confidently_exited.append(inst)
            else:
                uncertain_missing.append(inst)

        if uncertain_missing:
            logger.debug(f"[EDGE] Uncertain missing: {uncertain_missing} → deferring to Hungarian.")
            return None

        remaining_ref_ids = [i for i in ref_ids if i not in confidently_exited]
        remaining_pred_insts = ref_ids.tolist()

        if len(remaining_ref_ids) == 0:
            return self.inst_list

        ref_small = Track_Properties(
            centroids=ref.centroids[remaining_ref_ids],
            rotations=ref.rotations[remaining_ref_ids],
            poses=ref.poses[remaining_ref_ids],
            last_updated=ref.last_updated[remaining_ref_ids]
        )
        pred_small = Track_Properties(
            centroids=pred.centroids[remaining_pred_insts],
            rotations=pred.rotations[remaining_pred_insts],
            poses=pred.poses[remaining_pred_insts],
            last_updated=pred.last_updated[remaining_pred_insts]
        )

        try:
            hun = Hungarian(
                pred_small, ref_small,
                valid_pred_mask=np.ones(len(remaining_pred_insts), dtype=bool),
                valid_ref_mask=np.ones(len(remaining_ref_ids), dtype=bool),
                ref_frame_gap=np.ones(len(remaining_ref_ids), dtype=np.int64),
                oks_kappa=self.kappa,
                sigma=self.cr_sigma,
                weight=self.crp_weight
            )
            small_order = hun.hungarian_matching()
            if small_order is None:
                return None
        except Exception as e:
            logger.debug(f"[EDGE] Small Hungarian failed: {e}")
            return None

        new_order = self.inst_list
        for idx_ref, ref_id in enumerate(remaining_ref_ids):
            pred_inst = remaining_pred_insts[small_order[idx_ref]]
            new_order[ref_id] = pred_inst

        for inst in confidently_exited:
            new_order[inst] = inst

        logger.debug(f"[EDGE] Confident exit(s): {confidently_exited} → new_order: {new_order}")
        return new_order

    def _compute_blast_features(
            self,
            crp:Tuple[np.ndarray, np.ndarray, np.ndarray],
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        C, R, P = crp
        F, I = C.shape[0], C.shape[1]

        aligned_xy = P.reshape(F, I, self.num_keypoint, 3)[..., :2]
        canon_xy = self.canon_pose[None, None, :, :]
        d2 = np.sum((aligned_xy - canon_xy) ** 2, axis=-1)

        kappa_sq = (self.kappa ** 2)[None, None, :]
        exp_term = np.exp(-d2 / (2 * kappa_sq + 1e-9))

        orig_conf = P.reshape(F, I, self.num_keypoint, 3)[..., 2]
        visible = orig_conf > 0.2
        weighted = exp_term * orig_conf * visible

        numerator = np.sum(weighted, axis=-1)
        denominator = np.sum(orig_conf * visible, axis=-1) + 1e-9
        self_oks = np.nanmin(numerator / denominator, axis=1) 
    
        centroids = C.copy()
        rotations = R.copy() 

        curr_cent = centroids[1:, :, None, :]      
        prev_cent = centroids[:-1, None, :, :]     
        d_cent_matrix = np.linalg.norm(curr_cent - prev_cent, axis=-1)  

        curr_rot = rotations[1:, :, None]          
        prev_rot = rotations[:-1, None, :]         
        d_rot_matrix = np.abs(curr_rot - prev_rot)
        d_rot_matrix = np.minimum(d_rot_matrix, 2*np.pi - d_rot_matrix)  

        min_d_cent = np.nanmin(d_cent_matrix, axis=2)
        min_d_rot  = np.nanmin(d_rot_matrix, axis=2) 

        max_min_d_cent = np.nanmax(np.concatenate([np.zeros((1,I)), min_d_cent]), axis=1)  
        max_min_d_rot  = np.nanmax(np.concatenate([np.zeros((1,I)), min_d_rot]), axis=1)   
        sigma_c, sigma_r = self.cr_sigma

        norm_oks = 1.0 - self_oks
        norm_dist = (max_min_d_cent / sigma_c) ** 2
        norm_rot = (max_min_d_rot / sigma_r) ** 2

        norm_dist = np.clip(norm_dist, 0, 25)
        norm_rot = np.clip(norm_rot, 0, 25)

        return norm_oks, norm_dist, norm_rot

    def _detect_blasted_frames(
            self,
            crp:Tuple[np.ndarray, np.ndarray, np.ndarray],
            ) -> np.ndarray:
        F = crp[0].shape[0]

        w1, w2, w3 = self.blast_weight
        blast_threshold = self.blast_threshold

        norm_oks, norm_dist, norm_rot = self._compute_blast_features(crp)
        blast_score = w1 * norm_oks + w2 * norm_dist + w3 * norm_rot

        if not blast_threshold:
            combined = (norm_oks + norm_dist + norm_rot) / 3
            threshold = np.percentile(combined[~np.isnan(combined)], 95)
        else:
            threshold = blast_threshold

        blasted_mask = blast_score > threshold
        blast_count = np.sum(blasted_mask)

        logger.info(f"[BLAST] Auto-detected {blast_count}/{F} ({100*blast_count/F:.1f}%) blasted frames.")
        if blast_count > 0:
            blast_frames = np.where(blasted_mask)[0]
            logger.debug(f" → Frames: {blast_frames[:20]}{'...' if len(blast_frames) > 20 else ''}")

        return blasted_mask

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