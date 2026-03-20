import os
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict

from .reviewer import Swap_Correction_Dialog
from core.io import Frame_Extractor, Temp_Manager, Frame_Exporter_Threaded
from ui import Dual_Pixmap_Dialog
from utils.track import Kalman, swap_track
from utils.pose import (
    calculate_pose_centroids, calculate_pose_array_rotations, calculate_pose_array_bbox,
    outlier_rotation, outlier_confidence, outlier_size, outlier_bodypart,
    )
from utils.helper import get_instance_count_per_frame, get_instances_on_current_frame, indices_to_spans
from utils.dataclass import Loaded_DLC_Data, Cutout_Augments, Emb_Params
from utils.logger import logger

class Track_Fixer:
    KALMAN_RESET_THRESHOLD = 3
    USER_DIALOG_COOLDOWN = 20
    AMBIGUITY_THRESHOLD = 20.0
    MAX_DIST_THRESHOLD = 120.0
    KALMAN_MAX_ERROR = 80.0

    def __init__(
        self,
        pred_data_array: np.ndarray,
        dlc_data: Loaded_DLC_Data,
        tm: Temp_Manager,
        extractor: Frame_Extractor,
        anglemap: Dict[str, int],
        emp: Emb_Params,
        skip_sweep: bool = False,
        avtomat: bool = False,
        parent=None
    ):
        if pred_data_array.shape[1] != 2:
            raise NotImplementedError("Track_Fixer supports exactly 2 instances.")
        self.pred_data_array = pred_data_array.copy()
        self.dlc_data = dlc_data
        self.extractor = extractor
        self.tm = tm
        self.temp_dir = tm.create("track")
        self.anglemap = anglemap
        self.emp = emp
        self.skip_sweep = skip_sweep
        self.avtomat = avtomat
        self.main = parent
        self.total_frames = self.pred_data_array.shape[0]
        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)
        self.inst_count_per_frame = get_instance_count_per_frame(self.pred_data_array)
        self.kalman_filters = [None, None]
        self.last_known_pos = np.full((2, 2), np.nan)
        self.kalman_failure_count = [0, 0]
        
        self.eligible_frames = []

    def track_correction(self, start_idx: int = 0, end_idx: int = -1) -> np.ndarray:
        if end_idx == -1:
            end_idx = self.total_frames

        ambiguous_frames = []

        if not self.skip_sweep:
            for f in range(start_idx, end_idx):
                success = self._correct_frame_with_hungarian(f)
                if not success:
                    ambiguous_frames.append(f)
                    logger.debug(f"[TF] Ambiguous match, added to backtrack list")

        self.eligible_frames = self._find_eligible_frames(min_eligible=self.emp.triplets)
        self.eligible_frames = [f for f in self.eligible_frames if f >= start_idx and f < end_idx]
        self._run_contrain_magic(ambiguous_frames)
        return self.pred_data_array

    def _find_eligible_frames(
            self,
            inst_dist_threshold: float = 1.2,
            conf_threshold: float = 0.5,
            size_threshold: Tuple[float, float] = (0.3, 2.5),
            bp_threshold: int = 4,
            twist_angle_threshold: float = 10.0,
            min_eligible: int = 5000,
            ) -> List[int]:

        instance_count = get_instance_count_per_frame(self.pred_data_array)
        two_inst_mask = instance_count == 2
        if not np.any(two_inst_mask):
            raise ValueError("No frame with two insts, cannot perform contrastive learning.")

        two_inst_array = self.pred_data_array[two_inst_mask]
        centroids_two_inst = self.centroids[two_inst_mask]
        head_idx = self.anglemap["head_idx"]
        all_head = self.pred_data_array[..., head_idx * 3:head_idx * 3 + 2]
        tail_idx = self.anglemap["tail_idx"]
        all_tail = self.pred_data_array[..., tail_idx * 3:tail_idx * 3 + 2]
        mice_length = np.nanmedian(np.linalg.norm(all_head - all_tail, axis=-1))
        dists = np.linalg.norm(centroids_two_inst[:, 1, :] - centroids_two_inst[:, 0, :], axis=-1)
        dist_mask = dists > inst_dist_threshold * mice_length

        conf_mask = ~np.any(outlier_confidence(two_inst_array, conf_threshold), axis=-1)
        angle_mask = ~np.any(outlier_rotation(two_inst_array, self.anglemap, twist_angle_threshold), axis=-1)

        size_mask = ~np.any(outlier_size(two_inst_array, *size_threshold), axis=-1)
        bp_mask = ~np.any(outlier_bodypart(two_inst_array, bp_threshold), axis=-1)

        full_mask = np.zeros(self.total_frames, dtype=bool)

        target_count = min(min_eligible, self.total_frames // 2)
        constraint_sets = [
            dist_mask & angle_mask & size_mask & bp_mask & conf_mask,
            dist_mask & angle_mask & size_mask & bp_mask,
            dist_mask & angle_mask,
            dist_mask,
        ]
        for constraint in constraint_sets:
            eligible_count = np.sum(constraint)
            if eligible_count >= target_count:
                full_mask[two_inst_mask] = constraint
                return np.where(full_mask)[0].tolist()

        logger.warning(
            f"Masking yielded too few frames (<{target_count}). Falling back to ALL {np.sum(two_inst_mask)} two-instance frames. "
        )
        full_mask[two_inst_mask] = True
        return np.where(full_mask)[0].tolist()

    def _crop_rotate_and_export(self):
        I = self.pred_data_array.shape[1]
        angles = np.zeros((self.total_frames, I))
        crop_coords = calculate_pose_array_bbox(self.pred_data_array[self.eligible_frames], padding=15).astype(np.uint16)
        cutout_dim = int(np.percentile(crop_coords[..., 2:4] - crop_coords[..., 0:2], 90))

        if cutout_dim % 2:
            cutout_dim += 1

        angles[self.eligible_frames] = np.rad2deg(calculate_pose_array_rotations(self.pred_data_array[self.eligible_frames], self.anglemap))  # (F, I)

        ca = Cutout_Augments(
            centroids=self.centroids,
            cutout_dim=cutout_dim,
            angle_array=angles,
            grayscaling=True
            )
        co = Frame_Exporter_Threaded(
            video_filepath=self.extractor.get_video_filepath(),
            tm=self.tm,
            output_folder=self.temp_dir,
            frame_list=self.eligible_frames,
        )
        co.extract_frames(ca)

    def _run_contrain_magic(self, ambiguous_frames: List[int]):

        from utils.torch import Crop_Dataset, Embedding_Visualizer, Contrastive_Trainer

        embedding_filepath = os.path.join(self.temp_dir, 'embeddings.npz')
        if os.path.isfile(embedding_filepath):
            logger.info(f"Auto loading embedding file at {embedding_filepath}")
            cache = np.load(embedding_filepath)
            embeddings = cache['embeddings']
            motion_ids = cache['motion_ids'].tolist()
            frame_indices = cache['frame_indices'].tolist()
        else:
            from core.io import Cutout_Dataloader
            self._crop_rotate_and_export()
            loader = Cutout_Dataloader(self.temp_dir)
            crops, motion_ids, frame_indices, _, _ = loader.load_paired_tracks(n_mice=2)
            if len(crops) == 0:
                raise FileNotFoundError("[TF] No crops loaded. Check your folder path and filename format.")
            cds = Crop_Dataset(crops, motion_ids, frame_indices)
            trainer = Contrastive_Trainer()
            trainer.train(
                dataset=cds,
                batch_size=self.emp.batch_size,
                epochs=self.emp.epochs,
                max_triplet=self.emp.triplets,
                lr=self.emp.lr,
                )
            embeddings = trainer.extract_embeddings(cds)
            np.savez(embedding_filepath, embeddings=embeddings,
                     motion_ids=motion_ids, frame_indices=frame_indices)
        kmeans = KMeans(n_clusters=2, random_state=42)
        visual_labels = kmeans.fit_predict(embeddings)
        vis = Embedding_Visualizer(embeddings, motion_ids, frame_indices, visual_labels)
        tsne_pix = vis.plot_tsne_combined()
        agreement_pix, stable_swap_candidates = vis.plot_agreement_timeline()
        tsne_pix.save(os.path.join(self.temp_dir, "tsne_combined.png"))
        agreement_pix.save(os.path.join(self.temp_dir, "agreement_timeline.png"))
        pix_dialog = Dual_Pixmap_Dialog(agreement_pix, tsne_pix, max_height=480, parent=self.main)
        pix_dialog.show()
        stable_spans = indices_to_spans(stable_swap_candidates)
        for start, end in stable_spans:
            start_idx = max(0, start - 10)
            fin_idx = min(end + 11, self.total_frames-1)
            amb_in_range = [f for f in ambiguous_frames if f >= start and f < end]
            if self.avtomat:
                swap_orders = []
                if not amb_in_range:
                    swap_orders.append((start, "t"))
                    swap_orders.append((end, "t"))
                else:
                    swap_orders.append((amb_in_range[0], "t"))
                    if len(amb_in_range) >= 2:
                        swap_orders.append((amb_in_range[-1], "t"))
                    else:
                        swap_orders.append((end, "t"))
            else:
                swap_orders = self._launch_dialog_swap(amb_in_range, start_idx, fin_idx)

            if swap_orders == "exit":
                return
            else:
                for frame_idx, order in swap_orders:
                    if order == "i":
                        self.pred_data_array = swap_track(self.pred_data_array, frame_idx)
                    else:
                        self.pred_data_array = swap_track(self.pred_data_array, frame_idx,
                                                          swap_range=list(range(frame_idx, fin_idx)))

        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)
        if os.path.isfile(embedding_filepath):
            os.remove(embedding_filepath)

    def _correct_frame_with_hungarian(self, frame_idx: int) -> bool:
        n_obs = len(get_instances_on_current_frame(self.pred_data_array, frame_idx))
        if n_obs == 0:
            return True
        ref_pos = self._get_ref_pos(frame_idx)
        if n_obs == 1:
            result = self._handle_single_observation(frame_idx, ref_pos)
        elif n_obs == 2:
            result = self._handle_two_observations(frame_idx, ref_pos)
        frame_inst_after_update = get_instances_on_current_frame(self.pred_data_array, frame_idx)
        self.last_known_pos[frame_inst_after_update] = self.centroids[frame_idx, frame_inst_after_update]
        return result

    def _get_ref_pos(self, frame_idx: int, force_kalman: bool = False) -> np.ndarray:
        current_obs = self.centroids[frame_idx]
        kalman_refs = np.full((2, 2), np.nan)
        kalman_available = True
        for inst_idx in [0, 1]:
            if self.kalman_filters[inst_idx] is not None:
                state = self.kalman_filters[inst_idx].predict()
                kalman_refs[inst_idx] = state[:2]
            else:
                kalman_available = False
        logger.debug(f"[KALMAN] Kalman refs: Inst 0: ({kalman_refs[0, 0]:.1f}, {kalman_refs[0, 1]:.1f}), Inst 1: ({kalman_refs[1, 0]:.1f}, {kalman_refs[1, 1]:.1f})")
        if (kalman_available or force_kalman) and not np.isnan(kalman_refs).any():
            min_cost_kalman = self._compute_min_assignment_cost(kalman_refs, current_obs)
            if min_cost_kalman <= self.KALMAN_MAX_ERROR:
                logger.debug(f"[KALMAN] Kalman matching succeeded (min_cost={min_cost_kalman:.1f})")
                return kalman_refs
        logger.debug(f"[KALMAN] Kalman matching failed (min_cost={min_cost_kalman:.1f} > {self.KALMAN_MAX_ERROR}), falling back to last known positions")
        last0 = self.last_known_pos[0]
        last1 = self.last_known_pos[1]
        last0_str = f"({last0[0]:.1f}, {last0[1]:.1f})" if not np.isnan(last0).any() else "NaN"
        last1_str = f"({last1[0]:.1f}, {last1[1]:.1f})" if not np.isnan(last1).any() else "NaN"
        logger.debug(f"[KALMAN] Last known refs: Inst 0: {last0_str}, Inst 1: {last1_str}")
        return self.last_known_pos

    def _compute_min_assignment_cost(self, refs: np.ndarray, obs: np.ndarray) -> float:
        valid_obs = ~np.isnan(obs[:, 0])
        valid_obs_indices = np.where(valid_obs)[0]
        n_obs = np.sum(valid_obs)
        if n_obs == 1:
            obs_idx = valid_obs_indices[0]
            observation = obs[obs_idx]
            cost_0 = cost_1 = 1e6
            if not np.isnan(refs[0]).any():
                cost_0 = np.linalg.norm(refs[0] - observation)
            if not np.isnan(refs[1]).any():
                cost_1 = np.linalg.norm(refs[1] - observation)
            min_cost = min(cost_0, cost_1)
        elif n_obs == 2:
            cost_no_swap = cost_swap = 0.0
            cost_no_swap += np.linalg.norm(refs[0] - obs[0]) if not np.isnan(refs[0]).any() else 1e6
            cost_no_swap += np.linalg.norm(refs[1] - obs[1]) if not np.isnan(refs[1]).any() else 1e6
            cost_swap += np.linalg.norm(refs[0] - obs[1]) if not np.isnan(refs[0]).any() else 1e6
            cost_swap += np.linalg.norm(refs[1] - obs[0]) if not np.isnan(refs[1]).any() else 1e6
            min_cost = min(cost_no_swap, cost_swap)
        else:
            return 1e6
        return min_cost

    def _handle_single_observation(self, frame_idx: int, ref_positions: np.ndarray, compare_only: bool = False) -> bool:
        obs_idx = get_instances_on_current_frame(self.pred_data_array, frame_idx)[0]
        obs_pos = self.centroids[frame_idx, obs_idx]
        costs = []
        candidates = []
        for candidate_idx in [0, 1]:
            if not np.isnan(ref_positions[candidate_idx, 0]):
                d = np.linalg.norm(obs_pos - ref_positions[candidate_idx])
                costs.append(d)
                candidates.append(candidate_idx)
        if not candidates:
            return False
        if len(candidates) == 1:
            if not compare_only:
                assigned_idx = candidates[0]
                if assigned_idx != obs_idx:
                    self._swap_ids_in_frame(frame_idx)
                    logger.debug(f"Frame {frame_idx}: Single obs assigned to {assigned_idx} (was {obs_idx}), swapped")
                self._update_kalman_with_observation(frame_idx, [assigned_idx], [assigned_idx])
            cost = costs[0]
            if cost <= self.MAX_DIST_THRESHOLD:
                return True
            else:
                logger.debug(f"Frame {frame_idx}: Single obs assignment cost too large: (cost: {cost:.1f} > {self.MAX_DIST_THRESHOLD})")
                return False
        else:
            c0, c1 = costs
            winner = candidates[0] if c0 < c1 else candidates[1]
            if winner != obs_idx:
                self._swap_ids_in_frame(frame_idx)
                logger.debug(f"Frame {frame_idx}: Single obs winner {winner} (was {obs_idx}), swapped")
                self._update_kalman_with_observation(frame_idx, [winner], [winner])
            if abs(c0 - c1) < self.AMBIGUITY_THRESHOLD:
                logger.debug(f"Frame {frame_idx}: Ambiguous single observation (costs {c0:.1f}, {c1:.1f})")
                return False
            return True

    def _handle_two_observations(self, frame_idx: int, ref_positions: np.ndarray, get_min_cost: bool = False) -> bool | float:
        valid_ref = np.where(np.all(~np.isnan(ref_positions), axis=-1))[0]
        n_ref = len(valid_ref)
        if n_ref == 1:
            ref_idx = valid_ref[0]
            cost_swap = np.linalg.norm(self.centroids[frame_idx, 1 - ref_idx] - ref_positions[ref_idx])
            cost_no_swap = np.linalg.norm(self.centroids[frame_idx, ref_idx] - ref_positions[ref_idx])
        elif n_ref == 2:
            cost_matrix = np.full((2, 2), np.inf)
            for obs_idx in [0, 1]:
                for candidate_idx in [0, 1]:
                    d = np.linalg.norm(self.centroids[frame_idx, obs_idx] - ref_positions[candidate_idx])
                    cost_matrix[obs_idx, candidate_idx] = d
            cost_matrix_lap = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)
            cost_swap = cost_matrix_lap[0, 1] + cost_matrix_lap[1, 0]
            cost_no_swap = cost_matrix_lap[0, 0] + cost_matrix_lap[1, 1]
        else:
            return False
        if get_min_cost:
            return min(cost_swap, cost_no_swap)
        needs_swap = cost_swap < cost_no_swap
        if needs_swap:
            self._swap_ids_in_frame(frame_idx)
            logger.debug(f"Frame {frame_idx}: Two-obs swap needed (cost_swap={cost_swap:.1f} < no_swap={cost_no_swap:.1f})")
        self._update_kalman_with_observation(frame_idx, [0, 1], [0, 1])
        if abs(cost_swap - cost_no_swap) < self.AMBIGUITY_THRESHOLD:
            logger.debug(f"Frame {frame_idx}: Ambiguous two-obs match (Δcost={abs(cost_swap - cost_no_swap):.1f})")
            return False
        return True

    def _update_kalman_with_observation(self, frame_idx: int, observed_inst_ids: List[int], assigned_ids: List[int]):
        for obs_idx, global_idx in zip(observed_inst_ids, assigned_ids):
            z = self.centroids[frame_idx, obs_idx]
            if np.isnan(z).any():
                continue
            kf = self.kalman_filters[global_idx]
            if kf is None:
                self.kalman_filters[global_idx] = Kalman(initial_state=[z[0], z[1], 0.0, 0.0])
                self.kalman_failure_count[global_idx] = 0
            else:
                pred_state = kf.predict()
                pred_pos = pred_state[:2]
                error = np.linalg.norm(z - pred_pos)
                logger.debug(
                    f"[KALMAN] Global ID {global_idx} ← obs col {obs_idx} at ({z[0]:.1f}, {z[1]:.1f}), "
                    f"pred=({pred_pos[0]:.1f}, {pred_pos[1]:.1f}), error={error:.1f}"
                )
                if error > self.KALMAN_MAX_ERROR:
                    self.kalman_failure_count[global_idx] += 1
                    logger.debug(f"[KALMAN] Kalman error for ID {global_idx}: {error:.1f} > {self.KALMAN_MAX_ERROR}")
                    if self.kalman_failure_count[global_idx] >= self.KALMAN_RESET_THRESHOLD:
                        logger.debug(f"[KALMAN] Resetting Kalman filter for ID {global_idx} after {self.kalman_failure_count[global_idx]} failures")
                        self.kalman_filters[global_idx] = Kalman(initial_state=[z[0], z[1], 0.0, 0.0])
                        self.kalman_failure_count[global_idx] = 0
                else:
                    kf.update(z)
                    self.kalman_failure_count[global_idx] = 0

    def _clear_instance_at_frame(self, frame_idx: int, inst_idx: int):
        self.pred_data_array[frame_idx, inst_idx, :] = np.nan
        self.centroids[frame_idx, inst_idx, :] = np.nan
        self.inst_count_per_frame[frame_idx] = len(get_instances_on_current_frame(self.pred_data_array, frame_idx))

    def _swap_ids_in_frame(self, frame_idx: int):
        self.pred_data_array[frame_idx] = self.pred_data_array[frame_idx, ::-1]
        self.centroids[frame_idx] = self.centroids[frame_idx, ::-1]

    def _launch_dialog_swap(
        self,
        ambiguous_list: List[int],
        event_start_idx=None,
        event_end_idx=None
    ) -> List[Tuple[int, str]] | str:
        if self.avtomat and ambiguous_list:
            return [(ambiguous_list[0], "t")]
        dialog = Swap_Correction_Dialog(
            dlc_data=self.dlc_data,
            extractor=self.extractor,
            pred_data_array=self.pred_data_array,
            ambiguous_frames=ambiguous_list,
            event_start_idx=event_start_idx,
            event_end_idx=event_end_idx,
            parent=self.main
        )
        dialog.exec()
        if dialog.cancelled:
            return "exit"
        else:
            return dialog.user_decisions