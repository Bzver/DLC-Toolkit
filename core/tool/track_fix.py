import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict

from .reviewer import Swap_Correction_Dialog
from core.io import Frame_Extractor, Temp_Manager, Frame_Exporter_Threaded
from ui import Dual_Pixmap_Dialog
from utils.track import Kalman, swap_track
from utils.pose import (
    calculate_pose_centroids, calculate_pose_array_rotations, calculate_pose_array_bbox,
    outlier_rotation, outlier_size, outlier_bodypart,
    )
from utils.helper import get_instance_count_per_frame, get_instances_on_current_frame, indices_to_spans, array_to_iterable_runs
from utils.dataclass import Loaded_DLC_Data, Cutout_Augments, Emb_Params
from utils.logger import logger


class Track_Fixer:
    KALMAN_RESET_THRESHOLD = 3
    USER_DIALOG_COOLDOWN = 20
    AMBIGUITY_THRESHOLD = 0.15
    MAX_DIST_THRESHOLD = 1.2
    KALMAN_MAX_ERROR = 80.0
    VOTE_WINDOW_SIZE = 10

    def __init__(
        self,
        pred_data_array: np.ndarray,
        dlc_data: Loaded_DLC_Data,
        extractor: Frame_Extractor,
        anglemap: Dict[str, int],
        emp: Emb_Params,
        worker_num: int = 8,
        skip_sweep: bool = False,
        kp_smooth: bool = True,
        blob_array: np.ndarray|None = None,
        avtomat: bool = False,
        skip_contrast: bool = False,
        use_kalman: bool = True,
        use_cache: bool = True,
        parent = None
    ):
        if pred_data_array.shape[1] != 2:
            raise NotImplementedError("Track_Fixer supports exactly 2 instances.")
        self.pred_data_array = pred_data_array.copy()
        self.dlc_data = dlc_data
        self.extractor = extractor
        self.anglemap = anglemap
        self.emp = emp
        self.worker_num = worker_num
        self.skip_sweep = skip_sweep
        self.kp_smooth = kp_smooth
        self.blob_array = blob_array
        self.avtomat = avtomat
        self.skip_contrast = skip_contrast
        self.use_kalman = use_kalman
        self.main = parent
        self.total_frames = self.pred_data_array.shape[0]
        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)
        self.inst_count_per_frame = get_instance_count_per_frame(self.pred_data_array)
        self.mice_length = self._get_mice_length()

        self.kalman_filters = [None, None] if use_kalman else []
        self.last_known_pos = np.full((2, 2), np.nan)
        self.kalman_failure_count = [0, 0] if use_kalman else []
        
        tm = Temp_Manager(self.extractor.get_video_filepath())
        self.temp_dir = tm.create("track", use_existing=use_cache)
        self.use_cache = use_cache
        self.ambiguous_frames = []
        self.id_lock = self.blob_array is not None
        self.last_id_locker = 0
        self.locker_stable_swap = []
        self.locked_idx = 0

        self.seg_list = []

        if self.skip_contrast and not self.id_lock:
            raise RuntimeError("Lock-ID mode needs animal counter data. Please count animals before running track correction with skip_contrast enabled.")
        if self.skip_contrast and self.skip_sweep:
            raise RuntimeError("I don't know how you did it, cheers!")

    def track_correction(self, start_idx: int = 0, end_idx: int = -1) -> np.ndarray:
        if end_idx == -1:
            end_idx = self.total_frames

        if self.id_lock:
            self.co_single_array = np.array((self.blob_array[:, 0] == 1) & (self.inst_count_per_frame <= 1))
            for start, end, value in array_to_iterable_runs(self.co_single_array):
                if end - start + 1 < 5 and value:
                    self.co_single_array[start:end+1] = False

        self._motion_sweep(start_idx, end_idx)

        if self.skip_contrast:
            if self.blob_array is not None and self.locker_stable_swap:
                self._process_stable_swap_candidates(self.locker_stable_swap)
            elif self.ambiguous_frames:
                swap_orders = self._launch_dialog_swap(self.ambiguous_frames, start_idx, end_idx)
                self._execute_swap_orders(swap_orders, self.total_frames-1)
            return self.pred_data_array

        qc_mask = self._find_eligible_frames(start_idx, end_idx)

        crop_coords = calculate_pose_array_bbox(self.pred_data_array[qc_mask], padding=15).astype(np.uint16)
        cutout_dim = int(np.percentile(crop_coords[..., 2:4] - crop_coords[..., 0:2], 90))

        if cutout_dim % 2:
            cutout_dim += 1
    
        angles = np.zeros((self.total_frames, self.dlc_data.instance_count))
        angles[qc_mask] = np.rad2deg(calculate_pose_array_rotations(self.pred_data_array[qc_mask], self.anglemap))

        ca = Cutout_Augments(
            centroids=self.centroids,
            cutout_dim=cutout_dim,
            angle_array=angles,
            grayscaling=True
            )

        self._crop_rotate_and_export(ca)
        self._run_contrain_magic(start_idx, end_idx)
        self._run_kp_smoothing()
        return self.pred_data_array

    def _motion_sweep(self, start_idx, end_idx):
        if self.skip_sweep:
            return
        
        if self.id_lock:
            try:
                lock_mask = (self.inst_count_per_frame == 1) & (self.blob_array[:,0] == 1)
                lock_frame_indices = np.where(lock_mask)[0][:10]
            except:
                pass
            else:
                num_0 = np.sum(~np.isnan(self.centroids[lock_frame_indices, 0, 0]))
                self.locked_idx = 0 if num_0 >= 5 else 1

        pbar = tqdm(total=end_idx - start_idx, desc=f"Motion Sweep In Progress", leave=False, ncols=200)        
        try:
            for f in range(start_idx, end_idx):
                logger.debug(f"--------- frame {f} ---------")
                pbar.update(1)
                success = self._correct_frame_with_hungarian(f)
                if not success:
                    self.ambiguous_frames.append(f)
                    logger.debug(f"[TF] Ambiguous match, added to backtrack list")
        finally:
            pbar.close()

    def _run_kp_smoothing(self):
        if not self.kp_smooth:
            return
        
        n_frames, n_instances, n_values = self.pred_data_array.shape
        n_keypoints = n_values // 3
        window_length = min(11, n_frames // 4)
        if window_length % 2 == 0:
            window_length += 1
        polyorder = min(3, window_length - 1)
        
        smoothed = np.full_like(self.pred_data_array, np.nan)
        
        for inst_idx in range(n_instances):
            for kp_idx in range(n_keypoints):
                x_idx, y_idx, c_idx = kp_idx * 3, kp_idx * 3 + 1, kp_idx * 3 + 2

                x_traj = self.pred_data_array[:, inst_idx, x_idx]
                y_traj = self.pred_data_array[:, inst_idx, y_idx]
                conf_traj = self.pred_data_array[:, inst_idx, c_idx]

                conf_mask = conf_traj >= 0.3
                x_weighted = np.where(conf_mask, x_traj, np.nan)
                y_weighted = np.where(conf_mask, y_traj, np.nan)

                smoothed_x = self._savgol_smooth(x_weighted, window_length, polyorder)
                smoothed_y = self._savgol_smooth(y_weighted, window_length, polyorder)

                blend_factor = np.clip(conf_traj, 0, 1)[:, np.newaxis]
                smoothed[:, inst_idx, x_idx] = blend_factor.squeeze() * x_traj + (1 - blend_factor.squeeze()) * smoothed_x
                smoothed[:, inst_idx, y_idx] = blend_factor.squeeze() * y_traj + (1 - blend_factor.squeeze()) * smoothed_y
                smoothed[:, inst_idx, c_idx] = conf_traj

        self.pred_data_array = smoothed
        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)

    def _savgol_smooth(self, trajectory: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
        result = trajectory.copy()
        valid_mask = ~np.isnan(trajectory)
        
        if np.sum(valid_mask) < window_length:
            return result

        filled = trajectory.copy()
        if not valid_mask.all():
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 1:
                filled[~valid_mask] = np.interp(np.where(~valid_mask)[0], valid_idx, trajectory[valid_mask])
        try:
            filtered = savgol_filter(filled, window_length, polyorder, mode='nearest')
        except ValueError:
            return result
        
        result[valid_mask] = filtered[valid_mask]
        return result

    def _find_eligible_frames(
            self,
            start_idx: int,
            end_idx: int,
            inst_dist_threshold: float = 0.8,
            size_threshold: Tuple[float, float] = (0.3, 2.5),
            bp_threshold: int = 4,
            twist_angle_threshold: float = 50.0,
            ) -> np.ndarray:

        instance_count = get_instance_count_per_frame(self.pred_data_array)
        two_inst_mask = instance_count == 2
        
        if not np.any(two_inst_mask):
            raise ValueError("No frame with two insts, cannot perform contrastive learning.")

        two_inst_array = self.pred_data_array[two_inst_mask]
        centroids_two_inst = self.centroids[two_inst_mask]

        dists = np.linalg.norm(centroids_two_inst[:, 1, :] - centroids_two_inst[:, 0, :], axis=-1)
        dist_mask = dists > inst_dist_threshold * self.mice_length

        angle_mask = ~np.any(outlier_rotation(two_inst_array, self.anglemap, twist_angle_threshold), axis=-1)
        size_mask = ~np.any(outlier_size(two_inst_array, *size_threshold), axis=-1)
        bp_mask = ~np.any(outlier_bodypart(two_inst_array, bp_threshold), axis=-1)

        frame_valid_mask = np.zeros(self.total_frames, dtype=bool)
        frame_valid_mask[two_inst_mask] = dist_mask & angle_mask & size_mask & bp_mask

        segmentation_mask = two_inst_mask.copy()
        if self.ambiguous_frames:
            segmentation_mask[self.ambiguous_frames] = False # Break segment at ambiguous frames when there are ambiguous frames
            frame_valid_mask[self.ambiguous_frames] = False
        else:
            segmentation_mask[two_inst_mask] = dist_mask

        for seg_start, seg_end in indices_to_spans(np.where(segmentation_mask)[0]):
            if seg_end < start_idx or seg_start > end_idx:
                continue
            
            seg_start = max(seg_start, start_idx)
            seg_end = min(seg_end, end_idx-1)

            seg_frames = [i for i in range(seg_start, seg_end + 1) if frame_valid_mask[i]]
            if len(seg_frames) <= 3:
                continue
            self.seg_list.append(seg_frames)

        return frame_valid_mask

    def _crop_rotate_and_export(self, ca:Cutout_Augments):
        if self.use_cache:
            logger.info("[TF] Scanning for existing npz chunk in the folder,")

        frame_list = []
        for seg in self.seg_list:
            frame_list.extend(seg)

        co = Frame_Exporter_Threaded(
            video_filepath=self.extractor.get_video_filepath(),
            output_folder=self.temp_dir,
            frame_list=frame_list,
            max_workers=self.worker_num,
        )
        # self._log_seg_list()

        co.extract_frames(ca, self.seg_list, self.use_cache)
    
    def _run_contrain_magic(self, start_idx, end_idx, segment_samples: int = 80):

        from utils.torch import Crop_Dataset, Embedding_Visualizer, Contrastive_Trainer, Cutout_Dataloader

        dataloader = Cutout_Dataloader(folder_path=self.temp_dir, seg_list=self.seg_list)

        trainer = Contrastive_Trainer()
        if self.emp.pretrained_model_path and os.path.exists(self.emp.pretrained_model_path):
            logger.info(f"[TF] Loading pretrained model: {self.emp.pretrained_model_path}")
            trainer.load_checkpoint(self.emp.pretrained_model_path, strict=False)

        train_seg_indices = dataloader.select_training_segments(ratio=1.0)

        logger.info(f"[TF] Loading {len(train_seg_indices)} training segments...")
        train_datasets = dataloader.load_all_segments_for_training(train_seg_indices)

        trainer.double_train(dataset=train_datasets, emp=self.emp)

        if self.emp.save_model:
            model_path = os.path.join(self.extractor.get_video_dir(), f"{self.extractor.get_video_name(no_ext=True)}_contrastive_trained.pth")
            trainer.save_checkpoint(model_path)
            logger.info(f"[TF] Model saved to {model_path}")

        logger.info(f"[TF] Running inference on ALL segments ({segment_samples} frames per segment)")
        segment_embeddings = []
        segment_frame_indices = []
        segment_motion_ids = []

        for seg_idx in tqdm(range(len(dataloader.segment_index)), desc="Inferring Segments", ncols=200):
            images, frames, _ = dataloader.load_segment_samples(seg_idx, n_samples=segment_samples)

            crops = [images[i, mouse_idx] for i in range(len(frames)) for mouse_idx in range(2)]
            mids_flat = [mid for _ in frames for mid in [0, 1]]
            frames_flat = [f for f in frames for _ in range(2)]
            
            ds = Crop_Dataset(crops, mids_flat, frames_flat, is_ir=True)
            
            embs = trainer.extract_embeddings(ds)
            
            segment_embeddings.append(embs)
            segment_frame_indices.append(frames_flat)
            segment_motion_ids.append(mids_flat)
    
        all_embeddings = np.vstack(segment_embeddings)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        visual_labels = kmeans.fit_predict(all_embeddings)

        centroids = kmeans.cluster_centers_ 
        dist_to_c0 = np.linalg.norm(all_embeddings - centroids[0], axis=1)
        dist_to_c1 = np.linalg.norm(all_embeddings - centroids[1], axis=1)

        margin = np.abs(dist_to_c0 - dist_to_c1)
        margin_scale = np.percentile(margin, 75)
        confidence = 1 / (1 + np.exp(-(margin - margin_scale) / (margin_scale * 0.5)))

        segment_confidence = []
        idx = 0
        for seg_emb in segment_embeddings:
            n = len(seg_emb)
            segment_confidence.append(confidence[idx:idx+n])
            idx += n

        logger.info("[TF] Checking for low-confidence segments...")
        confidence_threshold = 0.5
        min_confident_ratio = 0.4

        for seg_idx in range(len(segment_embeddings)):
            confidences = segment_confidence[seg_idx]
            confident_ratio = np.sum(confidences >= confidence_threshold) / len(confidences)
            
            if confident_ratio < min_confident_ratio:
                logger.debug(f"[TF] Segment {seg_idx}: low confidence ({confident_ratio:.1%}), re-mining with more samples")

                images, frames, _ = dataloader.load_segment_samples(seg_idx, n_samples=2*segment_samples)
                
                crops = [images[i, mouse_idx] for i in range(len(frames)) for mouse_idx in range(2)]
                mids_flat = [mid for _ in frames for mid in [0, 1]]
                frames_flat = [f for f in frames for _ in range(2)]
                
                ds = Crop_Dataset(crops, mids_flat, frames_flat, is_ir=True)
                new_embs = trainer.extract_embeddings(ds)
                
                segment_embeddings[seg_idx] = new_embs
                segment_frame_indices[seg_idx] = frames_flat
                segment_motion_ids[seg_idx] = mids_flat
    
        logger.info("[TF] Re-clustering with updated embeddings...")

        all_embeddings = np.vstack(segment_embeddings)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        visual_labels = kmeans.fit_predict(all_embeddings)

        centroids = kmeans.cluster_centers_ 
        dist_to_c0 = np.linalg.norm(all_embeddings - centroids[0], axis=1)
        dist_to_c1 = np.linalg.norm(all_embeddings - centroids[1], axis=1)

        margin = np.abs(dist_to_c0 - dist_to_c1)
        margin_scale = np.percentile(margin, 75)
        confidence = 1 / (1 + np.exp(-(margin - margin_scale) / (margin_scale * 0.5)))

        segment_confidence = []
        idx = 0
        for seg_emb in segment_embeddings:
            n = len(seg_emb)
            segment_confidence.append(confidence[idx:idx+n])
            idx += n

        segment_visual_labels = []
        idx = 0
        for seg_emb in segment_embeddings:
            n = len(seg_emb)
            segment_visual_labels.append(visual_labels[idx:idx+n])
            idx += n

        vis = Embedding_Visualizer(
            segment_embeddings, 
            segment_motion_ids, 
            segment_frame_indices,
            visual_labels=segment_visual_labels,
            assignment_confidence=segment_confidence,
            total_frames=self.total_frames,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        tsne_pix = vis.plot_tsne_combined()
        tsne_pix.save(os.path.join(self.temp_dir, "tsne_combined.png"))

        agreement_pix, stable_swap_candidates, diagnosis_timeline = vis.plot_agreement_timeline()
        agreement_pix.save(os.path.join(self.temp_dir, "agreement_timeline.png"))

        df_diagnosis = pd.DataFrame(diagnosis_timeline, columns=[
            "frame_agreement_score", "segment_agreement_score", "segment_type_class"])

        df_diagnosis.insert(0, "frame_index", np.arange(self.total_frames))
        output_csv_path = os.path.join(self.temp_dir, "diagnosis_timeline.csv")
        try:
            df_diagnosis.to_csv(output_csv_path, index=False)
            logger.info(f"[VIS] Diagnosis timeline exported to {output_csv_path}")
        except Exception as e:
            logger.error(f"[VIS] Failed to export diagnosis timeline: {e}")
        
        pix_dialog = Dual_Pixmap_Dialog(agreement_pix, tsne_pix, max_height=480, parent=self.main)
        pix_dialog.show()
        logger.info("[TF] Visualization complete.")
        
        if self.id_lock and self.co_single_array is not None:
            stable_swap = [f for f in stable_swap_candidates if not self.co_single_array[f]]
            stable_swap_candidates = stable_swap
        
        self._process_stable_swap_candidates(stable_swap_candidates)
        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)

    def _process_stable_swap_candidates(self, stable_swap_candidates):
        stable_spans = indices_to_spans(stable_swap_candidates)
        for start, end in stable_spans:
            start_idx = max(0, start - 10)
            fin_idx = min(end + 11, self.total_frames-1)
            amb_in_range = [f for f in self.ambiguous_frames if f >= start and f <= end]
            if self.avtomat:
                swap_orders = []
                if not amb_in_range:
                    swap_orders.append((start, "t"))
                    swap_orders.append((end, "t"))
                elif amb_in_range[0] >= end - 1:
                    return
                else:
                    swap_orders.append((amb_in_range[0], "t"))
                    if len(amb_in_range) >= 2:
                        swap_orders.append((amb_in_range[-1], "t"))
                    else:
                        swap_orders.append((end, "t"))
            else:
                swap_orders = self._launch_dialog_swap(amb_in_range, start_idx, fin_idx)

            if not self._execute_swap_orders(swap_orders, fin_idx):
                return

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

        if self.use_kalman:
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

        if compare_only:
            assert len(candidates) == 1, "Compare_only arg only supports 1vs1 comparison!"
            cost = costs[0]
            return cost <= self.MAX_DIST_THRESHOLD * self.mice_length

        if self.id_lock and self.co_single_array[frame_idx] and self.co_single_array[max(0, frame_idx-1)]: #Continuous co-single block, not checkpoint
            logger.debug(f"Frame {frame_idx}: Blob array supplied, locking single obs to instance 0.")
            if obs_idx != self.locked_idx:
                self._swap_ids_in_frame(frame_idx)

            self.last_id_locker = frame_idx
            self._update_kalman_with_observation(frame_idx, [self.locked_idx], [self.locked_idx])
            return True

        if not candidates:
            return False
        if len(candidates) == 1:
            assigned_idx = candidates[0]
            if self.id_lock and self.co_single_array[frame_idx] and assigned_idx != self.locked_idx: # It attempts to assign the wrong id at checkpoint
                self.ambiguous_frames.append(frame_idx)
                self.locker_stable_swap.extend(range(self.last_id_locker+1, frame_idx+1))
                if obs_idx != self.locked_idx:
                    self._swap_ids_in_frame(frame_idx)
                self._update_kalman_with_observation(frame_idx, [self.locked_idx], [self.locked_idx])
                return True
            if assigned_idx != obs_idx:
                self._swap_ids_in_frame(frame_idx)
                logger.debug(f"Frame {frame_idx}: Single obs assigned to {assigned_idx} (was {obs_idx}), swapped")
            self._update_kalman_with_observation(frame_idx, [assigned_idx], [assigned_idx])
            cost = costs[0]
            if cost <= self.MAX_DIST_THRESHOLD * self.mice_length:
                return True
            else:
                logger.debug(f"Frame {frame_idx}: Single obs assignment cost too large: (cost: {cost:.1f} > {self.MAX_DIST_THRESHOLD})")
                return False
        else:
            c0, c1 = costs
            winner = candidates[0] if c0 < c1 else candidates[1]
            if self.id_lock and self.blob_array[frame_idx, 0] == 1 and winner != self.locked_idx: # same
                if frame_idx - self.last_id_locker > 2 and np.sum(self.inst_count_per_frame[self.last_id_locker:frame_idx]==2) > 2:
                    self.ambiguous_frames.append(frame_idx)
                    self.locker_stable_swap.extend(range(self.last_id_locker+1, frame_idx+1))
                if obs_idx != self.locked_idx:
                    self._swap_ids_in_frame(frame_idx)
                self._update_kalman_with_observation(frame_idx, [self.locked_idx], [self.locked_idx])
                return True
            if winner != obs_idx:
                self._swap_ids_in_frame(frame_idx)
                logger.debug(f"Frame {frame_idx}: Single obs winner {winner} (was {obs_idx}), swapped")
                self._update_kalman_with_observation(frame_idx, [winner], [winner])
            if abs(c0 - c1)/(c0 + c1) < self.AMBIGUITY_THRESHOLD:
                logger.debug(f"Frame {frame_idx}: Ambiguous single observation (costs {c0:.1f}, {c1:.1f})")
                return False
        return True

    def _handle_two_observations(self, frame_idx: int, ref_positions: np.ndarray) -> bool | float:
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
        needs_swap = cost_swap < cost_no_swap
        if needs_swap:
            self._swap_ids_in_frame(frame_idx)
            logger.debug(f"Frame {frame_idx}: Two-obs swap needed (cost_swap={cost_swap:.1f} < no_swap={cost_no_swap:.1f})")
        self._update_kalman_with_observation(frame_idx, [0, 1], [0, 1])
        if abs(cost_swap - cost_no_swap)/(cost_swap + cost_no_swap) < self.AMBIGUITY_THRESHOLD:
            logger.debug(f"Frame {frame_idx}: Ambiguous two-obs match (Δcost={abs(cost_swap - cost_no_swap):.1f})")
            return False
        return True

    def _update_kalman_with_observation(self, frame_idx: int, observed_inst_ids: List[int], assigned_ids: List[int]):
        if not self.use_kalman:
            return

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

    def _get_mice_length(self) -> float:
        head_idx = self.anglemap["head_idx"]
        all_head = self.pred_data_array[..., head_idx * 3:head_idx * 3 + 2]
        tail_idx = self.anglemap["tail_idx"]
        all_tail = self.pred_data_array[..., tail_idx * 3:tail_idx * 3 + 2]
        return np.nanmedian(np.linalg.norm(all_head - all_tail, axis=-1))

    def _swap_ids_in_frame(self, frame_idx: int):
        self.pred_data_array[frame_idx] = self.pred_data_array[frame_idx, ::-1]
        self.centroids[frame_idx] = self.centroids[frame_idx, ::-1]

    def _execute_swap_orders(self, swap_orders:List[Tuple[int, str]], fin_idx:int):
        if swap_orders == "exit":
            return False

        for frame_idx, order in swap_orders:
            if order == "i":
                self.pred_data_array = swap_track(self.pred_data_array, frame_idx)
            else:
                self.pred_data_array = swap_track(self.pred_data_array, frame_idx, swap_range=list(range(frame_idx, fin_idx)))
        
        return True

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
        
    def _log_seg_list(self):
        import json
        raw_seg = []
        seg_len = []
        for seg in self.seg_list:
            raw_seg.append((seg[0], seg[-1]))
            seg_len.append(seg[-1]-seg[0]+1)

        unique_elements, counts = np.unique(seg_len, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_elements = unique_elements[sorted_indices]

        data = {
            "median": int(np.median(seg_len)),
            "mean": int(np.mean(seg_len)),
            "min": int(np.min(seg_len)),
            "max": int(np.max(seg_len)),
            "len_occurences": sorted_elements.tolist(),
            "raw_seg": raw_seg
        }
        with open ("debug.json", "w") as file:
            json.dump(data, file)