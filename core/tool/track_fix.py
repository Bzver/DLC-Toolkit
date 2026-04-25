import os
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict

from .reviewer import Swap_Correction_Dialog
from core.io import Frame_Extractor, Temp_Manager, Frame_Exporter_Threaded
from ui import Dual_Pixmap_Dialog
from utils.track import swap_track
from utils.pose import (
    calculate_pose_array_rotations, calculate_pose_dim, calculate_anatomical_centers,
    outlier_rotation, outlier_size, outlier_bodypart, outlier_duplicate, outlier_removal
    )
from utils.helper import get_instance_count_per_frame, get_instances_on_current_frame, indices_to_spans, array_to_iterable_runs
from utils.dataclass import Loaded_DLC_Data, Cutout_Augments, Emb_Params
from utils.logger import logger


class Track_Fixer:
    USER_DIALOG_COOLDOWN = 20
    AMBIGUITY_THRESHOLD = 0.15
    MAX_DIST_THRESHOLD = 1.2
    VOTE_WINDOW_SIZE = 10
    MAX_POS_AGE_BEFORE_INVALIDATING = 10

    def __init__(
        self,
        pred_data_array: np.ndarray,
        dlc_data: Loaded_DLC_Data,
        extractor: Frame_Extractor,
        anglemap: Dict[str, int],
        emp: Emb_Params,
        worker_num: int = 8,
        skip_sweep: bool = False,
        blob_array: np.ndarray|None = None,
        force_locked_id: int = -1,
        avtomat: bool = False,
        skip_contrast: bool = False,
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
        self.blob_array = blob_array
        self.forced_lock = force_locked_id > 0
        self.avtomat = avtomat
        self.skip_contrast = skip_contrast
        self.main = parent
        self.total_frames = self.pred_data_array.shape[0]

        duplicate_mask = outlier_duplicate(self.pred_data_array)
        single_point_mask = outlier_bodypart(self.pred_data_array)
        self.pred_data_array = outlier_removal(self.pred_data_array, duplicate_mask|single_point_mask)

        self.centroids = calculate_anatomical_centers(self.pred_data_array, self.anglemap)
        self.inst_count_per_frame = get_instance_count_per_frame(self.pred_data_array)
        self.mice_length = self._get_mice_length()

        self.last_known_pos = np.full((2, 2), np.nan)
        self.last_update = np.zeros((2, 2), dtype=np.uint32)
        
        tm = Temp_Manager(self.extractor.get_video_filepath())
        self.temp_dir = tm.create("track", use_existing=use_cache)
        self.use_cache = use_cache
        self.ambiguous_frames = []
        self.id_lock = self.blob_array is not None
        self.last_id_locker = 0
        self.locker_stable_swap = []
        self.locked_idx = force_locked_id if self.forced_lock else 0

        self.seg_list = []

        if self.skip_contrast and self.skip_sweep:
            raise RuntimeError("I don't know how you did it, cheers!")

    def track_correction(self, start_idx: int = 0, end_idx: int = -1) -> np.ndarray:
        if end_idx == -1:
            end_idx = self.total_frames

        if self.id_lock:
            self.co_single_array = np.array((self.blob_array[:, 0] <= 1) & (self.inst_count_per_frame <= 1))
            for start, end, value in array_to_iterable_runs(self.co_single_array):
                if end - start + 1 < 5 and value:
                    self.co_single_array[start:end+1] = False

        self._motion_sweep(start_idx, end_idx)

        if self.blob_array is not None and self.locker_stable_swap:
            self._process_stable_swap_candidates(self.locker_stable_swap)

        if self.skip_contrast:
            if self.blob_array is not None and self.locker_stable_swap:
                self._audit_stable_swap_candidates(self.locker_stable_swap)
            elif self.ambiguous_frames:
                swap_orders = self._launch_dialog_swap(self.ambiguous_frames, start_idx, end_idx)
                self._execute_swap_orders(swap_orders, self.total_frames-1)
            return self.pred_data_array

        qc_mask = self._find_eligible_frames(start_idx, end_idx)
        dim_array = calculate_pose_dim(self.pred_data_array[qc_mask], self.anglemap)
        cutout_width = int(np.percentile(dim_array[..., 0], 90) * 1.15)
        cutout_height = int(np.percentile(dim_array[..., 1], 90) * 1.15)
        cutout_dim = max(cutout_width, cutout_height)

        if cutout_dim % 2:
            cutout_dim += 1
    
        angles = np.zeros((self.total_frames, self.dlc_data.instance_count))
        angles[qc_mask] = np.rad2deg(calculate_pose_array_rotations(self.pred_data_array[qc_mask], self.anglemap))

        ca = Cutout_Augments(
            centroids=self.centroids,
            cutout_dim=cutout_dim,
            angle_array=angles,
            grayscaling=False,
            )

        self._crop_rotate_and_export(ca)
        self._run_contrain_magic(start_idx, end_idx)
        return self.pred_data_array

    def _motion_sweep(self, start_idx, end_idx):
        if self.skip_sweep:
            return
        
        if self.id_lock and not self.forced_lock:
            try:
                lock_mask = (self.inst_count_per_frame == 1) & (self.blob_array[:,0] <= 1)
                lock_frame_indices = np.where(lock_mask)[0][:10]
            except:
                pass
            else:
                num_0 = np.sum(~np.isnan(self.centroids[lock_frame_indices, 0, 0]))
                self.locked_idx = 0 if num_0 >= 5 else 1

        pbar = tqdm(total=end_idx - start_idx, desc=f"Motion Sweep In Progress", leave=True, ncols=200)        
        try:
            for f in range(start_idx, end_idx):
                logger.debug(f"--------- frame {f} ---------")
                pbar.update(1)
                self._correct_frame_with_hungarian(f)
        finally:
            pbar.close()

    def _find_eligible_frames(
            self,
            start_idx: int,
            end_idx: int,
            inst_dist_threshold: float = 0.5,
            size_threshold: Tuple[float, float] = (0.5, 2.5),
            bp_threshold: int = 6,
            twist_angle_threshold: float = 90.0,
            ) -> np.ndarray:

        instance_count = get_instance_count_per_frame(self.pred_data_array)
        two_inst_mask = instance_count == 2
        
        if not np.any(two_inst_mask):
            raise ValueError("No frame with two insts, cannot perform contrastive learning.")

        two_inst_array = self.pred_data_array[two_inst_mask]
        centroids_two_inst = self.centroids[two_inst_mask]
        dists = np.linalg.norm(centroids_two_inst[:, 1, :] - centroids_two_inst[:, 0, :], axis=-1)
        dist_mask = dists > inst_dist_threshold * self.mice_length
        strict_dist_mask = dists > 0.3 * self.mice_length

        angle_mask = ~np.any(outlier_rotation(two_inst_array, self.anglemap, twist_angle_threshold), axis=-1)
        size_mask = ~np.any(outlier_size(two_inst_array, *size_threshold), axis=-1)
        bp_mask = ~np.any(outlier_bodypart(two_inst_array, bp_threshold), axis=-1)

        frame_valid_mask = np.zeros(self.total_frames, dtype=bool)
        frame_valid_mask[two_inst_mask] = dist_mask & angle_mask & size_mask & bp_mask

        segmentation_mask = two_inst_mask.copy()
        if self.ambiguous_frames:
            segmentation_mask[self.ambiguous_frames] = False # Break segment at ambiguous frames when there are ambiguous frames
            segmentation_mask[two_inst_mask] = strict_dist_mask # And frames that are dangerously close
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
    
    def _run_contrain_magic(self, start_idx, end_idx):

        from utils.torch import Crop_Dataset, Embedding_Visualizer, Contrastive_Trainer, Cutout_Dataloader

        dataloader = Cutout_Dataloader(folder_path=self.temp_dir, seg_list=self.seg_list)

        trainer = Contrastive_Trainer()

        pretrained = False
        if self.emp.pretrained_model_path and os.path.exists(self.emp.pretrained_model_path):
            logger.info(f"[TF] Loading pretrained model: {self.emp.pretrained_model_path}")
            trainer.load_checkpoint(self.emp.pretrained_model_path, strict=False)
            pretrained = True

        train_seg_indices = dataloader.select_training_segments(ratio=1.0)

        logger.info(f"[TF] Loading {len(train_seg_indices)} training segments...")
        train_datasets = dataloader.load_all_segments_for_training(train_seg_indices)

        if self.emp.epochs > 0:
            trainer.train(datasets=train_datasets, emp=self.emp, pretrained=pretrained)

        if self.emp.save_model:
            model_path = os.path.join(self.extractor.get_video_dir(), f"{self.extractor.get_video_name(no_ext=True)}_contrastive_trained.pth")
            trainer.save_checkpoint(model_path)
            logger.info(f"[TF] Model saved to {model_path}")

        logger.info(f"[TF] Running inference on ALL segments")
        segment_embeddings = []
        segment_frame_indices = []
        segment_motion_ids = []

        for seg_idx in tqdm(range(len(dataloader.segment_index)), desc="Inferring Segments", ncols=200):
            images, frames, _ = dataloader.load_segment(seg_idx)

            crops = [images[i, mouse_idx] for i in range(len(frames)) for mouse_idx in range(2)]
            mids_flat = [mid for _ in frames for mid in [0, 1]]
            frames_flat = [f for f in frames for _ in range(2)]
            
            ds = Crop_Dataset(crops, mids_flat, frames_flat)
            
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
        agreement_pix, stable_swap_candidates_raw, diagnosis_timeline = vis.plot_agreement_timeline()

        stable_swap_candidates = [f for f in stable_swap_candidates_raw if self.id_lock and not self.co_single_array[f]]
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
        
        if not self.avtomat:
            pix_dialog = Dual_Pixmap_Dialog(agreement_pix, tsne_pix, max_width=1080, parent=self.main)
            pix_dialog.show()

        logger.info("[TF] Visualization complete.")
        
        if self.avtomat:
            self._process_stable_swap_candidates(stable_swap_candidates)
        else:
            self._audit_stable_swap_candidates(stable_swap_candidates)

        self.centroids = calculate_anatomical_centers(self.pred_data_array, self.anglemap)

    def _process_stable_swap_candidates(self, stable_swap_candidates):
        stable_spans = indices_to_spans(stable_swap_candidates)

        for start, end in stable_spans:
            amb_in_range = [f for f in self.ambiguous_frames if f >= start and f <= end]

            if amb_in_range and self.skip_contrast:
                self.pred_data_array = swap_track(self.pred_data_array, 0, swap_range=list(range(amb_in_range[0], end+1)))

        if stable_spans:
            with open (os.path.join(self.temp_dir, "stable_spans.json"), "w") as file:
                json.dump(stable_spans, file, indent=4)

    def _audit_stable_swap_candidates(self, stable_swap_candidates):
        stable_spans = indices_to_spans(stable_swap_candidates)

        for start, end in stable_spans:
            start_idx = max(0, start - 10)
            fin_idx = min(end + 11, self.total_frames-1)
            amb_in_range = [f for f in self.ambiguous_frames if f >= start and f <= end]

            swap_orders = self._launch_dialog_swap(amb_in_range, start_idx, fin_idx)

            if not self._execute_swap_orders(swap_orders, fin_idx):
                return

    def _correct_frame_with_hungarian(self, frame_idx: int):
        valid_inst = get_instances_on_current_frame(self.pred_data_array, frame_idx)
        n_obs = len(valid_inst)
        if n_obs == 0:
            logger.debug("[TF] No observed pose on the current frame, skipping...")
            return
        
        curr_pose = self.centroids[frame_idx]

        pos_age = frame_idx - self.last_update
        self.last_known_pos[pos_age > self.MAX_POS_AGE_BEFORE_INVALIDATING] = np.nan

        ref_pos = self.last_known_pos
        valid_ref = []
        for i in range(2):
            if ~np.isnan(ref_pos[i, 0]):
                valid_ref.append(i)
        n_ref = len(valid_ref)
        if n_ref == 0:
            self.last_known_pos[valid_inst] = self.centroids[frame_idx, valid_inst]
            self.last_update[valid_inst] = frame_idx
            logger.debug("[TF] No valid reference for the current frame, skipping...")
            return

        logger.debug("[TF] Pose:"
            f" Inst 0: ({curr_pose[0, 0]:.1f}, {curr_pose[0, 1]:.1f}), Inst 1: ({curr_pose[1, 0]:.1f}, {curr_pose[1, 1]:.1f})")
        logger.debug("[TF] Refs:"
            f" Inst 0: ({ref_pos[0, 0]:.1f}, {ref_pos[0, 1]:.1f}), Inst 1: ({ref_pos[1, 0]:.1f}, {ref_pos[1, 1]:.1f})")
        
        max_dist_cost = self.MAX_DIST_THRESHOLD * self.mice_length
        if n_obs == 1:
            inst_idx = valid_inst[0]
            if n_ref == 1:
                ref_idx = valid_ref[0]
                if self.id_lock and self.co_single_array[frame_idx]:
                    if inst_idx != self.locked_idx:
                        logger.debug(f"[TF] ID Lock: {self.id_lock}, inst_idx {inst_idx} != {self.locked_idx}.")
                        self._swap_ids_in_frame(frame_idx)
                    self.last_id_locker = frame_idx
                else:
                    dist = np.linalg.norm(ref_pos[ref_idx]-curr_pose[inst_idx])
                    if dist > max_dist_cost:
                        logger.debug(f"[TF] Assgin cost too high ({dist:.2f} > {max_dist_cost:.2f}), add to ambiguous frame and skip ref update.")
                        self.ambiguous_frames.append(frame_idx)
                        return
                    elif inst_idx != ref_idx:
                        logger.debug(f"[TF] inst_idx {inst_idx} != {ref_idx}, swapping.")
                        self._swap_ids_in_frame(frame_idx)
        
            elif self.id_lock and self.co_single_array[frame_idx] and self.co_single_array[frame_idx-1]:
                if inst_idx != self.locked_idx:
                    logger.debug(f"[TF] ID Lock: {self.id_lock}, inst_idx {inst_idx} != {self.locked_idx}.")
                    self._swap_ids_in_frame(frame_idx)
                self.last_id_locker = frame_idx
            else:
                c0 = np.linalg.norm(ref_pos[inst_idx]-curr_pose[inst_idx])
                c1 = np.linalg.norm(ref_pos[1-inst_idx]-curr_pose[inst_idx])
                if min(c0, c1) > max_dist_cost:
                    logger.debug(f"[TF] Assgin cost too high ({min(c0, c1):.2f} > {max_dist_cost:.2f}), add to ambiguous frame and skip ref update.")
                    self.ambiguous_frames.append(frame_idx)
                    return
                if self.blob_array is not None and self.blob_array[frame_idx, 0] > 1 and c0+c1 < self.mice_length:
                    logger.debug("[TF] Potential instance overlapping/missing, add to ambiguous frame and skip ref update.")
                    self.ambiguous_frames.append(frame_idx)
                    return
                if abs(c0-c1) < self.AMBIGUITY_THRESHOLD * (c0+c1):
                    logger.debug(f"[TF] Distance ambiguous ({abs(c0-c1):.2f} < {self.AMBIGUITY_THRESHOLD * (c0+c1):.2f}), add to ambiguous frame.")
                    self.ambiguous_frames.append(frame_idx)
                if c0 > c1:
                    logger.debug(f"[TF] No swap cost ({c0}) larger than swap cost ({c1}) swapping.")
                    self._swap_ids_in_frame(frame_idx)
    
                if self.id_lock and self.co_single_array[frame_idx]:
                    new_inst_idx = get_instances_on_current_frame(self.pred_data_array, frame_idx)[0]
                    if new_inst_idx != self.locked_idx and frame_idx > self.last_id_locker+1:
                        logger.debug(f"[TF] Corrected inst {new_inst_idx} != locked inst {self.locked_idx} at checkpoint, logging swap segment.")
                        self._swap_ids_in_frame(frame_idx)
                        self.ambiguous_frames.append(frame_idx)
                        self.locker_stable_swap.extend(range(self.last_id_locker+1, frame_idx))
                    self.last_id_locker = frame_idx
        else:
            if n_ref == 1:
                ref_idx = valid_ref[0]
                c0 = np.linalg.norm(ref_pos[ref_idx]-curr_pose[ref_idx])
                c1 = np.linalg.norm(ref_pos[ref_idx]-curr_pose[1-ref_idx])
                if min(c0, c1) > max_dist_cost:
                    logger.debug(f"[TF] Assgin cost too high ({min(c0, c1):.2f} > {max_dist_cost:.2f}), add to ambiguous frame and skip ref update.")
                    self.ambiguous_frames.append(frame_idx)
                    return
                if abs(c0-c1) < self.AMBIGUITY_THRESHOLD * (c0+c1):
                    logger.debug(f"[TF] Distance ambiguous ({abs(c0-c1):.2f} < {self.AMBIGUITY_THRESHOLD * (c0+c1):.2f}), add to ambiguous frame.")
                    self.ambiguous_frames.append(frame_idx)
                if c0 > c1:
                    logger.debug(f"[TF] No swap cost ({c0}) larger than swap cost ({c1}) swapping.")
                    self._swap_ids_in_frame(frame_idx)
            else:
                cost_matrix = np.zeros((2, 2))
                for i in range(2):
                    for j in range(2):
                        cost_matrix[i, j] = np.linalg.norm(ref_pos[i]-curr_pose[j])
                c0 = cost_matrix[0, 0] + cost_matrix[1, 1]
                c1 = cost_matrix[1, 0] + cost_matrix[0, 1]
                if min(c0, c1) > max_dist_cost * 1.5:
                    logger.debug(f"[TF] Assgin cost too high ({min(c0, c1):.2f} > {max_dist_cost * 1.5:.2f}), add to ambiguous frame and skip ref update.")
                    self.ambiguous_frames.append(frame_idx)
                    return
                if abs(c0-c1) < self.AMBIGUITY_THRESHOLD * (c0+c1):
                    logger.debug(f"[TF] Distance ambiguous ({abs(c0-c1):.2f} < {self.AMBIGUITY_THRESHOLD * (c0+c1):.2f}), add to ambiguous frame.")
                    self.ambiguous_frames.append(frame_idx)
                if c0 > c1:
                    logger.debug(f"[TF] No swap cost ({c0}) larger than swap cost ({c1}) swapping.")
                    self._swap_ids_in_frame(frame_idx)

        frame_inst_after_update = get_instances_on_current_frame(self.pred_data_array, frame_idx)
        logger.debug(f"[TF] Frame instances after fixing: {frame_inst_after_update}, updating refs.")
        self.last_known_pos[frame_inst_after_update] = self.centroids[frame_idx, frame_inst_after_update]
        self.last_update[frame_inst_after_update] = frame_idx

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