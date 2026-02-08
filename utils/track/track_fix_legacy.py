
# Currently unused, will probably attempt a Transformer based implementation in unforseeable future


# import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.optimize import linear_sum_assignment
# from typing import Optional, Tuple, List
# from utils.dataclass import Track_Properties
# from utils.logger import logger


# HUNGARIAN_SUCCESS = "success"
# HUNGARIAN_LOW_SIM = "low_similarity"
# HUNGARIAN_LOW_GAP = "low_confidence"
# HUNGARIAN_FAILED = "mission_failed"

# class Hungarian:
#     def __init__(self,
#                 pred:Track_Properties,
#                 ref:Track_Properties,
#                 oks_kappa:np.ndarray,
#                 sigma:Tuple[float, float],
#                 weight:Tuple[float, float, float],
#                 min_sim:float = 0.15,
#                 gap_threshold = 0.05
#                 ):
#         self.min_sim = min_sim
#         self.gap_threshold = gap_threshold

#         self.pred = pred
#         self.ref = ref
#         self.kappa = oks_kappa
#         self.sigma = sigma
#         self.weight = weight

#         self.ref_exist = np.any(~np.isnan(ref.centroids), axis=1)

#         self.instance_count = pred.validity.shape[0]
#         self.inst_list = list(range(self.instance_count))

#         self.pred_indices = np.where(self.pred.validity)[0]
#         self.ref_indices  = np.where(self.ref.validity)[0]

#         self.new_order = None

#         assert len(self.weight) == 3, f"weight must be length 3, got {len(self.weight)}"
#         assert len(self.sigma) == 2, f"sigma must be length 2, got {len(self.sigma)}"
#         assert all(w >= 0 for w in self.weight), "weights must be non-negative"
#         assert all(s > 0 for s in self.sigma), "sigma values must be positive"
#         assert np.isclose(sum(self.weight), 1.0), f"weights sum to {sum(self.weight):.6f}, not 1"

#     def hungarian_matching(self) -> Optional[List[int]]:
#         """
#         Perform identity correction using Hungarian algorithm.

#         Returns:
#             new_order: List[int] of length N, where:
#                 new_order[target_identity] = source_instance_index_in_current_frame
#                 i.e., "Identity j comes from current instance new_order[j]"
#         """
#         ref_centroids, ref_rotations, ref_poses = self.mask_prop(prop=self.ref, mask=self.ref.validity)
#         pred_centroids, pred_rotations, pred_poses = self.mask_prop(prop=self.pred, mask=self.pred.validity)

#         sim_matrix = self._compute_similarity_matrix(pred_centroids, pred_rotations, pred_poses, ref_centroids, ref_rotations, ref_poses)
#         cost_matrix = 1 - sim_matrix

#         if sim_matrix.size == 0 or np.any(np.isnan(sim_matrix)):
#             return HUNGARIAN_FAILED
    
#         if np.max(sim_matrix) < self.min_sim:
#             logger.debug("")  #<- TBA
#             return HUNGARIAN_LOW_SIM

#         if sim_matrix.size == 1:
#             row_ind_conf, col_ind_conf = [0], [0]
#             logger.debug(f"[HUN] Accepted 1v1 valid match (sim={np.max(sim_matrix):.4f} ≥ {self.min_sim})")
#         else:
#             try:
#                 row_ind, col_ind = linear_sum_assignment(cost_matrix)
#                 gaps = self._compute_assignment_gaps(cost_matrix, row_ind, col_ind)
#                 confident_mask = gaps >= self.gap_threshold
#                 row_ind_conf = row_ind[confident_mask]
#                 col_ind_conf = col_ind[confident_mask]
#                 logger.debug(f"[HUN] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}")
#             except Exception as e:
#                 logger.debug(f"[HUN] Hungarian failed: {e}.")
#                 return HUNGARIAN_FAILED

#         if len(row_ind_conf) == 0:
#             logger.debug("[HUN] No confident matches.")
#             return HUNGARIAN_LOW_GAP

#         self._build_new_order(row_ind_conf, col_ind_conf)
#         logger.debug(f"[HUN] Final new_order: {self.new_order}")
#         return HUNGARIAN_SUCCESS

#     def get_new_order(self) -> List[int]:
#         return self.new_order

#     def _compute_similarity_matrix(self, pred_centroids, pred_rotations, pred_poses, ref_centroids, ref_rotations, ref_poses):
#         cent_matrix = cdist(pred_centroids, ref_centroids, metric='euclidean')
#         rota_matrix = self.angular_distance(pred_rotations[:, np.newaxis], ref_rotations[np.newaxis, :])

#         w1, w2, w3 = self.weight
#         s1, s2 = self.sigma
        
#         sim_c = np.exp(-cent_matrix**2 / s1**2)
#         sim_r = np.exp(-rota_matrix**2 / s2**2)
#         sim_p = self._object_keypoint_similarity(pred_poses, ref_poses)

#         sim_matrix = w1 * sim_c + w2 * sim_r + w3 * sim_p

#         if sim_matrix.size == 1:
#             sc = sim_c.item()
#             sr = sim_r.item()
#             sp = sim_p.item()
#             total = sim_matrix.item()
#             d = cent_matrix.item()
#             dtheta = rota_matrix.item()
#             logger.debug(
#                 "[SIM] Single match:"
#                 f"d={d:.1f}px (σ₁={s1:.1f}) → sim_c={sc:.3f}, Δθ={dtheta:.2f}rad (σ₂={s2:.2f}) → sim_r={sr:.3f}, "
#                 f"sim_p={sp:.3f} → total={total:.4f} (weights=({w1},{w2},{w3}))"
#             )
#         else:
#             logger.debug(f"[SIM] Components: | sim_c ={sim_c} | sim_r ={sim_r} | sim_p ={sim_p}")

#         assert np.all(np.isfinite(sim_matrix)), "pose_matrix contains NaN/inf — check pose validity masking"

#         return sim_matrix

#     def _object_keypoint_similarity(
#         self,
#         pred_poses:np.ndarray,
#         ref_poses:np.ndarray,
#         visibility_thresh:float=0.3,
#         min_shared_perc = 0.5,
#     ) -> np.ndarray:
#         x1, y1, p1 = pred_poses[:, 0::3], pred_poses[:, 1::3], pred_poses[:, 2::3]
#         x2, y2, p2 = ref_poses[:, 0::3], ref_poses[:, 1::3], ref_poses[:, 2::3]

#         x1_b = x1[:, None, :]
#         y1_b = y1[:, None, :]
#         p1_b = p1[:, None, :]
#         x2_b = x2[None, :, :]
#         y2_b = y2[None, :, :]
#         p2_b = p2[None, :, :]

#         x1_b = np.where(np.isfinite(x1_b), x1_b, 0.0)
#         y1_b = np.where(np.isfinite(y1_b), y1_b, 0.0)
#         p1_b = np.where(np.isfinite(p1_b), p1_b, 0.0)
#         x2_b = np.where(np.isfinite(x2_b), x2_b, 0.0)
#         y2_b = np.where(np.isfinite(y2_b), y2_b, 0.0)
#         p2_b = np.where(np.isfinite(p2_b), p2_b, 0.0)

#         d2 = (x1_b - x2_b)**2 + (y1_b - y2_b)**2

#         visible = (p1_b > visibility_thresh) & (p2_b > visibility_thresh)

#         all_kp = x1.shape[1]
#         min_shared = int(all_kp * min_shared_perc)

#         shared_count = np.sum(visible, axis=2)
#         logger.debug(f"[OKS] shared kps per pair: {shared_count.flatten()}")

#         exp_term = np.exp(-d2 / (2 * (self.kappa)**2 + 1e-9))
#         conf_weight = np.minimum(p1_b, p2_b)
        
#         weighted = exp_term * conf_weight * visible
#         numerator = np.sum(weighted, axis=2)
#         denominator = np.sum(conf_weight * visible, axis=2) + 1e-9
#         oks = numerator / denominator
#         oks = np.where(shared_count >= min_shared, oks, 0.0)
#         clamped = np.sum(shared_count < min_shared)
#         if clamped:
#             logger.debug(f"[OKS] clamped {clamped} pairs with <{min_shared} shared kps")

#         return oks

#     def _compute_assignment_gaps(self, cost_matrix: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> np.ndarray:
#         gaps = []
#         for r, c in zip(row_ind, col_ind):
#             best = cost_matrix[r, c]

#             row_costs = cost_matrix[r].copy()
#             row_costs[c] = np.inf  
#             second_best_row = np.min(row_costs)
#             col_costs = cost_matrix[:, c].copy()
#             col_costs[r] = np.inf  
#             second_best_col = np.min(col_costs)

#             gap = min(second_best_row - best, second_best_col - best)
#             gaps.append(gap)
        
#         return np.array(gaps)

#     def _build_new_order(self, row_ind:np.ndarray, col_ind:np.ndarray) -> List[int]:
#         all_inst = range(self.instance_count)

#         processed = {}
#         for r, c in zip(row_ind, col_ind):
#             target_identity = self.ref_indices[c]
#             source_instance = self.pred_indices[r]
#             processed[target_identity] = source_instance

#         unprocessed = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
#         unassigned = [inst_idx for inst_idx in all_inst if inst_idx not in processed.values()]

#         for target_identity in unprocessed:
#             if target_identity in unassigned:
#                 source_instance = target_identity
#                 processed[target_identity] = source_instance
#                 unassigned.remove(source_instance)
        
#         unprocessed[:] = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]

#         for target_identity in unprocessed:
#             source_instance = unassigned[-1]
#             processed[target_identity] = source_instance
#             unassigned.remove(source_instance)
            
#         sorted_processed = {k: processed[k] for k in sorted(processed)}
#         self.new_order = list(sorted_processed.values())

#     @staticmethod
#     def angular_distance(a, b):
#         d = np.abs(a - b)
#         return np.minimum(d, 2*np.pi - d)

#     @staticmethod
#     def mask_prop(prop:Track_Properties, mask:np.ndarray):
#         return prop.centroids[mask], prop.rotations[mask], prop.poses[mask]



# import numpy as np
# import random
# from itertools import combinations

# from scipy.optimize import differential_evolution

# from PySide6.QtWidgets import QProgressDialog
# from typing import List, Dict, Tuple, Optional

# from .hungarian import Hungarian
# from .sigma_kappa import sigma_estimation, kappa_estimation
# from utils.pose import calculate_pose_centroids, calculate_aligned_local, calculate_pose_rotations, calculate_anatomical_centers
# from utils.helper import get_instance_count_per_frame
# from utils.dataclass import Track_Properties
# from utils.logger import logger


# class Track_Fixer:
#     def __init__(
#             self,
#             pred_data_array:np.ndarray,
#             angle_map:Dict[str, int],
#             progress:QProgressDialog,
#             crp_weight:Tuple[float, float, float],
#             cr_sigma:Optional[Tuple[float, float]] = None,
#             kappa:Optional[np.ndarray] = None,
#             prefit_window_size:int = 1000,
#             learning_rate:float = 0.5,
#             lookback_window:int = 5,
#             minimum_similarity: float = 0.15,
#             gap_threshold: float = 0.05,
#             used_starts: List[int]=[]
#             ):
#         self.pred_data_array = pred_data_array.copy()
#         self.angle_map = angle_map
    
#         self.crp_weight = crp_weight
        
#         self.lr = learning_rate
#         self.lookback_window = lookback_window

#         self.pred_data_array = pred_data_array.copy()

#         self.total_frames, self.instance_count, xyconf = pred_data_array.shape
#         self.num_keypoint = xyconf // 3

#         self.video_dim = (np.nanmax(pred_data_array[..., 0::3]), np.nanmax(pred_data_array[..., 1::3]))
#         self.inst_count_per_frame = get_instance_count_per_frame(pred_data_array)
#         self.centroids, self.rotations, self.poses = self._pose_array_to_crp(pred_data_array)
#         self.canon_pose = np.nanmean(self.poses.reshape(self.total_frames, self.instance_count, self.num_keypoint, 3)[..., :2], axis=(0, 1))
#         self.mice_length = np.linalg.norm(self.canon_pose[self.angle_map["head_idx"]]-self.canon_pose[self.angle_map["tail_idx"]])

#         self.cr_sigma = cr_sigma if cr_sigma is not None else sigma_estimation((self.centroids, self.rotations, self.poses), max_disp_px=self.mice_length)
#         self.kappa = kappa if kappa is not None else kappa_estimation(self.poses)
#         self.min_sim = minimum_similarity
#         self.gap_threshold = gap_threshold

#         self.progress = progress
#         self.progress.setRange(0, self.total_frames)

#         self.inst_list = list(range(self.instance_count))
#         self.ambiguous_frames = []
#         self.used_starts = used_starts
        
#         self._precompute_valid_windows(window_step=prefit_window_size)

#     def fit_full_video(self):
#         return self._weight_repeater(range(self.total_frames))

#     def iter_orchestrator(self):
#         current_try = 0
#         max_try = 5
#         blast, ambi = [], []
#         pred_data_array = self.pred_data_array.copy()

#         while current_try < max_try:
#             if not self.window_list:
#                 self.window_list = self.window_list_org.copy()
#                 self.used_starts.clear()
            
#             window_idx = self.window_list.pop(0)
#             if self.valid_starts[window_idx] in self.used_starts:
#                 continue

#             logger.debug(f"[TMOD] Trying window {window_idx} with score {self.window_scores[window_idx]:.3f}")
#             slice_window = range(self.valid_starts[window_idx], self.valid_ends[window_idx])
            
#             pred_data_array, blast, ambi = self._weight_repeater(slice_window)

#             if ambi:
#                 logger.info(f"[TMOD] Window {window_idx} produced {len(blast)} frames with implausible assignment, {len(ambi)} frames with ambiguious assignment"
#                              " → selected for training")
#                 self.used_starts.append(self.valid_starts[window_idx])
#                 break
            
#             logger.debug(f"[TMOD] Window {window_idx} produced no issues → skipping to next window")

#             current_try += 1
        
#         if current_try == max_try:
#             logger.warning("Failed to find valid window after reaching max attempts, returning the last 'clean' window.")

#         return pred_data_array, blast, ambi, list(slice_window), self.used_starts

#     def get_params(self):
#         return self.crp_weight, self.cr_sigma, self.min_sim, self.gap_threshold, self.kappa

#     def _weight_repeater(self, slice_window:range) -> Tuple[np.ndarray, np.ndarray, List[int]]:
#         ref = Track_Properties(
#             centroids = np.full((self.instance_count, 2), np.nan),
#             rotations = np.full((self.instance_count,), np.nan),
#             poses = np.full((self.instance_count, self.num_keypoint*3), np.nan),
#             last_updated = np.full((self.instance_count,), -2*self.lookback_window),
#             validity = np.zeros((self.instance_count,), dtype=bool)
#         )
#         wr_pred_data = self.pred_data_array.copy()
#         centroids, rotations, poses = self.centroids.copy(), self.rotations.copy(), self.poses.copy()
#         array_to_sync = [wr_pred_data, centroids, rotations, poses]

#         ambiguous_frames = []

#         start, end = slice_window[0], slice_window[-1]

#         if abs((end-start) - self.total_frames) < 100:
#             self.progress.setLabelText("Fitting the entire video.")
#         else:
#             self.progress.setLabelText(f"Testing the weight with frames between {start} and {end}.")

#         self.progress.setRange(start, end)

#         for frame_idx in slice_window:
#             self.progress.setValue(frame_idx)
#             if self.progress.wasCanceled():
#                 raise RuntimeError("User cancelled the operation.")
            
#             logger.debug(f"---------- frame: {frame_idx} ---------- ")

#             pred = Track_Properties(
#                 centroids = centroids[frame_idx],
#                 rotations = rotations[frame_idx],
#                 poses = poses[frame_idx],
#                 last_updated = np.full((self.instance_count,), frame_idx),
#                 validity = np.any(~np.isnan(centroids[frame_idx]), axis=1)
#             )

#             if not np.any(pred.validity):
#                 logger.debug("[TMOD] Skipping due to no prediction on current frame.")
#                 continue

#             ref.validity = (ref.last_updated > frame_idx - self.lookback_window)

#             if not np.any(ref.validity):
#                 if not np.all(np.isnan(ref.centroids)):
#                     ambiguous_frames.append(frame_idx)
#                 logger.debug("[TMOD] No valid ref frame found. Rebasing the reference on current frame.")
#                 ref = self._rebase_ref(ref, array_to_sync[1:], frame_idx, pred.validity)
#                 continue

#             for i in range(self.instance_count):
#                 logger.debug(f"x,y in pred: inst {i}: ({pred.centroids[i,0]:.1f}, {pred.centroids[i,1]:.1f}) | angle: {pred.rotations[i]:.2f} ")
#             for i in range(self.instance_count):
#                 logger.debug(f"x,y in ref: inst {i}: ({ref.centroids[i,0]:.1f}, {ref.centroids[i,1]:.1f}) | angle: {ref.rotations[i]:.2f} | "
#                             f"last updated: {ref.last_updated[i]} | valid: {ref.validity[i]}")

#             hun = Hungarian(pred, ref, oks_kappa=self.kappa, sigma=self.cr_sigma, weight=self.crp_weight,
#                             min_sim=self.min_sim, gap_threshold=self.gap_threshold)

#             result = hun.hungarian_matching()

#             fixed_pred_mask = np.zeros((self.instance_count,), dtype=bool)
#             match result:
#                 case "mission_failed":
#                     ambiguous_frames.append(frame_idx)
#                 case "low_similarity":
#                     ambiguous_frames.append(frame_idx)
#                     fixed_pred_mask = pred.validity & (~ref.validity)
#                 case "low_confidence": 
#                     ambiguous_frames.append(frame_idx)
#                     fixed_pred_mask = pred.validity
#                 case "success":
#                     new_order = hun.get_new_order()
#                     match new_order:
#                         case self.inst_list: logger.debug("[TMOD] NO SWAP, already the best solution.")
#                         case _: logger.debug("[TMOD] SWAP, new_order: {new_order}.")
#                     self._sync_changes(array_to_sync, frame_idx, new_order)
#                     fixed_pred_mask = pred.validity[new_order]

#             if np.any(fixed_pred_mask):
#                 ref = self._rebase_ref(ref, array_to_sync[1:], frame_idx, fixed_pred_mask)

#         return array_to_sync[0], ambiguous_frames

#     def process_labels(self, pred_data_array, slice_window, status_array):
#         if np.all(status_array == 0):
#             return

#         slice_corrected = pred_data_array
#         self._train_swap(slice_corrected, slice_window, status_array)

#     def _train_swap(self, slice_corrected: np.ndarray, slice_window: range, status_array: np.ndarray):
#         valid_mask = np.all((status_array > 0) & (status_array < 3), axis=1)
#         if not np.any(valid_mask):
#             logger.debug("[S_TRAIN] Skipping training.")
#             return

#         def objective(x):
#             w1, w2, log_s1, log_s2, min_sim, gap_thresh = x
#             w3 = 1.0 - w1 - w2
#             if w1 < 0.6 or w2 < 0 or w3 < 0:
#                 return 1e6
            
#             weights = (float(w1), float(w2), float(w3))
#             sigma = (float(np.exp(log_s1)), float(np.exp(log_s2)))

#             try:
#                 error = self._swap_train_assessment(
#                     slice_corrected, slice_window, status_array,
#                     weights=weights,
#                     sigma=sigma,
#                     thresholds=(min_sim,gap_thresh)
#                 )
#                 return error
#             except Exception as e:
#                 logger.warning(f"[SASS] Evaluation failed: {e}")
#                 return 1e6

#         w1_curr, w2_curr, _ = self.crp_weight
#         s1, s2 = self.cr_sigma

#         bounds = [
#             (0.6, 1.0),           # w1
#             (0.0, 0.4),           # w2
#             (np.log(s1) - 1.0, np.log(s1) + 1.0),  # log(s1)
#             (np.log(s2) - 1.0, np.log(s2) + 1.0),  # log(s2)
#             (0.05, 0.30),         # min_sim ∈ [0.05, 0.30]
#             (0.01, 0.20),         # gap_threshold ∈ [0.01, 0.20]
#         ]

#         x0 = [
#             w1_curr,
#             w2_curr,
#             np.log(self.cr_sigma[0]),
#             np.log(self.cr_sigma[1]),
#             self.min_sim,
#             self.gap_threshold
#         ]

#         result = differential_evolution(
#             objective,
#             bounds,
#             x0 = x0,
#             maxiter=100,
#             popsize=7,
#             seed=42,
#             polish=False,
#             atol=1e-3,
#             updating='deferred',
#             workers=1
#         )

#         best_w1, best_w2, best_s1_log, best_s2_log, best_min_sim, best_gap_thresh = result.x
#         best_w3 = 1.0 - best_w1 - best_w2
#         best_weight = (float(best_w1), float(best_w2), float(best_w3))
#         best_s1, best_s2 =  float(np.exp(best_s1_log)), float(np.exp(best_s2_log))
#         best_sigma =  (best_s1, best_s2)
#         best_score = result.fun
#         logger.info(f"[S_TRAIN] Final params: weights={best_weight}, sigma={best_sigma},"
#                     f"min_sim={best_min_sim:.2f}, gap={best_gap_thresh:.2f}")

#         if best_score < 0.1:
#             self.crp_weight = tuple(
#                 (1 - self.lr) * old + self.lr * new 
#                 for old, new in zip(self.crp_weight, best_weight)
#             )
#             self.cr_sigma = (
#                 (1 - self.lr) * self.cr_sigma[0] + self.lr * best_s1,
#                 (1 - self.lr) * self.cr_sigma[1] + self.lr * best_s2
#             )
#             self.min_sim = (1 - self.lr) * self.min_sim + self.lr * best_min_sim
#             self.gap_threshold = (1 - self.lr) * self.gap_threshold + self.lr * best_gap_thresh

#             logger.info(f"[S_TRAIN] Updated CRP weights to {self.crp_weight}. Updated CR Sigma to {self.cr_sigma} "
#                         f"Updated threshold to {(self.min_sim, self.gap_threshold)}  (error: {best_score:.3f})")
#         else:
#             logger.warning(f"[S_TRAIN] DE failed to find good weights; keeping current. Best score: {best_score}")

#     def _swap_train_assessment(
#         self,
#         corrected_window: np.ndarray,
#         window_slice: range,
#         status_array: np.ndarray,
#         weights: Tuple[float, float, float],
#         sigma: Tuple[float, float],
#         thresholds: Tuple[float, float]
#     ) -> float:
#         F = len(window_slice)
#         logger.debug(f"[SASS] Window size: {F} frames, {self.instance_count} instances, "
#                      f"weights={weights}, sigma={sigma}, thresholds: {thresholds}")

#         raw_centroids = self.centroids[window_slice].copy()
#         raw_rotations = self.rotations[window_slice].copy()
#         raw_poses = self.poses[window_slice].copy()

#         raw_cent_corrected = raw_centroids.copy()

#         gt_centroids, gt_rotations, gt_poses = self._pose_array_to_crp(corrected_window)

#         ref = Track_Properties(
#             centroids = gt_centroids[0],
#             rotations = gt_rotations[0],
#             poses = gt_poses[0],
#             last_updated = np.full(self.instance_count, 0),
#             validity = np.any(~np.isnan(gt_centroids[0]), axis=1)
#         )

#         for f in range(1, F):
#             pred = Track_Properties(
#                 centroids=raw_centroids[f],
#                 rotations=raw_rotations[f],
#                 poses=raw_poses[f],
#                 last_updated=np.full(self.instance_count, f),
#                 validity = np.any(~np.isnan(raw_centroids[f]), axis=1)
#             )

#             gt_validity = np.any(~np.isnan(gt_centroids[f]), axis=1)
#             if not np.any(pred.validity) or not np.any(ref.validity):
#                 logger.debug(f"[SASS] Frame {f}: Skipping (no valid pred/ref)")
#                 ref = self._rebase_ref(ref, [gt_centroids, gt_rotations, gt_poses], f, gt_validity)
#                 continue

#             ref.validity = (ref.last_updated > f - self.lookback_window)

#             hun = Hungarian(pred, ref, oks_kappa=self.kappa,
#                             sigma=sigma, weight=weights, min_sim=thresholds[0], gap_threshold=thresholds[1])
#             result = hun.hungarian_matching()
    
#             fixed_pred_mask = np.zeros((self.instance_count,), dtype=bool)
#             match result:
#                 case "low_similarity":
#                     fixed_pred_mask = gt_validity & (~ref.validity)
#                 case "low_confidence": 
#                     fixed_pred_mask = gt_validity
#                 case "success":
#                     new_order = hun.get_new_order()
#                     match new_order:
#                         case self.inst_list: logger.debug("[TMOD] NO SWAP, already the best solution.")
#                         case _: logger.debug("[TMOD] SWAP, new_order: {new_order}.")
#                     raw_cent_corrected[f, :, :] = raw_centroids[f, new_order, :]
#                     fixed_pred_mask = gt_validity

#             if np.any(fixed_pred_mask):
#                 ref = self._rebase_ref(ref, [gt_centroids, gt_rotations, gt_poses], f, fixed_pred_mask)

#         raw_cent_corrected = np.nan_to_num(raw_cent_corrected, nan=-1.0)
#         gt_centroids = np.nan_to_num(gt_centroids, nan=-1.0)

#         total_error = np.sum(np.any(raw_cent_corrected != gt_centroids, axis=-1))
#         final_error = total_error / (F*self.instance_count)

#         logger.debug(f"[SASS] Final error: {final_error:.4f}.")
#         return final_error

#     def _precompute_valid_windows(self, window_step: int = 100):
#         cumsum = np.concatenate(([0], np.cumsum(self.inst_count_per_frame)))
#         starts = np.arange(0, self.total_frames, window_step)
#         ends = np.minimum(starts + window_step, self.total_frames)
#         lens = ends - starts

#         nonzero = lens > 0
#         starts, ends, lens = starts[nonzero], ends[nonzero], lens[nonzero]

#         sums = cumsum[ends] - cumsum[starts]
#         valid = sums >= lens

#         if not np.any(valid):
#             logger.warning("[TMOD] No windows satisfied medium condition (sum >= len); relaxing to 2*sum >= len")
#             valid = (2 * sums) >= lens

#         if not np.any(valid):
#             raise RuntimeError(f"No non-overlapping windows (step={window_step}) satisfy even minimal instance count fraction.")

#         starts = starts[valid]
#         ends = ends[valid]

#         proximity_threshold = 2 * self.mice_length
#         logger.debug(f"[TMOD] Canonical body length: {self.mice_length:.1f}px, proximity threshold: {proximity_threshold:.1f}px")

#         centroid_dist = []
#         for i, j in combinations(self.inst_list, 2):
#             centroid_dist.append(np.linalg.norm(self.centroids[:, i, :]- self.centroids[:, j, :], axis=-1))

#         centroid_dist_arr = np.array(centroid_dist)
#         min_pairwise_dist = np.nanmin(centroid_dist_arr, axis=0)
#         close_frame = min_pairwise_dist < proximity_threshold

#         speeds = np.linalg.norm(np.diff(self.centroids, axis=0), axis=-1)
#         speeds_full = np.zeros((self.total_frames, len(self.inst_list)))
#         speeds_full[1:, :] = speeds

#         moving_frame = np.any(speeds_full > 10, axis=1)
#         moving_frame = np.nan_to_num(moving_frame, nan=False).astype(bool)

#         close_and_moving = close_frame & moving_frame

#         scores = []
#         for s, e in zip(starts, ends):
#             close_count = np.sum(close_and_moving[s:e])
#             score = close_count / (e - s)
#             scores.append(score)
#             logger.debug(f"[TMOD] Window [{s}-{e}]: {close_count}/{e - s} relevant frames, score={score:.3f}")

#         sorted_indices = np.argsort(scores)[::-1]
#         self.valid_starts = starts[sorted_indices]
#         self.valid_ends = ends[sorted_indices]
#         self.window_scores = np.array(scores)[sorted_indices]

#         total_windows = len(self.valid_starts)
#         top_k = min(20, total_windows)

#         prioritized = list(range(top_k))
#         random.shuffle(prioritized)

#         if top_k < total_windows:
#             remaining = list(range(top_k, total_windows))
#             random.shuffle(remaining)
#             prioritized.extend(remaining)

#         self.window_list = prioritized
#         self.window_list_org = self.window_list.copy()

#         logger.info(f"[TMOD] Precomputed {len(self.valid_starts)} windows. Top ambiguity score: {self.window_scores[0]:.2f}")

#     def _pose_array_to_crp(self, pred_data_array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         _, local_coords = calculate_pose_centroids(pred_data_array)
#         local_x, local_y = local_coords[..., 0::2], local_coords[..., 1::2]

#         centroids = calculate_anatomical_centers(pred_data_array, self.angle_map)
#         rotations = calculate_pose_rotations(local_x, local_y, self.angle_map)
#         poses = calculate_aligned_local(pred_data_array, self.angle_map)

#         return centroids, rotations, poses

#     @staticmethod
#     def _sync_changes(array_list:List[np.ndarray], frame_idx:int, new_order:List[int]):
#         for array in array_list:
#             if array.ndim == 3:
#                 array[frame_idx, :, :] = array[frame_idx, new_order, :]
#             elif array.ndim == 2:
#                 array[frame_idx, :] = array[frame_idx, new_order]
#             else:
#                 raise RuntimeError(f"Incompatible array shape: {array.shape}")

#     @staticmethod
#     def _rebase_ref(ref:Track_Properties, crp_list:List[np.ndarray], frame_idx:int, mask:np.ndarray) -> Track_Properties:
#         centroids, rotations, poses = crp_list
#         ref.centroids[mask] = centroids[frame_idx][mask]
#         ref.rotations[mask] = rotations[frame_idx][mask]
#         ref.poses[mask] = poses[frame_idx][mask]
#         ref.last_updated[mask] = frame_idx
#         return ref


# import dataclass
# from numpy.typing import NDArray

# @dataclass
# class Track_Properties:
#     centroids: NDArray
#     rotations: NDArray
#     poses: NDArray
#     last_updated: NDArray
#     validity: NDArray