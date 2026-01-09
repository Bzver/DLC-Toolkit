import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple, List
from utils.dataclass import Track_Properties
from utils.logger import logger


class Hungarian:
    def __init__(self,
                pred:Track_Properties,
                ref:Track_Properties,
                valid_pred_mask:np.ndarray,
                valid_ref_mask:np.ndarray,
                ref_frame_gap:np.ndarray,
                oks_kappa:np.ndarray,
                sigma:Tuple[float, float],
                weight:Tuple[float, float, float] = (0.8, 0.1, 0.1)
                ):
        self.update_predictions(pred, ref, valid_pred_mask, valid_ref_mask, ref_frame_gap)
        self.update_ksw(oks_kappa, sigma, weight)

    def hungarian_matching(self, gap_threshold = 0.05) -> Optional[List[int]]:
        """
        Perform identity correction using Hungarian algorithm.

        Returns:
            new_order: List[int] of length N, where:
                new_order[target_identity] = source_instance_index_in_current_frame
                i.e., "Identity j comes from current instance new_order[j]"
        """
        ref_centroids, ref_rotations, ref_poses= self.mask_prop(prop=self.ref, mask=self.ref_mask)
        pred_centroids, pred_rotations, pred_poses = self.mask_prop(prop=self.pred, mask=self.pred_mask)

        cent_matrix = cdist(pred_centroids, ref_centroids, metric='euclidean')
        rota_matrix = self.angular_distance(pred_rotations[:, np.newaxis], ref_rotations[np.newaxis, :])

        w1, w2, w3 = self.weight
        s1, s2 = self.sigma
        
        sim_c = np.exp(-cent_matrix**2 / s1**2)
        sim_r = np.exp(-rota_matrix**2 / s2**2)
        sim_p = self._object_keypoint_similarity(pred_poses, ref_poses)

        sim_matrix = w1 * sim_c + w2 * sim_r + w3 * sim_p
        cost_matrix = 1 - sim_matrix
        assert np.all(np.isfinite(cost_matrix)), "pose_matrix contains NaN/inf — check pose validity masking"

        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            logger.debug(f"[HUN] Hungarian assignment: row_ind={row_ind}, col_ind={col_ind}")
        except Exception as e:
            logger.debug(f"[HUN] Hungarian failed: {e}. Returning None.")
            return None

        gaps = self._compute_assignment_gaps(cost_matrix, row_ind, col_ind)
        confident_mask = gaps >= gap_threshold
        row_ind_conf = row_ind[confident_mask]
        col_ind_conf = col_ind[confident_mask]

        if len(row_ind_conf) == 0:
            logger.debug("[HUN] No confident matches.")
            return None

        logger.debug(f"[HUN] Kept {len(row_ind_conf)}/{len(row_ind)} matches with gap ≥ {gap_threshold}")

        new_order = self._build_new_order(row_ind_conf, col_ind_conf)
        logger.debug(f"[HUN] Final new_order: {new_order}")
        return new_order

    def update_predictions(
        self,
        pred:Track_Properties,
        ref:Track_Properties,
        valid_pred_mask:np.ndarray,
        valid_ref_mask:np.ndarray,
        ref_frame_gap:np.ndarray
        ):
        self.pred = pred
        self.ref = ref

        self.pred_mask = valid_pred_mask
        self.ref_mask = valid_ref_mask
        self.frame_gap = ref_frame_gap[valid_ref_mask]

        self.ref_centroids_full = ref.centroids
        self.frame_gap_full = ref_frame_gap

        self.instance_count = valid_pred_mask.shape[0]
        self.inst_list = list(range(self.instance_count))
        self.full_set = np.all(self.pred_mask) and np.all(self.ref_mask)

        self.pred_indices = np.where(self.pred_mask)[0]
        self.ref_indices  = np.where(self.ref_mask)[0]

    def update_ksw(
            self,
            kappa:np.ndarray,
            sigma:Tuple[float, float],
            weight:Tuple[float, float, float] = (0.8, 0.1, 0.1),
            ):
        self.kappa = kappa
        self.sigma = sigma
        self.weight = weight

        assert len(self.weight) == 3, f"weight must be length 3, got {len(self.weight)}"
        assert len(self.sigma) == 2, f"sigma must be length 2, got {len(self.sigma)}"
        assert all(w >= 0 for w in self.weight), "weights must be non-negative"
        assert all(s > 0 for s in self.sigma), "sigma values must be positive"
        assert np.isclose(sum(self.weight), 1.0), f"weights sum to {sum(self.weight):.6f}, not 1"

    def _object_keypoint_similarity(
        self,
        pred_poses:np.ndarray,
        ref_poses:np.ndarray,
        visibility_thresh:float=0.3,
        min_shared = 3,
    ) -> np.ndarray:
        x1, y1, p1 = pred_poses[:, 0::3], pred_poses[:, 1::3], pred_poses[:, 2::3]
        x2, y2, p2 = ref_poses[:, 0::3], ref_poses[:, 1::3], ref_poses[:, 2::3]

        x1_b = x1[:, None, :]
        y1_b = y1[:, None, :]
        p1_b = p1[:, None, :]
        x2_b = x2[None, :, :]
        y2_b = y2[None, :, :]
        p2_b = p2[None, :, :]

        x1_b = np.where(np.isfinite(x1_b), x1_b, 0.0)
        y1_b = np.where(np.isfinite(y1_b), y1_b, 0.0)
        p1_b = np.where(np.isfinite(p1_b), p1_b, 0.0)
        x2_b = np.where(np.isfinite(x2_b), x2_b, 0.0)
        y2_b = np.where(np.isfinite(y2_b), y2_b, 0.0)
        p2_b = np.where(np.isfinite(p2_b), p2_b, 0.0)

        d2 = (x1_b - x2_b)**2 + (y1_b - y2_b)**2

        visible = (p1_b > visibility_thresh) & (p2_b > visibility_thresh)
        shared_count = np.sum(visible, axis=2)
        logger.debug(f"[OKS] shared kps per pair: {shared_count.flatten()}")

        exp_term = np.exp(-d2 / (2 * (self.kappa)**2 + 1e-9))
        conf_weight = np.minimum(p1_b, p2_b)
        
        weighted = exp_term * conf_weight * visible
        numerator = np.sum(weighted, axis=2)
        denominator = np.sum(conf_weight * visible, axis=2) + 1e-9
        oks = numerator / denominator
        oks = np.where(shared_count >= min_shared, oks, 0.0)
        logger.debug(f"[OKS] clamped {np.sum(shared_count < min_shared)} pairs with <{min_shared} shared kps")

        return oks

    def _compute_assignment_gaps(self, cost_matrix: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> np.ndarray:
        gaps = []
        for r, c in zip(row_ind, col_ind):
            best = cost_matrix[r, c]

            row_costs = cost_matrix[r].copy()
            row_costs[c] = np.inf  
            second_best_row = np.min(row_costs)
            col_costs = cost_matrix[:, c].copy()
            col_costs[r] = np.inf  
            second_best_col = np.min(col_costs)

            gap = min(second_best_row - best, second_best_col - best)
            gaps.append(gap)
        
        return np.array(gaps)

    def _build_new_order(self, row_ind:np.ndarray, col_ind:np.ndarray) -> List[int]:
        all_inst = range(self.instance_count)

        processed = {}
        for r, c in zip(row_ind, col_ind):
            target_identity = self.ref_indices[c]
            source_instance = self.pred_indices[r]
            processed[target_identity] = source_instance

        unprocessed = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]
        unassigned = [inst_idx for inst_idx in all_inst if inst_idx not in processed.values()]

        for target_identity in unprocessed:
            if target_identity in unassigned:
                source_instance = target_identity
                processed[target_identity] = source_instance
                unassigned.remove(source_instance)
        
        unprocessed[:] = [inst_idx for inst_idx in all_inst if inst_idx not in processed.keys()]

        for target_identity in unprocessed:
            source_instance = unassigned[-1]
            processed[target_identity] = source_instance
            unassigned.remove(source_instance)
            
        sorted_processed = {k: processed[k] for k in sorted(processed)}
        new_order = list(sorted_processed.values())

        return new_order

    @staticmethod
    def angular_distance(a, b):
        d = np.abs(a - b)
        return np.minimum(d, 2*np.pi - d)

    @staticmethod
    def mask_prop(prop:Track_Properties, mask:np.ndarray):
        return prop.centroids[mask], prop.rotations[mask], prop.poses[mask]