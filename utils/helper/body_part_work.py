import numpy as np
from typing import List, Tuple, Optional

from utils.logger import logger


def infer_head_tail_indices(keypoint_names:List[str]) -> Tuple[int,int]:
    """
    Infer head and tail keypoint indices from keypoint names with robust handling
    of capitalization, underscores, and common anatomical naming patterns.
    
    Args:
        keypoint_names: list of all the keypoint names

    Returns:
        idx of supposed head and tail keypoint
    """
    # Define priority-ordered keywords (lowercase, without underscores)
    head_keywords_priority = [
        'nose',
        'head',
        'forehead',
        'front',
        'snout',
        'face',
        'mouth',
        'muzzle',
        'spinF',
        'neck',
        'eye',
        'ear',
        'cheek',
        'chin',
        'anterior',
    ]
    tail_keywords_priority = [
        'tailbase',
        'base_tail',
        'tail_base',
        'butt',
        'hip',
        'rump',
        'thorax',
        'ass',
        'pelvis',
        'tail',
        'spineM',
        'cent',
        'posterior',
        'back',
    ]

    def normalize(name): # Normalize keypoint names: lowercase, remove non-alphanumeric, collapse underscores
        return ''.join(c.lower() for c in name if c.isalnum())

    normalized_names = [normalize(name) for name in keypoint_names]

    # Search with priority: return first match in priority list
    head_idx = None
    for kw in head_keywords_priority:
        normalized_kw = normalize(kw)
        for idx, norm_name in enumerate(normalized_names):
            if normalized_kw in norm_name:
                head_idx = idx
                break
        if head_idx is not None:
            break

    tail_idx = None
    for kw in tail_keywords_priority:
        normalized_kw = normalize(kw)
        for idx, norm_name in enumerate(normalized_names):
            if normalized_kw in norm_name:
                tail_idx = idx
                break
        if tail_idx is not None:
            break

    if head_idx is None:
        logger.warning("Could not infer head keypoint from keypoint names.")
    if tail_idx is None:
        logger.warning("Could not infer tail keypoint from keypoint names.")

    return head_idx, tail_idx

def build_angle_map(canon_pose:np.ndarray, head_idx:int, tail_idx:int) -> dict:
    sq_dist = np.sum(canon_pose**2, axis=1)
    center_kp = np.argmin(sq_dist)
    return {"head_idx": head_idx, "tail_idx": tail_idx, "center_idx": center_kp}

def build_weighted_pose_vectors(pred_data_array:np.ndarray, angle_map_data:dict, max_connections:int=6, min_weight: float = 0.1) -> Optional[np.ndarray]:
    angle_map = angle_map_data["angle_map"]
    
    reliable = [conn for conn in angle_map if conn["weight"] >= min_weight][:max_connections]
    
    if not reliable:
        logger.warning("No reliable connections found in angle_map.")
        return None
    
    M = len(reliable)
    T, N = pred_data_array.shape[:2]
    vectors = np.full((T, N, 2 * M), np.nan, dtype=np.float32)
    
    xs = pred_data_array[..., 0::3]
    ys = pred_data_array[..., 1::3]
    
    for idx, conn in enumerate(reliable):
        i, j, w = conn["i"], conn["j"], conn["weight"]

        dx = w * (xs[:, :, j] - xs[:, :, i])
        dy = w * (ys[:, :, j] - ys[:, :, i])
        
        valid = (~np.isnan(xs[:, :, i]) & ~np.isnan(xs[:, :, j]) & ~np.isnan(ys[:, :, i]) & ~np.isnan(ys[:, :, j]))
        
        vectors[:, :, 2*idx]     = np.where(valid, dx, np.nan)
        vectors[:, :, 2*idx + 1] = np.where(valid, dy, np.nan)
    
    return vectors
