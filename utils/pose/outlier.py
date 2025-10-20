import numpy as np
from itertools import combinations

from typing import Tuple

from utils.helper import get_instances_on_current_frame, get_instance_count_per_frame
from .pose_analysis import calculate_pose_centroids, calculate_pose_bbox

def outlier_removal(pred_data_array:np.ndarray, outlier_mask:np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Removes outliers from the prediction data by setting flagged values to NaN.

    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence).
        outlier_mask (np.ndarray): Boolean mask of shape (num_frames, num_instances) indicating 
            which instance-frame pairs are outliers.

    Returns:
        Tuple[np.ndarray, int, int]:
            - Modified prediction array with outliers set to NaN.
            - Number of frames that contained at least one outlier.
            - Total number of instance outliers removed.
    """
    removed_frames_count = np.sum(np.any(outlier_mask, axis=1))
    removed_instances_count = np.sum(outlier_mask)
    pred_data_array[outlier_mask] = np.nan
    return pred_data_array, removed_frames_count, removed_instances_count

########################################################################################################

def outlier_confidence(pred_data_array:np.ndarray, threshold:float=0.5) -> np.ndarray:
    """
    Flags instances with average keypoint confidence below a given threshold.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        threshold (float): Minimum average confidence for an instance to be considered valid.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates low confidence.
    """
    confidence_scores = pred_data_array[:, :, 2::3]
    inst_conf = np.nanmean(confidence_scores, axis=2)
    low_conf_mask = inst_conf < threshold
    return low_conf_mask

def outlier_bodypart(pred_data_array:np.ndarray, threshold:int=2) -> np.ndarray:
    """
    Flags instances that have fewer detected keypoints than the specified threshold.

    Only considers instances with at least one valid keypoint (excludes completely missing ones).

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        threshold (int): Minimum number of visible keypoints required.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates too few body parts.
    """
    x_coords_mask = ~np.isnan(pred_data_array[:, :, 0::3])
    y_coords_mask = ~np.isnan(pred_data_array[:, :, 1::3])

    discovered_bodyparts = np.sum(np.logical_and(x_coords_mask, y_coords_mask), axis=2)
    low_bp_mask = np.logical_and(discovered_bodyparts < threshold,
        discovered_bodyparts > 0) # No need to include empty instances
    
    return low_bp_mask

def outlier_instance(pred_data_array:np.ndarray, threshold:int) -> np.ndarray:
    """
    Flags all instances in frames where the total number of detected instances is below a threshold.

    Used to remove frames with abnormally low instance counts across the board.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        threshold (int): Minimum number of instances required per frame.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where all instances in underpopulated 
                    frames are flagged.
    """
    instance_count = pred_data_array.shape[1]

    instance_count_per_frame = get_instance_count_per_frame(pred_data_array)
    low_instance_mask_frame = instance_count_per_frame < threshold

    # Broadcast to all instances: if frame is bad, all instances are flagged
    low_instance_mask = np.tile(low_instance_mask_frame[:, np.newaxis], (1, instance_count))

    return low_instance_mask

def outlier_duplicate(
        pred_data_array:np.ndarray,
        bp_threshold:float=0.5,
        dist_threshold:float=10.0
        ) -> np.ndarray:
    """
    Detects duplicate detections by identifying instances whose keypoints are abnormally close.

    Compares all pairs of instances. An instance is flagged if a large proportion of its keypoints 
    are within a small distance of another instance's keypoints. The lower-confidence or smaller 
    instance is marked as the duplicate.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        bp_threshold (float): Proportion of overlapping keypoints required to flag duplicates.
        dist_threshold (float): Maximum distance (in pixels) for two keypoints to be considered overlapping.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates a duplicate instance.
    """
    total_frames, instance_count, _ = pred_data_array.shape
    instances = list(range(instance_count))
    duplicate_mask = np.zeros((total_frames, instance_count), dtype=bool)

    for inst_1_idx, inst_2_idx in combinations(instances, 2):
        inst_1_x = pred_data_array[:, inst_1_idx, 0::3]
        inst_1_y = pred_data_array[:, inst_1_idx, 1::3]
        inst_2_x = pred_data_array[:, inst_2_idx, 0::3]
        inst_2_y = pred_data_array[:, inst_2_idx, 1::3]

        if np.all(np.isnan(inst_1_x)) or np.all(np.isnan(inst_2_x)):
            continue
        
        inst_1_conf = np.nanmean(pred_data_array[:, inst_1_idx, 2::3], axis=1)
        inst_2_conf = np.nanmean(pred_data_array[:, inst_2_idx, 2::3], axis=1)
        inst_1_valid_kp = np.sum(~np.isnan(inst_1_x), axis=1)
        inst_2_valid_kp = np.sum(~np.isnan(inst_2_x), axis=1)

        euclidean_dist = np.sqrt((inst_1_x - inst_2_x)**2 + (inst_1_y - inst_2_y)**2)

        close_kp = np.sum(euclidean_dist <= dist_threshold, axis=1)
        suspect_list = np.where(close_kp > 0)[0].tolist()

        for frame_idx in suspect_list:
            if close_kp[frame_idx] < min(inst_1_valid_kp[frame_idx],inst_2_valid_kp[frame_idx]) * bp_threshold:
                continue
            elif close_kp[frame_idx] > max(inst_1_valid_kp[frame_idx],inst_2_valid_kp[frame_idx]) * bp_threshold:
                inst_to_mark = inst_2_idx if inst_1_conf[frame_idx] >= inst_2_conf[frame_idx] else inst_1_idx
            else:
                inst_to_mark = inst_2_idx if inst_1_valid_kp[frame_idx] > inst_2_valid_kp[frame_idx] else inst_1_idx
            
            duplicate_mask[frame_idx, inst_to_mark] = True
            
    return duplicate_mask

def outlier_size(
        pred_data_array:np.ndarray,
        canon_pose:np.ndarray,
        min_ratio:float=0.3,
        max_ratio:float=2.5
        ) -> np.ndarray:
    """
    Flags instances whose size (mean distance of keypoints from centroid) is significantly different 
    from the canonical pose size.

    Helps detect implausible detections (e.g., extremely small or large animal poses).

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        canon_pose (np.ndarray): Canonical 2D pose of shape (num_keypoints, 2) used as size reference.
        min_ratio (float): Instances smaller than this ratio of the canonical size are flagged.
        max_ratio (float): Instances larger than this ratio of the canonical size are flagged.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates abnormal size.
    """
    total_frames, instance_count, _ = pred_data_array.shape
    canon_radius = np.mean(np.sqrt(canon_pose[:, 0]**2 + canon_pose[:, 1]**2))
    _, local_coords = calculate_pose_centroids(pred_data_array)
    inst_radius = np.full((total_frames, instance_count), np.nan)
    for inst_idx in range(instance_count):
        inst_radius[:, inst_idx] = np.nanmean(
            np.sqrt(local_coords[:, inst_idx, 0::2]**2 + local_coords[:, inst_idx, 1::2]**2), axis=1)
        
    small_mask = inst_radius < canon_radius * min_ratio
    large_mask = inst_radius >= canon_radius * max_ratio

    return np.logical_or(small_mask, large_mask)

def outlier_flicker(pred_data_array:np.ndarray, min_duration:int=2) -> np.ndarray:
    """
    Flags instances that appear and disappear rapidly (flickering), indicating unstable detection.

    An instance is considered flickering if it appears for fewer than `min_duration` consecutive frames.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        min_duration (int): Minimum number of consecutive frames an instance must be present to avoid being flagged.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates flickering behavior.
    """
    total_frames, instance_count, xyconf = pred_data_array.shape

    inst_buffer = np.full((1, instance_count, xyconf), np.nan)
    pred_data_with_buffer = np.concatenate((inst_buffer, pred_data_array), axis=0)
    inst_presence = np.any(~np.isnan(pred_data_with_buffer), axis=2).astype(int)
    appear_mask = np.diff(inst_presence, axis=0) == 1
    disappear_mask = np.diff(inst_presence, axis=0) == -1

    flicker_mask = np.zeros((total_frames, instance_count), dtype=bool)

    for inst_idx in range(instance_count):
        appear_frames = np.where(appear_mask[:, inst_idx])[0]
        disappear_frames = np.where(disappear_mask[:, inst_idx])[0]

        # Extend disappear_frames to handle "still present at end"
        if inst_presence[-1, inst_idx]:
            disappear_frames = np.append(disappear_frames, total_frames)

        # Match each appearance with next disappearance
        for a_frame in appear_frames:
            post_dis = disappear_frames[disappear_frames > a_frame]
            if len(post_dis) == 0:
                continue
            d_frame = post_dis[0]
            duration = d_frame - a_frame  # how long it stayed

            if duration < min_duration:
                flicker_mask[a_frame:d_frame, inst_idx] = True

    return flicker_mask

def outlier_enveloped(pred_data_array:np.ndarray, threshold:float=0.6) -> np.ndarray:
    """
    Flags smaller instances that are largely contained within the bounding box of a larger, nearby instance.

    Used to detect potential occlusion or false positives where one animal's pose is incorrectly 
    detected inside another's.

    Args:
        pred_data_array (np.ndarray): Prediction array of shape (num_frames, num_instances, num_keypoints * 3).
        threshold (float): Proportion of a small instance's keypoints that must lie within 
                                     the larger instance's bounding box to be flagged.

    Returns:
        np.ndarray: Boolean mask of shape (num_frames, num_instances) where True indicates an enveloped instance.
    """
    total_frames, instance_count, _ = pred_data_array.shape

    all_x, all_y = pred_data_array[:, :, 0::3], pred_data_array[:, :, 1::3]
    min_x_array, min_y_array, max_x_array, max_y_array = calculate_pose_bbox(all_x, all_y)
    size_array = (max_y_array - min_y_array) * (max_x_array - min_x_array)
    valid_sizes = size_array[~np.isnan(size_array)]
    mean_size = np.nanmean(valid_sizes)
    bbox_length = np.sqrt(mean_size) 

    enveloped_mask = np.zeros((total_frames, instance_count), dtype=bool)

    for frame_idx in range(total_frames):
        valid_inst = get_instances_on_current_frame(pred_data_array, frame_idx)
        if len(valid_inst) < 2:
            continue

        close_pairs = []
        centroids, _ = calculate_pose_centroids(pred_data_array, frame_idx)
        for inst_1_idx, inst_2_idx in combinations(valid_inst, 2):
            inst_dist = np.linalg.norm(centroids[inst_2_idx] - centroids[inst_1_idx])
            if inst_dist < bbox_length * 0.5:
                close_pairs.append((inst_1_idx,inst_2_idx))
        
        if not close_pairs:
            continue

        for inst_1_idx, inst_2_idx in close_pairs:
            inst_1_size = size_array[frame_idx, inst_1_idx]
            inst_2_size = size_array[frame_idx, inst_2_idx]
            if inst_1_size > inst_2_size:
                small_inst = inst_2_idx
                large_inst = inst_1_idx
            else:
                small_inst = inst_1_idx
                large_inst = inst_2_idx

            small_x = all_x[frame_idx, small_inst]
            small_y = all_y[frame_idx, small_inst]

            valid = np.logical_and(~np.isnan(small_x), ~np.isnan(small_y))
            in_bbox_mask = (
                    (small_x >= min_x_array[frame_idx, large_inst]) &
                    (small_x <= max_x_array[frame_idx, large_inst]) &
                    (small_y >= min_y_array[frame_idx, large_inst]) &
                    (small_y <= max_y_array[frame_idx, large_inst])
                )

            in_ratio = np.sum(in_bbox_mask & valid) / np.sum(valid)
            if in_ratio >= threshold:
                enveloped_mask[frame_idx, small_inst] = True
            
    return enveloped_mask