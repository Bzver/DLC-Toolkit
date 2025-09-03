import numpy as np

def pose_alignment_worker(local_coords:np.ndarray, angles:np.ndarray) -> np.ndarray:
    """
    Rotates all poses to a common orientation using an anchor-reference vector.
    
    Args:
        local_coords (np.ndarray): Relative positions of keypoints, shape (N, K*2).
        angles (np.ndarray): Angles of rotation used to align the poses in radians, shape (N,).

    Returns:
        - Rotated relative positions, shape (N, K*2).
    """
    relative_x, relative_y = local_coords[:, 0::2], local_coords[:, 1::2]
    cos_angles, sin_angles = np.cos(-angles), np.sin(-angles)
    rotated_local = np.empty(local_coords.shape)
    rotated_relative_x = relative_x * cos_angles[:, np.newaxis] - relative_y * sin_angles[:, np.newaxis]
    rotated_relative_y = relative_x * sin_angles[:, np.newaxis] + relative_y * cos_angles[:, np.newaxis]
    rotated_local[:, 0::2], rotated_local[:, 1::2] = rotated_relative_x, rotated_relative_y
    
    return rotated_local

def pose_rotation_worker(angle:float, centroids:np.ndarray, local_coords:np.ndarray, confs:np.ndarray) -> np.ndarray:
    """
    Rotate average relative keypoint positions by a given angle around the average centroid, then compute an aligned pose.
    
    Args:
        angle (float): Rotation angle in radians.
        centroids (np.ndarray): Average centroids of shape (2,) or (N, 2) for same instance across multiple frames.
        local_coords (np.ndarray): Local coordinates of shape (K*2,) or (N, K*2) for same instance across multiple frames.
        confs (np.ndarray): Confidence scores of shape (K,) or (N, K) for same instance across multiple frames.

    Returns:
        np.ndarray: Averaged & rotated single pose in global coordinates, shape (K*3,).
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    avg_centroid = centroids if centroids.ndim == 1 else np.nanmean(centroids, axis=0)
    avg_confs = confs if confs.ndim == 1 else np.nanmean(confs, axis=0)
    if local_coords.ndim == 1:
        avg_local_x, avg_local_y = local_coords[..., 0::2], local_coords[..., 1::2]
    else:
        avg_local_x, avg_local_y = np.nanmean(local_coords[..., 0::2], axis=0), np.nanmean(local_coords[..., 1::2], axis=0)

    global_x = avg_local_x * cos_a - avg_local_y * sin_a + avg_centroid[0]
    global_y = avg_local_x * sin_a + avg_local_y * cos_a + avg_centroid[1]
    average_pose = np.empty(local_coords.shape[-1] // 2 * 3)
    average_pose[0::3], average_pose[1::3], average_pose[2::3] = global_x, global_y, avg_confs
    return average_pose