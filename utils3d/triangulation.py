import numpy as np
import cv2

from typing import List, Tuple, Union

def triangulate_point(
    num_views: int,
    projs: List[np.ndarray],
    pts_2d: List[Tuple[float, float]],
    confs: List[float],
    return_confidence: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Triangulates a single 3D point using Direct Linear Transformation (DLT) with confidence weighting.
    Optionally returns a confidence score based on weighted reprojection error and system condition.

    Args:
        num_views (int): Number of camera views.
        projs (list of np.array): List of 3x4 camera projection matrices.
        pts_2d (list of tuple): List of 2D image points (u, v).
        confs (list of float): Confidence values for each 2D point (e.g., detector confidence).
        return_confidence (bool): If True, returns both 3D point and a confidence score.

    Returns:
        If return_confidence is False:
            np.array: 3D point (x, y, z)
        If return_confidence is True:
            tuple: (point_3d: np.array (3,), confidence: float in [0, 1])
    """
    # === Step 1: Build DLT system with confidence weighting ===
    A = []
    valid_indices = []
    for i in range(num_views):
        if np.isnan(pts_2d[i][0]) or np.isnan(pts_2d[i][1]) or confs[i] <= 0:
            continue  # Skip invalid or zero-confidence views
        P_i = np.array(projs[i])
        u, v = pts_2d[i]
        w = confs[i]

        A.append(w * (u * P_i[2, :] - P_i[0, :]))
        A.append(w * (v * P_i[2, :] - P_i[1, :]))
        valid_indices.append(i)

    if len(A) < 4:  # Need at least 2 views (4 equations)
        if return_confidence:
            return np.full(3, np.nan), 0.0
        else:
            return np.full(3, np.nan)

    A = np.array(A)
    
    # === Step 2: Solve via SVD ===
    U, S, Vt = np.linalg.svd(A)
    point_4d_hom = Vt[-1]
    if abs(point_4d_hom[3]) < 1e-10:
        # Poorly conditioned
        if return_confidence:
            return np.full(3, np.nan), 0.0
        else:
            return np.full(3, np.nan)
    
    point_3d = (point_4d_hom / point_4d_hom[3])[:3]

    if not return_confidence:
        return point_3d

    # === Step 3: Compute Reprojection Error (Confidence Basis) ===
    reprojection_errors = []
    weighted_errors = []
    total_weight = 0.0

    for i in valid_indices:
        P_i = np.array(projs[i])
        u_obs, v_obs = pts_2d[i]
        conf = confs[i]

        # Reproject 3D point
        p_3d_hom = np.append(point_3d, 1.0)
        proj = P_i @ p_3d_hom
        u_proj, v_proj, w_proj = proj
        if abs(w_proj) < 1e-10:
            continue
        u_proj /= w_proj
        v_proj /= w_proj

        error = np.hypot(u_proj - u_obs, v_proj - v_obs)  # pixel error
        reprojection_errors.append(error)
        weighted_errors.append(conf * error)
        total_weight += conf

    if len(reprojection_errors) == 0:
        return point_3d, 0.0

    # Mean weighted reprojection error
    mean_weighted_error = sum(weighted_errors) / total_weight if total_weight > 0 else np.inf

    # === Step 4: Compute Condition Number (Numerical Stability) ===
    cond_num = S[0] / (S[-1] + 1e-10)  # Avoid division by zero
    condition_score = np.clip(1.0 / (cond_num / 100.0), 0, 1)  # Normalize to ~[0,1]

    # === Step 5: Map reprojection error to [0,1] confidence ===
    # Example: error > 10px → 0, error < 1px → 1
    max_allowed_error = 10.0  # pixels
    reproj_conf = np.clip(1.0 - (mean_weighted_error / max_allowed_error), 0.0, 1.0)

    # Combine with condition score
    final_confidence = 0.7 * reproj_conf + 0.3 * condition_score

    return point_3d, final_confidence

def triangulate_point_simple(
        proj1:np.ndarray,
        proj2:np.ndarray, 
        pts_2d1:Tuple[float],
        pts_2d2:Tuple[float], 
        conf1:float,
        conf2:float
        ) -> np.ndarray:
    """
    Triangulates a single 3D point from two 2D camera views.

    Args:
        proj1 (np.array): 3x4 projection matrix for the first camera view.
        proj2 (np.array): 3x4 projection matrix for the second camera view.
        pts_2d1 (tuple): 2D image point (u, v) from the first camera view.
        pts_2d2 (tuple): 2D image point (u, v) from the second camera view.
        conf1 (float): Confidence value for the first 2D point.
        conf2 (float): Confidence value for the second 2D point.

    Returns:
        np.array: The triangulated 3D point in Euclidean coordinates (x, y, z).
    """
    num_views = 2
    projs = [proj1, proj2]
    pts_2d = [pts_2d1, pts_2d2]
    confs = [conf1, conf2]
    
    point_3d = triangulate_point(num_views, projs, pts_2d, confs, return_confidence=False)
    return point_3d

def get_projection_matrix(K, R, t):

    """
    Computes the projection matrix from camera intrinsic and extrinsic parameters.

    The projection matrix P combines the camera's intrinsic properties (K)
    with its extrinsic pose (rotation R and translation t) relative to the
    world coordinate system. It maps 3D world points to 2D image points.

    Args:
        K (np.array): The camera intrinsic matrix (3x3).
                    Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        R (np.array): The 3x3 rotation matrix representing the camera's orientation
                    in the world coordinate system.
        t (np.array): The 3x1 or (3,) translation vector representing the camera's
                    position in the world coordinate system.

    Returns:
        np.array: The 3x4 projection matrix P = K * [R | t].
    """
    # Ensure t is a 3x1 column vector
    if t.shape == (3,):
        t = t.reshape(3, 1)
    elif t.shape == (3, 1):
        pass
    else:
        raise ValueError("Translation vector 't' must be of shape (3,) or (3,1)")

    # Concatenate R and t to form the extrinsic matrix [R | t]
    extrinsic_matrix = np.hstack((R, t))
    
    # Projection matrix
    P = K @ extrinsic_matrix
    return P

def undistort_points(points_xy_conf, K, RDistort, TDistort):
    """
    Undistorts 2D image points given camera intrinsic matrix and distortion coefficients.

    Args:
        points_xy_conf (list or np.array): A 1D array or list of (x, y, confidence) triplets.
                                        Example: [x1, y1, conf1, x2, y2, conf2, ...]
        K (np.array): The camera intrinsic matrix (3x3).
                    Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        RDistort (list or np.array): Radial distortion coefficients [k1, k2].
        TDistort (list or np.array): Tangential distortion coefficients [p1, p2].

    Returns:
        np.array: A 1D array of undistorted (x, y, confidence) triplets.
                Example: [undistorted_x1, undistorted_y1, conf1, ...]
    """
    if points_xy_conf.size % 3 != 0:
        raise ValueError("Input 'points_xy_conf' must contain triplets of (x, y, confidence).")
    
    num_points = points_xy_conf.size // 3
    dist_coeffs = np.array([RDistort[0], RDistort[1], TDistort[0], TDistort[1], 0])
    points_xy_conf = np.array(points_xy_conf, dtype=np.float32)

    reshaped_points = points_xy_conf.reshape(-1, 3) # Separate (x, y) coordinates from confidences
    xy_coords = reshaped_points[:, :2]    # Shape (N, 2)
    confidences = reshaped_points[:, 2]   # Shape (N,)

    points_xy_conf = xy_coords.reshape(-1, 1, 2).astype(np.float32) # Reshape points for OpenCV: (N, 1, 2)
    undistorted_pts = cv2.undistortPoints(points_xy_conf, K, dist_coeffs, P=K)

    undistorted_pts_clean = undistorted_pts.reshape(-1, 2) #Reshape the undistorted (x, y) back to a simple (N, 2) array

    output_combined = np.empty((num_points, 3), dtype=np.float32) #Create an empty array to hold the final (x, y, conf) triplets
    output_combined[:, :2] = undistorted_pts_clean
    output_combined[:, 2] = confidences

    return output_combined.flatten() # Reshape back to 1D array