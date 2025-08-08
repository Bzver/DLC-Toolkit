import numpy as np
import cv2

from typing import List, Tuple
from numpy.typing import NDArray

def triangulate_point(num_views:int, projs:List[NDArray], pts_2d:List[Tuple[float]], confs:List[float]) -> NDArray:
    """
    Triangulates a single 3D point from multiple 2D camera views using the Direct Linear Transformation (DLT) method.
    Each 2D point's contribution to the system of equations is weighted by its confidence.

    Args:
        num_views (int): The number of camera views providing observations for this point.
        projs (list of np.array): A list of 3x4 projection matrices, one for each camera view.
        pts_2d (list of tuple): A list of 2D image points (u, v), one for each camera view.
        confs (list of float): A list of confidence values, one for each 2D point. Used as weights in the triangulation.

    Returns:
        np.array: The triangulated 3D point in Euclidean coordinates (x, y, z).
    """
    A = []
    for i in range(num_views):
        P_i = projs[i]
        u, v = pts_2d[i]
        w = confs[i] # Weight by confidence

        P_i = np.array(P_i) # Ensure P_i is a numpy array for slicing

        # Equations for DLT:
        # u * P_i[2,:] - P_i[0,:] = 0
        # v * P_i[2,:] - P_i[1,:] = 0
        # Apply weight 'w' to each row
        A.append(w * (u * P_i[2,:] - P_i[0,:]))
        A.append(w * (v * P_i[2,:] - P_i[1,:]))

    A = np.array(A) # Solve Ax = 0 using SVD
    U, S, Vt = np.linalg.svd(A) # The 3D point is the last column of V (or last row of Vt)
    
    point_4d_hom = Vt[-1] 
    point_3d = (point_4d_hom / point_4d_hom[3]).flatten()[:3] # Convert from homogeneous to Euclidean coordinates (x/w, y/w, z/w)
    return point_3d

def triangulate_point_simple(proj1:NDArray, proj2:NDArray, pts_2d1:Tuple[float], pts_2d2:Tuple[float], conf1:float, conf2:float) -> NDArray:
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
    
    point_3d = triangulate_point(num_views, projs, pts_2d, confs)
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