import numpy as np
from typing import Tuple

from utils.logger import logger


def sigma_estimation(crp, percentile:float = 85.0, min_samples:int = 10, min_disp_px=10.0, max_disp_px=100.0) -> Tuple[float, float]:
    disp_mags = []   # list of (dx² + dy²)
    ang_diffs = []   # list of |Δθ| (wrapped)

    cent, rot, = crp[:2]

    for inst in range(cent.shape[1]):
        x = cent[:, inst, 0]
        y = cent[:, inst, 1]
        r = rot[:, inst]

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
        valid_frames = np.where(valid)[0]

        if len(valid_frames) < 2:
            continue

        for i in range(len(valid_frames) - 1):
            t0, t1 = valid_frames[i], valid_frames[i+1]
            if t1 != t0 + 1:  # Skip non-consecutive frames
                continue

            dx = x[t1] - x[t0]
            dy = y[t1] - y[t0]
            disp_sq = dx*dx + dy*dy

            if disp_sq < min_disp_px**2: # Stationary
                continue
            if disp_sq > max_disp_px**2: # Implausible
                continue

            disp_mags.append(disp_sq)

            dtheta = r[t1] - r[t0]
            dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
            ang_diffs.append(np.abs(dtheta))

    if len(disp_mags) >= min_samples:
        disp_sq_arr = np.array(disp_mags)
        med_disp = np.median(disp_sq_arr)
        if med_disp > 0:
            upper_bound = 9 * med_disp  # (3*sqrt(med))^2 = 9*med
            disp_sq_arr = disp_sq_arr[disp_sq_arr <= upper_bound]

        if len(disp_sq_arr) >= max(3, min_samples // 3):
            sigma_c = np.sqrt(np.percentile(disp_sq_arr, percentile))
        else:
            sigma_c = np.sqrt(med_disp) if med_disp > 0 else 30.0
    else:
        sigma_c = 30.0

    if len(ang_diffs) >= min_samples:
        ang_arr = np.array(ang_diffs)
        med_ang = np.median(ang_arr)
        if med_ang > 0:
            upper_bound = 3 * med_ang
            ang_arr = ang_arr[ang_arr <= upper_bound]
        sigma_r = np.percentile(ang_arr, percentile) if len(ang_arr) > 0 else med_ang
    else:
        sigma_r = 0.5236  # 30 degrees

    sigma_c = max(3.0, sigma_c)
    sigma_r = max(0.05, sigma_r)

    logger.debug(
        f"[SIGMA] ({percentile}th percentile): s1={sigma_c:.1f}px (from {len(disp_mags)} disp), s2={sigma_r:.2f}rad (from {len(ang_diffs)} Δθ)"
    )
    return float(sigma_c), float(sigma_r)

def kappa_estimation(
    aligned_local_coords: np.ndarray,
    confidence_thresh: float = 0.6,
    min_kappa: float = 0.5
) -> np.ndarray:
    """
    Compute per-keypoint κ (tolerance) from aligned canonical poses.
    
    Args:
        aligned_local_coords: (F, I, 3*K)
        confidence_thresh: min prob to consider a keypoint "visible"
        min_kappa: lower bound to avoid numerical issues
    
    Returns:
        kappa: (K,) array of per-keypoint tolerances
    """
    xyconf = aligned_local_coords.shape[-1]
    K = xyconf // 3

    A = aligned_local_coords.reshape(-1, K, 3)
    x, y, conf = A[:, :, 0], A[:, :, 1], A[:, :, 2]
    valid = conf > confidence_thresh

    x_masked = np.where(valid, x, np.nan)
    y_masked = np.where(valid, y, np.nan)

    sigma_x = np.nanstd(x_masked, axis=0, ddof=1)
    sigma_y = np.nanstd(y_masked, axis=0, ddof=1)

    kappa = np.sqrt(sigma_x**2 + sigma_y**2)
    kappa = np.clip(kappa, a_min=min_kappa, a_max=None)
    
    return kappa