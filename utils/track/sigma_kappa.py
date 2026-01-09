import numpy as np
from typing import Tuple

from utils.logger import logger


def sigma_estimation(crp, percentile:float = 90.0, min_samples:int = 10) -> Tuple[float, float]:
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
            if t1 != t0 + 1:  # skip non-consecutive frames (occlusion gap)
                continue

            dx = x[t1] - x[t0]
            dy = y[t1] - y[t0]
            disp_sq = dx*dx + dy*dy
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
        f"[SIGMA] Robust ({percentile}th percentile): "
        f"s1={sigma_c:.1f}px (from {len(disp_mags)} disp), "
        f"s2={sigma_r:.2f}rad (from {len(ang_diffs)} Δθ)"
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

def adaptive_sigma_nudge(
        crp_clean:Tuple[np.ndarray, np.ndarray, np.ndarray],
        cr_sigma_curr:Tuple[float, float],
        alpha: float = 0.2
        ) -> Tuple[float, float]:
    centroids, rotations = crp_clean[:2]
    F, I = centroids.shape[0], centroids.shape[1]

    disp_mags, ang_diffs = [], []
    for inst in range(I):
        x, y, r = centroids[:, inst, 0], centroids[:, inst, 1], rotations[:, inst]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
        valid_frames = np.where(valid)[0]
        for i in range(len(valid_frames) - 1):
            t0, t1 = valid_frames[i], valid_frames[i+1]
            if t1 == t0 + 1:
                disp_mags.append((x[t1]-x[t0])**2 + (y[t1]-y[t0])**2)
                dtheta = (r[t1] - r[t0] + np.pi) % (2*np.pi) - np.pi
                ang_diffs.append(abs(dtheta))

    sigma_c_win = 30.0
    sigma_r_win = 0.5236
    if len(disp_mags) >= 5:
        disp_arr = np.array(disp_mags)
        med = np.median(disp_arr)
        if med > 0:
            disp_arr = disp_arr[disp_arr <= 9*med]
        if len(disp_arr) >= 3:
            sigma_c_win = np.sqrt(np.percentile(disp_arr, 90))
    if len(ang_diffs) >= 5:
        ang_arr = np.array(ang_diffs)
        med = np.median(ang_arr)
        if med > 0:
            ang_arr = ang_arr[ang_arr <= 3*med]
        if len(ang_diffs) >= 3:
            sigma_r_win = np.percentile(ang_arr, 90)

    sigma_c_win = max(3.0, sigma_c_win)
    sigma_r_win = max(0.05, sigma_r_win)

    alpha_adapt = min(alpha, F / 200)
    alpha_adapt = max(alpha_adapt, 0.05)

    sigma_c_curr, sigma_r_curr = cr_sigma_curr
    sigma_c_new = (1 - alpha_adapt) * sigma_c_curr + alpha_adapt * sigma_c_win
    sigma_r_new = (1 - alpha_adapt) * sigma_r_curr + alpha_adapt * sigma_r_win

    logger.debug(f"[SIGMA] Nudged σ (α={alpha_adapt:.2f}): c={sigma_c_new:.1f}, r={sigma_r_new:.2f}")
    
    return float(sigma_c_new), float(sigma_r_new)

def adaptive_kappa_nudge(
        clean_pred: np.ndarray,
        kappa_curr: np.ndarray,
        alpha: float = 0.2
        ):
    F, _, xyconf = clean_pred.shape
    K = xyconf // 3
    A = clean_pred.reshape(-1, K, 3)
    x, y, conf = A[:, :, 0], A[:, :, 1], A[:, :, 2]
    valid = conf > 0.6

    x_masked = np.where(valid, x, np.nan)
    y_masked = np.where(valid, y, np.nan)

    sigma_x = np.nanstd(x_masked, axis=0, ddof=1)
    sigma_y = np.nanstd(y_masked, axis=0, ddof=1)
    kappa_window = np.sqrt(sigma_x**2 + sigma_y**2)
    kappa_window = np.clip(kappa_window, 0.5, None)

    alpha_adapt = min(alpha, F / 200)
    alpha_adapt = max(alpha_adapt, 0.05)

    kappa = (1 - alpha_adapt) * kappa_curr + alpha_adapt * kappa_window
    logger.debug(f"[KAPPA] Nudged κ (α={alpha_adapt:.2f}). Mean: {np.mean(kappa):.2f}")

    return kappa