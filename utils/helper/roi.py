import numpy as np
import cv2
from typing import Tuple, Optional


def get_roi_cv2(frame) -> Tuple[int, int, int, int] | None:
    cv2.namedWindow("Select ROI ('space' to accept, 'c' to cancel)", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI ('space' to accept, 'c' to cancel)", frame, fromCenter=False)
    cv2.destroyWindow("Select ROI ('space' to accept, 'c' to cancel)")
    
    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        return (x, y, x + w, y + h)
    else:
        return None

def plot_roi(frame, roi) -> np.ndarray:
    roi = validate_crop_coord(roi)
    if roi is None:
        return frame
    frame = frame.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame

def crop_coord_to_array(crop_coord:np.ndarray|Tuple[int, int, int, int], arr_shape:Tuple[int, int, int]):
    coord_array = np.zeros(arr_shape)
    x, y = crop_coord[0], crop_coord[1]
    coord_array[:, :, 0::3] = x
    coord_array[:, :, 1::3] = y
    return coord_array

def validate_crop_coord(crop_coord:np.ndarray|Tuple[int, int, int, int]|None) -> Optional[Tuple[int, int, int, int]]:
    if crop_coord is None:
        return None
    try:
        x1, y1, x2, y2 = crop_coord
    except Exception:
        return None
    else:
        return (x1, y1, x2, y2)