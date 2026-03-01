import numpy as np
import cv2
from PySide6 import QtGui
from typing import Union, Tuple, Literal


def frame_to_qimage(frame:np.ndarray, request_dim=False) -> Union[QtGui.QImage, Tuple[QtGui.QImage, int, int]]:
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty")

    h, w = frame.shape[:2]

    if frame.ndim == 3 and frame.shape[2] == 4: # BGRA → RGBA
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        bytes_per_line = 4 * w
        qt_image = QtGui.QImage(rgba.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    elif frame.ndim == 3 and frame.shape[2] == 3: # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qt_image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    elif frame.ndim == 2: # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        bytes_per_line = 3 * w
        qt_image = QtGui.QImage(gray.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    qt_image = qt_image.copy()

    if request_dim:
        return qt_image, w, h
    else:
        return qt_image

def frame_to_pixmap(frame, request_dim=False) -> QtGui.QPixmap | Tuple[QtGui.QPixmap, int, int]:
    qt_image, w, h = frame_to_qimage(frame, request_dim=True)
    pixmap = QtGui.QPixmap.fromImage(qt_image)
    if request_dim:
        return pixmap, w, h
    else:
        return pixmap
    
def get_smart_bg_masking(
    frame_batched: np.ndarray,
    background: np.ndarray,
    threshold: int = 25,
    intensity_margin: float = 0.3,
    min_fg_pixels: int = 100,
    polarity: Literal["Dark Blobs", "Light Blobs"] = "Dark Blobs",
) -> np.ndarray:

    H, W = frame_batched.shape[1:3]

    f = frame_batched.astype(np.int16)
    b = background.astype(np.int16)
    diff_bg = np.max(np.abs(f - b), axis=3)
    fg_mask = diff_bg >= threshold

    fg_pixel_count = np.sum(fg_mask)

    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    gray = np.tensordot(frame_batched.astype(np.float32), weights, axes=([3], [0]))  # (B, H, W)
    bg_gray = np.tensordot(background.astype(np.float32), weights, axes=([2], [0]))  # (H, W)

    artifact_mask = np.zeros((H, W), dtype=np.int16)

    if fg_pixel_count >= min_fg_pixels:
        fg_vals = gray[fg_mask]
        match polarity:
            case "Dark Blobs":
                mask_value = 255
                mouse_intensity = np.percentile(fg_vals, 20) if len(fg_vals) > 0 else 0.0
                low, high = 0, mouse_intensity * (1 + intensity_margin)
            case "Light Blobs":
                mask_value = -255
                mouse_intensity = np.percentile(fg_vals, 80) if len(fg_vals) > 0 else 255.0
                low, high = mouse_intensity * (1 - intensity_margin), 255

        intensity_condition = (bg_gray >= low) & (bg_gray <= high)
        artifact_mask[intensity_condition] = mask_value

        artifact_mask = np.repeat(artifact_mask[..., np.newaxis], 3, axis=2)

    return artifact_mask

def mask_to_qimage(mask):
    h, w = mask.shape[:2]
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    first_ch = mask[:, :, 0]
    white_mask = (first_ch == 255)
    black_mask = (first_ch == -255)
    arr[white_mask, :] = [255, 255, 255, 255]
    arr[black_mask, :] = [0, 0, 0, 255]

    return frame_to_qimage(arr)

def frame_to_grayscale(frame:np.ndarray, keep_as_bgr:bool=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if keep_as_bgr:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        return gray
