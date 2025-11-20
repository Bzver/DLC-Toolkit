import numpy as np
import cv2
from PySide6 import QtGui
from PySide6.QtWidgets import QMessageBox
from typing import List, Tuple, Callable, Union

def get_instances_on_current_frame(pred_data_array:np.ndarray, current_frame_idx:int) -> List[int]:
    """
    Identifies which instances are present in a given frame based on non-NaN keypoint data.

    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence) for each keypoint.
        current_frame_idx (int): Index of the frame to check.

    Returns:
        List[int]: List of instance indices that have at least one valid keypoint 
                   in the specified frame.
    """
    instance_count = pred_data_array.shape[1]
    current_frame_inst = []
    for inst_idx in range(instance_count):
        if np.any(~np.isnan(pred_data_array[current_frame_idx, inst_idx, :])):
            current_frame_inst.append(inst_idx)
    return current_frame_inst

def get_instance_count_per_frame(pred_data_array:np.ndarray) -> np.ndarray:
    """
    Count the number of non-empty instances per frame.
    
    Args:
        pred_data_array (np.ndarray): Array of shape (num_frames, num_instances, num_keypoints * 3) 
            containing flattened 2D predictions (x, y, confidence) for each keypoint.
    
    Returns:
        Array of shape (n_frames,) with count of valid instances per frame.
    """
    non_empty_instance_numerical = (np.any(~np.isnan(pred_data_array), axis=2)) * 1
    instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
    return instance_count_per_frame

#########################################################################################################################################################1

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
        print("Warning: Could not infer head keypoint from keypoint names.")
    if tail_idx is None:
        print("Warning: Could not infer tail keypoint from keypoint names.")

    return head_idx, tail_idx

def build_angle_map(canon_pose:np.ndarray, all_frame_poses:np.ndarray , head_idx:int, tail_idx:int) -> dict:
    canonical_vec = canon_pose[head_idx] - canon_pose[tail_idx]
    num_keypoint = canon_pose.shape[0]
    if np.linalg.norm(canonical_vec) < 1e-6:
        canonical_body_angle = 0.0
    else:
        canonical_body_angle = np.arctan2(canonical_vec[1], canonical_vec[0])

    # Build angle map for every possible connection
    angle_map = []  # (i, j, expected_offset, weight)
    all_angles = np.arctan2(all_frame_poses[:, 1::2], all_frame_poses[:, 0::2])  # (N, K)

    for i in range(num_keypoint):
        for j in range(num_keypoint):
            if i == j:
                continue

            # Vector from i to j in canon pose
            vec = canon_pose[j] - canon_pose[i]
            if np.linalg.norm(vec) < 1e-6:
                continue

            # Expected angle of this vector
            raw_angle = np.arctan2(vec[1], vec[0])

            # Offset relative to canonical body angle
            offset = np.arctan2(
                np.sin(raw_angle - canonical_body_angle),
                np.cos(raw_angle - canonical_body_angle)
            )  # Wrap to [-π, π]

            # Measure angular variation (in radians)
            ij_angles = all_angles[:, j] - all_angles[:, i]  # (N,)
            ij_angles = np.arctan2(np.sin(ij_angles), np.cos(ij_angles))  # Unwrap
            var = np.nanvar(ij_angles)

            # Weight: high if stable and aligned with body
            length = np.linalg.norm(vec)
            stability = 1.0 / (1.0 + var) if var > 0 else 1.0
            alignment = abs(np.dot(vec / np.linalg.norm(vec), canonical_vec / np.linalg.norm(canonical_vec)))

            weight = length * stability * alignment

            angle_map.append({"i": i, "j": j,"offset": offset,"weight": weight})

    # Sort by weight (most reliable first)
    angle_map.sort(key=lambda x: x["weight"], reverse=True)
    
    angle_map_data = {"head_idx": head_idx, "tail_idx": tail_idx, "angle_map": angle_map}

    return angle_map_data

#########################################################################################################################################################1

def log_print(*args, enabled=True, **kwargs):
    if not enabled:
        return
    try:
        log_file = "D:/Project/debug_log.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs)
    except:
        pass

def clean_log():
    try:
        log_file = "D:/Project/debug_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            pass
    except:
        pass

#########################################################################################################################################################1

def clean_inconsistent_nans(pred_data_array:np.ndarray):
    print("Cleaning up NaN keypoints that somehow has confidence value...")
    nan_mask = np.isnan(pred_data_array)
    x_is_nan = nan_mask[:, :, 0::3]
    y_is_nan = nan_mask[:, :, 1::3]
    keypoints_to_fully_nan = x_is_nan | y_is_nan
    full_nan_sweep_mask = np.repeat(keypoints_to_fully_nan, 3, axis=-1)
    pred_data_array[full_nan_sweep_mask] = np.nan
    print("NaN keypoint confidence cleaned.")
    return pred_data_array

#########################################################################################################################################################1

def format_title(base_title: str, debug_status: bool) -> str:
    return f"{base_title} --- DEBUG MODE" if debug_status else base_title

def handle_unsaved_changes_on_close(
        parent,
        event,
        is_saved:bool,
        save_callback:Callable[[], bool]
        ):
    """
    Prompts the user when attempting to close a window with unsaved changes, offering 
    options to save, discard, or cancel the close action.

    Args:
        parent: Parent widget (e.g., QMainWindow) used for modal dialog positioning.
        event: Close event object that will be accepted or ignored based on user choice.
        is_saved (bool): Flag indicating whether the current state is already saved. 
                         If True, the window closes immediately without prompting.
        save_callback (Callable[[], bool]): Function to call when the user chooses to save. 
                                            Should return True on successful save, False otherwise.

    Returns:
        None: This function directly controls the event's acceptance or rejection.
              It does not return a value but affects application flow by accepting 
              or ignoring the close event based on user interaction.
    """
    if is_saved:
        event.accept()
        return
    
    close_call = QMessageBox(parent)
    close_call.setWindowTitle("Changes Unsaved")
    close_call.setText("Do you want to save your changes before closing?")
    close_call.setIcon(QMessageBox.Icon.Question)

    save_btn = close_call.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
    discard_btn = close_call.addButton("Don't Save", QMessageBox.ButtonRole.DestructiveRole)
    close_btn = close_call.addButton("Close", QMessageBox.RejectRole)
    
    close_call.setDefaultButton(close_btn)

    close_call.exec()
    clicked_button = close_call.clickedButton()
    
    if clicked_button == save_btn:
        success = save_callback()
        if success:
            event.accept()
        else:
            event.ignore()
    elif clicked_button == discard_btn:
        event.accept()  # Close without saving
    else:
        event.ignore()  # Cancel the close action

###########################################################################################

def calculate_snapping_zoom_level(
        current_frame_data:np.ndarray,
        view_width:float,
        view_height:float
        )->Tuple[float,float,float]:
    """
    Calculates an optimal zoom level and center position to fit all visible keypoints 
    in the current frame within the view, with padding.

    The function computes the bounding box of all non-NaN 2D keypoint coordinates, 
    applies uniform padding, and determines the maximum zoom level that fits the padded 
    box within the given view dimensions. The result centers the keypoints in the view.

    Args:
        current_frame_data (np.ndarray): Array of shape (num_instances * num_keypoints, 3) 
            containing flattened x, y, confidence values for all keypoints in the frame.
        view_width (float): Width of the target view (e.g., graphics scene or display window).
        view_height (float): Height of the target view.

    Returns:
        Tuple[float, float, float]:
            - new_zoom_level (float): Scaling factor to apply (clamped between 0.1 and 10.0).
            - center_x (float): X-coordinate of the center of the bounding box.
            - center_y (float): Y-coordinate of the center of the bounding box.
    """
    x_vals_current_frame = current_frame_data[:, 0::3]
    y_vals_current_frame = current_frame_data[:, 1::3]

    if np.all(np.isnan(x_vals_current_frame)):
        return
    
    min_x = np.nanmin(x_vals_current_frame)
    max_x = np.nanmax(x_vals_current_frame)
    min_y = np.nanmin(y_vals_current_frame)
    max_y = np.nanmax(y_vals_current_frame)

    padding_factor = 1.25 # 25% padding
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    padded_width = width * padding_factor
    padded_height = height * padding_factor
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate new zoom level
    if padded_width > 0 and padded_height > 0:
        zoom_x = view_width / padded_width
        zoom_y = view_height / padded_height
        new_zoom_level = min(zoom_x, zoom_y)
    else:
        new_zoom_level = 1.0

    # Apply zoom limits
    new_zoom_level = max(0.1, min(new_zoom_level, 10.0))

    return new_zoom_level, center_x, center_y

###########################################################################################

def frame_to_pixmap(frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qt_image)
    return pixmap, w, h

###########################################################################################

def get_roi_cv2(frame) -> Tuple[int, int, int, int] | None:
    cv2.namedWindow("Select ROI ('space' to accept, 'c' to cancel)", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI ('space' to accept, 'c' to cancel)", frame, fromCenter=False)
    cv2.destroyWindow("Select ROI ('space' to accept, 'c' to cancel)")
    
    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        return (x, y, x + w, y + h)
    else:
        return None

def crop_coord_to_array(crop_coord:np.ndarray, arr_shape:Tuple[int, int, int], frame_list:List[int]):
    coord_array = np.zeros(arr_shape)
    
    if arr_shape[0] != len(frame_list):
        diff = abs(arr_shape[0]-len(frame_list))
        if diff > 1:
            raise ValueError(
                f"Dimension mismatch: arr_shape[0] = {arr_shape[0]}, "
                f"but len(frame_list) = {len(frame_list)} (difference = {diff} > 1). "
                f"Expected near-equal lengths for alignment."
            )
        final_len = min(arr_shape[0], len(frame_list))
        frame_list = frame_list[0:final_len]

    x, y = crop_coord[0], crop_coord[1]
    
    coord_array[:, :, 0::3] = x
    coord_array[:, :, 1::3] = y
    return coord_array

#########################################################################################################################################################1

def indices_to_spans(indices: Union[np.ndarray, List[int]]) -> List[Tuple[int, int]]:
    """
    Convert a list/array of frame indices into contiguous spans (start, end).
    
    Example:
        >>> indices_to_spans([1,2,3,5,6,9])
        [(1, 3), (5, 6), (9, 9)]
    """
    if len(indices) == 0:
        return []
    
    if isinstance(indices, list):
        indices = np.asarray(indices, dtype=np.int32)

    indices = np.sort(indices)

    n = indices.size
    if n == 1:
        i0 = int(indices[0])
        return [(i0, i0)]

    split_at = np.where(np.diff(indices) > 1)[0] + 1

    if split_at.size == 0:
        return [(int(indices[0]), int(indices[-1]))]

    chunks = np.split(indices, split_at)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks]