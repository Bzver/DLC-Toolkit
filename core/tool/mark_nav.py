import bisect
from typing import Optional, Callable, List, Literal

from utils.logger import Loggerbox

def navigate_to_marked_frame(
        parent,
        frame_list:List[int],
        current_frame_idx:int,
        change_frame_callback:Callable[[int], None],
        direction:Literal["prev","next"]
        ):
    """
    Navigates to the previous or next frame in a sorted list of frames.

    Args:
        parent: Parent widget for displaying warning or error messages.
        frame_list (List[int]): List of frame indices that are marked.
        current_frame_idx (int): Index of the currently displayed frame.
        change_frame_callback (Callable[[int], None]): Function to call with the destination frame index.
        direction (Literal["prev", "next"]): Direction of navigation — either "prev" or "next".

    Behavior:
        - Sorts the frame list and finds the nearest previous or next frame.
        - If no such frame exists, shows a warning.
        - Otherwise, calls the frame change callback with the target frame.
        - On exception during callback, shows a critical error message.
    """
    if not frame_list:
        Loggerbox.warning(parent, "No Marked Frames", "No marked frames to navigate.")
        return
    
    frame_list.sort()

    if direction == "prev":
        dest_frame_idx = _get_prev_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No previous marked frame found."
    elif direction == "next":
        dest_frame_idx = get_next_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No next marked frame found."
    
    if dest_frame_idx is None:
        Loggerbox.warning(parent, "Navigation", no_frame_message)
        return

    try:
        change_frame_callback(dest_frame_idx)
    except Exception as e:
        Loggerbox.error(parent, "Exception", e, exc=e)

def _get_prev_frame_in_list(frame_list:List[int], current_frame_idx:int) -> Optional[int]:
    try:
        current_idx_in_list = frame_list.index(current_frame_idx)
        prev_idx = current_idx_in_list - 1
    except ValueError:
        insertion_point = bisect.bisect_left(frame_list, current_frame_idx)
        prev_idx = insertion_point - 1

    if prev_idx >= 0:
        return frame_list[prev_idx]
    
    return None

def get_next_frame_in_list(frame_list:List[int], current_frame_idx:int) -> Optional[int]:
    try:
        current_idx_in_list = frame_list.index(current_frame_idx)
        next_idx = current_idx_in_list + 1
    except ValueError:
        insertion_point = bisect.bisect_right(frame_list, current_frame_idx)
        next_idx = insertion_point

    if next_idx < len(frame_list):
        return frame_list[next_idx]
    
    return None