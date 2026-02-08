import bisect
from typing import Optional, Callable, List, Literal

from utils.logger import Loggerbox

def navigate_to_marked_frame(
        parent,
        frame_list:List[int],
        current_frame_idx:int,
        change_frame_callback:Callable[[int], None],
        direction:Literal["prev","next"],
        midway:bool=False,
        ):
    if not frame_list:
        Loggerbox.warning(parent, "No Marked Frames", "No marked frames to navigate.")
        return
    
    frame_list.sort()

    if direction == "prev":
        dest_frame_idx = get_prev_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No previous marked frame found."
    elif direction == "next":
        dest_frame_idx = get_next_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No next marked frame found."
    
    if dest_frame_idx is None:
        Loggerbox.warning(parent, "Navigation", no_frame_message)
        return

    if midway and abs(dest_frame_idx - current_frame_idx) > 1:
        change_frame_callback((dest_frame_idx + current_frame_idx) // 2)
    else:
        change_frame_callback(dest_frame_idx)

def get_prev_frame_in_list(frame_list:List[int], current_frame_idx:int) -> Optional[int]:
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