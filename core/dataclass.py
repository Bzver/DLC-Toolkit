from dataclasses import dataclass

from typing import List, Optional, Literal, Dict, Callable
from numpy.typing import NDArray

@dataclass
class Loaded_DLC_Data:
    dlc_config_filepath: str
    scorer: str
    multi_animal: bool
    keypoints: List[str]
    skeleton: List[List[str]]
    individuals: Optional[List[str]]
    instance_count: int
    num_keypoint: int
    keypoint_to_idx: Dict[str, int]
    
    prediction_filepath: Optional[str]
    pred_data_array: Optional[NDArray]
    pred_frame_count: Optional[int]

@dataclass
class Export_Settings:
    video_filepath: str
    video_name: str
    save_path: str
    export_mode: Literal["Append", "Merge", "CSV"]

@dataclass
class Swap_Calculation_Config:
    show_progress: bool
    start_frame: int
    frame_count_min: int
    frame_count_max: int
    until_next_error: bool

@dataclass
class Plot_Config:
    plot_opacity:float
    point_size:float
    confidence_cutoff: float
    hide_text_labels:bool
    edit_mode:bool

@dataclass
class Labeler_Plotter_Callbacks:
    keypoint_coords_callback: Callable[[int, int, float, float], None]  # instance_id, keypoint_id, new_x, new_y
    keypoint_object_callback: Callable[[object], None]
    box_coords_callback: Callable[[int, float, float], None]
    box_object_callback: Callable[[object], None]