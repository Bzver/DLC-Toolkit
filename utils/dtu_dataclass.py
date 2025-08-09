from dataclasses import dataclass

from typing import List, Optional
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
    
    prediction_filepath: Optional[str]
    pred_data_array: Optional[NDArray]
    pred_frame_count: Optional[int]

@dataclass
class Export_Settings:
    video_filepath: str
    video_name: str
    save_path: str
    export_mode: str # "Append" or "Merge" or "CSV"
    
@dataclass
class Session_3D_Plotter:
    base_folder: str
    calibration_filepath: str
    dlc_config_filepath: str
    pred_data_array: NDArray
    current_frame_idx: int
    confidence_cutoff: float
    deviance_threshold: int
    roi_frame_list: List[Optional[int]]
    failed_frame_list: List[Optional[int]]
    sus_frame_list: List[Optional[int]]
    swap_detection_score_array: NDArray