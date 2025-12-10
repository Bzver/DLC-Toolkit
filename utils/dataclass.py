from dataclasses import dataclass, asdict, replace

from typing import List, Optional, Literal, Dict, Callable, Tuple
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

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.pred_data_array is not None:
            d["pred_data_array"] = self.pred_data_array
        return d

    @classmethod
    def from_dict(cls, data) -> "Loaded_DLC_Data":
        pred_data_array = data.get("pred_data_array")
        if pred_data_array is not None:
            data["pred_data_array"] = pred_data_array.copy()
        return cls(**data)

@dataclass
class Export_Settings:
    video_filepath: str
    video_name: str
    save_path: str
    export_mode: Literal["Append", "Merge", "CSV"]

    def copy(self, **kwargs):
        return replace(self, **kwargs)

@dataclass
class Swap_Calculation_Config:
    show_progress: bool
    start_frame: int
    frame_count_min: int
    frame_count_max: int
    until_next_error: bool

@dataclass
class Plot_Config:
    plot_opacity: float
    point_size: float
    hide_text_labels: bool
    edit_mode: bool

    plot_labeled: bool
    plot_pred: bool
    navigate_labeled: bool

    confidence_cutoff: float
    auto_snapping: bool
    navigate_roi: bool
    
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data) -> "Plot_Config":
        return cls(**data)

@dataclass
class Plotter_Callbacks:
    keypoint_coords_callback: Callable[[int, int, float, float], None]  # instance_id, keypoint_id, new_x, new_y
    keypoint_object_callback: Callable[[object], None]
    box_coords_callback: Callable[[int, float, float], None]
    box_object_callback: Callable[[object], None]

@dataclass
class Nav_Callback:
    change_frame_callback: Callable[[int], None]
    nav_prev_callback: Callable[[], None]
    nav_next_callback: Callable[[], None]

@dataclass
class Blob_Config:
    sample_frame_count: int
    threshold: int
    double_blob_area_threshold: int
    min_blob_area: int
    bg_removal_method: str
    blob_type: str
    background_frames: Dict[str, NDArray]
    roi: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data) -> "Blob_Config":
        return cls(**data)