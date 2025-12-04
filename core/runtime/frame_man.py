from typing import Dict, List, Callable, Any, Set
from collections import defaultdict

from utils.logger import logger

class Frame_Manager:
    PREDEFINED = {
        "marked":      {"display_name": "Pending Frames",       "color_hex": "#E28F13", "group": "fview"},
        "rejected":    {"display_name": "Rejected Frames",      "color_hex": "#F749C6", "group": "fview"},
        "approved":    {"display_name": "Approved Frames",      "color_hex": "#68b3ff", "group": "fview"},
        "refined":     {"display_name": "Refined Frames",       "color_hex": "#009979", "group": "fview"},
        "labeled":      {"display_name": "Labeled (DLC)",       "color_hex": "#1F32D7", "group": "fview"},

        "animal_0":    {"display_name": "0 Animals Frames",     "color_hex": "#A9A9A9", "group": "counting"},
        "animal_1":    {"display_name": "1 Animal Frames",      "color_hex": "#33FF00", "group": "counting"},
        "animal_n":    {"display_name": "2+ Animals Frames",    "color_hex": "#d3c54a", "group": "counting"},
        "blob_merged": {"display_name": "Merged Blob Frames",   "color_hex": "#ff00bf", "group": "counting"},

        "roi_change":  {"display_name": "Instance Change Frames",   "color_hex": "#FF1100", "group": "flabel"},
        "outlier":     {"display_name": "Outlier Frames",           "color_hex": "#75541F", "group": "flabel"},
    }

    def __init__(self, refresh_callback:Callable[[], None]):
        self.frames: Dict[str, Set[int]] = defaultdict(set)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.refresh_callback = refresh_callback

        for cat, meta in self.PREDEFINED.items():
            if cat not in self.metadata:
                self.metadata[cat] = meta.copy()
                self.frames[cat] = set()

        self.fview_cats = [cat for cat, meta in self.PREDEFINED.items() if meta.get("group") == "fview"]
        self.count_cats = [cat for cat, meta in self.PREDEFINED.items() if meta.get("group") == "counting"]
        self.flabel_cats = [cat for cat, meta in self.PREDEFINED.items() if meta.get("group") == "flabel"]

    def add_frame(self, category:str, frame_idx:int):
        if frame_idx not in self.frames[category]:
            self.frames[category].add(frame_idx)
            self.refresh_callback()

    def add_frames(self, category:str, frame_indices:List[int], no_refresh:bool=False):
        new_frames = set(frame_indices) - self.frames[category]
        if new_frames:
            self.frames[category] |= new_frames
            if not no_refresh:
                self.refresh_callback()

    def remove_frame(self, category:str, frame_idx:int, no_refresh:bool=False):
        if frame_idx in self.frames[category]:
            self.frames[category].discard(frame_idx)
            if not no_refresh:
                self.refresh_callback()

    def remove_frames(self, category:str, frame_indices:List[int], no_refresh:bool=False):
        current_frames = self.frames[category]
        to_remove = set(frame_indices) & current_frames
        if to_remove:
            current_frames -= to_remove
            if not no_refresh:
                self.refresh_callback()

    def move_frame(self, new_category:str, old_category:str, frame_idx:int):
        if old_category == new_category:
            return
        if frame_idx in self.frames[old_category]:
            self.frames[new_category].add(frame_idx)
            self.frames[old_category].discard(frame_idx)
            self.refresh_callback()

    def move_category(self,  dest_category:str, source_category:str) -> None:
        if source_category == dest_category:
            return
        source_frames = self.frames[source_category]
        if not source_frames:
            return
        self.frames[dest_category] |= source_frames
        source_frames.clear()
        self.refresh_callback()

    def toggle_frame(self, category:str, frame_idx:int):
        if frame_idx in self.frames[category]:
            self.remove_frame(category, frame_idx)
        else:
            self.add_frame(category, frame_idx)

    def clear_category(self, category:str, no_refresh:bool=False):
        if self.frames[category]:
            if category in ["rejected", "approved"]: # They get a second chance
                self.move_category("marked", category)
            else:
                self.frames[category].clear()
            if not no_refresh:
                self.refresh_callback()

    def clear_group(self, group:str):
        for cat, meta in self.metadata.items():
            if meta.get("group") == group:
                self.clear_category(cat)
    
    def clear_all(self):
        for cat in self.frames.keys():
            self.frames[cat].clear()
        self.refresh_callback()

    def reset_fm(self):
        if any(self.frames.values()):
            self.frames = {cat: set() for cat in self.PREDEFINED.keys()}
            self.metadata = {cat: meta.copy() for cat, meta in self.PREDEFINED.items()}
            self.refresh_callback()

    ##########################################################################################################################################

    def get_frames(self, category:str) -> List[int]:
        return sorted(self.frames[category])

    def get_len(self, category:str) -> int:
        return len(self.frames[category])

    def get_group_categories(self, group:str) -> List[str]:
        return [cat for cat, meta in self.PREDEFINED.items() if meta.get("group") == group]

    def has_frame(self, category:str, frame_idx:int) -> bool:
        return frame_idx in self.frames[category]

    def all_categories(self) -> List[str]:
        return list(self.frames.keys())
    
    def all_populated_categories(self) -> List[str]:
        return sorted(cat for cat, frames in self.frames.items() if frames)
    
    def frames_in_any(self, categories:List[str]) -> List[int]:
        union = set().union(*(self.frames[c] for c in categories if c in self.frames))
        return sorted(union)

    def frames_in_all(self, categories:List[str]) -> List[int]:
        intersection = set.intersection(*(self.frames[c] for c in categories if c in self.frames))
        return sorted(intersection)

    def frames_not_in(self, category:str, exclude_categories:List[str]) -> List[int]:
        base = self.frames[category]
        exclude = set().union(*(self.frames[c] for c in exclude_categories if c in self.frames))
        return sorted(base - exclude)

    ##########################################################################################################################################

    def set_metadata(self, category:str, **kwargs):
        """     
        Args:
            category (str):The name of the annotation category (e.g., "jumping", "refined").
            **kwargs: Arbitrary metadata key-value pairs. Common keys include:
                - ``display_name`` (str): Human-readable name for UI (e.g., "Jumping Behavior").
                - ``color`` (str): Hex color code (e.g., "#FF1493") for visualization/navigation.
                - ``group`` (str): The group of this category to determine the slider scheme.
        """
        self.metadata.setdefault(category, {}).update(kwargs)
        self.refresh_callback()

    def _get_metadata(self, category:str, key:str, default=None):
        return self.metadata.get(category, {}).get(key, default)

    def get_color(self, category:str) -> str:
        return self._get_metadata(category, "color_hex", "#888888")

    def get_display_name(self, category:str) -> str:
        return self._get_metadata(category, "display_name", category.replace("_", " ").title())

    ##########################################################################################################################################

    def to_dict(self) -> Dict[str, Any]:
        all_cat = set(self.metadata.keys())
        return {
            "frames": {cat: sorted(self.frames[cat]) for cat in all_cat},
            "metadata": self.metadata.copy()
        }

    @classmethod
    def from_dict(cls, data:Dict[str, Any], refresh_callback:Callable[[], None] = None) -> "Frame_Manager":
        store = cls(refresh_callback or (lambda: None))
        metadata = data.get("metadata", {})
        frames_data = data.get("frames", {})

        for cat in metadata:
            store.metadata[cat] = metadata[cat].copy()
            store.frames[cat] = set(frames_data.get(cat, []))
            
        return store
    
    ##########################################################################################################################################

    def print_all_category_len(self):
        all_cat = self.all_categories()
        logger.info("----------------------------------------")
        for cat in all_cat:
            logger.info(f"[FRMAN] Category: {cat} - {self.get_len(cat)}")