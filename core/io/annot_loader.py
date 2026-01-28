import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

from utils.logger import logger

def load_annotation(file_path:str, json_path:Optional[str]=None, config_only:bool=False) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, List[int]]]:
    annot_raw = defaultdict(list)
    behav_map_txt = {}

    if not os.path.exists(file_path):
        logger.error(f"[ANTLOAD] Error: Annotation file not found at {file_path}")
        return {}, {}

    with open(file_path, 'r') as f:
        found_start = False
        config_start = False
        for line in f:
            line = line.strip()

            if "Configuration file:" in line:
                config_start = True
                continue
            if "start\tend\ttype" in line:
                config_start = False
                found_start = True
                continue
            if found_start and line.startswith('---'):
                continue

            if config_start and line:
                parts = line.split()
                if len(parts) == 2:
                    behav_map_txt[parts[0]] = parts[1]
                else:
                    logger.warning(f"[ANTLOAD] Skipping malformed config line in {file_path}: {line}")

            if config_only:
                continue

            if found_start and line:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        start_frame = int(parts[0])
                        end_frame = int(parts[1])
                        type_name = parts[2]
                        annot_raw[type_name].append((start_frame, end_frame))
                    except ValueError:
                        logger.warning(f"[ANTLOAD] Skipping malformed data line in {file_path}: {line}")
                else:
                    logger.warning(f"[ANTLOAD] Skipping malformed data line in {file_path}: {line}")

    frame_dict = {}
    if not config_only:
        for type_name, ranges in annot_raw.items():
            all_frames = set()
            for start, end in ranges:
                all_frames.update(range(start, end + 1))
            frame_dict[type_name] = sorted(all_frames)

    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
                behav_map = meta.get("behav_map", {})
                return behav_map, frame_dict
        except Exception as e:
            logger.warning(f"[ANTLOAD] Failed to load JSON metadata {json_path}: {e}. Falling back to .txt config.")

    FALLBACK_COLORS = (
        "#B8B8B8", "#00BCD4", "#FF9800", "#4CAF50", "#FFEB3B",
        "#FFB3AD", "#795548", "#3F51B5", "#9BB0BB", "#009688",
        "#FF5722", "#CDDC39", "#FFC107", "#8BC34A", "#673AB7"
    )

    behav_map = {}
    for i, (name, key) in enumerate(behav_map_txt.items()):
        color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        behav_map[name] = (key, color)

    for name in frame_dict:
        if name not in behav_map:
            key = name[0] if name else "x"
            color = FALLBACK_COLORS[len(behav_map) % len(FALLBACK_COLORS)]
            behav_map[name] = (key, color)

    return behav_map, frame_dict

def load_onehot_csv(
    file_path: str, 
    json_path: str,
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, List[int]]]:
    if not os.path.exists(file_path):
        logger.error(f"[ANTLOAD] Error: Annotation file not found at {file_path}")
        return {}, {}

    if not os.path.exists(json_path):
        logger.error(f"[ANTLOAD] Error: Metadata file not found at {json_path}")
        return {}, {}

    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)
        
        used_frames = meta["used_frames"]
        
        behav_map = meta.get("behav_map", {})
        total_frames = meta.get("total_frames")

    except Exception as e:
        logger.error(f"[ANTLOAD] Failed to parse JSON metadata {json_path}: {e}")
        return {}, {}

    try:
        df = pd.read_csv(file_path, sep=',')
    except Exception as e:
        logger.error(f"[ANTLOAD] Failed to read CSV {file_path}: {e}")
        return {}, {}

    behavior_cols = [col for col in df.columns if col != "time"]
    for behav in behavior_cols:
        if behav not in behav_map:
            raise RuntimeError(f"Behavior '{behav}' in CSV not found in behav_map from metadata!")

    frame_dict = {}

    if "other" in behav_map:
        frame_dict["other"] = list(range(total_frames))
    else:
        frame_dict["other"] = []

    used_frames = np.array(used_frames, dtype=int)

    for behav in behavior_cols:
        if behav == "other":
            continue
        active_series = df[behav]
        active_mask = (active_series.values == 1)
        active_mask = np.concatenate([[False], active_mask])
    
        active_original_frames = used_frames[active_mask].tolist()
        frame_dict[behav] = sorted(active_original_frames)

        if "other" in frame_dict:
            frame_dict["other"] = sorted(set(frame_dict["other"]) - set(active_original_frames))

    if "other" in frame_dict and not frame_dict["other"]:
        del frame_dict["other"]

    return behav_map, frame_dict