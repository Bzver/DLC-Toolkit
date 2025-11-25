import os
from collections import defaultdict
from typing import Dict, Tuple, List

def load_annotation(file_path: str) -> Tuple[Dict[str, str], Dict[str, List[Tuple[int, int]]]]:
    annot_raw = defaultdict(list)
    annot_processed = {}
    behav_map = {}

    if not os.path.exists(file_path):
        print(f"Error: Annotation file not found at {file_path}")
        return annot_processed

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
                    try:
                        behav_map[parts[0]] = parts[1]
                    except ValueError:
                        print(f"Skipping malformed data line in {file_path}: {line}")
                else:
                    print(f"Skipping malformed data line (incorrect number of parts) in {file_path}: {line}")

            if found_start and line:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        start_frame = int(parts[0])
                        end_frame = int(parts[1])
                        type_name = parts[2]
                        annot_raw[type_name].append((start_frame, end_frame))
                    except ValueError:
                        print(f"Skipping malformed data line in {file_path}: {line}")
                else:
                    print(f"Skipping malformed data line (incorrect number of parts) in {file_path}: {line}")

    for type_name, ranges in annot_raw.items():
        all_frames = set()
        for start, end in ranges:
            all_frames.update(range(start, end + 1))
        annot_processed[type_name] = sorted(list(all_frames))

    return behav_map, annot_processed