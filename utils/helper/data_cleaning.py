import numpy as np
from typing import Union, List, Tuple, Dict, Iterable
from utils.logger import logger


def clean_inconsistent_nans(pred_data_array:np.ndarray):
    logger.info("Cleaning up NaN keypoints that somehow has confidence value...")
    nan_mask = np.isnan(pred_data_array)
    x_is_nan = nan_mask[:, :, 0::3]
    y_is_nan = nan_mask[:, :, 1::3]
    keypoints_to_fully_nan = x_is_nan | y_is_nan
    full_nan_sweep_mask = np.repeat(keypoints_to_fully_nan, 3, axis=-1)
    pred_data_array[full_nan_sweep_mask] = np.nan
    logger.info("NaN keypoint confidence cleaned.")
    return pred_data_array

def calculate_blob_inference_intervals(blob_array:np.ndarray, intervals:Dict[str, int], existing_frames:List[int]=[]) -> List[int]:
    existing_set = set(existing_frames)
    total_frames = blob_array.shape[0]

    inference_list = []

    animal_count_array, merge_array = clean_blob_array_for_inference(blob_array)

    last_inferenced_frame = -max(intervals.values())
    for frame_idx in range(total_frames):   
        animal_count = animal_count_array[frame_idx]
        merge_status = merge_array[frame_idx]

        if frame_idx in existing_set:
            last_inferenced_frame = frame_idx
            continue

        if animal_count == 0:
            current_interval = intervals["interval_0_animal"]
        elif animal_count == 1:
            current_interval = intervals["interval_1_animal"]
        elif merge_status == 1:
            current_interval = intervals["interval_merged"]
        else: # animal_count >= 2 and not merged
            current_interval = intervals["interval_n_animals"]
        
        if frame_idx - last_inferenced_frame >= current_interval:
            inference_list.append(frame_idx)
            last_inferenced_frame = frame_idx
    
    return inference_list

def clean_blob_array_for_inference(blob_array:np.ndarray, buffer_size:int=5) -> Tuple[np.ndarray, np.ndarray]:
    total_frames = blob_array.shape[0]

    animal_count_array, merge_array = blob_array[:, 0].copy(), blob_array[:, 1].copy()
    animal_count_array[animal_count_array>2] = 2

    for start, end, value in array_to_iterable_runs(merge_array): # merge array has only 1 and 0
        if end == start:
            try:
                merge_array[start] = merge_array[start - 1]
            except IndexError:
                pass
            finally:
                continue
        if value == 1:
            merge_array[max(0,start-buffer_size):end+buffer_size+1] = value

    merged_frames = set(np.where(merge_array==1)[0])

    two_indices = set()
    for start, end, value in array_to_iterable_runs(animal_count_array):
        if end == start:
            try:
                animal_count_array[start] = animal_count_array[start - 1]
            except IndexError:
                pass
            finally:
                continue
        if value == 2:
            two_indices.update(range(max(0,start-buffer_size), min(total_frames,end+buffer_size+1)))
        if value == 1:
            animal_count_array[max(0,start):min(total_frames,end+1)] = value

    two_indices = list(sorted(two_indices|merged_frames))
    animal_count_array[two_indices] = 2

    return animal_count_array, merge_array

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

def array_to_iterable_runs(arr:np.ndarray) -> Iterable[Tuple[int, int, int]]:
    if len(arr) == 0:
        return zip([], [], [])
    change_points = np.where(arr[1:] != arr[:-1])[0] + 1 
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points - 1, [len(arr) - 1]))
    values = arr[starts]
    return zip(starts, ends, values)