import numpy as np
from typing import Union, List, Tuple, Iterable


def clean_inconsistent_nans(pred_data_array:np.ndarray):
    nan_mask = np.isnan(pred_data_array)
    x_is_nan = nan_mask[:, :, 0::3]
    y_is_nan = nan_mask[:, :, 1::3]
    keypoints_to_fully_nan = x_is_nan | y_is_nan
    full_nan_sweep_mask = np.repeat(keypoints_to_fully_nan, 3, axis=-1)
    pred_data_array[full_nan_sweep_mask] = np.nan
    return pred_data_array

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

def clean_outside_roi_pred(pred_data_array:np.ndarray, roi:Tuple[int, int, int, int]|np.ndarray) -> np.ndarray:
    F, I, D = pred_data_array.shape
    K = D // 3

    poses = pred_data_array.reshape(F, I, K, 3)
    xs = poses[..., 0]
    ys = poses[..., 1]
    
    x1, y1, x2, y2 = roi

    xs[xs < x1] = np.nan
    xs[xs >= x2] = np.nan
    ys[ys < y1] = np.nan
    ys[ys >= y2] = np.nan

    return clean_inconsistent_nans(poses.reshape(F, I, D))

def clean_pred_in_mask(pred_data_array:np.ndarray, background_mask:np.ndarray) -> np.ndarray:
    F, I, D = pred_data_array.shape
    K = D // 3

    poses = pred_data_array.reshape(F, I, K, 3)
    
    poses_no_nan = np.nan_to_num(poses, nan=-1.0)
    xs = poses_no_nan[..., 0].astype(int)
    ys = poses_no_nan[..., 1].astype(int)
    
    mask_array = background_mask[..., 0]
    mask_bool = mask_array != 0
    
    H, W = mask_bool.shape

    xs[xs < 0] = 0
    xs[xs >= W] = W - 1
    ys[ys < 0] = 0
    ys[ys >= H] = H - 1

    in_mask_region = mask_bool[ys, xs]
    poses[in_mask_region, :] = np.nan

    return poses.reshape(F, I, D)

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


###################################################################################################################


import math
import numpy as np
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Tuple


def custom_round(value):
    return math.floor(value + 0.5)

def split_by_neutral(seq: np.ndarray, neutral_ids: List[int]) -> List[Tuple[int, int]]:
    is_neutral = np.isin(seq, neutral_ids)
    boundaries = np.where(np.diff(is_neutral.astype(int)) != 0)[0] + 1
    segments = []
    start = 0
    for b in boundaries:
        if not is_neutral[start]:
            segments.append((start, b))
        start = b
    if start < len(seq) and not is_neutral[start]:
        segments.append((start, len(seq)))
    return segments

def process_segment(
    seq:np.ndarray,
    eps_frames: int = 5,
    min_cluster_size: int = 3,
    ) -> np.ndarray:

    n = len(seq)
    if n < 3:
        return seq
    
    wseq = seq.copy()
    
    counts = np.bincount(wseq)
    rare_vals = np.where(counts < 3)[0]
    
    for val in rare_vals:
        mask = wseq == val
        indices = np.where(mask)[0]
        for idx in indices:
            if idx > 0:
                wseq[idx] = wseq[idx-1]
            elif idx < n-1:
                wseq[idx] = wseq[idx+1]
                
    clusters = []
    unique_vals = np.unique(wseq)

    for val in unique_vals:
        indices = np.where(wseq == val)[0].reshape(-1, 1)
        clustering = DBSCAN(eps=eps_frames, min_samples=1).fit(indices)
        labels = clustering.labels_
        
        for lbl in set(labels):
            cluster_idx = indices[labels == lbl].flatten()
            size = len(cluster_idx)
            com = np.mean(cluster_idx)

            start = int(custom_round(com)) - size // 2
            end = int(custom_round(com)) + size // 2 - (1 if size % 2 == 0 else 0)
            if end >= n:
                delta = end + 1 - n
                start -= delta
                end -= delta
            if start < 0:
                end += start
                start = 0

            clusters.append({
                'val': val,
                'com': com,
                'size': size,
                'start': start,
                'end': end
            })
        
    clusters.sort(key=lambda x: x['size'])
    
    assigned = -np.ones(n, dtype=np.int8)
    reserved = []
    
    for c in clusters:
        if c['size'] < min_cluster_size:
            reserved.append(c)
            continue
        s, e = c['start'], c['end']
        if np.all(assigned[s:e+1] == -1):
            assigned[s:e+1] = c['val']
        elif assigned[s] != -1 and assigned[e] != -1:
            reserved.append(c)
        elif assigned[s] == -1:
            delta = e - np.where(assigned[s:e+1] != -1)[0][0] + 1
            if np.all(assigned[s-delta:e+1-delta] == -1) and s-delta > 0:
                assigned[s-delta:e+1-delta] = c['val']
            else:
                reserved.append(c)
        elif assigned[e] == -1:
            delta = np.where(assigned[s:e+1] != -1)[0][0] + 1
            if np.all(assigned[s+delta:e+1+delta] == -1) and e+delta < n:
                assigned[s+delta:e+1+delta] = c['val']
            else:
                reserved.append(c)
        else:
            reserved.append(c)

    reserved.sort(key=lambda x: x['com']) 
    reserved_vals = []

    for rc in reserved:
        reserved_vals.extend([rc['val']]*rc['size'])

    assigned[assigned==-1] = reserved_vals

    for start, end, val in array_to_iterable_runs(assigned):
        if start > 0 and val == assigned[start-1]:
            continue
        if end - start + 1 < min_cluster_size:
            if start > 0:
                assigned[start:end+1] = assigned[start-1]
            else:
                assigned[start:end+1] = assigned[end+2]

    return assigned

def refine_bouts_parallel(
    seq: np.ndarray,
    neutral_ids: List[int],
    eps_frames: int = 5,
    min_cluster_size: int = 3,
    n_jobs: int = -1
) -> np.ndarray:

    segments = split_by_neutral(seq, neutral_ids)
    
    if not segments:
        return seq.copy()
    
    def process_one(seg_bounds):
        start, end = seg_bounds
        segment = seq[start:end]
        refined = process_segment(segment, eps_frames, min_cluster_size)
        return start, end, refined
    
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_one)(seg) for seg in tqdm(segments, total=len(segments), desc="Processing")
    )

    refined_seq = seq.copy()
    for start, end, refined_segment in results:
        refined_seq[start:end] = refined_segment
    
    return refined_seq