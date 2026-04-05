import os
import numpy as np
import random

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

from utils.logger import logger


class Crop_Dataset(Dataset):
    def __init__(self, crops, motion_ids, frame_indices, is_ir=False):
        self.crops = crops
        self.motion_ids = motion_ids
        self.frame_indices = frame_indices

        if is_ir:
            tr_nm_mean = [0.485, 0.485, 0.485]
            tr_nm_std = [0.229, 0.229, 0.229]
        else:
            tr_nm_mean = [0.485, 0.456, 0.406]
            tr_nm_std=[0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tr_nm_mean, std=tr_nm_std),
        ])

        self.frame_to_indices = {}
        for i, f_idx in enumerate(frame_indices):
            if f_idx not in self.frame_to_indices:
                self.frame_to_indices[f_idx] = []
            self.frame_to_indices[f_idx].append(i)
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self, idx):
        crop = self.crops[idx]
        if self.transform:
            crop = self.transform(crop)
        return crop, self.motion_ids[idx], self.frame_indices[idx], idx


class Cutout_Dataloader:
    def __init__(self, folder_path: str, seg_list: List[List[int]] = None):
        self.folder_path = folder_path
        self.seg_list = seg_list
        self.segment_index = self._build_segment_index()

        
        logger.info(f"[CLOAD] Found {len(self.segment_index)} segments in {folder_path}")
    
    def _build_segment_index(self) -> List[Dict]:
        index = []
        all_len = []
        for seg_idx, frame_list in enumerate(self.seg_list):
            if not frame_list:
                continue
            start_frame = frame_list[0]
            chunk_path = os.path.join(self.folder_path, f"chunk_{start_frame:08d}.npz")
            
            if os.path.exists(chunk_path):
                index.append({
                    'seg_idx': seg_idx,
                    'start_frame': start_frame,
                    'chunk_path': chunk_path,
                    'length': len(frame_list),
                    'frame_indices': frame_list
                })
                all_len.append(len(frame_list))
            else:
                logger.warning(f"[CLOAD] Chunk not found: {chunk_path}")

        self.min_segment_length = 10

        return index
    
    def load_segment(self, seg_idx: int) -> Tuple[np.ndarray, List[int], List[int]]:
        if seg_idx >= len(self.segment_index):
            raise IndexError(f"Segment {seg_idx} out of range (max: {len(self.segment_index)-1})")
        
        chunk_info = self.segment_index[seg_idx]
        chunk_path = chunk_info['chunk_path']
        
        with np.load(chunk_path, allow_pickle=True) as f:
            images = f['images']
            frame_indices = f['frame_indices'].tolist()

        n_frames = len(frame_indices)
        motion_ids = [mid for _ in range(n_frames) for mid in [0, 1]]
        
        return images, frame_indices, motion_ids
    
    def load_segment_samples(
        self, 
        seg_idx: int, 
        n_samples: int = 5
    ) -> Tuple[np.ndarray, List[int], List[int]]:

        all_images, all_frames, all_mids = self.load_segment(seg_idx)
        
        n_total = len(all_frames)
        
        if n_total <= n_samples:
            return all_images, all_frames, all_mids

        step = n_total / n_samples
        sample_indices = [int(i * step) for i in range(n_samples)]

        if sample_indices[-1] != n_total - 1:
            sample_indices[-1] = n_total - 1

        sampled_images = all_images[sample_indices]
        sampled_frames = [all_frames[i] for i in sample_indices]
        sampled_mids = [all_mids[i * 2] for i in sample_indices for _ in range(2)]  # Both mice per frame
        
        return sampled_images, sampled_frames, sampled_mids
    
    def load_all_segments_for_training(
        self, 
        train_seg_indices: List[int],
    ) -> List[Crop_Dataset]:
        datasets = []
        
        for seg_idx in train_seg_indices:
            images, frames, _ = self.load_segment(seg_idx)

            n_frames = len(frames)
            crops = []
            motion_ids = []
            frame_indices = []
            
            for i in range(n_frames):
                for mouse_idx in range(2):
                    crops.append(images[i, mouse_idx])
                    motion_ids.append(mouse_idx)
                    frame_indices.append(frames[i])
            
            ds = Crop_Dataset(crops, motion_ids, frame_indices)
            datasets.append(ds)
        
        logger.info(f"[CLOAD] Loaded {len(datasets)} segments for training")
        return datasets
    
    def select_training_segments(self, ratio: float = 0.1, use_random: bool = False) -> List[int]:
        available = list(range(len(self.segment_index)))
        available = [i for i in available if self.segment_index[i]['length'] >= self.min_segment_length]
        n = int(ratio * len(available))

        if use_random:
            return random.sample(available, n)
        else:
            sorted_segs = sorted(available, key=lambda i: self.segment_index[i]['start_frame'])
            step = len(sorted_segs) / n
            return [sorted_segs[int(i * step)] for i in range(n)]
    
    def select_validation_segments(self, exclude: List[int], ratio: float = 0.1) -> List[int]:
        available = [i for i in range(len(self.segment_index)) if i not in exclude and self.segment_index[i]['length'] >= self.min_segment_length]
        n = int(ratio * len(available))

        return random.sample(available, min(n, len(available)))

    def get_total_frames(self) -> int:
        return sum(seg['length'] for seg in self.segment_index)

    def get_segment_info(self, seg_idx: int) -> Dict:
        return self.segment_index[seg_idx].copy()