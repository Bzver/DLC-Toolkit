import os
import glob
import numpy as np
from typing import List

from core.io import Prediction_Loader
from utils.dataclass import Loaded_DLC_Data
from utils.logger import logger


class Pred_Manager_3D:
    def __init__(self, dlc_data: Loaded_DLC_Data, video_folders:List[str|None], total_frames:int):
        self.video_folders = video_folders
        self.total_frames = total_frames
        self.dlc_data = dlc_data

        self.comb_data_array = np.full(
            (total_frames, len(video_folders), dlc_data.instance_count, dlc_data.num_keypoint*3)
            , np.nan, dtype=np.float32)
        self.pred_files = [None] * len(video_folders)

        self._load_predictions()

    def _load_predictions(self):
        for i, vf in enumerate(self.video_folders):
            if not vf:
                continue
            h5_files = glob.glob(os.path.join(vf, "*.h5"))
            if not h5_files:
                logger.warning(f"[PM3D] No prediction found in {vf}.")
                continue
            h5_files.sort()
            h5_file = h5_files[-1]
            self.pred_files[i] = h5_file

            self._load_pred_h5(h5_file, i)

    def _load_pred_h5(self, h5_file, cam_idx):
        pl = Prediction_Loader(self.dlc_data.dlc_config_filepath, h5_file)

        temp_dlc_data = pl.load_data()
        if not temp_dlc_data:
            return

        temp_pred_len = temp_dlc_data.pred_data_array.shape[0]
        if temp_pred_len > self.total_frames:
            logger.warning("[PM3D] Reloaded prediction has more frames than total frames. Truncating.")
            self.comb_data_array[:, cam_idx, :, :] = temp_dlc_data.pred_data_array[:self.total_frames, :, :]
        else:
            self.comb_data_array[:temp_pred_len, cam_idx, :, :] = temp_dlc_data.pred_data_array[:]