import os
import numpy as np
import pandas as pd
import scipy.io as sio

import json
from typing import Dict, Tuple, Literal

from core.io import prediction_to_csv, Frame_Exporter_Threaded
from utils.track import interpolate_track_all
from utils.helper import get_instance_count_per_frame, array_to_iterable_runs
from utils.dataclass import Exporter_Augments, Loaded_DLC_Data


class Annot_Exporter:
    def __init__(
            self,
            annot_array:np.ndarray,
            video_name:str,
            behav_map:Dict[str, Tuple[str, str]],
            idx_to_cat:Dict[int, str],
            roi: Tuple[int, int, int, int] | np.ndarray
            ):

        self.annot_array = annot_array
        self.behav_map = behav_map
        self.total_frames = len(annot_array)
        self.idx_to_cat = idx_to_cat
        self.video_name = video_name
        self.roi = roi

    def to_txt(self, file_path, segments):
        content = "Caltech Behavior Annotator - Annotation File\n\n"
        content += "Configuration file:\n"
        for category, (key, _) in self.behav_map.items():
            content += f"{category}\t{key}\n"
        content += "\n"
        content += "S1:\tstart\tend\ttype\n"
        content += "-----------------------------\n"
        for category, start, end in segments:
            content += f"\t{start}\t{end}\t{category}\n"

        with open(file_path, 'w') as f:
            f.write(content)

        json_path = file_path.replace(".txt", ".json")
        with open(json_path, 'w') as f:
            json.dump({"behav_map": self.behav_map}, f, indent=2)

    def to_mat(self, file_path):
        behavior_struct = self.annot_array.copy()
        annotation_struct = {
            "streamID": 1,
            "annotation": behavior_struct.reshape(-1, 1),
            "behaviors": self.cat_to_idx
        }
        mat_to_save = {"annotation": annotation_struct}
        sio.savemat(file_path, mat_to_save)

        json_path = file_path.replace(".mat", ".json")
        with open(json_path, 'w') as f:
            json.dump({"behav_map": self.behav_map}, f, indent=2)

    def to_onehot(self, file_path, dlc_data:Loaded_DLC_Data, fps:int=10):
        behavior_list = [b for b in self.idx_to_cat.values()]
        onehot_array = np.zeros((self.total_frames, len(behavior_list)))
        onehot_array[np.arange(self.total_frames), self.annot_array] = 1

        df_annot = pd.DataFrame(onehot_array, columns=behavior_list)
        if "other" in behavior_list:
            behavior_cols = [col for col in df_annot.columns if col != "other"]
            new_column_order = behavior_cols + ["other"]
            df_annot = df_annot[new_column_order]

        df_annot.insert(0, "time", np.arange(self.total_frames) / fps)

        df_annot.to_csv(file_path, sep=',', index=False, float_format='%.6f')
        pred_path = file_path.replace(".csv", "_pred.csv")
        prediction_to_csv(dlc_data, dlc_data.pred_data_array, pred_path, keep_conf=True, no_scorer_row=True)

    def to_asoid_train(self, folder_path, dlc_data:Loaded_DLC_Data, min_duration:int=10, max_gap:int=3, fps:int=10):
        os.makedirs(os.path.join(folder_path, "behav"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "pred"), exist_ok=True)

        pred_data_array = dlc_data.pred_data_array.copy()
        I = pred_data_array.shape[1]
        for inst_idx in range(I):
            pred_data_array = interpolate_track_all(pred_data_array, inst_idx, max_gap)

        instance_array = get_instance_count_per_frame(pred_data_array)

        frames_to_use = np.zeros_like(instance_array, dtype=bool)
        truncated_arrays = []

        for start, end, value in array_to_iterable_runs(instance_array == I):
            if not value:
                continue
            if end - start + 1 < min_duration:
                continue

            frames_to_use[start:end+1] = True
            truncated_arrays.append(self.annot_array[start:end+1])

        frame_list = np.where(frames_to_use)[0].tolist()

        truncated_behavior_array = np.concatenate(truncated_arrays)
        truncated_pred_array = pred_data_array[frames_to_use]

        behavior_list = [b for b in self.idx_to_cat.values()]

        truncated_one_hot = np.zeros((truncated_behavior_array.size, len(behavior_list)), dtype=int)
        truncated_one_hot[np.arange(truncated_behavior_array.size), truncated_behavior_array] = 1

        df_annot = pd.DataFrame(truncated_one_hot, columns=behavior_list)
        if "other" in behavior_list:
            behavior_cols = [col for col in df_annot.columns if col != "other"]
            new_column_order = behavior_cols + ["other"]
            df_annot = df_annot[new_column_order]

        df_annot.insert(0, "time", np.arange(len(frame_list)) / fps)

        file_path = os.path.join(folder_path, "behav", f"{self.video_name}_ASOiD_train.csv")
        pred_path = os.path.join(folder_path, "pred", f"{self.video_name}_ASOiD_train.csv")
        self._csv_exporter_worker(
            df_annot=df_annot,
            annot_path=file_path,
            truncated_pred_array=truncated_pred_array,
            dlc_data=dlc_data,
            pred_path=pred_path,
            frame_list=frame_list,
            mode="train")

    def to_asoid_infer(self, file_path, dlc_data:Loaded_DLC_Data, min_duration:int=10, max_gap:int=3, fps:int=10):
        pred_data_array = dlc_data.pred_data_array.copy()
        I = pred_data_array.shape[1]
        for inst_idx in range(I):
            pred_data_array = interpolate_track_all(pred_data_array, inst_idx, max_gap)

        instance_array = get_instance_count_per_frame(pred_data_array)
        
        frames_to_use = np.zeros_like(instance_array, dtype=bool)
        truncated_arrays = []

        for start, end, value in array_to_iterable_runs(instance_array == I):
            if not value:
                continue
            if end - start + 1 < min_duration:
                continue

            frames_to_use[start:end+1] = True
            truncated_arrays.append(self.annot_array[start:end+1])

        frame_list = np.where(frames_to_use)[0].tolist()
        truncated_pred_array = pred_data_array[frames_to_use]

        pred_path = file_path.replace(".csv", "_pred.csv")
        self._csv_exporter_worker(
            df_annot=None,
            annot_path=file_path,
            truncated_pred_array=truncated_pred_array,
            dlc_data=dlc_data,
            pred_path=pred_path,
            frame_list=frame_list,
            mode="infer")

    def to_asoid_refine(self, folder_path, dlc_data:Loaded_DLC_Data, max_gap=10, min_length=600, seg_needed=10):
        os.makedirs(os.path.join(folder_path, "refine"), exist_ok=True)

        pred_data_array = dlc_data.pred_data_array.copy()
        I = pred_data_array.shape[1]
        for inst_idx in range(I):
            pred_data_array = interpolate_track_all(pred_data_array, inst_idx, max_gap)

        instance_array = get_instance_count_per_frame(pred_data_array)

        segs = []
        for start, end, value in array_to_iterable_runs(instance_array == I):
            if len(segs) >= seg_needed:
                break
            if not value:
                continue
            if end - start + 1 < min_length:
                continue
            
            segs.append((start, end+1))

        for seg in segs:
            frame_list = list(range(*seg))

            truncated_pred_array = pred_data_array[frame_list]
            pred_path = os.path.join(folder_path, "refine", f"{self.video_name}_frame{seg[0]}-{seg[1]}_ASOiD_refine.csv")
            self._csv_exporter_worker(
                df_annot=None,
                annot_path=pred_path,
                truncated_pred_array=truncated_pred_array,
                dlc_data=dlc_data,
                pred_path=pred_path,
                frame_list=frame_list,
                mode="refine")

    def _csv_exporter_worker(self, df_annot, annot_path, truncated_pred_array, dlc_data, pred_path, frame_list, mode:Literal["train", "infer", "refine"]="train"):
        match mode:
            case "train": df_annot.to_csv(annot_path, sep=',', index=False, float_format='%.6f')
            case "infer":
                json_path = annot_path.replace(".csv", ".json")
                with open(json_path, 'w') as f:
                    json.dump({
                        "used_frames": frame_list,
                        "total_frames": self.total_frames,
                        "total_exported_frames": len(frame_list),
                        "behav_map": self.behav_map,
                    }, f, indent=2)
            case "refine": 
                output_folder = os.path.basename(pred_path)
                fe = Frame_Exporter_Threaded(self.video_file, output_folder, frame_list)
                ea = Exporter_Augments(crop_coord=self.roi)
                fe.extract_frames_into_video(ea, video_name=f"{self.video_name}_frame{frame_list[0]}-{frame_list[-1]}.mp4")

        prediction_to_csv(dlc_data, truncated_pred_array, pred_path, keep_conf=True, no_scorer_row=mode!="train")