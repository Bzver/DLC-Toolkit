import os
import sys
import shutil
import time

from PySide6 import QtWidgets
from typing import List, Tuple, Optional

from utils.helper import calculate_blob_inference_intervals, log_print
from core.runtime import Data_Manager
from core.tool.inference import DLC_Inference

def batch_log(*args, **kwargs,):
    log_print(*args, with_time=True, **kwargs)

def inference_workspace_vid(
        workspace_file:str,
        dlc_config_path:Optional[str]=None,
        partial_infer:bool=False,
        partial_infer_indices:Optional[List[int]]=None,
        blob_based_infer:bool=False,
        infer_interval:Optional[Tuple[int,int,int,int]]=None,
        crop:bool=False,
        crop_region:Optional[Tuple[int,int,int,int]]=None,
        shuffle_idx:Optional[int]=None,
        batch_size:Optional[int]=None,
        detector_batch_size:Optional[int]=None,
    ):
    assert os.path.isfile(workspace_file), f"Workspace file not found: {workspace_file}"

    mmddhhmmss = time.strftime("%m%d%H%M%S", time.localtime(time.time()))

    file, ext = os.path.splitext(workspace_file)
    assert ext == ".pkl", f"Unsupported workspace extension: {ext}"

    workcopy = f"{file}_batchcop_{mmddhhmmss}{ext}"
    shutil.copy(workspace_file, workcopy)
    batch_log(f"Operating in a copy of {workspace_file}: {workcopy}")

    batch_log(f"Workspace in {workcopy} has been loaded.")

    dialog = QtWidgets.QDialog()
    dm = Data_Manager(init_vid_callback=_pseudo_callback, refresh_callback=_pseudo_callback, parent=dialog)
    
    dm._load_workspace(workcopy)
    
    if dlc_config_path is not None and os.path.isfile(dlc_config_path):
        dm.load_metadata_to_dm(dlc_config_path)
    else:
        assert dm.dlc_data is not None and dm.dlc_data.dlc_config_filepath is not None, "DLC configuration not found in workspace. Ensure the workspace includes a valid DLC project."
    assert dm.video_file is not None and os.path.isfile(dm.video_file), f"Video file missing or invalid: {dm.video_file}"

    if crop:
        assert crop_region is not None or dm.roi is not None, "Cropping enabled, but no crop region or ROI defined. Provide crop_region or ensure ROI is set in the workspace."
        if crop_region is None:
            try:
                x1, y1, x2, y2 = dm.roi
                crop_region = x1, y1, x2, y2
            except:
                raise RuntimeError("ROI in workspace is malformed, fail to translate the ROI in workspace.")

    inference_list = []
    if partial_infer:
        assert partial_infer_indices is not None, "partial_infer_indices must be provided when partial_infer is True"
        inference_list = partial_infer_indices
    elif blob_based_infer:
        assert infer_interval is not None, "infer_interval must be provided when blob_based_infer is True"
        batch_log(f"Blob-based inference selected with interval: {infer_interval}")
        if dm.blob_array is not None:
            intervals = {
                "interval_0_animal": infer_interval[0],
                "interval_1_animal": infer_interval[1],
                "interval_n_animals": infer_interval[2],
                "interval_merged": infer_interval[3],
            }
            inference_list = calculate_blob_inference_intervals(dm.blob_array, intervals)
        else:
            raise RuntimeError("Blob array has not been initialized in the workspace file!")
    else:
        inference_list = list(range(dm.total_frames))
    
    batch_log(f"DLC_Inference instantiated with inference_list of length: {len(inference_list)}")
    batch_log(f"First 10 frames in inference_list: {inference_list[:10]}")

    inference_window = DLC_Inference(
        dlc_data=dm.dlc_data,
        frame_list=inference_list,
        video_filepath=dm.video_file,
        roi=crop_region,
        parent=dialog
    )
    inference_window.cropping = crop
    if batch_size is not None:
        inference_window._batch_size_spinbox_changed(batch_size)
    if detector_batch_size is not None:
        inference_window._det_batch_size_spinbox_changed(detector_batch_size)
    if shuffle_idx is not None:
        available_shuffles = inference_window._check_available_shuffles()
        if shuffle_idx not in available_shuffles:
            batch_log(f"Supplied shuffle_idx - {shuffle_idx} not in available shuffles - {available_shuffles}, using the newest shuffle instead.")
        else:
            inference_window._shuffle_spinbox_changed(shuffle_idx)

    inference_window.fresh_pred = True
    batch_log("Inference process initiated.")
    inference_window._inference_pipe()

def _pseudo_callback(*arg, **kwargs):
    pass

if __name__ == "__main__":
    rootdir = "D:/Data/Videos/20251012 Marathon/"
    sys.stdout = open(f"{rootdir}/batch_log.txt", "a", encoding="utf-8")
    pkl_list = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".pkl") and "_batchcop_" not in file:
                pkl_list.append(os.path.join(root, file))

    print("About to batch process following workspace file.")
    for path in pkl_list:
        print(path)

    app = QtWidgets.QApplication([])
    success_count = 0
    for i, f in enumerate(pkl_list, 1):
        batch_log(f"\n[Batch {i}/{len(pkl_list)}] Starting: {os.path.basename(f)}")
        try:
            inference_workspace_vid(
                workspace_file=f,
                dlc_config_path="D:/Project/DLC-Models/NTD/config.yaml",
                blob_based_infer=True,
                infer_interval=(1,10,3,1),
                batch_size=128,
                detector_batch_size=32,
            )
            success_count += 1
            batch_log(f"Completed: {os.path.basename(f)}")
        except Exception as e:
            batch_log(f"FAILED: {os.path.basename(f)} — {e}")
            import traceback
            batch_log(f"Traceback:\n{traceback.format_exc()}")
            continue # Gliding over all

    batch_log(f"\nBatch finished: {success_count}/{len(pkl_list)} succeeded.")