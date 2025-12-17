import os
import numpy as np
from PySide6 import QtWidgets
from typing import List, Tuple, Optional

from core.runtime import Data_Manager
from core.tool.inference import DLC_Inference
from core.io import backup_existing_prediction
from utils.helper import calculate_blob_inference_intervals
from utils.logger import logger, set_headless_mode


def inference_workspace_vid(
        workspace_file:str,
        data_manager:Data_Manager,
        dlc_config_path:Optional[str]=None,
        partial_infer:bool=False,
        partial_infer_indices:Optional[List[int]]=None,
        blob_based_infer:bool=False,
        infer_interval:Optional[Tuple[int,int,int,int]]=None,
        infer_only_empty_frames:bool=False,
        crop:bool=False,
        crop_region:Optional[Tuple[int,int,int,int]]=None,
        shuffle_idx:Optional[int]=None,
        batch_size:Optional[int]=None,
        detector_batch_size:Optional[int]=None,
    ):
    assert os.path.isfile(workspace_file), f"[BATCH] Workspace file not found: {workspace_file}"

    logger.info(f"[BATCH] Workspace in {workspace_file} has been loaded.")
    dm = data_manager

    dm.load_workspace(workspace_file)
    
    if dlc_config_path is not None and os.path.isfile(dlc_config_path) and dm.dlc_data is None:
        dm.load_metadata_to_dm(dlc_config_path)
    else:
        assert dm.dlc_data is not None and dm.dlc_data.dlc_config_filepath is not None, "[BATCH] DLC configuration not found in workspace. Ensure the workspace includes a valid DLC project."
    assert dm.video_file is not None and os.path.isfile(dm.video_file), f"[BATCH] Video file missing or invalid: {dm.video_file}"

    if crop:
        assert crop_region is not None or dm.roi is not None, "[BATCH] Cropping enabled, but no crop region or ROI defined. Provide crop_region or ensure ROI is set in the workspace."
        if crop_region is None:
            try:
                x1, y1, x2, y2 = dm.roi
                crop_region = x1, y1, x2, y2
            except:
                raise RuntimeError("[BATCH] ROI in workspace is malformed, fail to translate the ROI in workspace.")

    inference_list = []
    if partial_infer:
        assert partial_infer_indices is not None, "[BATCH] partial_infer_indices must be provided when partial_infer is True"
        inference_list = partial_infer_indices
    elif blob_based_infer:
        assert infer_interval is not None, "[BATCH] infer_interval must be provided when blob_based_infer is True"
        logger.info(f"[BATCH] Blob-based inference selected with interval: {infer_interval}")
        if dm.blob_array is not None:
            intervals = {
                "interval_0_animal": infer_interval[0],
                "interval_1_animal": infer_interval[1],
                "interval_n_animals": infer_interval[2],
                "interval_merged": infer_interval[3],
            }
            if infer_only_empty_frames and dm.dlc_data.pred_data_array is not None:
                existing_frames = np.where(np.any(~np.isnan(dm.dlc_data.pred_data_array), axis=(1,2)))[0].tolist()
                inference_list = calculate_blob_inference_intervals(dm.blob_array, intervals, existing_frames)
            else:
                inference_list = calculate_blob_inference_intervals(dm.blob_array, intervals)
        else:
            raise RuntimeError("[BATCH] Blob array has not been initialized in the workspace file!")
    else:
        inference_list = list(range(dm.total_frames))
    
    logger.info(f"[BATCH] DLC_Inference instantiated with inference_list of length: {len(inference_list)}")

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
            logger.info(
                f"[BATCH] Supplied shuffle_idx - {shuffle_idx} not in available shuffles - {available_shuffles}, using the newest shuffle instead.")
        else:
            inference_window._shuffle_spinbox_changed(shuffle_idx)

    inference_window.fresh_pred = True
    logger.info("[BATCH] Inference process initiated.")
    inference_window._inference_pipe(headless=True)

def _pseudo_callback(*arg, **kwargs):
    pass

if __name__ == "__main__":
    set_headless_mode(True)
    rootdir = "D:/Data/Videos/20250918 Marathon"
    dlc_config_path = "D:/Project/DLC-Models/NTD/config.yaml"
    pkl_list = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".pkl") and "_batchcop_" not in file:
                pkl_list.append(os.path.join(root, file))

    logger.info("[BATCH] About to batch process following workspace file.")
    for i, path in enumerate(pkl_list):
        logger.info(f"[{i}]:{path}")

    app = QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog()
    success_count = 0
    for i, f in enumerate(pkl_list, 1):
        filename = os.path.basename(f)
        filefolder = os.path.dirname(f)
        logger.info(f"\n[Batch {i}/{len(pkl_list)}] Starting: {filename}")
        dm = Data_Manager(init_vid_callback=_pseudo_callback, refresh_callback=_pseudo_callback, parent=dialog)
        try:
            inference_workspace_vid(
                workspace_file=f,
                data_manager=dm,
                dlc_config_path=dlc_config_path,
                crop=True,
                blob_based_infer=True,
                infer_interval=(100,4,2,1),
                infer_only_empty_frames=True,
                batch_size=128,
                detector_batch_size=32,
            )
        except Exception as e:
            logger.error(f"[Batch {i}/{len(pkl_list)}] FAILED: {filename} — {e}")
            logger.exception(f"[Batch {i}/{len(pkl_list)}]")
            continue # Gliding over all
        else:
            success_count += 1
            logger.info(f"[Batch {i}/{len(pkl_list)}] Completed: {filename}")
            backup_existing_prediction(f)
            prediction_path = None
            for file in os.listdir(os.path.dirname(f)):
                if file.endswith(".h5") and filename.replace(".pkl", "") in file:
                    prediction_path = os.path.join(filefolder, file)
            if prediction_path:
                dm.load_pred_to_dm(dlc_config_path=dlc_config_path, prediction_path=prediction_path)
                dm.save_workspace()
            continue

    logger.info(f"[BATCH] Batch finished: {success_count}/{len(pkl_list)} succeeded.")