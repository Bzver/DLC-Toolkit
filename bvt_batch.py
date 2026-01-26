import os
import numpy as np
import yaml
from PySide6 import QtWidgets
from typing import List, Tuple, Optional

from core.runtime import Data_Manager, Video_Manager
from core.tool.inference import DLC_Inference
from core.io import backup_existing_prediction, get_existing_projects, csv_op
from utils.helper import calculate_blob_inference_intervals
from utils.logger import logger, set_headless_mode


MAX_FRAMES_PER_RUN = 100000


def batch_to_h5(dlc_config_path: str):
    if not os.path.isfile(dlc_config_path):
        logger.error("[BATCH] DLC config file not found: %s", dlc_config_path)
        return

    dlc_dir = os.path.dirname(dlc_config_path)

    try:
        with open(dlc_config_path, 'r') as f:
            config_org = yaml.safe_load(f)
        scorer = config_org.get("scorer")
        if not scorer:
            logger.error("[BATCH] 'scorer' not found in DLC config.")
            return
    except Exception as e:
        logger.error("[BATCH] Failed to read DLC config: %s", e)
        return

    labeled_data_dir = os.path.join(dlc_dir, "labeled-data")
    if not os.path.isdir(labeled_data_dir):
        logger.warning("[BATCH] labeled-data directory not found: %s", labeled_data_dir)
        return

    success_count = 0
    skipped = []
    failed = []

    for project_name in os.listdir(labeled_data_dir):
        project_dir = os.path.join(labeled_data_dir, project_name)
        if not os.path.isdir(project_dir):
            continue

        csv_path = os.path.join(project_dir, f"CollectedData_{scorer}.csv")
        h5_path = os.path.join(project_dir, f"CollectedData_{scorer}.h5")

        if not os.path.isfile(csv_path):
            skipped.append((project_name, "CSV missing"))
            logger.debug("[BATCH] Skipped (CSV not found): %s", csv_path)
            continue
        if os.path.isfile(h5_path):
            skipped.append((project_name, "H5 already exists"))
            logger.debug("[BATCH] Skipped (H5 already exists): %s", h5_path)
            continue

        try:
            csv_op.csv_to_h5(csv_path=csv_path, multi_animal=True, scorer=scorer)
            success_count += 1
            logger.info("[BATCH] Successfully converted: %s", csv_path)
        except Exception as e:
            failed.append((project_name, str(e)))
            logger.error("[BATCH] Failed to convert %s: %s", csv_path, e)

    total = len([d for d in os.listdir(labeled_data_dir) if os.path.isdir(os.path.join(labeled_data_dir, d))])
    logger.info("[BATCH] Conversion finished: %d/%d succeeded.", success_count, total)

    if skipped:
        logger.info("[BATCH] Skipped %d projects:", len(skipped))
        for name, reason in skipped:
            logger.info("  - %s (%s)", name, reason)

    if failed:
        logger.error("[BATCH] Failed %d projects:", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)

def batch_grayscale(dlc_config_path):
    success_count = 0
    failed = []
    skipped = []

    projects = get_existing_projects(dlc_config_path)
    if not projects:
        logger.info("[BATCH] No projects found.")
        return

    logger.info("[BATCH] Found following labeled projects in DLC.")
    for i, path in enumerate(projects):
        logger.info(f"[{i}]: {path}")

    for project in projects:
        if project.endswith("_GR"):
            skipped.append(project)
            logger.info(f"[BATCH] Skipped (already grayscale): {project}")
            continue

        grayscaled_project = f"{project}_GR"
        if os.path.isdir(grayscaled_project):
            skipped.append(project)
            logger.info(f"[BATCH] Skipped (grayscale version already exists): {project}")
            continue

        try:
            dm = Data_Manager(init_vid_callback=_pseudo_callback, refresh_callback=_pseudo_callback, parent=dialog)
            dm.load_metadata_to_dm(dlc_config_path)
            dm.load_dlc_label(project)
            dm.migrate_existing_project(grayscaled_project, grayscaling=True)
        except Exception as e:
            error_msg = f"Failed to process project at {project}: {e}"
            logger.error(f"[BATCH] {error_msg}")
            failed.append(project)
        else:
            success_count += 1
            logger.info(f"[BATCH] Successfully grayscaled project at {project}.")

    total = len(projects)
    logger.info(f"[BATCH] Grayscale conversion finished: {success_count}/{total} succeeded.")
    if skipped:
        logger.info(f"[BATCH] Skipped {len(skipped)} projects:")
        for s in skipped:
            logger.info(f"  - {s}")
    if failed:
        logger.info(f"[BATCH] Failed {len(failed)} projects:")
        for f in failed:
            logger.info(f"  - {f}")

def batch_kp_normalization(dlc_config_path, task, dialog):
    if not os.path.isfile(dlc_config_path):
        logger.error("[BATCH] DLC config file not found: %s", dlc_config_path)
        return

    dlc_dir = os.path.dirname(dlc_config_path)

    try:
        with open(dlc_config_path, 'r') as f:
            config_org = yaml.safe_load(f)
        scorer = config_org.get("scorer")
        if not scorer:
            logger.error("[BATCH] 'scorer' not found in DLC config.")
            return
    except Exception as e:
        logger.error("[BATCH] Failed to read DLC config: %s", e)
        return

    labeled_data_dir = os.path.join(dlc_dir, "labeled-data")
    if not os.path.isdir(labeled_data_dir):
        logger.warning("[BATCH] labeled-data directory not found: %s", labeled_data_dir)
        return

    failed = []

    for project_name in os.listdir(labeled_data_dir):
        project_dir = os.path.join(labeled_data_dir, project_name)
        if not os.path.isdir(project_dir):
            continue

        if project_dir.endswith("_NM"):
            continue

        dm = Data_Manager(_pseudo_callback, _pseudo_callback, dialog)
        dm.load_metadata_to_dm(dlc_config_path)
        dm.load_dlc_label(project_dir)
        pred_data_array = dm.dlc_data.pred_data_array.copy()

        for t_idx, r1_idx, r2_idx in task:
            pred_data_array[:, :, t_idx*3:t_idx*3+2] = (dm.dlc_data.pred_data_array[:, :, r1_idx*3:r1_idx*3+2] + dm.dlc_data.pred_data_array[:, :, r2_idx*3:r2_idx*3+2]) / 2
        dm.dlc_data.pred_data_array = pred_data_array

        dm.migrate_existing_project(f"{project_dir}_NM")

    if failed:
        logger.error("[BATCH] Failed %d projects:", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)

def batch_inference(rootdir, dlc_config_path, dialog):
    pkl_list = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".pkl") and "backup" not in root:
                pkl_list.append(os.path.join(root, file))

    logger.info("[BATCH] About to batch process following workspace file.")
    for i, path in enumerate(pkl_list):
        logger.info(f"[{i}]:{path}")

    success_count = 0
    failed = []

    for i, f in enumerate(pkl_list, 1):
        filename = os.path.basename(f)
        logger.info(f"\n[Batch {i}/{len(pkl_list)}] Starting: {filename}")
        dm = Data_Manager(init_vid_callback=_pseudo_callback, refresh_callback=_pseudo_callback, parent=dialog)
        try:
            _inference_workspace_vid(
                workspace_file=f,
                data_manager=dm,
                dlc_config_path=dlc_config_path,
                crop=True,
                mask=True,
                grayscale=True,
                batch_size=32,
                detector_batch_size=32,
            )
        except Exception as e:
            logger.error(f"[Batch {i}/{len(pkl_list)}] FAILED: {filename} — {e} | ")
            logger.exception(f"[Batch {i}/{len(pkl_list)}]")
            failed.append(f)
            continue # Gliding over all
        else:
            success_count += 1
            logger.info(f"[Batch {i}/{len(pkl_list)}] Completed: {filename}")
            backup_existing_prediction(f)
            _autoload_pred(f, dm, dlc_config_path)
            dm.save_workspace()
            continue

    logger.info(f"[BATCH] Batch finished: {success_count}/{len(pkl_list)} succeeded.")
    if failed:
        logger.info(f"[BATCH] Failed videos:")
        for f in failed:
            logger.info(f)

def _inference_workspace_vid(
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
        mask:bool=False,
        grayscale:bool=False,
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

    if dm.dlc_data.pred_data_array is None:
        _autoload_pred(workspace_file, dm, dlc_config_path)

    if crop:
        assert crop_region is not None or dm.roi is not None, "[BATCH] Cropping enabled, but no crop region or ROI defined. Provide crop_region or ensure ROI is set in the workspace."
        if crop_region is None:
            try:
                x1, y1, x2, y2 = dm.roi
                crop_region = x1, y1, x2, y2
            except:
                raise RuntimeError("[BATCH] ROI in workspace is malformed, fail to translate the ROI in workspace.")
            
    if mask:
        assert dm.background_mask is not None, "No mask is loaded in data manager when mask parameter is set to True."

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
                if existing_frames:
                    logger.info(f"[BATCH] Loaded {len(existing_frames)} inferenced frames.")
                    inference_list = calculate_blob_inference_intervals(dm.blob_array, intervals, existing_frames)
                else:
                    logger.info("[BATCH] No existing frames found.")
                    inference_list = calculate_blob_inference_intervals(dm.blob_array, intervals)
            else:
                inference_list = calculate_blob_inference_intervals(dm.blob_array, intervals)
        else:
            raise RuntimeError("[BATCH] Blob array has not been initialized in the workspace file!")
    else:
        inference_list = list(range(dm.total_frames))

    if len(inference_list) > MAX_FRAMES_PER_RUN:
        logger.info(f"[BATCH] Splitting {len(inference_list)} frames into chunks.")
        chunked_lists = [
            inference_list[i:i + MAX_FRAMES_PER_RUN]
            for i in range(0, len(inference_list), MAX_FRAMES_PER_RUN)
        ]
    else:
        chunked_lists = [inference_list]

    for chunk_list in chunked_lists:
        _autoload_pred(workspace_file, dm, dlc_config_path)

        logger.info(f"[BATCH] DLC_Inference instantiated with inference_list of length: {len(chunk_list)}")

        inference_window = DLC_Inference(
            dlc_data=dm.dlc_data,
            frame_list=chunk_list,
            video_filepath=dm.video_file,
            roi=crop_region,
            mask=dm.background_mask,
            parent=dialog
        )
        inference_window.cropping = crop
        inference_window.masking = mask
        inference_window.grayscaling = grayscale
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

        logger.info(f"[BATCH] Inference process initiated. cropping: {crop}, masking: {dm.background_mask is not None}, grayscaling: {grayscale}")
        inference_window._inference_pipe(headless=True)

def _pseudo_callback(*arg, **kwargs):
    pass

def _autoload_pred(workspace_file:str, dm:Data_Manager, dlc_config_path:Optional[str]=None):
    workspace_dir = os.path.dirname(workspace_file)
    workspace_base = os.path.splitext(os.path.basename(workspace_file))[0].replace("_workspace","")

    assert dlc_config_path or (dm.dlc_data and dm.dlc_data.pred_data_array is not None), "DLC Config Missing!"

    candidate_preds = [
        os.path.join(workspace_dir, f) for f in os.listdir(workspace_dir) if f.endswith('.h5') and workspace_base in f
    ]

    if candidate_preds:
        candidate_preds.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        newest_pred = candidate_preds[0]

        logger.info(f"[AUTOLOAD] No prediction in workspace; loading newest prediction: {newest_pred}")
        try:
            if dm.dlc_data and dm.dlc_data.dlc_config_filepath:
                dm.load_pred_to_dm(
                    dlc_config_path=dm.dlc_data.dlc_config_filepath,
                    prediction_path=newest_pred
                )
            elif dlc_config_path:
                dm.load_pred_to_dm(
                    dlc_config_path=dlc_config_path,
                    prediction_path=newest_pred
                )
        except Exception as e:
            logger.error(f"[AUTOLOAD] Failed to auto-load prediction {newest_pred}: {e}")
        else:
            logger.info(f"[AUTOLOAD] Loaded newest pred {newest_pred}.")
    else:
        logger.info("[AUTOLOAD] No matching .h5 prediction found for auto-load.")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog()
    set_headless_mode(True)
    rootdir = "D:/Project/WORKINGONIT/20251201 Marathon/"
    dlc_config_path = "D:/Project/DLC-Models/NTD/config.yaml"
 
    batch_inference(rootdir, dlc_config_path, dialog)
    # batch_grayscale(dlc_config_path)
    # batch_to_h5(dlc_config_path)

    # task = [(11, 7, 9), (10, 6, 8), (13, 9, 3), (12, 8, 3)]
    # batch_kp_normalization(dlc_config_path, task, dialog)