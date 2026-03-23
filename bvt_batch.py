import os
import shutil
import yaml
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
from PySide6 import QtWidgets
from typing import List, Tuple, Optional

from core.runtime import Data_Manager
from core.tool.inference import DLC_Inference
from core.tool.track_fix import Track_Fixer
from core.io import backup_existing_prediction, get_existing_projects, csv_op, prediction_to_csv, Frame_Extractor
from utils.helper import calculate_blob_inference_intervals, get_instance_count_per_frame
from utils.logger import logger, set_headless_mode


MAX_FRAMES_PER_RUN: int = 10000
WORKSPACE_EXTENSIONS: Tuple[str, ...] = (".joblib", ".pkl")
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mkv")


############################################################################################

def batch_backup_project(root_dir: str):
    workspaces = _find_files_by_extension(root_dir, WORKSPACE_EXTENSIONS)
    if not workspaces:
        logger.info("[BATCH] No old workspace files found for backup")
        return

    _log_batch_progress("Workspace Backup", workspaces)
    backup_dir = os.path.join(root_dir, "bvt_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    def process_workspace(ws_path:str) -> bool:
        rel_dir = ws_path.split(root_dir)[1]
        new_path = f"{backup_dir}{rel_dir}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        if os.path.isfile(new_path):
            backup_existing_prediction(new_path)
        shutil.copy(ws_path, new_path)
        return os.path.isfile(new_path)

    _process_batch(workspaces, process_workspace, "Workspace Backup")

def batch_migration_pkl_to_joblib(root_dir: str):
    workspaces = _find_files_by_extension(root_dir, ".pkl")
    if not workspaces:
        logger.info("[BATCH] No old workspace files found for migration")
        return

    _log_batch_progress("PKL Migration", workspaces)
    
    def process_workspace(ws_path:str) -> bool:
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback,
            refresh_callback=_pseudo_callback
        )
        dm.load_workspace(ws_path)
        ws_newpath = ws_path.replace(".pkl",".joblib")
        return os.path.isfile(ws_newpath)

    _process_batch(workspaces, process_workspace, "PKL Migration")

def batch_convert_csv_to_h5(dlc_config_path: str):
    projects = _find_labeled_projects(dlc_config_path)
    if not projects:
        return
    
    with open(dlc_config_path, 'r') as f:
        scorer = yaml.safe_load(f)["scorer"]
    
    def process_project(project_dir: str) -> bool:
        csv_path = os.path.join(project_dir, f"CollectedData_{scorer}.csv") 
        h5_path = os.path.join(project_dir, f"CollectedData_{scorer}.h5") 
        
        if not os.path.exists(csv_path):
            logger.debug(f"[BATCH] Skip (no CSV): {project_dir}")
            return False
        
        if os.path.exists(h5_path):
            logger.debug(f"[BATCH] Skip (H5 exists): {project_dir}")
            return False
        
        csv_op.csv_to_h5(csv_path=str(csv_path), multi_animal=True, scorer=scorer)
        logger.info(f"[BATCH] Converted: {csv_path}")
        return True
    
    _process_batch(projects, process_project, "CSV→H5 conversion")

def batch_convert_to_grayscale(dlc_config_path: str):
    projects = get_existing_projects(dlc_config_path)
    if not projects:
        logger.info("[BATCH] No projects found for grayscale conversion")
        return
    
    _log_batch_progress("grayscale conversion", projects)
    
    def process_project(project_path: str) -> bool:
        if project_path.endswith("_GR"):
            logger.debug(f"[BATCH] Skip (already GR): {project_path}")
            return False
        
        grayscaled_path = f"{project_path}_GR"
        if os.path.exists(grayscaled_path):
            logger.debug(f"[BATCH] Skip (GR exists): {project_path}")
            return False
        
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback, 
            refresh_callback=_pseudo_callback
        )
        dm.load_metadata_to_dm(str(dlc_config_path))
        dm.load_dlc_label(str(project_path))
        dm.migrate_existing_project(str(grayscaled_path), grayscaling=True)
        
        logger.info(f"[BATCH] Grayscaled: {project_path}")
        return True
    
    _process_batch(projects, process_project, "grayscale")

def batch_create_workspaces(rootdir: str, dlc_config_path: str):
    videos = _find_video_files(rootdir)
    if not videos:
        logger.info("[BATCH] No videos requiring workspace creation")
        return
    
    _log_batch_progress("workspace creation", videos)
    
    def process_video(video_path: str) -> bool:
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback,
            refresh_callback=_pseudo_callback
        )
        
        dm.update_video_path(video_path=str(video_path))
        
        with managed_frame_extractor(video_path) as extractor:
            dm.total_frames = extractor.get_total_frames()
        
        dm.load_metadata_to_dm(str(dlc_config_path))
        dm.save_workspace()
        
        return True
    
    _process_batch(videos, process_video, "workspace creation")

def batch_export_csv(
        root_dir: str, 
        with_conf:bool=True,
        animal_num_filtering:bool=False,
        min_animal_num:int=2,
        frame_count_filtering:bool=False,
        frame_count_max:int=6000,
        no_scorer_header:bool=False
    ):
    workspaces = _find_files_by_extension(root_dir, WORKSPACE_EXTENSIONS)
    if not workspaces:
        logger.info("[BATCH] No workspace files found for export")
        return
    
    _log_batch_progress("CSV export", workspaces)
    
    def process_workspace(ws_path:str) -> bool:
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback,
            refresh_callback=_pseudo_callback
        )
        dm.load_workspace(str(ws_path))
        
        pred_data = dm.dlc_data.pred_data_array

        if animal_num_filtering:
            counts = get_instance_count_per_frame(pred_data)
            pred_data = pred_data[counts >= min_animal_num]
        
        if frame_count_filtering:
            pred_data = pred_data[:frame_count_max]

        output_path = (
            str(ws_path)
            .replace("_workspace.joblib", "_auto_export.csv")
            .replace("_workspace.pkl", "_auto_export.csv")
        )
        
        prediction_to_csv(
            dm.dlc_data,
            pred_data,
            save_path=output_path,
            keep_conf=with_conf,
            no_scorer_row=no_scorer_header,
        )
        return True
    
    _process_batch(workspaces, process_workspace, "CSV export")


def batch_correct_tracking(root_dir:str) -> None:
    workspaces = _find_files_by_extension(root_dir, WORKSPACE_EXTENSIONS)
    if not workspaces:
        logger.info("[BATCH] No workspace files found for track correction")
        return
    
    _log_batch_progress("track correction", workspaces)
    
    def process_workspace(ws_path: str) -> bool:
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback,
            refresh_callback=_pseudo_callback
        )
        dm.load_workspace(str(ws_path))
        
        with managed_frame_extractor(dm.video_file) as extractor:
            if dm.angle_map_data is None:
                dm._init_canon_pose()
            
            tf = Track_Fixer(
                dm.dlc_data.pred_data_array,
                dm.dlc_data,
                extractor,
                dm.angle_map_data,
                avtomat=True
            )
            dm.dlc_data.pred_data_array = tf.track_correction()
        
        dm.save_workspace()
        return True
    
    _process_batch(workspaces, process_workspace, "track correction")

@contextmanager
def managed_frame_extractor(video_path:str):
    extractor = Frame_Extractor(video_path)
    try:
        yield extractor
    finally:
        extractor.close()

############################################################################################

def _find_video_files(root_dir: str) -> List[str]:
    videos = _find_files_by_extension(root_dir, VIDEO_EXTENSIONS)
    filtered = []
    for video in videos:
        workspace_candidates = [
            video.with_name(f"{video.stem}_workspace{ext}")
            for ext in WORKSPACE_EXTENSIONS
        ]
        if not any(os.path.exists(ws) for ws in workspace_candidates):
            filtered.append(video)
    
    return filtered

def _find_labeled_projects(dlc_config_path: str) -> List[str]:
    if not os.path.isfile(dlc_config_path):
        logger.error(f"[BATCH] DLC config not found: {dlc_config_path}")
        return []
    
    try:
        with open(dlc_config_path, 'r') as f:
            config = yaml.safe_load(f)
        scorer = config.get("scorer")
        if not scorer:
            logger.error("[BATCH] 'scorer' missing in DLC config")
            return []
    except Exception as e:
        logger.error(f"[BATCH] Failed to parse DLC config: {e}")
        return []
    
    labeled_dir = os.path.join(os.path.dirname(dlc_config_path), "labeled-data")
    if not os.path.isdir(labeled_dir):
        logger.warning(f"[BATCH] labeled-data directory not found: {labeled_dir}")
        return []
    
    all_dirs = []
    for d in os.listdir(labeled_dir):
        d_full = os.path.join(labeled_dir, d)
        all_dirs.append(d_full)

    return all_dirs

def _find_files_by_extension(root_dir:str, extensions: Tuple[str]) -> List[str]:
    found = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extensions) and "backup" not in root and "temp" not in root:
                found.append(os.path.join(root, file))
    return found

############################################################################################

def _process_batch(
    items: List[str], 
    processor: callable,
    operation_name: str
) -> Tuple[int, List[Tuple[str, str]]]:
    success_count = 0
    failures = []
    
    for idx, item in enumerate(items, start=1):
        filename = os.path.basename(item)
        logger.info(f"\n[{operation_name} {idx}/{len(items)}] Processing: {filename}")
        
        try:
            if processor(item):
                success_count += 1
                logger.info(f"[{operation_name} {idx}/{len(items)}] Completed: {filename}")
            else:
                logger.warning(f"[{operation_name} {idx}/{len(items)}] Skipped: {filename}")
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{operation_name} {idx}/{len(items)}] FAILED: {filename} — {error_msg}")
            logger.exception(f"[{operation_name} {idx}/{len(items)}] Stack trace:")
            failures.append((item, error_msg))

    logger.info(f"[BATCH] {operation_name} finished: {success_count}/{len(items)} succeeded.")
    
    if failures:
        logger.warning(f"[BATCH] {len(failures)} item(s) failed:")
        for path, error in failures:
            logger.warning(f"  - {path}: {error}")
    
    return success_count, failures

def _pseudo_callback(*arg, **kwargs) -> None:
    pass

def _log_batch_progress(operation: str, items: List[str]) -> None:
    logger.info(f"[BATCH] Starting {operation} for {len(items)} item(s).")
    for idx, path in enumerate(items):
        logger.debug(f"  [{idx}]: {path}")

############################################################################################

def batch_inference(
    rootdir: str, 
    dlc_config_path: str, 
    crop: bool = False,
    mask: bool = False,
    grayscale: bool = False,
    batch_size: int = 32,
    detector_batch_size: int = 32,
    shuffle_idx: Optional[int] = None,
    partial_infer: bool = False,
    partial_infer_indices: Optional[List[int]] = None,
    blob_based_infer: bool = False,
    infer_interval: Optional[Tuple[int, int, int, int]] = None,
    infer_only_empty_frames: bool = False,
    crop_region: Optional[Tuple[int, int, int, int]] = None
):
    workspaces = _find_files_by_extension(rootdir, WORKSPACE_EXTENSIONS)
    
    if not workspaces:
        logger.info("[BATCH] No workspace files found for inference")
        return

    _log_batch_progress("inference", workspaces)
    
    def process_workspace(ws_path: str) -> bool:
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback,
            refresh_callback=_pseudo_callback
        )

        _inference_workspace_vid(
            workspace_file=ws_path,
            data_manager=dm,
            dlc_config_path=dlc_config_path,
            partial_infer=partial_infer,
            partial_infer_indices=partial_infer_indices,
            blob_based_infer=blob_based_infer,
            infer_interval=infer_interval,
            infer_only_empty_frames=infer_only_empty_frames,
            crop=crop,
            crop_region=crop_region,
            use_mask=mask,
            grayscale=grayscale,
            shuffle_idx=shuffle_idx,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
        )
        backup_existing_prediction(ws_path)
        _autoload_pred(ws_path, dm, dlc_config_path)
        dm.save_workspace()
        
        if dm.video_file:
            _cleanup_old_auto_predictions(os.path.dirname(dm.video_file))
            
        return True

    _process_batch(workspaces, process_workspace, "inference")

def _inference_workspace_vid(
        workspace_file: str,
        data_manager: Data_Manager,
        dlc_config_path: Optional[str] = None,
        partial_infer: bool = False,
        partial_infer_indices: Optional[List[int]] = None,
        blob_based_infer: bool = False,
        infer_interval: Optional[Tuple[int, int, int, int]] = None,
        infer_only_empty_frames: bool = False,
        crop: bool = False,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        use_mask: bool = False,
        grayscale: bool = False,
        shuffle_idx: Optional[int] = None,
        batch_size: Optional[int] = None,
        detector_batch_size: Optional[int] = None,
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
            
    if use_mask:
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
    elif infer_only_empty_frames and dm.dlc_data.pred_data_array is not None:
        inference_list = np.where(np.all(np.isnan(dm.dlc_data.pred_data_array), axis=(1,2)))[0].tolist()
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
        _cleanup_old_auto_predictions(os.path.dirname(dm.video_file))

        logger.info(f"[BATCH] DLC_Inference instantiated with inference_list of length: {len(chunk_list)}")

        inference_window = DLC_Inference(
            dlc_data=dm.dlc_data,
            video_length=dm.total_frames,
            frame_list=chunk_list,
            video_filepath=dm.video_file,
            roi=crop_region,
            mask=dm.background_mask if use_mask else None
        )
        inference_window.hide()
        inference_window.cropping = crop
        inference_window.masking = use_mask
        inference_window.grayscaling = grayscale

        to_video = not any([blob_based_infer, infer_only_empty_frames, partial_infer])
        inference_window.to_video_checkbox.setChecked(to_video)
        if batch_size is not None:
            inference_window.batchsize_spinbox.setValue(batch_size)
        if detector_batch_size is not None:
            inference_window.detector_batchsize_spinbox.setValue(detector_batch_size)
        if shuffle_idx is not None:
            available_shuffles = inference_window._check_available_shuffles()
            if shuffle_idx not in available_shuffles:
                logger.info(f"[BATCH] Supplied shuffle_idx - {shuffle_idx} not in available shuffles - {available_shuffles}, using the newest shuffle instead.")
            else:
                inference_window._shuffle_spinbox_changed(shuffle_idx)

        logger.info(f"[BATCH] Inference process initiated. cropping: {crop}, masking: {use_mask}, grayscaling: {grayscale}")
        inference_window._inference_pipe(headless=True)


def _autoload_pred(workspace_file: str, dm: Data_Manager, dlc_config_path: Optional[str] = None):
    workspace_dir = os.path.dirname(workspace_file)
    workspace_base = os.path.splitext(os.path.basename(workspace_file))[0].replace("_workspace", "")

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

def _cleanup_old_auto_predictions(video_dir: str, keep: int = 2):
    groups = defaultdict(list)

    for filename in os.listdir(video_dir):
        video_name, dlc_params = _parse_auto_pred_filename(filename)
        if video_name is None:
            continue  # Not an auto file

        full_path = os.path.join(video_dir, filename)
        groups[(video_name, dlc_params)].append(full_path)

    total_deleted = 0

    for file_list in groups.values():
        if len(file_list) <= keep:
            continue

        file_list.sort(key=lambda p: os.path.getmtime(p))
        to_delete = file_list[:-keep]

        for filepath in to_delete:
            try:
                os.remove(filepath)
                total_deleted += 1
                logger.debug(f"[BATCH] Deleted old auto-pred: {os.path.basename(filepath)}")
                csv_path = filepath.replace(".h5", ".csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)
                    logger.debug(f"[BATCH] Deleted corresponding CSV: {os.path.basename(csv_path)}")
            except Exception as e:
                logger.warning(f"[BATCH] Failed to delete {filepath}: {e}")

    if total_deleted > 0:
        logger.info(f"[BATCH] Cleaned up {total_deleted} old auto-prediction files.")

def _parse_auto_pred_filename(filename: str):
    if not filename.endswith(".h5"):
        return None, None
    if "_auto_" not in filename:
        return None, None
    if "_inference_" not in filename:
        return None, None

    parts = filename.split("_auto_", 1)
    if len(parts) != 2:
        return None, None

    video_name = parts[0]
    rest = parts[1]

    if rest.endswith(".h5"):
        rest = rest[:-3]

    inference_pos = rest.rfind("_inference_")
    if inference_pos == -1:
        return None, None

    dlc_params = rest[:inference_pos]
    return video_name, dlc_params


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    set_headless_mode(True)
    rootdir = r"D:\DGH\Data\Videos\20250913 Marathon"
    dlc_config_path = "D:/Project/DLC-Models/NTD/config.yaml"
 
    dial_tone = 1

    match dial_tone:
        case 1:
            batch_inference(
                rootdir,
                dlc_config_path,
                crop=True,
                mask=False,
                grayscale=True,
                infer_only_empty_frames=False,
                batch_size=32,
                detector_batch_size=16
            )
        case 2:
            batch_export_csv(
                rootdir,
                with_conf=True,
                no_scorer_header=True,
                animal_num_filtering=True,
                min_animal_num=2,
                frame_count_filtering=True,
                frame_count_max=6000,
                )
        case 3: batch_create_workspaces(rootdir, dlc_config_path)
        case 4: batch_convert_to_grayscale(dlc_config_path)
        case 5: batch_convert_csv_to_h5(dlc_config_path)
        case 6: batch_migration_pkl_to_joblib(rootdir)
        case 7: batch_backup_project(rootdir)
