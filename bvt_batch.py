import os
import numpy as np
import pandas as pd
import shutil
import yaml
import json
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from PySide6 import QtWidgets
from typing import List, Tuple, Optional, Literal

from core.runtime import Data_Manager
from core.tool import DLC_Inference, Track_Fixer, Mark_Generator, Outlier_Finder, Annot_Exporter
from core.io import backup_existing_prediction, get_existing_projects, prediction_to_csv, Frame_Extractor, Temp_Manager, csv_op
from ui import Track_Fix_Config_Dialog
from utils.helper import get_instance_count_per_frame, clean_outside_roi_pred, clean_pred_in_mask, validate_crop_coord
from utils.logger import logger, set_headless_mode


MAX_FRAMES_PER_RUN: int = 10000
WORKSPACE_EXTENSIONS: Tuple[str, ...] = (".joblib", ".pkl")
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mkv")


def batch_backup_project(rootdir: str):
    workspaces = _find_files_by_extension(rootdir, ".joblib")
    if not workspaces:
        logger.info("[BATCH] No old workspace files found for backup")
        return

    _log_batch_progress("Workspace Backup", workspaces)
    backup_dir = os.path.join(rootdir, "bvt_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    def process_workspace(ws_path:str) -> bool:
        rel_dir = ws_path.split(rootdir)[1]
        new_path = f"{backup_dir}{rel_dir}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        if os.path.isfile(new_path):
            backup_existing_prediction(new_path)
        shutil.copy(ws_path, new_path)
        return os.path.isfile(new_path)

    _process_batch(workspaces, process_workspace, "Workspace Backup")

def batch_extract_track_fix_info(rootdir: str):
    target_files = {"agreement_timeline.png", "diagnosis_timeline.csv", "tsne_combined.png", "stable_spans.json"}
    levels = ("101T_", "103T_", "201T_", "203T_", "301T_", "303T_", "401T_", "403T_")
    for root, _, files in os.walk(rootdir):
        if "bvt_temp" not in root:
            continue
        for file in files:
            if file not in target_files:
                continue
            for level in levels:
                if level in root:
                    level_trunc = level.replace("T_", "")
                    base_dir = root.split("bvt_temp")[0]
                    dest_folder = os.path.join(base_dir, level_trunc)
                    os.makedirs(dest_folder, exist_ok=True)
                    shutil.move(os.path.join(root, file), os.path.join(dest_folder, file))
                    logger.info(f"Moved track fix info from {root} to {dest_folder}.")
                    break

def batch_migration_pkl_to_joblib(rootdir: str):
    workspaces = _find_files_by_extension(rootdir, ".pkl")
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
        rootdir: str, 
        with_conf:bool=True,
        animal_num_filtering:bool=False,
        min_animal_num:int=2,
        frame_count_filtering:bool=False,
        frame_count_max:int=6000,
        no_scorer_header:bool=False
    ):
    workspaces = _find_files_by_extension(rootdir, ".joblib")
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

def batch_data_clean(rootdir:str, kp_clean:bool=False, inst_clean:bool=False):
    workspaces = _find_files_by_extension(rootdir, ".joblib")
    if not workspaces:
        logger.info("[BATCH] No workspace files found for track correction")
        return

    _log_batch_progress("data cleaning", workspaces)
    def process_workspace(ws_path: str) -> bool:
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback,
            refresh_callback=_pseudo_callback
        )
        dm.load_workspace(str(ws_path))
    
        if dm.angle_map_data is None:
            dm._init_canon_pose()

        pred_data_array = dm.dlc_data.pred_data_array
        roi = validate_crop_coord(dm.roi)
        if roi is not None:
            pred_data_array = clean_outside_roi_pred(pred_data_array, roi)
        if dm.background_mask is not None:
            pred_data_array = clean_pred_in_mask(pred_data_array, dm.background_mask)

        dm.dlc_data.pred_data_array = pred_data_array

        of = Outlier_Finder(
            pred_data_array=dm.dlc_data.pred_data_array,
            skele_list=dm.dlc_data.skeleton,
            kp_to_idx=dm.dlc_data.keypoint_to_idx,
            angle_map_data=dm.angle_map_data)
        
        of.hide()

        if kp_clean:
            of.mode_combo.setCurrentText("Keypoint")
            of.keypoint_container.outlier_bone_gbox.setChecked(True)
            of.keypoint_container.outlier_confidence_gbox.setChecked(False)
            of._get_outlier_mask()
            outlier_kp_mask = of.outliers

            F, I, D = dm.dlc_data.pred_data_array.shape
            K = D//3
            pred_data_array = np.reshape(dm.dlc_data.pred_data_array, (F, I, K, 3))
            pred_data_array[outlier_kp_mask] = np.nan
            pred_data_array = np.reshape(pred_data_array, (F, I, K*3))

        if inst_clean:
            of.mode_combo.setCurrentText("Instance")
            of.instance_container.outlier_duplicate_gbox.setChecked(True)
            of.instance_container.confidence_spinbox.setValue(0.6)
            of.instance_container.outlier_size_gbox.setChecked(True)
            of.instance_container.min_size_spinbox.setValue(0.3)
            of.instance_container.outlier_bodypart_gbox.setChecked(True)
            of.instance_container.bodypart_spinbox.setValue(3)
            of._get_outlier_mask()
            outlier_inst_mask = of.outliers
            
            pred_data_array[outlier_inst_mask] = np.nan

        dm.dlc_data.pred_data_array = pred_data_array
        dm.save_workspace()

        return True
    
    _process_batch(workspaces, process_workspace, "data cleaning")

def batch_export_to_asoid_inference(rootdir:str, catalogue_file:Optional[str]=None, behav_map_file:Optional[str]=None):
    workspaces = _find_files_by_extension(rootdir, ".joblib")
    if not workspaces:
        logger.info("[BATCH] No workspace files.")
        return

    csv_lookup = None
    if catalogue_file:
        df = pd.read_csv(catalogue_file, header=None)
        df['date'] = df[0].str.split(' ').str[0]
        csv_lookup = dict(zip(zip(df['date'], df[1].astype(str)), df[2]))

    behav_map = None
    if behav_map_file:
        with open(behav_map_file, 'r') as f:
            meta = json.load(f)
            behav_map = meta["behav_map"]

    _log_batch_progress("Exporting to Asoid Inference", workspaces)
    def process_workspace(ws_path: str) -> bool:
        dm = Data_Manager(
            init_vid_callback=_pseudo_callback,
            refresh_callback=_pseudo_callback
        )
        dm.load_workspace(str(ws_path))
    
        def generate_pred_filename(file_path, csv_lookup):
            parts = file_path.replace('\\', '/').split('/')
            filename = parts[-1]

            m_idx = next(i for i, p in enumerate(parts) if 'Marathon' in p)
            
            proj_folder = parts[m_idx]
            sub_folder = parts[m_idx + 1] if len(parts) > m_idx + 2 else None
            
            proj_date_str = proj_folder.split(' ')[0]
            proj_dt = datetime.strptime(proj_date_str, "%Y%m%d")
            
            if sub_folder and len(sub_folder) == 4 and sub_folder.isdigit():
                sub_dt = datetime.strptime(f"{proj_dt.year}{sub_folder}", "%Y%m%d")
                day_num = (sub_dt - proj_dt).days + 1
            else:
                day_num = 1

            subj_id = filename.split('_')[0][:-1] 
            type_val = csv_lookup[(proj_date_str, subj_id[0])]
            
            last_two = subj_id[-2:]
            if type_val == 'L':
                role = 'sub' if last_two == '01' else 'dom'
            else:
                role = 'dom' if last_two == '01' else 'sub'

            mmdd = proj_date_str[4:]
            return f"{mmdd}_{role}_day{day_num}_{subj_id}.csv"

        if csv_lookup is not None:
            save_name = generate_pred_filename(ws_path, csv_lookup)
        else:
            save_name = ws_path.replace(".joblib", "_auto_export.csv")
        
        save_path = os.path.join(os.path.dirname(ws_path), save_name)
        ae = Annot_Exporter(np.zeros(dm.total_frames), dm.video_file, behav_map, None, dm.roi)
        ae.to_asoid_infer(save_path, dm.dlc_data)

        return True
    
    _process_batch(workspaces, process_workspace, "Exporting to Asoid Inference")

def batch_temp_dir_clean(rootdir:str):
    for root, dirs, _ in os.walk(rootdir):
        for dirr in dirs:
            if dirr != "bvt_temp":
                continue

            Temp_Manager(os.path.join(root, "blahblah.mp4"), force_clean=True)
            temp_dir_root = os.path.join(root, "bvt_temp")

            for entry in os.listdir(temp_dir_root):
                full_path = os.path.join(temp_dir_root, entry)
                if os.path.isdir(full_path):
                    logger.info(f"[TMCLEAN] Failed to remove {full_path}.")
            
    logger.info("[TMCLEAN] Finished.")

def batch_track_fix(
        rootdir:str,
        auto_load_weight:bool=False,
        inference_only:bool=False,
        lock_id:bool=False,
        force_locked_id:int=-1,
        ):

    workspaces = _find_files_by_extension(rootdir, ".joblib")
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
        tfd = Track_Fix_Config_Dialog(dm.total_frames)
        tfd.hide()

        if auto_load_weight:
            weight_path = ws_path.replace("_workspace.joblib", "_contrastive_trained.pth")
            if os.path.isfile(weight_path):
                logger.info(f"Auto loaded weight path at {weight_path}")
                tfd.pretrained_model_path = weight_path
        
        if inference_only and tfd.pretrained_model_path is not None:
            tfd.preset_combo.setCurrentText("Inference")

        tfd._on_accept()

        skip_sweep = tfd.skip_motion_sweep
        skip_contrast = tfd.skip_contrastive
        use_cache = tfd.use_cache
        emp = tfd.emp

        with managed_frame_extractor(dm.video_file) as extractor:
            if dm.angle_map_data is None:
                dm._init_canon_pose()

            tf = Track_Fixer(
                pred_data_array=dm.dlc_data.pred_data_array,
                dlc_data=dm.dlc_data,
                extractor=extractor,
                anglemap=dm.angle_map_data,
                skip_sweep=skip_sweep,
                blob_array=dm.blob_array if lock_id else None,
                force_locked_id=force_locked_id,
                avtomat=True,
                skip_contrast=skip_contrast,
                use_cache=use_cache,
                emp=emp,
            )
            dm.dlc_data.pred_data_array = tf.track_correction()
        
        dm.save_workspace()
        save_path = f"{ws_path.split('_workspace')[0]}_auto_pred_corrected.h5"
        dm.save_pred(dm.dlc_data.pred_data_array, save_path)
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

def _find_video_files(rootdir: str) -> List[str]:
    videos = _find_files_by_extension(rootdir, VIDEO_EXTENSIONS)
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

def _find_files_by_extension(rootdir:str, extensions: Tuple[str]) -> List[str]:
    found = []
    for root, _, files in os.walk(rootdir):
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
    force_load_new_config: bool = False,
    use_dm_list: bool = False,
    mark_gen_mode: Literal["None", "NM", "NM-I", "stride10"] = "None",
    crop: bool = False,
    mask: bool = False,
    grayscale: bool = False,
    infer_as_video: bool = False,
    batch_size: int = 32,
    detector_batch_size: int = 32,
    shuffle_idx: Optional[int] = None,
    crop_region: Optional[Tuple[int, int, int, int]] = None
):
    workspaces = _find_files_by_extension(rootdir, ".joblib")
    
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
            force_load_new_config=force_load_new_config,
            use_dm_list=use_dm_list,
            mark_gen_mode=mark_gen_mode,
            crop=crop,
            crop_region=crop_region,
            use_mask=mask,
            grayscale=grayscale,
            infer_as_video=infer_as_video,
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
        force_load_new_config: bool = False,
        use_dm_list: bool = False,
        mark_gen_mode: Literal["None", "NM", "NM-I", "stride10"] = "None",
        crop: bool = False,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        use_mask: bool = False,
        grayscale: bool = False,
        infer_as_video:bool = False,
        shuffle_idx: Optional[int] = None,
        batch_size: Optional[int] = None,
        detector_batch_size: Optional[int] = None,
    ):
    assert os.path.isfile(workspace_file), f"[BATCH] Workspace file not found: {workspace_file}"

    logger.info(f"[BATCH] Workspace in {workspace_file} has been loaded.")
    dm = data_manager

    dm.load_workspace(workspace_file)

    if dlc_config_path is not None and os.path.isfile(dlc_config_path) and (dm.dlc_data is None or force_load_new_config):
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

    if use_dm_list:
        inference_list = dm.get_frames("marked")
    elif mark_gen_mode != "None":
        inference_list = _acquire_inference_list_from_markgen(dm, mark_gen_mode)
    else:
        inference_list = sorted(range(dm.total_frames))

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
        inference_window.cropping_checkbox.setChecked(crop)
        inference_window.masking_checkbox.setChecked(use_mask)
        inference_window.grayscaling_checkbox.setChecked(grayscale)
        inference_window.to_video_checkbox.setChecked(infer_as_video)
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
        QtWidgets.QApplication.processEvents()
        inference_window._inference_pipe(headless=True)
        inference_window.close()

def _acquire_inference_list_from_markgen(dm:Data_Manager, mark_gen_mode:Literal["None", "NM", "NM-I", "stride10"]):
    mg = Mark_Generator(
        total_frames=dm.total_frames,
        pred_data_array=dm.dlc_data.pred_data_array if dm.dlc_data else None,
        blob_array=dm.blob_array,
        dlc_data=dm.dlc_data,
        angle_map_data=dm.angle_map_data
        )
    mg.hide()

    match mark_gen_mode:
        case "NM":
            inference_list = []
            if dm.blob_array is not None or np.any(dm.blob_array!=0):
                mg.mode_option.setCurrentText("Animal Num")
                mg.blob_source_radio.setChecked(True)
                mg.two_plus_animal_checkbox.setChecked(True)

                frames_by_count = mg.find_frames_to_mark()
                if frames_by_count:
                    inference_list.extend(frames_by_count)
                    mg.buffer_size_spin.setValue(20)
                    frames_buffer = mg._mark_count_change_frames()
                    inference_list.extend(frames_buffer)

            mg.mode_option.setCurrentText("Stride")
            mg.stride_spin.setValue(10)
            frames_by_interval = mg.find_frames_to_mark()
            if frames_by_interval:
                inference_list.extend(frames_by_interval)
        case "NM-I":
            nmi_prereqs = [
                dm.blob_array is not None,
                dm.dlc_data is not None,
                dm.dlc_data.pred_data_array is not None
                ]
            assert all(nmi_prereqs), "Inference list acquisition with 'NM=I' mode must have all of non None (blob_array, dlc_data.pred_data_array) in dm."

            full_mask = np.zeros(dm.total_frames, dtype=bool)

            mg.mode_option.setCurrentText("Animal Num")

            mg.dlc_source_radio.setChecked(True)
            mg.zero_animal_checkbox.setChecked(True)
            mask_0 = full_mask.copy()
            mask_0[mg.find_frames_to_mark()] = True
            mg.one_animal_checkbox.setChecked(True)
            mask_0_1 = full_mask.copy()
            mask_0_1[mg.find_frames_to_mark()] = True

            mg.zero_animal_checkbox.setChecked(False)
            mg.one_animal_checkbox.setChecked(False)

            mg.blob_source_radio.setChecked(True)
            mg.two_plus_animal_checkbox.setChecked(True)

            frames_by_count = mg.find_frames_to_mark()
            mask_2p = full_mask.copy()
            mask_2p[frames_by_count] = mask_0_1[frames_by_count]

            mg.buffer_size_spin.setValue(10)
            frames_buffer = mg._mark_count_change_frames()
            mask_bf = full_mask.copy()
            mask_bf[frames_buffer] = mask_0[frames_buffer]

            full_mask = mask_2p | mask_bf
            inference_list = np.where(full_mask)[0].tolist()
        case "stride10":
            mg.mode_option.setCurrentText("Stride")
            mg.stride_spin.setValue(10)
            inference_list = mg.find_frames_to_mark()

    mg.close()
    return sorted(set(inference_list))

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

############################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    set_headless_mode(True)

    quened_dirs = [
        # r'D:\Data\Videos\20250913 Marathon',
        # r'D:\Data\Videos\20250918 Marathon',
        # r'D:\Data\Videos\20251012 Marathon',
        # r'D:\Data\Videos\20251018 Marathon',
        # r'D:\Data\Videos\20251101 Marathon',
        # r'D:\Data\Videos\20251117 Marathon',
        # r'D:\Data\Videos\20251201 Marathon',
        # r'D:\Data\Videos\20260324 Marathon',
        r'D:\Data\Videos\20260416 Marathon',
    ]

    dial_tones = [13]

    """
    1 - inference; 2 - rerun; 3 - track fix; 4 - track clean; 5 - temp dir clean;
    6 - csv 2 h5; 7 - pkl migration; 8 - backup workspace; 9- create workspace;
    10 - extract track fix info; 11 - convert grayscale; 12 - export csv; 13 - export ASOID
    """

    dlc_config_path =r"D:/Project/DLC-Models/NTD-Blob/config.yaml"
    catalogue_file=r"D:\Data\Videos\catalogue.csv"
    behav_map_file=r"D:\Data\Videos\20250913 Marathon\301T_aaa_20251030160615_combined_cut_annot.json"

    CROPPING = True
    MASKING = False
    GRAYSCALING = False
    BATCH = 16
    DT_BATCH = 16

    for rootdir in quened_dirs:
        if not os.path.isdir(rootdir):
            print(f"{rootdir} does not exist!")
            continue
        for tone in dial_tones:
            match tone: 
                case 1:
                    batch_inference(
                        rootdir,
                        dlc_config_path,
                        force_load_new_config=True,
                        mark_gen_mode="NM",
                        use_dm_list=False,
                        crop=CROPPING,
                        mask=MASKING,
                        grayscale=GRAYSCALING,
                        infer_as_video=False,
                        batch_size=BATCH,
                        detector_batch_size=DT_BATCH
                    )
                    batch_temp_dir_clean(rootdir)
                case 2:
                    mgm = "NM-I"
                    batch_inference(
                        rootdir,
                        dlc_config_path,
                        mark_gen_mode=mgm,
                        crop=CROPPING,
                        mask=MASKING,
                        grayscale=GRAYSCALING,
                        infer_as_video=False,
                        batch_size=BATCH,
                        detector_batch_size=DT_BATCH
                    )
                case 3:
                    batch_track_fix(
                        rootdir,
                        auto_load_weight=True,
                        inference_only=True,
                        lock_id=True,
                        force_locked_id=0,
                    )
                    batch_extract_track_fix_info(rootdir)
                case 4: batch_data_clean(rootdir, kp_clean=True, inst_clean=True)
                case 5: batch_temp_dir_clean(rootdir)
                case 6: batch_convert_csv_to_h5(dlc_config_path)
                case 7: batch_migration_pkl_to_joblib(rootdir)
                case 8: batch_backup_project(rootdir)
                case 9: batch_create_workspaces(rootdir, dlc_config_path)
                case 10: batch_extract_track_fix_info(rootdir)
                case 11: batch_convert_to_grayscale(dlc_config_path)
                case 12: batch_export_csv(rootdir, with_conf=True, no_scorer_header=True, animal_num_filtering=True, min_animal_num=2)
                case 13: batch_export_to_asoid_inference(rootdir, catalogue_file, behav_map_file)