import os
import shutil
import yaml
import numpy as np

def backup_existing_prediction(save_filepath):
    if not os.path.isfile(save_filepath):
        return

    filename = os.path.basename(save_filepath)
    path = os.path.dirname(save_filepath)
    file, ext = filename.split(".")
    backup_idx = 0
    backup_dir = os.path.join(path, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_filepath = os.path.join(backup_dir, f"{file}_backup{backup_idx}.{ext}")

    while os.path.isfile(backup_filepath):
        backup_idx += 1
        backup_filepath = os.path.join(backup_dir, f"{file}_backup{backup_idx}.{ext}")

    shutil.copy(save_filepath, backup_filepath)

def determine_save_path(prediction_filepath:str, suffix:str) -> str:
    pred_file_dir = os.path.dirname(prediction_filepath)
    pred_file_name_without_ext = os.path.splitext(os.path.basename(prediction_filepath))[0]
    
    if not suffix in pred_file_name_without_ext:
        save_idx = 0
        base_name = pred_file_name_without_ext
    else:
        base_name, save_idx_str = pred_file_name_without_ext.split(suffix)
        try:
            save_idx = int(save_idx_str) + 1
        except ValueError:
            save_idx = 0 # Fallback if suffix is malformed
    
    pred_file_to_save_path = os.path.join(pred_file_dir,f"{base_name}{suffix}{save_idx}.h5")

    shutil.copy(prediction_filepath, pred_file_to_save_path)
    print(f"Saved modified prediction to: {pred_file_to_save_path}")
    return pred_file_to_save_path
    
def append_new_video_to_dlc_config(config_path, video_name):
    dlc_dir = os.path.dirname(config_path)
    config_backup = os.path.join(dlc_dir, "config_bak.yaml")
    print("Backup up the original config.yaml as config_bak.yaml")
    shutil.copy(config_path ,config_backup)

    video_filepath = os.path.join(dlc_dir, "videos", f"{video_name}.mp4")

    # Load original config
    with open(config_path, 'r') as f:
        try:
            config_org = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    if video_filepath in config_org["video_sets"]:
        print(f"Video {video_filepath} already exists in video_sets. Skipping update.")
        return

    with open(config_path, 'r') as f:
        config_org["video_sets"][video_filepath] = {"crop": "0, 0, 0, 0"}
        print("Appended new video_sets to the originals.")
    with open(config_path, 'w') as file:
        yaml.dump(config_org, file, default_flow_style=False, sort_keys=False)
        print(f"DeepLabCut config in {config_path} has been updated.")

def add_mock_confidence_score(array:np.ndarray) -> np.ndarray:
    array_dim = len(array.shape) # Always check for dimension first
    if array_dim not in (2, 3):
        raise ValueError("Input array must be 2D or 3D.")

    if array_dim == 2:
        rows, cols = array.shape
        new_array = np.full((rows, cols // 2 * 3), np.nan)
        new_array[:,0::3] = array[:,0::2]
        new_array[:,1::3] = array[:,1::2]

        x_nan_mask = np.isnan(new_array[:, 0::3])
        y_nan_mask = np.isnan(new_array[:, 1::3])
        xy_not_nan_mask = ~(x_nan_mask | y_nan_mask)
        new_array[:, 2::3][xy_not_nan_mask] = 1.0

    if array_dim == 3: # Unflattened (frame_idx, instance, bodyparts)
        dim_1, dim_2, dim_3 = array.shape
        new_array = np.full((dim_1, dim_2, dim_3 // 2 * 3), np.nan)
        new_array[:,:,0::3] = array[:,:,0::2]
        new_array[:,:,1::3] = array[:,:,1::2]

        x_nan_mask = np.isnan(new_array[:, :, 0::3])
        y_nan_mask = np.isnan(new_array[:, :, 1::3])
        xy_not_nan_mask = ~(x_nan_mask | y_nan_mask)
        new_array[:, :, 2::3][xy_not_nan_mask] = 1.0

    return new_array

def unflatten_data_array(array:np.ndarray, inst_count:int) -> np.ndarray:
    rows, cols = array.shape
    new_array = np.full((rows, inst_count, cols // inst_count), np.nan)

    for inst_idx in range(inst_count):
        start_col = inst_idx * cols // inst_count
        end_col = (inst_idx + 1) * cols // inst_count
        new_array[:, inst_idx, :] = array[:, start_col:end_col]
    return new_array

def remove_confidence_score(array:np.ndarray):
    array_dim = len(array.shape) # Always check for dimension first
    if array_dim == 2:
        rows, cols = array.shape
        new_array = np.full((rows, cols // 3 * 2), np.nan)
        new_array[:,0::2] = array[:,0::3]
        new_array[:,1::2] = array[:,1::3]
    if array_dim == 3: # Unflattened (frame_idx, instance, bodyparts)
        dim_1, dim_2, dim_3 = array.shape
        new_array = np.full((dim_1, dim_2, dim_3 // 3 * 2), np.nan)
        new_array[:,:,0::2] = array[:,:,0::3]
        new_array[:,:,1::2] = array[:,:,1::3]
    return new_array