import os
import shutil
import yaml

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