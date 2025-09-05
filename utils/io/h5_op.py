import os
import numpy as np
import h5py

from typing import Tuple
import traceback

from .parser import convert_prediction_array_to_save_format
from .csv_op import prediction_to_csv, csv_to_h5
from core.dataclass import Loaded_DLC_Data, Export_Settings

def save_prediction_to_existing_h5(
        prediction_filepath:str,
        pred_data_array:np.ndarray,
        keypoints:list=None,
        multi_animal:bool=False,
        ) -> Tuple[bool, str]:
    try:
        with h5py.File(prediction_filepath, "a") as pred_file:
            key, subkey = validate_h5_keys(pred_file)
            if not key or not subkey:
                return False, f"Error: No valid key in the {prediction_filepath}."
            if subkey == "table":
                pred_file[key][subkey][...] = convert_prediction_array_to_save_format(pred_data_array)
            else:
                F = pred_data_array.shape[0]
                pred_file[key][subkey][...] = pred_data_array.reshape(F, -1)
                if keypoints:
                    pred_file = fix_h5_key_order_on_save(pred_file, key, multi_animal, keypoints)

        return True, f"Successfully saved prediction to {prediction_filepath}."
    except Exception as e:
        print(f"Error saving prediction to HDF5: {e}")
        traceback.print_exc()
        return False, e
    
def save_predictions_to_new_h5(dlc_data:Loaded_DLC_Data, pred_data_array:np.ndarray, export_settings:Export_Settings):
    export_settings.export_mode = "CSV"
    try:
        prediction_to_csv(
            dlc_data=dlc_data,
            pred_data_array=pred_data_array,
            export_settings=export_settings
            )
        prediction_filename = os.path.basename(dlc_data.prediction_filepath).split(".")[0]
        csv_to_h5(
            project_dir=export_settings.save_path,
            multi_animal=dlc_data.multi_animal,
            scorer=dlc_data.scorer,
            csv_name=prediction_filename
            )
        return True, "Prediction successfully saved to new HDF5 file!"
    except Exception as e:
        print(f"Error saving prediction to HDF5: {e}")
        traceback.print_exc()
        return False, e
    
def validate_h5_keys(pred_file:dict):
    pred_header = ["tracks", "df_with_missing", "predictions", "keypoints"]
    found_keys = [key for key in pred_header if key in pred_file]
    if not found_keys:
        return None, None
    key = found_keys[-1]

    subkey_header = ["block0_values", "table"]
    found_subkeys = [subkey for subkey in subkey_header if subkey in pred_file[key]]
    if not found_subkeys:
        return key, None
    subkey = found_subkeys[-1]

    return key, subkey

def fix_h5_kp_order(pred_file: dict, key: str, multi_animal: bool, keypoints: list):
    data = np.array(pred_file[key]["block0_values"])
    n_cols = data.shape[1]

    ref_key = "axis0_level2" if multi_animal else "axis0_level1"
    ref_list = [kp.decode('utf-8') if isinstance(kp, bytes) else kp 
                for kp in pred_file[key][ref_key]]

    for kp in ref_list:
        if kp not in keypoints:
            print(f"Keypoint '{kp}' from prediction file not found in provided keypoints list.")
            return data

    label_key = "axis0_label2" if multi_animal else "axis0_label1"
    label_array = np.array(pred_file[key][label_key])  # shape: (n_cols,), int indices

    n_keypoints_expected = len(keypoints)
    n_coords_per_kp = 2

    if n_cols % (n_keypoints_expected * n_coords_per_kp) != 0:
        n_coords_per_kp = 3
        if n_cols % (n_keypoints_expected * n_coords_per_kp) != 0:
            raise ValueError("Data shape does not match expected number of keypoints.")
    
    n_animals = n_cols // (n_keypoints_expected * n_coords_per_kp)

    # Create mapping: new order index -> original keypoint index
    try:
        remap_kp_indices = [ref_list.index(kp) for kp in keypoints]  # len = n_keypoints
    except ValueError as e:
        raise ValueError(f"Missing keypoint in reference list: {e}")

    # Prepare output array
    data_reordered = np.empty_like(data)

    # For each animal and each keypoint, copy x and y columns
    for animal_idx in range(n_animals):
        animal_slice = slice(animal_idx * n_keypoints_expected * 2,
                             (animal_idx + 1) * n_keypoints_expected * 2)

        for new_idx, orig_kp_idx in enumerate(remap_kp_indices):
            old_x_col = np.where(label_array[animal_slice] == orig_kp_idx)[0][0] + animal_slice.start
            old_y_col = np.where(label_array[animal_slice] == orig_kp_idx)[0][1] + animal_slice.start

            new_x_col = animal_idx * n_keypoints_expected * 2 + new_idx * 2
            new_y_col = new_x_col + 1

            data_reordered[:, new_x_col] = data[:, old_x_col]
            data_reordered[:, new_y_col] = data[:, old_y_col]

    return data_reordered

def fix_h5_frame_order(pred_file):
    # 2 B Implemented
    pass

def fix_h5_key_order_on_save(pred_file, key: str, multi_animal: bool, keypoints: list):
    target_key = "axis0_label2" if multi_animal else "axis0_label1"
    ref_key = "axis0_level2" if multi_animal else "axis0_level1"

    ref_list = [kp.decode('utf-8') if isinstance(kp, bytes) else kp 
                for kp in pred_file[key][ref_key]]
    
    label_array = np.array(pred_file[key][target_key])
    n_cols = label_array.shape[0]
    n_kp = len(keypoints)

    if multi_animal:
        n_animals = n_cols // (n_kp * 2)
    else:
        n_animals = 1

    # Build mapping: desired keypoint -> original index in ref_list
    try:
        remap = [ref_list.index(kp) for kp in keypoints]  # new order → old index
    except ValueError as e:
        raise ValueError(f"Keypoint not found in reference list: {e}")

    # Update label_array in place
    for animal_idx in range(n_animals):
        start_col = animal_idx * n_kp * 2
        for new_kp_idx, orig_kp_idx in enumerate(remap):
            label_array[start_col + new_kp_idx * 2]     = orig_kp_idx  # x
            label_array[start_col + new_kp_idx * 2 + 1] = orig_kp_idx  # y

    # Write back to file
    pred_file[key][target_key][...] = label_array
    return pred_file