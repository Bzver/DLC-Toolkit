import numpy as np
import h5py

from typing import Tuple

from .io_helper import convert_prediction_array_to_save_format, remove_confidence_score
from .csv_op import prediction_to_csv, csv_to_h5
from utils.logger import logger
from utils.dataclass import Loaded_DLC_Data


def save_prediction_to_existing_h5(
        prediction_filepath:str,
        pred_data_array:np.ndarray,
        keypoints:list=None,
        multi_animal:bool=False,
        ) -> Tuple[bool, str]:
    
    with h5py.File(prediction_filepath, "a") as pred_file:
        key, subkey = validate_h5_keys(pred_file)
        logger.debug(f"[H5OP] Validated H5 keys: key='{key}', subkey='{subkey}'")
        if not key or not subkey:
            logger.error(f"[H5OP] Error: No valid key or subkey found in {prediction_filepath}.")
            return False, f"Error: No valid key in the {prediction_filepath}."
        if subkey == "table":
            pred_file[key][subkey][...] = convert_prediction_array_to_save_format(pred_data_array)
            logger.debug(f"[H5OP] Saved data to '{key}/{subkey}' as table format.")
        else:
            F = pred_data_array.shape[0]
            logger.debug(f"[H5OP] Data shape for saving: {pred_data_array.shape}")
            try:
                pred_file[key][subkey][...] = pred_data_array.reshape(F, -1)
                logger.debug(f"[H5OP] Saved data to '{key}/{subkey}' as reshaped array.")
            except TypeError as e:
                logger.warning(f"[H5OP] TypeError encountered when saving to H5. Removing confidence scores. Error: {e}")
                pred_data_array = remove_confidence_score(pred_data_array)
                pred_file[key][subkey][...] = pred_data_array.reshape(F, -1)
                logger.debug(f"[H5OP] Saved data to '{key}/{subkey}' after removing confidence scores.")
            if keypoints:
                logger.debug(f"[H5OP] Fixing H5 key order on save with keypoints: {keypoints}")
                pred_file = fix_h5_key_order_on_save(pred_file, key, multi_animal, keypoints)
                logger.debug(f"[H5OP] H5 key order fixed.")
    logger.debug(f"[H5OP] Successfully saved prediction to existing H5: {prediction_filepath}")
    return True, ""
    
def save_predictions_to_new_h5(
        dlc_data:Loaded_DLC_Data,
        pred_data_array:np.ndarray,
        save_path:str,
        frame_list:list|None=None,
        to_dlc:bool=False,
        ):
    logger.debug(f"[H5OP] Attempting to save predictions to new H5. Save path: {save_path}")
    prediction_to_csv(
        dlc_data=dlc_data,
        pred_data_array=pred_data_array,
        frame_list=frame_list,
        save_path=save_path.replace(".h5", ".csv"),
        keep_conf=not to_dlc,
        to_dlc=to_dlc,
        )
    csv_to_h5(
        csv_path=save_path.replace(".h5", ".csv"),
        multi_animal=dlc_data.multi_animal,
        scorer=dlc_data.scorer
        )
    logger.debug(f"[H5OP] CSV converted to new H5 file at {save_path}")
    
def validate_h5_keys(pred_file:dict) -> Tuple[str, str]:
    logger.debug("[H5OP] Validating H5 keys.")
    pred_header = ["tracks", "df_with_missing", "predictions", "keypoints"]
    found_keys = [key for key in pred_header if key in pred_file]
    logger.debug(f"[H5OP] Found potential top-level keys: {found_keys}")
    if not found_keys:
        logger.warning("[H5OP] No valid top-level key found.")
        return None, None
    key = found_keys[-1]
    logger.debug(f"[H5OP] Selected top-level key: '{key}'")

    subkey_header = ["block0_values", "table"]
    found_subkeys = [subkey for subkey in subkey_header if subkey in pred_file[key]]
    logger.debug(f"[H5OP] Found potential subkeys under '{key}': {found_subkeys}")
    if not found_subkeys:
        logger.warning(f"[H5OP] No valid subkey found under '{key}'.")
        return key, None
    subkey = found_subkeys[-1]
    logger.debug(f"[H5OP] Selected subkey: '{subkey}'")

    return key, subkey

def fix_h5_kp_order(pred_file: dict, key: str, config_data:dict) -> np.ndarray:
    logger.debug(f"[H5OP] Fixing H5 keypoint order for key: '{key}'. Multi-animal: {config_data['multi_animal']}")
    data = np.array(pred_file[key]["block0_values"])
    n_cols = data.shape[1]
    multi_animal = config_data["multi_animal"]
    keypoints = config_data["keypoints"]
    num_keypoint = config_data["num_keypoint"]
    instance_count = config_data["instance_count"]
    logger.debug(f"[H5OP] Config data: num_keypoint={num_keypoint}, instance_count={instance_count}")

    ref_key = "axis0_level2" if multi_animal else "axis0_level1"
    ref_list = [kp.decode('utf-8') if isinstance(kp, bytes) else kp
                for kp in pred_file[key][ref_key]]
    logger.debug(f"[H5OP] Reference key list from H5: {ref_list}")
    logger.debug(f"[H5OP] Expected keypoints: {keypoints}")

    for kp in ref_list:
        if kp not in keypoints:
            logger.warning(f"[H5OP] Keypoint '{kp}' from prediction file not found in provided keypoints list. Returning original data.")
            return data

    label_key = "axis0_label2" if multi_animal else "axis0_label1"
    label_array = np.array(pred_file[key][label_key])

    if n_cols % (num_keypoint * 2) != 0 or  n_cols // (num_keypoint * 2) != instance_count:
        return data

    try:
        remap_kp_indices = [ref_list.index(kp) for kp in keypoints]  # len = n_keypoints
        logger.debug(f"[H5OP] Remapped keypoint indices: {remap_kp_indices}")
    except ValueError as e:
        logger.error(f"[H5OP] Missing keypoint in reference list during remapping: {e}")
        raise ValueError(f"Missing keypoint in reference list: {e}")

    data_reordered = np.empty_like(data)
    logger.debug(f"[H5OP] Reordering data for {instance_count} animals.")

    for animal_idx in range(instance_count):
        animal_slice = slice(animal_idx * num_keypoint * 2, (animal_idx + 1) * num_keypoint * 2)
        logger.debug(f"[H5OP] Processing animal index {animal_idx}, slice: {animal_slice}")

        for new_idx, orig_kp_idx in enumerate(remap_kp_indices):
            old_x_col = np.where(label_array[animal_slice] == orig_kp_idx)[0][0] + animal_slice.start
            old_y_col = np.where(label_array[animal_slice] == orig_kp_idx)[0][1] + animal_slice.start

            new_x_col = animal_idx * num_keypoint * 2 + new_idx * 2
            new_y_col = new_x_col + 1
            logger.debug(f"[H5OP]   Keypoint '{keypoints[new_idx]}' (orig_idx:{orig_kp_idx}): old_x={old_x_col}, old_y={old_y_col} -> new_x={new_x_col}, new_y={new_y_col}")

            data_reordered[:, new_x_col] = data[:, old_x_col]
            data_reordered[:, new_y_col] = data[:, old_y_col]

    logger.debug("[H5OP] Finished fixing H5 keypoint order.")
    return data_reordered

def fix_h5_key_order_on_save(pred_file, key: str, multi_animal: bool, keypoints: list) -> dict:
    logger.debug(f"[H5OP] Fixing H5 key order on save for key: '{key}'. Multi-animal: {multi_animal}. Keypoints: {keypoints}")
    target_key_axis = "axis0_label2" if multi_animal else "axis0_label1"
    target_key_block = "block0_items_label2" if multi_animal else "block0_items_label1"
    ref_key = "axis0_level2" if multi_animal else "axis0_level1"

    ref_list = [kp.decode('utf-8') if isinstance(kp, bytes) else kp
                for kp in pred_file[key][ref_key]]
    logger.debug(f"[H5OP] Reference key list from H5 on save: {ref_list}")
    
    label_array_axis = np.array(pred_file[key][target_key_axis])
    label_array_block = np.array(pred_file[key][target_key_block])
    n_cols = label_array_axis.shape[0]
    n_kp = len(keypoints)
    logger.debug(f"[H5OP] Initial label_array_axis on save: {label_array_axis}")
    logger.debug(f"[H5OP] Number of columns: {n_cols}, Number of keypoints: {n_kp}")

    if multi_animal:
        n_animals = n_cols // (n_kp * 2)
    else:
        n_animals = 1
    logger.debug(f"[H5OP] Number of animals detected: {n_animals}")

    try:
        remap = [ref_list.index(kp) for kp in keypoints]  # new order → old index
        logger.debug(f"[H5OP] Remapping indices for save: {remap}")
    except ValueError as e:
        logger.error(f"[H5OP] Keypoint not found in reference list on save: {e}")
        raise ValueError(f"Keypoint not found in reference list: {e}")

    for animal_idx in range(n_animals):
        start_col = animal_idx * n_kp * 2
        logger.debug(f"[H5OP] Processing animal index {animal_idx}, starting column: {start_col}")
        for new_kp_idx, orig_kp_idx in enumerate(remap):
            label_array_axis[start_col + new_kp_idx * 2]     = orig_kp_idx  # x
            label_array_axis[start_col + new_kp_idx * 2 + 1] = orig_kp_idx  # y
            logger.debug(f"[H5OP]   Updating label_array_axis[{start_col + new_kp_idx * 2}] (x) and [{start_col + new_kp_idx * 2 + 1}] (y) to original keypoint index {orig_kp_idx}")

    label_array_block = label_array_axis.copy()
    logger.debug(f"[H5OP] Updated label_array_axis on save: {label_array_axis}")

    pred_file[key][target_key_axis][...] = label_array_axis
    pred_file[key][target_key_block][...] = label_array_block
    logger.debug("[H5OP] Finished fixing H5 key order on save.")
    return pred_file

def get_frame_list_from_h5(filepath):
    with h5py.File(filepath, "a") as pred_file:
        key, subkey = validate_h5_keys(pred_file)
        if not key or not subkey:
            raise ValueError(f"No valid key found in '{filepath}'")
        labeled_frame_list = pred_file[key]["axis1_level2"].asstr()[()]
        labeled_frame_list = [int(f.split("img")[1].split(".")[0]) for f in labeled_frame_list]
        labeled_frame_list.sort()
        return labeled_frame_list