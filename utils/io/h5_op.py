import os
import numpy as np
import h5py

from typing import Tuple
import traceback

from .parser import convert_prediction_array_to_save_format
from .csv_op import prediction_to_csv, csv_to_h5
from core.dataclass import Loaded_DLC_Data, Export_Settings

def save_prediction_to_existing_h5(prediction_filepath: str, pred_data_array: np.ndarray) -> Tuple[bool, str]:
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
    pred_header = ["tracks", "df_with_missing", "predictions"]
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
