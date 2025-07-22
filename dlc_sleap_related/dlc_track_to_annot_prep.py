import h5py
import numpy as np

#################   W   ##################   I   ##################   P   ##################   

def prediction_loader(prediction_file, instance_count, num_keypoints):
    print("a")
    with h5py.File(prediction_file, "r") as pred_file:
        if not "tracks" in pred_file.keys():
            print("Error: Prediction file not valid, no 'tracks' key found in prediction file.")
            return False
        pred_data = pred_file["tracks"]["table"]
        pred_data_values = np.array([item[1] for item in pred_data])
        pred_frame_count = pred_data.size
        pred_data_array = np.full((pred_frame_count, instance_count, num_keypoints*3),np.nan)
        for inst in range(instance_count):
                pred_data_array[:,inst,:] = pred_data_values[:, inst*num_keypoints*3:(inst+1)*num_keypoints*3]
        instance_count_per_frame = check_instance_count_per_frame(pred_data_array)
        average_confidence_per_frame = check_confidence_per_frame(pred_data_values)

def check_instance_count_per_frame(pred_data_array):
    print("b")
    nan_mask = np.isnan(pred_data_array)
    empty_instance = np.all(nan_mask, axis=2)
    non_empty_instance_numerical = (~empty_instance)*1
    instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
    print(instance_count_per_frame)
    return instance_count_per_frame

def check_confidence_per_frame(pred_data_values): # Confidence used when both prediction detected 2 instances at the same time
    print("c")
    confidence_all = pred_data_values[:,2::3]
    average_confidence_per_frame = np.nanmean(confidence_all, axis=1)
    print(average_confidence_per_frame)
    return average_confidence_per_frame

def compare_pred_inst(pred_1, pred_2, config=None):
    print("f")
    prediction_loader(pred_1, 2, 14)
    prediction_loader(pred_2, 2, 14)

if __name__ == "__init__":
    pred_1 = "D:/Project/A-SOID/Data/20250709/20250709-first3h-D-convDLC_HrnetW32_bezver-SD-20250605M-cam52025-06-26shuffle1_detector_090_snapshot_080_el_tr.h5"
    pred_2 = "D:/Project/A-SOID/Data/20250709/20250709-first3h-S-convDLC_HrnetW32_bezver-SD-20250605M-cam52025-06-26shuffle1_detector_090_snapshot_080_el_tr.h5"
    compare_pred_inst(pred_1, pred_2)
