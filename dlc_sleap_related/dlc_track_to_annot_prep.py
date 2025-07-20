import os

import h5py

import pandas as pd
import numpy as np

#################   W   ##################   I   ##################   P   ##################   

def prediction_loader(prediction_file, instance_count, num_keypoints):
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
        return pred_data_values, pred_data_array

def check_instance_count_per_frame(pred_data_array):
    nan_mask = np.isnan(pred_data_array)
    empty_instance = np.all(nan_mask, axis=2)
    non_empty_instance_numerical = (~empty_instance)*1
    instance_count_per_frame = non_empty_instance_numerical.sum(axis=1)
    return instance_count_per_frame

def check_confidence_per_frame(pred_data_values):
    confidence_all = pred_data_values[:,2::3]