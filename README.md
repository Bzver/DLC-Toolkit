# DLC-Toolkits
GUI tools to view and manually correct DeepLabCut pose estimation, among other things.

## Why This Toolkit? 🤔
DeepLabCut, alongside its napari integration, offers powerful pose estimation. However, while napari excels as a general image viewer, I've found its performance can crumble when dealing with predictions containing 4000+ frames, leading to frustrating lags and unresponsiveness during tasks like manual frame extraction and track refinement. This isn't a critique of napari or the fantastic DeepLabCut developers, but rather a recognition that highly demanding, specialized workflows require tailored optimization. This toolkit, built with PySide6, hopes to addresses these performance bottlenecks, and to offer a significantly faster, more stable, and smoother user experience for navigating, correcting, and refining large-scale behavioral video data, ensuring that even the most extensive DeepLabCut projects can be managed efficiently without enduring the common pain of performance limitations.

## Important Note on Development ⚠️
This project is currently in active development, which means you might encounter changes or unexpected bugs. I strongly recommend backing up your original DeepLabCut project files and data before using these tools. If you unfortunately encounter one of these bugs, please open an issue here so that your insights can help refine this toolkit for everyone!

**Table of Contents**

- [dlc_3D_skeleton_plotter.py](#dlc_3d_skeleton_plotterpy)
- [dlc_dataset_augumenter.py](#dlc_dataset_augumenterpy)

- [dlc_manual_frame_extract.py](#dlc_manual_frame_extractpy)
- [dlc_obsolete_train_img_trimmer.py](#dlc_obsolete_train_img_trimmerpy)
- [dlc_track_refiner.py](#dlc_track_refinerpy)
- [sleap_keypoint_fill.py](#sleap_keypoint_fillpy)
- [sleap_viewer.ipynb](#sleap_vieweripynb)
- [dlc_h5_to_csv.py](#dlc_h5_to_csvpy)
- [dlc_track_to_annot_prep.py](#dlc_track_to_annot_preppy)


### dlc_3D_skeleton_plotter.py

#### Functionality
A GUI application for visualizing 3D skeleton plots from multiple DeepLabCut predictions ( currently support 4 views only ), synchronized with corresponding 2D video feeds from multiple cameras. Key features include:
- **Multi-camera 2D video display:** Shows synchronized video frames from up to 4 cameras.
- **3D skeleton visualization:** Plots the 3D reconstructed skeleton based on 2D predictions and camera calibrations.
- **Track refinement integration:** Can launch the `DLC_Track_Refiner` tool for selected camera views to correct 2D tracks.

#### Required Folder Structure
The script expects a specific folder structure for videos and prediction files, especially for multi-camera setups.
```
your_video_folder/
├── Camera1/
│   ├── 0.mp4
│   └── your_prediction_file.h5 (e.g., 'video_nameDLC_resnet50_modelname.h5')
├── Camera2/
│   ├── 0.mp4
│   └── your_prediction_file.h5
└── ...
```

**Inputs:**
- **DLC Config File (`config.yaml`):** A DeepLabCut project configuration file, which defines body parts, skeleton, and multi-animal settings.
- **Calibration File (`.mat`):** A MATLAB `.mat` file containing camera calibration parameters (e.g., `sync_dannce.mat`) ( produced in Label3D / DANNCE / SDANNCE workflow ). This file should contain camera intrinsic and extrinsic parameters.
- **Video Folder:** A directory containing subfolders for each camera (e.g., `Camera1`, `Camera2`), where each subfolder contains the video file (`0.mp4`) and its corresponding DLC prediction `.h5` file.

### dlc_dataset_augumenter.py

#### Functionality
A tool for augmenting DLC datasets by adjusting the tone (brightness, contrast, L/A/B channels) of images to match a reference image or a set of reference images. Features an interactive GUI for manual adjustment and an auto-apply option for batch processing. Useful for improving model robustness when training data comes from different lighting conditions or camera settings.

#### Required Folder Structure
The script expects the following structure for DLC labeled data and reference images:

```
your_dlc_project/
└── labeled-data/
    └── your_video_name/
        ├── img0000.png
        ├── img0001.png
        └── ...
        └── CollectedData_YourName.h5 (or .csv)

your_reference_folder/
└── frame_0000.png
└── frame_0001.png
└── ...
```

**Inputs:**
- **DLC Dataset Folder:** The `labeled-data` directory within a DLC project, containing subfolders for each video with labeled images (e.g., `imgXXXX.png`) and their corresponding `CollectedData_*.h5` or `.csv` files.
- **Reference Folder(s):** Directory/directories containing images whose tone you want to match. These images are used as a reference for tone adjustment.

**Outputs:**
- **Augmented Images:** A new folder named `your_video_name_augmented` will be created alongside the original `your_video_name` folder, containing the tone-adjusted images.

### dlc_manual_frame_extract.py

#### Functionality
A GUI application for manually extracting and marking frames from a video, primarily for DLC labeling purposes. It allows users to:
- **Load videos and DLC predictions:** Visualize predictions overlaid on video frames.
- **Mark/unmark frames:** Easily select frames for further labeling or analysis.
- **Navigate frames:** Step through frames, jump to marked frames, and autoplays through the frames.
- **Adjust confidence cutoff:** Filter out low-confidence keypoints from prediction view.
- **Save marked frames:** Export a list of marked frames to a YAML file.
- **Save to DLC format:** Extract marked frames as images and convert prediction data to a DLC-compatible HDF5 format for labeling.

#### Required Folder Structure
For saving marked frames and predictions in a DLC-compatible format, the script interacts with the standard DLC project structure.

```
your_dlc_project/
└── labeled-data/
    └── your_video_name/
        ├── img0000.png (extracted frames)
        └── CollectedData_YourName.h5 (generated prediction data for labeling)
```

**Inputs:**
- **Video File:** The video file from which frames will be extracted.
- **Prediction File (`.h5`):** (Optional) A DeepLabCut prediction `.h5` file to overlay pose estimations on the video.
- **DLC Config File (`config.yaml`):** (Optional) A DeepLabCut project configuration file to display keypoint labels and skeletons.
- **Marked Frames File (`.yaml`):** (Optional) A YAML file containing a previously saved list of marked frames.

**Outputs:**
- **Marked Frames File (`_frame_list.yaml`):** A YAML file containing the list of manually marked frame indices. ( When using Ctrl+S for manual saving)
- **Extracted Images:** If "Save to DLC" is used, selected frames are extracted as `.png` images into the `labeled-data/your_video_name/` directory within the specified DLC project.
- **DLC-compatible HDF5:** If "Save to DLC" is used, a `CollectedData_*.h5` file is generated in the `labeled-data/your_video_name/` directory, containing the prediction data for the extracted frames, ready for DLC labeling.

### dlc_obsolete_train_img_trimmer.py

#### Functionality
This script helps manage DLC training datasets by removing image files from a `labeled-data` directory that are no longer referenced in the corresponding `CollectedData_*.csv` file. This is useful in synergy with frame_extractor to remove the frames you extracted but don't feel like labeling later.

#### Required Folder Structure
The script operates directly on the `labeled-data` subfolder within a DLC project.

```
your_dlc_project/
└── labeled-data/
    └── your_video_name/
        ├── img0000.png
        ├── img0001.png
        └── ...
        └── CollectedData_YourName.csv
```

**Inputs:**
- **CSV File (`CollectedData_*.csv`):** The DeepLabCut CSV file that contains the list of currently labeled frames.
- **Project Directory:** The specific `labeled-data/your_video_name/` directory where the images and CSV file are located.

**Outputs:**
- **Cleaned Image Directory:** Obsolete `.png` image files (those not listed in the CSV) are deleted from the specified project directory.

### dlc_track_refiner.py

#### Functionality
A comprehensive GUI tool for refining DLC tracking data. It allows users to manually correct, interpolate, delete, and swap tracks for individual instances (animals) across video frames. Key features include:
- **Interactive video and prediction display:** Overlays DLC predictions (keypoints, skeletons, bounding boxes) on video frames.
- **Manual keypoint and bounding box editing:** Users can drag and drop keypoints and entire bounding boxes to correct mislabeled or inaccurate predictions.
- **Track manipulation:**
    - **Swap Track:** Exchange the identities of two instances for a single frame or a batch of frames.
    - **Delete Track:** Remove an instance's track data for a single frame or a batch of frames.
    - **Interpolate Track:** Fill in missing or inaccurate track data between two valid points.
    - **Retroactive Fill:** Propagate the last valid keypoint position forward to fill gaps.
- **Advanced refinement options:**
    - **Purge Instance by Confidence:** Delete all tracks below a set confidence threshold.
    - **Interpolate All Frames for One Instance:** Apply interpolation across all frames for a selected instance.
    - **Remove All Prediction Inside Area:** Define a region of interest to remove all predictions within it.
    - **Segmental Auto Correct:** (Experimental) Automated correction for segments of tracks in a specific setup.
- **Undo/Redo functionality:** Allows users to revert or reapply changes.
- **Save refined predictions:** Save the modified prediction data back to a new `.h5` file.

**Inputs:**
- **Video File:** The video file to be refined.
- **DLC Config File (`config.yaml`):** A DeepLabCut project configuration file, essential for defining body parts, skeleton, and multi-animal settings.
- **Prediction File (`.h5`):** The DeepLabCut prediction `.h5` file containing the tracking data to be refined.

**Outputs:**
- **Refined Prediction File (`_refiner_modified_*.h5`):** A new `.h5` file is generated with the refined tracking data. The original prediction file is preserved.

### sleap_keypoint_fill.py

#### Functionality
This script is designed to fill in missing keypoints (NaN values) in SLEAP HDF5 files. It can duplicate and extend keypoints based on two main schemes:
- **Tail Extension:** Extends keypoints along a body vector (e.g., from head to tail) to fill in missing tail points. If a head-to-tail vector is not provided, it uses a random direction.
- **Limb Extension:** Extends keypoints from a source point using a small random vector, useful for filling in missing limb points.

This is particularly useful for pre-processing SLEAP data to ensure complete keypoint sets.

**Inputs:**
- **SLEAP File (`.slp`):** The SLEAP HDF5 file containing the keypoint data to be processed.
- **`num_bodypart` (integer):** The total number of body parts defined in the SLEAP project.
- **`head_tail` (tuple, optional):** A tuple `(head_index, tail_index)` (1-based) used to define the body vector for tail extension.
- **`tail_scheme` (list of tuples, optional):** A list of `(source_index, new_index)` tuples (1-based) for tail keypoint duplication.
- **`limb_scheme` (list of tuples, optional):** A list of `(source_index, new_index)` tuples (1-based) for limb keypoint duplication.

**Outputs:**
- **Modified SLEAP File:** The input `.slp` file is modified in-place with the filled keypoint data.

### sleap_viewer.ipynb

A basic viewer to load and inspect data from a SLEAP HDF5 file (`.slp`). 
- List the keys (datasets) available within the SLEAP HDF5 file.
- Access and print information about the `instances` dataset, including its fields and a few entries.
- Access and parse the `tracks_json` dataset to understand the mapping between internal track indices and user-defined track identifiers.
- Load the `points` dataset and map individual keypoint instances to their respective tracks and frames.
- Print an example of tracked points by identity and frame, showing how to retrieve specific keypoint coordinates.

### dlc_h5_to_csv.py

A simple utility script borrowed from DLC repo that converts DLC prediction `.h5` files to `.csv` format leveraging the `analyze_videos_converth5_to_csv` function within deeplabcut.


### dlc_track_to_annot_prep.py

WIP
