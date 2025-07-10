import os

import pandas as pd
from itertools import islice

import h5py
import yaml

import cv2
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QtGui

#######################    W    #######################    I    #######################    P    #######################

class DLCOutlierFinder(QtWidgets.QMainWindow):  # GUI for manually select the outliers
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Outlier Finder")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Video display area
        self.video_label = QtWidgets.QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.layout.addWidget(self.video_label)

        # Buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.load_video_button = QtWidgets.QPushButton("Load Video")
        self.load_prediction_button = QtWidgets.QPushButton("Load Prediction")
        self.save_frame_button = QtWidgets.QPushButton("Save Current Frame")

        self.button_layout.addWidget(self.load_video_button)
        self.button_layout.addWidget(self.load_prediction_button)
        self.button_layout.addWidget(self.save_frame_button)
        self.layout.addLayout(self.button_layout)

        # Connect buttons to placeholder methods
        self.load_video_button.clicked.connect(self.load_video)
        self.load_prediction_button.clicked.connect(self.load_prediction)
        self.save_frame_button.clicked.connect(self.save_current_frame)

        self.original_vid = None
        self.prediction = None
        self.frame_list = None
        self.deeplabcut_dir = None
        self.cap = None # Video capture object
        self.current_frame = None # Current frame image
        self.frame_list = []

    def load_video(self):
        file_dialog = QtWidgets.QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)")
        if video_path:
            self.original_vid = video_path
            self.cap = cv2.VideoCapture(self.original_vid)
            if not self.cap.isOpened():
                print(f"Error: Could not open video {self.original_vid}")
                self.video_label.setText("Error: Could not open video")
                self.cap = None
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0
            self.display_current_frame()
            print(f"Video loaded: {self.original_vid}")

    def load_prediction(self):
        file_dialog = QtWidgets.QFileDialog(self)
        prediction_path, _ = file_dialog.getOpenFileName(self, "Load Prediction", "", "HDF5 Files (*.h5);;All Files (*)")
        if prediction_path:
            self.prediction = prediction_path
            print(f"Prediction loaded: {self.prediction}")

    def change_current_frame_status(self):
        if self.current_frame is not None and not self.current_frame in self.frame_list:
            self.frame_list.append(self.current_frame_idx)
        elif self.current_frame in self.frame_list:
            self.frame_list.remove(self.current_frame_idx)

    def display_current_frame(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Convert OpenCV image to QPixmap
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                # Scale pixmap to fit label
                scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
                self.video_label.setText("") # Clear "No video loaded" text
            else:
                self.video_label.setText("Error: Could not read frame")
        else:
            self.video_label.setText("No video loaded")

class DLCOutlierExtractor:  # Backend for extracting outliers for further labelling
    def __init__(self, original_vid, prediction, frame_list, deeplabcut_dir):
        self.original_vid = original_vid
        self.prediction = prediction
        self.frame_list = frame_list
        self.deeplabcut_dir = deeplabcut_dir

        self.project_dir = None
        self.multi_animal = False
        self.scorer = "machine-labeled"
        self.data = []
        self.existing_project = False
        
    def extract_frame_and_label(self):

        if not os.path.isfile(self.original_vid):
            print(f"Original video not found at {self.original_vid}")
            return False
        elif not os.path.isfile(self.prediction):
            print(f"Prediction file not found at {self.prediction}")
            return False
        else:
            video_name = os.path.basename(self.original_vid).split(".")[0]
            video_path = os.path.dirname(self.original_vid)

            if self.deeplabcut_dir is not None:
                self.project_dir = os.path.join(self.deeplabcut_dir,"labeled-data",video_name)
                if os.path.isdir(self.project_dir) and any(os.scandir(self.project_dir)):
                    print(f"Existing project folder found, checking for already labelled image...")
                    self.existing_project = True
                    existing_img = [ int(f.split("img")[1].split(".")[0]) for f in os.listdir(self.project_dir) if f.endswith(".png") and f.startswith("img") ]
                else:
                    existing_img = []
                for frame in self.frame_list:
                    if frame in existing_img:
                        self.frame_list.remove(frame)
                        print(f"Frame {frame} already in the {self.project_dir}, skipping...")
            else:
                self.project_dir = video_path

            os.makedirs(self.project_dir, exist_ok=True)
            frame_extraction = self.extract_frame()
            label_extraction = self.extract_label()

            if frame_extraction and label_extraction:
                print("Extraction successful.")
                return True
            else:
                fail_message = ""
                fail_message += " extract frame" if not frame_extraction else ""
                fail_message += " extract label" if not label_extraction else ""
                print(f"Extraction failed. Error:{fail_message}")
        
    def extract_frame(self):
        """Extract frame pictures from original videos"""
        cap = cv2.VideoCapture(self.original_vid)

        if not cap.isOpened():
            print(f"Error: Could not open video {self.original_vid}")
            return False
        
        frames_to_extract = set(self.frame_list)

        for frame in frames_to_extract:
            image_path = f"img{str(int(frame)).zfill(8)}.png"
            image_output_path = os.path.join(self.project_dir,image_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_output_path, frame)
        return True

    def extract_label(self): # Adapted from DeepLabCuT
        """Extract label from predictions"""
        with h5py.File(self.prediction, "r") as pred_file:
            if not "tracks" in pred_file.keys():
                print("Error: Prediction file not valid, no 'tracks' key found in prediction file.")
                return False
            else:
                if not "table" in pred_file["tracks"].keys():
                    print("Errpr: Prediction file not valid, no prediction table found in 'tracks'.")
                    return False
            for frame in self.frame_list:
                frame_idx = pred_file["tracks"]["table"][frame][0]
                frame_data = pred_file["tracks"]["table"][frame][1]
                filtered_frame_data = [val for i, val in enumerate(frame_data) if (i % 3 != 2)] # Remove likelihood
                self.data.append([frame_idx] + filtered_frame_data)
            if not self.prediction_to_csv():
                print("Error exporting predictions to csv.")
                return False
            else:
                print("Prediction of selected frames successfully exported to csv.")
                if not self.csv_to_h5():
                    print("Error transforming to h5.")
                    return False
                else:
                    print("Exported csv transformed into h5.")
                    return True

    def prediction_to_csv(self):
        """Save the extracted predictions into csv"""
        config = os.path.join(self.deeplabcut_dir,"config.yaml")

        with open(config, "r") as conf:
            cfg = yaml.safe_load(conf)
        self.multi_animal = cfg["multianimalproject"]
        keypoints = cfg["bodyparts"] if not self.multi_animal else cfg["multianimalbodyparts"]
            
        data_frames = []
        columns = ["frame"]
        bodyparts_row = ["bodyparts"]
        coords_row = ["coords"]

        num_keypoints = len(keypoints)
        max_instances = 1

        if not self.multi_animal:
            columns += ([f"{kp}_x" for kp in keypoints] + [f"{kp}_y" for kp in keypoints])
            bodyparts_row += [ f"{kp}" for kp in keypoints for _ in (0, 1) ]
            coords_row += (["x", "y"] * num_keypoints)
        else:
            individuals_row = ["individuals"]
            individuals = cfg["individuals"]
            max_instances = len(individuals)
            individuals = [str(k) for k in range(1,max_instances+1)]
            for m in range(max_instances):
                columns += ([f"{kp}_x" for kp in keypoints] + [f"{kp}_y" for kp in keypoints])
                bodyparts_row += [ f"{kp}" for kp in keypoints for _ in (0, 1) ]
                coords_row += (["x", "y"] * num_keypoints)
                for _ in range(num_keypoints*2):
                    individuals_row += [individuals[m]]
        scorer_row = ["scorer"] + [f"{self.scorer}"] * (len(columns) - 1)

        labels_df = pd.DataFrame(self.data, columns=columns)
        labels_df["frame"] = labels_df["frame"].apply(
            lambda x: (
                f"labeled-data/{video_name}/"
                f"img{str(int(x)).zfill(8)}.png"
            )
        )
        labels_df = labels_df.groupby("frame", as_index=False).first()
        data_frames.append(labels_df)
        combined_df = pd.concat(data_frames, ignore_index=True)

        header_df = pd.DataFrame(
        [row for row in [scorer_row, individuals_row, bodyparts_row, coords_row] if row != individuals_row or self.multi_animal],
            columns=combined_df.columns
        )

        final_df = pd.concat([header_df, combined_df], ignore_index=True)
        final_df.columns = [None] * len(final_df.columns)

        final_df.to_csv(
            os.path.join(self.project_dir, f"MachineLabelsRefine.csv"),
            index=False,
            header=None,
        )
        return True

    def csv_to_h5(self):
        """Export the csv data to h5"""
        try:
            fn = os.path.join(self.project_dir, f"MachineLabelsRefine.csv")
            with open(fn) as datafile:
                head = list(islice(datafile, 0, 5))
            if self.multi_animal:
                header = list(range(4))
            else:
                header = list(range(3))
            if head[-1].split(",")[0] == "labeled-data":
                index_col = [0, 1, 2]
            else:
                index_col = 0
            data = pd.read_csv(fn, index_col=index_col, header=header)
            data.columns = data.columns.set_levels([self.scorer], level="scorer")
            self.guarantee_multiindex_rows(data)
            data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
            data.to_csv(fn)
            return True
        except FileNotFoundError:
            print("Attention:", self.project_dir, "does not appear to have labeled data!")

    def guarantee_multiindex_rows(self, df):
        # Make paths platform-agnostic if they are not already
        if not isinstance(df.index, pd.MultiIndex):  # Backwards compatibility
            path = df.index[0]
            try:
                sep = "/" if "/" in path else "\\"
                splits = tuple(df.index.str.split(sep))
                df.index = pd.MultiIndex.from_tuples(splits)
            except TypeError:  #  Ignore numerical index of frame indices
                pass
        # Ensure folder names are strings
        try:
            df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
        except AttributeError:
            pass

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DLCOutlierFinder()
    window.show()
    app.exec()