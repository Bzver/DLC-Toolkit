import deeplabcut
import sys

sys.setrecursionlimit(10000)

config_path = "D:/Project/DLC-Models/NTD/config.yaml"
videofile_path = "D:/Project/DLC-Models/NTD/videos/jobs/20250626C1D.mp4"

deeplabcut.stitch_tracklets(
    config_path,
    [f'{videofile_path}'],
    videotype='mp4',
    shuffle=1,
    trainingsetindex=0,
)