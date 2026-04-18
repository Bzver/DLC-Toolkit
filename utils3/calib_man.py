import numpy as np
import scipy.io as sio

from .triangulation import get_projection_matrix


class Calib_Manager:
    def __init__(self, calib_filepath:str):
        self.calib_filepath = calib_filepath

        if not self.calib_filepath.endswith(".mat"):
            raise NotImplemented("Only support DANNCE/Label 3D Calibration for now.")

        calib = sio.loadmat(calib_filepath)
        self.num_cam = calib["params"].size
        self.cam_params = [{} for _ in range(self.num_cam)]

        self._parse_calib_mat(calib)

    def _parse_calib_mat(self, calib):
        cam_pos = [None] * self.num_cam
        cam_dir = [None] * self.num_cam
        frame_count = [None] * self.num_cam
        for i in range(self.num_cam):
            self.cam_params[i]["RDistort"] = calib["params"][i,0][0,0]["RDistort"][0]
            self.cam_params[i]["TDistort"] = calib["params"][i,0][0,0]["TDistort"][0]
            K = calib["params"][i,0][0,0]["K"].T
            r = calib["params"][i,0][0,0]["r"].T
            t = calib["params"][i,0][0,0]["t"].flatten()
            cam_pos[i] = -np.dot(r.T, t)
            cam_dir[i] = r[2, :]
            self.cam_params[i]["K"] = K
            self.cam_params[i]["P"] = get_projection_matrix(K,r,t)
            frame_count[i] = len(calib["sync"][i,0][0,0]["data_sampleID"][0])
        self.cam_pos = np.array(cam_pos)
        self.cam_dir = np.array(cam_dir)