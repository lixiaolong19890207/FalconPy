from typing import Dict

import SimpleITK as sitk
import numpy as np

from common.stopwatch import stopwatch
from core.dataset import Datasets
from core.defines import RGBA


class Render:
    def __init__(self):
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0

        self.rotate_matrix = np.identity(3)
        self.t_rotate_matrix = np.identity(3)
        self.transform_matrix = np.identity(3)
        self.t_transform_matrix = np.identity(3)

        self.datasets = Datasets()

    @stopwatch(__file__)
    def load_volume(self, vol_file):
        self.datasets.load_volume(vol_file)

        transfer_func = {
            5: RGBA(0.8, 0.8, 0.8, 0),
            90: RGBA(0.8, 0.8, 0.8, 0),
        }

    def set_vr_ww_wl(self, ww: float, wl: float, label: str = None):
        pass

    def set_transfer_function(self, func: Dict[int, RGBA]):
        pass

    def rotate(self, rotate_x: float, rotate_y: float):
        pass

    def zoom(self, ratio: float):
        self.scale *= ratio

    def pan(self, pan_x: float, pan_y: float):
        self.offset_x += pan_x
        self.offset_y += pan_y

    def cpy_transfer_to_device(self):
        pass

