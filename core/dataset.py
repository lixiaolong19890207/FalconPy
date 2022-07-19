import math

import pyfpng as fpng
from common.stopwatch import stopwatch
from core.direction import Direction3
from core.point import Point3


class Dataset:
    def __init__(self):
        self.volume = None
        self.mask = None
        self.is_volume_inverted = False

        self.slice_thickness = 1.0
        self.dims = Point3(0.0, 0.0, 0.0)
        self.spacing = Point3(0.0, 0.0, 0.0)
        self.origin = Point3(0.0, 0.0, 0.0)

        self.dir_x = Direction3(1, 0, 0)
        self.dir_y = Direction3(0, 1, 0)
        self.dir_z = Direction3(0, 0, -1)

    def clear(self):
        self.volume = None
        self.mask = None
        del self.volume
        del self.mask

        self.is_volume_inverted = False

    @stopwatch(__file__)
    def load_volume(self, vol_file):
        pass

    def add_mask(self, mask_file):
        pass

    def update_mask(self, mask):
        pass

    def set_direction(self, dir_x: Direction3, dir_y: Direction3, dir_z: Direction3):
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z

    def is_need_invert_z(self):
        dir_norm = self.dir_x.cross(self.dir_y)
        return dir_norm.dot(self.dir_z) > 0

    def is_vertical_coord(self):
        dir_norm = self.dir_x.cross(self.dir_y)
        dot_v = math.fabs(dir_norm.dot(self.dir_z))
        return dot_v >= math.sin(math.pi * 85 / 180)

    @classmethod
    def normalize_mask(cls, mask):
        pass

    def patient2voxel(self, patient_pt: Point3):
        pass

    def voxel2patient(self, voxel_pt: Point3):
        pass
