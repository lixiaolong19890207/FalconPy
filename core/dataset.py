import math
import SimpleITK as sitk

from common.stopwatch import stopwatch
from core.defines import RGBA
from core.direction import Direction3
from core.point import Point3


class Datasets:
    def __init__(self):
        self.volume = None
        self.label_to_mask = {}
        self.active_label = 0
        self.is_volume_inverted = False

        self.slice_thickness = 1.0
        self.dims = Point3(0.0, 0.0, 0.0)
        self.spacing = Point3(0.0, 0.0, 0.0)
        self.origin = Point3(0.0, 0.0, 0.0)

        self.dir_x = Direction3(1, 0, 0)
        self.dir_y = Direction3(0, 1, 0)
        self.dir_z = Direction3(0, 0, -1)

    def clear(self):
        del self.volume
        del self.label_to_mask
        self.volume = None
        self.label_to_mask = {}

        self.is_volume_inverted = False

    @stopwatch(__file__)
    def load_volume(self, vol_file):
        itk_img = sitk.ReadImage(vol_file)
        direction = itk_img.GetDirection()
        self.dims = Point3(*itk_img.GetDimension())
        self.dir_x = Direction3(*direction[0:3])
        self.dir_y = Direction3(*direction[3:6])
        self.dir_z = Direction3(*direction[6:9])

        self.spacing = Point3(*itk_img.GetSpacing())
        self.origin = Point3(*itk_img.GetOrigin())
        self.volume = sitk.GetArrayFromImage(itk_img)

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

