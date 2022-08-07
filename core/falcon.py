import ctypes

import SimpleITK as sitk
import pyfpng as fpng
import numpy as np
from typing import Dict
from PIL import Image

from common.stopwatch import stopwatch
from core.defines import RGBA, BoundingBox
from core.direction import Direction3
from core.enums import PlaneType, ShadingType
from core.gpu import Kernel
from core.point import Point3
from core.transfer_func import TransferFunc


class Falcon:
    def __init__(self):
        self.dir_z = None
        self.dir_y = None
        self.dir_x = None
        self.spacing = None
        self.origin = None
        self.depth = None
        self.dimension = None
        self.array_img = None
        self.transfer_func = None
        self.background_color = RGBA()

        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0

        self.bounding_box = BoundingBox()
        self.kernel = Kernel()

    def load_volume(self, vol_file):
        itk_img = sitk.ReadImage(vol_file)
        direction = itk_img.GetDirection()
        self.dir_x = Direction3(*direction[0:3])
        self.dir_y = Direction3(*direction[3:6])
        self.dir_z = Direction3(*direction[6:9])

        self.spacing = itk_img.GetSpacing()
        self.depth = itk_img.GetDepth()
        self.array_img = sitk.GetArrayFromImage(itk_img)
        self.dimension = Point3(*itk_img.GetSize())

        self.transfer_func = {
            5: RGBA(red=0.8, green=0.8, blue=0.8, alpha=0),
            90: RGBA(red=0.8, green=0.8, blue=0.8, alpha=0),
        }
        self.kernel.copy_volume(self.array_img)
        return self

    def set_direction(self, dir_x, dir_y, dir_z):
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z
        return self

    def set_spacing(self, x, y, z):
        self.spacing = [x, y, z]
        self.kernel.reset(x, y, z)
        return self

    def set_bounding_box(self, box: BoundingBox):
        self.bounding_box.xMin = box.xMin
        self.bounding_box.xMax = box.xMax
        self.bounding_box.yMin = box.yMin
        self.bounding_box.yMax = box.yMax
        self.bounding_box.zMin = box.zMin
        self.bounding_box.zMax = box.zMax

    def set_origin(self, x, y, z):
        self.origin = [x, y, z]
        return self

    def reset(self):
        pass

    def set_transfer_function(self, func: Dict[int, RGBA]):
        tf = TransferFunc()
        tf.set_control_pts(func)
        fun = tf.get_transfer_func()

        self.kernel.copy_transfer_function(fun)

    def set_background_color(self, clr: RGBA):
        self.background_color.red = clr.red
        self.background_color.blue = clr.blue
        self.background_color.green = clr.green
        self.background_color.alpha = clr.alpha

    def get_plane_data(self, plane_type: PlaneType):
        pass

    def get_plane_b64png(self, plane_type: PlaneType):
        pass

    def get_origin_b64png(self, z: int):
        pass

    @stopwatch(__file__)
    def get_vr_data(self, width: int, height: int):
        self.bounding_box.xMax = self.dimension[0]
        self.bounding_box.yMax = self.dimension[1]
        self.bounding_box.zMax = self.dimension[2]
        self.kernel.set_bounding_box(self.bounding_box)
        return self.kernel.render(width, height, self.offset_x, self.offset_y, self.scale, ctypes.c_bool(False), self.background_color)

    def get_vr_b64png(self, width: int, height: int):
        data = self.get_vr_data(width, height)
        data = data.reshape(512, 512, 3)
        success, encoded = fpng.encode_image_to_memory(data)
        if not success:
            return None

        return encoded

    def get_plane_idx(self, plane_type: PlaneType):
        pass

    def get_plane_count(self, plane_type: PlaneType):
        pass

    def get_plane_total_matrix(self, plane_type: PlaneType):
        pass

    def front(self):
        pass

    def back(self):
        pass

    def left(self):
        pass

    def right(self):
        pass

    def top(self):
        pass

    def bottom(self):
        pass

    def rotate(self, rotate_x: float, rotate_y: float):
        pass

    def zoom(self, ratio: float):
        pass

    def pan(self, pan_x: float, pan_y: float):
        pass

    def browse(self, delta: float, plane_type: PlaneType):
        pass

    def pan_crosshair(self, x: int, y: int, plane_type: PlaneType):
        pass

    def rotate_crosshair(self, angle: float, plane_type: PlaneType):
        pass

    def get_crosshair(self, plane_type: PlaneType):
        pass

    def get_crosshair_3d(self):
        pass

    def get_direction(self, plane_type: PlaneType):
        pass

    def get_direction_3d(self, plane_type: PlaneType):
        pass

    def get_batch_direction_3d(self, angle: float, plane_type: PlaneType):
        pass

    def set_vr_ww_wl(self, ww: float, wl: float, label: str = None):
        pass

    def set_plane_idx(self, index: int, plane_type: PlaneType):
        pass

    def get_pixel_spacing(self, plane_type: PlaneType):
        pass

    def image_to_voxel(self, x: float, y: float, plane_type: PlaneType):
        pass

    def update_thickness(self, v: float):
        pass

    def set_thickness(self, v: float, plane_type: PlaneType):
        pass

    def get_thickness(self, plane_type: PlaneType):
        pass

    def set_mpr_type(self, t: ShadingType):
        pass

    def set_cpr_line_patient(self, line: list):
        pass

    def set_cpr_line_voxel(self, line: list):
        pass

    def rotate_cpr(self, angle: float, plane_type: PlaneType):
        pass

    def set_segment_alpha(self, alpha: float, label: str = None):
        pass

    def add_segment(self, mask, width: int, height: int, depth: int):
        pass

    def update_segment(self, mask, width: int, height: int, depth: int):
        pass
