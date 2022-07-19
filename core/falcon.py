from typing import Dict, List

# from core.defines import RGBA
from pydantic.color import RGBA

from core.enums import PlaneType, ShadingType


class Falcon:
    def __init__(self):
        pass

    def load_volume(self, vol_file, width, height, depth):
        pass
        return self

    def set_direction(self, x, y, z):
        pass
        return self

    def set_spacing(self, x, y, z):
        pass
        return self

    def set_origin(self, x, y, z):
        pass
        return self

    def reset(self):
        pass

    def set_transfer_function(self, func: Dict[int, RGBA]):
        pass

    def set_background_color(self, clr: RGBA):
        pass

    def get_plane_data(self, plane_type: PlaneType):
        pass

    def get_plane_b64png(self, plane_type: PlaneType):
        pass

    def get_origin_b64png(self, z: int):
        pass

    def get_vr_data(self, width: int, height: int):
        pass

    def get_vr_b64png(self, width: int, height: int):
        pass

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

    def set_vr_wwwl(self, ww: float, wl: float, label: str = None):
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
