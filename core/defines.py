import ctypes
from typing import Optional, Union

from pydantic import BaseModel
from sympy import Point3D


class RGB(BaseModel):
    red: Union[int, float] = 0.0
    green: Union[int, float] = 0.0
    blue: Union[int, float] = 0.0


class RGBA(RGB):
    alpha: Union[int, float] = 0.0


class BoundingBox(ctypes.Structure):
    _fields_ = [
        ('xMin', ctypes.c_int),
        ('xMax', ctypes.c_int),
        ('yMin', ctypes.c_int),
        ('yMax', ctypes.c_int),
        ('zMin', ctypes.c_int),
        ('zMax', ctypes.c_int),
    ]


class WWWL(BaseModel):
    ww: Union[int, float] = 0.0
    wl: Union[int, float] = 0.0
    alpha: Union[int, float] = 0.0


class Point3D(list):
    pass


if __name__ == '__main__':
    c1 = RGB()
    c2 = RGBA()

