from typing import Optional, Union

from pydantic import BaseModel
from sympy import Point3D


class RGB(BaseModel):
    red: Union[int, float] = 0.0
    green: Union[int, float] = 0.0
    blue: Union[int, float] = 0.0


class RGBA(RGB):
    alpha: Union[int, float] = 0.0


class Border(BaseModel):
    left: Union[int, float] = -1
    right: Union[int, float] = -1
    front: Union[int, float] = -1
    back: Union[int, float] = -1
    top: Union[int, float] = -1
    bottom: Union[int, float] = -1


class WWWL(BaseModel):
    ww: Union[int, float] = 0.0
    wl: Union[int, float] = 0.0
    alpha: Union[int, float] = 0.0


class Point3D(list):
    pass


if __name__ == '__main__':
    c1 = RGB()
    c2 = RGBA()

