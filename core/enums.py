from enum import Enum, auto


class PlaneType(Enum):
    Undefined = -1
    Axial = auto()
    Sagittal = auto()
    Coronal = auto()
    AxialOblique = auto()
    SagittalOblique = auto()
    CoronalOblique = auto()
    VR = auto()
    StretchedCPR = auto()
    StraightenedCPR = auto()


class ShadingType(Enum):
    Undefined = -1,
    Average = auto()
    MIP = auto()
    MinIP = auto()


if __name__ == '__main__':
    pass

