import SimpleITK as sitk
from pathlib import Path

from core.defines import RGBA
from core.direction import Direction3
from core.falcon import Falcon

CUR_PATH = Path(__file__).resolve().parent
DATA_PATH = CUR_PATH / 'data'
LUNG_PATH = DATA_PATH / 'ct_lung.original.nrrd'


def test_load_volumes():
    itk_img = sitk.ReadImage(str(LUNG_PATH))
    direction = itk_img.GetDirection()
    dir_x = Direction3(*direction[0:3])
    dir_y = Direction3(*direction[3:6])
    dir_z = Direction3(*direction[6:9])

    spacing = itk_img.GetSpacing()
    depth = itk_img.GetDepth()
    array_img = sitk.GetArrayFromImage(itk_img)

    transfer_func = {
        5: RGBA(0.8, 0.8, 0.8, 0),
        90: RGBA(0.8, 0.8, 0.8, 0),
    }

    falcon = Falcon()


if __name__ == '__main__':
    test_load_volumes()
