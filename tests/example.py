import SimpleITK as sitk
from pathlib import Path

from core.defines import RGBA
from core.direction import Direction3
from core.falcon import Falcon

CUR_PATH = Path(__file__).resolve().parent
DATA_PATH = CUR_PATH / 'data'
LUNG_PATH = DATA_PATH / 'cardiac.mhd'


def test_load_volumes():
    # itk_img = sitk.ReadImage(str(LUNG_PATH))
    # direction = itk_img.GetDirection()
    # dir_x = Direction3(*direction[0:3])
    # dir_y = Direction3(*direction[3:6])
    # dir_z = Direction3(*direction[6:9])
    #
    # spacing = itk_img.GetSpacing()
    # depth = itk_img.GetDepth()
    # array_img = sitk.GetArrayFromImage(itk_img)
    #
    # transfer_func = {
    #     5: RGBA(0.8, 0.8, 0.8, 0),
    #     90: RGBA(0.8, 0.8, 0.8, 0),
    # }

    falcon = Falcon()
    falcon.load_volume(str(LUNG_PATH))

    pos_to_rgba = {
        0: RGBA(red=0.8, green=0, blue=0, alpha=0),
        10: RGBA(red=0.8, green=0, blue=0, alpha=0.3),
        40: RGBA(red=0.8, green=0.8, blue=0, alpha=0),
        99: RGBA(red=1.0, green=0.8, blue=1.0, alpha=1.0)
    }

    falcon.set_transfer_function(pos_to_rgba)
    b64png = falcon.get_vr_b64png(512, 512)
    pass


if __name__ == '__main__':
    test_load_volumes()
