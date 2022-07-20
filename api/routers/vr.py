from fastapi import APIRouter

from core.falcon_mgr import falcon_mgr

router = APIRouter()


@router.get('/rotate')
async def get_image(
    falcon_uid: str,
    x_angle: float,
    y_angle: float,
    width: int = 512,
    height: int = 512
):
    instance = falcon_mgr.get_falcon(falcon_uid)
    assert instance, f'Failed to find falcon by uid: {falcon_uid}'

    instance.rotate(x_angle, y_angle)
    b64 = instance.get_vr_b64png(width, height)

    return {
        'data': {
            'image': b64
        },
        'message': 'successful'
    }


@router.get('/pan')
async def get_image(
    falcon_uid: str,
    x_pan: float,
    y_pan: float,
    width: int = 512,
    height: int = 512
):
    instance = falcon_mgr.get_falcon(falcon_uid)
    assert instance, f'Failed to find falcon by uid: {falcon_uid}'

    instance.pan(x_pan, y_pan)
    b64 = instance.get_vr_b64png(width, height)

    return {
        'data': {
            'image': b64
        },
        'message': 'successful'
    }


@router.get('/zoom')
async def get_image(
    falcon_uid: str,
    delta: float,
    width: int = 512,
    height: int = 512
):
    instance = falcon_mgr.get_falcon(falcon_uid)
    assert instance, f'Failed to find falcon by uid: {falcon_uid}'

    instance.zoom(delta)
    b64 = instance.get_vr_b64png(width, height)

    return {
        'data': {
            'image': b64
        },
        'message': 'successful'
    }