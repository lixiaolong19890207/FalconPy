import numpy as np
from pathlib import Path

from helper import cuda, cudart
from core.helper.common import KernelHelper
from core.helper.helper_cuda import findCudaDevice, checkCudaErrors
from core.point import Point3

__cur_path = Path(__file__).resolve().parent
__cuda_path = __cur_path / 'kernel.cu'
with open(__cuda_path, 'r') as f:
    KERNEL_CODE = f.read()

DEVICE_ID = findCudaDevice()


class Kernel:
    def __init__(self):
        self.volume_of_interest = [-1, -1, -1, -1, -1, -1]
        self.width_vr = 0
        self.height_vr = 0
        self.spacing = Point3(0, 0, 0)
        self.normal = Point3(0, 0, 0)
        self.max_per = Point3(0, 0, 0)
        self.dimensions = Point3(0, 0, 0)

        kernel_mods = KernelHelper(KERNEL_CODE, DEVICE_ID)

        self.const_transform_matrix = kernel_mods.getGlobal(b'const_transform_matrix')
        self.volume_text_obj = kernel_mods.getGlobal(b'volume_text_obj')
        self.d_volume_array = kernel_mods.getGlobal(b'd_volume_array')
        # self.d_volume_array = None

        self.__cu_render = kernel_mods.getFunction(b'cu_render')

    def free(self):
        checkCudaErrors(cudart.cudaDestroyTextureObject(self.const_transform_matrix))
        checkCudaErrors(cudart.cudaFreeArray(self.d_volume_array))
        # checkCudaErrors(cudart.cudaFree(cu_3darray))

    def reset(self, spacing_x, spacing_y, spacing_z):
        self.volume_of_interest = [-1, -1, -1, -1, -1, -1]
        self.width_vr = 0
        self.height_vr = 0

        self.spacing.x = spacing_x
        self.spacing.y = spacing_y
        self.spacing.z = spacing_z

        self.normal.x = 1.0 / self.dimensions[0]
        self.normal.y = 1.0 / self.dimensions[1]
        self.normal.z = 1.0 / self.dimensions[2]

        max_spacing = max(spacing_x, max(spacing_y, spacing_z))
        max_length = max(
            volume_size.width * spacing_x,
            max(
                volume_size.height * spacing_y,
                volume_size.depth * spacing_z
            )
        )

        self.max_per.x = 1.0 * max_length / volume_size.width / spacing_x
        self.max_per.y = 1.0 * max_length / volume_size.height / spacing_y
        self.max_per.z = 1.0 * max_length / volume_size.depth / spacing_z

    def copy_volume(self, volume_array):
        shape = volume_array.shape
        volume_size = cudart.make_cudaExtent(shape[0], shape[1], shape[2])
        if self.d_volume_array:
            checkCudaErrors(cudart.cudaFreeArray(self.d_volume_array))
            self.d_volume_array = 0
            self.volume_text_obj = 0

        channel_desc = checkCudaErrors(
            cudart.cudaCreateChannelDesc(32, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat))
        self.d_volume_array = checkCudaErrors(
            cudart.cudaMalloc3DArray(channel_desc, volume_size, cudart.cudaArrayCubemap))

        memcpy_params = cudart.cudaMemcpy3DParms()
        memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
        memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
        memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(
            volume_array,
            width * np.dtype(np.float32).itemsize,
            width,
            width
        )
        memcpy_params.dstArray = self.d_volume_array
        memcpy_params.extent = volume_size
        memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        checkCudaErrors(cudart.cudaMemcpy3D(memcpy_params))

        tex_res = cudart.cudaResourceDesc()
        tex_res.resType = cudart.cudaResourceType.cudaResourceTypeArray
        tex_res.res.array.array = self.d_volume_array

        tex_descr = cudart.cudaTextureDesc()
        tex_descr.normalizedCoords = True
        tex_descr.filterMode = cudart.cudaTextureFilterMode.cudaFilterModeLinear
        tex_descr.addressMode[0] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
        tex_descr.addressMode[1] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
        tex_descr.addressMode[2] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
        tex_descr.readMode = cudart.cudaTextureReadMode.cudaReadModeNormalizedFloat

        tex = checkCudaErrors(cudart.cudaCreateTextureObject(tex_res, tex_descr, None))

    def set_voi(self, voi: VOI):
        pass

    def copy_operator_matrix(self, transform_matrix):
        checkCudaErrors(cudart.cudaMemcpy(
            self.const_transform_matrix, transform_matrix, 9 * np.dtype(np.float32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        ))

    def render(self):
        pass
