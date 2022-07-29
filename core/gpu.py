import ctypes

import numpy as np
from pathlib import Path

from cuda import cuda, cudart

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
        self.bounding_box = None
        self.width_vr = 0
        self.height_vr = 0
        self.h_spacing = cudart.make_cudaPos(0, 0, 0)
        self.h_normal = cudart.make_cudaPos(0, 0, 0)
        self.h_max_per = cudart.make_cudaPos(0, 0, 0)

        self.h_dims_extent = cudart.make_cudaExtent(0, 0, 0)
        self.h_volume_text_obj = None
        self.d_volume_image = None

        kernel_mods = KernelHelper(KERNEL_CODE, DEVICE_ID)
        self.d_transform_matrix, self.d_transform_matrix_size = kernel_mods.getGlobal(b'const_transform_matrix')
        self.d_const_transfer_func_text, self.d_const_transfer_func_text_size = kernel_mods.getGlobal(b'const_transfer_func_text')
        self.d_volume_array, _ = kernel_mods.getGlobal(b'd_volume_array')
        self.d_transfer_func_array, _ = kernel_mods.getGlobal(b'd_transfer_func_array')

        self.__cuda_render_func = kernel_mods.getFunction(b'cu_render')

    def set_bounding_box(self, box):
        self.bounding_box = box

    def free(self):
        checkCudaErrors(cudart.cudaDestroyTextureObject(self.d_transform_matrix))
        checkCudaErrors(cudart.cudaFreeArray(self.d_volume_array))

    def reset(self, spacing_x, spacing_y, spacing_z):
        self.width_vr = 0
        self.height_vr = 0

        self.h_spacing = cudart.make_cudaPos(0, 0, 0)
        self.h_normal = cudart.make_cudaPos(0, 0, 0)

        self.h_normal.x = 1.0 / self.h_dims_extent.width
        self.h_normal.y = 1.0 / self.h_dims_extent.height
        self.h_normal.z = 1.0 / self.h_dims_extent.depth

        max_length = max(
            self.h_dims_extent.width * spacing_x,
            max(
                self.h_dims_extent.height * spacing_y,
                self.h_dims_extent.depth * spacing_z
            )
        )

        self.h_max_per.x = 1.0 * max_length / self.h_dims_extent.width / spacing_x
        self.h_max_per.y = 1.0 * max_length / self.h_dims_extent.height / spacing_y
        self.h_max_per.z = 1.0 * max_length / self.h_dims_extent.depth / spacing_z

    def copy_volume(self, volume_array):
        volume_array = volume_array.astype(np.int16)

        shape = volume_array.shape
        self.h_dims_extent.width = shape[1]
        self.h_dims_extent.height = shape[2]
        self.h_dims_extent.depth = shape[0]

        if self.d_volume_array:
            # checkCudaErrors(cudart.cudaFreeArray(self.d_volume_array))
            self.d_volume_array = None
            self.h_volume_text_obj = None

        channel_desc = checkCudaErrors(
            cudart.cudaCreateChannelDesc(np.dtype(np.int16).itemsize * 8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindSigned))
        self.d_volume_array = checkCudaErrors(cudart.cudaMalloc3DArray(channel_desc, self.h_dims_extent, cudart.cudaArrayDefault))

        memcpy_params = cudart.cudaMemcpy3DParms()
        memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
        memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
        memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(
            volume_array,
            self.h_dims_extent.width * np.dtype(np.int16).itemsize,
            self.h_dims_extent.width,
            self.h_dims_extent.height
        )
        memcpy_params.dstArray = self.d_volume_array
        memcpy_params.extent = self.h_dims_extent
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

        self.h_volume_text_obj = checkCudaErrors(cudart.cudaCreateTextureObject(tex_res, tex_descr, None))

    def copy_operator_matrix(self, transform_matrix):
        checkCudaErrors(cuda.cuMemcpyHtoD(
            self.d_transform_matrix, transform_matrix, self.d_transform_matrix_size
        ))

    def copy_transfer_function(self, transfer_func, label):

        if self.d_transfer_func_array:
            checkCudaErrors(cudart.cudaFreeArray(self.d_transfer_func_array))
            self.d_transfer_func_array = None

        tex_res = cudart.cudaResourceDesc()
        tex_res.resType = cudart.cudaResourceType.cudaResourceTypeArray
        tex_res.res.array.array = self.d_transfer_func_array

        tex_descr = cudart.cudaTextureDesc()
        tex_descr.normalizedCoords = True
        tex_descr.filterMode = cudart.cudaTextureFilterMode.cudaFilterModeLinear
        tex_descr.addressMode[0] = cudart.cudaTextureAddressMode.cudaAddressModeClamp
        tex_descr.readMode = cudart.cudaTextureReadMode.cudaReadModeElementType

        channel_desc = checkCudaErrors(cudart.cudaCreateChannelDesc(32, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat))
        if self.d_transfer_func_array:
            checkCudaErrors(cudart.cudaFreeArray(self.d_transfer_func_array))
            self.d_transfer_func_array = None

        self.d_transfer_func_array = checkCudaErrors(
            cudart.cudaMallocArray(channel_desc, 888, 0, 0))

        checkCudaErrors(cudart.cudaMemcpy2DToArray(
            self.d_transfer_func_array, 0, 0, transfer_func, size, size, 1,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        ))

        self.transferFuncText = checkCudaErrors(cudart.cudaCreateTextureObject(tex_res, tex_descr, None))

        checkCudaErrors(cuda.cuMemcpyHtoD(
            self.d_const_transfer_func_text, self.transferFuncText, constTransferFuncTexts_size
        ))

    def render(self, width, height, x_pan, y_pan, scale, invert_z, color):
        if width > self.width_vr or height > self.height_vr:
            if self.d_volume_image:
                checkCudaErrors(cudart.cudaFree(self.d_volume_image))
                self.d_volume_image = None

        if not self.d_volume_image:
            self.width_vr = width
            self.height_vr = height
            buffer_size = np.dtype(np.uint8).itemsize * self.width_vr * self.height_vr * 3
            self.d_volume_image = checkCudaErrors(cuda.cuMemAlloc(buffer_size))

        dim_block = cudart.dim3()
        dim_block.x = 32
        dim_block.y = 32
        dim_block.z = 1
        dim_grid = cudart.dim3()
        dim_grid.x = width / dim_block.x
        dim_grid.y = width / dim_block.y
        dim_grid.z = 1

        kernel_args = (
            (
                self.d_volume_image, self.h_volume_text_obj,
                ctypes.c_int(width), ctypes.c_int(height), ctypes.c_float(x_pan), ctypes.c_float(y_pan), ctypes.c_float(scale),
                self.h_max_per, self.h_spacing, self.h_normal,
                self.bounding_box, self.h_dims_extent, invert_z
            ),
            (
                None, None,
                ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                None, None, None,
                None, None, ctypes.c_bool
            )
        )

        checkCudaErrors(cuda.cuLaunchKernel(
            self.__cuda_render_func,
            dim_grid.x, dim_grid.y, dim_grid.z,
            dim_block.x, dim_block.y, dim_block.z,
            0, 0,
            kernel_args, 0
        ))

        h_volume_image = np.zeros((self.width_vr, self.height_vr, 3), dtype='uint8')
        checkCudaErrors(cuda.cuMemcpyDtoH(
            h_volume_image,
            self.d_volume_image,
            np.dtype(np.uint8).itemsize * self.width_vr * self.height_vr * 3
        ))

        return h_volume_image
