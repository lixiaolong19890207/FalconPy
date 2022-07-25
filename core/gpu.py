import ctypes

import numpy as np
from pathlib import Path

from cuda import cuda, cudart
from cuda.ccudart import cudaFree

from core.helper.common import KernelHelper
from core.helper.helper_cuda import findCudaDevice, checkCudaErrors
from core.point import Point3

__cur_path = Path(__file__).resolve().parent
__cuda_path = __cur_path / 'kernel.cu'
with open(__cuda_path, 'r') as f:
    KERNEL_CODE = f.read()

DEVICE_ID = findCudaDevice()


class BoundingBox(ctypes.Structure):
    _fields_ = [
        ('xMin', ctypes.c_int),
        ('xMax', ctypes.c_int),
        ('yMin', ctypes.c_int),
        ('yMax', ctypes.c_int),
        ('zMin', ctypes.c_int),
        ('zMax', ctypes.c_int),
    ]


class Kernel:
    def __init__(self):
        self.bounding_box = BoundingBox()
        self.width_vr = ctypes.c_int(2)
        self.height_vr = ctypes.c_int(2)
        self.spacing = cudart.make_cudaPos(0, 0, 0)
        self.normal = cudart.make_cudaPos(0, 0, 0)
        self.max_per = cudart.make_cudaPos(0, 0, 0)

        self.dims_extent = cudart.make_cudaExtent(0, 0, 0)
        self.volume_text_obj = None

        kernel_mods = KernelHelper(KERNEL_CODE, DEVICE_ID)
        self.transform_matrix, self.transform_matrix_size = kernel_mods.getGlobal(b'const_transform_matrix')
        self.d_volume_array = kernel_mods.getGlobal(b'd_volume_array')
        # self.d_volume_array = None
        self.__cu_render = kernel_mods.getFunction(b'cu_render')

    def free(self):
        checkCudaErrors(cudart.cudaDestroyTextureObject(self.transform_matrix))
        checkCudaErrors(cudart.cudaFreeArray(self.d_volume_array))

    def reset(self, spacing_x, spacing_y, spacing_z):
        self.bounding_box = BoundingBox()
        self.width_vr = ctypes.c_int(2)
        self.height_vr = ctypes.c_int(2)

        self.spacing = cudart.make_cudaPos(0, 0, 0)
        self.normal = cudart.make_cudaPos(0, 0, 0)

        self.normal.x = 1.0 / self.dims_extent.width
        self.normal.y = 1.0 / self.dims_extent.height
        self.normal.z = 1.0 / self.dims_extent.depth

        max_spacing = max(spacing_x, max(spacing_y, spacing_z))
        max_length = max(
            self.dims_extent.width * spacing_x,
            max(
                self.dims_extent.height * spacing_y,
                self.dims_extent.depth * spacing_z
            )
        )

        self.max_per.x = 1.0 * max_length / self.dims_extent.width / spacing_x
        self.max_per.y = 1.0 * max_length / self.dims_extent.height / spacing_y
        self.max_per.z = 1.0 * max_length / self.dims_extent.depth / spacing_z

    def copy_volume(self, volume_array):
        volume_array = volume_array.astype(np.int16)

        shape = volume_array.shape
        self.dims_extent.width = shape[0]
        self.dims_extent.height = shape[1]
        self.dims_extent.depth = shape[2]

        if self.d_volume_array:
            checkCudaErrors(cudart.cudaFreeArray(self.d_volume_array))
            self.d_volume_array = None
            self.volume_text_obj = None

        channel_desc = checkCudaErrors(
            cudart.cudaCreateChannelDesc(32, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat))
        self.d_volume_array = checkCudaErrors(
            cudart.cudaMalloc3DArray(channel_desc, self.dims_extent, cudart.cudaArrayCubemap))

        memcpy_params = cudart.cudaMemcpy3DParms()
        memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
        memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
        memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(
            volume_array,
            self.dims_extent.width * np.dtype(np.int16).itemsize,
            self.dims_extent.width,
            self.dims_extent.height
        )
        memcpy_params.dstArray = self.d_volume_array
        memcpy_params.extent = self.dims_extent
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

        self.volume_text_obj = checkCudaErrors(cudart.cudaCreateTextureObject(tex_res, tex_descr, None))

    def set_bounding_box(self, box: BoundingBox):
        self.bounding_box.xMin = box.xMin
        self.bounding_box.xmax = box.xMax
        self.bounding_box.yMin = box.yMin
        self.bounding_box.yMax = box.yMax
        self.bounding_box.zMin = box.zMin
        self.bounding_box.zMax = box.zMax

    def copy_operator_matrix(self, transform_matrix):
        checkCudaErrors(cuda.cuMemcpyHtoD(
            self.transform_matrix, transform_matrix, self.transform_matrix_size
        ))

    def render(self, width, height, x_pan, y_pan, scale, invert_z, color):
        if width > self.width_vr or height > self.height_vr:
            if self.p_vr:
                checkCudaErrors(cudart.cudaFree(self.p_vr))

            self.width_vr = width
            self.height_vr = height
            checkCudaErrors(cudart)

        dim_block = cudart.dim3()
        dim_block.x = 32
        dim_block.y = 32
        dim_block.z = 1
        dim_grid = cudart.dim3()
        dim_grid.x = width / dim_block.x
        dim_grid.y = width / dim_block.y
        dim_grid.z = 1

        kernelArgs = (
            (self.p_vr, self.volume_text_obj, mask_text,
             width, height, x_pan, y_pan, scale,
             self.max_per, self.spacing, self.normal,
             self.bounding_box, self.dims_extent, invert_z, color),
            (ctypes.c_void_p, ctypes.c_int, None)
        )

        checkCudaErrors(cuda.cuLaunchKernel(
            self.__cu_render,
            dim_grid.x, dim_grid.y, dim_grid.z,  # grid dim
            dim_block.x, dim_block.y, dim_block.z,  # block dim
            0, 0,  # shared mem and stream
            kernelArgs, 0
        ))

        cudaError_t t = cudaMemcpy(pVR, p_vr, width * height * 3 * sizeof(unsigned
        char), cudaMemcpyDeviceToHost );
