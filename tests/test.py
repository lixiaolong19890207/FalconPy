# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import ctypes

import numpy as np
from cuda import cuda, cudart

from core.helper import common
from core.helper.helper_cuda import checkCudaErrors, findCudaDeviceDRV

vectorAddDrv = '''\
typedef struct {
    int left;
    int right;
} VOI;

__constant__ float const_transform_matrix[9];
cudaArray* d_volume_array = 0;

unsigned char* p_vr = 0;
float3 normal;
int volume_of_interest[6];
VOI voi;
int width_vr = 0;

// Device code
extern "C" __global__ void TestCharPtr(unsigned char* pPixelData, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        pPixelData[i] = i % 127;
}
'''


def test_const(kernel_helper):
    transform_matrix = np.identity(3).astype(dtype=np.float32)
    d_transform_matrix, d_transform_matrix_size = kernel_helper.getGlobal(b'const_transform_matrix')

    checkCudaErrors(cuda.cuMemcpyHtoD(
        d_transform_matrix, transform_matrix, d_transform_matrix_size
    ))

    transform_matrix = transform_matrix * 3
    checkCudaErrors(cuda.cuMemcpyDtoH(
        transform_matrix, d_transform_matrix, d_transform_matrix_size,
    ))


def test_intarr(kernel_helper):
    h_array = np.random.randint(0, 9, size=6).astype(dtype=np.int32)
    d_ptr, d_size = kernel_helper.getGlobal(b'volume_of_interest')

    checkCudaErrors(cuda.cuMemcpyHtoD(
        d_ptr, h_array, d_size
    ))

    h_array2 = np.zeros(6).astype(dtype=np.int32)
    checkCudaErrors(cuda.cuMemcpyDtoH(
        h_array2, d_ptr, d_size,
    ))

    assert (h_array == h_array2).all()


def test_int(kernel_helper):
    h_width = ctypes.c_float(2)
    d_ptr, d_size = kernel_helper.getGlobal(b'width_vr')

    checkCudaErrors(cuda.cuMemcpyHtoD(
        d_ptr, h_width, d_size
    ))

    h_width2 = ctypes.c_float(0)
    checkCudaErrors(cuda.cuMemcpyDtoH(
        h_width2, d_ptr, d_size,
    ))

    assert h_width.value == h_width2.value


def test_float3(kernel_helper):
    h_array = np.random.rand(3).astype(dtype=np.float32)
    d_ptr, d_size = kernel_helper.getGlobal(b'normal')

    checkCudaErrors(cuda.cuMemcpyHtoD(
        d_ptr, h_array, d_size
    ))

    h_array2 = np.zeros(3).astype(dtype=np.float32)
    checkCudaErrors(cuda.cuMemcpyDtoH(
        h_array2, d_ptr, d_size,
    ))

    assert (h_array == h_array2).all()


def test_struct(kernel_helper):
    d_ptr, d_size = kernel_helper.getGlobal(b'voi')

    class VOI(ctypes.Structure):
        _fields_ = [
            ('left', ctypes.c_int),
            ('right', ctypes.c_int),
        ]

    h_voi = VOI(4, 9)

    checkCudaErrors(cuda.cuMemcpyHtoD(
        d_ptr, h_voi, d_size
    ))

    h_voi2 = VOI(0, 0)
    checkCudaErrors(cuda.cuMemcpyDtoH(
        h_voi2, d_ptr, d_size,
    ))

    assert (h_voi.left, h_voi.right) == (h_voi2.left, h_voi2.right)


def test_cuda_extent(kernel_helper):
    cudaExtent = cudart.make_cudaExtent(1, 2, 3)
    assert (cudaExtent.width == 1)
    assert (cudaExtent.height == 2)
    assert (cudaExtent.depth == 3)

    extent = cudart.cudaExtent()
    extent.width = 1000
    extent.height = 500
    extent.depth = 0


def test_char_ptr(kernel_helper):
    N = 512 * 512

    buffer_size = np.dtype(np.int8).itemsize * N
    dbuf = checkCudaErrors(cuda.cuMemAlloc(buffer_size))

    checkCudaErrors(cuda.cuMemsetD8(dbuf, 0, buffer_size))

    _TestCharPtr = kernel_helper.getFunction(b'TestCharPtr')
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) / threads_per_block

    kernel_args = ((dbuf, N), (None, ctypes.c_int))
    checkCudaErrors(cuda.cuLaunchKernel(
        _TestCharPtr,
        blocks_per_grid, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        kernel_args, 0
    ))

    buf = np.empty(N, dtype='int8')
    checkCudaErrors(cuda.cuMemcpyDtoH(buf, dbuf, np.dtype(np.int8).itemsize * N))
    pass


def test_cudaArray(kernel_helper):
    pass


def main():
    print("Vector Addition (Driver API)")
    # Initialize
    checkCudaErrors(cuda.cuInit(0))

    cu_device = findCudaDeviceDRV()
    uva_supported = checkCudaErrors(
        cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cu_device))
    if not uva_supported:
        print("Accessing pageable memory directly requires UVA")
        return

    kernel_helper = common.KernelHelper(vectorAddDrv, int(cu_device))

    test_const(kernel_helper)
    test_intarr(kernel_helper)
    test_struct(kernel_helper)
    test_int(kernel_helper)
    test_float3(kernel_helper)
    test_cuda_extent(kernel_helper)
    test_char_ptr(kernel_helper)

    print('Test Finished')


if __name__ == "__main__":
    main()
