# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import ctypes
import math
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
cudaTextureObject_t volume_text_obj;
cudaArray* d_volume_array = 0;
cudaExtent volume_size;

unsigned char* p_vr = 0;
float3 normal, spacing, max_per;
int volume_of_interest[6];
VOI voi;
int width_vr = 0;

// Device code
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}
'''


def test_modify_const(kernel_helper):
    transform_matrix = np.identity(3).astype(dtype=np.float32)
    d_transform_matrix, d_transform_matrix_size = kernel_helper.getGlobal(b'const_transform_matrix')

    checkCudaErrors(cuda.cuMemcpyHtoD(
        d_transform_matrix, transform_matrix, d_transform_matrix_size
    ))

    transform_matrix = transform_matrix * 3
    checkCudaErrors(cuda.cuMemcpyDtoH(
        transform_matrix, d_transform_matrix, d_transform_matrix_size,
    ))


def test_modify_intarr(kernel_helper):
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


def test_modify_int(kernel_helper):
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


def test_modify_float3(kernel_helper):
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


def test_cudaArray(kernel_helper):
    pass


def main():
    print("Vector Addition (Driver API)")
    N = 50000
    devID = 0
    size = N * np.dtype(np.float32).itemsize

    # Initialize
    checkCudaErrors(cuda.cuInit(0));

    cuDevice = findCudaDeviceDRV()
    # Create context
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cuDevice))

    uvaSupported = checkCudaErrors(
        cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice))
    if not uvaSupported:
        print("Accessing pageable memory directly requires UVA")
        return

    kernelHelper = common.KernelHelper(vectorAddDrv, int(cuDevice))
    _VecAdd_kernel = kernelHelper.getFunction(b'VecAdd_kernel')

    # Allocate input vectors h_A and h_B in host memory
    h_A = np.random.rand(size).astype(dtype=np.float32)
    h_B = np.random.rand(size).astype(dtype=np.float32)
    h_C = np.random.rand(size).astype(dtype=np.float32)

    # Allocate vectors in device memory
    d_A = checkCudaErrors(cuda.cuMemAlloc(size))
    d_B = checkCudaErrors(cuda.cuMemAlloc(size))
    d_C = checkCudaErrors(cuda.cuMemAlloc(size))

    # Copy vectors from host memory to device memory
    checkCudaErrors(cuda.cuMemcpyHtoD(d_A, h_A, size))
    checkCudaErrors(cuda.cuMemcpyHtoD(d_B, h_B, size))

    # Grid/Block configuration
    threadsPerBlock = 256
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock

    kernelArgs = ((d_A, d_B, d_C, N),
                  (None, None, None, ctypes.c_int))

    # Launch the CUDA kernel
    checkCudaErrors(cuda.cuLaunchKernel(_VecAdd_kernel,
                                        blocksPerGrid, 1, 1,
                                        threadsPerBlock, 1, 1,
                                        0, 0,
                                        kernelArgs, 0))

    # Copy result from device memory to host memory
    # h_C contains the result in host memory
    checkCudaErrors(cuda.cuMemcpyDtoH(h_C, d_C, size))

    for i in range(N):
        sum_all = h_A[i] + h_B[i]
        if math.fabs(h_C[i] - sum_all) > 1e-7:
            break

    # Free device memory
    checkCudaErrors(cuda.cuMemFree(d_A))
    checkCudaErrors(cuda.cuMemFree(d_B))
    checkCudaErrors(cuda.cuMemFree(d_C))

    checkCudaErrors(cuda.cuCtxDestroy(cuContext))
    print("{}".format("Result = PASS" if i + 1 == N else "Result = FAIL"))

    test_modify_const(kernelHelper)
    test_modify_intarr(kernelHelper)
    test_struct(kernelHelper)
    test_modify_int(kernelHelper)
    test_modify_float3(kernelHelper)
    test_cuda_extent(kernelHelper)


if __name__ == "__main__":
    main()
