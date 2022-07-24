import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule, DynamicSourceModule

module = DynamicSourceModule(
    """
typedef struct {
	float3 m[3];
} float3x3;

cudaExtent m_volumeSize;

__constant__ float3x3 constTransposeTransformMatrix;
__constant__ float3x3 constTransformMatrix;

cudaTextureObject_t volumeText;
cudaArray* d_volumeArray = 0;
__constant__ float const_array[32];

unsigned char* d_pVR = 0;
int nWidth_VR = 0;
int nHeight_VR = 0;
short* d_pMPR = 0;
int nWidth_MPR = 0;
int nHeight_MPR = 0;

__global__ void copy_constant_into_global(float* global_result_array)
{
    global_result_array[threadIdx.x] = const_array[threadIdx.x];
}
""", include_dirs=[r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include']
)

copy_constant_into_global = module.get_function("copy_constant_into_global")
const_array, _ = module.get_global("const_array")

# host_array = np.random.randint(0, 255, (32,)).astype(np.float32)
#
# global_result_array = drv.mem_alloc_like(host_array)
# drv.memcpy_htod(const_array, host_array)
#
# copy_constant_into_global(global_result_array, grid=(1, 1), block=(32, 1, 1))
#
# host_result_array = np.zeros_like(host_array)
# drv.memcpy_dtoh(host_result_array, global_result_array)
#
# assert (host_result_array == host_array).all
