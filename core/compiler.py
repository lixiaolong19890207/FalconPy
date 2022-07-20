from pathlib import Path

import pycuda.driver as drv
import pycuda.autoinit
import numpy

from pycuda.compiler import SourceModule


CUR_PATH = Path(__file__).resolve().parent
CUDA_PATH = CUR_PATH / 'kernel.cu'
with open(CUDA_PATH, 'r') as f:
    KERNEL = f.read()


MODULE = SourceModule(KERNEL)


