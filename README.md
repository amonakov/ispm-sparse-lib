= ispm-sparse-lib

CUDA sparse matrix-vector multiplication for a custom matrix format based on
Sliced ELLPACK and a few associated routines for building a GPU iterative
linear solver.  The associated [foam-extend](http://www.extend-project.de/)
solver code is currently provided at
https://github.com/amonakov/openfoam-extend-foam-extend-3.1, branch
feature/ispmCudaSolvers.  [See the respective README and
source](https://github.com/amonakov/openfoam-extend-foam-extend-3.1/tree/feature/ispmCudaSolvers/src/ispmCudaSolvers).

== Building

    make -j4 CUDA_ARCH=20
    export ISPM_DIR=`pwd`

The CUDA compiler `nvcc` must be in PATH.

`CUDA_ARCH` denotes the target CUDA compute capability.  See this 
[link](https://developer.nvidia.com/cuda-gpus) for the list.  The above
example is for Tesla M20xx, for Tesla K10 specify `CUDA_ARCH=30` and for
Tesla K20 and K40 — `CUDA_ARCH=35` (the default).

After building this library proceed to build the foam-extend solver with
`./Allwmake` in its source subdirectory.  The `ISPM_DIR` environment variable
should point to the checkout of this library, as above.
