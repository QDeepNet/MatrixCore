#include <cuda_runtime.h>
#include "ising_converter.cuh"
#include "device_matrix.cuh"


__device__ void qubo_to_ising(device_matrix_t *m) {
    __shared__ int64_t h[1024];

    h[threadIdx.x] = m->Q[threadIdx.x + blockIdx.x * m->n[0]] / 2; // Q[i,j]
    __syncthreads();

    for (unsigned int stride = 1024 / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) h[threadIdx.x] += h[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) m->h[blockIdx.x] = (float_t)h[0];
    __syncthreads();

    m->J[threadIdx.x + blockIdx.x * m->n[0]] = (float_t)(threadIdx.x != blockIdx.x ? m->Q[threadIdx.x + blockIdx.x * m->n[0]] / 2 : 0);
}

