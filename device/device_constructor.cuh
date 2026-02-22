#ifndef MATRIXCORE_DEVICE_CONSTRUCTOR_CUH
#define MATRIXCORE_DEVICE_CONSTRUCTOR_CUH

#include <stdint.h>
#include <stdlib.h>

#include "device_submatrix.cuh"

typedef struct {
    int64_t *stack;

    int32_t offset_j;
    int64_t *ids_j; // size: n ; sub qubo indexes
    int64_t *q_j;   // size: n x v ; spins of the j variable (1 if included in subqubo)
} device_constructor_t;


static __inline__ device_constructor_t *device_constructor_init() {
    const auto c = static_cast<device_constructor_t *>(malloc(sizeof(device_constructor_t)));
    cudaMalloc(&c->stack,     MATRIX_MAX_N * MATRIX_MAX_N * MATRIX_STACK * sizeof(int64_t));

    cudaMalloc(&c->ids_j,     MATRIX_MAX_N * sizeof(int64_t));
    cudaMalloc(&c->q_j,       MATRIX_MAX_N * MATRIX_MAX_V * sizeof(int64_t));

    return c;
}
static __inline__ void device_constructor_free(device_constructor_t *m) {
    cudaFree(m->stack);

    cudaFree(m->ids_j);
    cudaFree(m->q_j);
}

#endif //MATRIXCORE_DEVICE_CONSTRUCTOR_CUH