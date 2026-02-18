#ifndef MATRIXCORE_DEVICE_MATRIX_CUH
#define MATRIXCORE_DEVICE_MATRIX_CUH

#include <stdint.h>
#include <stdlib.h>


typedef struct {
    int32_t *n;
    int64_t *ids_i; // indexes to make sub qubo

    int64_t *Q; // matrix n by n
    int64_t *J; // matrix n by n
    int64_t *h;
} device_matrix_t;


static __inline__ device_matrix_t *device_matrix_init(const uint64_t n) {
    const auto m = static_cast<device_matrix_t *>(malloc(sizeof(device_matrix_t)));
    cudaMalloc(&m->n, sizeof(int32_t));
    cudaMalloc(&m->ids_i, n * sizeof(int64_t));

    cudaMalloc(&m->Q, n * n * sizeof(int64_t));
    cudaMalloc(&m->J, n * n * sizeof(int64_t));
    cudaMalloc(&m->h, n * sizeof(int64_t));



    return m;
}
static __inline__ void device_matrix_free(device_matrix_t *m) {
    cudaFree(&m->n);
    cudaFree(&m->ids_i);

    cudaFree(&m->Q);
    cudaFree(&m->J);
    cudaFree(&m->h);
}

#endif //MATRIXCORE_DEVICE_MATRIX_CUH